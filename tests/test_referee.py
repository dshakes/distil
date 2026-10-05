"""`distil audit` (distil/referee.py): the referee's sampler, cap, estimator and report.

All offline: replays go to a fake `post`, or to a local stub upstream in the e2e test.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pytest

from distil import referee as rf
from distil.shadow import SIG_VERSION

MODEL = "claude-sonnet-5-5"


def _req(**extra) -> bytes:
    body = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "go"}]}
    body.update(extra)
    return json.dumps(body).encode()


def _resp(tool_input: dict, *, inp=1000, out=20, applied=False) -> bytes:
    body: dict = {
        "content": [{"type": "tool_use", "name": "bash", "input": tool_input}],
        "usage": {"input_tokens": inp, "output_tokens": out},
    }
    if applied:
        body["context_management"] = {
            "applied_edits": [{"type": "clear_tool_uses_20250919", "cleared_input_tokens": 9}]
        }
    return json.dumps(body).encode()


class FakePost:
    """A provider that answers A/A' with one action and B (context_management present)
    with `b_action`, billing B `b_in` input tokens."""

    def __init__(self, b_action: dict | None = None, b_in: int = 400, status: int = 200):
        self.calls: list[tuple[bytes, dict]] = []
        self.b_action = b_action
        self.b_in = b_in
        self.status = status

    def __call__(self, path, body, headers):
        self.calls.append((body, dict(headers)))
        if b'"context_management"' in body:
            return (
                self.status,
                {},
                _resp(self.b_action or {"cmd": "ls"}, inp=self.b_in, applied=True),
            )
        return self.status, {}, _resp({"cmd": "ls"})


def _auditor(compressor="anthropic-context-editing", cap=1.0, via=None, seed=0):
    return rf.Auditor(rf.AuditConfig(compressor, 1.0, cap, via), rng=random.Random(seed))


# ------------------------------------------------------------------ config


@pytest.mark.parametrize(
    ("cfg", "needle"),
    [
        (rf.AuditConfig("rtk"), "no faithful A"),
        (rf.AuditConfig("headroom"), "--via"),
        (rf.AuditConfig("headroom", via="https://evil.example.com"), "loopback"),
        (rf.AuditConfig("distil", via="http://127.0.0.1:1"), "only to --compressor headroom"),
        (rf.AuditConfig("distil", rate=0), "--rate"),
        (rf.AuditConfig("distil", cap_usd=0), "--cap-usd"),
        (rf.AuditConfig("nope"), "unknown"),
    ],
)
def test_validate_refuses_what_cannot_run_faithfully_or_safely(cfg, needle):
    assert needle in (rf.validate(cfg) or "")


def test_validate_accepts_the_auditable_ones():
    assert rf.validate(rf.AuditConfig("headroom", via="http://localhost:8787")) is None
    for c in ("distil", "anthropic-context-editing", "openai-compaction"):
        assert rf.validate(rf.AuditConfig(c)) is None


def test_config_roundtrip_and_auditor_is_off_by_default():
    assert rf.Auditor.load() is None, "no config file means no audit"
    rf.AuditConfig("distil", 0.5, 2.0).save()
    a = rf.Auditor.load()
    assert a is not None and a.cfg.rate == 0.5 and a.cfg.cap_usd == 2.0
    rf.config_path().write_text('{"compressor": "rtk"}')
    assert rf.Auditor.load() is None, "an unrunnable config is ignored, not obeyed"
    rf.config_path().write_text("not json")
    assert rf.Auditor.load() is None


# ------------------------------------------------------------------ shaping


def test_context_editing_b_carries_the_documented_parameter_and_beta():
    sh = rf.shape(
        "anthropic-context-editing", "/v1/messages", _req(), b"", {"Anthropic-Beta": "x-1"}
    )
    assert sh is not None
    assert "context_management" not in json.loads(sh.a)
    assert json.loads(sh.b)["context_management"] == {
        "edits": [{"type": "clear_tool_uses_20250919"}]
    }
    assert sh.b_headers["Anthropic-Beta"] == "x-1,context-management-2025-06-27"
    assert rf.shape("anthropic-context-editing", "/v1/chat/completions", _req(), b"", {}) is None


def test_upstream_context_editing_is_audited_against_off():
    on = _req(context_management={"edits": [{"type": "clear_tool_uses_20250919"}]})
    sh = rf.shape("anthropic-context-editing", "/v1/messages", on, b"", {})
    assert sh is not None
    assert "context_management" not in json.loads(sh.a)
    assert "context_management" in json.loads(sh.b)


def test_openai_compaction_is_responses_only():
    body = json.dumps({"model": "gpt-5", "input": [{"role": "user", "content": "x"}]}).encode()
    sh = rf.shape("openai-compaction", "/v1/responses", body, b"", {})
    assert sh is not None
    assert json.loads(sh.b)["context_management"] == [
        {"type": "compaction", "compact_threshold": rf.COMPACT_THRESHOLD}
    ]
    assert rf.shape("openai-compaction", "/v1/chat/completions", body, b"", {}) is None


def test_distil_b_is_the_served_body_and_headroom_b_goes_via():
    sh = rf.shape("distil", "/v1/messages", _req(), _req(system="s"), {})
    assert sh is not None and sh.a != sh.b
    hr = rf.shape("headroom", "/v1/messages", _req(), b"", {}, via="http://127.0.0.1:9")
    assert hr is not None and hr.a == hr.b and hr.via == "http://127.0.0.1:9"
    assert rf.shape("distil", "/v1/messages", b"not json", b"", {}) is None
    assert rf.shape("distil", "/v1/messages", _req(), b"not json", {}) is None
    assert rf.shape("bogus", "/v1/messages", _req(), b"", {}) is None


def test_fired_reads_each_providers_own_ground_truth():
    sh = rf.Shaped(b"a", b"b", {}, None, MODEL)
    assert rf.fired("anthropic-context-editing", sh, _resp({}, applied=True)) is True
    assert (
        rf.fired("anthropic-context-editing", sh, b'{"context_management":{"applied_edits":[]}}')
        is False
    )
    assert rf.fired("openai-compaction", sh, b'{"output":[{"type": "compaction"}]}') is True
    assert rf.fired("distil", sh, b"") is True
    assert rf.fired("distil", rf.Shaped(b"a", b"a", {}, None, MODEL), b"") is False
    assert rf.fired("headroom", sh, b"") is None


def test_signature_reads_the_responses_api_too():
    out = [{"type": "function_call", "name": "shell", "arguments": '{"cmd":"ls"}'}]
    js = rf.signature(json.dumps({"output": out}).encode())
    sse = rf.signature(
        b'data: {"type":"response.created"}\n\n'
        + b"data: "
        + json.dumps({"type": "response.completed", "response": {"output": out}}).encode()
        + b"\n\n"
    )
    assert js.startswith("tool:") and js == sse
    assert rf.signature(_resp({"cmd": "ls"})).startswith("tool:")
    assert rf.signature(b"data: nothing\n\n") == "none"
    assert rf.signature(b'{"x": 1}') == "none"


# ------------------------------------------------------------------ money + cap


def test_estimate_and_prices():
    est = rf.estimate_usd({"max_tokens": 32000}, b"x" * 350, MODEL)
    assert est == pytest.approx(100 * 3e-6 + rf.OUT_TOKENS_EST * 15e-6)
    assert rf.estimate_usd({}, b"x", "unknown-model") is None
    u = {"input_tokens": 10, "cache_read_input_tokens": 1000, "output_tokens": 5}
    assert rf.total_input(u) == 1010
    assert rf.list_usd(u, MODEL) == pytest.approx(1010 * 3e-6 + 5 * 15e-6)
    assert rf.billed_usd(u, MODEL) < rf.list_usd(u, MODEL), "cache reads bill at 0.1x"
    assert rf.billed_usd(u, "unknown-model") == 0.0 and rf.list_usd(u, "unknown-model") is None


def test_budget_reserves_settles_and_rolls_over(tmp_path):
    b = rf.Budget(1.0, tmp_path / "b.json")
    day1, day2 = 1_800_000_000.0, 1_800_000_000.0 + 86400 * 2
    assert b.reserve(0.6, now=day1)
    assert not b.reserve(0.6, now=day1), "an open reservation counts against the cap"
    b.settle(0.6, 0.2, now=day1)
    assert b.reserve(0.6, now=day1)
    b.settle(0.6, 0.7, now=day1)
    t = b.today(now=day1)
    assert t["spent"] == pytest.approx(0.9) and t["samples"] == 2 and t["skipped_cap"] == 1
    assert t["reserved"] == pytest.approx(0.0)
    assert not b.reserve(0.2, now=day1)
    assert b.reserve(0.2, now=day2), "a new day starts a new budget"


def test_the_cap_stops_the_sample_before_any_replay(tmp_path):
    post = FakePost()
    row = _auditor(cap=0.0001).run("/v1/messages", _req(), b"", {}, post, ledger=tmp_path / "l")
    assert row is None and post.calls == [], "no replay is paid for when the estimate won't fit"
    assert rf.Budget(1).today()["skipped_cap"] == 1


def test_unpriced_model_is_never_audited(tmp_path):
    post = FakePost()
    body = json.dumps({"model": "mystery", "messages": []}).encode()
    assert _auditor().run("/v1/messages", body, b"", {}, post, ledger=tmp_path / "l") is None
    assert post.calls == []


def test_a_sample_replays_three_arms_records_and_settles_actual_spend(tmp_path):
    post, booked = FakePost(b_action={"cmd": "cat"}), []
    led = tmp_path / "audit.jsonl"
    row = _auditor().run(
        "/v1/messages", _req(), b"", {}, post, book=lambda *a: booked.append(a), ledger=led
    )
    assert row is not None and len(post.calls) == 3
    assert sum(b'"context_management"' in c[0] for c in post.calls) == 1
    assert row["equivalent"] is False and row["aa_equal"] is True and row["fired"] is True
    assert (row["in_a"], row["in_b"]) == (1000, 400)
    assert row["usd_a"] > row["usd_b"]
    rec = json.loads(led.read_text())
    assert rec["kind"] == "paired" and rec["sig"] == SIG_VERSION
    assert rec["compressor"] == "anthropic-context-editing"
    assert "go" not in led.read_text(), "content-free"
    assert booked and booked[0][0] == "audit" and len(booked[0][2]) == 3
    t = rf.Budget(1).today()
    assert t["samples"] == 1 and t["reserved"] == pytest.approx(0.0)
    # billed (output at 5x input), not the reservation
    assert t["spent"] == pytest.approx((2 * (1000 + 100) + (400 + 100)) * 3e-6)


def test_a_failed_arm_stops_paying_and_records_nothing(tmp_path):
    post = FakePost(status=500)
    led = tmp_path / "l"
    assert _auditor().run("/v1/messages", _req(), b"", {}, post, ledger=led) is None
    assert len(post.calls) == 1 and not led.exists()
    assert rf.Budget(1).today()["samples"] == 1


def test_headroom_b_goes_through_the_via_proxy_only(tmp_path):
    seen = []

    def via(base, path, body, headers):
        seen.append(base)
        return 200, {}, _resp({"cmd": "ls"}, inp=300)

    a = _auditor("headroom", via="http://127.0.0.1:8787")
    a.via_post = via
    post = FakePost()
    row = a.run("/v1/messages", _req(), b"", {}, post, ledger=tmp_path / "l")
    assert row is not None and seen == ["http://127.0.0.1:8787"] and len(post.calls) == 2
    assert row["fired"] is None and row["in_b"] == 300


def test_thread_never_raises(tmp_path):
    def boom(*a):
        raise RuntimeError("upstream exploded")

    t = _auditor().thread("/v1/messages", _req(), b"", {}, boom)
    t.start()
    t.join(5)
    assert not t.is_alive()


# ------------------------------------------------------------------ verdicts


def _row(eq: bool, aa: bool, usd_a: float, usd_b: float, fired=True) -> dict:
    return {
        "kind": "paired",
        "sig": SIG_VERSION,
        "compressor": "x",
        "equivalent": eq,
        "aa_equal": aa,
        "fired": fired,
        "usd_a": usd_a,
        "usd_b": usd_b,
    }


def test_underpowered_says_cant_tell_and_how_many_more():
    rep = rf.summarize("x", [_row(True, True, 1.0, 0.5)] * 10 + [_row(True, True, 1, 1, False)] * 5)
    assert rep.verdict == "can't tell yet"
    assert rep.need == 40 and rep.audited == 15 and rep.fired == 10
    assert "where x fired" in rep.why


def test_helped_needs_money_saved_and_decisions_within_the_margin():
    rows = [_row(True, True, 1.0, 0.6 + 0.01 * (i % 5)) for i in range(60)]
    rep = rf.summarize("x", rows)
    assert rep.verdict == "helped", rep.why
    assert rep.saved_usd_ci is not None and rep.saved_usd_ci[0] > 0


def test_hurt_on_decisions_even_when_it_saves_money():
    rows = [_row(i % 2 == 0, True, 1.0, 0.5) for i in range(60)]
    rep = rf.summarize("x", rows)
    assert rep.verdict == "hurt" and "next action" in rep.why


def test_hurt_on_money():
    rows = [_row(True, True, 0.5, 0.9 + 0.01 * (i % 3)) for i in range(60)]
    rep = rf.summarize("x", rows)
    assert rep.verdict == "hurt" and "costs more" in rep.why


def test_straddling_intervals_stay_undecided_with_a_sample_size():
    rows = [_row(True, True, 1.0, 1.0 + (0.3 if i % 2 else -0.29)) for i in range(60)]
    rep = rf.summarize("x", rows)
    assert rep.verdict == "can't tell yet"
    assert rep.need is not None and rep.need > 0
    zero = rf.summarize(
        "x", [_row(True, True, 1.0, 1.0 + (0.3 if i % 2 else -0.3)) for i in range(60)]
    )
    assert zero.verdict == "can't tell yet"


def test_report_groups_by_compressor_and_renders(tmp_path):
    led = tmp_path / "audit.jsonl"
    rows = [dict(_row(True, True, 1.0, 0.5), compressor="distil") for _ in range(3)]
    rows += [dict(_row(True, True, 1.0, 0.5), compressor="headroom", sig=1)]  # old algorithm
    led.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    reps = rf.report(led)
    assert [r.compressor for r in reps] == ["distil"]
    text = rf.render(reps, {"spent": 0.5, "samples": 3, "skipped_cap": 2}, 1.0)
    assert "distil: CAN'T TELL YET (need 47 more)" in text
    assert "2 skipped at the cap" in text and "turns: not measurable" in text
    assert "nothing audited yet" in rf.render([], {}, None)


# ------------------------------------------------------------------ CLI


def _cli(*argv: str) -> int:
    from distil import cli

    args = cli.build_parser().parse_args(["audit", *argv])
    return args.func(args)


def test_cli_turns_on_only_with_consent(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _cli("--compressor", "anthropic-context-editing") == 2
    assert not rf.config_path().exists()
    assert _cli("--compressor", "anthropic-context-editing", "--yes", "--cap-usd", "0.5") == 0
    out = capsys.readouterr().out
    assert "YOUR key" in out and "$0.50/day" in out
    assert rf.AuditConfig.load().cap_usd == 0.5
    assert _cli() == 0 and "today:" in capsys.readouterr().out
    assert _cli("report", "--compressor", "distil") == 2
    assert _cli("off") == 0 and not rf.config_path().exists()
    assert _cli("off") == 0
    assert _cli() == 0 and "audit: off" in capsys.readouterr().out


def test_cli_interactive_prompt(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert _cli("--compressor", "distil") == 1 and not rf.config_path().exists()
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert _cli("--compressor", "distil") == 0 and rf.config_path().exists()


def test_cli_refuses_rtk_with_the_reason(capsys):
    assert _cli("--compressor", "rtk", "--yes") == 2
    assert "no faithful A" in capsys.readouterr().err


def test_cli_report_text_and_json(capsys):
    assert _cli("report") == 0 and "nothing audited yet" in capsys.readouterr().out
    assert _cli("report", "--json") == 0
    assert json.loads(capsys.readouterr().out)["compressors"] == []


def test_audit_is_hidden_from_the_front_door_but_listed_in_help_all():
    from distil import cli

    p = cli.build_parser()
    assert "audit" not in cli.front_help(p)
    assert "audit" in p.format_help()


# ------------------------------------------------------------------ proxy e2e


def test_proxy_audits_context_editing_end_to_end(monkeypatch):
    """One real request through the in-thread proxy with the audit on at rate 1: the
    served request plus A, A' and B, and one content-free paired row."""
    import http.server
    import os
    import sys
    import threading

    from distil import proxy as proxy_mod

    monkeypatch.setenv("DISTIL_HOT_SWAP", "0")
    rf.AuditConfig("anthropic-context-editing", 1.0, 5.0).save()
    posts: list[tuple[bytes, str]] = []

    class Up(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            posts.append((body, self.headers.get("anthropic-beta", "")))
            resp = _resp({"x": 1}, applied=b'"context_management"' in body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *a):  # noqa: ANN002
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    child = (
        "import os, json, urllib.request\n"
        "base = os.environ['ANTHROPIC_BASE_URL']\n"
        f"body = json.dumps({{'model': '{MODEL}', 'max_tokens': 64, 'messages': "
        "[{'role': 'user', 'content': 'go'}]}).encode()\n"
        "req = urllib.request.Request(base + '/v1/messages', data=body,"
        " headers={'Content-Type': 'application/json'}, method='POST')\n"
        "urllib.request.urlopen(req, timeout=5)\n"
    )
    try:
        code = proxy_mod.wrap_run(
            [sys.executable, "-c", child],
            upstream=f"http://127.0.0.1:{srv.server_address[1]}",
            record=False,
            shadow_rate=0.0,
        )
    finally:
        srv.shutdown()
    assert code == 0
    assert len(posts) == 4, "the served request plus A, A' and B"
    edited = [beta for body, beta in posts if b'"context_management"' in body]
    assert edited == ["context-management-2025-06-27"]
    led = Path(os.environ["DISTIL_HOME"]) / "audit.jsonl"
    (row,) = [json.loads(x) for x in led.read_text().splitlines()]
    assert row["compressor"] == "anthropic-context-editing" and row["fired"] is True
    assert row["equivalent"] is True and row["aa_equal"] is True


def test_cli_namespace_defaults():
    from distil import cli

    a = cli.build_parser().parse_args(["audit"])
    assert isinstance(a, argparse.Namespace)
    assert (a.action, a.rate, a.cap_usd) == ("status", rf.DEFAULT_RATE, rf.DEFAULT_CAP_USD)
