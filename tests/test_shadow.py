"""Shadow-mode live decision-equivalence — sampling, decision extraction, ledger."""

from __future__ import annotations

from distil.shadow import (
    ShadowLedger,
    ShadowSampler,
    compare_decisions,
    decision_signature,
    decision_signature_from_body,
)


# --- streaming (SSE / chunk-array) decision extraction --------------------- #
# The core property: a STREAMED response must yield the SAME signature as the
# equivalent non-streamed JSON, so shadow-mode works on Claude Code / Codex /
# Gemini sessions (which all stream).

_ANTHROPIC_SSE = (
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"tool_use","id":"t1","name":"get_weather","input":{}}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":"}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"\\"SF\\"}"}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)
_ANTHROPIC_JSON = {
    "content": [{"type": "tool_use", "name": "get_weather", "input": {"city": "SF"}}]
}

_OPENAI_SSE = (
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
    '"function":{"name":"get_weather","arguments":""}}]}}]}\n\n'
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"{\\"city\\":"}}]}}]}\n\n'
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"\\"SF\\"}"}}]}}]}\n\n'
    "data: [DONE]\n\n"
)
_OPENAI_JSON = {
    "choices": [
        {
            "message": {
                "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"SF"}'}}]
            }
        }
    ]
}

_GEMINI_SSE = (
    'data: {"candidates":[{"content":{"parts":['
    '{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]}}]}\n\n'
)
_GEMINI_JSON = {
    "candidates": [
        {"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}]}}
    ]
}


def test_anthropic_stream_matches_json():
    sig = decision_signature_from_body(_ANTHROPIC_SSE)
    assert sig.startswith("tool:")
    assert sig == decision_signature(_ANTHROPIC_JSON)


def test_openai_stream_matches_json():
    sig = decision_signature_from_body(_OPENAI_SSE)
    assert sig.startswith("tool:")
    assert sig == decision_signature(_OPENAI_JSON)


def test_gemini_stream_matches_json():
    sig = decision_signature_from_body(_GEMINI_SSE)
    assert sig.startswith("tool:")
    assert sig == decision_signature(_GEMINI_JSON)


def test_gemini_chunk_array_form():
    # Gemini streamGenerateContent without alt=sse returns a JSON array of chunks.
    import json

    body = json.dumps([_GEMINI_JSON])
    assert decision_signature_from_body(body) == decision_signature(_GEMINI_JSON)


def test_stream_text_responses_are_text():
    anth = (
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
    )
    oai = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    gem = 'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}\n\n'
    assert decision_signature_from_body(anth) == "text"
    assert decision_signature_from_body(oai) == "text"
    assert decision_signature_from_body(gem) == "text"


def test_body_json_dict_and_bytes_and_empty():
    import json

    assert decision_signature_from_body(json.dumps(_ANTHROPIC_JSON)) == decision_signature(
        _ANTHROPIC_JSON
    )
    assert decision_signature_from_body(_GEMINI_SSE.encode()).startswith("tool:")  # bytes ok
    assert decision_signature_from_body("") == "none"
    assert decision_signature_from_body("not json, not sse") == "none"


def test_compressed_vs_uncompressed_stream_equivalence():
    # Same decision, one streamed one not -> equivalent (shadow records no change).
    assert decision_signature_from_body(_OPENAI_SSE) == decision_signature_from_body(
        __import__("json").dumps(_OPENAI_JSON)
    )


def test_decision_signature_anthropic_tool_use():
    a = {"content": [{"type": "tool_use", "name": "rotate_logs", "input": {"node": "N7"}}]}
    b = {"content": [{"type": "tool_use", "name": "rotate_logs", "input": {"node": "N7"}}]}
    c = {"content": [{"type": "tool_use", "name": "rotate_logs", "input": {"node": "N8"}}]}
    assert decision_signature(a) == decision_signature(b)  # same action+target
    assert decision_signature(a) != decision_signature(c)  # different target
    assert decision_signature(a).startswith("tool:")


def test_signature_v2_normalizes_argument_whitespace():
    """v2: formatting-whitespace jitter in a tool argument is NOT a decision change,
    but genuinely different arguments still are. This is the fix for the ~27% A/A
    self-noise that made identical replayed requests read as 'decision changed'."""
    from distil.shadow import SIG_VERSION

    assert SIG_VERSION >= 2

    def call(cmd):
        return {"content": [{"type": "tool_use", "name": "bash", "input": {"command": cmd}}]}

    # Same command, different spacing / trailing whitespace → SAME decision.
    assert decision_signature(call("ls -la")) == decision_signature(call("ls   -la"))
    assert decision_signature(call("ls -la")) == decision_signature(call("  ls -la\n"))
    # Genuinely different command → still a different decision (no over-coarsening).
    assert decision_signature(call("ls -la")) != decision_signature(call("rm -rf x"))


def test_rows_stamped_with_sig_version(tmp_path):
    """Every recorded row carries the signature-algorithm version and build, so
    load(current_only=True) can scope a verdict to comparable evidence."""
    import json as _json
    from distil.shadow import SIG_VERSION, ShadowLedger

    p = tmp_path / "shadow.jsonl"
    ShadowLedger().record(True, path=p)
    row = _json.loads(p.read_text().splitlines()[0])
    assert row["sig"] == SIG_VERSION
    assert "v" in row  # build attribution present


def test_load_current_only_excludes_old_signature_rows(tmp_path):
    """A verdict must not mix signature algorithms: current_only drops rows whose
    sig != current (and legacy rows with no sig at all)."""
    import json as _json
    from distil.shadow import SIG_VERSION, ShadowLedger

    p = tmp_path / "shadow.jsonl"
    rows = [
        {"equivalent": False, "ts": 1.0, "kind": "ab"},  # legacy, no sig
        {"equivalent": False, "ts": 2.0, "kind": "ab", "sig": SIG_VERSION - 1},  # old algo
        {"equivalent": True, "ts": 3.0, "kind": "ab", "sig": SIG_VERSION},  # current
        {"equivalent": True, "ts": 4.0, "kind": "ab", "sig": SIG_VERSION},  # current
    ]
    p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")

    assert ShadowLedger.load(p).samples == 4  # unscoped: everything
    scoped = ShadowLedger.load(p, current_only=True)
    assert scoped.samples == 2  # only the two current-sig rows
    assert scoped.changes == 0  # and their equivalence, not the old rows'


def test_decision_signature_openai_tool_call():
    a = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"name": "f", "arguments": '{"x":1}'}}]}}
        ]
    }
    b = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"name": "f", "arguments": '{"x":1}'}}]}}
        ]
    }
    assert decision_signature(a) == decision_signature(b)
    assert decision_signature(a).startswith("tool:")


def test_decision_signature_text_and_none():
    assert decision_signature({"content": [{"type": "text", "text": "hi"}]}) == "text"
    assert decision_signature({"stop_reason": "end_turn"}) == "none"
    assert decision_signature("not a dict") == "none"


def test_compare_decisions():
    tool_a = {"content": [{"type": "tool_use", "name": "x", "input": {"a": 1}}]}
    tool_b = {"content": [{"type": "tool_use", "name": "x", "input": {"a": 2}}]}
    text = {"content": [{"type": "text", "text": "answer"}]}
    assert compare_decisions(tool_a, tool_a) is True
    assert compare_decisions(tool_a, tool_b) is False
    assert compare_decisions(tool_a, text) is False  # compression suppressed the tool call
    assert compare_decisions(text, text) is True


def test_sampler_is_probabilistic_and_seedable():
    import random

    # Same seed → identical draw sequence, so shadow tests stay deterministic.
    a = ShadowSampler(0.2, rng=random.Random(42))
    b = ShadowSampler(0.2, rng=random.Random(42))
    assert [a.should_sample() for _ in range(50)] == [b.should_sample() for _ in range(50)]
    # ~10 expected at rate 0.2 over 50 draws; wide band, just not degenerate.
    c = ShadowSampler(0.2, rng=random.Random(7))
    assert 2 <= sum(c.should_sample() for _ in range(50)) <= 18
    assert ShadowSampler(0.0).should_sample() is False  # disabled
    assert all(ShadowSampler(1.0).should_sample() for _ in range(10))  # rate 1 always samples


def test_ledger_records_and_rates(tmp_path):
    led = ShadowLedger()
    p = tmp_path / "shadow.jsonl"
    for eq in [True, True, True, False, True]:
        led.record(eq, path=p)
    assert led.samples == 5
    assert led.changes == 1
    assert abs(led.rate() - 0.2) < 1e-9  # 1/5 changed
    # persisted content-free (no prompt/response text)
    text = p.read_text()
    assert "equivalent" in text and "content" not in text


def test_ledger_load_roundtrip(tmp_path):
    p = tmp_path / "shadow.jsonl"
    led = ShadowLedger()
    for eq in [True, False, True]:
        led.record(eq, path=p)
    reloaded = ShadowLedger.load(p)
    assert reloaded.samples == 3
    assert reloaded.changes == 1


# --- edit-equivalence: AST-normalized code in decision signatures ---------- #


def _anthropic_edit(new_str: str) -> dict:
    return {
        "content": [
            {"type": "tool_use", "name": "Edit", "input": {"path": "x.py", "new_str": new_str}}
        ]
    }


def test_edit_equivalence_ignores_formatting_and_comments():
    a = decision_signature(_anthropic_edit("def f():\n    return 1"))
    b = decision_signature(_anthropic_edit("def f():\n    # a comment\n    return 1"))
    c = decision_signature(_anthropic_edit("def f():\n        return 1"))  # reindented body
    assert a == b == c  # same code, different formatting/comments -> same decision


def test_edit_equivalence_detects_real_logic_change():
    a = decision_signature(_anthropic_edit("def f():\n    return 1"))
    d = decision_signature(_anthropic_edit("def f():\n    return 2"))  # different value
    assert a != d  # a genuine logic change is still a decision change


def test_non_code_inputs_still_distinguished():
    s1 = decision_signature(
        {"content": [{"type": "tool_use", "name": "weather", "input": {"city": "SF"}}]}
    )
    s2 = decision_signature(
        {"content": [{"type": "tool_use", "name": "weather", "input": {"city": "NYC"}}]}
    )
    assert s1 != s2 and s1.startswith("tool:")


def test_edit_equivalence_openai_arguments():
    import json as _j

    a = decision_signature(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "Edit",
                                    "arguments": _j.dumps({"new_str": "def f():\n    return 1"}),
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    b = decision_signature(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "Edit",
                                    "arguments": _j.dumps(
                                        {"new_str": "def f():\n\n    return 1  # x"}
                                    ),
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert a == b


def test_edit_equivalence_holds_across_streaming():
    # Streamed and non-streamed forms of the same edit must still match (shared sig path).
    nonstream = decision_signature(_anthropic_edit("def f():\n    return 1"))
    sse = (
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"tool_use","id":"t","name":"Edit","input":{}}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"{\\"path\\": \\"x.py\\", \\"new_str\\": \\"def f():\\\\n    return 1\\"}"}}\n\n'
    )
    assert decision_signature_from_body(sse) == nonstream


def test_shadow_discriminates_changed_decision_cross_provider():
    """Shadow must flag a *changed* next action (not just confirm matches) for
    every provider's response shape — the basis of cross-provider validation."""
    # OpenAI — same tool, different target argument → changed
    oa_a = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "edit", "arguments": '{"path":"a.py"}'}}]
                }
            }
        ]
    }
    oa_b = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "edit", "arguments": '{"path":"b.py"}'}}]
                }
            }
        ]
    }
    assert compare_decisions(oa_a, oa_a) is True
    assert compare_decisions(oa_a, oa_b) is False

    # Gemini — different function name → changed
    gm_a = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "read", "args": {"f": "x"}}}]}}
        ]
    }
    gm_b = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "write", "args": {"f": "x"}}}]}}
        ]
    }
    assert compare_decisions(gm_a, gm_a) is True
    assert compare_decisions(gm_a, gm_b) is False

    # Anthropic — different tool input → changed
    an_a = {"content": [{"type": "tool_use", "name": "bash", "input": {"cmd": "ls"}}]}
    an_b = {"content": [{"type": "tool_use", "name": "bash", "input": {"cmd": "rm"}}]}
    assert compare_decisions(an_a, an_a) is True
    assert compare_decisions(an_a, an_b) is False


# --- A/A noise baseline (rc4): raw A/B disagreement conflates compression ---
# harm with sampling nondeterminism; the baseline is what makes it readable.


def test_ledger_aa_kind_counts_separately(tmp_path):
    p = tmp_path / "shadow.jsonl"
    led = ShadowLedger()
    led.record(True, path=p)  # default kind: ab
    led.record(False, kind="aa", path=p)
    led.record(True, kind="aa", path=p)
    assert (led.samples, led.changes) == (1, 0)  # ab meaning unchanged
    assert (led.aa_samples, led.aa_changes) == (2, 1)
    reloaded = ShadowLedger.load(p)
    assert (reloaded.samples, reloaded.aa_samples, reloaded.aa_changes) == (1, 2, 1)


def test_ledger_load_pre_rc4_rows_count_as_ab(tmp_path):
    p = tmp_path / "shadow.jsonl"
    p.write_text('{"equivalent": false, "ts": 1.0}\n', encoding="utf-8")  # no "kind"
    led = ShadowLedger.load(p)
    assert (led.samples, led.changes, led.aa_samples) == (1, 1, 0)


def test_aa_agreement_needs_ten_samples(tmp_path):
    led = ShadowLedger()
    p = tmp_path / "shadow.jsonl"
    for _ in range(9):
        led.record(True, kind="aa", path=p)
    assert led.aa_agreement() is None
    led.record(True, kind="aa", path=p)
    assert led.aa_agreement() == 1.0


def test_adjusted_rate_factors_out_model_nondeterminism(tmp_path):
    """47% raw agreement against a 52% self-agreement baseline ≈ compression
    adds ~10% — the exact confusion the raw number invites."""
    led = ShadowLedger()
    p = tmp_path / "shadow.jsonl"
    for i in range(100):
        led.record(i < 47, path=p)  # ab: 47% equivalent
    for i in range(100):
        led.record(i < 52, kind="aa", path=p)  # aa: model agrees with itself 52%
    assert abs(led.rate() - 0.53) < 1e-9
    assert abs(led.aa_agreement() - 0.52) < 1e-9
    assert abs(led.adjusted_rate() - (1 - 0.47 / 0.52)) < 1e-9
    # and a perfect baseline changes nothing
    led2 = ShadowLedger()
    for i in range(100):
        led2.record(i < 47, path=p)
    assert led2.aa_agreement() is None
    assert led2.adjusted_rate() == led2.rate()  # no baseline → raw


def test_shadow_stats_json_nulls_adjusted_without_baseline(tmp_path, monkeypatch, capsys):
    """shadow-stats --json must NOT emit the raw fallback under an 'adjusted_*'
    label when the A/A baseline is missing — a consumer would read sampling
    noise as compression harm. Null until the baseline exists; real once it does."""
    import argparse
    import json
    import distil.shadow as shadow
    from distil import cli

    p = tmp_path / "shadow.jsonl"
    led = shadow.ShadowLedger()
    for i in range(30):
        led.record(i < 18, path=p)  # A/B samples, no A/A baseline yet
    monkeypatch.setattr(shadow.ShadowLedger, "load", classmethod(lambda cls, *a, **k: led))

    cli.cmd_shadow_stats(argparse.Namespace(json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["aa_self_agreement"] is None
    assert out["adjusted_change_rate"] is None
    assert out["adjusted_equivalence"] is None

    # baseline lands → the adjusted fields become real numbers
    for i in range(20):
        led.record(i < 10, kind="aa", path=p)
    cli.cmd_shadow_stats(argparse.Namespace(json=True))
    out2 = json.loads(capsys.readouterr().out)
    assert out2["adjusted_change_rate"] is not None
    assert out2["adjusted_equivalence"] is not None


def test_record_persists_content_free_evidence(tmp_path):
    import json as _json

    p = tmp_path / "shadow.jsonl"
    ShadowLedger().record(
        False,
        kind="ab",
        evidence={"digest": "ab12", "sig_served": "tool:x1", "sig_replay": "tool:y2"},
        path=p,
    )
    rec = _json.loads(p.read_text().strip())
    assert rec["kind"] == "ab" and rec["digest"] == "ab12"
    assert rec["sig_served"] != rec["sig_replay"]  # the divergence is now diagnosable


def _shadow_e2e(monkeypatch, **wrap_kwargs):
    """Drive one real request through the in-thread proxy at shadow_rate=1.0 and
    return (shadow rows, upstream POST count).

    Pinned to the in-thread proxy so the handler runs in this process; the shadow
    machinery is identical in a hot-swap worker."""
    import http.server
    import json as _json
    import os
    import sys
    import threading
    from pathlib import Path

    from distil import proxy as proxy_mod

    monkeypatch.setenv("DISTIL_HOT_SWAP", "0")
    RESP = _json.dumps(
        {
            "content": [{"type": "tool_use", "name": "t", "input": {"x": 1}}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
    ).encode()
    posts = []

    class Up(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            posts.append(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(RESP)))
            self.end_headers()
            self.wfile.write(RESP)

        def log_message(self, *a):  # noqa: ANN002
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    child = (
        "import os, json, urllib.request\n"
        "base = os.environ['ANTHROPIC_BASE_URL']\n"
        "body = json.dumps({'model': 'claude-opus-4-8', 'messages': "
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
            shadow_rate=1.0,
            **wrap_kwargs,
        )
    finally:
        srv.shutdown()
    assert code == 0
    sj = Path(os.environ["DISTIL_HOME"]) / "shadow.jsonl"
    rows = [_json.loads(line) for line in sj.read_text().splitlines() if line.strip()]
    return rows, len(posts)


def test_proxy_paired_replay_books_both_arms_e2e(monkeypatch, tmp_path):
    """v5: one sampled request, three replays (A, A\' on the original and B on the
    compressed body), one row carrying BOTH arms.

    Under v4 the arms were disjoint request sets chosen by a 1/3 coin, so the A/A
    baseline was measured on different traffic than the A/B number it corrected."""
    rows, posts = _shadow_e2e(monkeypatch, expand=True)
    paired = [r for r in rows if r.get("kind") == "paired"]
    assert paired, rows
    row = paired[0]
    assert row["equivalent"] is True and row["aa_equal"] is True
    assert row["sig_a"] == row["sig_b"] == row["sig_aa"]
    assert row["identical"] is False, "expand injects a tool, so the bodies differ"
    assert row["sig"] == 5 and row["digest"]
    assert row["mode"] and row["pinned_temperature"] is False  # no temperature field
    assert (row["out_a"], row["out_b"]) == (20, 20)
    assert posts == 4, "the served request plus three paired replays"


def test_byte_identical_sample_is_booked_as_aa_not_an_ab_win(monkeypatch, tmp_path):
    """Compression that changed NOTHING is an A/A sample, whatever the coin said.

    Under v4 a byte-identical body was still labelled A/B — 32 of 44 "A/B" rows on
    the maintainer\'s lossless-only traffic — so the headline was mostly the model
    agreeing with itself, reported as evidence that compression is safe."""
    rows, posts = _shadow_e2e(monkeypatch, verbatim=True, lossless_only=True)
    assert [r["kind"] for r in rows] == ["aa"], rows
    row = rows[0]
    assert row["identical"] is True and row["bytes_saved"] == 0
    assert "aa_equal" not in row and "sig_b" not in row
    assert posts == 3, "nothing to compare, so only two replays are paid for"


def test_proxy_aa_replay_records_baseline_e2e_legacy(monkeypatch, tmp_path):
    """DISTIL_SHADOW_PAIRED=0 keeps the cheap two-replay design — with the A/A arm
    replaying the ORIGINAL body. v4 replayed the COMPRESSED body twice, which is a
    B/B arm: under digest mode it measured self-agreement on the wrong distribution
    entirely, and could not detect a compressor that changed every decision
    consistently."""
    monkeypatch.setenv("DISTIL_SHADOW_PAIRED", "0")
    monkeypatch.setattr("random.random", lambda: 0.0)  # sampler fires AND aa branch taken
    rows, posts = _shadow_e2e(monkeypatch, expand=True)
    assert [r["kind"] for r in rows] == ["aa"], rows
    assert rows[0]["equivalent"] is True
    assert posts == 3, "two replays, not three"


def _hammer_append(args):
    """Worker for the cross-process append test (must be module-level to pickle)."""
    path_str, worker_id, n_rows = args
    from pathlib import Path

    from distil.shadow import ShadowLedger

    led = ShadowLedger()
    fat = f"sig-{worker_id}-" + "x" * 4096  # well past PIPE_BUF atomicity
    for i in range(n_rows):
        led.record(
            i % 2 == 0,
            kind="ab",
            evidence={"digest": f"d{worker_id}-{i}", "sig_served": fat, "sig_replay": fat},
            path=Path(path_str),
        )
    return worker_id


def test_concurrent_cross_process_appends_stay_intact(tmp_path):
    """Multiple wrap sessions append to one shadow.jsonl. Without the flock a
    >PIPE_BUF line can interleave with another writer's and tear both rows —
    every line must parse and every row must survive."""
    import json as _json
    import sys
    from concurrent.futures import ProcessPoolExecutor

    import pytest

    if sys.platform == "win32":
        pytest.skip("fcntl advisory locking is POSIX-only; unlocked on Windows by design")

    p = tmp_path / "shadow.jsonl"
    workers, rows_each = 4, 25
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_hammer_append, [(str(p), w, rows_each) for w in range(workers)]))

    lines = [line for line in p.read_text().splitlines() if line.strip()]
    assert len(lines) == workers * rows_each  # nothing lost
    for line in lines:
        _json.loads(line)  # nothing torn


# --- v3: deterministic (temp-0) replay -----------------------------------------
# The gate measures whether COMPRESSION changed the decision, not whether the model
# sampled differently. force_deterministic pins temperature so the A/A noise floor
# collapses (~38% under v2 hot sampling -> ~100%), leaving A/B a real signal.


def test_force_deterministic_pins_temperature_and_preserves_prompt():
    import json

    from distil.shadow import force_deterministic

    body = json.dumps(
        {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}], "temperature": 1}
    ).encode()
    obj = json.loads(force_deterministic(body))
    assert obj["temperature"] == 0  # sampling pinned
    assert obj["model"] == "claude-x"  # same request otherwise replayed
    assert obj["messages"] == [{"role": "user", "content": "hi"}]
    # ONLY temperature — no top_p/seed, which Anthropic (Claude Code's upstream) 400s on
    assert "seed" not in obj and "top_p" not in obj


def test_force_deterministic_does_not_inject_temperature_when_absent():
    """Regression: Opus 4.7+ REMOVED temperature entirely (any value 400s) and the
    client omits it, so INJECTING temperature 0 where it was absent 400s the replay.
    force_deterministic must only pin an EXISTING temperature, never add one."""
    import json

    from distil.shadow import force_deterministic

    obj = json.loads(force_deterministic(b'{"model":"claude-opus-4-8","messages":[]}'))
    assert "temperature" not in obj  # not injected → replay stays API-valid on 4.7+


def test_force_deterministic_leaves_thinking_requests_valid():
    """Regression: Anthropic 400s on temperature != 1 with extended thinking, and
    Claude Code runs thinking by default. Forcing temp 0 unconditionally 400'd ~every
    sampled request (295/323 replay_failed). A thinking request must be replayed
    exactly as sent — temperature NOT pinned to 0 — so the replay is API-valid."""
    import json

    from distil.shadow import force_deterministic

    body = json.dumps(
        {
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 4000},
            "temperature": 1,
        }
    ).encode()
    obj = json.loads(force_deterministic(body))
    assert obj["temperature"] == 1  # NOT forced to 0 → no 400
    assert obj["thinking"] == {"type": "enabled", "budget_tokens": 4000}  # untouched
    assert obj["messages"] == [{"role": "user", "content": "hi"}]  # decision input intact
    # Opus 4.7+ "adaptive" thinking is also "on" → an existing temperature is left alone
    # (verified live: adaptive + temperature 0 → 400, as-sent → 200).
    adaptive = json.loads(
        force_deterministic(
            json.dumps(
                {
                    "model": "claude-opus-4-8",
                    "messages": [],
                    "thinking": {"type": "adaptive"},
                    "temperature": 1,
                }
            ).encode()
        )
    )
    assert adaptive["temperature"] == 1  # adaptive counts as thinking-on → not pinned to 0
    # thinking DISABLED + an EXISTING temperature is still pinned to 0 (greedy allowed)
    off = json.loads(
        force_deterministic(
            json.dumps(
                {"model": "m", "messages": [], "thinking": {"type": "disabled"}, "temperature": 0.7}
            ).encode()
        )
    )
    assert off["temperature"] == 0


def test_force_deterministic_skips_non_json():
    from distil.shadow import force_deterministic

    assert force_deterministic(None) is None
    assert force_deterministic(b"") is None
    assert force_deterministic(b"not json") is None
    assert force_deterministic(b"[1,2,3]") is None  # JSON but not an object -> skip


# --- thinking blocks in replays -----------------------------------------------
def test_prior_thinking_blocks_are_stripped_from_a_replay():
    """A thinking block is signed and bound to the request that produced it.

    Replaying the conversation as a NEW request makes the provider re-validate the
    signature, which fails with `Invalid signature in thinking block` (verified
    live). Claude Code runs thinking by default, so most turns of a real session
    carry these — shadow could only ever sample the minority that had none, which
    did not merely shrink the sample, it biased it.
    """
    import json as _json

    from distil.shadow import force_deterministic

    tb = {"type": "thinking", "thinking": "deliberating", "signature": "sig-abc"}
    body = {
        "model": "claude-opus-4-5",
        "max_tokens": 4096,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [tb, {"type": "text", "text": "Hello"}]},
            {"role": "user", "content": "again"},
        ],
    }
    out = _json.loads(force_deterministic(_json.dumps(body).encode()))
    blocks = out["messages"][1]["content"]
    assert all(b.get("type") != "thinking" for b in blocks), "signed block must not replay"
    assert any(b.get("type") == "text" for b in blocks), "the assistant's answer must remain"
    assert len(out["messages"]) == 3, "no turn should be lost when text survives"


def test_redacted_thinking_is_stripped_too():
    import json as _json

    from distil.shadow import force_deterministic

    rb = {"type": "redacted_thinking", "data": "opaque"}
    body = {
        "model": "m",
        "messages": [
            {"role": "assistant", "content": [rb, {"type": "text", "text": "hi"}]},
        ],
    }
    out = _json.loads(force_deterministic(_json.dumps(body).encode()))
    assert all(b.get("type") != "redacted_thinking" for b in out["messages"][0]["content"])


def test_a_thinking_only_turn_is_dropped_not_left_empty():
    """The API rejects an assistant turn with an empty content array."""
    import json as _json

    from distil.shadow import force_deterministic

    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "x", "signature": "s"}],
            },
            {"role": "user", "content": "again"},
        ],
    }
    out = _json.loads(force_deterministic(_json.dumps(body).encode()))
    assert len(out["messages"]) == 2, "a thinking-only turn must be dropped entirely"
    assert all(m["role"] == "user" for m in out["messages"])


def test_user_turns_and_tool_results_are_untouched():
    """Only assistant thinking is signed. Nothing else may be altered, or the two
    replay sides would no longer be comparing the same conversation."""
    import json as _json

    from distil.shadow import force_deterministic

    tr = {"type": "tool_result", "tool_use_id": "t1", "content": "output"}
    body = {"model": "m", "messages": [{"role": "user", "content": [tr]}]}
    out = _json.loads(force_deterministic(_json.dumps(body).encode()))
    assert out["messages"][0]["content"] == [tr]


def test_thinking_strip_bumped_the_signature_version(tmp_path):
    """Changing HOW the sample is generated must bump SIG_VERSION.

    The rule is stated on the constant itself — bump on any change to how a
    signature is computed OR how the compared sample is generated — because rows
    from two methods are not comparable. Stripping thinking blocks changes which
    turns can be sampled at all (v3 could only reach thinking-free turns), so v3
    rows must be discarded rather than averaged into v4's.

    Cross-audit caught this omission on the PR that introduced the stripping.
    """
    import json as _json

    from distil.shadow import SIG_VERSION, ShadowLedger

    assert SIG_VERSION >= 4, "the thinking-strip change requires a version bump"

    p = tmp_path / "shadow.jsonl"
    rows = [
        {"equivalent": True, "kind": "ab", "sig": 3},
        {"equivalent": False, "kind": "ab", "sig": 3},
        {"equivalent": True, "kind": "ab", "sig": SIG_VERSION},
    ]
    p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    led = ShadowLedger.load(path=p, current_only=True)
    assert led.samples == 1, "rows from an older sampling method must not be counted"
    assert led.changes == 0, "the discarded v3 change must not appear in the rate"


# --- v5: the paired estimator, and what it is allowed to say -------------------
# v4 computed p_BA / p_AA between two DISJOINT request sets and clamped the result
# at 0. It could print 100% by chance and could never print harm. v5 measures both
# arms on the same request and reports a difference with an interval.


def _paired(tmp_path, n, *, ab_changes=0, aa_changes=0, mode="digest"):
    led = ShadowLedger()
    for i in range(n):
        led.record(
            i >= ab_changes,
            kind="paired",
            evidence={"aa_equal": i >= aa_changes, "mode": mode},
            path=tmp_path / "shadow.jsonl",
        )
    return led


def test_paired_row_feeds_both_arms_from_one_request(tmp_path):
    led = _paired(tmp_path, 60, ab_changes=6, aa_changes=3)
    assert (led.samples, led.changes) == (60, 6)
    assert (led.aa_samples, led.aa_changes) == (60, 3)
    eq = led.equivalence()
    assert eq.estimator == "paired"
    assert eq.p_ab == 0.9 and eq.p_aa == 0.95
    assert abs(eq.diff - (-0.05)) < 1e-9  # compression costs 5 points
    assert abs(eq.pct - 95.0) < 1e-9
    # and it survives a round-trip through the ledger file
    again = ShadowLedger.load(tmp_path / "shadow.jsonl").equivalence()
    assert again.diff == eq.diff and again.estimator == "paired"


def test_paired_difference_can_report_harm(tmp_path):
    """The estimator must be able to say no. `adjusted_rate` clamped at 0, so a
    compressor that changed decisions outright still printed 100%."""
    led = _paired(tmp_path, 60, ab_changes=30)  # half the decisions changed
    eq = led.equivalence()
    assert eq.diff is not None and eq.diff < -0.4
    assert eq.pct < 60.0
    assert eq.pct_ci is not None and eq.pct_ci[1] < 100.0
    assert led.adjusted_rate() >= 0.0  # the legacy number, kept and labelled


def test_paired_difference_may_exceed_a_hundred_percent(tmp_path):
    """A/B agreeing more often than A/A is a real outcome of a noisy model, not an
    error to be clipped away. Clipping is what made the old estimator one-sided."""
    led = _paired(tmp_path, 60, aa_changes=6)
    eq = led.equivalence()
    assert eq.diff == 0.1 and eq.pct > 100.0


def test_equivalence_is_silent_below_the_reporting_floor(tmp_path):
    from distil.shadow import VERDICT_MIN_AA, VERDICT_MIN_AB, floor_note

    eq = _paired(tmp_path, VERDICT_MIN_AB - 1).equivalence()
    assert eq.below_floor and eq.pct is None and eq.pct_ci is None
    assert eq.line() == floor_note(VERDICT_MIN_AB - 1, VERDICT_MIN_AB - 1)
    assert f"need {VERDICT_MIN_AB}/{VERDICT_MIN_AA}" in eq.line()
    # the raw arms are still computed — suppression is about the CLAIM, not the data
    assert eq.p_ab == 1.0 and eq.p_ab_ci is not None


def test_a_paired_verdict_needs_a_paired_sample_not_a_padded_total(tmp_path):
    """Legacy rows count toward the arm totals. Without a floor on the paired pool
    itself, 50 unpaired A/B rows plus ONE paired row would publish a difference and
    an interval computed from n=1."""
    from distil.shadow import VERDICT_MIN_AA, VERDICT_MIN_AB

    led = ShadowLedger()
    p = tmp_path / "shadow.jsonl"
    for _ in range(VERDICT_MIN_AB):
        led.record(True, path=p)
    for _ in range(VERDICT_MIN_AA):
        led.record(True, kind="aa", path=p)
    led.record(False, kind="paired", evidence={"aa_equal": True}, path=p)

    eq = led.equivalence()
    assert eq.estimator == "paired" and eq.n_paired == 1
    assert eq.n_ab > VERDICT_MIN_AB and eq.n_aa > VERDICT_MIN_AA
    assert eq.below_floor and eq.pct is None
    assert "only 1 of them paired" in eq.line()


def test_legacy_unpaired_rows_are_labelled_as_such(tmp_path):
    led = ShadowLedger()
    p = tmp_path / "shadow.jsonl"
    for i in range(60):
        led.record(i >= 6, path=p)
    for i in range(40):
        led.record(True, kind="aa", path=p)
    eq = led.equivalence()
    assert eq.estimator == "legacy-unpaired" and eq.diff is None
    assert eq.pct == 90.0  # raw agreement, not a difference
    assert "legacy-unpaired" in eq.line()


def test_modes_are_reported_separately(tmp_path):
    """lossless-only and digest are different experiments; pooling them reports the
    average of two things nobody runs."""
    led = _paired(tmp_path, 40, mode="lossless-only")
    for i in range(20):
        led.record(
            i >= 10,
            kind="paired",
            evidence={"aa_equal": True, "mode": "digest"},
            path=tmp_path / "shadow.jsonl",
        )
    assert set(led.by_mode) == {"lossless-only", "digest"}
    assert led.by_mode["lossless-only"].ab_n == 40
    assert sum(led.by_mode["lossless-only"].diffs) == 0
    assert sum(led.by_mode["digest"].diffs) == -10


def test_byte_identical_and_hot_replays_are_counted(tmp_path):
    p = tmp_path / "shadow.jsonl"
    led = ShadowLedger()
    led.record(True, kind="aa", evidence={"identical": True, "pinned_temperature": False}, path=p)
    led.record(
        True,
        kind="paired",
        evidence={"aa_equal": True, "identical": False, "pinned_temperature": True},
        path=p,
    )
    assert (led.identical, led.pinned, led.unpinned) == (1, 1, 1)
    reloaded = ShadowLedger.load(p)
    assert (reloaded.identical, reloaded.pinned, reloaded.unpinned) == (1, 1, 1)


def test_wilson_interval_does_not_collapse_at_a_perfect_arm():
    from distil.shadow import wilson_ci

    lo, hi = wilson_ci(30, 30)
    assert lo < 1.0 and hi == 1.0, "a Wald interval would report [1.0, 1.0] here"
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0)  # no evidence is the whole interval
    lo, hi = wilson_ci(45, 50)
    assert lo < 0.9 < hi


def test_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    from distil.shadow import bootstrap_ci

    xs = [0] * 90 + [-1] * 10
    lo, hi = bootstrap_ci(xs)
    assert lo <= -0.1 <= hi
    assert bootstrap_ci(xs) == (lo, hi), "the same rows must give the same interval"
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([3]) == (3.0, 3.0)


# --- output-token accounting (the compression paradox) -------------------------


def test_output_inflation_can_flip_the_net_negative():
    """Compression that shortens the prompt but lengthens the answer can cost more
    than it saved: output is priced 5x input on this model."""
    from distil.shadow import ReplayCost, cost_delta

    rows = [
        ReplayCost("claude-opus-4-8", in_a=1000, in_b=900, out_a=100, out_b=300) for _ in range(20)
    ]
    cd = cost_delta(rows)
    assert cd is not None and cd.priced == 20
    assert cd.out_delta_mean == 200.0
    assert cd.input_saved_usd > 0 and cd.output_extra_usd > 0
    assert cd.net_usd < 0, "200 extra output tokens outweigh 100 saved input tokens"
    lo, hi = cd.out_delta_ci
    assert lo == hi == 200.0  # zero variance, so the interval is a point


def test_cost_delta_skips_unpriceable_models_but_still_counts_tokens():
    from distil.shadow import ReplayCost, cost_delta

    cd = cost_delta([ReplayCost("some-other-vendor-model", 1000, 900, 100, 90)])
    assert cd is not None and cd.priced == 0
    assert cd.out_delta_mean == -10.0 and cd.net_usd == 0.0
    assert cost_delta([]) is None


def test_ledger_carries_output_tokens_through_the_file(tmp_path):
    p = tmp_path / "shadow.jsonl"
    led = ShadowLedger()
    led.record(
        True,
        kind="paired",
        evidence={
            "aa_equal": True,
            "model": "claude-opus-4-8",
            "in_a": 1000,
            "in_b": 800,
            "out_a": 50,
            "out_b": 60,
        },
        path=p,
    )
    cd = ShadowLedger.load(p).cost()
    assert cd is not None and cd.out_delta_mean == 10.0 and cd.n == 1
    assert cd.net_usd > 0, "200 input tokens saved beats 10 extra output tokens"


def test_a_malformed_cost_row_does_not_sink_the_read(tmp_path):
    import json as _json

    p = tmp_path / "shadow.jsonl"
    rows = [
        {"equivalent": True, "kind": "paired", "sig": 5, "aa_equal": True},
        {
            "equivalent": True,
            "kind": "paired",
            "sig": 5,
            "aa_equal": True,
            "in_a": "not-a-number",
            "in_b": 1,
            "out_a": 1,
            "out_b": 1,
        },
    ]
    p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    led = ShadowLedger.load(p, current_only=True)
    assert led.samples == 2 and led.cost() is None


# --- replay provenance ---------------------------------------------------------


def test_deterministic_body_reports_whether_it_actually_pinned():
    """Claude Code runs thinking by default and current models carry no temperature
    field, so most replays are HOT. The reports must say so rather than imply a
    determinism the replay never had."""
    import json as _json

    from distil.shadow import deterministic_body

    hot = deterministic_body(_json.dumps({"model": "m", "messages": []}).encode())
    assert hot is not None and hot.pinned_temperature is False and hot.model == "m"

    pinned = deterministic_body(_json.dumps({"model": "m", "temperature": 0.7}).encode())
    assert pinned is not None and pinned.pinned_temperature is True
    assert _json.loads(pinned.body)["temperature"] == 0

    thinking = deterministic_body(
        _json.dumps({"temperature": 0.7, "thinking": {"type": "enabled"}}).encode()
    )
    assert thinking is not None and thinking.pinned_temperature is False
    assert _json.loads(thinking.body)["temperature"] == 0.7  # pinning here 400s
    assert deterministic_body(b"not json") is None


def test_paired_design_bumped_the_signature_version():
    """The rule on SIG_VERSION: bump on any change to how the compared sample is
    GENERATED. v5 replays different bodies in a different arm structure, so v4 rows
    are not comparable and must not be averaged in."""
    import json as _json

    from distil.shadow import SIG_VERSION, ShadowLedger

    assert SIG_VERSION >= 5
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "shadow.jsonl"
        rows = [
            {"equivalent": False, "kind": "ab", "sig": 4},
            {"equivalent": True, "kind": "paired", "aa_equal": True, "sig": SIG_VERSION},
        ]
        p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        led = ShadowLedger.load(path=p, current_only=True)
        assert led.samples == 1 and led.changes == 0
