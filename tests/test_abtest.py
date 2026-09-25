"""distil.abtest — session-randomised, model-aware task-level A/B.

Known-answer tests use simulated sessions with a KNOWN true effect: a model change
that moves turns/task in both arms must not be attributed to distil; a change in one
stratum must not touch another; CUPED must narrow the interval on correlated data;
the sequential interval must hold its error rate under repeated looks (Monte Carlo,
loose tolerance — see test_sequential_type_one_error_under_repeated_looks).
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchmarks.abtest_montecarlo import sim
from distil import abtest
from distil.abtest import outcomes, stats
from distil.abtest.outcomes import SessionOutcome, client_tag, fold_session, is_user_turn
from distil.abtest.report import family, render, split_eras, summarize

# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def test_assignment_is_deterministic_keyed_and_near_the_rate(monkeypatch):
    monkeypatch.setenv(abtest.RATE_ENV, "0.05")
    arms = [abtest.assign(f"s{i}-1").arm for i in range(4000)]
    assert arms == [abtest.assign(f"s{i}-1").arm for i in range(4000)]
    share = arms.count(abtest.HOLDOUT) / len(arms)
    assert 0.035 < share < 0.065  # binomial sd ≈ 0.0034
    assert set(arms) == {abtest.DISTIL, abtest.HOLDOUT}
    # A different install secret draws differently: the assignment is keyed.
    before = [abtest.bucket(f"s{i}") for i in range(50)]
    abtest.config_path().unlink()
    assert [abtest.bucket(f"s{i}") for i in range(50)] != before


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_secret_is_owner_only():
    abtest.bucket("x")
    assert abtest.config_path().stat().st_mode & 0o077 == 0


def test_resumed_session_keeps_its_arm_even_if_the_rate_changes(monkeypatch):
    monkeypatch.setenv(abtest.RATE_ENV, "0")
    prev = {"ab": {"arm": "holdout", "rate": 0.05}}
    assert abtest.assign("s1-1", prev) == abtest.Assignment("holdout", 0.05)
    assert abtest.assign("s1-1", None).arm == ""  # disabled → not randomised
    assert abtest.assign(None, None).arm == ""
    # A corrupt previous record is ignored, not trusted.
    monkeypatch.setenv(abtest.RATE_ENV, "0.5")
    assert abtest.assign("s1-1", {"ab": {"arm": "holdout", "rate": "x"}}).arm in (
        "distil",
        "holdout",
    )


@pytest.mark.parametrize(
    ("raw", "want"),
    [("0.05", 0.05), ("5%", 0.05), (" 0 ", 0.0), ("off", 0.0), ("false", 0.0), (0.2, 0.2)],
)
def test_parse_rate_accepts(raw, want):
    assert abtest.parse_rate(raw) == pytest.approx(want)


@pytest.mark.parametrize("raw", ["5", "-0.1", "0.9", "abc", True])
def test_parse_rate_rejects(raw):
    with pytest.raises(ValueError):
        abtest.parse_rate(raw)


def test_rate_precedence_and_opt_out(monkeypatch, caplog):
    monkeypatch.delenv(abtest.RATE_ENV, raising=False)
    assert abtest.holdout_rate() == abtest.DEFAULT_RATE
    abtest.set_holdout_rate(0.1)
    assert abtest.holdout_rate() == 0.1
    monkeypatch.setenv(abtest.RATE_ENV, "0")
    assert abtest.holdout_rate() == 0.0  # env beats config: the opt-out wins
    monkeypatch.setenv(abtest.RATE_ENV, "lots")
    assert abtest.holdout_rate() == 0.0  # unparseable → off, never the default
    monkeypatch.delenv(abtest.RATE_ENV)
    cfg = json.loads(abtest.config_path().read_text())
    cfg["holdout_rate"] = 7
    abtest.config_path().write_text(json.dumps(cfg))
    assert abtest.holdout_rate() == 0.0
    abtest.config_path().write_text("{not json")
    assert abtest.holdout_rate() == abtest.DEFAULT_RATE


def test_disclosure_names_the_rate_and_the_opt_out():
    assert "5%" in abtest.disclosure(0.05) and "--holdout-rate 0" in abtest.disclosure(0.05)
    assert "off" in abtest.disclosure(0.0)


# --------------------------------------------------------------------------- #
# Per-request fields and the session fold
# --------------------------------------------------------------------------- #


def test_client_tag_is_name_and_major_minor_only():
    assert client_tag("claude-cli/2.1.3 (external, cli)") == "claude-cli/2.1"
    assert client_tag("OpenAI/Python 1.40.0") == ""  # not a product/version token
    assert client_tag("codex/1") == "codex/1"
    assert client_tag(None) == "" and client_tag("<script>") == ""


def test_is_user_turn_by_roles_and_block_types_only():
    tr = {"type": "tool_result", "tool_use_id": "t", "content": "x"}
    assert is_user_turn({"messages": [{"role": "user", "content": "fix it"}]}) is True
    assert is_user_turn({"messages": [{"role": "user", "content": [tr]}]}) is False
    assert is_user_turn({"messages": [{"role": "tool", "content": "x"}]}) is False
    assert is_user_turn({"input": [{"type": "function_call_output", "output": "x"}]}) is False
    assert is_user_turn({"input": [{"role": "user", "content": "hi"}]}) is True
    assert (
        is_user_turn({"contents": [{"role": "user", "parts": [{"functionResponse": {}}]}]}) is False
    )
    assert is_user_turn({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}) is True
    assert is_user_turn({"messages": []}) is None and is_user_turn(None) is None
    assert is_user_turn({"messages": ["x"]}) is None


def _row(ts, model="claude-opus-4-8", status=200, user_turn=None, **usage):
    r = {"ts": ts, "model": model, "status": status, "client": "claude-cli/2.1"}
    r.update(
        usage_input_tokens=usage.get("inp", 1000),
        usage_output_tokens=usage.get("out", 100),
        usage_cache_read=usage.get("cr", 0),
        usage_cache_create=usage.get("cw", 0),
    )
    if user_turn is not None:
        r["user_turn"] = user_turn
    return r


def test_fold_prices_cache_inclusive_and_counts_tasks():
    man = {"sid": "s1", "ab": {"arm": "holdout", "rate": 0.05}, "cwd": "/w", "started_ts": 10.0}
    rows = [
        _row(11, user_turn=True, inp=1_000_000, cr=1_000_000, cw=1_000_000, out=1_000_000),
        _row(12, user_turn=False),
        _row(13, user_turn=True),
        _row(14, model="claude-haiku-4-5", user_turn=True),  # side call, not a task
        _row(15, status=529),
        {"ts": 16, "model": "gemini-2.5-pro", "status": 200, "usage_input_tokens": 5},
    ]
    o = fold_session(man, rows, exit_text="child exit code 0 at …", src=20.0, now=30.0)
    assert o is not None
    # opus-4-8: $5 in, $25 out, read 0.1x, write 1.25x — per Mtok
    first = 5 + 0.5 + 6.25 + 25
    assert o.cost == pytest.approx(
        first + 2 * (1000 * 5e-6 + 100 * 25e-6) + (1000 * 1e-6 + 100 * 5e-6)
    )
    assert (o.model, o.client, o.turns, o.tasks, o.errors, o.unpriced) == (
        "claude-opus-4-8",
        "claude-cli/2.1",
        5,
        2,
        1,
        1,
    )
    assert o.ended == "ok" and o.arm == "holdout" and o.ws and o.ws != "/w"
    assert o.wall_s == pytest.approx(6.0)


def test_fold_end_states_and_unrandomised():
    man = {"sid": "s1", "ab": {"arm": "distil", "rate": 0.05}}
    assert fold_session(man, [_row(1)], exit_text="child signal SIGKILL", now=2).ended == "error"
    assert fold_session(man, [_row(1)], now=2).ended == "open"
    assert fold_session(man, [_row(1)], now=1 + outcomes.ABANDON_S + 1).ended == "abandoned"
    o = fold_session(man, [], now=5)
    assert o is not None and o.cost is None and o.tasks == 1 and o.turns == 0
    assert fold_session({"sid": "s2"}, [_row(1)]) is None
    assert fold_session({"sid": "s2", "ab": {"arm": ""}}, [_row(1)]) is None


def _write_session(home: Path, sid: str, arm: str, rows, exit_text=None):
    d = home / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.json").write_text(
        json.dumps({"sid": sid, "ab": {"arm": arm, "rate": 0.05}, "started_ts": 1.0})
    )
    (d / f"{sid}.requests.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\nnot json\n"
    )
    if exit_text:
        (d / f"{sid}.exit").write_text(exit_text)


def test_harvest_is_idempotent_refolds_open_sessions_and_outlives_the_sweep():
    home = Path(os.environ["DISTIL_HOME"])
    _write_session(home, "sA", "distil", [_row(2)], "child exit code 0")
    _write_session(home, "sB", "holdout", [_row(3)])
    (home / "sessions" / "sC.json").write_text(json.dumps({"sid": "sC"}))  # unrandomised
    (home / "sessions" / "sD.json").write_text("{broken")
    now = time.time()
    assert outcomes.harvest(now=now) == 2
    assert outcomes.harvest(now=now) == 0  # sB re-folded (still open), row unchanged
    assert outcomes.load()["sB"].ended == "open"
    assert outcomes.harvest(now=now + outcomes.ABANDON_S * 2) == 1  # open → abandoned
    assert outcomes.load()["sB"].ended == "abandoned"
    for f in (home / "sessions").iterdir():  # the 7-day TTL sweep
        f.unlink()
    assert set(outcomes.load()) == {"sA", "sB"}
    with open(outcomes.store_path(), "a") as fh:
        fh.write('{"v": 99}\nnope\n{"v": 1, "sid": "bad"}\n')
    assert set(outcomes.load()) == {"sA", "sB"}


def test_harvest_never_raises(monkeypatch):
    monkeypatch.setattr(outcomes, "load", lambda: 1 / 0)
    assert outcomes.harvest() == 0


# --------------------------------------------------------------------------- #
# Proxy: a holdout session is forwarded byte-for-byte, and recorded
# --------------------------------------------------------------------------- #

_TOOL_OUT = "\n".join(
    f"line {i}: some verbose tool output that digest would fold" for i in range(40)
)


class _Echo(BaseHTTPRequestHandler):
    seen: list[bytes] = []

    def log_message(self, fmt, *args):  # noqa: ARG002
        pass

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Echo.seen.append(body)
        out = json.dumps(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 100, "output_tokens": 10},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _payload() -> bytes:
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _TOOL_OUT}],
        },
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "and   now   this"},
    ]
    # Non-canonical spacing on purpose: any re-serialisation would change these bytes.
    return json.dumps(
        {"model": "claude-opus-4-8", "max_tokens": 64, "messages": msgs}, indent=1
    ).encode()


@pytest.mark.parametrize("arm", ["holdout", "distil"])
def test_proxy_holdout_forwards_original_bytes_and_tags_the_record(monkeypatch, arm):
    from distil.proxy import build_handler

    monkeypatch.setenv("DISTIL_SESSION", "s9-9")
    _Echo.seen = []
    up = _serve(_Echo)
    try:
        proxy = _serve(
            build_handler(f"http://127.0.0.1:{up.server_address[1]}", arm=arm, shadow_rate=0.5)
        )
        try:
            body = _payload()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_address[1]}/v1/messages",
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "claude-cli/2.1.7 (x)"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                arm_hdr = resp.headers.get("x-distil-arm")
                resp.read()
        finally:
            proxy.shutdown()
    finally:
        up.shutdown()
    rec = json.loads(
        (Path(os.environ["DISTIL_HOME"]) / "sessions" / "s9-9.requests.jsonl")
        .read_text()
        .splitlines()[-1]
    )
    assert rec["arm"] == arm and rec["client"] == "claude-cli/2.1" and rec["user_turn"] is True
    if arm == "holdout":
        assert _Echo.seen == [body]  # byte-for-byte: no Tier-0, no expand tool, no shaping
        assert arm_hdr == "holdout" and rec["booked"] is False and rec["tokens_saved"] == 0
    else:
        assert _Echo.seen[0] != body and arm_hdr is None


def test_wrap_run_assigns_records_and_folds_a_holdout_session(monkeypatch, capsys):
    """End to end: a resumed session keeps the holdout arm from its manifest, the agent's
    request reaches the upstream untouched, and the exit folds its outcome."""
    from distil import ledger
    from distil.proxy import wrap_run

    monkeypatch.setenv("DISTIL_HOT_SWAP", "0")
    monkeypatch.setenv("DISTIL_SESSION", "s5-5")
    monkeypatch.setenv(abtest.RATE_ENV, "0.05")
    ledger.write_session_manifest({"sid": "s5-5", "ab": {"arm": "holdout", "rate": 0.05}})
    _Echo.seen = []
    up = _serve(_Echo)
    Path(os.environ["DISTIL_HOME"], "payload.json").write_bytes(_payload())
    child = (
        "import os,urllib.request;"
        "b=open(os.path.join(os.environ['DISTIL_HOME'],'payload.json'),'rb').read();"
        "r=urllib.request.Request(os.environ['ANTHROPIC_BASE_URL']+'/v1/messages',data=b,"
        "headers={'Content-Type':'application/json'},method='POST');"
        "urllib.request.urlopen(r).read()"
    )
    try:
        code = wrap_run(
            [sys.executable, "-c", child],
            upstream=f"http://127.0.0.1:{up.server_address[1]}",
            record=True,
        )
    finally:
        up.shutdown()
    assert code == 0
    assert _Echo.seen == [_payload()]
    assert "A/B holdout: this session runs uncompressed" in capsys.readouterr().out
    man = json.loads(ledger.session_manifest_path("s5-5").read_text())
    assert man["ab"] == {"arm": "holdout", "rate": 0.05}
    o = outcomes.load()["s5-5"]
    assert (o.arm, o.ended, o.turns) == ("holdout", "ok", 1)
    assert o.cost == pytest.approx(100 * 5e-6 + 10 * 25e-6)


# --------------------------------------------------------------------------- #
# Statistics — primitives
# --------------------------------------------------------------------------- #


def test_ratio_contrast_edges():
    assert stats.ratio_contrast([stats.Unit(1, 1, True)]) is None  # one arm, one unit
    zero = [
        stats.Unit(0, 1, True),
        stats.Unit(0, 1, True),
        stats.Unit(1, 1, False),
        stats.Unit(2, 1, False),
    ]
    assert stats.ratio_contrast(zero) is None
    flat = [stats.Unit(1, 1, t) for t in (True, True, False, False)]
    assert stats.ratio_contrast(flat) is None  # zero variance → no claim
    assert stats.pool([]) is None


def test_cs_matches_the_evalue_threshold():
    """The CS boundary and the e-value crossing 1/α are the same event."""
    a, var = 0.05, 0.004
    h = stats.cs_halfwidth(var, a)
    at = stats.msprt_log_evalue(h, var)
    assert at == pytest.approx(math.log(1 / a), rel=1e-9)
    assert stats.msprt_log_evalue(h * 0.99, var) < math.log(1 / a)


def test_holdout_needed_shrinks_with_less_noise_and_bigger_rate():
    assert stats.holdout_needed(1.0, 0.05) > stats.holdout_needed(0.25, 0.05)
    # Fewer HOLDOUT sessions at a small rate (the distil arm is huge), but far more in total.
    assert stats.holdout_needed(1.0, 0.05) / 0.05 > stats.holdout_needed(1.0, 0.5) / 0.5


def test_cusum_detects_a_shift_and_places_it():
    rng = random.Random(3)
    ys = [rng.gauss(0, 1) for _ in range(200)] + [rng.gauss(1.5, 1) for _ in range(100)]
    cuts = stats.cusum_changepoints(ys)
    assert cuts and 195 <= cuts[0][0] <= 215 and cuts[0][0] <= cuts[0][1] <= 240
    assert stats.cusum_changepoints([1.0] * 5) == []


def test_cusum_false_alarm_rate_is_low_on_stationary_data():
    rng = random.Random(11)
    alarms = sum(
        bool(stats.cusum_changepoints([rng.gauss(0, 1) for _ in range(150)])) for _ in range(200)
    )
    assert alarms / 200 < 0.03  # measured 0% at 150 points, ~1% by 3,000, with h = 12


# --------------------------------------------------------------------------- #
# Statistics — known answers
# --------------------------------------------------------------------------- #


def test_known_effect_is_recovered_with_coverage():
    s = summarize(sim(1200, effect=-0.2, seed=1), rate=0.5, now=3e9)
    assert s.status == "ok" and s.cost is not None
    assert s.cost.lo < -0.2 < s.cost.hi and s.cost.significant
    assert abs(s.cost.rel + 0.2) < 0.06
    assert s.turns is not None and s.turns.lo < 0 < s.turns.hi  # distil does not move turns
    assert s.bootstrap is not None and s.bootstrap[0] < -0.2 < s.bootstrap[1]
    assert s.holdout_cost is not None and s.holdout_cost[0] > 0  # holding out cost money
    assert "cost per task" in s.message


def test_silent_model_change_in_both_arms_is_not_attributed_to_distil():
    """Same model id, +40% turns per task from session 600 on, in BOTH arms.

    A before/after on the distil arm reads it as +40% cost; the concurrent contrast
    still reports the true −20% and CUSUM opens a new era."""
    before = sim(600, effect=-0.2, seed=2, sigma=0.3)
    after = sim(600, effect=-0.2, seed=3, sigma=0.3, turns_mult=1.4, t0=before[-1].start + 1000)
    naive = (
        sum(o.cost for o in after if o.arm == "distil") / sum(1 for o in after if o.arm == "distil")
    ) / (
        sum(o.cost for o in before if o.arm == "distil")
        / sum(1 for o in before if o.arm == "distil")
    ) - 1
    assert naive > 0.25  # what a before/after would book to distil
    s = summarize(before + after, rate=0.5, now=3e9)
    assert s.cost is not None and s.cost.lo < -0.2 < s.cost.hi and abs(s.cost.rel + 0.2) < 0.08
    assert any(e.kind == "shift" for e in s.events)
    assert s.model_change and "behaviour changed" in s.model_change
    assert s.did == []  # a CUSUM era is not a rollout: no DiD across it
    # Pooling across the change (no eras at all) is still unbiased — randomisation,
    # not the change point, is what protects the contrast.
    pooled = stats.ratio_contrast(
        [stats.Unit(o.cost, 1, o.arm == "distil") for o in before + after]
    )
    assert pooled is not None and abs(pooled.rel + 0.2) < 0.06


def test_bias_under_model_change_monte_carlo():
    errs = []
    for rep in range(25):
        a = sim(300, seed=100 + rep)
        b = sim(300, seed=200 + rep, turns_mult=1.4, model="claude-opus-5", t0=a[-1].start + 1000)
        s = summarize(a + b, rate=0.5, now=3e9)
        assert s.cost is not None
        errs.append(s.cost.rel + 0.2)
    assert abs(sum(errs) / len(errs)) < 0.02


def test_explicit_model_change_compares_within_the_new_model_only():
    a = sim(500, effect=-0.1, seed=4)
    b = sim(500, effect=-0.3, seed=5, turns_mult=1.4, model="claude-opus-5", t0=a[-1].start + 1000)
    s = summarize(a + b, rate=0.5, now=3e9)
    assert s.cost is not None and s.cost.lo < -0.3 < s.cost.hi
    assert [c.model for c in s.cells if c.current] == ["claude-opus-5"]
    assert s.model_change and "model changed" in s.model_change
    assert any("claude-opus-4-8 → claude-opus-5" in e.text for e in s.events)
    d = s.did[0]
    assert d.effect.lo < (0.7 / 0.9 - 1) < d.effect.hi and d.effect.rel < 0
    assert "claude-opus-5" in render(s)


def test_change_in_one_stratum_leaves_the_other_alone():
    a1 = sim(400, effect=-0.3, seed=6)
    a2 = sim(
        400, effect=-0.3, seed=7, turns_mult=1.4, model="claude-opus-5", t0=a1[-1].start + 1000
    )
    b = sim(800, effect=-0.1, seed=8, model="claude-sonnet-4-6", prefix="b", t0=1_000_500.0)
    s = summarize(a1 + a2 + b, rate=0.5, now=3e9)
    sonnet = [c for c in s.cells if c.model == "claude-sonnet-4-6"]
    assert len(sonnet) == 1 and sonnet[0].current and sonnet[0].cost is not None
    assert sonnet[0].cost.lo < -0.1 < sonnet[0].cost.hi
    assert {e.group for e in s.events} == {"claude-opus | claude-cli"}
    assert {c.model for c in s.cells if c.current} == {"claude-opus-5", "claude-sonnet-4-6"}
    assert s.cost is not None and -0.3 < s.cost.rel < -0.1  # precision-weighted blend


def test_cuped_narrows_the_interval_on_correlated_data():
    data = sim(1000, seed=9, ws_sd=1.0, sigma=0.4, n_ws=40)
    with_cov = summarize(data, rate=0.5, now=3e9)
    no_cov = summarize([_no_ws(o) for o in data], rate=0.5, now=3e9)
    assert with_cov.cost is not None and no_cov.cost is not None
    assert with_cov.cost.variance_reduction > 0.3
    assert (with_cov.cost.hi - with_cov.cost.lo) < 0.8 * (no_cov.cost.hi - no_cov.cost.lo)
    assert with_cov.cost.lo < -0.2 < with_cov.cost.hi


def _no_ws(o: SessionOutcome) -> SessionOutcome:
    from dataclasses import replace

    return replace(o, ws="")


def test_sequential_type_one_error_under_repeated_looks():
    """Null effect, a look every 20 sessions up to 400 (19 looks), 300 replications.

    The mixture CS is anytime-valid only asymptotically (plug-in variance), so the
    tolerance is loose: ≤ 2α. The naive fixed-n 95% interval checked at every look
    is shown to inflate well past α on the same data — that is the failure mode the
    sequence exists to prevent."""
    a = 0.05
    z = 1.959964
    seq = naive = 0
    for rep in range(300):
        data = sim(400, effect=0.0, seed=1000 + rep, sigma=0.8)
        hit_s = hit_n = False
        for n in range(40, 401, 20):
            c = stats.ratio_contrast(
                [stats.Unit(o.cost, 1, o.arm == "distil") for o in data[:n]], cuped=False
            )
            if c is None:
                continue
            hit_s = hit_s or c.significant(a)
            hit_n = hit_n or abs(c.delta) > z * math.sqrt(c.var)
        seq += hit_s
        naive += hit_n
    assert seq / 300 <= 2 * a
    assert naive / 300 > seq / 300 and naive / 300 > 1.5 * a


def test_power_at_a_realistic_size():
    """−20% at session-cost CV ≈ 0.75, one look, 50/50 split: ~40% power at n=400
    (the price of anytime validity at a single look), ≥80% at n=800."""
    hits = 0
    for rep in range(60):
        c = stats.ratio_contrast(
            [
                stats.Unit(o.cost, 1, o.arm == "distil")
                for o in sim(800, effect=-0.2, seed=2000 + rep)
            ],
            cuped=False,
        )
        hits += bool(c and c.significant(0.05))
    assert hits / 60 >= 0.8


# --------------------------------------------------------------------------- #
# Floors, eras, surfaces
# --------------------------------------------------------------------------- #


def test_not_enough_data_names_n_and_need():
    s = summarize(sim(60, p=0.05, seed=12), rate=0.05, now=3e9)
    assert s.status == "insufficient" and s.need_holdout and s.need_holdout >= stats.MIN_HOLDOUT
    assert s.message.startswith("not enough data yet (n=") and "need ≈" in s.message
    assert "not enough data yet" in render(s)


def test_no_data_disabled_open_and_window():
    assert summarize([], rate=0.05).status == "no-data"
    assert summarize([], rate=0.0).status == "disabled"
    data = sim(100, seed=13)
    from dataclasses import replace

    data[0] = replace(data[0], ended="open")
    data[1] = replace(data[1], cost=None)
    s = summarize(data, rate=0.5, now=data[-1].start + 10, window=(50 * 1000) / 86400)
    assert s.n_sessions <= 51 and s.n_open == 0
    s = summarize(data, rate=0.5, now=3e9)
    assert s.n_open == 1 and s.n_unpriced == 1


def test_rate_change_opens_an_era():
    a = sim(200, seed=14, rate=0.05)
    b = sim(200, seed=15, rate=0.1, t0=a[-1].start + 1000)
    eras = split_eras(a + b)
    assert len(eras) >= 2 and any(kind == "rate" for _, kind, _ in eras)


def test_family_strips_versions_not_names():
    assert family("claude-opus-4-8-20260101") == "claude-opus"
    assert family("anthropic.claude-sonnet-4-5@2025") == "claude-sonnet"
    assert family("gpt-4o-2024-08-06") == "gpt-4o"


def test_render_full_report_and_caveats():
    from dataclasses import replace

    data = sim(800, seed=16, ws_sd=0.8)
    data = [
        replace(o, billing="subscription", expanded=1) if i == 0 else o for i, o in enumerate(data)
    ]
    out = render(summarize(data, rate=0.5, now=3e9))
    for needle in (
        "cost per task",
        "turns per task",
        "cost per turn",
        "bootstrap cross-check",
        "cost of the holdout",
        "notional",
        "strata (model | client, era)",
    ):
        assert needle in out


def test_cli_ab_text_json_and_set_rate(capsys, monkeypatch):
    from distil.cli import main

    outcomes.append(sim(300, seed=17))
    monkeypatch.setenv(abtest.RATE_ENV, "0.5")
    assert main(["ab"]) == 0
    assert "distil ab — task-level A/B" in capsys.readouterr().out
    assert main(["ab", "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["status"] == "ok" and js["cost"]["n_holdout"] > 0
    monkeypatch.delenv(abtest.RATE_ENV)
    assert main(["ab", "--holdout-rate", "0"]) == 0
    assert "off" in capsys.readouterr().out and abtest.holdout_rate() == 0.0
    assert main(["ab", "--holdout-rate", "0.9"]) == 2
    assert abtest.abtest_summary().status == "ok"
