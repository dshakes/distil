"""`distil statusline` — the Claude Code plugin status line.

It must always print one line, never raise, and reflect the local savings ledger.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import pytest


from distil import ledger
from distil.cli import cmd_statusline


def test_humanize_tokens():
    assert ledger._human(0) == "0"
    assert ledger._human(999) == "999"
    assert ledger._human(14_417) == "14.4K"
    assert ledger._human(1_200_000) == "1.2M"


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path):
    """A real wrap session exports DISTIL_SESSION (switching the live slice to
    session= lookups) and the de segment reads $DISTIL_HOME/shadow.jsonl — both
    must be pinned or these tests' results depend on the developer's machine."""
    monkeypatch.delenv("DISTIL_SESSION", raising=False)
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "distil-home"))
    # These tests exercise the live/idle displays of a ROUTED session; route via
    # loopback base URL (not DISTIL_SESSION, which would switch the live slice
    # to session= lookups). The unrouted "off" label is covered in test_cli_onboard.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")


def _run(monkeypatch, capsys, summary, recent=None, stdin="{}", no_color=True):
    """summary() → lifetime; summary(since=…) → recent window (empty by default,
    so these tests exercise the lifetime/idle view unless a test sets `recent`)."""
    empty = ledger.LedgerSummary(0, 0.0, 0, {})

    def _summary(*a, **k):
        live = k.get("since") is not None or k.get("session")  # _live() slice, either form
        return (recent if recent is not None else empty) if live else summary

    monkeypatch.setattr(ledger, "summary", _summary)
    monkeypatch.setattr(ledger, "latest_session", lambda *a, **k: ("", 0.0))
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    rc = cmd_statusline(argparse.Namespace(no_color=no_color))
    return rc, capsys.readouterr().out.strip()


def test_empty_ledger_shows_hint(monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, ledger.LedgerSummary(0, 0.0, 0, {}))
    assert rc == 0
    assert out.startswith("distil")
    assert "no savings yet" in out


def test_populated_ledger(monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")  # force metered: dollars shown
    s = ledger.LedgerSummary(
        3,
        0.0400,
        21_600,
        {"live-proxy": 0.04},
        total_baseline_tokens=50_000,
        total_distil_tokens=28_400,
        total_baseline_dollars=0.10,
        total_distil_dollars=0.06,
    )
    rc, out = _run(monkeypatch, capsys, s, recent=s)  # recent activity with savings
    assert rc == 0
    # unified grammar: live ▼ + rate + $, then a bare `total ▼`
    assert "▼21.6K" in out
    assert "43% smaller" in out  # trim rate, the glanceable number
    assert "$0.04" in out
    assert "total ▼21.6K" in out  # lifetime, always the same bare format
    assert "runs" not in out  # run counts live in `distil stats`, not the line


def test_subscription_hides_notional_dollars(monkeypatch, capsys):
    # On a flat-rate subscription the dollar figure is notional, so it's hidden;
    # the orig -> compressed token reduction (the real win) still shows.
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    s = ledger.LedgerSummary(
        3,
        0.04,
        21_600,
        {},
        total_baseline_tokens=50_000,
        total_distil_tokens=28_400,
        total_baseline_dollars=0.10,
        total_distil_dollars=0.06,
    )
    rc, out = _run(monkeypatch, capsys, s, recent=s)
    assert rc == 0
    assert "▼21.6K" in out and "total ▼21.6K" in out
    assert "$" not in out


def test_equivalence_health_color(monkeypatch, capsys, tmp_path):
    """eq% is colored by health: green >=99%, yellow >=95%, red below."""
    import distil.shadow as shadow

    s = ledger.LedgerSummary(1, 0.01, 100, {}, total_baseline_tokens=1000, total_distil_tokens=600)

    def led_with(changes: int, samples: int = 100):
        led = shadow.ShadowLedger()
        # Perfect A/A baseline (30 self-agreements, 0 changes) so adjusted_rate()
        # == rate() and the verdict path runs — the point of THIS test.
        led.aa_samples = 30
        led.aa_changes = 0
        for i in range(samples):
            led.samples += 1
            eq = i >= changes
            if not eq:
                led.changes += 1
            led.recent.append(1 if eq else 0)
        return led

    # theme-stable 256-color hues + health glyph: ✓ teal, ⚠ yellow, ✗ red
    for changes, code in (
        (0, "\033[38;5;86m✓"),
        (3, "\033[38;5;220m⚠"),
        (10, "\033[38;5;196m✗"),
    ):
        monkeypatch.setattr(
            shadow.ShadowLedger, "load", classmethod(lambda cls, *a, _l=led_with(changes), **k: _l)
        )
        _rc, out = _run(monkeypatch, capsys, s, no_color=False)
        assert f"{code}de " in out, out


def test_no_verdict_glyph_before_aa_baseline(monkeypatch, capsys, tmp_path):
    """Enough A/B samples but no robust A/A noise baseline → show 'de baseline N/30',
    NOT a red ✗ verdict. adjusted_rate() falls back to the raw rate when the
    baseline is missing; painting a degraded glyph over it reads sampling
    nondeterminism as compression harm (the false-alarm bug)."""
    import distil.shadow as shadow

    s = ledger.LedgerSummary(1, 0.01, 100, {}, total_baseline_tokens=1000, total_distil_tokens=600)

    led = shadow.ShadowLedger()
    led.aa_samples = 3  # baseline not ready (< 30)
    for i in range(60):  # A/B sample count clears the gate, so baseline is the blocker
        led.samples += 1
        eq = i >= 24
        if not eq:
            led.changes += 1
        led.recent.append(1 if eq else 0)

    # fresh shadow.jsonl so the segment isn't "de idle"
    (tmp_path / "shadow.jsonl").write_text("{}\n")
    monkeypatch.setattr("distil.shadow._state_dir", lambda: tmp_path)
    monkeypatch.setattr(shadow.ShadowLedger, "load", classmethod(lambda cls, *a, _l=led, **k: _l))

    _rc, out = _run(monkeypatch, capsys, s, no_color=False)
    assert "✗" not in out, out  # no false degraded verdict
    assert "de baseline 3/30" in out, out


def test_run_counts_not_in_line(monkeypatch, capsys):
    # run counts moved to `distil stats` — the composite line stays compact
    s = ledger.LedgerSummary(1, 0.01, 500, {"x": 0.01})
    _rc, out = _run(monkeypatch, capsys, s)
    assert "run" not in out
    assert out.startswith("distil")


def test_model_name_from_stdin(monkeypatch, capsys):
    s = ledger.LedgerSummary(2, 0.02, 1000, {})
    _rc, out = _run(monkeypatch, capsys, s, stdin='{"model":{"display_name":"Claude Opus 4.8"}}')
    assert "Claude Opus 4.8" in out


def test_never_raises_on_bad_stdin(monkeypatch, capsys):
    s = ledger.LedgerSummary(1, 0.01, 100, {})
    rc, out = _run(monkeypatch, capsys, s, stdin="not json {{{")
    assert rc == 0
    assert out.startswith("distil")


def test_color_codes_present_by_default(monkeypatch, capsys):
    s = ledger.LedgerSummary(1, 0.01, 100, {})
    _rc, out = _run(monkeypatch, capsys, s, no_color=False)
    assert "\033[" in out


_REPO = Path(__file__).resolve().parent.parent


def test_main_statusline_flushes_and_hard_exits(monkeypatch, capsys):
    """The fix's mechanism, version-independently: main(["statusline"]) delivers
    its line, then hard-exits via os._exit so the interpreter's shutdown flush
    can never fault on a pipe the consumer already closed. We patch os._exit
    (which would otherwise kill the test runner) to observe the call."""
    import os

    import pytest

    import distil.cli as cli

    exited = {}

    def fake_exit(code):
        exited["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    monkeypatch.setattr(ledger, "summary", lambda *a, **k: ledger.LedgerSummary(1, 0.01, 100, {}))

    with pytest.raises(SystemExit):
        cli.main(["statusline"])

    assert exited.get("code") == 0
    assert "distil" in capsys.readouterr().out


def test_no_broken_pipe_when_reader_closes_early(tmp_path):
    """End-to-end: invoke distil the way its console-script does — a .py run as
    __main__ that imports the package — with the reader closing the pipe early.
    No BrokenPipeError may reach stderr. (The `-m` form does NOT reproduce this;
    it only manifests on Python 3.13+, so on older interpreters this is a smoke
    test that must simply stay clean.)"""
    wrapper = tmp_path / "run.py"
    wrapper.write_text("import sys\nfrom distil.cli import main\nsys.exit(main())\n")
    for _ in range(8):
        proc = subprocess.run(
            f"{sys.executable} {wrapper} statusline | true",
            shell=True,
            cwd=_REPO,
            input="{}",
            text=True,
            stderr=subprocess.PIPE,
        )
        assert "BrokenPipeError" not in proc.stderr, proc.stderr
        assert "Broken pipe" not in proc.stderr, proc.stderr


def test_render_dashboard_shows_orig_and_compressed():
    s = ledger.LedgerSummary(
        2,
        5.0,
        1_000_000,
        {"live-proxy": 5.0},
        total_baseline_tokens=2_000_000,
        total_distil_tokens=1_000_000,
        total_baseline_dollars=10.0,
        total_distil_dollars=5.0,
    )
    out = ledger.render_dashboard(s, change_rate=0.01, samples=200, color=False)
    assert "2.0M → 1.0M" in out  # orig -> compressed tokens
    assert "50.0% trimmed" in out
    assert "$10.00 → $5.00" in out
    assert "99.0%" in out  # decision-equivalence = 1 - change_rate
    assert "200 samples" in out


def test_render_dashboard_subscription_hides_cost():
    s = ledger.LedgerSummary(
        1,
        5.0,
        1_000_000,
        {},
        total_baseline_tokens=2_000_000,
        total_distil_tokens=1_000_000,
        total_baseline_dollars=10.0,
        total_distil_dollars=5.0,
    )
    out = ledger.render_dashboard(s, subscription=True, color=False)
    assert "notional" in out
    assert "$10.00 → $5.00" not in out


def test_render_dashboard_recent_strip():
    # The live shadow panel: recent decisions shown as a strip (▰ same / ▱ changed).
    s = ledger.LedgerSummary(
        5,
        5.0,
        1000,
        {},
        total_baseline_tokens=2_000_000,
        total_distil_tokens=1_000_000,
        total_baseline_dollars=10.0,
        total_distil_dollars=5.0,
    )
    out = ledger.render_dashboard(
        s, change_rate=0.2, samples=10, recent=[1, 1, 0, 1, 1], color=False
    )
    assert "recent" in out
    assert "▰" in out and "▱" in out  # equivalent + changed marks present


def test_recent_window_leads_lifetime_follows(monkeypatch, capsys, tmp_path):
    """Live ▼ = RECENT activity (15-min window, aggregates all terminals — no
    single-session flicker); total = lifetime."""
    import json as _json
    import time

    from distil import ledger as led_mod

    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    path = tmp_path / "savings.jsonl"
    # an OLD record (ts=1000, outside the 15-min window) — lifetime only
    old = {
        "trajectory_id": "live-proxy",
        "model": "claude-opus-4-8",
        "turns": 5,
        "baseline_dollars": 1.0,
        "distil_dollars": 0.4,
        "baseline_input_tokens": 200_000,
        "distil_input_tokens": 80_000,
        "tokenizer": "heuristic",
        "ts": 1000.0,
        "session": "old",
    }
    path.write_text(_json.dumps(old) + "\n")
    # a RECENT record (now) — this is the live number, from a different session
    led_mod.record(
        trajectory_id="live-proxy",
        model="claude-opus-4-8",
        turns=3,
        baseline_dollars=0.5,
        distil_dollars=0.2,
        baseline_input_tokens=100_000,
        distil_input_tokens=40_000,
        session=f"s{int(time.time())}-99",
        path=path,
    )
    monkeypatch.setattr(led_mod, "default_path", lambda: path)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))
    rc = cmd_statusline(argparse.Namespace(no_color=True))
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert "▼60.0K · 60% smaller" in out  # recent activity, not lifetime
    assert "total ▼180.0K" in out  # lifetime = old + recent
    assert "$0.30" in out  # recent dollars, not lifetime


def test_lifetime_fallback_when_session_stale(monkeypatch, capsys, tmp_path):
    """No live session (>4h idle) → the familiar lifetime view."""
    import json as _json

    from distil import ledger as led_mod

    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    path = tmp_path / "savings.jsonl"
    rec = {
        "trajectory_id": "live-proxy",
        "model": "claude-opus-4-8",
        "turns": 2,
        "baseline_dollars": 1.0,
        "distil_dollars": 0.5,
        "baseline_input_tokens": 50_000,
        "distil_input_tokens": 25_000,
        "tokenizer": "heuristic",
        "ts": 1000.0,
        "session": "ancient",
    }
    path.write_text(_json.dumps(rec) + "\n")
    monkeypatch.setattr(led_mod, "default_path", lambda: path)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))
    cmd_statusline(argparse.Namespace(no_color=True))
    out = capsys.readouterr().out.strip()
    # idle (no recent traffic): set-up-and-on, then the same bare `total ▼`
    assert "✓ on" in out
    assert "total ▼25.0K" in out
    assert "saved · 50% smaller" not in out  # no special idle formatting anymore
    assert "▼" not in out.split("total")[0]  # no live savings figure when idle


def test_eq_suppressed_below_min_samples(monkeypatch, capsys):
    """de 100.0% (a rate) over 1 sample is noise wearing a number — suppressed until the gate."""
    import distil.shadow as shadow

    s = ledger.LedgerSummary(1, 0.01, 100, {}, total_baseline_tokens=1000, total_distil_tokens=600)
    led = shadow.ShadowLedger()
    led.samples = 1
    led.recent.append(1)
    monkeypatch.setattr(shadow.ShadowLedger, "load", classmethod(lambda cls, *a, **k: led))
    sj = shadow._state_dir() / "shadow.jsonl"  # sampler is live: file freshly fed
    sj.parent.mkdir(parents=True, exist_ok=True)
    sj.write_text('{"equivalent": true}\n', encoding="utf-8")
    _rc, out = _run(monkeypatch, capsys, s)
    assert "%" not in out  # no rate claimed below the sample gate
    assert "de 1/50" in out  # but collection progress is shown ("warming up")


def test_zero_savings_session_says_watching(monkeypatch, capsys, tmp_path):
    """Traffic flowing but nothing trimmed yet must read as 'watching', not ▼0 −0%."""
    import time

    from distil import ledger as led_mod

    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    path = tmp_path / "savings.jsonl"
    led_mod.record(
        trajectory_id="live-proxy",
        model="claude-opus-4-8",
        turns=4,
        baseline_dollars=0.1,
        distil_dollars=0.1,
        baseline_input_tokens=12_000,
        distil_input_tokens=12_000,
        session=f"s{int(time.time())}-7",
        path=path,
    )
    monkeypatch.setattr(led_mod, "default_path", lambda: path)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))
    cmd_statusline(argparse.Namespace(no_color=True))
    out = capsys.readouterr().out.strip()
    # unmistakable: distil is ON, waiting for large content — not a bare "watching"
    assert "✓ on" in out and "waiting for a large read" in out
    assert "▼" not in out.split("total")[0]  # no live ▼ savings before 'total'


def test_flush_skips_zero_baseline_records(tmp_path):
    """A flush window with zero measured tokens writes nothing (no noise records)."""
    from distil.runtime import RuntimeSavings

    led = tmp_path / "savings.jsonl"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    rs.record(0, 0)  # a request passed through with nothing measurable
    assert rs.flush() is True  # counters reset...
    assert not led.exists()  # ...but no record was written


def test_total_segment_identical_across_all_states(monkeypatch, capsys):
    """CONSISTENCY GUARD (the bug a user caught): the lifetime `total ▼` segment
    and the overall `distil · <live> · total ▼…` shape must hold in EVERY state —
    idle, watching, and saving. This is the cross-state check my per-state tests
    were missing."""
    import re

    from distil import ledger as led_mod

    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    monkeypatch.setattr(led_mod, "latest_session", lambda *a, **k: ("", 0.0))
    # Isolate the total-consistency invariant from the shadow segment: real on-disk
    # samples would otherwise append eq/de after `total ▼`, which this test asserts is last.
    import distil.shadow as _shadow

    monkeypatch.setattr(
        _shadow.ShadowLedger, "load", classmethod(lambda cls, *a, **k: _shadow.ShadowLedger())
    )
    life = ledger.LedgerSummary(
        9,
        0.0,
        27_000_000,
        {"live-proxy": 1.0},
        total_baseline_tokens=54_000_000,
        total_distil_tokens=27_000_000,
    )
    states = {
        "idle": ledger.LedgerSummary(0, 0.0, 0, {}),
        "watching": ledger.LedgerSummary(
            2, 0.0, 0, {}, total_baseline_tokens=46_000, total_distil_tokens=46_000
        ),
        "saving": ledger.LedgerSummary(
            3, 0.0, 12_000, {}, total_baseline_tokens=30_000, total_distil_tokens=18_000
        ),
    }
    outs = {}
    for name, recent in states.items():
        monkeypatch.setattr(
            led_mod,
            "summary",
            lambda *a, _r=recent, **k: _r if k.get("since") is not None else life,
        )
        monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
        cmd_statusline(argparse.Namespace(no_color=True))
        outs[name] = capsys.readouterr().out.strip()

    totals = {name: re.search(r"total ▼\S+", o).group(0) for name, o in outs.items()}
    assert len(set(totals.values())) == 1, f"total segment differs across states: {totals}"
    for name, o in outs.items():
        assert o.startswith("distil ·") and o.endswith("total ▼27.0M"), (name, o)


def test_per_session_via_distil_session_env(monkeypatch, tmp_path):
    """PROD per-session: DISTIL_SESSION (stamped by `distil wrap`, inherited by
    the status line) filters the live ▼ to THIS session; total stays lifetime.
    Two terminals on the same machine must NOT bleed into each other."""
    import io

    from distil import ledger as led_mod
    from distil.runtime import RuntimeSavings

    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    path = tmp_path / "savings.jsonl"
    monkeypatch.setattr(led_mod, "default_path", lambda: path)

    # session A records 30K saved; the RuntimeSavings id comes from DISTIL_SESSION
    monkeypatch.setenv("DISTIL_SESSION", "sess-A")
    rs = RuntimeSavings(model="claude-fable-5", ledger_path=path)
    assert rs.session_id == "sess-A"
    rs.record(50_000, 20_000)
    rs.flush()

    import contextlib

    outA, outB = io.StringIO(), io.StringIO()
    monkeypatch.setenv("DISTIL_SESSION", "sess-A")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    with contextlib.redirect_stdout(outA):
        cmd_statusline(argparse.Namespace(no_color=True))
    monkeypatch.setenv("DISTIL_SESSION", "sess-B")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    with contextlib.redirect_stdout(outB):
        cmd_statusline(argparse.Namespace(no_color=True))

    a, b = outA.getvalue().strip(), outB.getvalue().strip()
    assert "▼30.0K" in a and "60% smaller" in a  # session A sees its own savings
    assert "✓ on" in b and "▼" not in b.split("total")[0]  # session B: none of its own
    assert "total ▼30.0K" in a and "total ▼30.0K" in b  # shared lifetime total


# ---------------------------------------------------------------------------
# Bypass detection: "✓ on" must mean traffic actually flows through the proxy.
# A wrapped OAuth agent can keep the env vars while sending its model calls
# straight to the provider — the marker (written "0" by wrap_run, flipped "1"
# by the proxy's first POST) is how the statusline tells the two apart.
# ---------------------------------------------------------------------------


def _mk_marker(value: str, age_s: float = 600.0) -> Path:
    import os
    import time

    mp = ledger.session_marker_path()
    assert mp is not None
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(value, encoding="utf-8")
    old = time.time() - age_s
    os.utime(mp, (old, old))
    return mp


def test_bypass_warning_when_marker_stays_zero(monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SESSION", "sTEST-bypass")
    _mk_marker("0")
    rc, out = _run(monkeypatch, capsys, ledger.LedgerSummary(0, 0.0, 0, {}))
    assert rc == 0
    assert "bypassing" in out
    assert "✓ on" not in out


def test_no_bypass_warning_within_grace_period(monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SESSION", "sTEST-fresh")
    _mk_marker("0", age_s=10.0)  # wrap just started — agent hasn't called yet
    rc, out = _run(monkeypatch, capsys, ledger.LedgerSummary(0, 0.0, 0, {}))
    assert rc == 0
    assert "✓ on" in out
    assert "bypassing" not in out


def test_no_bypass_warning_once_traffic_flows(monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SESSION", "sTEST-flows")
    _mk_marker("1")  # proxy saw a request
    rc, out = _run(monkeypatch, capsys, ledger.LedgerSummary(0, 0.0, 0, {}))
    assert rc == 0
    assert "✓ on" in out
    assert "bypassing" not in out


def test_bypass_warning_replaces_idle_on_with_lifetime_total(monkeypatch, capsys):
    """Populated lifetime ledger + idle bypassed session: warn, keep the total."""
    monkeypatch.setenv("DISTIL_SESSION", "sTEST-idle")
    _mk_marker("0")
    s = ledger.LedgerSummary(
        3,
        0.04,
        21_600,
        {"live-proxy": 0.04},
        total_baseline_tokens=50_000,
        total_distil_tokens=28_400,
    )
    rc, out = _run(monkeypatch, capsys, s)  # recent slice empty → idle branch
    assert rc == 0
    assert "bypassing" in out
    assert "total ▼21.6K" in out
    assert "✓ on" not in out


def _write_row(ts: float) -> None:
    """One ledger row stamped *ts*, as an always-on proxy would record it —
    under its OWN session id, never the wrap session's."""
    import json

    p = ledger.default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, "session": "s-always-on-1234", "mode": "digest"}) + "\n")


def test_no_bypass_warning_when_an_always_on_proxy_is_serving(monkeypatch, capsys):
    """The false positive: wrap + always-on together.

    An always-on install pins ANTHROPIC_BASE_URL in a settings file, and that
    pin outranks the env var `distil wrap` injects — so the agent talks to the
    always-on proxy, which stamps rows with its own session id and cannot know
    the wrap session's. The marker therefore stays "0" for the whole session and
    the old check warned "agent bypassing proxy" at a session that was fully
    routed, forever, with no way for the user to clear it.
    """
    monkeypatch.setenv("DISTIL_SESSION", "sTEST-alwayson")
    mp = _mk_marker("0", age_s=600.0)
    _write_row(mp.stat().st_mtime + 60)  # a request served after this wrap started
    rc, out = _run(monkeypatch, capsys, ledger.LedgerSummary(0, 0.0, 0, {}))
    assert rc == 0
    assert "bypassing" not in out, f"warned at a routed session: {out}"
    assert "✓ on" in out


def test_bypass_warning_survives_a_ledger_that_went_quiet(monkeypatch, capsys):
    """Rows from BEFORE this session started are not evidence about it.

    Otherwise any machine with history would suppress the warning permanently
    and the detection would be dead code.
    """
    monkeypatch.setenv("DISTIL_SESSION", "sTEST-quiet")
    mp = _mk_marker("0", age_s=600.0)
    _write_row(mp.stat().st_mtime - 60)  # last traffic predates this wrap
    rc, out = _run(monkeypatch, capsys, ledger.LedgerSummary(0, 0.0, 0, {}))
    assert rc == 0
    assert "bypassing" in out


def test_bypass_warning_minimal_mode(monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_STATUSLINE", "minimal")
    monkeypatch.setenv("DISTIL_SESSION", "sTEST-min")
    _mk_marker("0")
    rc, out = _run(monkeypatch, capsys, ledger.LedgerSummary(0, 0.0, 0, {}))
    assert rc == 0
    assert "⚠ bypassed" in out


def test_de_shows_progress_only_while_sampler_is_live(monkeypatch, capsys):
    """Honesty gap #3: a frozen sub-25 counter must read "de idle", not imply
    active measurement; a recently-fed shadow ledger shows real progress."""
    import os
    import time

    import distil.shadow as shadow

    led = shadow.ShadowLedger()
    led.samples = 3
    monkeypatch.setattr(shadow.ShadowLedger, "load", classmethod(lambda cls, *a, **k: led))
    s = ledger.LedgerSummary(1, 0.01, 100, {}, total_baseline_tokens=1000, total_distil_tokens=600)

    sj = shadow._state_dir() / "shadow.jsonl"
    sj.parent.mkdir(parents=True, exist_ok=True)
    sj.write_text('{"equivalent": true}\n', encoding="utf-8")

    _rc, out = _run(monkeypatch, capsys, s, recent=s)
    assert "de 3/50" in out  # fresh file → genuinely collecting

    old = time.time() - 2 * 86400
    os.utime(sj, (old, old))
    _rc, out = _run(monkeypatch, capsys, s, recent=s)
    assert "de idle" in out  # stale >24h → no sampler feeding it
    assert "3/50" not in out


def test_wrap_shadow_defaults_on():
    """Intelligence is the default: decision-equivalence sampling runs at 2%
    unless explicitly disabled (--shadow 0 is the opt-out)."""
    from distil.cli import build_parser

    ns = build_parser().parse_args(["wrap", "--", "echo", "hi"])
    assert ns.shadow == 0.02
