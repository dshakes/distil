"""`distil statusline` — the Claude Code plugin status line.

It must always print one line, never raise, and reflect the local savings ledger.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path


from distil import ledger
from distil.cli import cmd_statusline


def test_humanize_tokens():
    assert ledger._human(0) == "0"
    assert ledger._human(999) == "999"
    assert ledger._human(14_417) == "14.4K"
    assert ledger._human(1_200_000) == "1.2M"


def _run(monkeypatch, capsys, summary, stdin="{}", no_color=True):
    monkeypatch.setattr(ledger, "summary", lambda *a, **k: summary)
    # Isolate from the developer's real ledger: no live session unless a test
    # sets one up itself (otherwise a proxy running on the dev machine flips
    # these lifetime-view tests into the session view).
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
    rc, out = _run(monkeypatch, capsys, s)
    assert rc == 0
    # compact grammar: lifetime = one total ▼ figure + trim + $
    assert "total ▼21.6K saved" in out
    assert "43% smaller" in out  # trim rate, the glanceable number
    assert "$0.04" in out
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
    rc, out = _run(monkeypatch, capsys, s)
    assert rc == 0
    assert "total ▼21.6K saved" in out
    assert "$" not in out


def test_equivalence_health_color(monkeypatch, capsys, tmp_path):
    """eq% is colored by health: green >=99%, yellow >=95%, red below."""
    import distil.shadow as shadow

    s = ledger.LedgerSummary(1, 0.01, 100, {}, total_baseline_tokens=1000, total_distil_tokens=600)

    def led_with(changes: int, samples: int = 100):
        led = shadow.ShadowLedger()
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
        assert f"{code}eq " in out, out


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


def test_session_first_when_live_session(monkeypatch, capsys, tmp_path):
    """A live session leads the line; lifetime collapses to one Σ figure."""
    import time

    from distil import ledger as led_mod

    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    path = tmp_path / "savings.jsonl"
    led_mod.record(
        trajectory_id="live-proxy", model="claude-opus-4-8", turns=5,
        baseline_dollars=1.0, distil_dollars=0.4,
        baseline_input_tokens=200_000, distil_input_tokens=80_000,
        session="old-session", path=path,
    )
    led_mod.record(
        trajectory_id="live-proxy", model="claude-opus-4-8", turns=3,
        baseline_dollars=0.5, distil_dollars=0.2,
        baseline_input_tokens=100_000, distil_input_tokens=40_000,
        session=f"s{int(time.time())}-99", path=path,
    )
    monkeypatch.setattr(led_mod, "default_path", lambda: path)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))
    rc = cmd_statusline(argparse.Namespace(no_color=True))
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert "session ▼60.0K · 60% smaller" in out  # THIS session
    assert "total ▼180.0K" in out  # lifetime
    assert "$0.30" in out  # session dollars, not lifetime


def test_lifetime_fallback_when_session_stale(monkeypatch, capsys, tmp_path):
    """No live session (>4h idle) → the familiar lifetime view."""
    import json as _json

    from distil import ledger as led_mod

    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    path = tmp_path / "savings.jsonl"
    rec = {
        "trajectory_id": "live-proxy", "model": "claude-opus-4-8", "turns": 2,
        "baseline_dollars": 1.0, "distil_dollars": 0.5,
        "baseline_input_tokens": 50_000, "distil_input_tokens": 25_000,
        "tokenizer": "heuristic", "ts": 1000.0, "session": "ancient",
    }
    path.write_text(_json.dumps(rec) + "\n")
    monkeypatch.setattr(led_mod, "default_path", lambda: path)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))
    cmd_statusline(argparse.Namespace(no_color=True))
    out = capsys.readouterr().out.strip()
    assert "total ▼25.0K saved · 50% smaller" in out
    assert "session ▼" not in out  # no live-session segment


def test_eq_suppressed_below_min_samples(monkeypatch, capsys):
    """eq 100.0% over 1 sample is noise wearing a number — suppressed until 25."""
    import distil.shadow as shadow

    s = ledger.LedgerSummary(1, 0.01, 100, {}, total_baseline_tokens=1000, total_distil_tokens=600)
    led = shadow.ShadowLedger()
    led.samples = 1
    led.recent.append(1)
    monkeypatch.setattr(shadow.ShadowLedger, "load", classmethod(lambda cls, *a, **k: led))
    _rc, out = _run(monkeypatch, capsys, s)
    assert "eq" not in out


def test_zero_savings_session_says_watching(monkeypatch, capsys, tmp_path):
    """Traffic flowing but nothing trimmed yet must read as 'watching', not ▼0 −0%."""
    import time

    from distil import ledger as led_mod

    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    path = tmp_path / "savings.jsonl"
    led_mod.record(
        trajectory_id="live-proxy", model="claude-opus-4-8", turns=4,
        baseline_dollars=0.1, distil_dollars=0.1,
        baseline_input_tokens=12_000, distil_input_tokens=12_000,
        session=f"s{int(time.time())}-7", path=path,
    )
    monkeypatch.setattr(led_mod, "default_path", lambda: path)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))
    cmd_statusline(argparse.Namespace(no_color=True))
    out = capsys.readouterr().out.strip()
    assert "watching · 12.0K seen" in out
    assert "session ▼" not in out and "smaller" not in out.split("total")[0]


def test_flush_skips_zero_baseline_records(tmp_path):
    """A flush window with zero measured tokens writes nothing (no noise records)."""
    from distil.runtime import RuntimeSavings

    led = tmp_path / "savings.jsonl"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    rs.record(0, 0)  # a request passed through with nothing measurable
    assert rs.flush() is True  # counters reset...
    assert not led.exists()  # ...but no record was written
