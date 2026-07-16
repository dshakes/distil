"""Proof Ledger — end-of-session certified printout tests.

Covers: session summary content, suppression labeling, zero-request silence,
opt-out env var, fail-open guarantee, restorability stats.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


from distil import calibration as _calib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_ledger_row(
    home: Path,
    session_id: str,
    *,
    baseline_tok: int = 1000,
    distil_tok: int = 500,
    baseline_usd: float = 0.02,
    distil_usd: float = 0.01,
) -> None:
    """Write one savings row to $DISTIL_HOME/savings.jsonl."""
    row = {
        "trajectory_id": "live-proxy",
        "model": "claude-opus-4-8",
        "turns": 1,
        "baseline_dollars": baseline_usd,
        "distil_dollars": distil_usd,
        "baseline_input_tokens": baseline_tok,
        "distil_input_tokens": distil_tok,
        "tokenizer": "heuristic",
        "ts": time.time(),
        "session": session_id,
        "mode": "digest",
        "acct": 2,
    }
    p = home / "savings.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _write_shadow_rows(
    home: Path,
    *,
    ab_count: int,
    aa_count: int,
    equivalent: bool = True,
    since: float | None = None,
) -> None:
    """Write shadow rows to $DISTIL_HOME/shadow.jsonl."""
    ts = since if since is not None else time.time()
    p = home / "shadow.jsonl"
    with p.open("a", encoding="utf-8") as f:
        for _ in range(ab_count):
            f.write(json.dumps({"equivalent": equivalent, "ts": ts, "kind": "ab", "sig": 3}) + "\n")
        for _ in range(aa_count):
            f.write(json.dumps({"equivalent": True, "ts": ts, "kind": "aa", "sig": 3}) + "\n")


# ---------------------------------------------------------------------------
# Core output
# ---------------------------------------------------------------------------


def test_returns_none_for_zero_requests(tmp_path, monkeypatch):
    """No ledger rows → build_ledger_text returns None (print nothing)."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil.proof_ledger import build_ledger_text

    result = build_ledger_text("s12345-9999", time.time() - 60)
    assert result is None


def test_returns_none_for_different_session(tmp_path, monkeypatch):
    """Rows for another session don't count for the requested session."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    _write_ledger_row(tmp_path, "other-session")
    from distil.proof_ledger import build_ledger_text

    result = build_ledger_text("my-session", time.time() - 60)
    assert result is None


def test_returns_text_for_matching_session(tmp_path, monkeypatch):
    """A matching ledger row produces a formatted proof ledger block."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s12345-9999"
    _write_ledger_row(
        tmp_path, sid, baseline_tok=2000, distil_tok=1000, baseline_usd=0.04, distil_usd=0.02
    )
    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 120)
    assert text is not None
    assert "distil proof ledger" in text
    assert "50.0% smaller" in text
    assert "$0.04" in text
    assert "$0.02" in text


def test_duration_label_appears(tmp_path, monkeypatch):
    """Session duration is shown in the header."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-dur"
    _write_ledger_row(tmp_path, sid)
    from distil.proof_ledger import build_ledger_text

    # 5 minutes ago
    text = build_ledger_text(sid, time.time() - 300)
    assert text is not None
    assert "session 5m" in text


def test_token_counts_in_output(tmp_path, monkeypatch):
    """Token baseline and distil counts appear in the text."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-tok"
    _write_ledger_row(tmp_path, sid, baseline_tok=1000, distil_tok=600)
    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    assert "1,000" in text
    assert "600" in text
    assert "40.0% smaller" in text


# ---------------------------------------------------------------------------
# Shadow suppression labeling
# ---------------------------------------------------------------------------


def test_shadow_zero_samples_shows_hint(tmp_path, monkeypatch):
    """Zero shadow samples → 'no shadow samples' with enable hint."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-sh0"
    _write_ledger_row(tmp_path, sid)
    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 60)
    assert text is not None
    assert "no shadow samples" in text
    assert "shadow-rate" in text


def test_shadow_below_threshold_shows_gathering(tmp_path, monkeypatch):
    """Shadow below VERDICT_MIN_AB or VERDICT_MIN_AA → suppression label."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-sh-low"
    _write_ledger_row(tmp_path, sid)
    # 10 A/B < VERDICT_MIN_AB (50), 5 A/A < VERDICT_MIN_AA (30)
    _write_shadow_rows(tmp_path, ab_count=10, aa_count=5, since=time.time() - 60)
    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 120)
    assert text is not None
    assert "below verdict threshold, gathering" in text
    assert "n=10 A/B" in text
    assert "5 A/A" in text


def test_shadow_above_threshold_no_suppression(tmp_path, monkeypatch):
    """Shadow above both thresholds → sample counts without suppression label."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-sh-hi"
    _write_ledger_row(tmp_path, sid)
    from distil.shadow import VERDICT_MIN_AA, VERDICT_MIN_AB

    _write_shadow_rows(
        tmp_path, ab_count=VERDICT_MIN_AB, aa_count=VERDICT_MIN_AA, since=time.time() - 60
    )
    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 120)
    assert text is not None
    assert "below verdict threshold" not in text
    assert f"n={VERDICT_MIN_AB} A/B" in text
    assert f"{VERDICT_MIN_AA} A/A" in text


def test_shadow_filtered_by_session_start(tmp_path, monkeypatch):
    """Shadow rows BEFORE session start are excluded from the session tally."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-sh-ts"
    _write_ledger_row(tmp_path, sid)
    # Write 60 A/B rows timestamped in the past (before session started)
    _write_shadow_rows(tmp_path, ab_count=60, aa_count=40, since=time.time() - 3600)
    from distil.proof_ledger import build_ledger_text

    # Session start = 30 seconds ago → old rows excluded
    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    assert "no shadow samples" in text  # all pre-session rows excluded


def test_shadow_changes_counted(tmp_path, monkeypatch):
    """Decision changes (not-equivalent) are counted correctly."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-sh-chg"
    _write_ledger_row(tmp_path, sid)
    p = tmp_path / "shadow.jsonl"
    ts = time.time()
    with p.open("a") as f:
        for _ in range(3):
            f.write(json.dumps({"equivalent": False, "ts": ts, "kind": "ab", "sig": 3}) + "\n")
        for _ in range(7):
            f.write(json.dumps({"equivalent": True, "ts": ts, "kind": "ab", "sig": 3}) + "\n")
        for _ in range(5):
            f.write(json.dumps({"equivalent": True, "ts": ts, "kind": "aa", "sig": 3}) + "\n")
    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 60)
    assert text is not None
    assert "3 decision changes" in text


# ---------------------------------------------------------------------------
# Calibration labeling
# ---------------------------------------------------------------------------


def test_calibration_uncalibrated(tmp_path, monkeypatch):
    """Below MIN_SAMPLES → 'estimated · calibrating'."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-cal-low"
    _write_ledger_row(tmp_path, sid)
    # 1 sample — well below MIN_SAMPLES (20)
    _calib.record("claude-opus-4-8", 1000, 1200, path=tmp_path / "calibration.json")
    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    assert "estimated · calibrating" in text


def test_calibration_calibrated(tmp_path, monkeypatch):
    """Above MIN_SAMPLES with factor != 1.0 → 'calibrated to your billed usage'."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-cal-hi"
    _write_ledger_row(tmp_path, sid)
    cal_path = tmp_path / "calibration.json"
    for i in range(25):
        _calib.record("claude-opus-4-8", 1000 + i, int((1000 + i) * 1.2), path=cal_path)
    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    assert "calibrated to your billed usage" in text


# ---------------------------------------------------------------------------
# Restorability stats
# ---------------------------------------------------------------------------


def test_restore_no_requests_file_shows_zero(tmp_path, monkeypatch):
    """No session requests file → 0 digests, all handles live."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-rest-0"
    _write_ledger_row(tmp_path, sid)
    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    assert "0 digests" in text
    assert "100% recoverable" in text


def test_restore_live_handles(tmp_path, monkeypatch):
    """Session blocks in restore store (fresh mtime) → '100% recoverable'."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-rest-live"
    _write_ledger_row(tmp_path, sid)

    # Write session requests file with two block handles
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    req = sessions_dir / (sid + ".requests.jsonl")
    rec = {
        "ts": time.time(),
        "blocks": [
            {"h": "abcd1234", "sig": "abc", "tokens": 100},
            {"h": "ef567890", "sig": "xyz", "tokens": 50},
        ],
    }
    req.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    # Write both handles to restore store
    restore_dir = tmp_path / "restore"
    restore_dir.mkdir(parents=True, exist_ok=True)
    (restore_dir / "abcd1234").write_text("content A", encoding="utf-8")
    (restore_dir / "ef567890").write_text("content B", encoding="utf-8")

    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    assert "100% recoverable" in text
    assert "2 digests" in text


# ---------------------------------------------------------------------------
# Opt-out and fail-open
# ---------------------------------------------------------------------------


def test_opt_out_env_var_suppresses_output(tmp_path, monkeypatch, capsys):
    """DISTIL_NO_LEDGER=1 → nothing printed to stderr."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_NO_LEDGER", "1")
    sid = "s-opt"
    _write_ledger_row(tmp_path, sid)
    from distil.proof_ledger import print_proof_ledger

    print_proof_ledger(sid, time.time() - 60)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_zero_requests_no_output_to_stderr(tmp_path, monkeypatch, capsys):
    """Zero proxied requests → print_proof_ledger prints nothing."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-zero"
    from distil.proof_ledger import print_proof_ledger

    print_proof_ledger(sid, time.time() - 60)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_print_fail_open(monkeypatch):
    """A crash inside build_ledger_text must not propagate from print_proof_ledger."""
    import distil.proof_ledger as _pl

    original = _pl.build_ledger_text

    def _boom(*_a, **_k):
        raise RuntimeError("simulated internal failure")

    _pl.build_ledger_text = _boom
    try:
        _pl.print_proof_ledger("s-crash", time.time())  # must not raise
    finally:
        _pl.build_ledger_text = original


def test_print_outputs_to_stderr(tmp_path, monkeypatch, capsys):
    """Ledger text goes to stderr (not stdout) so it doesn't pollute piped output."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-err"
    _write_ledger_row(tmp_path, sid)
    from distil.proof_ledger import print_proof_ledger

    print_proof_ledger(sid, time.time() - 60)
    captured = capsys.readouterr()
    assert "distil proof ledger" in captured.err
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Exit-code preservation (integration)
# ---------------------------------------------------------------------------


def test_wrap_run_exit_code_preserved_with_ledger(tmp_path, monkeypatch):
    """wrap_run returns the child's exit code unchanged even when the ledger prints."""
    import sys

    from distil.proxy import wrap_run

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    code = wrap_run([sys.executable, "-c", "raise SystemExit(42)"], record=False)
    assert code == 42


def test_wrap_run_exit_code_preserved_when_ledger_crashes(tmp_path, monkeypatch):
    """wrap_run exit code is unaffected even if proof_ledger raises internally."""
    import sys

    import distil.proof_ledger as _pl

    original = _pl.print_proof_ledger

    def _boom(*_a, **_k):
        raise RuntimeError("ledger on fire")

    _pl.print_proof_ledger = _boom
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    try:
        from distil.proxy import wrap_run

        code = wrap_run([sys.executable, "-c", "raise SystemExit(7)"], record=False)
        assert code == 7
    finally:
        _pl.print_proof_ledger = original


# ---------------------------------------------------------------------------
# _dur: hour-range branch (lines 28-29)
# ---------------------------------------------------------------------------


def test_dur_hours_and_minutes() -> None:
    """_dur with >= 3600s and leftover minutes → 'Xh Ym' format."""
    from distil.proof_ledger import _dur

    assert _dur(7380) == "2h3m"  # 2h 3m = 7380s


def test_dur_hours_exact() -> None:
    """_dur with exactly N hours and no leftover minutes → 'Xh' (no trailing 'm')."""
    from distil.proof_ledger import _dur

    assert _dur(7200) == "2h"  # exactly 2 hours, no minutes


# ---------------------------------------------------------------------------
# _session_digest_stats: n==0 branch (line 64)
# ---------------------------------------------------------------------------


def test_session_digest_stats_no_block_handles(tmp_path, monkeypatch):
    """Requests file exists but records carry no block handles → n=0 → 100% recoverable."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-no-handles"
    _write_ledger_row(tmp_path, sid)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    req = sessions_dir / (sid + ".requests.jsonl")
    # Record with no 'blocks' key → session_handles stays empty → n=0
    req.write_text(json.dumps({"ts": time.time(), "tokens": 500}) + "\n", encoding="utf-8")

    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    assert "0 digests" in text
    assert "100% recoverable" in text


# ---------------------------------------------------------------------------
# _session_digest_stats: expired / missing restore files (lines 73-74, 160)
# ---------------------------------------------------------------------------


def test_session_expired_handles_not_in_restore(tmp_path, monkeypatch):
    """Block handles referenced in the session file but absent from restore → not all live."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-expired"
    _write_ledger_row(tmp_path, sid)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    req = sessions_dir / (sid + ".requests.jsonl")
    rec = {"ts": time.time(), "blocks": [{"h": "dead1234", "sig": "abc", "tokens": 100}]}
    req.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    # Create the restore directory but NOT the handle file — handle is "missing"
    (tmp_path / "restore").mkdir(parents=True, exist_ok=True)

    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    assert "some handles expired" in text


# ---------------------------------------------------------------------------
# _session_digest_stats: invalid JSON lines (lines 57-58)
# ---------------------------------------------------------------------------


def test_session_requests_file_invalid_json_lines(tmp_path, monkeypatch):
    """Corrupt lines in the session requests file are skipped (fail-open)."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-badjson"
    _write_ledger_row(tmp_path, sid)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    req = sessions_dir / (sid + ".requests.jsonl")
    # Mix of bad JSON, valid JSON with no blocks, and a partially valid record
    content = (
        "not-json-at-all\n"
        + json.dumps({"ts": time.time()})
        + "\n"
        + "{broken_json\n"
        + "null\n"  # valid JSON but not a dict → AttributeError caught
    )
    req.write_text(content, encoding="utf-8")

    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None  # fail-open: bad lines skipped gracefully


# ---------------------------------------------------------------------------
# _session_digest_stats: OSError on file read (lines 59-60)
# ---------------------------------------------------------------------------


def test_session_requests_oserror(tmp_path, monkeypatch):
    """OSError reading the session requests file → fail-open returns (0, True)."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-oserr"
    _write_ledger_row(tmp_path, sid)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    req = sessions_dir / (sid + ".requests.jsonl")
    req.write_text("{}\n", encoding="utf-8")
    req.chmod(0o000)

    try:
        from distil.proof_ledger import build_ledger_text

        text = build_ledger_text(sid, time.time() - 30)
        assert text is not None  # fail-open: 0 digests, all handles "live"
    finally:
        req.chmod(0o644)


# ---------------------------------------------------------------------------
# _calib_note: calibrated with factor == 1.0 (line 97)
# ---------------------------------------------------------------------------


def test_calib_factor_unity_label(tmp_path, monkeypatch):
    """n >= MIN_SAMPLES with factor=1.0 (estimated==actual) → 'calibrated (n=N, factor≈1.0)'."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-cal-unity"
    _write_ledger_row(tmp_path, sid)
    cal_path = tmp_path / "calibration.json"
    # Record MIN_SAMPLES+5 samples where estimated == actual → factor = 1.0
    for i in range(25):
        _calib.record("claude-opus-4-8", 1000 + i, 1000 + i, path=cal_path)

    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    assert "factor≈1.0" in text


# ---------------------------------------------------------------------------
# build_ledger_text: singular 'digest' label (line 156 area)
# ---------------------------------------------------------------------------


def test_session_single_digest_label(tmp_path, monkeypatch):
    """n_digests=1 → singular 'digest', not 'digests'."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    sid = "s-single-digest"
    _write_ledger_row(tmp_path, sid)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    req = sessions_dir / (sid + ".requests.jsonl")
    rec = {"ts": time.time(), "blocks": [{"h": "abc12345", "sig": "x", "tokens": 50}]}
    req.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    restore_dir = tmp_path / "restore"
    restore_dir.mkdir(parents=True, exist_ok=True)
    (restore_dir / "abc12345").write_text("content", encoding="utf-8")

    from distil.proof_ledger import build_ledger_text

    text = build_ledger_text(sid, time.time() - 30)
    assert text is not None
    # Singular form when exactly 1 digest
    assert "1 digest," in text or "1 digest " in text
    assert "1 digests" not in text
