"""Shadow counter diagnostics — content-free sampling observability."""

from __future__ import annotations

import json
from pathlib import Path


from distil.shadow import ShadowCounters


# ---------------------------------------------------------------------------
# ShadowCounters unit tests
# ---------------------------------------------------------------------------


def test_note_seen_increments_in_memory(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_seen()
    c.note_seen()
    with c._lock:
        assert c._pending.get("requests_seen") == 2


def test_note_sampled_increments_in_memory(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_sampled()
    with c._lock:
        assert c._pending.get("sampled") == 1


def test_flush_with_record_path(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_seen()
    c.note_sampled()
    c.flush_with(replay_attempted=True, recorded=True)
    data = json.loads(p.read_text())
    assert data["requests_seen"] == 1
    assert data["sampled"] == 1
    assert data["replay_attempted"] == 1
    assert data.get("replay_failed", 0) == 0
    assert data["recorded"] == 1
    # pending should be drained
    with c._lock:
        assert c._pending == {}


def test_flush_with_fail_path(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_seen()
    c.note_sampled()
    c.flush_with(replay_attempted=True, replay_failed=True, fail_reason="401")
    data = json.loads(p.read_text())
    assert data["replay_attempted"] == 1
    assert data["replay_failed"] == 1
    assert data["last_fail_reason"] == "401"
    assert data.get("recorded", 0) == 0


def test_flush_with_sig_none_skipped(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.flush_with(replay_attempted=True, sig_none_skipped=True)
    data = json.loads(p.read_text())
    assert data["replay_attempted"] == 1
    assert data["signature_none_skipped"] == 1
    assert data.get("replay_failed", 0) == 0


def test_flush_accumulates(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.flush_with(replay_attempted=True, recorded=True)
    c.flush_with(replay_attempted=True, replay_failed=True, fail_reason="401")
    data = json.loads(p.read_text())
    assert data["replay_attempted"] == 2
    assert data["recorded"] == 1
    assert data["replay_failed"] == 1


def test_load_returns_empty_when_missing(tmp_path: Path) -> None:
    p = tmp_path / "nonexistent.json"
    assert ShadowCounters.load(path=p) == {}


def test_load_returns_empty_on_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    p.write_text("not json")
    assert ShadowCounters.load(path=p) == {}


def test_load_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_seen()
    c.note_seen()
    c.note_sampled()
    c.flush_with(replay_attempted=True, recorded=True)
    loaded = ShadowCounters.load(path=p)
    assert loaded["requests_seen"] == 2
    assert loaded["sampled"] == 1
    assert loaded["recorded"] == 1


# ---------------------------------------------------------------------------
# Stats renderer: diagnostic line when recorded==0 but attempted>0
# ---------------------------------------------------------------------------


def _render_shadow_stats(counters: dict, *, samples: int = 0) -> str:
    """Minimal inline reproduction of the cmd_shadow_stats diagnostic logic."""
    lines: list[str] = []
    if samples == 0:
        attempted = counters.get("replay_attempted", 0)
        if attempted > 0:
            seen = counters.get("requests_seen", 0)
            sampled = counters.get("sampled", 0)
            failed = counters.get("replay_failed", 0)
            reason = counters.get("last_fail_reason", "")
            reason_str = f" (last: {reason})" if reason else ""
            lines.append(
                f"Shadow counters: {seen} seen, {sampled} sampled, "
                f"{attempted} replay{'s' if attempted != 1 else ''} attempted, "
                f"{failed} failed{reason_str}"
            )
        else:
            lines.append("No shadow samples yet.")
    else:
        attempted = counters.get("replay_attempted", 0)
        failed = counters.get("replay_failed", 0)
        reason = counters.get("last_fail_reason", "")
        recorded = counters.get("recorded", 0)
        seen = counters.get("requests_seen", 0)
        sampled = counters.get("sampled", 0)
        reason_str = f" (last: {reason})" if reason and failed else ""
        lines.append(
            f"Sampling: {seen} seen, {sampled} sampled, "
            f"{attempted} attempted, {failed} failed{reason_str}, {recorded} recorded"
        )
    return "\n".join(lines)


def test_renderer_shows_diagnostic_when_attempted_but_no_samples() -> None:
    ctrs = {
        "requests_seen": 19,
        "sampled": 2,
        "replay_attempted": 2,
        "replay_failed": 2,
        "last_fail_reason": "401",
    }
    out = _render_shadow_stats(ctrs, samples=0)
    assert "19 seen" in out
    assert "2 sampled" in out
    assert "2 replays attempted" in out
    assert "2 failed" in out
    assert "401" in out
    assert "No shadow samples yet" not in out


def test_renderer_shows_no_samples_when_no_attempts() -> None:
    out = _render_shadow_stats({}, samples=0)
    assert "No shadow samples yet" in out


def test_renderer_shows_counter_line_when_samples_exist() -> None:
    ctrs = {
        "requests_seen": 50,
        "sampled": 5,
        "replay_attempted": 5,
        "replay_failed": 0,
        "recorded": 5,
    }
    out = _render_shadow_stats(ctrs, samples=30)
    assert "50 seen" in out
    assert "5 sampled" in out
    assert "5 recorded" in out
    assert "0 failed" in out


# ---------------------------------------------------------------------------
# Per-version buckets — a fixed bug must not depress the rate forever (#96)
# ---------------------------------------------------------------------------


def test_counters_are_bucketed_by_version(tmp_path: Path, monkeypatch) -> None:
    """Lifetime totals accumulate for the life of the install, so failures from an
    ALREADY-FIXED bug stay in the displayed rate forever.

    Concretely, the case that prompted this: the temperature-0 replay bug (fixed in
    8c744df) left 295/323 failures behind, which pinned the lifetime rate at 42%
    while the real rate since the fix was 5.3%. Two consequences, and the second is
    the one that matters — a fixed bug makes the sampler look broken, and the next
    REAL regression has to clear that stale noise floor before anyone can see it.
    """
    import distil.shadow as sh

    p = tmp_path / "ctr.json"
    monkeypatch.setattr(sh.ShadowCounters, "_VERSION_CACHE", "1.0.0")
    c = ShadowCounters(path=p)
    c.note_seen()
    c.flush_with(replay_attempted=True, replay_failed=True, fail_reason="400")

    # ... a fix ships, and the next build behaves.
    monkeypatch.setattr(sh.ShadowCounters, "_VERSION_CACHE", "1.1.0")
    c2 = ShadowCounters(path=p)
    c2.note_seen()
    c2.flush_with(replay_attempted=True, recorded=True)

    data = ShadowCounters.load(p)
    assert data["replay_failed"] == 1, "lifetime totals stay, for continuity"
    assert data["replay_attempted"] == 2
    by_v = data["by_version"]
    assert by_v["1.0.0"]["replay_failed"] == 1
    assert by_v["1.1.0"].get("replay_failed", 0) == 0, (
        "the fixed build must not inherit the broken build's failures"
    )
    assert by_v["1.1.0"]["recorded"] == 1


def test_a_version_change_does_not_reset_the_lifetime_totals(tmp_path: Path, monkeypatch) -> None:
    """Deliberately NOT resetting: that loses the trend and makes the opposite
    mistake — a rate degrading across releases would become invisible."""
    import distil.shadow as sh

    p = tmp_path / "ctr.json"
    for ver in ("1.0.0", "1.1.0", "1.2.0"):
        monkeypatch.setattr(sh.ShadowCounters, "_VERSION_CACHE", ver)
        ShadowCounters(path=p).flush_with(replay_attempted=True)
    data = ShadowCounters.load(p)
    assert data["replay_attempted"] == 3, "history must survive an upgrade"
    assert set(data["by_version"]) == {"1.0.0", "1.1.0", "1.2.0"}


def test_fail_reasons_are_a_histogram_not_the_last_one(tmp_path: Path, monkeypatch) -> None:
    """`last_fail_reason` keeps only the most recent value, so a run mixing 400s,
    429s and exceptions reported whichever landed last — which turned diagnosing a
    real failure into archaeology rather than a read."""
    import distil.shadow as sh

    p = tmp_path / "ctr.json"
    monkeypatch.setattr(sh.ShadowCounters, "_VERSION_CACHE", "1.0.0")
    c = ShadowCounters(path=p)
    for reason in ("400", "400", "429", "exception"):
        c.flush_with(replay_attempted=True, replay_failed=True, fail_reason=reason)

    bucket = ShadowCounters.load(p)["by_version"]["1.0.0"]
    assert bucket["fail_reasons"] == {"400": 2, "429": 1, "exception": 1}
    assert ShadowCounters.load(p)["last_fail_reason"] == "exception", "kept for continuity"


def test_a_counter_file_written_before_buckets_existed_still_loads(tmp_path: Path) -> None:
    """Every existing install has a bucket-less file; upgrading must not lose it."""
    p = tmp_path / "ctr.json"
    p.write_text(json.dumps({"requests_seen": 7, "replay_failed": 2}))
    ShadowCounters(path=p).flush_with(replay_attempted=True)
    data = ShadowCounters.load(p)
    assert data["requests_seen"] == 7 and data["replay_failed"] == 2
    assert data["replay_attempted"] == 1
    assert "by_version" in data


def test_a_corrupt_by_version_value_is_replaced_not_crashed(tmp_path: Path) -> None:
    """Counters are diagnostics: a mangled file must never break the request path."""
    p = tmp_path / "ctr.json"
    p.write_text(json.dumps({"by_version": "not-a-dict"}))
    ShadowCounters(path=p).flush_with(replay_attempted=True)
    assert isinstance(ShadowCounters.load(p)["by_version"], dict)


# ---------------------------------------------------------------------------
# What `distil shadow-stats` actually prints (#96)
# ---------------------------------------------------------------------------


def _stats(home: Path, counters: dict, monkeypatch, capsys) -> str:
    """Run `shadow-stats` IN-PROCESS.

    A subprocess would be more end-to-end but invisible to coverage, so the
    reporting branches these tests exist to pin would read as untested — and an
    untested branch is the next one to rot.
    """
    import argparse

    from distil import cli

    (home / "shadow_counters.json").write_text(json.dumps(counters))
    monkeypatch.setenv("DISTIL_HOME", str(home))
    cli.cmd_shadow_stats(argparse.Namespace(record=False, max_change_rate=None))
    return capsys.readouterr().out


#: The real shape from the field: a lifetime dominated by an already-fixed bug.
_FIELD = {
    "requests_seen": 34315,
    "sampled": 757,
    "replay_attempted": 757,
    "replay_failed": 318,
    "recorded": 438,
    "last_fail_reason": "exception",
}


def test_stats_leads_with_the_running_build_not_the_lifetime(tmp_path, monkeypatch, capsys) -> None:
    """The headline number must describe the build the user is running.

    Reading `318 failed` off a 42% lifetime rate says the sampler is broken. It
    isn't — nearly all of those are 400s from a bug fixed months earlier.
    """
    import distil.shadow as sh

    ctrs = dict(_FIELD)
    ctrs["by_version"] = {
        sh._counter_version(): {
            "requests_seen": 900,
            "sampled": 12,
            "replay_attempted": 12,
            "replay_failed": 1,
            "recorded": 11,
            "fail_reasons": {"429": 1},
        }
    }
    out = _stats(tmp_path, ctrs, monkeypatch, capsys)
    assert sh._counter_version() in out, "the running version must be named"
    assert "12 replays attempted, 1 failed" in out or "12 attempted, 1 failed" in out
    assert "318 failed" not in out.split("lifetime")[0], (
        "the stale lifetime failure count must not be the headline"
    )
    assert "429×1" in out, "fail reasons are a histogram now"


def test_stats_says_so_when_this_build_has_no_samples_yet(tmp_path, monkeypatch, capsys) -> None:
    """Falling back to lifetime is fine — presenting it as current is not."""
    out = _stats(tmp_path, dict(_FIELD), monkeypatch, capsys)
    assert "lifetime" in out.lower()
    assert "757" in out, "with nothing else to show, lifetime is what there is"


def test_stats_with_samples_reports_this_build_and_labels_lifetime(
    tmp_path, monkeypatch, capsys
) -> None:
    """The main reporting path — the one a user with real traffic actually sees.

    The lifetime line must still be shown (dropping it would hide a rate degrading
    across releases) but must be LABELLED, so nobody reads a fixed bug's failures
    as the current state of the sampler.
    """
    import argparse

    import distil.shadow as sh
    from distil import cli
    from distil.shadow import ShadowLedger

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    led = ShadowLedger()
    for i in range(60):
        led.record(i % 20 != 0, path=tmp_path / "shadow.jsonl")
    for _ in range(40):
        led.record(True, kind="aa", path=tmp_path / "shadow.jsonl")

    ctrs = dict(_FIELD)
    ctrs["by_version"] = {
        sh._counter_version(): {
            "requests_seen": 900,
            "sampled": 12,
            "replay_attempted": 12,
            "replay_failed": 1,
            "recorded": 11,
            "fail_reasons": {"429": 1},
        }
    }
    (tmp_path / "shadow_counters.json").write_text(json.dumps(ctrs))

    cli.cmd_shadow_stats(argparse.Namespace(record=False, max_change_rate=None))
    out = capsys.readouterr().out

    assert f"Sampling ({sh._counter_version()})" in out, "the running build leads"
    assert "429×1" in out, "fail reasons are a histogram"
    assert "lifetime" in out, "history is kept as context, not dropped"
    head = out.split("lifetime")[0]
    assert "318 failed" not in head, "the stale failure count must not be the headline"
    assert "8.3%" in out or "% of replays" in out, "the rate is stated, not left to arithmetic"


def test_stats_falls_back_to_lifetime_but_says_so(tmp_path, monkeypatch, capsys) -> None:
    """Just-upgraded install: samples exist, but none yet on this build.

    Falling back to lifetime is correct — it is all there is. Presenting it as the
    running build's rate is what this whole change exists to stop, so the label
    carries the honesty.
    """
    import argparse

    from distil import cli
    from distil.shadow import ShadowLedger

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    led = ShadowLedger()
    for i in range(60):
        led.record(i % 20 != 0, path=tmp_path / "shadow.jsonl")

    (tmp_path / "shadow_counters.json").write_text(json.dumps(dict(_FIELD)))  # no by_version
    cli.cmd_shadow_stats(argparse.Namespace(record=False, max_change_rate=None))
    out = capsys.readouterr().out

    assert "no samples yet on this build" in out, "the fallback must announce itself"
    assert "757" in out, "and still show what history there is"
