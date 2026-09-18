"""Live drift wiring: the loss definition, the persistence, and the statistical gate.

The gate is the point. An alarm that false-alarms above its stated delta is worse than
no alarm, because it teaches the reader to ignore a line that can come back negative —
so the null simulation runs here, at the same budget the code actually ships with, and
fails the build if the e-process stops holding its level.
"""

from __future__ import annotations

import json
import math
import random
import statistics

from distil.drift import (
    BUDGET_ALPHA,
    BUDGET_DELTA,
    DriftMonitor,
    LiveDrift,
    live_monitor,
    paired_loss,
)

NULL_MEAN = (1.0 + BUDGET_ALPHA) / 2.0


def _draw(rng: random.Random, harm: float, u: float) -> int:
    """One paired difference with ``E[d] = -harm`` and ``P(d == -1) == u``.

    ``u`` is the dispersion the model's own nondeterminism creates: on the maintainer's
    real traffic (A/A self-agreement 55.6%) it sits near 0.26, which is the regime the
    null has to hold in.
    """
    r = rng.random()
    if r < u:
        return -1
    if r < u + (u - harm):
        return 1
    return 0


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


def test_paired_loss_is_affine_not_clipped():
    """-1/0/+1 maps to 1/0.5/0, so a request where compression WON pulls the mean down.

    Clipping (``max(0, -d)``) would count self-noise as harm: on the real sample that
    reads ~26% against a 5% budget, which is a wrong verdict, not a conservative one.
    """
    assert paired_loss(-1) == 1.0
    assert paired_loss(0) == 0.5
    assert paired_loss(1) == 0.0


def test_loss_mean_recovers_the_harm_rate():
    diffs = [-1] * 15 + [1] * 5 + [0] * 80
    harm = -sum(diffs) / len(diffs)
    mean = sum(paired_loss(d) for d in diffs) / len(diffs)
    assert math.isclose(2 * mean - 1, harm)


def test_budget_matches_the_cli_certificate_default():
    """One budget, or the live alarm and the offline certificate mean different things."""
    from distil.cli import build_parser

    parser = build_parser()
    for cmd, flag in (("conformal", "alpha"), ("conformal", "delta")):
        action = next(
            a
            for a in parser._subparsers._group_actions[0].choices[cmd]._actions  # type: ignore[attr-defined]
            if a.dest == flag
        )
        assert action.default == (BUDGET_ALPHA if flag == "alpha" else BUDGET_DELTA)


# ---------------------------------------------------------------------------
# The statistical gate — mandatory, not decorative
# ---------------------------------------------------------------------------


def test_null_false_alarm_stays_under_delta():
    """2000 null runs at exactly the budget; checked after EVERY sample (anytime-valid)."""
    trials, horizon = 2000, 2000
    tolerance = BUDGET_DELTA + 2 * math.sqrt(BUDGET_DELTA * (1 - BUDGET_DELTA) / trials)
    for u in (0.26, 0.10):  # real-traffic dispersion, and a quieter stream
        alarms = 0
        for tr in range(trials):
            rng = random.Random(0xD1F7 ^ tr)
            mon = DriftMonitor(alpha=NULL_MEAN, delta=BUDGET_DELTA)
            for _ in range(horizon):
                if mon.update(paired_loss(_draw(rng, BUDGET_ALPHA, u))):
                    break
            alarms += mon.tripped
        rate = alarms / trials
        assert rate <= tolerance, f"u={u}: false-alarm {rate:.4f} > {tolerance:.4f}"


def test_detects_an_alternative_ten_points_over_budget():
    """Budget + 0.10 must be caught, and caught soon enough to be worth printing."""
    trials, horizon = 500, 2000
    detected, ns = 0, []
    for tr in range(trials):
        rng = random.Random(0xBADF00D ^ tr)
        mon = DriftMonitor(alpha=NULL_MEAN, delta=BUDGET_DELTA)
        for i in range(horizon):
            if mon.update(paired_loss(_draw(rng, BUDGET_ALPHA + 0.10, 0.26))):
                ns.append(i + 1)
                detected += 1
                break
    assert detected / trials >= 0.95
    # Documented detection horizon: median 172 samples at the maintainer's dispersion.
    assert statistics.median(ns) <= 400


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_state_round_trips_and_never_double_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    diffs = [0] * 60
    first = live_monitor(diffs)
    assert first.consumed == 60
    # Same history again: nothing new to bet on, so capital must not move.
    again = live_monitor(diffs)
    assert again.consumed == 60
    assert again.monitor.capital == first.monitor.capital
    # Two more rows advance it by exactly two.
    more = live_monitor(diffs + [0, 0])
    assert more.consumed == 62


def test_breach_is_sticky_and_names_the_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    state = live_monitor([-1] * 200)
    assert state.monitor.tripped
    assert 0 < state.tripped_at <= 200
    line = state.line()
    assert line is not None and line.startswith("BREACHED at sample ")
    assert "distil calibrate" in line
    # A clean run afterwards does not un-breach it — the reset is explicit.
    healed = live_monitor([-1] * 200 + [1] * 500)
    assert healed.monitor.tripped
    assert healed.line().startswith("BREACHED")


# ---------------------------------------------------------------------------
# Provenance — the count is bound to the stream it counted
# ---------------------------------------------------------------------------


def test_a_truncated_stream_rebuilds_instead_of_ignoring_every_new_row(tmp_path, monkeypatch):
    """The failure a bare index hides: archive shadow.jsonl outside `reset --shadow` and
    `consumed` outruns the file, so `diffs[consumed:]` is empty and the alarm reports a
    stale n while ignoring live traffic until the new file outgrows the old count."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    live_monitor([0] * 200)
    fresh = live_monitor([0] * 60)  # the archived file, regrown to 60 rows
    assert fresh.rebuilt
    assert fresh.consumed == 60
    assert fresh.line() == (
        f"intact (e-value {fresh.monitor.evalue:.2f}, n=60)"
        " — restarted: the shadow stream was replaced"
    )


def test_a_rewritten_stream_of_the_same_length_is_still_detected(tmp_path, monkeypatch):
    """The case a length check cannot see, and the reason the fingerprint exists: the file
    was replaced by a different one that happens to hold the same number of rows."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    first = live_monitor([0] * 80)
    second = live_monitor([-1] * 80)  # same count, entirely different evidence
    assert second.rebuilt
    assert second.consumed == 80
    assert second.monitor.capital != first.monitor.capital


def test_a_signature_bump_rebuilds_and_does_not_carry_a_breach(tmp_path, monkeypatch):
    """A SIG_VERSION bump filters the old rows out of the ledger. Capital bet on evidence
    the ledger no longer returns is not evidence about the stream that replaced it."""
    import distil.shadow as _shadow

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    breached = live_monitor([-1] * 200)
    assert breached.monitor.tripped

    monkeypatch.setattr(_shadow, "SIG_VERSION", _shadow.SIG_VERSION + 1)
    after = live_monitor([0] * 60)
    assert after.rebuilt
    assert not after.monitor.tripped and after.consumed == 60
    assert json.loads((tmp_path / "drift.json").read_text())["sig"] == _shadow.SIG_VERSION


def test_a_plain_append_carries_capital_and_does_not_rebuild(tmp_path, monkeypatch):
    """The common path must stay the common path — provenance is a guard, not a reset."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    diffs = [0] * 60
    first = live_monitor(diffs)
    assert not first.rebuilt
    grown = live_monitor(diffs + [0] * 40)
    assert not grown.rebuilt
    assert grown.consumed == 100
    assert grown.monitor.capital != first.monitor.capital  # it kept betting, from where it was
    assert grown.monitor.n == 100


def test_a_state_file_without_provenance_rebuilds_once(tmp_path, monkeypatch):
    """Upgrading over a pre-provenance state file: the prefix cannot be checked, so it
    cannot be claimed. Rebuild rather than carry an unverifiable number."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    live_monitor([0] * 60)
    p = tmp_path / "drift.json"
    raw = json.loads(p.read_text())
    del raw["stream"], raw["sig"]  # what 1.53.0 wrote
    p.write_text(json.dumps(raw))
    upgraded = live_monitor([0] * 60)
    assert upgraded.rebuilt and upgraded.consumed == 60
    assert not live_monitor([0] * 60).rebuilt  # and only once


def test_reset_clears_the_alarm(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    live_monitor([-1] * 200)
    (tmp_path / "drift.json").unlink()
    fresh = LiveDrift.load()
    assert fresh.consumed == 0 and not fresh.monitor.tripped


def test_budget_change_discards_stale_capital(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    live_monitor([-1] * 100)
    p = tmp_path / "drift.json"
    raw = json.loads(p.read_text())
    raw["alpha"] = 0.99  # capital was bet against a different null
    p.write_text(json.dumps(raw))
    assert LiveDrift.load().consumed == 0


def test_corrupt_state_degrades_to_a_fresh_monitor(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    for junk in ("{not json", "[]", '"a string"'):  # unparseable, and parseable-but-wrong
        (tmp_path / "drift.json").write_text(junk)
        assert LiveDrift.load().consumed == 0, junk


def test_below_floor_says_so_and_prints_no_evalue(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    line = live_monitor([0] * 10).line()
    assert line == "not enough samples yet (10/50)"
    assert "e-value" not in line


def test_capital_is_floored_at_zero_for_an_aggressive_stake():
    """The shipped stake cap (c=0.5) cannot drive capital negative, but the clamp is what
    keeps that a property of the tuning rather than an assumption. A bankrupt bettor has
    zero capital, not negative capital — the e-value must stay a valid e-value."""
    mon = DriftMonitor(alpha=0.5, delta=BUDGET_DELTA, c=3.0)
    mon.update(0.0)  # 1 + 3*(0 - 0.5) = -0.5 before the floor
    assert mon.capital == 0.0
    assert not mon.tripped


def test_a_wrongly_typed_field_degrades_to_a_fresh_monitor(tmp_path, monkeypatch):
    """Corrupt JSON is one failure; well-formed JSON with a junk value is the other.
    Either way the answer is a fresh monitor, never a capital number nobody can explain."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    live_monitor([0] * 10)
    p = tmp_path / "drift.json"
    raw = json.loads(p.read_text())
    raw["capital"] = "not a number"
    p.write_text(json.dumps(raw))
    fresh = LiveDrift.load()
    assert fresh.consumed == 0 and fresh.monitor.capital == 1.0


def test_an_unwritable_state_path_still_returns_a_verdict(tmp_path, monkeypatch):
    """The alarm is bookkeeping: it must never be the reason a wrap session raises."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "blocked"))
    (tmp_path / "blocked").write_text("this is a file, not a directory")
    state = live_monitor([0] * 60)
    assert state.consumed == 60
    assert state.line() == f"intact (e-value {state.monitor.evalue:.2f}, n=60)"


def test_an_interrupted_write_leaves_the_previous_state_intact(tmp_path, monkeypatch):
    """`tripped` is sticky and capital accumulates across sessions, so a torn write is not
    a loud corruption — it is a silent reset that reads "intact" over evidence that said
    BREACHED. The temp file absorbs the tear; the rename is what the reader ever sees."""
    import distil._filelock as _fl

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    breached = live_monitor([-1] * 200)
    assert breached.monitor.tripped
    before = (tmp_path / "drift.json").read_text()

    def _die(src, dst):  # the crash lands after the temp file exists, before the rename
        raise OSError("interrupted")

    monkeypatch.setattr(_fl, "replace_retrying", _die)
    live_monitor([-1] * 200 + [0] * 5)  # must not raise

    assert (tmp_path / "drift.json").read_text() == before, "the live state was damaged"
    assert not (tmp_path / "drift.json.tmp").exists(), "the torn temp file was left behind"
    # Loaded by explicit path: undoing the monkeypatch here would also undo DISTIL_HOME
    # and point this assertion at the developer's real ~/.distil.
    assert LiveDrift.load(tmp_path / "drift.json").monitor.tripped, "the breach was lost"
