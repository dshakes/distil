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

import pytest

from distil.conformal import BUDGET_ALPHA, BUDGET_DELTA
from distil.drift import DriftGuard, DriftMonitor, LiveDrift, _bootstrap, fold, paired_loss, release

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
# One e-process: restarts and hot-swaps continue it, they never re-branch it
# ---------------------------------------------------------------------------


def test_restarts_continue_one_capital_path_exactly(tmp_path, monkeypatch):
    """The same rows folded by one long-lived proxy, or by a proxy that restarts (or is
    hot-swapped) every few rows, must produce the SAME capital, bit for bit. Anything else
    means a restart took a fresh look at the evidence — an extra look Ville does not cover."""
    rng = random.Random(7)
    diffs = [_draw(rng, BUDGET_ALPHA, 0.26) for _ in range(400)]

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "one"))
    for d in diffs:
        fold([d])
    single = LiveDrift.load()

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "many"))
    for chunk in range(0, len(diffs), 7):
        guard = DriftGuard.start(watch=False)  # a restart / hot-swap worker
        for d in diffs[chunk : chunk + 7]:
            guard.observe(d)
    restarted = LiveDrift.load()

    assert restarted.monitor.n == single.monitor.n == len(diffs)
    assert restarted.monitor.capital == single.monitor.capital
    assert restarted.monitor.tripped == single.monitor.tripped


def test_null_false_alarm_stays_under_delta_across_restarts(tmp_path, monkeypatch):
    """REGRESSION GUARD, not a δ-level check. Under the null, a proxy restarts every 20
    samples. When each restart re-branched from stale capital, the rate grew as (k+1)·δ
    — far above this bound. With 150 trials (file I/O per fold) the tolerance is
    δ + 2σ ≈ 0.086, loose by construction; the exact-capital test above is the sharp
    check that restarts add no looks, and the in-memory gate is the δ-level one."""
    trials, horizon, every = 150, 300, 20
    tolerance = BUDGET_DELTA + 2 * math.sqrt(BUDGET_DELTA * (1 - BUDGET_DELTA) / trials)
    alarms = 0
    for tr in range(trials):
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path / str(tr)))
        rng = random.Random(0x5EED ^ tr)
        guard = DriftGuard.start(watch=False)
        for i in range(horizon):
            if i and i % every == 0:
                guard = DriftGuard.start(watch=False)
            guard.observe(_draw(rng, BUDGET_ALPHA, 0.26))
            if guard.engaged:
                break
        alarms += guard.engaged
    assert alarms / trials <= tolerance, f"false-alarm {alarms / trials:.4f} > {tolerance:.4f}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_every_folded_row_is_counted_once(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    fold([0] * 60)
    assert LiveDrift.load().monitor.n == 60
    assert LiveDrift.load().monitor.n == 60  # reading is not folding
    assert fold([0, 0]).monitor.n == 62


def test_breach_is_sticky_and_names_the_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    state = fold([-1] * 200)
    assert state.monitor.tripped
    assert 0 < state.tripped_at <= 200 and state.tripped_ts > 0
    line = state.line()
    assert line.startswith("BREACHED at sample ")
    assert "distil calibrate" in line and "distil reset --drift-guard" in line
    # A clean run afterwards does not un-breach it — the release is explicit.
    healed = fold([1] * 500)
    assert healed.monitor.tripped
    assert LiveDrift.load().line().startswith("BREACHED")


def test_bootstrap_rebuilds_a_pre_schema_state_from_the_ledger_once(tmp_path, monkeypatch):
    """Upgrading: the old e-process was folded from shadow.jsonl at wrap exit. The first
    proxy start rebuilds it from that file once; after that only fold() writes."""
    from distil.shadow import SIG_VERSION

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    with (tmp_path / "shadow.jsonl").open("w", encoding="utf-8") as f:
        for _ in range(60):
            row = {"equivalent": True, "aa_equal": True, "kind": "paired", "sig": SIG_VERSION}
            f.write(json.dumps(row) + "\n")
    old = {"alpha": NULL_MEAN, "delta": BUDGET_DELTA, "n": 10, "consumed": 10, "stream": "x"}
    (tmp_path / "drift.json").write_text(json.dumps(old))

    assert _bootstrap().monitor.n == 60
    fold([0])
    assert _bootstrap().monitor.n == 61  # schema 2 now: no second rebuild


def test_release_leaves_a_fresh_state_that_bootstrap_does_not_refold(tmp_path, monkeypatch):
    """Otherwise the next proxy start would fold the rows that caused the trip straight
    back into the same breach, and the release would last until the first restart."""
    from distil.shadow import SIG_VERSION

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    with (tmp_path / "shadow.jsonl").open("w", encoding="utf-8") as f:
        for _ in range(200):
            row = {"equivalent": False, "aa_equal": True, "kind": "paired", "sig": SIG_VERSION}
            f.write(json.dumps(row) + "\n")
    old = {"alpha": NULL_MEAN, "delta": BUDGET_DELTA, "n": 0, "consumed": 0, "stream": "x"}
    (tmp_path / "drift.json").write_text(json.dumps(old))  # a pre-schema-2 file: migrates
    assert _bootstrap().monitor.tripped
    assert release("t1")
    assert (tmp_path / "drift.json.reset-t1").exists()
    after = _bootstrap()
    assert not after.monitor.tripped and after.monitor.n == 0


def test_budget_change_discards_stale_capital(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    fold([-1] * 100)
    p = tmp_path / "drift.json"
    raw = json.loads(p.read_text())
    raw["alpha"] = 0.99  # capital was bet against a different null
    p.write_text(json.dumps(raw))
    assert LiveDrift.load().monitor.n == 0


@pytest.mark.parametrize("junk", ["", "{not json", "[]", '"a string"'])
def test_an_unreadable_state_file_is_held_and_quarantined_not_overwritten(
    tmp_path, monkeypatch, junk
):
    """Zero-length, garbage, parseable-but-wrong: the file may have held a breach, so it
    is held (fail-safe), moved aside rather than overwritten, and the hold says how to
    release it. Before, it loaded as fresh and the next fold destroyed it."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    (tmp_path / "drift.json").write_text(junk)
    assert LiveDrift.load().held  # a reader holds without touching the file
    assert (tmp_path / "drift.json").read_text() == junk

    state = fold([0])
    assert state.held and not state.monitor.tripped
    moved = list(tmp_path.glob("drift.json.corrupt-*"))
    assert len(moved) == 1 and moved[0].read_text() == junk
    line = LiveDrift.load().line()
    assert line.startswith("HELD") and moved[0].name in line
    assert "distil reset --drift-guard" in line
    assert DriftGuard.start(watch=False).engaged

    release("t")
    assert not LiveDrift.load().held


def test_below_floor_says_so_and_prints_no_evalue(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    line = fold([0] * 10).line()
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


def test_a_wrongly_typed_field_is_held_as_corrupt(tmp_path, monkeypatch):
    """Well-formed JSON with a junk value is still a state nobody can vouch for."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    fold([0] * 10)
    p = tmp_path / "drift.json"
    for field, junk in (("capital", "not a number"), ("n", "bad"), ("n", -5)):
        raw = json.loads(p.read_text())
        raw[field] = junk
        p.write_text(json.dumps(raw))
        assert LiveDrift.load().held, (field, junk)
        release(field + str(junk))
        fold([0] * 10)


def test_a_missing_state_is_fresh_and_never_refolds_released_history(tmp_path, monkeypatch):
    """The review's blocker: release, then lose drift.json (deleted, or the fresh write
    failed). The next start must NOT re-fold the shadow history into the same breach."""
    from distil.shadow import SIG_VERSION

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    with (tmp_path / "shadow.jsonl").open("w", encoding="utf-8") as f:
        for _ in range(200):
            row = {"equivalent": False, "aa_equal": True, "kind": "paired", "sig": SIG_VERSION}
            f.write(json.dumps(row) + "\n")
    fold([-1] * 200)
    assert release("t")
    (tmp_path / "drift.json").unlink()
    guard = DriftGuard.start(watch=False)  # a restart
    assert not guard.engaged
    assert LiveDrift.load().monitor.n == 0


def test_release_raises_when_the_fresh_state_cannot_be_written(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    fold([-1] * 200)
    monkeypatch.setattr(LiveDrift, "_write", lambda self, p: False)
    with pytest.raises(OSError):
        release("t")


def test_an_unwritable_state_path_still_returns_a_verdict(tmp_path, monkeypatch):
    """The alarm is bookkeeping: it must never be the reason a proxy raises."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "blocked"))
    (tmp_path / "blocked").write_text("this is a file, not a directory")
    state = fold([0] * 60)
    assert state.monitor.n == 60
    assert state.line() == f"intact (e-value {state.monitor.evalue:.2f}, n=60)"


def test_an_interrupted_write_leaves_the_previous_state_intact(tmp_path, monkeypatch):
    """`tripped` is sticky and capital accumulates across sessions, so a torn write is not
    a loud corruption — it is a silent reset that reads "intact" over evidence that said
    BREACHED. The temp file absorbs the tear; the rename is what the reader ever sees."""
    import distil._filelock as _fl

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    assert fold([-1] * 200).monitor.tripped
    before = (tmp_path / "drift.json").read_text()

    def _die(src, dst):  # the crash lands after the temp file exists, before the rename
        raise OSError("interrupted")

    monkeypatch.setattr(_fl, "replace_retrying", _die)
    fold([0] * 5)  # must not raise

    assert (tmp_path / "drift.json").read_text() == before, "the live state was damaged"
    assert not (tmp_path / "drift.json.tmp").exists(), "the torn temp file was left behind"
    # Loaded by explicit path: undoing the monkeypatch here would also undo DISTIL_HOME
    # and point this assertion at the developer's real ~/.distil.
    assert LiveDrift.load(tmp_path / "drift.json").monitor.tripped, "the breach was lost"
