"""The gate must PASS lossless strategies and FAIL quality-degrading ones."""

from distil.certify.gate import certify
from distil.certify.stats import tost
from distil.replay.ablation import discover
from distil.trajectory import Trajectory

CORPUS = "corpus/sample_trajectory.json"


def test_tost_passes_when_lossless():
    # every paired difference is zero -> non-inferior at any positive margin
    res = tost([0.0, 0.0, 0.0, 0.0], margin=0.02)
    assert res.non_inferior


def test_tost_fails_on_real_degradation():
    res = tost([0.0, -1.0, -1.0, 0.0], margin=0.02)
    assert not res.non_inferior


def test_distil_strategy_is_certified_non_inferior():
    report = certify(Trajectory.load(CORPUS), "distil")
    assert report.match_rate == 1.0
    assert report.verdict == "PASS"


def test_aggressive_strategy_is_rejected():
    report = certify(Trajectory.load(CORPUS), "aggressive")
    assert report.match_rate < 1.0
    assert report.verdict == "FAIL"


def test_ablation_finds_the_speculative_docs_prunable():
    report = discover(Trajectory.load(CORPUS))
    prunable_ids = {v.block_id for v in report.prunable}
    # the speculative retrieved docs never changed a decision
    assert {"doc-0", "doc-1", "doc-2", "doc-3"} <= prunable_ids
    # the system prompt and tool schema DID drive decisions -> kept
    assert "system" not in prunable_ids
    assert "tools" not in prunable_ids
    assert report.tokens_freed > 0


# ---------------------------------------------------------------------------
# A/A control + pooled certify (the live-gate methodology, tested offline)
# ---------------------------------------------------------------------------

class _UniqueRunner:
    """Stochastic stand-in with a 0% self-agreement floor: every draw is a new
    decision. Under the A/A control such a model cannot indict compression —
    every A/B mismatch is cancelled by an A/A mismatch."""

    def __init__(self):
        self.calls = 0

    def decide(self, blocks):
        self.calls += 1
        return f"decision-{self.calls}"


def test_aa_control_is_noop_for_deterministic_runner():
    """With a deterministic runner the second A draw is identical, the floor is
    exactly 1.0, and the report reduces to the plain gate — per-commit CI
    semantics are provably unchanged by the control."""
    from distil.certify.gate import certify
    from distil.trajectory import Trajectory

    traj = Trajectory.load(CORPUS)
    plain = certify(traj, "distil")
    controlled = certify(traj, "distil", aa_control=True)
    assert plain.aa_match_rate is None
    assert controlled.aa_match_rate == 1.0
    assert controlled.tost.mean_diff == plain.tost.mean_diff
    assert controlled.verdict == plain.verdict


def test_aa_control_absorbs_model_self_disagreement():
    """A model that never agrees with itself must not indict compression: the
    paired diff is (A/B match) - (A/A match) = 0 - 0 on every turn."""
    from distil.certify.gate import certify
    from distil.trajectory import Trajectory

    traj = Trajectory.load(CORPUS)
    report = certify(traj, "none", runner=_UniqueRunner(), aa_control=True, margin=0.1)
    assert report.aa_match_rate == 0.0  # total self-disagreement
    assert report.match_rate == 0.0  # A/B also never matches
    assert report.tost.mean_diff == 0.0  # ...and the control cancels it exactly
    assert report.tost.non_inferior  # no divergence beyond the floor -> PASS


def test_without_aa_control_the_same_runner_fails():
    """The raw gate on the same self-disagreeing model FAILS — proving the
    control is what separates model variance from compression divergence."""
    from distil.certify.gate import certify
    from distil.trajectory import Trajectory

    traj = Trajectory.load(CORPUS)
    report = certify(traj, "none", runner=_UniqueRunner(), aa_control=False, margin=0.1)
    assert report.tost.mean_diff == -1.0
    assert not report.tost.non_inferior


def test_certify_pooled_pools_turns_across_trajectories():
    from distil.certify.gate import certify, certify_pooled
    from distil.trajectory import Trajectory

    t1 = Trajectory.load(CORPUS)
    single = certify(t1, "distil")
    pooled = certify_pooled([t1, t1], "distil")
    assert len(pooled.divergences) == 2 * len(single.divergences)
    assert all(d.traj_id == t1.id for d in pooled.divergences)
    assert pooled.verdict == "PASS"


class _ProseTargetRunner:
    """Stable ACTION, free-text TARGET that never repeats — models the live
    failure mode (annotateticket with a paragraph-long target) that made pooled
    verdicts flip between identical runs."""

    def __init__(self):
        self.calls = 0

    def decide(self, blocks):
        self.calls += 1
        return f'{{"action":"annotateticket","target":"prose variant {self.calls}"}}'


def test_aa_probe_downgrades_unmeasurable_targets_to_action_grading():
    """When the baseline's own two draws disagree on the target, the turn is
    graded action-only — a stable action then matches on both arms, so a
    prose-target model contributes zero diff (and zero variance) instead of a
    coin flip per turn."""
    from distil.certify.gate import certify
    from distil.trajectory import Trajectory

    traj = Trajectory.load(CORPUS)
    report = certify(traj, "none", runner=_ProseTargetRunner(), aa_control=True, margin=0.1)
    assert all(d.granularity == "action" for d in report.divergences)
    assert all(d.matched for d in report.divergences)  # action stable -> graded ok
    assert report.aa_match_rate == 1.0  # at the granularity the turn supports
    assert report.tost.mean_diff == 0.0
    assert report.tost.non_inferior


def test_aa_probe_still_catches_real_action_changes():
    """The downgrade must never mask a genuine action change: a runner whose
    compressed arm picks a DIFFERENT action stays a divergence."""
    from distil.certify.gate import certify
    from distil.trajectory import Trajectory

    class _ActionFlipRunner:
        def __init__(self):
            self.calls = 0

        def decide(self, blocks):
            self.calls += 1
            # call order per turn: base, comp, base2 — comp gets a different action
            phase = (self.calls - 1) % 3
            action = "readfile" if phase == 1 else "runtests"
            return f'{{"action":"{action}","target":"prose variant {self.calls}"}}'

    traj = Trajectory.load(CORPUS)
    report = certify(traj, "none", runner=_ActionFlipRunner(), aa_control=True, margin=0.1)
    assert all(d.granularity == "action" for d in report.divergences)
    assert not any(d.matched for d in report.divergences)  # action change caught
    assert report.tost.mean_diff == -1.0
    assert not report.tost.non_inferior
