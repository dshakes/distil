"""The certification gate — decision-equivalence on a corpus, then TOST.

For each turn we compare the agent's decision on the compressed context against
its decision on the uncompressed context. The per-turn outcome is 1.0 if they
match, else 0.0; the paired difference vs the (always-1.0) baseline feeds TOST.
A strategy ships only if it is certified non-inferior.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..compress.strategies import REGISTRY, Strategy
from ..replay.runner import AgentRunner, DeterministicRunner
from ..trajectory import Trajectory
from .stats import TostResult, tost


@dataclass
class TurnDivergence:
    turn: int
    baseline_decision: str
    compressed_decision: str
    traj_id: str = ""  # set by certify_pooled so grouped output stays readable

    @property
    def matched(self) -> bool:
        return self.baseline_decision == self.compressed_decision


@dataclass
class CertReport:
    strategy: str
    divergences: list[TurnDivergence]
    tost: TostResult
    # A/A self-agreement rate when the aa_control ran (None otherwise). A live
    # model on an ambiguous turn disagrees with ITSELF across draws; divergence
    # below that floor cannot indict compression (the shadow v2->v3 lesson).
    aa_match_rate: float | None = None

    @property
    def match_rate(self) -> float:
        if not self.divergences:
            return 1.0
        return sum(d.matched for d in self.divergences) / len(self.divergences)

    @property
    def verdict(self) -> str:
        return self.tost.verdict


def certify(
    traj: Trajectory,
    strategy: str | Strategy,
    *,
    runner: AgentRunner | None = None,
    margin: float = 0.02,
    alpha: float = 0.05,
    aa_control: bool = False,
) -> CertReport:
    """Certify decision-equivalence of *strategy* on *traj*.

    With ``aa_control=True`` each turn also draws a SECOND baseline decision and
    the paired TOST diff becomes (A/B match) − (A/A match): compression is
    certified non-inferior to the model's own self-agreement floor rather than
    to perfect determinism. Live models can't pin temperature to 0, so grading
    an ambiguous turn against 1.0 confounds the model's run-to-run variance
    with compression-induced divergence — exactly the false-positive mode the
    shadow-signature v3 fix removed. With the offline DeterministicRunner the
    second draw is identical, the A/A floor is exactly 1.0, and the numbers
    reduce to the plain gate — so the per-commit CI semantics are unchanged.
    """
    runner = runner or DeterministicRunner()
    fn: Strategy = REGISTRY[strategy] if isinstance(strategy, str) else strategy
    name = strategy if isinstance(strategy, str) else getattr(strategy, "__name__", "custom")

    divergences, diffs, aa_matches = _turn_outcomes(traj, fn, runner, aa_control)
    aa_rate = (sum(aa_matches) / len(aa_matches)) if aa_matches else None
    return CertReport(name, divergences, tost(diffs, margin=margin, alpha=alpha), aa_rate)


def certify_pooled(
    trajs: list[Trajectory],
    strategy: str | Strategy,
    *,
    runner: AgentRunner | None = None,
    margin: float = 0.02,
    alpha: float = 0.05,
    aa_control: bool = False,
) -> CertReport:
    """One pooled TOST over EVERY turn of every trajectory.

    Per-trajectory verdicts on a live (stochastic) grader are underpowered — a
    handful of turns cannot separate compression-induced divergence from the
    model's own run-to-run variance, so verdicts flip between runs. Pooling all
    corpus turns into a single paired test is the statistically honest unit for
    an unattended live gate; per-trajectory verdicts remain the right unit for
    the offline deterministic oracle, where variance is exactly zero.
    """
    runner = runner or DeterministicRunner()
    fn: Strategy = REGISTRY[strategy] if isinstance(strategy, str) else strategy
    name = strategy if isinstance(strategy, str) else getattr(strategy, "__name__", "custom")

    divergences: list[TurnDivergence] = []
    diffs: list[float] = []
    aa_matches: list[bool] = []
    for traj in trajs:
        d, f, a = _turn_outcomes(traj, fn, runner, aa_control)
        for dv in d:
            dv.traj_id = traj.id
        divergences.extend(d)
        diffs.extend(f)
        aa_matches.extend(a)

    aa_rate = (sum(aa_matches) / len(aa_matches)) if aa_matches else None
    return CertReport(name, divergences, tost(diffs, margin=margin, alpha=alpha), aa_rate)


def _turn_outcomes(
    traj: Trajectory,
    fn: Strategy,
    runner: AgentRunner,
    aa_control: bool,
) -> tuple[list[TurnDivergence], list[float], list[bool]]:
    divergences: list[TurnDivergence] = []
    diffs: list[float] = []
    aa_matches: list[bool] = []
    for turn in traj.turns:
        base = runner.decide(turn.blocks)
        comp = runner.decide(fn(turn.blocks, turn.index))
        divergences.append(TurnDivergence(turn.index, base, comp))
        if aa_control:
            aa = base == runner.decide(turn.blocks)  # independent second draw
            aa_matches.append(aa)
            # (A/B) − (A/A): 0 when compression tracks the self-agreement floor,
            # negative only when compression diverges where the model is stable.
            diffs.append((1.0 if base == comp else 0.0) - (1.0 if aa else 0.0))
        else:
            # compressed_score - baseline_score; baseline scores 1.0 against
            # itself, compressed scores 1.0 only if its decision matches.
            diffs.append((1.0 if base == comp else 0.0) - 1.0)
    return divergences, diffs, aa_matches
