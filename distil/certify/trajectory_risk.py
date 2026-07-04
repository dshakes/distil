"""Trajectory-level risk certificate — certify what users actually care about.

The lesson of distil's own E7 experiment (and of the 2024–26 agent-compression
literature, e.g. arXiv 2412.17483, 2510.00615): a statistically valid PER-STEP
certificate — "the compressed context yields the same next action" — can pass
while END-TO-END task success collapses, because per-step fidelity metrics
systematically overpredict multi-step success. Certifying next-action
equivalence and implying task fidelity is a category error.

This module certifies the right invariant instead: the TRAJECTORY-level loss —
did the task still succeed end-to-end under compression? — using Conformal
Risk Control and Learn-Then-Test (Angelopoulos et al., arXiv 2208.02814) on
matched full-context/compressed runs of complete tasks.

Honest scope (stated, because a guarantee without its assumptions is
advertising):

* **Exchangeability.** The bound holds for future trajectories exchangeable
  with the calibration set. Agent workloads drift; pair the certificate with
  :func:`drift_monitor` (an anytime-valid e-process) and RE-CALIBRATE when it
  alarms — a static certificate silently loses coverage under shift.
* **The loss is the task outcome you measured.** ``degraded`` should come from
  a real end-to-end evaluation (test suite passed, task rubric met), not a
  proxy metric — proxies are exactly the failure this module exists to fix.
* **No adaptivity.** Trajectories used to tune the compressor must not be
  reused as calibration (that breaks exchangeability a second way); keep a
  held-out calibration stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..conformal import hb_pvalue, ltt_certify, tight_risk_bound


@dataclass(frozen=True)
class TrajectoryOutcome:
    """One matched pair: the same task run with full context and compressed.

    ``full_success``/``compressed_success`` are the END-TO-END task outcomes
    (e.g. SWE-bench test suite passed) — never per-step similarity scores.
    """

    task_id: str
    full_success: bool
    compressed_success: bool

    @property
    def loss(self) -> float:
        """1.0 when compression degraded a task the full context solved.

        A task the full context also failed carries no loss — the certificate
        bounds the *degradation* attributable to compression, not the agent's
        base failure rate.
        """
        return 1.0 if (self.full_success and not self.compressed_success) else 0.0


@dataclass(frozen=True)
class TrajectoryRiskCertificate:
    """A distribution-free bound on end-to-end degradation risk."""

    n: int
    empirical_risk: float
    risk_bound: float  # (1-delta)-confidence upper bound on E[loss]
    alpha: float  # the target risk level tested
    delta: float  # confidence budget
    certified: bool  # P(risk <= alpha) >= 1 - delta held
    p_value: float
    assumptions: str

    @property
    def statement(self) -> str:
        if self.certified:
            return (
                f"With confidence {(1 - self.delta) * 100:.0f}%, compression degrades at most "
                f"{self.alpha * 100:.1f}% of tasks the full context would have solved "
                f"(observed {self.empirical_risk * 100:.1f}% over {self.n} matched trajector{'y' if self.n == 1 else 'ies'}). "
                f"Valid under: {self.assumptions}"
            )
        return (
            f"NOT CERTIFIED at α={self.alpha}: observed {self.empirical_risk * 100:.1f}% "
            f"degradation over {self.n} matched trajector{'y' if self.n == 1 else 'ies'} "
            f"(upper bound {self.risk_bound * 100:.1f}%). Collect more calibration "
            "trajectories or reduce compression aggressiveness."
        )


_ASSUMPTIONS = (
    "future trajectories exchangeable with calibration; loss = measured end-to-end "
    "task outcome; calibration held out from compressor tuning; recalibrate on drift"
)


def certify_trajectory_risk(
    outcomes: Sequence[TrajectoryOutcome],
    *,
    alpha: float = 0.05,
    delta: float = 0.05,
    min_n: int = 20,
) -> TrajectoryRiskCertificate:
    """Test H0: trajectory-degradation risk > *alpha* at confidence 1-*delta*.

    Refuses to certify (rather than overclaiming) below *min_n* matched
    trajectories — a certificate from a handful of samples is noise dressed
    up as a guarantee.
    """
    losses = [o.loss for o in outcomes]
    n = len(losses)
    rhat = (sum(losses) / n) if n else 1.0
    if n < min_n:
        return TrajectoryRiskCertificate(
            n=n,
            empirical_risk=rhat,
            risk_bound=1.0,
            alpha=alpha,
            delta=delta,
            certified=False,
            p_value=1.0,
            assumptions=_ASSUMPTIONS,
        )
    p = hb_pvalue(rhat, n, alpha)
    bound = tight_risk_bound(losses, delta)
    return TrajectoryRiskCertificate(
        n=n,
        empirical_risk=rhat,
        risk_bound=bound,
        alpha=alpha,
        delta=delta,
        certified=p <= delta,
        p_value=p,
        assumptions=_ASSUMPTIONS,
    )


def select_compression_level(
    outcomes_by_level: Sequence[Sequence[TrajectoryOutcome]],
    *,
    alpha: float = 0.05,
    delta: float = 0.05,
) -> int:
    """Learn-Then-Test over a compression ladder, on TRAJECTORY losses.

    ``outcomes_by_level`` is ordered least→most aggressive (risk monotone
    non-decreasing). Returns the most aggressive level whose end-to-end
    degradation risk is certified ≤ alpha at confidence 1-delta, or -1 when
    even the mildest level fails. Fixed-sequence testing gives family-wise
    validity with no multiplicity penalty.
    """
    level_losses = [[o.loss for o in level] for level in outcomes_by_level]
    index, _pvals = ltt_certify(level_losses, alpha=alpha, delta=delta)
    return index


def drift_monitor(alpha: float = 0.05, delta: float = 0.05):
    """An anytime-valid monitor for the certificate's exchangeability assumption.

    Feed it every post-deployment trajectory loss (``monitor.update(loss)``);
    it returns True — and the certificate must be considered STALE and
    recalibrated — as soon as an e-process accumulates evidence that live risk
    exceeds the certified alpha. Wraps :class:`distil.drift.DriftMonitor`.
    """
    from ..drift import DriftMonitor

    return DriftMonitor(alpha=alpha, delta=delta)
