"""RESEARCH-ONLY — not on the request path; see docs/research.html.

Speculative expansion — pay for full context only when divergence risk is certified low.

The idea, borrowed from speculative decoding: most turns are safe to run on cheap compressed
context; only a few actually need the full context to keep the agent's decision. A cheap
*risk score* predicts, per turn, whether the compressed decision would diverge from the full
decision. The controller escalates to full context only when the score crosses a threshold —
so cost is mostly-cheap-plus-occasionally-full instead of always-full.

The motto is preserved by **certifying the miss rate**: among the turns we did NOT escalate,
the rate at which the compressed decision actually diverges from full is bounded by a
distribution-free (1−δ) conformal upper bound (:mod:`distil.conformal`). The threshold is
chosen as the *least* escalation (cheapest) whose certified miss rate is ≤ α.

Scope (honest): this module is the controller + its certificate. It needs (a) a risk-score
function and (b) labeled calibration turns (compressed-vs-full decision) from your traffic;
producing those requires a live calibration run, which is why end-to-end speculative savings
are a tracked GA item (`docs/GA_READINESS.md`) rather than a shipped default. The control
logic and the certificate are implemented and tested here.
"""

from __future__ import annotations

from dataclasses import dataclass

from distil.conformal import tight_risk_bound


@dataclass(frozen=True)
class SpeculativeController:
    """A calibrated escalation policy: escalate to full context iff ``score >= threshold``."""

    threshold: float
    certified_miss_rate: float  # (1−δ) upper bound on P(diverge | kept compressed)
    escalation_rate: float  # fraction of turns escalated to full (the cost paid)
    alpha: float
    n: int
    feasible: bool  # False => no threshold certifies; fail safe to always-full

    def decide(self, score: float) -> str:
        """Return ``"full"`` (escalate) or ``"compressed"`` (cheap) for a turn's risk score."""
        if not self.feasible:
            return "full"
        return "full" if score >= self.threshold else "compressed"


def calibrate_speculative(
    scores: list[float], diverged: list[int], *, alpha: float = 0.05, delta: float = 0.05
) -> SpeculativeController:
    """Calibrate the escalation threshold on labeled turns.

    Args:
        scores: per-turn risk score in [0,1] (higher = more likely the compressed decision
            diverges from the full-context decision).
        diverged: per-turn label, 1 iff the compressed decision differs from the full decision.
        alpha: tolerated certified miss rate among non-escalated turns.
        delta: certificate failure probability (1−confidence).

    Returns the controller with the **smallest** threshold (least escalation, lowest cost)
    whose certified miss rate is ≤ ``alpha``. If none qualifies, ``feasible=False`` and the
    controller fails safe to always-escalate (full context).
    """
    if len(scores) != len(diverged):
        raise ValueError("scores and diverged must align")
    n = len(scores)
    if n == 0:
        return SpeculativeController(1.0, 1.0, 1.0, alpha, 0, feasible=False)

    # Escalate (full) iff score >= threshold, so a HIGHER threshold keeps MORE turns compressed
    # (cheaper) but risks more misses; a LOWER threshold escalates more (safer). Feasibility
    # (certified miss <= alpha) grows as the threshold drops. We want the cheapest feasible
    # point = the HIGHEST threshold that still certifies, so scan candidates descending and
    # take the first feasible. Threshold = min(scores) escalates everything (zero misses), so a
    # feasible point always exists — the worst case is escalation_rate ~= 1.0 (no savings).
    candidates = sorted(set(scores), reverse=True) + [min(scores)]
    for thr in candidates:
        kept = [d for s, d in zip(scores, diverged) if s < thr]  # non-escalated turns
        escalation = (n - len(kept)) / n
        miss_bound = tight_risk_bound([float(d) for d in kept], delta) if kept else 0.0
        if miss_bound <= alpha:
            return SpeculativeController(
                threshold=thr,
                certified_miss_rate=round(miss_bound, 4),
                escalation_rate=round(escalation, 4),
                alpha=alpha,
                n=n,
                feasible=True,
            )
    # Unreachable in practice (escalate-all certifies), but fail safe to full if it happens.
    return SpeculativeController(min(scores), 0.0, 1.0, alpha, n, feasible=False)
