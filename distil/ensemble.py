"""RESEARCH-ONLY — not on the request path; see docs/research.html.

Cross-family grader ensemble — robustness to a single unfaithful grader.

Decision-equivalence is graded by a model that maps context to a canonical next-action
fingerprint (:class:`distil.replay.runner.AgentRunner`). A standing limitation of the
certificate is that every reported number uses *one* grader family, so a blind spot in that
family could bias the measured decision-change rate.

This module grades with several graders (ideally different model families) and aggregates their
per-turn change judgments. The principled default is **"any"**: a turn counts as a decision
change if *any* grader sees one. That choice is deliberately conservative — it can only *raise*
the measured risk, never lower it — so the Decision-Equivalence Risk Certificate built on top of
an "any"-ensemble loss stays valid even if one grader family is unfaithful (an unfaithful grader
can make us *over*-report risk and refuse to certify, never *under*-report and over-certify).
``"majority"`` and ``"unanimous"`` are available when over-conservatism costs too much savings.

Scope: the aggregation logic is implemented and tested here; running it end-to-end needs real
grader models from more than one family (the Anthropic/OpenAI/CLI runners already exist), which
is a live-API step rather than a unit test. Multi-family validation is a tracked GA item
(`docs/GA_READINESS.md`).
"""

from __future__ import annotations

from collections.abc import Sequence

from distil.replay.runner import AgentRunner
from distil.trajectory import Block

AGGREGATIONS = ("any", "majority", "unanimous")


class EnsembleGrader:
    """Grade decision-change with several graders, aggregated conservatively by default."""

    def __init__(self, graders: Sequence[AgentRunner], aggregate: str = "any"):
        if not graders:
            raise ValueError("need at least one grader")
        if aggregate not in AGGREGATIONS:
            raise ValueError(f"aggregate must be one of {AGGREGATIONS}, got {aggregate!r}")
        self.graders = list(graders)
        self.aggregate = aggregate

    def votes(self, base: list[Block], compressed: list[Block]) -> list[bool]:
        """Per-grader judgments: did the decision change under compression? (each grader
        compares its own decision on ``base`` vs ``compressed`` — fingerprints are only
        comparable within a family)."""
        return [g.decide(base) != g.decide(compressed) for g in self.graders]

    def changed(self, base: list[Block], compressed: list[Block]) -> bool:
        """Aggregated decision-change judgment for one turn."""
        votes = self.votes(base, compressed)
        k = sum(votes)
        if self.aggregate == "any":
            return k >= 1
        if self.aggregate == "unanimous":
            return k == len(votes)
        return k > len(votes) / 2  # majority

    def loss(self, base: list[Block], compressed: list[Block]) -> float:
        """0/1 decision-change loss for one turn under the chosen aggregation."""
        return 1.0 if self.changed(base, compressed) else 0.0


def ensemble_losses(
    grader: EnsembleGrader, pairs: Sequence[tuple[list[Block], list[Block]]]
) -> list[float]:
    """Per-turn ensemble losses over ``(base, compressed)`` pairs — feeds the certificate
    (:func:`distil.conformal.tight_risk_bound` / ``ltt_certify``)."""
    return [grader.loss(base, comp) for base, comp in pairs]
