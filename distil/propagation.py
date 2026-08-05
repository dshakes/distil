"""Error propagation — does a compression error at turn k cause a failure later?

Every other probe in distil scores one turn against its own original. That measures
*incidence* and misses *consequence*: a fact dropped at turn 3 does no visible damage
until turn 9, when the agent re-reads a file it already read, or re-derives a decision
it already made, or acts on the gap.

The 2026 survey on context compression for LLM agents names error propagation as a
missing metric and does not operationalise it. This module does, with the weakest
claim the data actually supports.

Method. Given per-turn compression fidelity and per-turn decision changes, compute for
each lag L the *lift*:

    lift(L) = P(decision changed at turn t | fidelity dropped at turn t-L)
              ------------------------------------------------------------
                            P(decision changed at turn t)

lift ≈ 1 means a drop L turns back tells you nothing — errors are local. lift > 1
means damage is arriving late, and the lag profile says how late.

**This is association, not causation, and the module says so in its own output.** A
turn-3 drop and a turn-9 change can share a cause (a hard trajectory) without one
producing the other. What the lift profile does establish is a *bound*: if lift is
flat at 1 across all lags, propagation is not happening at a measurable rate, which
is the claim distil actually wants to make about a conservative tier. Asserting more
than that from observational turn data would be overclaiming — which, per
:mod:`distil.overclaim`, is a thing this codebase measures rather than does.

**Known limit: periodic workloads alias.** If a trajectory drops fidelity every k
turns and also changes decision every k turns — a polling loop, a retry cadence, a
fixed tool rotation — every lag that is a multiple of k shows elevated lift with no
propagation whatsoever. The lag axis cannot distinguish a real delayed effect from a
shared period. Read an elevated verdict on a visibly rhythmic trajectory as "inspect
this", never as "confirmed". Aperiodic traffic does not have this problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# Minimum exposed turns before a lag's rate is worth comparing to the base rate.
# At 3, the bundled corpus produced a "propagating" verdict off 2 changes in 8 turns —
# a difference of one event either way would have flipped it. Publishing that as a
# finding would be exactly the overclaim `distil.overclaim` exists to catch. Ten is
# still small; it is the floor at which a rate is worth quoting at all, not a
# significance test. Below it the profile is reported and no verdict is asserted.
_MIN_EXPOSED = 10


@dataclass
class TurnSignal:
    """One turn's observation: how well it compressed, and whether behaviour changed.

    ``decision_changed`` MUST come from an independent decision oracle — a real
    baseline-vs-compressed comparison of what the agent would do. It must NOT be
    derived from the fidelity probes, however tempting: feeding a probe failure in as
    the "decision" makes the whole analysis circular. Lift then measures the
    correlation of probe failures with themselves, lag 0 is elevated by construction,
    and the verdict can fire without a single decision having changed. An early
    version of the caller did exactly that; this note exists so it is not repeated.
    """

    fidelity: float  # 1.0 = nothing lost this turn
    decision_changed: bool  # from a decision oracle, NOT from a fidelity probe


@dataclass
class LagLift:
    lag: int
    exposed: int  # turns preceded, L back, by a fidelity drop
    changed: int  # of those, how many changed decision
    lift: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lag": self.lag,
            "exposed": self.exposed,
            "changed": self.changed,
            "lift": round(self.lift, 4),
        }


@dataclass
class PropagationReport:
    base_rate: float = 0.0
    turns: int = 0
    lags: list[LagLift] = field(default_factory=list)
    threshold: float = 1.0

    @property
    def worst(self) -> LagLift | None:
        """The lag with the strongest association, if any lag had exposure."""
        scored = [lag for lag in self.lags if lag.exposed]
        return max(scored, key=lambda lag: lag.lift) if scored else None

    @property
    def propagates(self) -> bool:
        """True when some lag ≥ 1 shows meaningfully elevated risk.

        Lag 0 is excluded deliberately: a fidelity drop and a decision change in the
        SAME turn is co-incidence, not propagation, and including it would let local
        damage masquerade as a downstream effect.
        """
        return any(
            lag.lift > 1.25 and lag.exposed >= _MIN_EXPOSED for lag in self.lags if lag.lag >= 1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "turns": self.turns,
            "base_rate": round(self.base_rate, 4),
            "propagates": self.propagates,
            "lags": [lag.to_dict() for lag in self.lags],
        }


def analyse(
    signals: Sequence[TurnSignal], *, max_lag: int = 5, drop_threshold: float = 1.0
) -> PropagationReport:
    """Lag-lift profile for one trajectory.

    ``drop_threshold`` is the fidelity below which a turn counts as damaged; the
    default of 1.0 means "anything less than perfect", which is the strictest reading
    and the one that makes a flat profile a strong result.
    """
    n = len(signals)
    changed = [s.decision_changed for s in signals]
    dropped = [s.fidelity < drop_threshold for s in signals]
    total_changed = sum(changed)
    base = total_changed / n if n else 0.0
    report = PropagationReport(base_rate=base, turns=n, threshold=drop_threshold)
    for lag in range(0, max_lag + 1):
        exposed = 0
        hits = 0
        for t in range(lag, n):
            if dropped[t - lag]:
                exposed += 1
                if changed[t]:
                    hits += 1
        rate = hits / exposed if exposed else 0.0
        lift = (rate / base) if base else 0.0
        report.lags.append(LagLift(lag=lag, exposed=exposed, changed=hits, lift=lift))
    return report


def analyse_many(
    sequences: Sequence[Sequence[TurnSignal]], *, max_lag: int = 5, drop_threshold: float = 1.0
) -> PropagationReport:
    """Lag-lift across several INDEPENDENT trajectories, pooled without crossing them.

    Concatenating trajectories and analysing the result as one sequence is wrong and
    not obviously so: it lets a fidelity drop in the last turn of trajectory A be
    scored as causing a decision change in the first turn of trajectory B. There is
    no causal path across that seam — different task, different context, often a
    different day — and with many short trajectories the seams can dominate. A
    cross-audit reproduced `propagates=True` from a corpus of independent one-turn
    trajectories, where by construction nothing can propagate at all.

    Counts are pooled per lag across sequences; only the *pairing* is confined.
    """
    seqs = [list(s) for s in sequences if s]
    total_turns = sum(len(s) for s in seqs)
    total_changed = sum(1 for s in seqs for x in s if x.decision_changed)
    base = total_changed / total_turns if total_turns else 0.0
    report = PropagationReport(base_rate=base, turns=total_turns, threshold=drop_threshold)

    for lag in range(0, max_lag + 1):
        exposed = 0
        hits = 0
        for signals in seqs:
            changed = [x.decision_changed for x in signals]
            dropped = [x.fidelity < drop_threshold for x in signals]
            for t in range(lag, len(signals)):
                if dropped[t - lag]:
                    exposed += 1
                    if changed[t]:
                        hits += 1
        rate = hits / exposed if exposed else 0.0
        lift = (rate / base) if base else 0.0
        report.lags.append(LagLift(lag=lag, exposed=exposed, changed=hits, lift=lift))
    return report


def format_report(report: PropagationReport) -> str:
    if not report.turns:
        return "propagation: no turns to analyse."
    lines = [
        f"error propagation  ({report.turns} turns, base decision-change rate {report.base_rate:.1%})",
    ]
    for lag in report.lags:
        if not lag.exposed:
            continue
        marker = (
            "  <-- elevated"
            if lag.lag >= 1 and lag.lift > 1.25 and lag.exposed >= _MIN_EXPOSED
            else ""
        )
        lines.append(
            f"  lag {lag.lag}: {lag.changed:>3}/{lag.exposed:<3} changed   lift {lag.lift:5.2f}{marker}"
        )
    if report.base_rate == 0.0:
        # Nothing changed, so nothing propagated — but saying "no propagation" here
        # would dress a vacuous truth as an earned result. The analysis had no events
        # to work with and should say so.
        verdict = "no decisions changed at all — nothing to propagate, and nothing tested"
    elif report.propagates:
        verdict = "damage is arriving in LATER turns — compression errors are propagating"
    else:
        verdict = "no measurable propagation — errors stay local to their turn"
    lines.append(f"  verdict: {verdict}")
    lines.append("  (association, not causation: shared difficulty can produce the same profile)")
    return "\n".join(lines)
