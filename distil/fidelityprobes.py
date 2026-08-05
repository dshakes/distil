"""The fidelity suite — run every state probe over a corpus and gate on the results.

:mod:`distil.retention` answers "is the fact still there". These four answer the
questions that survive a yes:

  * :mod:`distil.artifacts`    — is the file's *state* still right, or just its name?
  * :mod:`distil.overclaim`    — did the value keep its uncertainty?
  * :mod:`distil.continuation` — does the agent still know what is left to do?
  * :mod:`distil.propagation`  — does a loss at turn k show up as a change at turn k+n?

Each is scored per turn against that turn's own original, so the suite runs on any
trajectory without an answer key — the same property that lets the live meter run on
real traffic.

The gates are asymmetric on purpose. `stale` artifacts, `overclaimed` values and
`dropped_work` are *silent* failures: the agent proceeds confidently on a false
belief and nothing in the transcript says so. They gate at zero by default. Plain
loss is loud — the agent can see the gap — so it is reported but not gated here;
:mod:`distil.retention` already owns that number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from . import artifacts, continuation, overclaim, propagation
from .compress.tier1 import Tier1Reversible


@dataclass
class FidelityReport:
    trajectories: int = 0
    turns: int = 0
    state: artifacts.StateProbe = field(default_factory=artifacts.StateProbe)
    hedges: overclaim.OverclaimProbe = field(default_factory=overclaim.OverclaimProbe)
    plan: continuation.ContinuationProbe = field(default_factory=continuation.ContinuationProbe)
    prop: propagation.PropagationReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories": self.trajectories,
            "turns": self.turns,
            "artifact_state": self.state.to_dict(),
            "overclaim": self.hedges.to_dict(),
            "continuation": self.plan.to_dict(),
            "propagation": self.prop.to_dict() if self.prop else None,
        }

    @property
    def silent_failures(self) -> int:
        """Every failure the agent cannot see. The number that should be zero."""
        return self.state.stale + self.hedges.overclaimed + self.plan.dropped_work


def run(entries: Iterable[Any], *, compressor: Any = None) -> FidelityReport:
    """Score every turn of every trajectory, with real compression in the loop."""
    comp = compressor or Tier1Reversible()
    report = FidelityReport()
    signals: list[propagation.TurnSignal] = []

    for entry in entries:
        report.trajectories += 1
        # Artifact state and plan state are CROSS-TURN properties: a file created at
        # turn 2 and deleted at turn 4 only has a final state when both turns are in
        # scope. Scoring turn-by-turn would report the lost delete as a missing path
        # (`lost`) instead of a surviving path with the wrong state (`stale`) — which
        # is precisely the failure these probes exist to distinguish. So they are
        # scored once over the whole trajectory. Overclaim is genuinely local to a
        # block, so it stays per-block.
        traj_original: list[str] = []
        traj_compressed: list[str] = []

        for turn in entry.trajectory.turns:
            report.turns += 1
            originals = [b.text for b in turn.blocks]
            # The real compressors take a LIST OF BLOCKS and return a CompressResult.
            # An earlier version of this loop passed `b.text` (a str) block-by-block;
            # every call raised, a blanket `except` swallowed it, and the suite spent
            # its whole life comparing each original against itself and reporting
            # 100% fidelity. There is deliberately no try/except here now: a
            # compressor that cannot run must fail the gate loudly, because a probe
            # that silently grades a no-op is worse than no probe at all.
            compressed = [b.text for b in comp.compress(list(turn.blocks)).blocks]

            hd = overclaim.OverclaimProbe()
            for original, small in zip(originals, compressed):
                hd.add(overclaim.score(original, small))
            report.hedges.add(hd)

            # Per-turn signal for the lag analysis. Uses the LOCAL view deliberately:
            # propagation asks when damage becomes visible, so each turn is scored on
            # what that turn alone lost.
            local_state = artifacts.score(originals, compressed)
            local_plan = continuation.score(originals, compressed)
            signals.append(
                propagation.TurnSignal(
                    fidelity=min(local_state.fidelity, hd.fidelity, local_plan.pending_recall),
                    decision_changed=bool(
                        local_state.stale or hd.overclaimed or local_plan.dropped_work
                    ),
                )
            )
            traj_original.extend(originals)
            traj_compressed.extend(compressed)

        report.state.add(artifacts.score(traj_original, traj_compressed))
        report.plan.add(continuation.score(traj_original, traj_compressed))

    report.prop = propagation.analyse(signals)
    return report


def format_report(report: FidelityReport) -> str:
    lines = [
        f"Fidelity probes — {report.trajectories} trajectories, {report.turns} turns",
        "",
        artifacts.format_probe(report.state),
        "",
        overclaim.format_probe(report.hedges),
        "",
        continuation.format_probe(report.plan),
    ]
    if report.prop:
        lines += ["", propagation.format_report(report.prop)]
    lines += [
        "",
        f"silent failures (stale + overclaimed + dropped work): {report.silent_failures}",
    ]
    return "\n".join(lines)
