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
from .compress import strategies
from .output import digest_output_blocks


@dataclass
class SurfaceProbe:
    """The three state probes, scored against one compression surface."""

    state: artifacts.StateProbe = field(default_factory=artifacts.StateProbe)
    hedges: overclaim.OverclaimProbe = field(default_factory=overclaim.OverclaimProbe)
    plan: continuation.ContinuationProbe = field(default_factory=continuation.ContinuationProbe)
    # False when this surface was never scored. An unscored surface previously
    # reported 100% on every probe and contributed 0 silent failures — a green
    # nobody earned, and indistinguishable from a clean result. `scored` makes the
    # difference explicit in the payload rather than leaving it to be inferred.
    scored: bool = True

    @property
    def silent_failures(self) -> int:
        return self.state.stale + self.hedges.overclaimed + self.plan.dropped_work

    def to_dict(self) -> dict[str, Any]:
        if not self.scored:
            return {
                "scored": False,
                "reason": "a custom compressor was supplied; output digestion is the "
                "serving path's own transform and cannot be attributed to it",
            }
        return {
            "scored": True,
            "artifact_state": self.state.to_dict(),
            "overclaim": self.hedges.to_dict(),
            "continuation": self.plan.to_dict(),
            "silent_failures": self.silent_failures,
        }


@dataclass
class FidelityReport:
    trajectories: int = 0
    turns: int = 0
    state: artifacts.StateProbe = field(default_factory=artifacts.StateProbe)
    hedges: overclaim.OverclaimProbe = field(default_factory=overclaim.OverclaimProbe)
    plan: continuation.ContinuationProbe = field(default_factory=continuation.ContinuationProbe)
    prop: propagation.PropagationReport | None = None
    # distil compresses BOTH sides of the bill. `strategies.distil` shrinks what the
    # model READS (volatile tail); `output.digest_output_blocks` shrinks what its own
    # past ANSWERS cost when they re-enter as history. The probes above graded only
    # the first — and since output digestion targets assistant/history blocks, which
    # the serving strategy deliberately leaves alone, an entire priced surface was
    # going ungraded. Same three probes, second surface.
    output: SurfaceProbe = field(default_factory=SurfaceProbe)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories": self.trajectories,
            "turns": self.turns,
            "artifact_state": self.state.to_dict(),
            "overclaim": self.hedges.to_dict(),
            "continuation": self.plan.to_dict(),
            "propagation": self.prop.to_dict() if self.prop else None,
            "output_surface": self.output.to_dict(),
        }

    @property
    def silent_failures(self) -> int:
        """Every failure the agent cannot see, across BOTH priced surfaces.

        Counting only the input surface would let a regression in output digestion
        pass a gate named for the whole system.
        """
        return (
            self.state.stale
            + self.hedges.overclaimed
            + self.plan.dropped_work
            + self.output.silent_failures
        )


_DECISION = "DECISION:"


def _decisions(blocks: Iterable[str]) -> set[str]:
    """The decisions a turn's context asserts, per the corpus `DECISION:` convention.

    Same oracle the bench gate uses. It is synthetic — a marker in fixture text, not a
    model's actual next action — and that limit is reported as provenance rather than
    left for a reader to assume otherwise.
    """
    out: set[str] = set()
    for text in blocks:
        for line in (text or "").splitlines():
            head, sep, tail = line.partition(_DECISION)
            if sep:
                cleaned = tail.strip()
                if cleaned:
                    out.add(cleaned)
    return out


def _serving_surface(blocks: list[Any], turn: int) -> list[Any]:
    """Compress the way distil actually serves.

    `strategies.distil` leaves every non-VOLATILE block untouched and runs Tier-1/0
    on the volatile tail only. Calling `Tier1Reversible()` over the whole turn — as
    the first version of this runner did — grades a surface no user is ever served:
    it can fail on stable-prefix behaviour that never happens in production, or pass
    on volatile-tail behaviour it never actually exercised.

    A gate is only worth having if it grades the thing that ships.
    """
    return strategies.distil(blocks, turn)


def run(entries: Iterable[Any], *, compressor: Any = None) -> FidelityReport:
    """Score every turn of every trajectory, with real compression in the loop.

    `compressor` accepts a `compress(blocks) -> CompressResult` object for tests that
    need a specific failure mode. Default is the SERVING strategy, not a bare tier.
    """
    report = FidelityReport()
    if compressor is not None:
        report.output.scored = False
    # Propagation is analysed PER TRAJECTORY and aggregated. Accumulating every
    # trajectory into one sequence let a fidelity drop in the last turn of one
    # trajectory be scored as causing a decision change in the first turn of the
    # next — a lag across a boundary no causal path crosses. Cross-audit reproduced
    # a corpus of independent one-turn trajectories reporting `propagates=True`
    # purely from those seams.
    per_traj: list[list[propagation.TurnSignal]] = []

    for entry in entries:
        report.trajectories += 1
        signals: list[propagation.TurnSignal] = []
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
            if compressor is None:
                out_blocks = _serving_surface(list(turn.blocks), turn.index)
            else:
                out_blocks = compressor.compress(list(turn.blocks)).blocks
            compressed = [b.text for b in out_blocks]

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
                    # A REAL decision signal, from the same `DECISION:` oracle the bench
                    # gate uses — not a re-badged probe failure.
                    #
                    # The first version derived this from `stale or overclaimed or
                    # dropped_work`, which made the whole analysis circular: lift then
                    # correlated probe failures with probe failures, lag 0 was high by
                    # construction, and the gate could report "propagation" without a
                    # single decision changing. This repo already refuses that
                    # conflation elsewhere (`conformal.render_grader` insists the
                    # synthetic oracle is "NOT a model") and the same standard applies
                    # here.
                    decision_changed=_decisions(originals) != _decisions(compressed),
                )
            )
            traj_original.extend(originals)
            traj_compressed.extend(compressed)

        report.state.add(artifacts.score(traj_original, traj_compressed))
        report.plan.add(continuation.score(traj_original, traj_compressed))
        per_traj.append(signals)

        # --- the OTHER priced surface -------------------------------------------
        # Output-on-re-entry digestion, graded with the same three probes. Runs over
        # the whole trajectory because artifact and plan state are cross-turn.
        if compressor is None:
            out_orig: list[str] = []
            out_small: list[str] = []
            for turn in entry.trajectory.turns:
                blocks = list(turn.blocks)
                digested, _restore = digest_output_blocks(blocks)
                out_orig.extend(b.text for b in blocks)
                out_small.extend(b.text for b in digested)
            report.output.state.add(artifacts.score(out_orig, out_small))
            report.output.plan.add(continuation.score(out_orig, out_small))
            for o, c in zip(out_orig, out_small):
                report.output.hedges.add(overclaim.score(o, c))

    report.prop = propagation.analyse_many(per_traj)
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
    o = report.output
    if not o.scored:
        lines += [
            "",
            "OUTPUT surface — NOT SCORED (custom compressor supplied)",
            "",
            f"silent failures, input surface only: {report.silent_failures}",
        ]
        return "\n".join(lines)
    lines += [
        "",
        "OUTPUT surface — past answers digested on re-entry (the other half of the bill)",
        f"  artifact state          {o.state.fidelity:6.1%}  ({o.state.exact}/{o.state.total})"
        f"   stale {o.state.stale}",
        f"  hedge fidelity          {o.hedges.fidelity:6.1%}  ({o.hedges.preserved}/{o.hedges.total})"
        f"   overclaimed {o.hedges.overclaimed}",
        f"  pending-work recall     {o.plan.pending_recall:6.1%}"
        f"  ({o.plan.pending_kept}/{o.plan.pending_total})   dropped {o.plan.dropped_work}",
        "",
        f"silent failures, BOTH surfaces: {report.silent_failures}"
        f"  (input {report.state.stale + report.hedges.overclaimed + report.plan.dropped_work},"
        f" output {o.silent_failures})",
    ]
    return "\n".join(lines)
