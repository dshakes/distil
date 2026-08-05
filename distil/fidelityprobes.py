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

The gates are asymmetric on purpose. `stale` artifacts, `overclaimed` and
`inverted` values and `dropped_work` are *silent* failures: the agent proceeds
confidently on a false belief and nothing in the transcript says so. Plain loss is
loud — the agent can see the gap — so it is reported but not gated here;
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
    # Why it was not scored. "Unscored" and "clean" must never look alike in the
    # payload, and a reader cannot act on the difference without being told which.
    reason: str = ""
    # How many blocks the surface's transform actually CHANGED. A surface that
    # changed nothing is comparing text to itself, and every probe on it returns
    # 100% by construction — see `scored`.
    changed_blocks: int = 0
    # Facts (artifact ops, obligations) that the transform actually removed from a
    # block. Cross-turn probes are scored over the whole served context, so a high
    # score can otherwise come entirely from blocks the transform never touched.
    # This is the honest denominator for "did the transform put anything at risk",
    # and it answers that WITHOUT shrinking the ledger — shrinking it flips the
    # dangerous stale case into the harmless lost one and lets the gate pass.
    at_risk_facts: int = 0

    @property
    def silent_failures(self) -> int:
        return (
            self.state.stale
            + self.hedges.overclaimed
            + self.hedges.inverted
            + self.plan.dropped_work
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.scored:
            return {"scored": False, "reason": self.reason}
        return {
            "scored": True,
            "changed_blocks": self.changed_blocks,
            "at_risk_facts": self.at_risk_facts,
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
            + self.hedges.inverted
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
        report.output.reason = (
            "a custom compressor was supplied; output digestion is the serving path's "
            "own transform and cannot be attributed to it"
        )
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
            changed = [(o, c) for o, c in zip(out_orig, out_small) if o != c]
            report.output.changed_blocks += len(changed)
            # Cross-turn folds are scored over the FULL served context, because that
            # is what the model is handed: unchanged blocks stay in the prompt and
            # keep asserting state.
            #
            # Scoring only the changed blocks — which is where this landed for one
            # commit — makes the probe blind to the failure it exists for. Take an
            # untouched block saying `created a.py` and a digested one whose elided
            # middle said `deleted a.py`. The served context now asserts the create
            # and not the delete, so the agent believes a live file exists: STALE,
            # silent, the dangerous case. Dropping the untouched block hides the
            # create, and the same input scores LOST — loud, harmless, and worth
            # zero silent failures, so `--max-silent` passes a silent regression.
            #
            #   full context  -> total=1 stale=1 lost=0  silent share 1.00
            #   changed only  -> total=1 stale=0 lost=1  silent share 0.00
            #
            # The concern that motivated the narrower version is real too — a surface
            # can otherwise report 100% from evidence the transform never touched —
            # but the answer to it is a denominator, not a smaller ledger. See
            # `at_risk_facts`. Overclaim is per-block and stays restricted to changed
            # blocks: identical pairs there are padding, not context.
            report.output.at_risk_facts += sum(
                len(set(artifacts.extract_ops(o)) - set(artifacts.extract_ops(c)))
                + len(
                    set(continuation.extract_obligations(o))
                    - set(continuation.extract_obligations(c))
                )
                for o, c in changed
            )
            report.output.state.add(artifacts.score(out_orig, out_small))
            report.output.plan.add(continuation.score(out_orig, out_small))
            for o, c in changed:
                report.output.hedges.add(overclaim.score(o, c))

    # A surface whose transform changed nothing was not measured, it was echoed. Every
    # probe on it compares text to itself and returns 100% by construction — the exact
    # "green against nothing" this module gates against elsewhere. On the bundled
    # corpus that is the live case, not a hypothetical: `digest_output_blocks` only
    # touches HISTORY blocks of at least six lines, and all 58 history blocks are
    # shorter, so 0 of 238 blocks changed while the surface reported 7/7 artifacts and
    # 179/179 hedges preserved. Those numbers were real arithmetic over untouched
    # input-side text, and they read exactly like a clean result.
    if compressor is None and not report.output.changed_blocks:
        report.output.scored = False
        report.output.reason = (
            "output digestion changed no blocks on this corpus — `digest_output_blocks` "
            "only digests HISTORY blocks of >= 6 lines and none qualify, so there is "
            "nothing to grade. Not a pass: a surface that was never exercised cannot "
            "report fidelity."
        )
    report.prop = propagation.analyse_many(per_traj)
    return report


def _surface_line(label: str, total: int, rate: float, kept: int, bad_label: str, bad: int) -> str:
    """One probe's line, refusing to print a percentage it did not measure.

    A probe with a zero denominator returns 1.0 by definition, so `0/0` rendered as
    `100.0%` — a perfect score on nothing, sitting in a column beside scores that were
    real. This is the same failure as an unscored SURFACE, one level down: the surface
    was exercised, but this particular property had nothing in it to grade.
    """
    if not total:
        return f"  {label:<22}  not measured  (0 graded — nothing of this kind survived to compare)"
    return f"  {label:<22} {rate:6.1%}  ({kept}/{total})   {bad_label} {bad}"


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
            "OUTPUT surface — NOT SCORED",
            f"  {o.reason}",
            "",
            f"silent failures, input surface only: {report.silent_failures}",
        ]
        return "\n".join(lines)
    lines += [
        "",
        "OUTPUT surface — past answers digested on re-entry (the other half of the bill)",
        f"  ({o.changed_blocks} blocks changed by digestion, "
        f"{o.at_risk_facts} facts removed from them — the graded evidence)"
        + (
            ""
            if o.at_risk_facts
            else "\n  NOTE: digestion removed no artifact or plan facts, so the state and plan"
            "\n  figures below come entirely from blocks it never touched — true, but not"
            "\n  evidence about the transform."
        ),
        _surface_line(
            "artifact state", o.state.total, o.state.fidelity, o.state.exact, "stale", o.state.stale
        ),
        _surface_line(
            "hedge fidelity",
            o.hedges.total,
            o.hedges.fidelity,
            o.hedges.preserved,
            "overclaimed",
            o.hedges.overclaimed,
        ),
        _surface_line(
            "pending-work recall",
            o.plan.pending_total,
            o.plan.pending_recall,
            o.plan.pending_kept,
            "dropped",
            o.plan.dropped_work,
        ),
        "",
        f"silent failures, BOTH surfaces: {report.silent_failures}"
        f"  (input {report.state.stale + report.hedges.overclaimed + report.hedges.inverted + report.plan.dropped_work},"
        f" output {o.silent_failures})",
    ]
    return "\n".join(lines)
