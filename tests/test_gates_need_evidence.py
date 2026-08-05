"""No gate may pass without evidence.

This rule has now had to be applied in four separate places, each found by a
different audit round rather than by me:

  1. `EvalRecord.passed` — an empty gate list is not a pass (built in from the start)
  2. `shadow-stats --max-change-rate` — passed on a fresh ledger with 0 samples
  3. the output surface — reported 100% on three probes it had never scored
  4. `fidelity --no-propagation` — passed while the report said "nothing tested"

Four instances of one class, patched one at a time, is the pattern that kept this
PR in audit for seven rounds. This file sweeps the class instead: every gate the
CLI can record is exercised against an evidence-free input, and must NOT come back
green.

The failure being prevented is specific and bad: a CI job asks to be gated, the
gate has nothing to measure, and it answers "fine". That is worse than no gate,
because someone is now relying on it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile


def _cli(*args: str, home: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if home:
        env["DISTIL_HOME"] = home
    return subprocess.run(
        [sys.executable, "-m", "distil.cli", *args], capture_output=True, text=True, env=env
    )


class TestEveryGateRefusesToCertifyNothing:
    def test_shadow_change_rate_with_no_samples(self) -> None:
        out = _cli("shadow-stats", "--record", "--max-change-rate", "0.01", home=tempfile.mkdtemp())
        gate = next(g for g in json.loads(out.stdout)["gates"] if g["name"] == "max_change_rate")
        assert gate["passed"] is False
        assert out.returncode == 1
        assert "INCONCLUSIVE" in gate["rationale"]

    def test_propagation_with_no_decision_changes(self) -> None:
        """The bundled corpus has a zero decision-change rate, so the lag profile has
        no events. The report already says "nothing tested"; the gate must agree."""
        out = _cli("fidelity", "--json", "--no-propagation")
        d = json.loads(out.stdout)
        assert d["metrics"]["propagation"]["base_rate"] == 0.0, "fixture assumption"
        gate = next(g for g in d["gates"] if g["name"] == "no_propagation")
        assert gate["passed"] is False, "a gate must not certify what was never measured"
        assert out.returncode == 1
        assert "INCONCLUSIVE" in gate["rationale"]

    def test_a_record_with_no_gates_is_not_passed(self) -> None:
        from distil.evalrecord import EvalRecord

        assert EvalRecord().passed is False

    def test_unscored_surface_emits_no_metrics(self) -> None:
        from distil.compress.base import CompressResult
        from distil.corpus import load_corpus
        from distil.fidelityprobes import run

        class _Noop:
            def compress(self, blocks):
                return CompressResult(blocks=blocks, restore={})

        d = run(load_corpus(), compressor=_Noop()).output.to_dict()
        assert d["scored"] is False and "artifact_state" not in d


class TestGatesStillPassOnRealEvidence:
    """The counterpart: a gate that can never pass is equally useless."""

    def test_max_silent_passes_at_the_measured_bound(self) -> None:
        assert _cli("fidelity", "--max-silent", "15").returncode == 0

    def test_max_silent_fails_below_it(self) -> None:
        assert _cli("fidelity", "--max-silent", "0").returncode == 1

    def test_no_gate_requested_exits_zero(self) -> None:
        assert _cli("fidelity").returncode == 0
        assert _cli("shadow-stats", "--record", home=tempfile.mkdtemp()).returncode == 0


class TestTheRuleIsDocumentedWhereItIsEnforced:
    """A rule applied in four places and written down in none is a rule that gets
    re-broken in a fifth."""

    def test_evalrecord_states_it(self) -> None:
        from distil.evalrecord import EvalRecord

        assert "empty gate list is not a pass" in (EvalRecord.passed.__doc__ or "")


class TestEveryFailureClassReachesAGate:
    """A failure class no gate counts is a failure class nobody is protected from.

    `inverted` was added as a distinct outcome (a floor read as a ceiling) but the
    silent-failure total summed only stale + overclaimed + dropped_work, so an
    inverted bound could not move `--max-silent` at all. That is the same shape as
    the four vacuous gates above: the number exists, the gate ignores it, and the
    run comes back green.
    """

    def test_an_inverted_bound_counts_as_a_silent_failure(self) -> None:
        from distil.fidelityprobes import SurfaceProbe
        from distil.overclaim import OverclaimProbe

        s = SurfaceProbe(hedges=OverclaimProbe(total=1, inverted=1))
        assert s.silent_failures == 1, "an inverted bound must reach the gate"

    def test_the_report_total_counts_it_on_both_surfaces(self) -> None:
        from distil.fidelityprobes import FidelityReport, SurfaceProbe
        from distil.overclaim import OverclaimProbe

        r = FidelityReport(
            hedges=OverclaimProbe(total=1, inverted=1),
            output=SurfaceProbe(hedges=OverclaimProbe(total=1, inverted=1)),
        )
        assert r.silent_failures == 2


class TestAskingForAGateRunsAGate:
    """`shadow-stats --max-change-rate 0.01` printed the table and exited 0.

    The gate was evaluated only inside the `--record` branch, so a CI job passing the
    threshold alone believed it was gated and was not. Same vacuous-gate failure as
    the four above, arriving through the argument parser rather than through empty
    evidence.
    """

    def test_the_threshold_alone_still_gates(self) -> None:
        out = _cli("shadow-stats", "--max-change-rate", "0.01", home=tempfile.mkdtemp())
        assert out.returncode == 1, "a requested gate must run"
        gate = next(g for g in json.loads(out.stdout)["gates"] if g["name"] == "max_change_rate")
        assert gate["passed"] is False and "INCONCLUSIVE" in gate["rationale"]

    def test_no_threshold_still_exits_zero(self) -> None:
        assert _cli("shadow-stats", home=tempfile.mkdtemp()).returncode == 0


class TestTheOutputSurfaceIsActuallyExercised:
    """The output surface reported 100% on blocks digestion never touched.

    `digest_output_blocks` only digests HISTORY blocks of >= 6 lines. Every history
    block in the corpus was 5 lines, so 0 of 238 blocks changed — and the surface
    still reported `scored: true`, 7/7 artifacts and 179/179 hedges preserved. Those
    were real percentages computed over untouched input-side text: a guaranteed 100%
    that reads exactly like a clean result.

    Two things have to hold, and a test for either alone lets the other rot:
    the transform must change something, and each probe must have something of its
    own kind to grade.
    """

    def test_digestion_changes_blocks_on_the_bundled_corpus(self) -> None:
        from distil.corpus import load_corpus
        from distil.output import digest_output_blocks

        changed = 0
        for entry in load_corpus():
            for turn in entry.trajectory.turns:
                blocks = list(turn.blocks)
                digested, _ = digest_output_blocks(blocks)
                changed += sum(1 for a, b in zip(blocks, digested) if a.text != b.text)
        assert changed > 0, "no block is digestible: the output surface grades nothing"

    def test_every_output_probe_has_a_real_denominator(self) -> None:
        from distil.corpus import load_corpus
        from distil.fidelityprobes import run

        out = run(load_corpus()).output
        assert out.scored is True and out.changed_blocks > 0
        # A zero denominator makes the probe return 1.0 by definition.
        assert out.state.total > 0, "artifact probe graded nothing on the output surface"
        assert out.hedges.total > 0, "hedge probe graded nothing on the output surface"
        assert out.plan.pending_total > 0, "plan probe graded nothing on the output surface"

    def test_an_untouched_surface_is_reported_unscored(self) -> None:
        """The guard itself: if digestion stops firing, say so rather than score it."""
        from distil.fidelityprobes import SurfaceProbe

        s = SurfaceProbe(scored=False, reason="nothing digestible")
        assert s.to_dict() == {"scored": False, "reason": "nothing digestible"}
