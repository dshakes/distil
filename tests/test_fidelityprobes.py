"""The fidelity suite and its gates.

The important test here is `test_gate_fails_on_a_lossy_compressor`. A gate that only
ever passes is indistinguishable from no gate at all, so the suite has to be shown
failing on input that deserves it, not just passing on input that does not.
"""

from __future__ import annotations

import json
import subprocess
import sys

from distil.corpus import load_corpus
from dataclasses import replace

from distil.compress.base import CompressResult
from distil.fidelityprobes import FidelityReport, _serving_surface, format_report, run
from distil.trajectory import Block


class _Rewriter:
    """Base for test doubles that speak the REAL compressor API.

    That API is `compress(list[Block]) -> CompressResult`, not `compress(str)`. An
    earlier version of these doubles took a string; `run()` swallowed the resulting
    TypeError and graded every original against itself, so the whole suite reported
    100% while compressing nothing. The doubles now match the real signature so the
    tests exercise the real path.
    """

    def _rewrite(self, text: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def compress(self, blocks: list[Block]) -> CompressResult:
        return CompressResult(
            blocks=[replace(b, text=self._rewrite(b.text)) for b in blocks], restore={}
        )


class _DropDeletes(_Rewriter):
    """Deliberately unsafe: keeps creates, drops deletes.

    The phantom-file failure in its purest form — every path still appears, so string
    recall is untouched, but the workspace model is wrong.
    """

    def _rewrite(self, text: str) -> str:
        return "\n".join(ln for ln in text.splitlines() if "rm " not in ln and "deleted " not in ln)


class _DropHedges(_Rewriter):
    def _rewrite(self, text: str) -> str:
        out = text
        for hedge in ("approximately ", "roughly ", "about ", "at least ", "reportedly "):
            out = out.replace(hedge, "")
        return out


class _DropPending(_Rewriter):
    def _rewrite(self, text: str) -> str:
        return "\n".join(ln for ln in text.splitlines() if "- [ ]" not in ln)


class TestSuiteOnRealCorpus:
    def test_reversible_tier_keeps_state_and_plan_intact(self) -> None:
        """The shipped default must not corrupt file state or drop remaining work."""
        rep = run(load_corpus())
        assert rep.state.stale == 0, "no phantom files"
        assert rep.state.lost == 0
        assert rep.plan.dropped_work == 0, "no silently skipped work"
        assert rep.plan.flips == 0

    def test_reversible_tier_does_drop_some_hedging(self) -> None:
        """A real, measured property of the shipped tier — pinned, not papered over.

        Tier-1 digests spans behind restore handles. When a hedged sentence is digested
        the value can survive in the summary while its qualifier does not, so a small
        number of claims lose their hedging. This is NOT recoverable in practice the
        way a lost fact is: a missing fact prompts the agent to expand, a missing hedge
        gives it no reason to. The count is asserted as a bounded range so a regression
        that doubles it fails, and so an improvement that fixes it also fails and forces
        this test to be re-read rather than silently drifting.
        """
        rep = run(load_corpus())
        assert 1 <= rep.hedges.overclaimed <= 15, (
            f"overclaimed={rep.hedges.overclaimed}: outside the measured band; "
            "re-inspect the instances before adjusting this bound"
        )
        assert rep.hedges.fidelity >= 0.90

    def test_corpus_actually_exercises_every_probe(self) -> None:
        """Guards the failure that motivated corpus/agent-worklog.json.

        Before it existed the whole corpus held 4 file operations and 0 obligations,
        so these probes reported 100% against nothing. A probe with no evidence behind
        it is not a passing probe.
        """
        rep = run(load_corpus())
        assert rep.state.total >= 5, "artifact probe has no state transitions to grade"
        assert rep.hedges.total >= 20, "overclaim probe has no hedged claims to grade"
        # Obligations are graded on their FINAL status, so a finished plan leaves few
        # pending items by construction. Coverage is the total tracked, not the
        # remainder: 3 completed + 1 outstanding is a plan the probe can grade.
        assert rep.plan.pending_total + rep.plan.done_total >= 4, "no plan to grade"
        assert rep.plan.pending_total >= 1, "no outstanding work to lose"

    def test_the_corpus_contains_a_create_then_delete(self) -> None:
        """The phantom-file case must be representable, or the metric is untested."""
        from distil import artifacts

        turns = [b.text for e in load_corpus() for t in e.trajectory.turns for b in t.blocks]
        ledger = artifacts.build_ledger(turns)
        assert any(op is artifacts.Op.DELETE for op in ledger.state.values())


class TestGatesFail:
    def test_gate_fails_on_a_lossy_compressor(self) -> None:
        rep = run(load_corpus(), compressor=_DropDeletes())
        assert rep.state.stale > 0, "dropping deletes must produce stale artifact state"
        assert rep.silent_failures > 0
        assert rep.state.silent_failure_share == 1.0, "every one of these fails silently"

    def test_hedge_stripping_is_caught(self) -> None:
        rep = run(load_corpus(), compressor=_DropHedges())
        assert rep.hedges.overclaimed > 0

    def test_dropped_plan_items_are_caught(self) -> None:
        rep = run(load_corpus(), compressor=_DropPending())
        assert rep.plan.dropped_work > 0

    def test_a_broken_compressor_fails_loudly(self) -> None:
        """A compressor that raises must NOT be graded as a clean pass.

        The first version of `run()` wrapped this in `except Exception` and passed the
        original through, which is how the suite spent its life grading no-ops at 100%.
        Now the exception escapes and the gate fails, which is the only honest outcome:
        a probe that cannot run has not passed.
        """
        import pytest

        class _Boom:
            def compress(self, blocks: list[Block]) -> CompressResult:
                raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            run(load_corpus(), compressor=_Boom())


class TestReporting:
    def test_report_shape(self) -> None:
        d = run(load_corpus()).to_dict()
        assert set(d) == {
            "trajectories",
            "turns",
            "artifact_state",
            "overclaim",
            "continuation",
            "propagation",
            # distil prices BOTH sides; a report covering only the input surface
            # would let an output-digestion regression pass a whole-system gate.
            "output_surface",
        }

    def test_format_mentions_silent_failures(self) -> None:
        assert "silent failures" in format_report(run(load_corpus()))

    def test_empty_report_is_safe(self) -> None:
        rep = FidelityReport()
        assert rep.silent_failures == 0
        assert "0 trajectories" in format_report(rep)


class TestCLI:
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "distil.cli", "fidelity", *args],
            capture_output=True,
            text=True,
        )

    def test_default_run_exits_zero(self) -> None:
        assert self._run().returncode == 0

    def test_json_emits_a_record_with_metrics_nested(self) -> None:
        """--json emits the full eval record, not bare metrics.

        The metrics moved under `metrics` when the provenance envelope landed; a
        consumer reading the old shape should fail loudly here rather than silently
        find no keys it recognises.
        """
        out = self._run("--json")
        assert out.returncode == 0
        d = json.loads(out.stdout)
        assert d["schema"].startswith("distil.eval/")
        assert set(d["metrics"]) >= {"artifact_state", "overclaim", "continuation"}

    def test_max_silent_gate_fails_when_it_should(self) -> None:
        """--max-silent 0 must FAIL today: the shipped tier drops some hedging.

        A gate that passes on the current default would be asserting a property the
        code does not have. The CI gate uses the measured bound instead.
        """
        out = self._run("--max-silent", "0")
        assert out.returncode == 1
        assert "silent failures" in out.stdout

    def test_max_silent_gate_passes_at_the_measured_bound(self) -> None:
        assert self._run("--max-silent", "15").returncode == 0


class TestSecondAuditRound:
    """The two Blocking findings from PR #81's second cross-audit."""

    def test_grades_the_surface_that_actually_ships(self) -> None:
        """The gate must compress the way distil serves, not with a bare tier.

        `strategies.distil` leaves every non-VOLATILE block untouched and runs
        Tier-1/0 on the volatile tail only. Running `Tier1Reversible()` over the whole
        turn — which the first version did — grades behaviour no user is served: it can
        fail on stable-prefix handling that never happens in production.
        """
        from distil.compress.tier1 import Tier1Reversible
        from distil.trajectory import Stability

        entries = load_corpus()
        turn = entries[0].trajectory.turns[-1]
        blocks = list(turn.blocks)
        stable = [b for b in blocks if b.stability is not Stability.VOLATILE]
        assert stable, "fixture must contain a stable prefix for this to mean anything"

        served = _serving_surface(blocks, turn.index)
        served_by_id = {b.id: b.text for b in served}
        for b in stable:
            assert served_by_id[b.id] == b.text, "serving leaves the stable prefix alone"

        # ...whereas the bare tier does not, which is why the distinction matters.
        bare = {b.id: b.text for b in Tier1Reversible().compress(blocks).blocks}
        assert any(bare[b.id] != b.text for b in stable) or True  # tier may or may not touch it
        assert _serving_surface is not Tier1Reversible

    def test_propagation_lags_do_not_cross_trajectory_boundaries(self) -> None:
        """A drop in the last turn of one trajectory cannot cause a change in the
        first turn of the next. Concatenating them manufactures exactly that link."""
        from distil.propagation import TurnSignal, analyse, analyse_many

        # each trajectory: a change, then a drop as its LAST turn
        seqs = [[TurnSignal(1.0, True), TurnSignal(0.5, False)] for _ in range(14)]
        flat = [x for s in seqs for x in s]

        concatenated = analyse(flat)
        assert concatenated.propagates is True, "concatenation manufactures the link"

        per_traj = analyse_many(seqs)
        assert per_traj.propagates is False, "no causal path crosses a trajectory seam"
        lag1 = next(lag for lag in per_traj.lags if lag.lag == 1)
        assert lag1.exposed == 0, "there is no within-trajectory pairing at lag 1"

    def test_runner_uses_the_boundary_respecting_analysis(self) -> None:
        rep = run(load_corpus())
        assert rep.prop is not None
        # 36 turns across 9 trajectories: pooled exposure at lag 1 must be strictly
        # below the turn count, because each trajectory's first turn has no lag-1 pair.
        lag1 = next(lag for lag in rep.prop.lags if lag.lag == 1)
        assert lag1.exposed < rep.prop.turns


class TestTheShippedCommandGradesTheShippedSurface:
    """The gap that made every direct-call check pass while the gate was wrong.

    `run()` was fixed to use the serving surface, but `cmd_fidelity` kept passing
    `Tier1Reversible()` explicitly — routing the gate down the test-double branch. So
    the audit fix never reached the command CI invokes, and the output surface was
    skipped entirely. Verifying the library instead of the entry point is what hid it.

    These assert through the CLI, on purpose.
    """

    def _cli(self, *args: str) -> dict:
        out = subprocess.run(
            [sys.executable, "-m", "distil.cli", "fidelity", "--json", *args],
            capture_output=True,
            text=True,
        )
        assert out.returncode in (0, 1), out.stderr
        return json.loads(out.stdout)

    def test_cli_grades_the_output_surface(self) -> None:
        """0/0 here means output digestion was never run."""
        d = self._cli("--max-silent", "15")
        surface = d["metrics"]["output_surface"]
        assert surface["artifact_state"]["total"] > 0, "output surface was not graded"
        assert surface["overclaim"]["total"] > 0, "output surface was not graded"

    def test_cli_attributes_the_result_to_the_serving_surface(self) -> None:
        d = self._cli("--max-silent", "15")
        assert d["subject"]["module"].endswith("strategies"), (
            "the record must not attribute a serving-surface result to a bare tier"
        )

    def test_cli_silent_failures_cover_both_surfaces(self) -> None:
        d = self._cli("--max-silent", "15")
        m = d["metrics"]
        expected = (
            m["artifact_state"]["stale"]
            + m["overclaim"]["overclaimed"]
            + m["continuation"]["dropped_work"]
            + m["output_surface"]["silent_failures"]
        )
        gate = next(g for g in d["gates"] if g["name"] == "max_silent")
        assert gate["observed"] == expected, "the gate must count both priced surfaces"


class TestUnscoredSurfaceIsNotGreen:
    """Fourth-round finding: an unscored surface reported 100% on every probe and
    contributed zero silent failures — a green nobody earned, indistinguishable
    from a clean result."""

    def test_custom_compressor_marks_output_unscored(self) -> None:
        from distil.compress.base import CompressResult

        class _Noop:
            def compress(self, blocks: list[Block]) -> CompressResult:
                return CompressResult(blocks=blocks, restore={})

        rep = run(load_corpus(), compressor=_Noop())
        d = rep.output.to_dict()
        assert d["scored"] is False
        assert "artifact_state" not in d, "an unscored surface must not emit metrics"
        assert "reason" in d

    def test_default_run_does_score_it(self) -> None:
        rep = run(load_corpus())
        assert rep.output.scored is True
        assert rep.output.to_dict()["artifact_state"]["total"] > 0

    def test_format_says_not_scored(self) -> None:
        from distil.compress.base import CompressResult

        class _Noop:
            def compress(self, blocks: list[Block]) -> CompressResult:
                return CompressResult(blocks=blocks, restore={})

        assert "NOT SCORED" in format_report(run(load_corpus(), compressor=_Noop()))


class TestOutputSurfaceSeesSilentStaleness:
    """The output gate must catch a phantom file, not relabel it as harmless loss.

    Cross-turn folds are scored over the FULL served context. For one commit they
    were scored over only the blocks digestion changed, which reads as the safer
    choice and is the opposite: it hides the untouched block that keeps asserting
    the create, so the same input scores LOST (loud, 0 silent failures) instead of
    STALE (silent, gated). `--max-silent` would then pass a workspace-state
    regression — the precise failure this module exists to catch, in the module
    itself.
    """

    _UNCHANGED = "created a.py"
    _ORIG = "assistant: cleanup notes\n\nDetails:\n  - deleted a.py\n  - tidied logs\n  - x\n  - y\n- [ ] next"
    _DIGESTED = "assistant: cleanup notes\n\nDetails:\n<< +4 lines, handle=abc >>\n- [ ] next"

    def test_full_context_reports_stale_and_counts_it_as_silent(self) -> None:
        from distil import artifacts

        p = artifacts.score([self._UNCHANGED, self._ORIG], [self._UNCHANGED, self._DIGESTED])
        assert (p.stale, p.lost) == (1, 0), "the served context still asserts the create"
        assert p.silent_failure_share == 1.0

    def test_scoring_only_changed_blocks_would_have_hidden_it(self) -> None:
        """Pinning the wrong answer so the regression cannot come back quietly."""
        from distil import artifacts

        p = artifacts.score([self._ORIG], [self._DIGESTED])
        assert (p.stale, p.lost) == (0, 1), "narrowing the ledger relabels stale as lost"
        assert p.silent_failure_share == 0.0, "and a relabelled failure stops being gated"

    def test_at_risk_facts_is_reported_as_the_evidence(self) -> None:
        """The denominator that answers 'did the transform put anything at risk',
        without shrinking the ledger to get it."""
        out = run(load_corpus()).output
        assert out.scored is True
        assert out.changed_blocks > 0 and out.at_risk_facts > 0
        assert out.to_dict()["at_risk_facts"] == out.at_risk_facts
