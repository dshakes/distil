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
from distil.fidelityprobes import FidelityReport, format_report, run
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

    def test_json_is_parseable(self) -> None:
        out = self._run("--json")
        assert out.returncode == 0
        assert set(json.loads(out.stdout)) >= {"artifact_state", "overclaim", "continuation"}

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
