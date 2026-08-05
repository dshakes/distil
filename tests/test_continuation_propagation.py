"""Continuation and error-propagation probes."""

from __future__ import annotations

from distil.continuation import (
    ContinuationProbe,
    Status,
    extract_obligations,
    format_probe,
    score_texts,
)
from distil.propagation import TurnSignal, analyse, format_report


class TestObligationExtraction:
    def test_todo_is_pending(self) -> None:
        obs = extract_obligations("- TODO: wire the retry path into the client")
        assert {o.status for o in obs} == {Status.PENDING}

    def test_checkbox_states(self) -> None:
        obs = extract_obligations("- [x] added the parser\n- [ ] add the serialiser")
        assert {o.status for o in obs} == {Status.DONE, Status.PENDING}

    def test_stated_goal_is_pending(self) -> None:
        obs = extract_obligations("Your task is to migrate the billing module to v2")
        assert any(o.status is Status.PENDING for o in obs)

    def test_short_items_ignored(self) -> None:
        assert extract_obligations("TODO: x") == []

    def test_reflow_does_not_change_identity(self) -> None:
        """Compression rewraps lists; that must not read as lost work."""
        a = extract_obligations("- TODO: wire the retry path into the client module")
        b = extract_obligations("- TODO: wire the retry path into the\n  client module today")
        assert {o.key for o in a} == {o.key for o in b}


class TestContinuationScoring:
    def test_all_work_preserved(self) -> None:
        t = "- [ ] add the serialiser\n- [x] added the parser"
        p = score_texts(t, t)
        assert p.pending_recall == 1.0 and p.done_recall == 1.0 and p.flips == 0

    def test_dropped_pending_work_is_the_silent_failure(self) -> None:
        original = "- [x] added the parser\n- [ ] add the serialiser"
        compressed = "- [x] added the parser"
        p = score_texts(original, compressed)
        assert p.dropped_work == 1
        assert p.pending_recall == 0.0
        assert p.done_recall == 1.0, "keeping finished work while dropping remaining work"

    def test_dropped_done_work_is_cheap(self) -> None:
        p = score_texts(
            "- [x] added the parser\n- [ ] add the serialiser", "- [ ] add the serialiser"
        )
        assert p.pending_recall == 1.0 and p.dropped_work == 0

    def test_status_flip_counted_separately(self) -> None:
        p = score_texts("- [ ] add the serialiser", "- [x] add the serialiser")
        assert p.flips == 1
        assert p.pending_kept == 0, "a flipped item is not a kept item"

    def test_vacuous_when_no_obligations(self) -> None:
        p = score_texts("just prose", "just prose")
        assert p.pending_recall == 1.0 and p.pending_total == 0

    def test_to_dict_content_free(self) -> None:
        d = score_texts("- [ ] deploy secretservice to prod", "").to_dict()
        assert "secretservice" not in str(d)

    def test_format_names_the_consequence(self) -> None:
        out = format_probe(ContinuationProbe(pending_total=2, pending_kept=1))
        assert "without doing it" in out

    def test_format_nothing_to_grade(self) -> None:
        assert "nothing to grade" in format_probe(ContinuationProbe())


def _signals(n: int, *, drops: set[int], changes: set[int]) -> list[TurnSignal]:
    return [TurnSignal(0.5 if i in drops else 1.0, i in changes) for i in range(n)]


class TestPropagation:
    def test_flat_profile_when_errors_are_local(self) -> None:
        # Drops and changes co-occur, and the drops are spaced further apart than
        # max_lag so nothing aliases. This is the "errors stay local" shape.
        at = {0, 7, 16, 27}
        rep = analyse(_signals(35, drops=at, changes=at))
        assert rep.propagates is False

    def test_periodic_workload_aliases_and_the_module_admits_it(self) -> None:
        """A documented limit, pinned so it cannot regress into a silent claim.

        Period-3 drops with period-3 changes and NO causal link still light up lag 3.
        The verdict is elevated; the docstring tells the reader why not to trust it.
        """
        # 12 repeats so the aliased lag clears the minimum-exposure floor; with
        # fewer, the module correctly refuses to assert anything either way.
        sig = [TurnSignal(1.0, False), TurnSignal(0.5, True), TurnSignal(1.0, False)] * 12
        assert analyse(sig).propagates is True, "aliasing is real; do not pretend otherwise"
        import distil.propagation as mod

        assert "alias" in (mod.__doc__ or "").lower(), "the limit must be documented"

    def test_detects_delayed_damage(self) -> None:
        # Every drop is followed two turns later by a decision change.
        sig: list[TurnSignal] = []
        for _ in range(12):
            sig.append(TurnSignal(0.5, False))
            sig.append(TurnSignal(1.0, False))
            sig.append(TurnSignal(1.0, True))
        rep = analyse(sig)
        assert rep.propagates is True
        assert rep.worst is not None and rep.worst.lag == 2

    def test_lag_zero_alone_is_not_propagation(self) -> None:
        """Same-turn co-incidence must not be reported as downstream damage."""
        at = {0, 7, 16, 27}
        rep = analyse(_signals(35, drops=at, changes=at))
        lag0 = next(lag for lag in rep.lags if lag.lag == 0)
        assert lag0.lift > 1.25, "lag 0 is strongly associated by construction"
        assert rep.propagates is False, "yet lag 0 alone must not trip the verdict"

    def test_requires_minimum_exposure(self) -> None:
        # One lucky coincidence must not trip the verdict.
        sig = [TurnSignal(1.0, False)] * 8 + [TurnSignal(0.5, False), TurnSignal(1.0, True)]
        assert analyse(sig).propagates is False

    def test_empty_input(self) -> None:
        rep = analyse([])
        assert rep.turns == 0 and rep.worst is None
        assert "no turns" in format_report(rep)

    def test_report_discloses_the_causal_limit(self) -> None:
        sig = [TurnSignal(0.5, False), TurnSignal(1.0, True)] * 5
        out = format_report(analyse(sig))
        assert "association, not causation" in out, "the metric must state its own limit"

    def test_to_dict_shape(self) -> None:
        d = analyse([TurnSignal(1.0, False)] * 3).to_dict()
        assert set(d) == {"turns", "base_rate", "propagates", "lags"}
        assert isinstance(d["lags"], list)

    def test_zero_base_rate_does_not_divide_by_zero(self) -> None:
        rep = analyse([TurnSignal(0.5, False)] * 5)
        assert all(lag.lift == 0.0 for lag in rep.lags)
        assert rep.propagates is False


class TestPropagationInputDiscipline:
    """The Blocking finding from PR #81's cross-audit.

    `decision_changed` must come from a decision oracle. Deriving it from the fidelity
    probes makes the analysis circular: lift correlates probe failures with themselves,
    lag 0 is elevated by construction, and the gate can fire without a single decision
    having changed.
    """

    def test_zero_base_rate_says_nothing_was_tested(self) -> None:
        """Vacuous truth must not be dressed as an earned result."""
        rep = analyse([TurnSignal(0.5, False)] * 12)
        out = format_report(rep)
        assert "nothing to propagate, and nothing tested" in out
        assert "no measurable propagation" not in out, "that would overclaim the result"

    def test_the_contract_is_documented_on_the_input(self) -> None:
        # Whitespace-normalised: the docstring is reflowed by the formatter, and a test
        # that breaks on rewrapping is testing the formatter, not the contract.
        doc = " ".join((TurnSignal.__doc__ or "").split())
        assert "MUST come from an independent decision oracle" in doc
        assert "not be derived from the fidelity probes" in doc.lower()


class TestDecisionOracle:
    def test_extracts_decision_markers(self) -> None:
        from distil.fidelityprobes import _decisions

        assert _decisions(["noise\nDECISION: stop and report\nmore"]) == {"stop and report"}

    def test_change_is_detected(self) -> None:
        from distil.fidelityprobes import _decisions

        assert _decisions(["DECISION: a"]) != _decisions(["DECISION: b"])

    def test_no_marker_is_empty_not_an_error(self) -> None:
        from distil.fidelityprobes import _decisions

        assert _decisions(["no marker here", ""]) == set()

    def test_signal_is_not_derived_from_probes(self) -> None:
        """A turn with probe damage but an unchanged decision must report no change."""
        from distil.corpus import load_corpus
        from distil.fidelityprobes import run

        rep = run(load_corpus())
        assert rep.hedges.overclaimed > 0, "there IS probe damage on the corpus"
        assert rep.prop is not None
        assert rep.prop.base_rate == 0.0, (
            "yet no decision changed — proving the signal is independent of the probes"
        )


class TestThirdAuditRound:
    def test_same_block_checklist_update_is_deterministic(self) -> None:
        """`extract_obligations` returned a set, so `_fold` had no 'later'.

        A block holding both states of one item folded to whichever status hash
        ordering happened to yield last — `pending` in one checkout, `done` in
        another. A metric whose value depends on hash ordering is not a metric.
        """
        from distil.continuation import Status, _fold, extract_obligations

        blk = "- [ ] wire the endpoint handler\n- [x] wire the endpoint handler"
        assert isinstance(extract_obligations(blk), list), "must preserve document order"
        assert _fold([blk]) == {"wire the endpoint handler": Status.DONE}

    def test_reverse_order_gives_the_other_answer(self) -> None:
        """Order must actually drive the result, not merely be preserved."""
        from distil.continuation import Status, _fold

        blk = "- [x] wire the endpoint handler\n- [ ] wire the endpoint handler"
        assert _fold([blk]) == {"wire the endpoint handler": Status.PENDING}

    def test_extraction_is_stable_across_runs(self) -> None:
        from distil.continuation import extract_obligations

        blk = "- [ ] alpha item here\n- [x] beta item here\n- [ ] gamma item here"
        first = extract_obligations(blk)
        for _ in range(20):
            assert extract_obligations(blk) == first


class TestGoalMarkers:
    """Round-6 Blocking: `Objective:` never matched.

    The trailing `\\b` landed after the colon, where the next character is a space —
    no boundary exists there. A dropped objective left `dropped_work` at zero, so the
    gate silently undercounted pending work. Same misplaced-`\\b` bug already fixed
    for `[x]` checkboxes in this file.
    """

    def test_objective_marker_is_extracted(self) -> None:
        obs = extract_obligations("Objective: add retry wrapper now")
        assert [o.status for o in obs] == [Status.PENDING]

    def test_every_documented_goal_marker_works(self) -> None:
        for text in (
            "Your task is to add the retry wrapper",
            "The goal is to migrate the billing module",
            "Objective: ship the parser rewrite",
            "You must remove the legacy client",
        ):
            assert extract_obligations(text), f"no obligation from: {text!r}"

    def test_dropping_an_objective_is_counted(self) -> None:
        p = score_texts("Objective: add retry wrapper now", "unrelated prose")
        assert p.dropped_work == 1, "a dropped objective must register as lost work"
