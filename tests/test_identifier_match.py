"""Identifier matching for benchmarks whose golds are bare code names.

Two failures this guards, both measured on the real BFCL sample:

1. The prose matcher credited 11 of 85 golds by accident — `'a'` matched inside
   `"tool-schemas"`. A benchmark that passes on coincidence is worse than one that
   fails, because it reports confidence it has not earned.
2. Requiring a bare `"base"` read 2.9% on a payload where every name was plainly
   present. A tool schema is JSON nested inside a JSON payload, so the schema's own
   quotes arrive escaped as `\\"base\\"`. That number measured the matcher.
"""

from __future__ import annotations

from distil import retention as R


class TestIdentifierRule:
    def test_a_quoted_identifier_is_found(self) -> None:
        assert R._identifier_survives("base", '{"base": 10, "height": 5}')

    def test_escaped_quotes_still_match(self) -> None:
        """The case that made the strict rule read 2.9%: schema-in-payload escaping."""
        assert R._identifier_survives("base", '{"text": "{\\"base\\": 10}"}')

    def test_a_substring_is_not_a_match(self) -> None:
        """`unit` must not be credited by `units` — the exact overstatement measured."""
        assert not R._identifier_survives("unit", '{"units": "cm", "database": 1}')
        assert not R._identifier_survives("base", '{"database": true}')

    def test_an_unquoted_occurrence_is_not_a_match(self) -> None:
        assert not R._identifier_survives("base", "the base of a triangle")


class TestUnadjudicableGolds:
    """One-letter names are real BFCL arguments and unmatchable by any honest rule."""

    def test_short_golds_are_excluded_and_counted(self) -> None:
        case = _case(
            support=["a", "b", "calculate_area"], docs_text='{"a": 1, "calculate_area": 2}'
        )
        score = R._score_case(case, shape="json", match="identifier")
        assert score.support_unadjudicable == 2
        assert score.support_total == 1, "only the real identifier is graded"

    def test_they_are_not_counted_as_lost(self) -> None:
        """Excluding them must not silently become a failure — that would gate on a
        property no text rule can check."""
        case = _case(support=["a", "b", "c"], docs_text='{"z": 1}')
        score = R._score_case(case, shape="json", match="identifier")
        assert score.support_total == 0 and score.support_retained == 0
        assert score.support_unadjudicable == 3

    def test_the_count_reaches_the_report(self) -> None:
        """A shrunk denominator nobody reports is an inflated result."""
        cases = [_case(support=["a", "widget_id"], docs_text='{"widget_id": 1}')]
        data = R.score_dataset(cases, "bfcl").to_dict()
        assert data["distil"]["support_unadjudicable"] == 1


class TestPhraseModeUnchanged:
    def test_prose_benchmarks_still_use_the_phrase_rule(self) -> None:
        """The identifier rule must not leak into prose: a supporting SENTENCE is not a
        quoted token and would score zero under it."""
        case = _case(
            support=["The Eiffel Tower is in Paris."], docs_text="The Eiffel Tower is in Paris."
        )
        score = R._score_case(case, shape="json", match="phrase")
        assert score.support_total == 1, "the sentence was graded, not skipped"
        # Retained or recoverable both count as found; which one depends on whether the
        # tier digested the span. Lost is the only outcome that would mean the phrase
        # rule stopped seeing prose.
        assert score.support_retained + score.support_recoverable == 1

    def test_the_mode_comes_from_the_dataset_spec(self) -> None:
        """`distil suite` and `distil retention --dataset` must grade identically; a rule
        chosen by the caller means the number depends on which command you ran."""
        from distil.datasets import SPECS

        assert SPECS["bfcl"].match == "identifier"
        assert SPECS["hotpotqa"].match == "phrase"
        assert SPECS["squad"].match == "phrase"


def _case(*, support: list[str], docs_text: str):
    from distil.datasets import GroundTruthCase

    return GroundTruthCase(
        id="t",
        question="q",
        docs=[("tool-schemas", docs_text)],
        answer="",
        support=support,
        answerable=False,
        dataset="bfcl",
    )
