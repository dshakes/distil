"""Fact-level retention probe — the metric must be able to FAIL.

A recall gate that can only ever report 100% is theater. These tests pin the three
things that make the number real: it detects genuine loss, it does not credit a
handle whose restore bytes lack the fact (the block-scoped, *verified* recoverability
that distil claims over marker-presence heuristics), and it tolerates the format
conversions Tier-0 legitimately performs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from distil.retention import (
    RetentionReport,
    extract_targets,
    format_report,
    probe_block,
    run,
)
from distil.corpus import load_corpus
from distil.trajectory import Block, Kind, Stability


def _vol(text: str, bid: str = "b1") -> Block:
    return Block(id=bid, kind=Kind.TOOL_OUTPUT, text=text, stability=Stability.VOLATILE)


ORIGINAL = (
    'status: ok\n"latency_ms": 1234\n'
    "/var/log/app/worker.log\n"
    "ERROR: connection refused to db-primary\n"
)


def test_verbatim_is_fully_retained() -> None:
    probe = probe_block(_vol(ORIGINAL), ORIGINAL, {})
    for dim, tally in probe.dims.items():
        assert tally.total > 0, dim
        assert tally.lost == 0
        assert tally.visible_recall == 1.0


def test_truncation_loses_facts() -> None:
    """The gate's teeth: lossy compression with no restore table must report loss."""
    probe = probe_block(_vol(ORIGINAL), "status: ok\n", {})
    overall_lost = sum(t.lost for t in probe.dims.values())
    assert overall_lost > 0
    assert probe.dims["errors"].lost == 1
    assert probe.dims["errors"].recall == 0.0


def test_handle_without_the_fact_is_lost_not_recoverable() -> None:
    """Recoverability is verified against the restore BYTES, never inferred from a
    marker. A handle whose original lacks the fact must not launder it."""
    compressed = "<< +3 lines, handle=deadbeef >>"
    probe = probe_block(_vol(ORIGINAL), compressed, {"deadbeef": "unrelated content"})
    assert probe.dims["errors"].recoverable == 0
    assert probe.dims["errors"].lost == 1

    # Same marker, restore bytes that DO contain the fact -> recoverable, not lost.
    probe = probe_block(_vol(ORIGINAL), compressed, {"deadbeef": ORIGINAL})
    assert probe.dims["errors"].recoverable == 1
    assert probe.dims["errors"].lost == 0
    assert probe.dims["errors"].recall == 1.0
    assert probe.dims["errors"].visible_recall == 0.0


def test_foreign_handle_is_not_credited() -> None:
    """Recoverability is block-scoped: a handle this block does not carry cannot
    make its missing facts recoverable, even if the restore entry exists."""
    probe = probe_block(_vol(ORIGINAL), "status: ok\n", {"deadbeef": ORIGINAL})
    assert sum(t.recoverable for t in probe.dims.values()) == 0
    assert probe.dims["errors"].lost == 1


def test_tier0_reshape_counts_as_retained() -> None:
    """A JSON->table conversion keeps key and value in different cells; both present
    means the fact survived, so it must not be scored as loss."""
    probe = probe_block(_vol('{"latency_ms": 1234}'), "«cols=latency_ms«\n1234", {})
    assert probe.dims["numerics"].retained == 1


def test_numeric_needs_key_and_value() -> None:
    """A stray matching integer must not fake a hit when its key is gone."""
    probe = probe_block(_vol('"latency_ms": 1234'), "unrelated 1234", {})
    assert probe.dims["numerics"].retained == 0
    assert probe.dims["numerics"].lost == 1


def test_bare_numbers_are_not_probed() -> None:
    """Numbers with no key context are unverifiable noise, not facts."""
    assert extract_targets("1234 5678 9012")["numerics"] == set()


def test_empty_report_is_neutral() -> None:
    empty = RetentionReport()
    assert empty.overall().recall == 1.0
    assert empty.lost_facts == 0
    assert "0 compressed blocks probed" in format_report(empty)


def test_error_survives_losing_its_json_key() -> None:
    """'"msg": "ERROR: refused"' reshaped to a bare cell still carries the substance."""
    probe = probe_block(_vol('"msg": "ERROR: connection refused"'), "ERROR connection refused", {})
    assert probe.dims["errors"].retained == 1


def test_block_id_restore_key_is_credited() -> None:
    """Tier-0 keys its restore table by block id, not by an in-text handle."""
    probe = probe_block(_vol(ORIGINAL, bid="blk7"), "status: ok\n", {"blk7": ORIGINAL})
    assert probe.dims["errors"].recoverable == 1
    assert probe.dims["errors"].lost == 0


def test_turn_without_volatile_blocks_is_skipped() -> None:
    from distil.corpus import CorpusEntry
    from distil.retention import probe_trajectory
    from distil.trajectory import Trajectory, Turn

    stable = Block(id="s", kind=Kind.SYSTEM, text="fixed", stability=Stability.STABLE)
    traj = Trajectory(id="t", model="claude-sonnet-4-5", turns=[Turn(index=0, blocks=[stable])])
    report = probe_trajectory(CorpusEntry("f.json", "test", "t", traj))
    assert report.probes == []


class _Case:
    """A GroundTruthCase stand-in (no network, no dataset dependency)."""

    def __init__(
        self,
        docs: list[tuple[str, str]],
        answer: str,
        support: list[str] | None = None,
        answerable: bool = True,
    ) -> None:
        self.id = "c1"
        self.question = "q?"
        self.docs = docs
        self.answer = answer
        self.support = support or []
        self.answerable = answerable
        self.dataset = "synthetic"


def _rows(n: int, marker: str = "") -> list[tuple[str, str]]:
    return [(f"doc{i}", f"Paragraph {i} filler sentence about things. {marker}") for i in range(n)]


def test_unanswerable_is_excluded_not_counted_as_a_pass() -> None:
    from distil.retention import score_dataset

    rep = score_dataset([_Case(_rows(3), "", answerable=False)], "t")
    assert rep.ungraded("unanswerable") == 1
    assert rep._answer_recall(rep.scores, with_recovery=True) == (1.0, 0)  # 0 graded


def test_abstractive_answer_is_excluded_not_scored_as_loss() -> None:
    """HotpotQA's yes/no answers are never spans in the passage. Grading them would
    report the dataset's answer FORMAT as compression loss — the defect this pins."""
    from distil.retention import score_dataset

    rep = score_dataset([_Case(_rows(3), "yes")], "t")
    assert rep.ungraded("abstractive") == 1
    assert rep.scores[0].answer_retained is None


def test_answer_present_in_context_is_graded() -> None:
    from distil.retention import score_dataset

    case = _Case([("d", "The director was Tim Burton.")], "Tim Burton")
    # Graded (not excluded as abstractive), and reachable either way.
    rep = score_dataset([case], "t")
    assert rep.ungraded("abstractive") == 0
    assert rep.scores[0].answer_retained is not None
    assert rep._answer_recall(rep.scores, with_recovery=True) == (1.0, 1)

    # Under `prose` there is nothing to compress, so it must be VISIBLY retained.
    prose = score_dataset([case], "t", shape="prose")
    assert prose.scores[0].answer_retained is True


def test_support_is_gated_on_presence_in_the_original() -> None:
    """A gold sentence that was never in the context cannot be 'lost' by compression."""
    from distil.retention import score_dataset

    case = _Case([("d", "Real sentence here.")], "Real", support=["Absent sentence."])
    rep = score_dataset([case], "t")
    assert rep.scores[0].support_total == 0


def test_unengaged_compression_is_flagged_not_celebrated() -> None:
    """Short prose does not compress; a 100% recall on an identity transform must warn."""
    from distil.retention import format_dataset_report, score_dataset

    rep = score_dataset([_Case([("d", "Tiny prose.")], "Tiny")], "t", shape="prose")
    assert not rep.engaged
    assert "WARNING: compression barely engaged" in format_dataset_report(rep)


def test_json_shape_engages_compression() -> None:
    """The production shape (a retrieval tool's JSON array) is what distil compresses."""
    from distil.retention import score_dataset

    rep = score_dataset([_Case(_rows(12), "filler")], "t", shape="json")
    assert rep.engaged
    assert rep.savings > 0.0


def test_baseline_savings_match_distil() -> None:
    """The comparison is only fair at equal cost — pin that the baseline is not cheaper."""
    from distil.retention import DatasetReport, score_dataset

    rep = score_dataset([_Case(_rows(12), "filler")], "t")
    distil_savings = DatasetReport._savings(rep.scores)
    baseline_savings = DatasetReport._savings(rep.baseline)
    assert baseline_savings >= distil_savings - 0.05


def test_baseline_has_no_recovery_path() -> None:
    """Truncation is lossy by construction: nothing may be scored as recoverable."""
    from distil.retention import score_dataset

    rep = score_dataset([_Case(_rows(12), "filler")], "t")
    assert all(not s.answer_recoverable for s in rep.baseline)
    assert all(s.support_recoverable == 0 for s in rep.baseline)


def test_support_recall_reported_when_gold_sentences_survive() -> None:
    from distil.retention import format_dataset_report, score_dataset

    gold = "It was directed by Tim Burton."
    case = _Case([("d", f"Ed Wood is a film. {gold}")], "Tim Burton", support=[gold])
    rep = score_dataset([case], "t", shape="prose")
    assert rep.scores[0].support_total == 1
    assert rep.scores[0].support_retained == 1
    assert rep._support_recall(rep.scores, with_recovery=False) == (1.0, 1)
    assert "support recall" in format_dataset_report(rep)


def test_support_behind_a_handle_is_recoverable_not_lost() -> None:
    """Gold sentences digested into the restore table count as recoverable — the
    distinction a lossy compressor cannot make."""
    from distil.retention import score_dataset

    gold = "The director was Tim Burton."
    case = _Case([("d", gold * 40)], "Tim Burton", support=[gold])
    rep = score_dataset([case], "t")
    score = rep.scores[0]
    assert score.support_total == 1
    assert score.support_retained == 0  # digested out of the visible text
    assert score.support_recoverable == 1  # but proven present in the restore bytes
    assert rep._support_recall(rep.scores, with_recovery=False) == (0.0, 1)
    assert rep._support_recall(rep.scores, with_recovery=True) == (1.0, 1)


def test_exclusions_appear_in_the_rendered_report() -> None:
    from distil.retention import format_dataset_report, score_dataset

    rep = score_dataset(
        [_Case(_rows(3), "", answerable=False), _Case(_rows(3), "yes")], "t", shape="prose"
    )
    rendered = format_dataset_report(rep)
    assert "marked unanswerable" in rendered
    assert "not a span in the passage" in rendered


def test_low_visible_recall_is_called_out() -> None:
    """100% recall with everything behind a handle is lossless but costs round trips,
    and the report must say so rather than only showing the reassuring number."""
    from distil.retention import format_dataset_report, score_dataset

    rep = score_dataset([_Case([("d", "The director was Tim Burton. " * 40)], "Tim Burton")], "t")
    visible, graded = rep._answer_recall(rep.scores, with_recovery=False)
    true, _ = rep._answer_recall(rep.scores, with_recovery=True)
    assert graded == 1 and visible == 0.0 and true == 1.0
    assert "NOTE: only 0.0% of answers are visible" in format_dataset_report(rep)


def test_bad_shape_rejected() -> None:
    from distil.retention import score_dataset

    with pytest.raises(ValueError, match="shape must be"):
        score_dataset([_Case(_rows(2), "x")], "t", shape="yaml")


def test_dataset_report_serializes_both_arms() -> None:
    from distil.retention import score_dataset

    payload = score_dataset([_Case(_rows(12), "filler")], "hotpotqa").to_dict()
    assert payload["dataset"] == "hotpotqa" and payload["shape"] == "json"
    assert payload["compression_engaged"] is True
    assert set(payload["distil"]) == set(payload["baseline"]) - {"label"}


def _probe(domain: str, total: int, retained: int, recoverable: int) -> object:
    from distil.retention import BlockProbe, Tally

    return BlockProbe(
        block_id=f"{domain}-b",
        savings=0.5,
        dims={"numerics": Tally(total=total, retained=retained, recoverable=recoverable)},
        domain=domain,
    )


def test_macro_average_is_not_swung_by_one_dominant_domain() -> None:
    """The reason this metric exists. A fact-weighted mean is set by whichever domain
    carries the most probes — adding one HTML fixture moved it 9.8% -> 62.6% without
    anything about the compressor changing. Macro counts each domain once."""
    from distil.retention import RetentionReport

    report = RetentionReport(
        probes=[
            _probe("a", total=10, retained=10, recoverable=0),  # gap 0%
            _probe("b", total=10, retained=10, recoverable=0),  # gap 0%
            _probe("huge", total=1000, retained=0, recoverable=1000),  # gap 100%
        ]
    )
    overall = report.overall()
    micro_gap = overall.recall - overall.visible_recall
    macro = report.macro()

    assert macro.domains == 3
    assert macro.gap == pytest.approx(1 / 3, abs=1e-6)  # (0 + 0 + 1) / 3
    assert micro_gap > 0.97  # the 1000-fact domain sets the fact-weighted number
    assert macro.gap < micro_gap  # and macro resists it


def test_macro_ignores_empty_domains() -> None:
    from distil.retention import RetentionReport

    report = RetentionReport(
        probes=[
            _probe("real", total=4, retained=2, recoverable=2),
            _probe("empty", total=0, retained=0, recoverable=0),
        ]
    )
    assert report.macro().domains == 1
    assert report.macro().visible_recall == pytest.approx(0.5)


def test_macro_on_an_empty_report_is_neutral() -> None:
    from distil.retention import RetentionReport

    macro = RetentionReport().macro()
    assert macro.domains == 0 and macro.gap == 0.0 and macro.recall == 1.0


def test_report_leads_with_macro_and_labels_micro() -> None:
    """A reader must not be able to quote the swingable number by accident."""
    from distil.retention import RetentionReport, format_report

    rendered = format_report(
        RetentionReport(
            probes=[
                _probe("a", total=10, retained=10, recoverable=0),
                _probe("huge", total=1000, retained=0, recoverable=1000),
            ]
        )
    )
    assert "the mean across 2 domains, each counted once" in rendered
    assert "fact-weighted, the same figure reads" in rendered
    assert "by domain (each counts once" in rendered


def test_single_domain_report_keeps_the_plain_sentence() -> None:
    """With one domain a macro average says nothing extra, so do not clutter it."""
    from distil.retention import RetentionReport, format_report

    rendered = format_report(RetentionReport(probes=[_probe("only", 10, 4, 6)]))
    assert "reversibility is worth 60.0% recall here" in rendered
    assert "fact-weighted" not in rendered


def test_corpus_macro_is_reported_and_beats_micro_for_stability() -> None:
    """On the real corpus the two must actually differ, or the distinction is untested."""
    report = run()
    macro = report.macro()
    overall = report.overall()
    # Derived from the manifest, not hardcoded: pinning the literal count made adding a
    # trajectory look like a retention regression, which is a test failing for a reason
    # it was never written to detect.
    assert macro.domains == len({e.domain for e in load_corpus()})
    assert macro.domains >= 8, "the corpus must stay broad enough for macro to mean something"
    assert macro.recall == 1.0 and overall.lost == 0
    # web-research carries 61% of the facts, so fact-weighting inflates the gap.
    assert macro.gap < (overall.recall - overall.visible_recall)

    payload = report.to_dict()
    assert list(payload)[1] == "macro", "macro must lead the payload, before micro"
    assert payload["micro"]["note"].startswith("fact-weighted")
    assert set(payload["by_domain"]) >= {"web-research", "coding"}


def test_long_domain_labels_do_not_break_the_table() -> None:
    """Audit finding. `{x:<14}` pads but never truncates, so a custom corpus domain
    longer than the column would shove every later column right. The clamp marker must
    stay ASCII — `…` has no cp1252 code point and would resurrect the Windows crash."""
    from distil.retention import RetentionReport, _fit, format_report

    assert _fit("coding") == "coding"
    assert _fit("exactly14chars") == "exactly14chars"  # boundary: untouched
    assert _fit("vision/ui-automation") == "vision/ui-aut+"
    assert len(_fit("vision/ui-automation")) == 14

    rendered = format_report(
        RetentionReport(
            probes=[_probe("vision/ui-automation", 10, 5, 5), _probe("coding", 10, 10, 0)]
        )
    )
    # `_row` lays out "  " + 14-wide label + right-aligned recall, so the numeric
    # column always begins at index 16. Check that directly rather than hunting for a
    # "%" — bucket labels like "50%-75%" contain one, and so does the closing prose.
    rows = [ln for ln in rendered.splitlines() if ln[2:16].strip() in {"vision/ui-aut+", "coding"}]
    assert len(rows) == 2, rendered
    for line in rows:
        assert line[16:23].strip().endswith("%"), f"column drifted: {line!r}"
    # The clamp marker itself must be cp1252-safe. (The full report legitimately
    # carries U+2588 bars under a UTF-8 console; that path is covered by
    # test_report_encodes_on_a_windows_console, which patches stdout.)
    _fit("vision/ui-automation").encode("cp1252")


def test_blocks_without_a_domain_get_a_label_that_fits() -> None:
    from distil.retention import RetentionReport

    report = RetentionReport(probes=[_probe("", 4, 4, 0)])
    (label,) = report.by_domain()
    assert label == "(no domain)" and len(label) <= 14


def test_report_encodes_on_a_windows_console() -> None:
    """U+2588 is not in cp1252, so printing the bar to a stock Windows console raised
    UnicodeEncodeError and took down the retention CI gate on windows-latest while
    every POSIX leg passed. Every rendered surface must survive that encoding."""
    import io
    import sys

    from distil.retention import LiveMeter, _bar_char, format_live, load_live

    real = sys.stdout
    try:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        assert _bar_char() == "#"  # degrades rather than raising
        corpus = format_report(run())
        dims, requests = load_live(Path(__file__).parent / "does-not-exist.jsonl")
        live = format_live(dims, requests)
    finally:
        sys.stdout = real

    for rendered in (corpus, live):
        rendered.encode("cp1252")  # raises if any glyph is unencodable

    assert _bar_char() == "█"  # and keeps the block glyph where it is supported
    assert LiveMeter(0.0).enabled is False


def test_corpus_is_lossless_and_gains_from_reversibility() -> None:
    """The real corpus on the real pipeline: Tier-1/0 are lossless by construction,
    so nothing may be lost — and some facts must land behind a handle, otherwise the
    recoverable path is untested by this suite."""
    report = run()
    overall = report.overall()
    assert overall.total > 0
    assert overall.lost == 0
    assert overall.recall == 1.0
    assert overall.recoverable > 0, "no digested facts — the recoverable path is unexercised"
    assert overall.visible_recall < 1.0

    rendered = format_report(report)
    assert "reversibility is worth" in rendered
    assert "by block savings:" in rendered

    payload = report.to_dict()
    assert payload["overall"] == overall.to_dict()
    assert set(payload["by_dimension"]) == {"numerics", "artifacts", "errors"}
    assert any(b["total"] for b in payload["by_savings_bucket"].values())
