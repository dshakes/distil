"""The public-benchmark suite.

The load-bearing tests are the ones about EVIDENCE, not about arithmetic. A suite
like this fails in two ways that both look like success:

  * a benchmark cannot be fetched, is quietly dropped, and the run reports a clean
    sheet for less work than it claimed;
  * every graded benchmark is a thin-payload control, so the table is full of
    unchanged scores that demonstrate nothing about compression.

Both exit non-zero here. Everything hits cached rows only — a suite you skip in CI
because it needs a network or a key is not a gate.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from distil import benchsuite, datasets


def _seed(name: str, rows: list[dict]) -> None:
    """Write cache rows into the sandboxed DISTIL_HOME.

    `conftest` gives every test a fresh DISTIL_HOME, so these tests must bring their
    own data. An earlier version read the developer's real ~/.distil at import time
    and passed on this machine while testing nothing in CI — a fixture that only
    exists on one laptop is not a fixture.
    """
    home = pathlib.Path(os.environ["DISTIL_HOME"]) / "datasets"
    home.mkdir(parents=True, exist_ok=True)
    (home / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


# Minimal rows in each upstream's real shape, verified against live data.
BFCL_ROWS = [
    {
        "id": f"simple_{i}",
        "question": [[{"role": "user", "content": f"Find the area of triangle {i}"}]],
        "function": [
            {
                "name": "calculate_triangle_area",
                "description": "Calculate the area of a triangle. " + "padding " * 40,
                "parameters": {"type": "dict", "properties": {"base": {"type": "integer"}}},
            }
        ],
        "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5]}}],
    }
    for i in range(5)
]

GSM8K_ROWS = [
    {
        "question": f"Janet has {i} eggs and sells 3. How many remain?",
        "answer": f"She has {i} minus 3.\n#### {i - 3}",
    }
    for i in range(5, 10)
]


@pytest.fixture
def seeded():
    """A rich benchmark and a control, both cached, no network."""
    _seed("bfcl", BFCL_ROWS)
    _seed("gsm8k", GSM8K_ROWS)
    return {"rich": "bfcl", "thin": "gsm8k"}


class TestPayloadClassification:
    """Only rich-payload benchmarks can demonstrate compression quality."""

    def test_every_registered_benchmark_declares_a_payload_class(self) -> None:
        for name in datasets.SPECS:
            assert datasets.payload_class(name) in ("rich", "thin"), name

    def test_the_controls_are_the_ones_with_nothing_to_compress(self) -> None:
        """GSM8K/MMLU/ARC/TruthfulQA/TriviaQA are one-line questions. Labelling them
        as evidence is how a benchmark table overstates itself."""
        for name in ("gsm8k", "mmlu", "arc", "truthfulqa", "triviaqa"):
            assert datasets.payload_class(name) == "thin", f"{name} is not evidence"

    def test_the_agent_payloads_are_evidence(self) -> None:
        for name in ("bfcl", "hotpotqa", "squad", "msmarco", "narrativeqa"):
            assert datasets.payload_class(name) == "rich", f"{name} should be evidence"

    def test_an_unknown_name_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(datasets.DatasetUnavailable):
            datasets.payload_class("not-a-benchmark")


class TestTiers:
    def test_tier_one_leads_with_tool_calling(self) -> None:
        """BFCL is the failure an agent proxy is most likely to cause and least
        likely to notice: a QA benchmark never asks the model to act."""
        assert benchsuite.TIERS[1][0] == "bfcl"

    def test_every_tiered_name_is_registered(self) -> None:
        for tier, names in benchsuite.TIERS.items():
            for name in names:
                assert name in datasets.SPECS, f"tier {tier} references unknown {name!r}"

    def test_every_registered_benchmark_appears_in_some_tier(self) -> None:
        """A benchmark nobody runs is a benchmark nobody maintains."""
        tiered = {n for names in benchsuite.TIERS.values() for n in names}
        assert set(datasets.SPECS) == tiered, set(datasets.SPECS) ^ tiered


class TestAFailedBenchmarkIsNeverSilent:
    def test_an_unfetchable_benchmark_becomes_a_failed_row(self) -> None:
        report = benchsuite.run(names=["definitely-not-a-dataset"], offline=True)
        assert len(report.rows) == 1
        assert report.failed and not report.graded
        assert "definitely-not-a-dataset" in report.to_dict()["failed"]

    def test_offline_without_cache_fails_loudly(self, tmp_path) -> None:
        env = dict(os.environ, DISTIL_HOME=str(tmp_path))
        out = subprocess.run(
            [sys.executable, "-m", "distil.cli", "suite", "--tier", "1", "--offline"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert out.returncode == 1, "an empty cache must not report a clean suite"
        assert "could not be graded" in out.stdout + out.stderr


class TestGradingRealCachedBenchmarks:
    def test_a_cached_benchmark_grades(self, seeded) -> None:
        report = benchsuite.run(names=[seeded["rich"]], n=3, offline=True)
        row = report.rows[0]
        assert row.ok, row.error
        assert row.cases > 0 and 0.0 <= row.savings <= 1.0
        assert 0.0 <= row.answer_recall <= 1.0 and 0.0 <= row.support_recall <= 1.0

    def test_metrics_are_not_all_zero(self, seeded) -> None:
        """`_num` reads a NESTED block; reading the top level returned 0.0 for
        everything and still rendered a complete-looking table."""
        report = benchsuite.run(names=[seeded["rich"]], n=3, offline=True)
        assert report.rows[0].savings > 0.0, "a table of zeros is not a passing suite"

    def test_json_report_is_machine_readable_and_content_free(self, seeded) -> None:
        report = benchsuite.run(names=[seeded["rich"]], n=3, offline=True)
        blob = json.dumps(report.to_dict())
        parsed = json.loads(blob)
        assert parsed["rows"] and "evidence_benchmarks" in parsed
        # counts and names only — never a question, passage or answer
        assert "?" not in blob.replace('"payload": "?"', ""), (
            "benchmark text leaked into the report"
        )


class TestControlsAloneCannotPass:
    def test_a_controls_only_run_exits_nonzero(self, seeded) -> None:
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "distil.cli",
                "suite",
                "--only",
                seeded["thin"],
                "-n",
                "3",
                "--offline",
            ],
            capture_output=True,
            text=True,
        )
        assert out.returncode == 1, "controls alone must not certify compression quality"
        assert "rich-payload" in out.stdout + out.stderr

    def test_the_report_separates_evidence_from_controls(self) -> None:
        report = benchsuite.SuiteReport(
            rows=[
                benchsuite.BenchRow("a", "rich", 5, 0.5, 1.0, 1.0, 0, support_facts=9),
                benchsuite.BenchRow("b", "thin", 5, 0.1, 1.0, 1.0, 0, support_facts=9),
            ]
        )
        assert [r.name for r in report.evidence] == ["a"]
        assert [r.name for r in report.controls] == ["b"]
        assert "controls" in benchsuite.format_report(report)


class TestReportRendering:
    """The renderer must show a failure as a failure, not as a blank row."""

    def test_an_empty_selection_says_so(self) -> None:
        assert "nothing to grade" in benchsuite.format_report(benchsuite.SuiteReport())

    def test_a_failed_row_renders_as_FAILED_with_its_cause(self) -> None:
        report = benchsuite.SuiteReport(
            rows=[benchsuite.BenchRow("x", "rich", 0, 0.0, 0.0, 0.0, 0, error="HTTP 500 upstream")]
        )
        text = benchsuite.format_report(report)
        assert "FAILED" in text and "HTTP 500" in text
        assert "could not be graded" in text, "the summary must name the count too"

    def test_num_falls_back_rather_than_raising_on_a_shape_change(self) -> None:
        class _Odd:
            def to_dict(self):
                return {"distil": {"savings": "not-a-number"}}

        assert benchsuite._num(_Odd(), "savings") == 0.0
        assert benchsuite._num(_Odd(), "missing_field") == 0.0


class TestTheSuiteCannotPassOnACollapse:
    """A rich benchmark that graded cases and recalled nothing is a BROKEN run.

    Before this, `distil suite` exited 0 on it unless someone happened to pass a
    threshold flag — and `--max-lost` could not catch it either, because loss counted
    only support facts and benchmarks like SQuAD carry `support=[]`. So a total
    answer regression scored zero loss and reported success.
    """

    def test_a_collapsed_rich_benchmark_is_detected(self) -> None:
        report = benchsuite.SuiteReport(
            rows=[
                benchsuite.BenchRow(
                    "bfcl", "rich", 100, 0.9, 0.0, 0.0, 0, answer_graded=100, support_facts=50
                )
            ]
        )
        assert [r.name for r in report.collapsed] == ["bfcl"]
        assert report.to_dict()["collapsed"] == ["bfcl"]

    def test_a_control_at_zero_is_not_a_collapse(self) -> None:
        """Controls carry no compressible payload; they cannot demonstrate a collapse."""
        report = benchsuite.SuiteReport(
            rows=[benchsuite.BenchRow("gsm8k", "thin", 100, 0.0, 0.0, 0.0, 0)]
        )
        assert report.collapsed == []

    def test_a_benchmark_with_no_cases_is_not_a_collapse(self) -> None:
        report = benchsuite.SuiteReport(
            rows=[benchsuite.BenchRow("bfcl", "rich", 0, 0.0, 0.0, 0.0, 0)]
        )
        assert report.collapsed == []

    def test_the_cli_exits_one_on_a_collapse_with_no_flags(self, seeded, monkeypatch) -> None:
        """No threshold flag passed — the guard must still fire."""
        import distil.benchsuite as bs

        real = bs.run

        def _collapsed(*a, **k):
            report = real(*a, **k)
            for row in report.rows:
                row.answer_recall = row.support_recall = 0.0
            return report

        monkeypatch.setattr(bs, "run", _collapsed)
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            ["suite", "--only", seeded["rich"], "-n", "3", "--offline"]
        )
        assert cmd_suite(args) == 1

    def test_min_recall_flags_gate_rich_benchmarks(self) -> None:
        from distil.cli import build_parser

        args = build_parser().parse_args(["suite", "--min-answer-recall", "0.9"])
        assert args.min_answer_recall == 0.9
        args = build_parser().parse_args(["suite", "--min-support-recall", "0.8"])
        assert args.min_support_recall == 0.8

    def test_answer_loss_now_counts_toward_lost(self, seeded, monkeypatch) -> None:
        """SQuAD-style benchmarks have `support=[]`; counting only support facts made
        `--max-lost` structurally unable to see an answer regression."""
        import distil.retention as rt

        class _Report:
            def to_dict(self):
                return {
                    "distil": {
                        "savings": 0.9,
                        "answer_recall": 0.5,
                        "answer_graded": 10,
                        "support_recall": 1.0,
                        "support_facts": 0,
                    }
                }

        monkeypatch.setattr(rt, "score_dataset", lambda *a, **k: _Report())
        report = benchsuite.run(names=[seeded["rich"]], n=3, offline=True)
        assert report.rows[0].lost == 5, "half of 10 graded answers lost must count"


class TestAShortfallIsVisible:
    """`-n 100` graded on 7 cases is a different measurement.

    A split legitimately ends early and an adapter legitimately drops an unusable
    row — neither is an error, and `load()` documents that. But a thin run that
    presents itself as a full one is how a suite overstates its sample size, so the
    count is reported against what was asked for.
    """

    def test_a_short_row_knows_it_is_short(self) -> None:
        row = benchsuite.BenchRow("bfcl", "rich", 7, 0.9, 1.0, 1.0, 0, requested=100)
        assert row.short and row.to_dict()["short"] is True
        assert row.to_dict()["requested"] == 100

    def test_a_full_row_is_not_flagged(self) -> None:
        assert not benchsuite.BenchRow("bfcl", "rich", 100, 0.9, 1.0, 1.0, 0, requested=100).short

    def test_no_request_size_means_no_claim(self) -> None:
        """Without -n the dataset's own default applies; there is nothing to be short of."""
        assert not benchsuite.BenchRow("bfcl", "rich", 7, 0.9, 1.0, 1.0, 0).short

    def test_the_report_names_the_shortfall(self) -> None:
        report = benchsuite.SuiteReport(
            rows=[benchsuite.BenchRow("bfcl", "rich", 7, 0.9, 1.0, 1.0, 0, requested=100)]
        )
        text = benchsuite.format_report(report)
        assert "7/100" in text, "the table must show the shortfall inline"
        assert "not the sample size you asked for" in text

    def test_a_real_run_records_what_was_requested(self, seeded) -> None:
        report = benchsuite.run(names=[seeded["rich"]], n=3, offline=True)
        assert report.rows[0].requested == 3 and not report.rows[0].short


class TestCollapseUsesOnlyMeasuredDimensions:
    """A recall over zero golds is vacuously 1.0, and that made the guard blind.

    `retention` returns support_recall=1.0 when there are no support facts, so a
    SQuAD-shaped benchmark at 0% ANSWER recall satisfied "both recalls zero" only in
    the answer dimension and escaped. That is the identical blind spot fixed one
    property earlier in the loss count — reintroduced, and caught by the reviewer.
    """

    def test_squad_shaped_collapse_is_caught(self) -> None:
        """No support facts at all: the answer dimension is the only witness."""
        row = benchsuite.BenchRow(
            "squad", "rich", 100, 0.9, 0.0, 1.0, 100, answer_graded=100, support_facts=0
        )
        assert [r.name for r in benchsuite.SuiteReport(rows=[row]).collapsed] == ["squad"]

    def test_support_only_collapse_is_caught(self) -> None:
        """BFCL grades no answer spans; support (function names) is the witness."""
        row = benchsuite.BenchRow(
            "bfcl", "rich", 100, 0.9, 1.0, 0.0, 50, answer_graded=0, support_facts=50
        )
        assert [r.name for r in benchsuite.SuiteReport(rows=[row]).collapsed] == ["bfcl"]

    def test_a_healthy_row_is_not_flagged(self) -> None:
        row = benchsuite.BenchRow(
            "bfcl", "rich", 100, 0.9, 1.0, 1.0, 0, answer_graded=100, support_facts=50
        )
        assert benchsuite.SuiteReport(rows=[row]).collapsed == []

    def test_a_row_with_nothing_measured_is_not_a_collapse(self) -> None:
        """Zero golds of either kind means nothing testified — not a failure, and not
        a pass either. It is reported as graded-nothing rather than collapsed."""
        row = benchsuite.BenchRow("x", "rich", 100, 0.9, 1.0, 1.0, 0)
        assert benchsuite.SuiteReport(rows=[row]).collapsed == []

    def test_one_measured_dimension_at_zero_is_enough(self) -> None:
        """Partial evidence still convicts: if the only graded dimension is zero, the
        run collapsed regardless of what the ungraded one vacuously reports."""
        row = benchsuite.BenchRow(
            "m", "rich", 10, 0.9, 0.0, 1.0, 10, answer_graded=10, support_facts=0
        )
        assert [r.name for r in benchsuite.SuiteReport(rows=[row]).collapsed] == ["m"]

    def test_counts_reach_the_json(self, seeded) -> None:
        report = benchsuite.run(names=[seeded["rich"]], n=3, offline=True)
        blob = report.to_dict()["rows"][0]
        assert "answer_graded" in blob and "support_facts" in blob


class TestUnmeasuredIsNotEvidence:
    """A recall over zero golds renders as 100%. It is not a result.

    A rich row with cases but no golds of either kind graded nothing, yet counted as
    evidence and exited 0 — the vacuous pass this suite exists to refuse, arriving
    through the one row shape the collapse guard deliberately skips.
    """

    def test_a_row_with_no_golds_is_not_evidence(self) -> None:
        row = benchsuite.BenchRow("x", "rich", 100, 0.9, 1.0, 1.0, 0)
        report = benchsuite.SuiteReport(rows=[row])
        assert report.evidence == []
        assert [r.name for r in report.unmeasured] == ["x"]
        assert report.to_dict()["unmeasured"] == ["x"]

    def test_a_row_with_golds_is_evidence(self) -> None:
        row = benchsuite.BenchRow(
            "b", "rich", 100, 0.9, 1.0, 1.0, 0, answer_graded=0, support_facts=23
        )
        assert [r.name for r in benchsuite.SuiteReport(rows=[row]).evidence] == ["b"]

    def test_the_cli_fails_on_an_unmeasured_rich_row(self, seeded, monkeypatch) -> None:
        import distil.benchsuite as bs

        real = bs.run

        def _blank(*a, **k):
            rep = real(*a, **k)
            for row in rep.rows:
                row.answer_graded = row.support_facts = 0
            return rep

        monkeypatch.setattr(bs, "run", _blank)
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            ["suite", "--only", seeded["rich"], "-n", "3", "--offline"]
        )
        assert cmd_suite(args) == 1


class TestBfclGradesEveryNameTheCallNeeds:
    def test_argument_names_are_required_to_survive(self) -> None:
        from distil import datasets

        case = datasets._bfcl(
            {
                "id": "s0",
                "question": [[{"content": "area?"}]],
                "function": [{"name": "calc_area", "parameters": {}}],
                "ground_truth": [{"calc_area": {"base": [10], "height": [5]}}],
            }
        )
        assert case is not None
        assert set(case.support) >= {"calc_area", "base", "height"}, (
            "a schema can keep the function name and lose the parameter it needs"
        )


class TestTheCiGateHasARealThreshold:
    """A gate that only fires on total collapse is a loader smoke test.

    CI first ran `distil suite --tier 1 -n 25` with no thresholds, so an ordinary
    regression with unrecoverable gold loss exited 0. The gate now carries the
    MEASURED band on rich rows — and controls are excluded by construction, because
    gsm8k loses 17 bare-number answers at n=25 and that says nothing about
    compression.
    """

    def _gate_args(self, path: str) -> str:
        import pathlib

        return pathlib.Path(path).read_text(encoding="utf-8")

    def test_ci_passes_recall_thresholds(self) -> None:
        ci = self._gate_args(".github/workflows/ci.yml")
        assert "distil suite" in ci
        line = next(ln for ln in ci.splitlines() if "distil suite" in ln)
        assert "--min-answer-recall" in line and "--min-support-recall" in line, (
            "the CI gate must carry a threshold or it only catches a total collapse"
        )

    def test_make_gate_matches_ci(self) -> None:
        mk = self._gate_args("Makefile")
        line = next(ln for ln in mk.splitlines() if "distil suite" in ln)
        assert "--min-answer-recall" in line and "--min-support-recall" in line

    def test_a_default_run_records_the_spec_default_as_requested(self) -> None:
        """`requested` was 0 without -n, so a default run could never report a
        shortfall — and the default is the invocation users are told to run.

        (An insufficient OFFLINE cache is a hard failure by existing design, not a
        short row, so this seeds enough rows to let the run succeed.)
        """
        default_n = datasets.SPECS["bfcl"].default_n
        _seed("bfcl", BFCL_ROWS * (default_n // len(BFCL_ROWS) + 1))
        report = benchsuite.run(names=["bfcl"], offline=True)
        row = report.rows[0]
        assert row.ok, row.error
        assert row.requested == default_n, "a default run must know what it asked for"
        assert not row.short, "enough rows were cached"


class TestAnOutageIsNotARegression:
    """A third party's 504 is not a compression regression.

    The suite was added to the required CI gate and immediately failed it on
    `bfcl: HTTP Error 504` from the HuggingFace CDN. A required gate that fails on
    someone else's uptime teaches people to ignore it — but downgrading a fetch
    failure to silence would recreate the vacuous pass. So it warns loudly, and the
    run still fails if the outage took out everything that could testify.
    """

    def test_unavailable_is_a_warning_when_allowed(self, seeded) -> None:
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            [
                "suite",
                "--only",
                f"{seeded['rich']},msmarco",
                "-n",
                "3",
                "--offline",
                "--allow-unavailable",
            ]
        )
        assert cmd_suite(args) == 0, "a graded rich benchmark survives one outage"

    def test_unavailable_still_fails_without_the_flag(self, seeded) -> None:
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            ["suite", "--only", f"{seeded['rich']},msmarco", "-n", "3", "--offline"]
        )
        assert cmd_suite(args) == 1, "strict by default: a local run must not hide it"

    def test_a_total_outage_still_fails_even_when_allowed(self) -> None:
        """No rich evidence graded means nothing testified — the flag cannot rescue it."""
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            ["suite", "--only", "msmarco,narrativeqa", "--offline", "--allow-unavailable"]
        )
        assert cmd_suite(args) == 1

    def test_the_ci_gate_tolerates_an_outage_but_keeps_its_thresholds(self) -> None:
        import pathlib

        line = next(
            ln
            for ln in pathlib.Path(".github/workflows/ci.yml").read_text().splitlines()
            if "distil suite" in ln
        )
        assert "--allow-unavailable" in line
        assert "--min-answer-recall" in line and "--min-support-recall" in line, (
            "tolerating an outage must not mean dropping the quality bar"
        )


class TestAThresholdMustHaveSomethingToTest:
    """A recall of 1.0 over zero golds satisfies any threshold.

    BFCL grades no answer spans, so `--min-answer-recall 0.95` passed on it
    vacuously — and with an upstream outage tolerated, it could be the only row
    left. A threshold is now scoped to rows that carry golds in that dimension, and
    fails outright if none do.
    """

    def test_an_answer_threshold_fails_when_nothing_graded_answers(self, seeded) -> None:
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            [
                "suite",
                "--only",
                seeded["rich"],
                "-n",
                "3",
                "--offline",
                "--min-answer-recall",
                "0.95",
            ]
        )
        assert cmd_suite(args) == 1, "BFCL grades no answers; the threshold tested nothing"

    def test_a_support_threshold_passes_where_support_is_graded(self, seeded) -> None:
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            [
                "suite",
                "--only",
                seeded["rich"],
                "-n",
                "3",
                "--offline",
                "--min-support-recall",
                "0.5",
            ]
        )
        assert cmd_suite(args) == 0


class TestCompressionMustActuallyEngage:
    """The deepest hole this suite had: it could certify no compression at all.

    A regression that disables compression returns savings=0, perfect recall and
    non-empty evidence, and passes every threshold. `retention` already names this —
    "the arithmetic of an identity function" — and the suite was dropping the signal.
    """

    def test_an_idle_rich_row_is_detected(self) -> None:
        row = benchsuite.BenchRow(
            "bfcl", "rich", 10, 0.0, 1.0, 1.0, 0, support_facts=5, engaged=False
        )
        report = benchsuite.SuiteReport(rows=[row])
        assert [r.name for r in report.idle] == ["bfcl"]
        assert report.to_dict()["idle"] == ["bfcl"]

    def test_an_engaged_row_is_not_idle(self) -> None:
        row = benchsuite.BenchRow(
            "bfcl", "rich", 10, 0.9, 1.0, 1.0, 0, support_facts=5, engaged=True
        )
        assert benchsuite.SuiteReport(rows=[row]).idle == []

    def test_the_cli_fails_on_an_idle_benchmark(self, seeded, monkeypatch) -> None:
        import distil.benchsuite as bs

        real = bs.run

        def _idle(*a, **k):
            rep = real(*a, **k)
            for row in rep.rows:
                row.engaged = False
            return rep

        monkeypatch.setattr(bs, "run", _idle)
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            ["suite", "--only", seeded["rich"], "-n", "3", "--offline"]
        )
        assert cmd_suite(args) == 1, "no compression must never certify as a pass"

    def test_real_runs_report_engaged(self, seeded) -> None:
        report = benchsuite.run(names=[seeded["rich"]], n=3, offline=True)
        assert report.rows[0].engaged is True and report.idle == []


class TestAnUnknownNameIsNotAnOutage:
    """`--allow-unavailable` tolerates a CDN, not a typo.

    `--only bfcl,hotptoqa --allow-unavailable` graded one benchmark and exited 0,
    silently omitting the one that was misspelled. An unknown name is a mistake in
    the invocation and can never be waited out.
    """

    def test_a_typo_is_fatal_even_when_outages_are_allowed(self, seeded) -> None:
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            [
                "suite",
                "--only",
                f"{seeded['rich']},hotptoqa",
                "-n",
                "3",
                "--offline",
                "--allow-unavailable",
            ]
        )
        assert cmd_suite(args) == 1


class TestOnlyAvailabilityCanBeWaitedOut:
    """`--allow-unavailable` covers a CDN, never our own defect.

    `benchsuite.run` catches broad `Exception` so every failure surfaces as a row.
    That meant a `TypeError` in an adapter, or a schema drift leaving no adaptable
    rows, was downgraded exactly like a 504 — and CI passes the flag, so an internal
    regression exited 0 whenever another rich benchmark happened to pass.
    """

    def test_a_scoring_bug_is_not_downgradeable(self, seeded, monkeypatch) -> None:
        import distil.retention as rt
        from distil.cli import build_parser, cmd_suite

        def _boom(*a, **k):
            raise TypeError("scoring bug")

        monkeypatch.setattr(rt, "score_dataset", _boom)
        args = build_parser().parse_args(
            ["suite", "--only", seeded["rich"], "-n", "3", "--offline", "--allow-unavailable"]
        )
        assert cmd_suite(args) == 1, "our own exception must never be waited out"

    def test_a_transport_failure_is_downgradeable(self, seeded) -> None:
        from distil.cli import build_parser, cmd_suite

        args = build_parser().parse_args(
            [
                "suite",
                "--only",
                f"{seeded['rich']},msmarco",
                "-n",
                "3",
                "--offline",
                "--allow-unavailable",
            ]
        )
        assert cmd_suite(args) == 0

    def test_schema_drift_is_a_defect_not_an_outage(self, monkeypatch, tmp_path) -> None:
        """`no rows survived adaptation` means the upstream shape changed. Waiting
        does not fix that."""
        import json

        home = tmp_path / "datasets"
        home.mkdir(parents=True)
        (home / "gsm8k.jsonl").write_text(json.dumps({"unexpected": "shape"}) + "\n")
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        report = benchsuite.run(names=["gsm8k"], n=1, offline=True)
        assert report.rows[0].failure == "bug", report.rows[0].error

    def test_failure_kind_reaches_the_json(self, seeded) -> None:
        report = benchsuite.run(names=[seeded["rich"], "msmarco"], n=3, offline=True)
        kinds = {r["name"]: r["failure"] for r in report.to_dict()["rows"]}
        assert kinds[seeded["rich"]] == "" and kinds["msmarco"] == "unavailable"
