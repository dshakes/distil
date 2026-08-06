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
                benchsuite.BenchRow("a", "rich", 5, 0.5, 1.0, 1.0, 0),
                benchsuite.BenchRow("b", "thin", 5, 0.1, 1.0, 1.0, 0),
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
