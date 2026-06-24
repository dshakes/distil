"""Schema validation for committed paper result JSONs (``docs/paper/results/*``).

These files are produced by ``benchmarks/prove.py --report`` and consumed by
``benchmarks/report_to_latex.py`` to fill the paper's tables/macros. A drift in the
report structure would silently break table generation, so we lock the schema here.

In particular the shuffled-position E5 variant
(``swe_localization_e5_shuffled_headtohead.json``) must mirror the original E5
schema exactly, so the paper's table generator consumes both with one code path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "paper" / "results"

# E5 head-to-head reports (original + any additive variants) share one schema.
E5_REPORTS = sorted(RESULTS.glob("swe_localization_e5*headtohead.json"))


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _assert_e5_schema(d: dict) -> None:
    """Validate one prove.py E5 report against the locked structure."""
    assert set(d) >= {
        "args",
        "n_trajectories",
        "n_turns",
        "frontier",
        "head_to_head",
        "coverage",
        "shift",
        "task_success",
    }, f"missing top-level keys: {set(d)}"
    assert isinstance(d["n_trajectories"], int) and d["n_trajectories"] > 0
    assert isinstance(d["n_turns"], int) and d["n_turns"] > 0

    # frontier rows
    assert d["frontier"], "frontier must be non-empty"
    for r in d["frontier"]:
        assert isinstance(r["level"], str)
        assert isinstance(r["n"], int)
        for k in (
            "decision_change",
            "decision_change_effective",
            "trivial_frac",
            "savings",
        ):
            assert _is_number(r[k]), f"frontier.{k} not numeric"
        assert isinstance(r["effective_n"], int)

    # head-to-head rows
    assert d["head_to_head"], "head_to_head must be non-empty (run with --baselines)"
    for r in d["head_to_head"]:
        assert isinstance(r["method"], str)
        assert r["kind"] in ("distil", "baseline")
        assert _is_number(r["savings"]) and _is_number(r["decision_change"])
        assert isinstance(r["certifies"], bool)

    # coverage block (E2)
    cov = d["coverage"]
    for k in (
        "alpha",
        "delta",
        "certified_frac",
        "empirical_coverage",
        "mean_realized_risk",
        "mean_test_savings",
    ):
        assert _is_number(cov[k]), f"coverage.{k} not numeric"
    assert isinstance(cov["reps"], int)
    assert isinstance(cov["detail"], list) and cov["detail"]

    # shift (E3) — list, may be empty for single-domain corpora
    assert isinstance(d["shift"], list)

    # task_success (E4) — dict or None
    ts = d["task_success"]
    if ts is not None:
        assert isinstance(ts["n"], int)
        assert _is_number(ts["baseline_success"])
        assert isinstance(ts["outcome_evidential"], bool)
        for r in ts["levels"]:
            for k in (
                "savings",
                "retained_success",
                "ci_low",
                "ci_high",
                "preserved_frac",
            ):
                assert _is_number(r[k]), f"task_success.levels.{k} not numeric"


def test_at_least_one_e5_report_present():
    assert E5_REPORTS, "no E5 head-to-head report JSONs found under docs/paper/results/"


@pytest.mark.parametrize("path", E5_REPORTS, ids=lambda p: p.name)
def test_e5_report_schema(path: Path):
    _assert_e5_schema(json.loads(path.read_text()))


def test_e5_variants_share_schema():
    """All E5 reports must expose the same head-to-head method *set* and ladder, so a
    cross-corpus comparison table is apples-to-apples."""
    if len(E5_REPORTS) < 2:
        pytest.skip("only one E5 report committed so far")
    method_sets = []
    for p in E5_REPORTS:
        d = json.loads(p.read_text())
        method_sets.append(frozenset(r["method"] for r in d["head_to_head"]))
    assert len(set(method_sets)) == 1, f"E5 variants disagree on methods: {method_sets}"


# Phase-2 operating-point sweep reports.
SWEEP_REPORTS = sorted(RESULTS.glob("swe_localization_sweep*.json"))


def _assert_sweep_schema(d: dict) -> None:
    """Validate one sweep_operating_point.py report against its locked structure."""
    assert set(d) >= {
        "args",
        "n_cal",
        "n_test",
        "alpha",
        "delta",
        "grid",
        "calibration",
        "test",
        "full",
        "selected",
    }
    assert isinstance(d["n_cal"], int) and isinstance(d["n_test"], int)
    for split in ("calibration", "test", "full"):
        assert d[split], f"{split} stats must be non-empty"
        for name, st in d[split].items():
            assert isinstance(name, str)
            assert isinstance(st["n"], int)
            assert _is_number(st["decision_change"]) and _is_number(st["savings"])
    # cal and test must cover the same operating points (apples-to-apples selection)
    assert set(d["calibration"]) == set(d["test"]) == set(d["full"])
    # the disjoint-split invariant the whole experiment rests on (holds unconditionally)
    assert d["n_cal"] + d["n_test"] == d["n_trajectories"]
    sel = d["selected"]
    if sel is not None:
        assert sel["operating_point"] in d["test"]
        for k in ("cal_savings", "test_savings", "test_decision_change"):
            assert _is_number(sel[k])
        assert isinstance(sel["test_certifies"], bool)


@pytest.mark.parametrize("path", SWEEP_REPORTS, ids=lambda p: p.name)
def test_sweep_report_schema(path: Path):
    _assert_sweep_schema(json.loads(path.read_text()))
