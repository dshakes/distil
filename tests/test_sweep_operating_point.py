"""Tests for the Phase-2 operating-point sweep (benchmarks/sweep_operating_point.py).

The grading itself needs a live model, but the selection logic — building the
candidate ladder, computing per-point decision-change/savings, and picking the
highest-savings point that certifies on CALIBRATION — is pure and must be locked:
it is the part that guarantees *no test-set tuning*.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sweep():
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "sweep_operating_point.py"
    spec = importlib.util.spec_from_file_location("sweep_operating_point", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_candidate_ladder_shape_and_order():
    s = _load_sweep()
    ladder = s.candidate_ladder([120, 500], [3.2], [10])
    names = [n for n, _ in ladder]
    # anchors first, then protected (high→low budget), then plain truncations
    assert names[0] == "byte-exact" and names[1] == "lossless"
    assert "protect+trunc@500,e3.2,l10" in names
    assert "protect+trunc@120,e3.2,l10" in names
    assert names.index("protect+trunc@500,e3.2,l10") < names.index("protect+trunc@120,e3.2,l10")
    assert "trunc@500" in names and "trunc@120" in names
    # every entry is (name, callable strategy)
    assert all(callable(strat) for _, strat in ladder)


def _matrix(spec):
    """Build a minimal loss matrix. spec: {tid: {name: (loss, comp_tok)}}, base_tok=100."""
    m = {}
    for tid, levels in spec.items():
        m[tid] = {
            "turns": [
                {
                    "base_tok": 100,
                    "levels": {
                        name: {"loss": loss, "comp_tok": ct} for name, (loss, ct) in levels.items()
                    },
                }
            ]
        }
    return m


def test_point_stats_decision_change_and_savings():
    s = _load_sweep()
    # point "A": flips on 1 of 2 trajectories; keeps 60 of 100 tokens → 40% savings
    m = _matrix(
        {
            "t1": {"A": (1.0, 60)},
            "t2": {"A": (0.0, 60)},
        }
    )
    stats = s.point_stats(m, ["A"], ["t1", "t2"])["A"]
    assert stats["n"] == 2
    assert abs(stats["decision_change"] - 0.5) < 1e-9
    assert abs(stats["savings"] - 0.40) < 1e-9


def test_select_prefers_highest_savings_among_certifying():
    s = _load_sweep()
    # build 40 trajectories so HB has power. "safe" never flips (certifies) at 25% savings;
    # "greedy" flips 50% (won't certify) at 60% savings; "tiny" never flips at 5% savings.
    spec = {}
    for i in range(40):
        spec[f"t{i}"] = {
            "safe": (0.0, 75),  # 25% savings, 0% dec-change
            "greedy": (1.0 if i % 2 == 0 else 0.0, 40),  # 60% savings, 50% dec-change
            "tiny": (0.0, 95),  # 5% savings, 0% dec-change
        }
    m = _matrix(spec)
    cal_stats = s.point_stats(m, ["safe", "greedy", "tiny"], list(spec))
    winner = s.select_on_calibration(cal_stats, alpha=0.15, delta=0.05)
    # greedy saves most but flips 50% (won't certify); safe is the best CERTIFYING point
    assert winner == "safe"


def test_select_returns_none_when_nothing_certifies():
    s = _load_sweep()
    # the only point flips on every trajectory → cannot certify at α=0.15
    spec = {f"t{i}": {"bad": (1.0, 50)} for i in range(40)}
    m = _matrix(spec)
    cal_stats = s.point_stats(m, ["bad"], list(spec))
    assert s.select_on_calibration(cal_stats, alpha=0.15, delta=0.05) is None


def test_sweep_latex_marks_selected_and_lists_families():
    s = _load_sweep()
    import json

    rep = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "docs/paper/results/swe_localization_sweep_shuffled.json"
        ).read_text()
    )
    tex = s.sweep_latex(rep)
    assert "\\begin{tabular}" in tex and "\\bottomrule" in tex
    # the selected operating point is bolded
    sel = rep["selected"]["operating_point"].replace("trunc", "t")
    assert f"\\textbf{{{sel}}}" in tex
    macros = s.sweep_macros(rep)
    assert "\\SweepTestSav" in macros and "\\SweepPoint" in macros
