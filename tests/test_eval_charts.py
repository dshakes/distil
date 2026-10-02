"""The eval bubble charts in docs/assets/ must match a fresh build and the source artifacts."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

from benchmarks.swebench_outcome.report import analyse

_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _ROOT / "docs" / "assets"
_spec = importlib.util.spec_from_file_location(
    "build_eval_charts", _ROOT / "scripts" / "build_eval_charts.py"
)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod  # dataclasses resolve string annotations via sys.modules
_spec.loader.exec_module(mod)

_CIRCLE = re.compile(r'data-label="([^"]*)" data-x="([^"]*)" data-y="([^"]*)" data-size="([^"]*)"')


def _bubbles(name: str) -> dict[str, tuple[float, float, float]]:
    text = (_ASSETS / name).read_text()
    return {m[0]: (float(m[1]), float(m[2]), float(m[3])) for m in _CIRCLE.findall(text)}


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def test_charts_match_fresh_build() -> None:
    """Regenerating must be a no-op. If this fails, run: python3 scripts/build_eval_charts.py"""
    for name, svg in mod.build().items():
        assert (_ASSETS / name).read_text() == svg, (
            f"{name} stale. Run: python3 scripts/build_eval_charts.py"
        )


def test_certifier_bubbles_equal_summary() -> None:
    summary = json.loads(mod.SUMMARY.read_text())["certifier_migration_real"]["variants"]
    got = _bubbles("eval-bubble-certifier.svg")
    assert len(got) == len(summary) == 5
    for v in summary.values():
        label = v["model"] + (f" {v['effort']}" if v["effort"] else "")
        t = v["test_rates"]
        assert got[label] == (
            v["cost_per_case_usd"],
            t["equiv_expand_act"] * 100,
            t["self_consist_act"] * 100,
        )


def test_outcome_bubbles_cover_every_run_and_match_source() -> None:
    runs = sorted((_ROOT / "benchmarks" / "results").glob("swebench-outcome-*"))
    runs = [d for d in runs if (d / "grades.jsonl").exists()]
    assert runs
    got = _bubbles("eval-bubble-outcome.svg")
    assert len(got) == 2 * len(runs)
    for d in runs:
        res, grades = _jsonl(d / "results.jsonl"), _jsonl(d / "grades.jsonl")
        a = analyse(res, grades)
        run = d.name.removeprefix("swebench-outcome-")
        for arm in ("plain", "distil"):
            rs = [r for r in res if r["arm"] == arm]
            cost = round(math.fsum(r.get("cost_usd", 0) for r in rs) / len(rs), 6)
            n, k = a["arms"][arm]["n"], a["arms"][arm]["resolved"]
            assert got[f"{run} {arm}"] == (cost, k / n * 100, float(n))
