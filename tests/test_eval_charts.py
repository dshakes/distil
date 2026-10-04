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
        if not any((r.get("usage") or {}).get("cache_read") for r in res):
            run += " (uncached)"  # pre-caching runs are labelled as not cost-comparable
        for arm in ("plain", "distil"):
            rs = [r for r in res if r["arm"] == arm]
            cost = round(math.fsum(r.get("cost_usd", 0) for r in rs) / len(rs), 6)
            n, k = a["arms"][arm]["n"], a["arms"][arm]["resolved"]
            assert got[f"{run} {arm}"] == (cost, k / n * 100, float(n))


def _write_run(root: Path, name: str, arms: tuple[str, ...]) -> None:
    d = root / f"swebench-outcome-{name}"
    d.mkdir(parents=True)
    res, grades = [], []
    for i in range(4):
        for k, arm in enumerate(arms):
            res.append(
                {
                    "instance_id": f"i{i}",
                    "arm": arm,
                    "failure_class": None,
                    "cost_usd": 0.01 * (k + 1),
                    "usage": {"cache_read": 1},
                }
            )
            status = "resolved" if i <= k else "unresolved"
            grades.append({"instance_id": f"i{i}", "arm": arm, "status": status})
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
    (d / "grades.jsonl").write_text("".join(json.dumps(g) + "\n" for g in grades))


def test_outcome_chart_handles_n_arms(tmp_path) -> None:
    _write_run(tmp_path, "7", ("plain", "distil", "rtk", "provider-cm", "zzz-new"))
    _write_run(tmp_path, "8", ("plain", "selective"))
    bubbles = mod.outcome_bubbles(tmp_path)
    assert [b.label for b in bubbles] == [
        "7 plain",
        "7 distil",
        "7 rtk",
        "7 provider-cm",
        "7 zzz-new",
        "8 plain",
        "8 selective",
    ]
    by = {b.label: b for b in bubbles}
    # rtk is the 3rd arm (k=2): solves i0..i2 of 4 paired tasks at $0.03 each
    assert by["7 rtk"].x == 0.03 and by["7 rtk"].y == 75.0 and by["7 rtk"].size == 4.0
    assert by["7 plain"].color == by["8 plain"].color == mod.ACC2
    assert by["7 zzz-new"].color == mod.EXTRA_COLORS[0]  # unknown arms still get a colour
    assert len({b.color for b in bubbles}) == 6
    assert mod.outcome_key(tmp_path) == [
        ("plain agent", mod.ACC2),
        ("distil-served agent", mod.ACC),
        ("RTK (rewrite hook)", mod.ARM_STYLE["rtk"][0]),
        ("Selective Context", mod.ARM_STYLE["selective"][0]),
        ("Anthropic context editing", mod.ARM_STYLE["provider-cm"][0]),
        ("zzz-new", mod.EXTRA_COLORS[0]),
    ]
    assert mod.outcome_key(_ROOT / "benchmarks" / "results") == [
        ("plain agent", mod.ACC2),
        ("distil-served agent", mod.ACC),
    ]
