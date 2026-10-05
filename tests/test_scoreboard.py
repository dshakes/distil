"""docs/scoreboard.{json,html} are generated from committed artifacts (scripts/build_scoreboard.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location(
        "build_scoreboard", ROOT / "scripts" / "build_scoreboard.py"
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_committed_scoreboard_matches_its_artifacts():
    """Regenerate-and-compare: a number on the page that its run files do not produce fails."""
    sb = _mod()
    data = sb.build()
    assert (ROOT / "docs/scoreboard.json").read_text(encoding="utf-8") == json.dumps(
        data, indent=2, sort_keys=True
    ) + "\n", "stale: run python3 scripts/build_scoreboard.py"
    assert (ROOT / "docs/scoreboard.html").read_text(encoding="utf-8") == sb.render_html(data)
    assert sb.main(["x", "--check"]) == 0


def test_every_compressor_has_a_row_and_unrun_arms_say_pending():
    data = _mod().build()
    for s in data["sections"]:
        assert [r["arm"] for r in s["rows"]] == [
            "plain",
            "distil",
            "rtk",
            "headroom",
            "selective",
            "provider-cm",
        ]
        for r in s["rows"]:
            if r["status"] != "measured":
                assert set(r) == {"arm", "label", "status"}, "nothing invented for an unrun arm"
    ct = next(s for s in data["sections"] if s["id"] == "terminal-bench")
    assert {r["status"] for r in ct["rows"]} == {"pending run", "not in this harness"}


def test_missing_arms_and_cost_truth_results_render(tmp_path, monkeypatch):
    sb = _mod()
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {
            "instance_id": f"i{i}",
            "arm": a,
            "model": "m",
            "effort": "low",
            "cost_usd": 0.5,
            "usage": {"input": 10, "output": 2},
            "steps": 3,
            "arm_meta": {"library": "rtk", "version": "0.51.0"} if a == "rtk" else {},
        }
        for i in range(4)
        for a in ("plain", "rtk")
    ]
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    grades = [
        {
            "instance_id": f"i{i}",
            "arm": a,
            "status": "resolved" if i < (3 if a == "plain" else 2) else "unresolved",
        }
        for i in range(4)
        for a in ("plain", "rtk")
    ]
    (run / "grades.jsonl").write_text("\n".join(json.dumps(g) for g in grades))
    ct = tmp_path / "ct"
    ct.mkdir()
    (ct / "analysis.json").write_text(
        json.dumps(
            {
                "tasks": 2,
                "per_arm": {
                    "control": {
                        "attempts": 4,
                        "solved": 2,
                        "success_rate": 0.5,
                        "cost_usd": 2.0,
                        "usd_per_solved": 1.0,
                    },
                    "headroom": {
                        "attempts": 4,
                        "solved": 0,
                        "success_rate": 0.0,
                        "cost_usd": 1.0,
                        "usd_per_solved": None,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    idx = tmp_path / "index.json"
    idx.write_text(
        json.dumps(
            {
                "runs": [
                    {"id": "a", "kind": "swebench_outcome", "dir": str(run), "date": "2026-10-04"},
                    {"id": "b", "kind": "cost_truth", "dir": str(ct), "date": "2026-10-04"},
                ]
            }
        )
    )
    monkeypatch.setattr(sb, "INDEX", idx)
    data = sb.build()
    swe, ctr = data["sections"]
    by = {r["arm"]: r for r in swe["rows"]}
    assert by["distil"]["status"] == "pending run"
    assert by["rtk"]["version"] == "0.51.0" and by["rtk"]["usd_per_solved"] == 1.0
    assert by["plain"]["usd_per_solved"] == round(2.0 / 3, 4), "failed attempts' spend is counted"
    assert by["rtk"]["verdict"] == "PILOT (no verdict)"
    cby = {r["arm"]: r for r in ctr["rows"]}
    assert cby["plain"]["usd_per_solved"] == 1.0 and cby["headroom"]["usd_per_solved"] is None
    assert (
        cby["rtk"]["status"] == "pending run"
        and cby["selective"]["status"] == "not in this harness"
    )
    page = sb.render_html(data)
    assert "pending run" in page and "$1.0000" in page and "rtk 0.51.0" in page
