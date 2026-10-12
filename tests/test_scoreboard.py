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
    ct = next(s for s in data["sections"] if s["id"] == "terminal-bench-pilot")
    assert {r["status"] for r in ct["rows"]} == {"measured", "not in this harness"}
    for r in ct["rows"]:
        if r["status"] == "measured" and r["arm"] != "plain":
            assert r["verdict"] == "pilot, not shown", "a pilot never names a winner"


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


def test_cross_arm_cache_reads_never_show_as_an_arms_own_cost(tmp_path, monkeypatch):
    """The 2026-10-05 Lite arms read each other's prompt cache. Their billed dollars must not be
    shown as theirs: a full cold re-pricing, or 'confounded', never the billed figure."""
    sb = _mod()
    h2h = next(s for s in sb.build()["sections"] if s["id"] == "swebench-lite-300-h2h")
    by = {r["arm"]: r for r in h2h["rows"]}
    assert by["plain"]["cost_basis"] == by["distil"]["cost_basis"] == "cold"
    assert by["rtk"]["cost_basis"] == "cold re-priced" and by["rtk"]["precached_rows"] == [190, 300]
    assert by["rtk"]["usd_per_solved"] == 0.036 and by["rtk"]["usd_per_solved_billed"] == 0.0306
    for arm in ("selective", "provider-cm"):
        assert by[arm]["cost_basis"] == "confounded" and by[arm]["usd_per_solved"] is None
    assert by["selective"]["verdict"] == "INCONCLUSIVE"  # success columns untouched
    assert h2h["cost_confound"]["rtk"] == {
        "per_task_vs_plain_pct_cold": -6.5,
        "per_task_vs_plain_ci_pct_cold": [-13.1, 0.1],
        "per_task_vs_plain_pct_billed": -20.7,
    }
    for s in sb.build()["sections"]:
        if s["id"] != "swebench-lite-300-h2h":
            assert all(r.get("cost_basis", "cold") == "cold" for r in s["rows"]), s["id"]
    page = sb.render_html(sb.build())
    assert "$ per solved task (cold cache)" in page and "$0.0360†" in page
    assert "confounded — " in page and 'id="swebench-lite-300-h2h-cost-confound"' in page

    # A contaminated run with no committed cold re-pricing shows no dollar figure at all.
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {
            "instance_id": f"i{i}",
            "arm": a,
            "cost_usd": 0.5,
            "steps": 5,
            "usage": {
                "input": 2,
                "output": 9,
                "cache_write": 10 if a == "rtk" else 5000,
                "cache_read": 4000,
            },
        }
        for i in range(3)
        for a in ("plain", "rtk")
    ]
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run / "grades.jsonl").write_text(
        "\n".join(
            json.dumps({"instance_id": f"i{i}", "arm": a, "status": "resolved"})
            for i in range(3)
            for a in ("plain", "rtk")
        )
    )
    idx = tmp_path / "index.json"
    idx.write_text(
        json.dumps(
            {"runs": [{"id": "a", "kind": "swebench_outcome", "dir": str(run), "date": "d"}]}
        )
    )
    monkeypatch.setattr(sb, "INDEX", idx)
    rtk = {r["arm"]: r for r in sb.build()["sections"][0]["rows"]}["rtk"]
    assert rtk["cost_basis"] == "confounded" and rtk["cost_usd"] is None
    assert rtk["cost_usd_billed"] == 1.5 and rtk["precached_rows"] == [3, 3]
