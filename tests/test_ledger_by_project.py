"""Per-project savings breakdown: ledger rollup + dashboard section, content-free."""

from __future__ import annotations

import json

from distil import ledger
from distil.runtime import RuntimeSavings


def _rec(path, project, base, got):
    ledger.record(
        trajectory_id="live-proxy",
        model="m",
        turns=1,
        baseline_dollars=0.0,
        distil_dollars=0.0,
        baseline_input_tokens=base,
        distil_input_tokens=got,
        project=project,
        path=path,
    )


def test_summary_groups_by_project_and_skips_unlabelled(tmp_path):
    p = tmp_path / "l.jsonl"
    _rec(p, "alpha", 1000, 400)
    _rec(p, "alpha", 500, 500)
    _rec(p, "beta", 2000, 1000)
    _rec(p, "", 9999, 1)  # always-on proxy: no project
    p.open("a").write(
        json.dumps(
            {
                "trajectory_id": "old",
                "baseline_dollars": 0,
                "distil_dollars": 0,
                "baseline_input_tokens": 5,
                "distil_input_tokens": 4,
            }
        )
        + "\n"
    )
    s = ledger.summary(p)
    assert s.by_project == {"alpha": [2, 1500, 900], "beta": [1, 2000, 1000]}


def test_dashboard_lists_top_projects_by_tokens_saved(tmp_path):
    p = tmp_path / "l.jsonl"
    _rec(p, "alpha", 1000, 400)
    _rec(p, "beta", 3000, 1000)
    out = ledger.render_dashboard(ledger.summary(p), color=False)
    assert "by project" in out
    assert out.index("beta") < out.index("alpha")  # beta saved more


def test_dashboard_has_no_project_section_without_projects(tmp_path):
    p = tmp_path / "l.jsonl"
    _rec(p, "", 1000, 400)
    assert "by project" not in ledger.render_dashboard(ledger.summary(p), color=False)


def test_runtime_flush_stamps_distil_project(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_PROJECT", "myrepo")
    p = tmp_path / "l.jsonl"
    rs = RuntimeSavings(ledger_path=p)
    rs.record(1000, 600, "claude-sonnet-4-6")
    assert rs.flush()
    assert json.loads(p.read_text().splitlines()[0])["project"] == "myrepo"
