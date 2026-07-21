"""Local real-time dashboard: snapshot is content-free; page has no external calls."""

from __future__ import annotations

import json

from distil import webdash


def test_snapshot_is_content_free_and_correct(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    (tmp_path / "savings.jsonl").write_text(
        json.dumps(
            {
                "trajectory_id": "t",
                "model": "m",
                "turns": 1,
                "baseline_input_tokens": 1000,
                "distil_input_tokens": 400,
                "baseline_dollars": 1.0,
                "distil_dollars": 0.4,
                "tokenizer": "heuristic",
                "ts": 1.0,
                "acct": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    s = webdash._snapshot()
    assert set(s) == {
        "tokens_saved",
        "dollars_saved",
        "pct",
        "runs",
        "baseline_tokens",
        "distil_tokens",
        "equivalence",
        "subscription",
        "ts",
    }
    assert s["tokens_saved"] == 600 and s["pct"] == 60.0 and s["runs"] == 1
    assert set(s["equivalence"]) == {"pct", "shadowed"}
    # numbers / bools only — nothing that can carry prompt or path content
    for k, v in s.items():
        if k == "equivalence":
            continue
        assert isinstance(v, (int, float, bool)), k


def test_page_is_self_contained_local_only():
    # the served page must never call out to a remote host (local-only promise)
    assert "http://" not in webdash._PAGE and "https://" not in webdash._PAGE
    assert 'fetch("/data"' in webdash._PAGE  # reads the LOCAL endpoint only
    assert "nothing leaves this machine" in webdash._PAGE
