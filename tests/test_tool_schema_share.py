"""benchmarks/tool_schema_share.py — the measurement behind ADR 0012."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.tool_schema_share import _compact, _load_tools, lossless_headroom, measure


def test_compact_drops_only_provably_lossless_keys() -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "repo_path": {"type": "string", "title": "Repo Path", "description": "d"},
            "q": {"type": "string", "title": "Search query"},  # adds meaning: kept
            "nested": {
                "type": "object",
                "$schema": "x",
                "properties": {"id": {"type": "integer", "title": "ID"}},
            },
        },
    }
    out = _compact(schema)
    assert "$schema" not in out and "$schema" not in out["properties"]["nested"]
    assert out["additionalProperties"] is False  # strict tool use needs it
    assert "title" not in out["properties"]["repo_path"]
    assert out["properties"]["repo_path"]["description"] == "d"
    assert out["properties"]["q"]["title"] == "Search query"
    assert "title" not in out["properties"]["nested"]["properties"]["id"]
    assert _compact(out) == out  # idempotent
    assert "$schema" in schema  # input not mutated


def test_headroom_is_zero_on_already_minimal_tools() -> None:
    tools = [{"name": "t", "description": "x", "input_schema": {"type": "object"}}]
    assert lossless_headroom(tools)["lossless_reduction"] == 0.0
    assert lossless_headroom([])["lossless_reduction"] == 0.0


def test_load_tools_json_jsonl_and_garbage(tmp_path: Path) -> None:
    tools = [{"name": "t", "input_schema": {}}]
    (tmp_path / "a.json").write_text(json.dumps({"tools": tools}))
    (tmp_path / "b.jsonl").write_text(json.dumps({"tools": tools}) + "\n{}\n")
    (tmp_path / "c.json").write_text(json.dumps({"no": 1}))
    assert _load_tools(tmp_path / "a.json") == tools
    assert _load_tools(tmp_path / "b.jsonl") == tools
    with pytest.raises(ValueError):
        _load_tools(tmp_path / "c.json")


def test_measure_prices_tools_as_prefix_cache_reads(tmp_path: Path) -> None:
    s = tmp_path / "sessions"
    s.mkdir()
    good = {
        "status": 200,
        "model": "claude-opus-4-8",
        "usage_input_tokens": 0,
        "usage_cache_read": 900,
        "usage_cache_create": 100,
        "usage_output_tokens": 0,
        "overhead_tokens": 500,  # all tools
        "tools_tokens": 500,
        "compressible_tokens": 500,
        "tokens_saved": 0,
    }
    lines = [good, {**good, "status": 529}, {**good, "usage_input_tokens": None}]
    (s / "x.requests.jsonl").write_text(
        "\n".join(json.dumps(r) for r in lines) + "\nnot json\n", encoding="utf-8"
    )
    res = measure(tmp_path)
    assert res["requests"] == 1  # non-200, no-usage and corrupt lines skipped
    assert res["tool_share_of_input_tokens"] == 0.5
    assert res["tool_tokens_cache_read_fraction"] == 1.0  # tools lead the prefix
    # tools: 500 reads at 0.1x; total: 900 reads + 100 writes at 1.25x
    assert res["tool_share_of_billed_usd"] == pytest.approx(50 / 215, abs=1e-4)


def test_measure_empty_root(tmp_path: Path) -> None:
    res = measure(tmp_path)
    assert res["requests"] == 0 and res["tool_share_of_billed_usd"] == 0.0


def test_main_combines_share_and_headroom_into_the_deciding_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The ADR's headline (share of bill a perfect lossless compactor saves) is the
    # product main() computes — pin it end to end, not just its two inputs.
    import benchmarks.tool_schema_share as T

    test_measure_prices_tools_as_prefix_cache_reads(tmp_path)  # writes the sessions fixture
    tools = [{"name": "t", "input_schema": {"$schema": "x" * 400, "type": "object"}}]
    (tmp_path / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        ["x", "--root", str(tmp_path), "--tools", str(tmp_path / "tools.json"), "--out", str(out)],
    )
    T.main()
    capsys.readouterr()
    res = json.loads(out.read_text(encoding="utf-8"))
    red = res["lossless_headroom"]["lossless_reduction"]
    assert red > 0
    assert res["ev_share_of_billed_usd_lossless"] == round(res["tool_share_of_billed_usd"] * red, 4)
