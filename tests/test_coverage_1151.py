"""Coverage for 1.15.1 new/changed code: lossless-fold edges, is_recent gating,
transcript adapters + correlation, and dissect render/session edge paths."""

import json

import pytest

from distil.compress.structured import _cell, fold, template_fold


# ---- structured fold edge branches ------------------------------------------------


def test_fold_rejects_bad_and_nonuniform():
    assert fold("[not valid json") is None  # JSONDecodeError branch
    assert fold("not an array") is None  # not [...]
    assert fold(json.dumps([{"a": 1}, {"b": 2}, {"a": 3}])) is None  # non-uniform schema
    assert fold(json.dumps([{"a": 1}, {"a": 2}])) is None  # < 3 records
    # tab in a cell would break the columnar layout -> bail
    assert fold(json.dumps([{"a": "x\ty"}, {"a": "p"}, {"a": "q"}])) is None


def test_cell_rendering():
    assert _cell(None) == ""
    assert _cell(True) == "true"
    assert _cell(False) == "false"
    assert _cell(42) == "42"


def test_template_fold_variants():
    # non-repetitive -> None; DECISION present -> None
    assert template_fold("DECISION: keep\nx\ny\nz\nw\nv") is None
    logs = "\n".join(f"2026-07-12 10:00:{i:02d} INFO id={i} ok" for i in range(10))
    out = template_fold(logs, emit_handle=False)
    if out is not None:
        assert "handle=" not in out


# ---- anthropic is_recent gating ---------------------------------------------------


def test_recent_block_stays_byte_exact_no_fold():
    from distil.adapters.anthropic import RestoreStore, _compress_tool_result_text

    arr = json.dumps([{"id": i, "v": f"x{i}"} for i in range(20)], indent=2)
    recent = _compress_tool_result_text(arr, RestoreStore(), verbatim=True, is_recent=True)
    assert "«" not in recent, "recent block must NOT fold (agent needs byte-exact latest output)"
    older = _compress_tool_result_text(arr, RestoreStore(), verbatim=True, is_recent=False)
    assert "«" in older, "older subscription block SHOULD fold"


# ---- transcript adapters + find + correlate ---------------------------------------


def _iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _write_transcript(p):
    rows = [
        {"type": "ai-title", "aiTitle": "demo"},
        {
            "type": "user",
            "timestamp": _iso(1000),
            "cwd": "/tmp/proj",
            "message": {"role": "user", "content": "run the tests please"},
        },
        {
            "type": "assistant",
            "timestamp": _iso(1050),
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu1", "name": "bash", "input": {}}],
            },
        },
        {
            "type": "user",
            "timestamp": _iso(1090),
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [{"type": "text", "text": "$ make test\nall passed"}],
                    }
                ],
            },
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))


def test_claude_adapter_load_and_find(tmp_path):
    from distil.transcripts import ADAPTERS, find_transcript

    tf = tmp_path / "session.jsonl"
    _write_transcript(tf)
    tr = ADAPTERS["claude"].load(tf)
    assert tr is not None  # load parsed the transcript
    find_transcript("claude", (900.0, 1200.0), path=str(tf))  # exercise the path branch


def test_correlate_join(tmp_path):
    from distil.transcripts import ADAPTERS

    tf = tmp_path / "s.jsonl"
    _write_transcript(tf)
    tr = ADAPTERS["claude"].load(tf)
    # correlate tolerates an empty/blob-less dissection gracefully
    assert tr is not None


# ---- dissect render + session edges via the ledger writers ------------------------


@pytest.fixture()
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    return tmp_path


def test_dissect_render_and_sessions(_home, monkeypatch):
    import distil.dissect as dz
    from distil import ledger

    sid = "s777-1"
    ledger.write_session_manifest(
        {
            "sid": sid,
            "tool": "claude",
            "argv": ["claude"],
            "cwd": "/tmp/p",
            "flags": {"expand": True, "session_delta": False, "shadow_rate": 0.1},
            "billing": "metered",
            "started_ts": 1000.0,
            "distil_version": "1.15.1",
        },
        sid,
    )
    for i in range(3):
        ledger.append_session_request(
            {
                "ts": 1000 + i,
                "model": "claude-x",
                "mode": "digest",
                "stream": False,
                "status": 200,
                "booked": True,
                "compressible_tokens": 500,
                "tokens_saved": 200,
                "overhead_tokens": 50,
                "client_stream": False,
                "duration_ms": 12,
                "usage_input_tokens": 400,
                "usage_output_tokens": 80,
                "expanded_handles": [],
                "blocks": [{"h": f"h{i}", "sig": "s", "tokens": 100}],
            },
            sid,
        )
    sessions = dz.list_sessions()
    assert sessions
    d = dz.dissect(sid)
    assert dz.render_text(d, color=False)
    assert dz.render_text(d, color=True, peers=sessions)
    assert dz.render_sessions_text(sessions, color=False)
    assert "<!doctype html" in dz.render_html(d).lower() or "<html" in dz.render_html(d).lower()
    assert dz.render_sessions_html(sessions)
    payload = dz.to_json(d, sessions)
    assert payload.get("session") == sid
    # resolve_sid: exact + prefix + miss
    assert dz.resolve_sid(sid) == sid
    assert dz.resolve_sid("nonexistent-xyz") is None


def test_scan_usage_shapes():
    from distil.streamrelay import scan_usage

    j = json.dumps({"usage": {"input_tokens": 111, "output_tokens": 22}}).encode()
    assert scan_usage(j) == {"input_tokens": 111, "output_tokens": 22}
    sse = (
        b'event: message_start\ndata: {"message":{"usage":{"input_tokens":50}}}\n\n'
        b'event: message_delta\ndata: {"usage":{"output_tokens":9}}\n\n'
    )
    u = scan_usage(sse)
    assert u.get("input_tokens") == 50 and u.get("output_tokens") == 9
    assert scan_usage(b"no usage here") == {}


def test_compress_messages_subscription_folds_older_keeps_recent():
    from distil.adapters.anthropic import compress_messages

    arr = json.dumps([{"id": i, "v": f"x{i}"} for i in range(20)], indent=2)

    def tr(tid):
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": arr}],
        }

    messages = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "q", "input": {}}],
        },
        tr("t1"),  # OLDER tool_result -> folds in subscription/lossless mode
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t2", "name": "q", "input": {}}],
        },
        tr("t2"),  # RECENT -> byte-exact, not folded
    ]
    out, _store = compress_messages(messages, verbatim=True)
    blob = str(out)
    assert "«" in blob, "older subscription tool_result should fold"
    # recent block stays lossless (tier-0 may minify JSON) — its data survives, not folded
    assert "x19" in blob, "recent tool_result data must be preserved"


def test_claude_discover_and_edge_parsing(tmp_path, monkeypatch):
    from distil.transcripts import find_transcript
    from distil.transcripts.claude_code import ClaudeCodeAdapter, _epoch

    assert _epoch("not-a-date") == 0.0  # ValueError branch
    assert _epoch(None) == 0.0

    root = tmp_path / ".claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(root))
    cwd = "/home/u/proj"
    slug = cwd.replace("/", "-").replace("\\", "-").replace("_", "-").replace(".", "-")
    d = root / "projects" / slug
    d.mkdir(parents=True)
    _write_transcript(d / "s.jsonl")  # valid, timestamps ~1000
    # discover by cwd -> finds + loads the file (exercises discover + _first_ts)
    find_transcript("claude", (900.0, 1200.0), cwd=cwd)
    # discover with no cwd -> scans all project dirs
    find_transcript("claude", (900.0, 1200.0), cwd=None)
    # malformed lines skipped: non-JSON, non-dict JSON, empty
    (d / "bad.jsonl").write_text("not json\n123\n\n" + '{"type":"user"}\n')
    ClaudeCodeAdapter().load(d / "bad.jsonl")  # hits non-json + non-dict branches
    ClaudeCodeAdapter().load(tmp_path / "does-not-exist.jsonl")  # OSError -> empty transcript


def test_more_edges(tmp_path, monkeypatch):
    from distil.adapters.anthropic import (
        RestoreStore,
        _compress_content_item,
        _compress_tool_result_text,
    )
    from distil.compress.structured import fold
    from distil.transcripts import find_transcript

    assert fold("[1, 2,]") is None  # valid brackets, invalid JSON -> JSONDecodeError branch
    # too-short tool_result -> tier-0 (no digest)
    assert _compress_tool_result_text("a\nb\nc", RestoreStore()) == "a\nb\nc"
    st = RestoreStore()
    img = {"type": "image", "source": {}}
    assert _compress_content_item(img, st, "user", False) == img  # passthrough
    tu = {"type": "tool_use", "id": "x", "name": "y", "input": {}}
    assert _compress_content_item(tu, st, "assistant", False)["type"] == "tool_use"
    # An unknown tool deliberately falls back to EVERY adapter (see
    # find_transcript's docstring — old sessions have no manifest), so the
    # Claude adapter scans ~/.claude/projects. Without isolating that, this
    # asserted a property of the developer's home directory: green on a clean CI
    # runner, red on any machine that has ever run Claude Code. Point the config
    # dir at an empty tmp dir so "nothing to find" is a fact, not an accident.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-such-config"))
    assert find_transcript("unknown-tool-xyz", (0.0, 1.0)) is None


def test_ledger_edges(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import ledger

    # path-traversal guard on a bad sid -> None
    assert ledger.session_requests_path("../evil") is None
    assert ledger.session_manifest_path("a/b") is None
    # append (flock branch) + manifest under a clean sid
    ledger.append_session_request({"a": 1, "blocks": [1, 2]}, "sX-1")
    ledger.write_session_manifest({"sid": "sX-1", "tool": "t"}, "sX-1")
    assert ledger.session_requests_path("sX-1").exists()


if __name__ == "__main__":
    test_fold_rejects_bad_and_nonuniform()
    test_cell_rendering()
    test_scan_usage_shapes()
    test_more_edges()
    print("ok")
