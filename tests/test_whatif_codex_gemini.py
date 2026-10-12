"""Codex CLI and Gemini CLI readers: transcript adapters + the what-if replay.

Fixtures are synthetic but structurally exact (Codex rollout JSONL, Gemini CLI JSONL with
mutation lines, and the legacy Gemini ``.json``). conftest sandboxes HOME / DISTIL_HOME.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from distil import pricing, whatif
from distil.transcripts import ADAPTERS, find_transcript

T0 = 1_790_000_000.0
SECRET = "SECRET-LOG-LINE-9f2c"
TURNS = 8


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _log(i: int) -> str:
    return "\n".join(
        f"2026-10-0{1 + j % 3}T12:{j % 60:02d}:00 INFO worker-{j % 7} {SECRET} step={i}.{j} "
        f"latency_ms={(j * 37 + i) % 997} path=/srv/app/module_{j % 13}.py ok={j % 5 != 0}"
        for j in range(150)
    )


def _jl(rows: list[Any]) -> str:
    return "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n"


def _codex_rows(cwd: str = "/work/proj") -> list[Any]:
    def rec(i: float, tag: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"timestamp": _iso(T0 + i), "type": tag, "payload": payload}

    rows: list[Any] = [
        rec(0, "session_meta", {"id": "u", "timestamp": _iso(T0), "cwd": cwd, "cli_version": "1"}),
        rec(0, "turn_context", {"turn_id": "t", "cwd": cwd, "model": "gpt-5.2", "effort": "high"}),
        rec(0, "event_msg", {"type": "token_count", "info": None, "rate_limits": None}),
        rec(0, "world_state", {"anything": 1}),  # unknown tag: ignored
        "{not json",
        rec(
            1,
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "<environment_context>cwd</environment_context>"}
                ],
            },
        ),
    ]
    for i in range(TURNS):
        t = 10.0 * (i + 1)
        out: Any = _log(i) if i % 2 == 0 else [{"type": "input_text", "text": _log(i)}]
        rows += [
            rec(
                t,
                "response_item",
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"please run step {i}"}],
                },
            ),
            rec(
                t + 1,
                "response_item",
                {"type": "reasoning", "summary": [], "encrypted_content": "x"},
            ),
            rec(
                t + 2,
                "response_item",
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps({"command": ["ls"]}),
                    "call_id": f"c{i}",
                },
            ),
            rec(
                t + 3,
                "event_msg",
                {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {},
                        "model_context_window": 1,
                        "last_token_usage": {
                            "input_tokens": 1000 + i,
                            "cached_input_tokens": 600,
                            "output_tokens": 5,
                            "reasoning_output_tokens": 0,
                            "total_tokens": 1005,
                        },
                    },
                },
            ),
            rec(
                t + 4,
                "response_item",
                {"type": "function_call_output", "call_id": f"c{i}", "output": out},
            ),
            rec(
                t + 5,
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"done {i}"}],
                },
            ),
            rec(
                t + 6,
                "event_msg",
                {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 3000 + i,
                            "cached_input_tokens": 2000,
                            "output_tokens": 5,
                            "total_tokens": 3005,
                        }
                    },
                },
            ),
        ]
    return rows


def _gemini_msgs() -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    for i in range(TURNS):
        t = T0 + 10.0 * (i + 1)
        msgs += [
            {
                "id": f"u{i}",
                "timestamp": _iso(t),
                "type": "user",
                "content": [{"text": f"run step {i}"}],
            },
            {
                "id": f"g{i}",
                "timestamp": _iso(t + 1),
                "type": "gemini",
                "content": "",
                "model": "gemini-2.5-pro",
                "tokens": {"input": 1000 + i, "output": 5, "cached": 400, "total": 1005},
                "toolCalls": [
                    {
                        "id": f"tc{i}",
                        "name": "run_shell_command",
                        "args": {"command": "ls"},
                        "status": "success",
                        "timestamp": _iso(t + 1),
                        "result": [
                            {
                                "functionResponse": {
                                    "id": f"tc{i}",
                                    "name": "run_shell_command",
                                    "response": {"output": _log(i)},
                                }
                            }
                        ],
                    }
                ],
            },
            {"id": f"i{i}", "timestamp": _iso(t + 2), "type": "info", "content": "ignored"},
            {
                "id": f"d{i}",
                "timestamp": _iso(t + 3),
                "type": "gemini",
                "content": f"done {i}",
                "model": "gemini-2.5-pro",
                "tokens": {"input": 2000, "output": 5, "cached": 1500, "total": 2005},
            },
        ]
    return msgs


@pytest.fixture
def codex_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    d = tmp_path / "codex" / "sessions" / "2026" / "10" / "01"
    d.mkdir(parents=True)
    f = d / "rollout-2026-10-01T12-00-00-abc.jsonl"
    f.write_text(_jl(_codex_rows()))
    os.utime(f, (T0 + 500, T0 + 500))
    return f


@pytest.fixture
def gemini_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path / "ghome"))
    d = tmp_path / "ghome" / ".gemini" / "tmp" / "proj" / "chats"
    d.mkdir(parents=True)
    meta = {
        "sessionId": "s",
        "projectHash": "h",
        "startTime": _iso(T0),
        "lastUpdated": _iso(T0 + 90),
        "kind": "main",
    }
    msgs = _gemini_msgs()
    rows: list[Any] = [meta, *msgs[:4], {"$set": {"summary": "x"}}, "{bad", {"$patch": {}}]
    rows += [{**msgs[3], "content": "done 0 (updated)"}, *msgs[4:]]
    # a rewind drops d5 and everything after it (upstream splice(idx)), as a /rewind does
    rows += [{"$rewindTo": "d5"}]
    f = d / "session-2026-10-01T12-00-abcd1234.jsonl"
    f.write_text(_jl(rows))
    os.utime(f, (T0 + 500, T0 + 500))
    return f


def test_registered_under_manifest_names():
    assert {"claude", "codex", "gemini"} <= set(ADAPTERS)


def test_codex_load_and_discover(codex_file: Path):
    a = ADAPTERS["codex"]
    tr = a.load(codex_file)
    assert tr.agent == "codex" and tr.cwd == "/work/proj"
    assert [t.text for t in tr.turns][:2] == ["please run step 0", "please run step 1"]
    assert len(tr.turns) == TURNS  # environment_context is not a human turn
    assert len(tr.tool_calls) == TURNS and tr.tool_calls[0].name == "shell"
    assert len(tr.tool_results) == TURNS  # string AND array outputs
    assert all(r.tool == "shell" and SECRET in r.text for r in tr.tool_results)
    window = (T0, T0 + 600)
    assert a.discover(window, "/work/proj") == [codex_file]
    assert a.discover(window, "/elsewhere") == [codex_file]  # no cwd match: not emptied
    assert a.discover((T0 + 10_000, T0 + 11_000), None) == []
    assert find_transcript("codex", window, "/work/proj") is not None
    assert a.load(codex_file.parent / "missing.jsonl").turns == []


def test_codex_session_maps_usage(codex_file: Path):
    sess = whatif._read(codex_file, "responses")
    assert sess.shape == "responses" and sess.model == "gpt-5.2"
    assert len(sess.ends or []) == 2 * TURNS  # token_count info=None starts nothing
    # OpenAI input_tokens includes cached ones: billed is the input, no writes invented
    assert sorted(sess.billed.values())[:2] == [1000, 1001] and sess.cache_write == 0
    for e in sess.ends or []:
        assert e + 1 in sess.billed
    assert all(m.get("type") != "reasoning" for m in sess.messages)


def test_codex_cache_write_only_when_reported(tmp_path: Path):
    rows = _codex_rows()
    for r in rows:
        u = (
            (r.get("payload", {}).get("info") or {}).get("last_token_usage")
            if isinstance(r, dict)
            else None
        )
        if u:
            u["cache_write_input_tokens"] = 50
            break
    f = tmp_path / "rollout-x.jsonl"
    f.write_text(_jl(rows))
    sess = whatif._read(f, "responses")
    assert sess.cache_write == 50 and 1050 in sess.billed.values()


def test_gemini_load_discover_rewind(gemini_file: Path):
    a = ADAPTERS["gemini"]
    tr = a.load(gemini_file)
    assert tr.agent == "gemini" and tr.started == pytest.approx(T0)
    # rewound to d5: u0..u5 + g0..g5 kept (6 turns), d5 and later dropped
    assert len(tr.turns) == 6 and tr.turns[0].text == "run step 0"
    assert len(tr.tool_calls) == 6 and tr.tool_calls[0].name == "run_shell_command"
    assert len(tr.tool_results) == 6 and SECRET in tr.tool_results[0].text
    assert a.discover((T0, T0 + 600), None) == [gemini_file]
    assert a.discover((T0 + 10_000, T0 + 11_000), None) == []


def test_gemini_legacy_json(tmp_path: Path):
    f = tmp_path / "session-2026-10-01T12-00-ffff0000.json"
    f.write_text(
        json.dumps(
            {
                "sessionId": "s",
                "projectHash": "h",
                "startTime": _iso(T0),
                "lastUpdated": _iso(T0 + 9),
                "messages": [*_gemini_msgs()[:4], "junk"],
            }
        )
    )
    tr = ADAPTERS["gemini"].load(f)
    assert len(tr.turns) == 1 and len(tr.tool_results) == 1
    sess = whatif._read(f, "gemini")
    assert [m["role"] for m in sess.messages] == ["user", "model", "user", "model"]
    assert sess.billed == {1: 1000, 3: 2000}  # input includes cached; keyed by the response
    assert sess.ends == [0, 2] and sess.model == "gemini-2.5-pro"
    f.write_text("{not json")
    assert ADAPTERS["gemini"].load(f).turns == []


def _check_replay(w: whatif.WhatIf, model: str) -> None:
    assert w.sessions == 1 and w.failed_sessions == 0
    assert w.requests_replayed > 0 and w.billed_input_tokens > 0
    assert w.digest.removed > 0  # the digest took the old log-shaped outputs
    assert w.digest.removed >= w.lossless.removed >= 0
    # priced iff the catalog knows the model; unpriced keeps tokens, never invents dollars
    assert (w.unpriced_requests == 0) == (pricing.resolve(model) is not None)
    assert (w.digest.usd_before > 0) == (pricing.resolve(model) is not None)


def test_codex_replay_through_openai_adapter(codex_file: Path, monkeypatch: pytest.MonkeyPatch):
    from distil.adapters import openai

    seen: list[list[dict[str, Any]]] = []
    real = openai.compress_responses_input

    def spy(items: list[dict[str, Any]], **kw: Any) -> Any:
        out = real(items, **kw)
        seen.append(out[0])
        return out

    monkeypatch.setattr(openai, "compress_responses_input", spy)
    _check_replay(whatif.run(None), "gpt-5.2")
    digested = [
        it
        for it in seen[-1]
        if it.get("type") == "function_call_output" and "handle=" in str(it["output"])
    ]
    assert digested
    assert not any("cache_control" in json.dumps(req) for req in seen)


def test_gemini_replay_through_gemini_adapter(gemini_file: Path, monkeypatch: pytest.MonkeyPatch):
    from distil.adapters import gemini

    seen: list[str] = []
    real = gemini.compress_generate_request

    def spy(body: dict[str, Any], **kw: Any) -> Any:
        out = real(body, **kw)
        seen.append(json.dumps(out[0]))
        return out

    monkeypatch.setattr(gemini, "compress_generate_request", spy)
    _check_replay(whatif.run(None), "gemini-2.5-pro")
    assert any("handle=" in s for s in seen)
    assert not any("cache_control" in s for s in seen)


def test_replay_is_read_only(codex_file: Path, gemini_file: Path):
    whatif.run(None)
    assert not (Path(os.environ["DISTIL_HOME"]) / "restore").exists()
    from distil.adapters import gemini, openai

    assert openai.RestoreStore.__name__ == gemini.RestoreStore.__name__ == "RestoreStore"
