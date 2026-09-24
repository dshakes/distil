"""Claude Code's MCP tool search must survive the proxy byte-for-byte.

Claude Code turns tool search off behind any non-first-party ANTHROPIC_BASE_URL
because "most proxies don't forward tool_reference blocks"; `distil wrap -- claude`
turns it back on (onboard.AGENT_PRESETS). That is only safe while every piece of the
protocol reaches the provider untouched: the ``defer_loading`` flag on each tool
definition, the ``tool_reference`` blocks inside a ToolSearch result, and the beta
header. A stub upstream records exactly what arrived, with the digest path active.

Also pins the request record: deferred definitions are NOT billed and so are not
overhead, and the per-MCP-server accounting stays names-only.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from distil.ledger import session_requests_path
from distil.proxy import _mcp_record, build_handler

_LOG = "\n".join(f"[2026-09-24 12:00:{i:02d}] INFO worker-{i}: ok depth {i * 3}" for i in range(60))

_DEFERRED = {
    "name": "mcp__claude_ai_Deploys__list_projects",
    "description": "List projects",
    "input_schema": {"type": "object", "properties": {"team": {"type": "string"}}},
    "defer_loading": True,
}
_LOADED_MCP = {
    "name": "mcp__codeindex__find_symbol",
    "description": "Find a symbol",
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}
_BUILTIN = {
    "name": "ToolSearch",
    "description": "Search deferred tools",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}
_REFERENCE = {"type": "tool_reference", "tool_name": "mcp__claude_ai_Deploys__list_projects"}


def _payload() -> dict[str, Any]:
    return {
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "system": "agent",
        "tools": [_BUILTIN, _LOADED_MCP, _DEFERRED],
        "messages": [
            {"role": "user", "content": "deploy status?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "ts1",
                        "name": "ToolSearch",
                        "input": {"query": "v"},
                    },
                    {
                        "type": "tool_use",
                        "id": "sr1",
                        "name": "mcp__codeindex__find_symbol",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "ts1", "content": [_REFERENCE]},
                    {"type": "tool_result", "tool_use_id": "sr1", "content": _LOG},
                ],
            },
            {"role": "user", "content": "next"},
            {"role": "user", "content": "next again"},
            {"role": "user", "content": "and again"},
        ],
    }


class _Capture(BaseHTTPRequestHandler):
    seen: list[tuple[dict[str, str], dict[str, Any]]] = []

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Capture.seen.append(({k.lower(): v for k, v in self.headers.items()}, json.loads(raw)))
        out = json.dumps(
            {
                "id": "m",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture()
def port() -> Any:
    _Capture.seen = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    proxy = ThreadingHTTPServer(
        ("127.0.0.1", 0), build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    yield proxy.server_address[1]
    proxy.shutdown()
    upstream.shutdown()


def _send(port: int) -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(_payload()).encode(),
        headers={"Content-Type": "application/json", "anthropic-beta": "tool-search-2026"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200


def test_tool_search_protocol_reaches_the_provider_untouched(
    port: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_SESSION", "s1-1")
    _send(port)
    headers, body = _Capture.seen[-1]
    assert headers.get("anthropic-beta") == "tool-search-2026"
    # The client's definitions arrive first, in order, byte-identical (distil may
    # append its own distil_expand AFTER them; ADR 0012).
    sent = [json.dumps(t, sort_keys=True) for t in _payload()["tools"]]
    got = [json.dumps(t, sort_keys=True) for t in body["tools"][: len(sent)]]
    assert got == sent
    assert body["tools"][2]["defer_loading"] is True
    # The ToolSearch result keeps its tool_reference block while the neighbouring
    # log in the SAME message was digested — so the compression path really ran.
    results = body["messages"][2]["content"]
    assert results[0]["content"] == [_REFERENCE]
    assert results[1]["content"] != _LOG, "digest did not run; the test proves nothing"


def test_record_bills_only_loaded_definitions_and_names_called_servers(
    port: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_SESSION", "s2-1")
    _send(port)
    path = session_requests_path("s2-1")
    assert path is not None
    rec = None
    for _ in range(250):
        if path.exists() and path.read_text(encoding="utf-8").strip():
            rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            break
        time.sleep(0.02)
    assert rec is not None
    names = {t["name"] for t in rec["tools"]}
    assert _DEFERRED["name"] not in names  # deferred: not in the prompt, not billed
    assert rec["tools_deferred"] == 1
    assert rec["tools_tokens"] == sum(t["tokens"] for t in rec["tools"])
    assert set(rec["mcp_servers"]) == {"mcp__codeindex"}
    assert rec["mcp_called"] == ["mcp__codeindex"]
    # names only: nothing from a schema or an argument reaches the record
    flat = json.dumps(rec)
    assert "Find a symbol" not in flat and '"q"' not in flat


def test_mcp_record_shapes() -> None:
    body = {
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "name": "mcp__a__x"}]},
            {"role": "user", "content": [{"type": "tool_use", "name": "mcp__b__y"}]},  # not a call
            {"role": "assistant", "content": "text only"},
            "garbage",
        ]
    }
    got = _mcp_record(body, {"mcp__a__x": 5, "mcp__a__z": 7, "mcp__b__y": 3, "Bash": 100})
    assert got == {"mcp_servers": {"mcp__a": 12, "mcp__b": 3}, "mcp_called": ["mcp__a"]}
    # no MCP definitions at all -> absent, so "no connectors" != "never called"
    assert _mcp_record(body, {"Bash": 100}) == {}
    assert _mcp_record(None, {"mcp__a__x": 1}) == {"mcp_servers": {"mcp__a": 1}, "mcp_called": []}
    assert _mcp_record(body, {"mcp____bad": 1}) == {}
