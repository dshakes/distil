"""A stdio MCP server that serves a vendored fixture catalog with canned results.

For the proxy's integration tests, the bench's dry run, and trying ``distil mcp`` out
without installing anything::

    distil mcp wrap --name git --level L2 -- python -m distil.mcpproxy.fakeserver git

Results are deterministic and synthetic: listing-shaped tools return a long listing
(so result compression has something to do), everything else echoes its arguments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import IO, Any

FIXTURES = Path(__file__).with_name("fixtures")


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


def load_fixture(name: str) -> dict[str, Any]:
    if name not in fixture_names():
        raise ValueError(f"no fixture {name!r}; have {', '.join(fixture_names())}")
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return data


def canned_result(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    if any(w in tool for w in ("list", "search", "log", "tree")):
        rows = [f"entry-{i:04d}  {tool}  size={i * 37 % 9973}  ok" for i in range(400)]
        rows.insert(200, "ERROR: entry-0200 failed checksum")
        return {"content": [{"type": "text", "text": "\n".join(rows)}]}
    return {"content": [{"type": "text", "text": f"{tool} ok: {json.dumps(args, sort_keys=True)}"}]}


def handle(fixture: dict[str, Any], msg: dict[str, Any]) -> dict[str, Any] | None:
    method, msg_id = msg.get("method"), msg.get("id")
    if msg_id is None:
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": (msg.get("params") or {}).get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": fixture.get("serverInfo")
                or {"name": fixture["server"], "version": "0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": fixture["tools"]}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        if name not in {t["name"] for t in fixture["tools"]}:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": f"unknown tool {name!r}"},
            }
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": canned_result(str(name), params.get("arguments") or {}),
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"method not found: {method!r}"},
    }


def serve(name: str, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
    fixture = load_fixture(name)
    src, dst = stdin or sys.stdin, stdout or sys.stdout
    for line in src:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        out = handle(fixture, msg) if isinstance(msg, dict) else None
        if out is not None:
            dst.write(json.dumps(out) + "\n")
            dst.flush()


if __name__ == "__main__":  # pragma: no cover — exercised as a subprocess
    serve(sys.argv[1] if len(sys.argv) > 1 else "git")
