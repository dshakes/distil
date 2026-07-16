"""Minimal, zero-dependency MCP server for distil.

Exposes distil's reversible compression to any MCP client (Claude Desktop, IDEs,
agents) over stdio JSON-RPC 2.0 — **stdlib only**, no third-party SDK, so it keeps
distil's zero-runtime-deps promise.

Tools
-----
* ``distil_compress(text)`` — reversibly digest a blob; returns the digest, an
  8-hex handle, and tokens saved. The original is kept in a local on-disk store
  (never returned to the model until asked), so it costs zero tokens on the wire.
* ``distil_expand(handle)`` — return the original text for a handle.
* ``distil_savings()`` — cumulative savings from the local ledger.

Run
---
``distil mcp``  (or ``python -m distil.mcp_server``). Wire it into an MCP client's
server config as a stdio command. The protocol is newline-delimited JSON-RPC 2.0;
``handle_message`` is a pure function (testable without real stdio).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import atrest
from .compress.tier1 import _handle, digest
from .tokenizer import DEFAULT as _tokenizer

SERVER_NAME = "distil"
DEFAULT_PROTOCOL = "2025-06-18"


# ---------------------------------------------------------------------------
# Persistent handle store (so expand works across calls / processes)
# ---------------------------------------------------------------------------


def _store_path() -> Path:
    import os

    base = Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))
    return base / "mcp_store.json"


try:
    import fcntl  # POSIX advisory locking; absent on Windows

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    _HAVE_FCNTL = False


def _store_add(handle: str, text: str) -> None:
    """Read-modify-write the store under an advisory lock.

    Two concurrent ``distil_compress`` calls (e.g. two agent sessions sharing
    this MCP server's store) would otherwise race load/load/save/save and
    silently drop one handle — a later ``distil_expand`` on it then fails.
    Locks a sidecar file so the save itself can stay a simple rewrite.
    """
    lock_path = _store_path().with_suffix(".lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lk:
            if _HAVE_FCNTL:
                fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
            store = _load_store()
            store[handle] = text
            _save_store(store)
    except OSError:
        pass  # best-effort; never crash a tool call


def _load_store() -> dict[str, str]:
    p = _store_path()
    try:
        raw = p.read_bytes()
        decrypted = atrest.decrypt_bytes(raw)
        if decrypted is None:
            return {}  # auth failure — treat as missing (fail-open)
        return json.loads(decrypted.decode())
    except (OSError, json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return {}


# The store holds ORIGINAL tool-output content (that's its purpose — expand
# must survive across processes), so it is bounded and owner-readable only.
_MAX_STORE_ENTRIES = 512


def _save_store(store: dict[str, str]) -> None:
    p = _store_path()
    try:
        # FIFO-bound the store so it can't grow without limit across sessions
        # (dict preserves insertion order; oldest handles age out first).
        while len(store) > _MAX_STORE_ENTRIES:
            store.pop(next(iter(store)))
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(store).encode()
        p.write_bytes(atrest.encrypt_bytes(raw))
        p.chmod(0o600)  # encrypted content at rest — owner-only
    except OSError:
        pass  # best-effort; never crash a tool call


_RESTORE_CAP = (
    500  # ponytail: FIFO-by-mtime cap; raise or make configurable if long sessions outgrow it
)
# Age cap on top of the count cap: digest originals are real agent content
# (can include secrets/PII), so a low-traffic store must not hold them forever.
# 0 disables. Expired handles simply fail to expand — same as capped-out ones.
_RESTORE_TTL_DAYS = float(os.environ.get("DISTIL_RESTORE_TTL_DAYS", "14") or 0)
_HANDLE_RE = re.compile(r"[0-9a-f]{8}")


def _restore_dir() -> Path:
    return _store_path().parent / "restore"


def _read_restore_text(p: Path) -> str | None:
    """Read a restore file (encrypted or legacy plaintext).

    Returns the plaintext string, or None on I/O error or authentication
    failure. Handles both the new DSTL1 format and pre-encryption legacy files
    (which atrest.decrypt_bytes passes through unchanged).
    """
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    decrypted = atrest.decrypt_bytes(raw)
    if decrypted is None:
        return None  # authentication failure — treat as missing
    try:
        return decrypted.decode("utf-8")
    except UnicodeDecodeError:
        return None


def record_restore(handle: str, original: str) -> None:
    """Persist a digest original to disk so handles survive proxy restarts/upgrades
    and can be expanded from other processes (e.g. this MCP server)."""
    if not _HANDLE_RE.fullmatch(handle):
        return
    try:
        d = _restore_dir()
        d.mkdir(parents=True, exist_ok=True)
        p = d / handle
        # Collision guard, mirroring RestoreStore._record's in-memory check: if this
        # handle already maps to *different* bytes on disk, a 32-bit handle collided
        # across sessions. Do NOT clobber the earlier block — its stub would then expand
        # to the wrong content. Keep the first writer; skip the second.
        # When existing content matches, fall through and rewrite — this refreshes mtime
        # for the TTL sweep AND upgrades legacy plaintext files to encrypted format.
        if p.exists():
            existing = _read_restore_text(p)
            if existing is not None and existing != original:
                return  # genuine collision — keep first writer
            # existing is None (auth failure/corrupt) or same content → rewrite
        p.write_bytes(atrest.encrypt_bytes(original.encode("utf-8")))
        p.chmod(0o600)  # encrypted content at rest — owner-only
        stale = sorted(d.iterdir(), key=lambda f: f.stat().st_mtime)[:-_RESTORE_CAP]
        if _RESTORE_TTL_DAYS > 0:
            cutoff = time.time() - _RESTORE_TTL_DAYS * 86400
            fresh = sorted(d.iterdir(), key=lambda f: f.stat().st_mtime)[-_RESTORE_CAP:]
            stale += [f for f in fresh if f.stat().st_mtime < cutoff]
        for old in stale:
            old.unlink()
    except OSError:
        pass  # best-effort; never crash a compress call


def load_restore(handle: str) -> str | None:
    """Return the persisted original for *handle*, or None."""
    if not _HANDLE_RE.fullmatch(handle):  # untrusted MCP arg — no path traversal
        return None
    return _read_restore_text(_restore_dir() / handle)


# ---------------------------------------------------------------------------
# Tool catalog + implementations
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "distil_compress",
        "description": (
            "Reversibly compress a text blob (e.g. a large tool output) with distil. "
            "Returns a compact digest, an 8-hex handle, and the tokens saved. The "
            "original is kept locally and can be recovered with distil_expand."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "the text to compress"}},
            "required": ["text"],
        },
    },
    {
        "name": "distil_expand",
        "description": "Return the original text for a handle produced by distil_compress.",
        "inputSchema": {
            "type": "object",
            "properties": {"handle": {"type": "string", "description": "the 8-hex content handle"}},
            "required": ["handle"],
        },
    },
    {
        "name": "distil_savings",
        "description": "Report cumulative token/dollar savings from the local distil ledger.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _tool_compress(args: dict[str, Any]) -> str:
    text = args.get("text")
    if not isinstance(text, str):
        return "error: 'text' must be a string"
    digested, changed = digest(text)
    if not changed:
        return json.dumps({"compressed": text, "handle": None, "tokens_saved": 0})
    h = _handle(text)
    _store_add(h, text)
    saved = max(0, _tokenizer.count(text) - _tokenizer.count(digested))
    return json.dumps({"compressed": digested, "handle": h, "tokens_saved": saved})


def _tool_expand(args: dict[str, Any]) -> str:
    handle = args.get("handle")
    if not isinstance(handle, str):
        return "error: 'handle' must be a string"
    original = _load_store().get(handle)
    if original is None:
        original = load_restore(handle)  # proxy-side digests persisted by record_restore
    if original is None:
        return f"error: no original found for handle {handle!r}"
    return original


def _tool_savings(_args: dict[str, Any]) -> str:
    from . import ledger

    s = ledger.summary()
    return json.dumps(
        {
            "runs": s.runs,
            "tokens_saved": s.total_tokens_saved,
            "dollars_saved": round(s.total_dollars_saved, 6),
        }
    )


_DISPATCH = {
    "distil_compress": _tool_compress,
    "distil_expand": _tool_expand,
    "distil_savings": _tool_savings,
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 message handling (pure — unit-testable)
# ---------------------------------------------------------------------------


def _result(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message; return a response dict, or None for notifications."""
    method = msg.get("method")
    msg_id = msg.get("id")

    # Notifications (no id) get no response.
    if method is not None and msg_id is None:
        return None

    if method == "initialize":
        requested = (msg.get("params") or {}).get("protocolVersion")
        return _result(
            msg_id,
            {
                "protocolVersion": requested or DEFAULT_PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": _server_version()},
            },
        )

    if method == "tools/list":
        return _result(msg_id, {"tools": TOOLS})

    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name") or ""
        args = params.get("arguments") or {}
        fn = _DISPATCH.get(name)
        if fn is None:
            return _error(msg_id, -32602, f"unknown tool: {name!r}")
        try:
            text = fn(args)
            is_error = isinstance(text, str) and text.startswith("error:")
            return _result(
                msg_id, {"content": [{"type": "text", "text": text}], "isError": is_error}
            )
        except Exception as exc:  # noqa: BLE001 — surface as a tool error, never crash the server
            return _result(
                msg_id,
                {"content": [{"type": "text", "text": f"error: {exc}"}], "isError": True},
            )

    if method == "ping":
        return _result(msg_id, {})

    return _error(msg_id, -32601, f"method not found: {method!r}")


def _server_version() -> str:
    from . import __version__

    return __version__


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------


def serve(stdin: Any = None, stdout: Any = None) -> None:
    """Run the newline-delimited JSON-RPC 2.0 stdio loop until EOF."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue  # skip unparseable input rather than crash
        response = handle_message(msg)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    serve()
