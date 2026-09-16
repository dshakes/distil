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

import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import _filelock, atrest
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


def _store_add(handle: str, text: str) -> None:
    """Read-modify-write the store under an advisory lock.

    Two concurrent ``distil_compress`` calls (e.g. two agent sessions sharing
    this MCP server's store) would otherwise race load/load/save/save and
    silently drop one handle — a later ``distil_expand`` on it then fails.
    Locks a sidecar file (cross-platform, see ``_filelock``) so the save
    itself can stay a simple rewrite.
    """
    try:
        with _filelock.locked(_store_path()):
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
        # Owner-only AT CREATION, not by a chmod after the write: chmod-ing
        # afterwards leaves the file at the process umask (0644 on a default box)
        # for the whole write, and under DISTIL_NO_ENCRYPT_AT_REST what sits in
        # that window is plaintext agent tool output.
        atrest.write_owner_only(p, atrest.encrypt_bytes(raw))
    except OSError:
        pass  # best-effort; never crash a tool call


# FIFO-by-mtime cap. 500 was too small for a single long agent session: a 93-minute
# run folded 704 blocks and evicted 204 of them — including its most re-used fold —
# while still running, which made "everything stays recoverable" false mid-session.
_RESTORE_CAP = max(0, int(os.environ.get("DISTIL_RESTORE_CAP", "5000") or 0))
# Age cap on top of the count cap: digest originals are real agent content
# (can include secrets/PII), so a low-traffic store must not hold them forever.
# 0 disables. Expired handles simply fail to expand — same as capped-out ones.
# Enforced on the READ (`_live_restore_text`), not only by the sweep: the sweep is
# amortized, so on its own it would let a quiet store keep serving expired content.
_RESTORE_TTL_DAYS = float(os.environ.get("DISTIL_RESTORE_TTL_DAYS", "14") or 0)
_HANDLE_RE = re.compile(r"[0-9a-f]{8}")
# Sweeping the store is O(files) in ``stat()`` calls, and it used to run on every single
# recorded handle — twice, once for the count cap and once for the TTL — so at the 5,000
# cap one handle cost up to 10,000 stats. The re-read delta records several handles a
# turn, which is the ms/turn ADR 0010 attributes to the restore store. Amortize it.
# ponytail: a counter, not an mtime index. The ceiling it buys is an overshoot of at most
# this many files above the cap (and expired blobs living that much longer); swap in an
# index if the store ever needs to be exact between sweeps.
_SWEEP_EVERY = 64
_since_sweep = _SWEEP_EVERY  # sweep on the first record of a process, then every N
# ...and every TTL/24 regardless of how little traffic there is. A count alone stops
# being a schedule the moment the sweep is amortized: a store that never receives a
# 64th handle never reaches the trigger, so expired blobs would sit on disk for as long
# as the machine stayed quiet. 0 when the TTL is disabled — nothing to expire on time.
_last_sweep = 0.0  # epoch of the last sweep; 0 = never, so the first record sweeps


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


def _live_restore_text(p: Path) -> str | None:
    """``_read_restore_text`` with the TTL applied: an expired blob reads as absent.

    The TTL is a retention boundary, not a housekeeping preference — a restore blob is
    real agent content and can hold secrets or PII — so it is enforced HERE, on the read,
    where no caller can skip it. The sweep is bulk cleanup and cannot carry the guarantee
    on its own: since it was amortized to one run per ``_SWEEP_EVERY`` records, a store
    that goes quiet never reaches the trigger, and every expand against it would keep
    serving content that is past its retention date indefinitely.

    Expired blobs are unlinked on sight, so a read that finds one also cleans it up.
    Fail-open on ``OSError``: if the unlink loses a race with the sweep, the answer is
    still "absent", which is the answer that matters.
    """
    if _RESTORE_TTL_DAYS > 0:
        try:
            expired = p.stat().st_mtime < time.time() - _RESTORE_TTL_DAYS * 86400
        except OSError:
            expired = False  # cannot tell — fall through to the ordinary read
        if expired:
            with contextlib.suppress(OSError):
                p.unlink()
            return None
    return _read_restore_text(p)


def _maybe_sweep(d: Path) -> None:
    """Run the sweep when enough records OR enough time has passed, whichever first."""
    global _since_sweep, _last_sweep
    _since_sweep += 1
    now = time.time()
    interval = _RESTORE_TTL_DAYS * 3600  # a twenty-fourth of the TTL, in seconds
    if _since_sweep < _SWEEP_EVERY and not (interval > 0 and now - _last_sweep >= interval):
        return
    _since_sweep = 0
    _last_sweep = now
    _sweep(d)


def _sweep(d: Path) -> None:
    """Evict by count, then by age. One listing and one ``stat()`` per file."""
    ordered = sorted((f.stat().st_mtime, f) for f in d.iterdir())
    # Guard the 0 case: [:-0] is the WHOLE list, so an unguarded cap of 0 would
    # evict every blob rather than disabling the cap.
    stale = [f for _, f in ordered[:-_RESTORE_CAP]] if _RESTORE_CAP > 0 else []
    if _RESTORE_TTL_DAYS > 0:
        cutoff = time.time() - _RESTORE_TTL_DAYS * 86400
        stale += [f for mtime, f in ordered[-_RESTORE_CAP:] if mtime < cutoff]
    for old in stale:
        old.unlink()


def record_restore(handle: str, original: str) -> bool:
    """Persist a digest original to disk so handles survive proxy restarts/upgrades
    and can be expanded from other processes (e.g. this MCP server).

    Returns ``False`` only on a genuine on-disk COLLISION — *handle* already maps to
    different bytes. The caller must then not emit a stub for it: the running process
    would expand it correctly from memory, but a post-restart or cross-process
    ``distil_expand`` reads this file and would hand back the other block's content.
    Every other outcome returns ``True``, including a write that fails — persistence is
    best-effort and the in-memory store still answers for this session.
    """
    if not _HANDLE_RE.fullmatch(handle):
        return True
    try:
        d = _restore_dir()
        d.mkdir(parents=True, exist_ok=True)
        p = d / handle
        payload = atrest.encrypt_bytes(original.encode("utf-8"))
        # Exclusive create, so the FIRST writer wins atomically and 0600 is the mode the
        # file is born with. `p.exists()` then write was check-then-act across processes:
        # two proxies folding the same block could both see "absent", and on a genuine
        # 32-bit collision the second would clobber the first — precisely the outcome
        # this guard exists to prevent, in precisely the concurrent case it was added for.
        try:
            with open(p, "xb", opener=atrest.owner_only) as fh:
                fh.write(payload)
        except FileExistsError:
            # Collision guard, mirroring RestoreStore._record's in-memory check: if this
            # handle already maps to *different* bytes on disk, a 32-bit handle collided
            # across sessions. Do NOT clobber the earlier block — its stub would then
            # expand to the wrong content. Keep the first writer; refuse the second.
            existing = _live_restore_text(p)
            if existing is not None and existing != original:
                return False  # genuine collision — keep first writer
            # Same content, unreadable (auth failure/corrupt), or past its TTL — all
            # rewrite. The rewrite refreshes mtime for the sweep AND upgrades a legacy
            # plaintext file to the encrypted format.
            with open(p, "wb", opener=atrest.owner_only) as fh:
                fh.write(payload)
        p.chmod(0o600)  # belt-and-braces: a pre-existing file keeps its old mode
        _maybe_sweep(d)
    except OSError:
        pass  # best-effort; never crash a compress call
    return True


def load_restore(handle: str) -> str | None:
    """Return the persisted original for *handle*, or None."""
    if not _HANDLE_RE.fullmatch(handle):  # untrusted MCP arg — no path traversal
        return None
    return _live_restore_text(_restore_dir() / handle)


# ---------------------------------------------------------------------------
# Tool catalog + implementations
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "distil_compress",
        "description": (
            "Reversibly compress a text blob — typically a large tool output you want to "
            "keep in context cheaply. Returns JSON "
            '{"compressed": str, "handle": str|null, "tokens_saved": int}: a compact digest '
            "to keep in the conversation, plus an 8-hex handle that recovers the exact "
            "original bytes via distil_expand. The original is stored locally (encrypted, "
            "owner-only) and never sent anywhere. Use when a tool result is large enough "
            "that carrying it verbatim is wasteful; skip it for short text, which comes back "
            'unchanged with handle=null and tokens_saved=0. Errors return "error: ..." with '
            "isError set; nothing is stored."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Raw text to compress, passed verbatim — do not pre-summarize or "
                        "truncate it, or the recovered original will be lossy."
                    ),
                }
            },
            "required": ["text"],
        },
        "annotations": {
            "title": "Compress text (reversibly)",
            "readOnlyHint": False,  # writes the original to the local store
            "destructiveHint": False,  # additive only; never mutates prior handles
            "idempotentHint": True,  # same text → same handle, same stored bytes
            "openWorldHint": False,  # purely local; no network
        },
    },
    {
        "name": "distil_expand",
        "description": (
            "Recover the exact original text for an 8-hex handle returned by "
            "distil_compress. Returns the original bytes as plain text — not JSON, not a "
            "summary. Use when the digest in context lacks a detail you now need (an exact "
            "line, value, or stack frame); prefer reading the digest first, since expanding "
            "spends the tokens compression saved. Reads a local, encrypted store, so it "
            "works across sessions and processes but not across machines. An unknown, "
            'expired, or evicted handle returns "error: no original found for handle ..." '
            "with isError set — re-run the original tool rather than retrying; the answer "
            "will not change. Handles age out after DISTIL_RESTORE_TTL_DAYS (default 14)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": (
                        "The 8-hex handle from a prior distil_compress result or a digest "
                        "stub in context, e.g. '3f9a1c07'. Content-addressed, so it is "
                        "stable across runs; any other shape is rejected."
                    ),
                    "pattern": "^[0-9a-f]{8}$",
                }
            },
            "required": ["handle"],
        },
        "annotations": {
            "title": "Expand a handle to its original text",
            "readOnlyHint": True,  # pure lookup
            "destructiveHint": False,
            "idempotentHint": True,  # same handle → same bytes, until it ages out
            "openWorldHint": False,  # purely local; no network
        },
    },
    {
        "name": "distil_savings",
        "description": (
            "Report cumulative savings from the local distil ledger as JSON "
            '{"runs": int, "tokens_saved": int, "dollars_saved": float}. Covers every '
            "request distil has compressed on this machine, not just this session. Use to "
            "answer 'how much has distil saved me'; it says nothing about whether any one "
            "compression was correct. Takes no arguments; an empty ledger reports zeros."
        ),
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "title": "Report cumulative savings",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
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
