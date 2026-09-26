"""Content-free local event log and per-server learned state for ``distil mcp``.

Everything here lives under ``$DISTIL_HOME/mcp/`` (default ``~/.distil/mcp``), is
created owner-only, and never leaves the machine — nothing in this module is read by
the census or any telemetry path. What is written:

* ``events.jsonl`` — one line per proxy event: timestamp, session id, server name,
  tool NAME, event kind, level, and integer token sizes. Never arguments, never
  results, never descriptions. Enforced by ``EventLog.emit``'s field allowlist.
* ``state/<server>.json`` — per-server tool-call counts (L3 learns its pin set from
  these) and each session's unlocked-tool list and list-change count.
* ``catalog/<server>.json`` — the server's tool definitions before and after
  compression, for the webdash diff view. Tool metadata the server publishes, not
  user content; local only, like the rest.

Every write is best-effort: a full disk or a read-only home degrades the watch view,
never the proxy.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .. import _filelock, atrest

#: Keys an event may carry, and the only types their values may have. A string field
#: outside this set cannot be logged, which is what keeps the log content-free.
_STR_FIELDS = frozenset({"session", "server", "tool", "ev", "level", "via", "err", "skipped"})
_NUM_FIELDS = frozenset({"ts", "tokens_before", "tokens_after", "n", "ms"})
#: Rotate ``events.jsonl`` past this size (one previous generation is kept).
MAX_LOG_BYTES = 4 * 1024 * 1024
#: Sessions kept per server in the state file (oldest dropped first).
MAX_SESSIONS = 32


def mcp_dir() -> Path:
    base = Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))
    return base / "mcp"


def new_session_id() -> str:
    return os.environ.get("DISTIL_MCP_SESSION") or uuid.uuid4().hex[:12]


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "mcp"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    atrest.write_owner_only(tmp, (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    _filelock.replace_retrying(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class EventLog:
    """Append-only, content-free, best-effort."""

    def __init__(self, session: str, path: Path | None = None) -> None:
        self.session = session
        self.path = path or mcp_dir() / "events.jsonl"

    #: False on ``NullLog`` — lets callers skip the token math an event would carry.
    enabled = True

    def emit(self, ev: str, server: str, **fields: Any) -> None:
        row: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "session": self.session,
            "server": server,
            "ev": ev,
        }
        for key, val in fields.items():
            if val is None:
                continue
            if key in _STR_FIELDS and isinstance(val, str):
                row[key] = val
            elif key in _NUM_FIELDS and isinstance(val, (int, float)) and not isinstance(val, bool):
                row[key] = val
            else:
                raise ValueError(f"event field {key!r} is not in the content-free allowlist")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                if self.path.stat().st_size > MAX_LOG_BYTES:
                    _filelock.replace_retrying(self.path, self.path.with_name("events.jsonl.1"))
            with (
                _filelock.locked(self.path),
                open(self.path, "a", encoding="utf-8", opener=atrest.owner_only) as fh,
            ):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass  # the log is for the watch view; never fail a tool call over it


class NullLog(EventLog):
    """Discards everything (the bench runs thousands of sessions)."""

    enabled = False

    def __init__(self, session: str = "null") -> None:
        super().__init__(session, Path("/dev/null"))

    def emit(self, ev: str, server: str, **fields: Any) -> None:
        return None


def read_events(path: Path | None = None, limit: int = 20000) -> list[dict[str, Any]]:
    """The last *limit* events (previous generation first, when rotated)."""
    path = path or mcp_dir() / "events.jsonl"
    rows: list[dict[str, Any]] = []
    for p in (path.with_name(path.name + ".1"), path):
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for ln in lines:
            with contextlib.suppress(ValueError):
                row = json.loads(ln)
                if isinstance(row, dict):
                    rows.append(row)
    return rows[-limit:]


class ServerState:
    """Per-server learned state: tool-call counts and per-session unlocks."""

    def __init__(self, server: str, root: Path | None = None) -> None:
        self.server = server
        self.path = (root or mcp_dir()) / "state" / f"{_safe(server)}.json"

    def load(self) -> dict[str, Any]:
        data = _read_json(self.path)
        if not isinstance(data, dict):
            data = {}
        usage = data.get("usage")
        sessions = data.get("sessions")
        return {
            "usage": {str(k): int(v) for k, v in usage.items() if isinstance(v, int)}
            if isinstance(usage, dict)
            else {},
            "sessions": sessions if isinstance(sessions, dict) else {},
        }

    def usage(self) -> dict[str, int]:
        return self.load()["usage"]

    def unlocked(self, session: str) -> list[str]:
        entry = self.load()["sessions"].get(session)
        names = entry.get("unlocked") if isinstance(entry, dict) else None
        return [str(n) for n in names] if isinstance(names, list) else []

    def update(
        self,
        session: str,
        *,
        used: str | None = None,
        unlocked: list[str] | None = None,
        list_changed: bool = False,
    ) -> None:
        try:
            with _filelock.locked(self.path):
                data = self.load()
                if used:
                    data["usage"][used] = data["usage"].get(used, 0) + 1
                sess = data["sessions"].get(session)
                if not isinstance(sess, dict):
                    sess = {"unlocked": [], "list_changes": 0}
                if unlocked is not None:
                    sess["unlocked"] = list(unlocked)
                if list_changed:
                    sess["list_changes"] = int(sess.get("list_changes", 0)) + 1
                sess["ts"] = round(time.time(), 3)
                data["sessions"][session] = sess
                if len(data["sessions"]) > MAX_SESSIONS:
                    oldest = sorted(
                        data["sessions"].items(),
                        key=lambda kv: kv[1].get("ts", 0) if isinstance(kv[1], dict) else 0,
                    )
                    for key, _ in oldest[: len(data["sessions"]) - MAX_SESSIONS]:
                        del data["sessions"][key]
                _write_json(self.path, data)
        except OSError:
            pass


def write_catalog(server: str, snapshot: dict[str, Any], root: Path | None = None) -> None:
    with contextlib.suppress(OSError):
        _write_json((root or mcp_dir()) / "catalog" / f"{_safe(server)}.json", snapshot)


def read_catalogs(root: Path | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    base = (root or mcp_dir()) / "catalog"
    try:
        paths = sorted(base.glob("*.json"))
    except OSError:
        return out
    for p in paths:
        data = _read_json(p)
        if isinstance(data, dict) and isinstance(data.get("server"), str):
            out[data["server"]] = data
    return out


def read_states(root: Path | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    base = (root or mcp_dir()) / "state"
    try:
        paths = sorted(base.glob("*.json"))
    except OSError:
        return out
    for p in paths:
        data = _read_json(p)
        if isinstance(data, dict):
            out[p.stem] = data
    return out
