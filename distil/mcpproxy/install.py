"""``distil mcp install`` — route a client's MCP servers through the proxy, with exact undo.

Each stdio server entry in the client's config is rewritten in place to launch
``distil mcp wrap --name <server> -- <original command> <original args>``; ``env``,
``cwd`` and every other key are left as they were (the proxy inherits them and hands
them to the real server). Remote (``url``) entries are never touched.

Writes follow the config-file pattern already used by ``distil wrap``'s config presets
(``config_wrap._atomic_write_secure``): a byte-exact ``.distil-mcp-backup`` of the
pre-install file, then a same-directory temp file created 0600 and swapped in with
``os.replace``. Ownership is recorded in ``$DISTIL_HOME/mcp/installs.json`` — the
SHA-256 of the bytes distil wrote and the file's original mode — so undo can tell
"still exactly what distil wrote" (restore the backup byte-for-byte, and the mode)
from "edited since" (unwrap distil's entries only, keep the user's later edits, and
leave the backup where it is).

Client config locations (verified 2026-09-24 against each client's docs):
Cursor ``~/.cursor/mcp.json``; Claude Desktop ``claude_desktop_config.json``; Gemini
CLI ``~/.gemini/settings.json``; Windsurf ``~/.codeium/windsurf/mcp_config.json``
(all ``mcpServers``); opencode ``~/.config/opencode/opencode.json`` (``mcp``, local
entries with a ``command`` array); Codex ``~/.codex/config.toml``
(``[mcp_servers.<name>]``). Claude Code is deliberately absent — its native tool
search defers unused tools better than a proxy can (ADR 0013).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import _filelock
from ..config_wrap import _atomic_write_secure
from . import events, levels

BACKUP_SUFFIX = ".distil-mcp-backup"


@dataclass(frozen=True)
class Client:
    key: str
    label: str
    fmt: str  # "json" | "opencode" | "toml"
    path: Callable[[], Path]


def _claude_desktop_path() -> Path:
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if os.name == "nt":
        return (
            Path(os.environ.get("APPDATA", str(Path.home())))
            / "Claude"
            / "claude_desktop_config.json"
        )
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


CLIENTS: dict[str, Client] = {
    c.key: c
    for c in (
        Client("cursor", "Cursor", "json", lambda: Path.home() / ".cursor" / "mcp.json"),
        Client("claude-desktop", "Claude Desktop", "json", _claude_desktop_path),
        Client("gemini", "Gemini CLI", "json", lambda: Path.home() / ".gemini" / "settings.json"),
        Client(
            "windsurf",
            "Windsurf",
            "json",
            lambda: Path.home() / ".codeium" / "windsurf" / "mcp_config.json",
        ),
        Client(
            "opencode",
            "opencode",
            "opencode",
            lambda: Path.home() / ".config" / "opencode" / "opencode.json",
        ),
        Client("codex", "Codex", "toml", lambda: Path.home() / ".codex" / "config.toml"),
    )
}


class InstallError(RuntimeError):
    pass


def distil_command() -> str:
    """The launcher written into client configs: the absolute ``distil`` on PATH."""
    return shutil.which("distil") or "distil"


def _is_wrapped(argv: list[Any]) -> bool:
    return len(argv) >= 3 and "--" in argv and argv[:2] == ["mcp", "wrap"]


def wrap_argv(name: str, command: list[str], level: str, results: bool) -> list[str]:
    extra = ["--level", level] + ([] if results else ["--no-results"])
    return ["mcp", "wrap", "--name", name, *extra, "--", *command]


def unwrap_argv(argv: list[str]) -> list[str]:
    """The original command line from a wrapped ``distil mcp wrap … -- cmd args``."""
    return argv[argv.index("--") + 1 :]


# ---------------------------------------------------------------------------
# JSON clients
# ---------------------------------------------------------------------------


def _json_section(doc: dict[str, Any], fmt: str) -> dict[str, Any]:
    key = "mcp" if fmt == "opencode" else "mcpServers"
    section = doc.get(key)
    return section if isinstance(section, dict) else {}


def _rewrite_json(
    text: str, fmt: str, fn: Callable[[str, dict[str, Any]], dict[str, Any] | None]
) -> tuple[str, list[str]]:
    try:
        doc = json.loads(text) if text.strip() else {}
    except ValueError as exc:
        raise InstallError(
            f"not valid JSON ({exc}) — distil will not rewrite a file it cannot parse"
        ) from exc
    if not isinstance(doc, dict):
        raise InstallError("config root is not a JSON object")
    section = _json_section(doc, fmt)
    changed: list[str] = []
    for name, entry in list(section.items()):
        if isinstance(entry, dict):
            new = fn(str(name), entry)
            if new is not None:
                section[name] = new
                changed.append(str(name))
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n", changed


def _wrap_json_entry(
    fmt: str, level: str, results: bool
) -> Callable[[str, dict[str, Any]], dict[str, Any] | None]:
    launcher = distil_command()

    def fn(name: str, entry: dict[str, Any]) -> dict[str, Any] | None:
        if fmt == "opencode":
            cmd = entry.get("command")
            if entry.get("type", "local") != "local" or not isinstance(cmd, list) or not cmd:
                return None
            if _is_wrapped(cmd[1:]):
                return None
            return {
                **entry,
                "command": [launcher, *wrap_argv(name, [str(c) for c in cmd], level, results)],
            }
        cmd = entry.get("command")
        args = entry.get("args", [])
        if "url" in entry or not isinstance(cmd, str) or not isinstance(args, list):
            return None
        if _is_wrapped(args):
            return None
        return {
            **entry,
            "command": launcher,
            "args": wrap_argv(name, [cmd, *map(str, args)], level, results),
        }

    return fn


def _unwrap_json_entry(fmt: str) -> Callable[[str, dict[str, Any]], dict[str, Any] | None]:
    def fn(name: str, entry: dict[str, Any]) -> dict[str, Any] | None:
        if fmt == "opencode":
            cmd = entry.get("command")
            if not isinstance(cmd, list) or not _is_wrapped(cmd[1:]):
                return None
            return {**entry, "command": unwrap_argv(cmd[1:])}
        args = entry.get("args")
        if not isinstance(args, list) or not _is_wrapped(args):
            return None
        original = unwrap_argv(args)
        if not original:
            return None
        return {**entry, "command": original[0], "args": original[1:]}

    return fn


# ---------------------------------------------------------------------------
# Codex (TOML): a line-level patch, verified by re-parsing — or refused
# ---------------------------------------------------------------------------


def _toml() -> Any:
    try:
        import tomllib
    except ImportError as exc:  # Python < 3.11
        raise InstallError("rewriting Codex's config.toml needs Python 3.11+ (tomllib)") from exc
    return tomllib


def _toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # a JSON string is a valid TOML basic string


def _bracket_delta(line: str) -> int:
    depth, quote, esc = 0, "", False
    for ch in line:
        if quote:
            if esc:
                esc = False
            elif ch == "\\" and quote == '"':
                esc = True
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            break
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
    return depth


_TABLE_RE = re.compile(r"^\s*\[\[?[^\]=]+\]\]?\s*(?:#.*)?$")


def _section_end(lines: list[str], start: int) -> int:
    """Index of the next table header after *start*, skipping multi-line arrays."""
    depth = 0
    for i in range(start + 1, len(lines)):
        if depth == 0 and _TABLE_RE.match(lines[i].rstrip("\r\n")):
            return i
        depth += _bracket_delta(lines[i])
    return len(lines)


def _rewrite_toml(
    text: str, fn: Callable[[str, list[str]], list[str] | None]
) -> tuple[str, list[str]]:
    """Rewrite ``command``/``args`` of each ``[mcp_servers.<name>]`` table.

    Only the two key lines (and an ``args`` array's continuation lines) are replaced;
    every other byte of the file is kept. The result is re-parsed and compared with
    the intended document — anything that does not round-trip is refused, not written.
    """
    tomllib = _toml()
    try:
        doc = tomllib.loads(text)
    except ValueError as exc:
        raise InstallError(f"not valid TOML ({exc})") from exc
    servers = doc.get("mcp_servers")
    if not isinstance(servers, dict):
        return text, []
    lines = text.splitlines(keepends=True)
    changed: list[str] = []
    expected = json.loads(json.dumps(doc))
    for name, entry in servers.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("command"), str):
            continue
        new_cmd = fn(str(name), [entry["command"], *map(str, entry.get("args") or [])])
        if new_cmd is None:
            continue
        header = re.compile(
            r"^\s*\[\s*mcp_servers\s*\.\s*(?:"
            + re.escape(name)
            + r'|"'
            + re.escape(name)
            + r'")\s*\]\s*(?:#.*)?$'
        )
        start = next((i for i, ln in enumerate(lines) if header.match(ln.rstrip("\r\n"))), None)
        if start is None:
            raise InstallError(
                f"[mcp_servers.{name}] is not a plain table header — edit it by hand"
            )
        end = _section_end(lines, start)
        body = lines[start + 1 : end]
        kept: list[str] = []
        insert_at: int | None = None
        i = 0
        while i < len(body):
            ln = body[i]
            if re.match(r"^\s*(command|args)\s*=", ln):
                insert_at = len(kept) if insert_at is None else insert_at
                depth = _bracket_delta(ln)
                while depth > 0 and i + 1 < len(body):
                    i += 1
                    depth += _bracket_delta(body[i])
                i += 1
                continue
            kept.append(ln)
            i += 1
        newline = "\r\n" if lines[start].endswith("\r\n") else "\n"
        block = [
            f"command = {_toml_str(new_cmd[0])}{newline}",
            f"args = [{', '.join(_toml_str(a) for a in new_cmd[1:])}]{newline}",
        ]
        at = insert_at if insert_at is not None else 0
        lines[start + 1 : end] = kept[:at] + block + kept[at:]
        expected["mcp_servers"][name]["command"] = new_cmd[0]
        expected["mcp_servers"][name]["args"] = new_cmd[1:]
        changed.append(str(name))
    out = "".join(lines)
    try:
        ok = json.loads(json.dumps(tomllib.loads(out))) == expected
    except ValueError:
        ok = False
    if not ok:
        raise InstallError("the rewritten config.toml did not round-trip — nothing was written")
    return out, changed


# ---------------------------------------------------------------------------
# Ownership records
# ---------------------------------------------------------------------------


def _records_path() -> Path:
    return events.mcp_dir() / "installs.json"


def _load_records() -> dict[str, Any]:
    try:
        data = json.loads(_records_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_records(records: dict[str, Any]) -> None:
    path = _records_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_secure(path, (json.dumps(records, indent=2) + "\n").encode("utf-8"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve(client: str, path: Path | None) -> tuple[Client, Path]:
    if client == "custom":
        if path is None:
            raise InstallError("--path is required for a custom config")
        return Client("custom", "custom", "json", lambda: path), path
    if client not in CLIENTS:
        raise InstallError(
            f"unknown client {client!r}; choose one of {', '.join(CLIENTS)} or custom"
        )
    c = CLIENTS[client]
    return c, path or c.path()


def _render(c: Client, text: str, wrap: bool, level: str, results: bool) -> tuple[str, list[str]]:
    if c.fmt == "toml":
        launcher = distil_command()

        def tfn(name: str, argv: list[str]) -> list[str] | None:
            if wrap:
                return (
                    None
                    if _is_wrapped(argv[1:])
                    else [launcher, *wrap_argv(name, argv, level, results)]
                )
            return unwrap_argv(argv[1:]) if _is_wrapped(argv[1:]) else None

        return _rewrite_toml(text, tfn)
    fn = _wrap_json_entry(c.fmt, level, results) if wrap else _unwrap_json_entry(c.fmt)
    return _rewrite_json(text, c.fmt, fn)


def install(
    client: str,
    *,
    path: Path | None = None,
    level: str = levels.DEFAULT_LEVEL,
    results: bool = levels.DEFAULT_RESULTS,
    dry_run: bool = False,
) -> tuple[str, str]:
    """Wrap every stdio server in *client*'s config. ``(status, message)``.

    status: ``ok`` | ``dry-run`` | ``exists`` (nothing left to wrap) | ``absent``.
    Raises ``InstallError`` for anything it refuses to write.
    """
    if level not in levels.LEVELS:
        raise InstallError(f"unknown level {level!r}")
    c, target = _resolve(client, path)
    if not target.exists():
        return "absent", f"no {c.label} MCP config at {target}"
    with _filelock.locked(target):
        before = target.read_bytes()
        text, changed = _render(c, before.decode("utf-8"), True, level, results)
        if not changed:
            return (
                "exists",
                f"every stdio server in {target} already routes through distil (or none to wrap)",
            )
        if dry_run:
            return "dry-run", f"would wrap {', '.join(changed)} in {target}:\n{text}"
        backup = target.with_name(target.name + BACKUP_SUFFIX)
        records = _load_records()
        rec = records.get(str(target))
        if not backup.exists():  # the pre-distil original, written once, never overwritten
            _atomic_write_secure(backup, before)
            rec = None
        mode = rec.get("mode") if isinstance(rec, dict) else target.stat().st_mode & 0o777
        new = text.encode("utf-8")
        _atomic_write_secure(target, new)
        records[str(target)] = {
            "client": c.key,
            "sha256": _sha(new),
            "backup": str(backup),
            "mode": mode,
            "servers": sorted(
                set(changed) | set(rec.get("servers", []) if isinstance(rec, dict) else [])
            ),
            "ts": round(time.time(), 3),
        }
        _save_records(records)
    return (
        "ok",
        f"wrapped {', '.join(changed)} in {target} (backup: {backup.name}; undo: distil mcp install {c.key} --undo)",
    )


def uninstall(client: str, *, path: Path | None = None) -> tuple[str, str]:
    """Undo ``install``. ``(status, message)``: ``restored`` | ``unwrapped`` | ``absent``.

    Byte-exact restore when the file is still exactly what distil wrote; otherwise
    distil's entries are unwrapped in place and everything else the user changed since
    is kept (the backup is left on disk and named in the message).
    """
    c, target = _resolve(client, path)
    records = _load_records()
    rec = records.get(str(target))
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not target.exists():
        return "absent", f"no {c.label} MCP config at {target}"
    with _filelock.locked(target):
        current = target.read_bytes()
        if isinstance(rec, dict) and backup.exists() and _sha(current) == rec.get("sha256"):
            _atomic_write_secure(target, backup.read_bytes())
            if isinstance(rec.get("mode"), int):
                target.chmod(rec["mode"])
            backup.unlink()
            records.pop(str(target), None)
            _save_records(records)
            return "restored", f"restored {target} byte-for-byte from before distil mcp install"
        text, changed = _render(c, current.decode("utf-8"), False, levels.DEFAULT_LEVEL, True)
        if changed:
            _atomic_write_secure(target, text.encode("utf-8"))
            if isinstance(rec, dict) and isinstance(rec.get("mode"), int):
                target.chmod(rec["mode"])
        if str(target) in records:
            records.pop(str(target))
            _save_records(records)
    if not changed:
        return "absent", f"nothing in {target} routes through distil mcp"
    left = f"; the pre-install backup is kept at {backup}" if backup.exists() else ""
    return (
        "unwrapped",
        f"unwrapped {', '.join(changed)} in {target}, keeping your later edits{left}",
    )
