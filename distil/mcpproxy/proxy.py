"""The MCP proxy: a JSON-RPC session in front of one or more stdio MCP servers.

Transparent by default — every method the proxy does not compress (resources,
prompts, completion, logging, pings, notifications, server-to-client requests such as
``roots/list`` or ``sampling/createMessage``) is relayed unchanged, with request ids
preserved so ``notifications/cancelled`` and progress tokens keep working. Only
``tools/list`` and ``tools/call`` are rewritten, per ``levels``.

Fail-open at every layer: a compression error on ``tools/list`` or ``tools/call``
relays the backend's raw answer instead (and logs the exception's class name, never
its message); ``distil mcp wrap`` execs the backend directly if the proxy cannot even
start (see ``cli.cmd_mcp``).

Threads, not asyncio: stdio JSON-RPC is line-at-a-time blocking I/O on both sides, the
existing ``distil.mcp_server`` is sync, and a thread per in-flight call keeps a slow
tool from blocking ``ping`` without needing an event loop on Windows pipes.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from .. import mcp_server
from . import events, levels

#: Seconds to wait for a backend's answer. MCP tools can be slow (builds, crawls), so
#: generous; override with ``DISTIL_MCP_TIMEOUT``.
DEFAULT_TIMEOUT = float(os.environ.get("DISTIL_MCP_TIMEOUT", "600") or 600)
_REMOTE_TYPES = frozenset({"http", "sse", "streamable-http", "streamableHttp"})


class ConfigError(ValueError):
    """An MCP client config that cannot be proxied as written."""


class BackendError(RuntimeError):
    """A backend server did not answer (exited, timed out, or its pipe broke)."""


@dataclass
class ServerSpec:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


def parse_servers(data: Any) -> tuple[list[ServerSpec], list[str]]:
    """``(stdio servers, skipped-with-reason)`` from a Claude-Desktop/Cursor-style config.

    Accepts ``{"mcpServers": {...}}`` and VS Code's ``{"servers": {...}}``. Remote
    (``url``) servers and ``disabled`` ones are skipped with a reason, not an error.
    """
    if not isinstance(data, dict):
        raise ConfigError("config root must be a JSON object")
    servers = data.get("mcpServers", data.get("servers"))
    if not isinstance(servers, dict):
        raise ConfigError('config has no "mcpServers" object')
    specs: list[ServerSpec] = []
    skipped: list[str] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"server {name!r}: entry must be an object")
        if entry.get("disabled") is True:
            skipped.append(f"{name}: disabled")
            continue
        if "url" in entry or entry.get("type") in _REMOTE_TYPES:
            skipped.append(f"{name}: remote (url) servers are not proxied — stdio only")
            continue
        command = entry.get("command")
        args = entry.get("args", [])
        env = entry.get("env", {})
        cwd = entry.get("cwd")
        if not isinstance(command, str) or not command:
            raise ConfigError(f'server {name!r}: "command" must be a non-empty string')
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ConfigError(f'server {name!r}: "args" must be a list of strings')
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            raise ConfigError(f'server {name!r}: "env" must map strings to strings')
        if cwd is not None and not isinstance(cwd, str):
            raise ConfigError(f'server {name!r}: "cwd" must be a string')
        specs.append(ServerSpec(str(name), command, list(args), dict(env), cwd))
    if not specs:
        raise ConfigError(
            "no stdio MCP servers to proxy" + (f" ({'; '.join(skipped)})" if skipped else "")
        )
    return specs, skipped


def load_config(path: Path) -> tuple[list[ServerSpec], list[str]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    return parse_servers(data)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Backend(Protocol):
    name: str
    on_message: Callable[[dict[str, Any]], None]

    def request(self, method: str, params: Any = None, *, id: Any = None) -> dict[str, Any]: ...

    def notify(self, method: str, params: Any = None) -> None: ...

    def send(self, msg: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class StdioBackend:
    """One MCP server subprocess spoken to over newline-delimited JSON-RPC.

    The server's stderr is inherited, so its logs land wherever the client already
    collects this proxy's stderr.
    """

    def __init__(self, spec: ServerSpec, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.name = spec.name
        self.spec = spec
        self.timeout = timeout
        self.on_message: Callable[[dict[str, Any]], None] = lambda msg: None
        self._pending: dict[Any, list[Any]] = {}
        self._lock = threading.Lock()
        self._wlock = threading.Lock()
        self._ids = itertools.count(1)
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        env = {**os.environ, **self.spec.env}
        self.proc = subprocess.Popen(
            [self.spec.command, *self.spec.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            cwd=self.spec.cwd,
        )
        threading.Thread(target=self._read, name=f"mcp-{self.name}", daemon=True).start()

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue  # a server printing junk to stdout must not kill the session
            if not isinstance(msg, dict):
                continue
            if "method" not in msg and "id" in msg:
                with self._lock:
                    slot = self._pending.pop(msg["id"], None)
                if slot is not None:
                    slot[1] = msg
                    slot[0].set()
                continue
            try:
                self.on_message(msg)
            except Exception:  # noqa: BLE001 — a relay failure must not stop the reader
                pass
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot[0].set()  # slot[1] stays None: "the server exited"

    def _write(self, msg: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise BackendError(f"{self.name}: not started")
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self._wlock:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def request(self, method: str, params: Any = None, *, id: Any = None) -> dict[str, Any]:
        rid = id if id is not None else f"distil-mcp-{next(self._ids)}"
        slot: list[Any] = [threading.Event(), None]
        with self._lock:
            self._pending[rid] = slot
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self._write(msg)
        except (OSError, ValueError) as exc:
            with self._lock:
                self._pending.pop(rid, None)
            raise BackendError(f"{self.name}: cannot write to server ({exc})") from exc
        if not slot[0].wait(self.timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise BackendError(f"{self.name}: no answer to {method} within {self.timeout:.0f}s")
        if slot[1] is None:
            raise BackendError(f"{self.name}: server exited")
        return dict(slot[1])

    def notify(self, method: str, params: Any = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self._write(msg)
        except (OSError, ValueError, BackendError):
            pass  # a notification to a dead server has nowhere to go

    def send(self, msg: dict[str, Any]) -> None:
        try:
            self._write(msg)
        except (OSError, ValueError, BackendError):
            pass

    def close(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


def _result(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _text_result(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


_LIST_CHANGED = {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
# Method families a backend advertises in its ``capabilities`` and the proxy relays.
_FAMILY = {
    "resources": "resources",
    "prompts": "prompts",
    "completion": "completions",
    "logging": "logging",
}


class Proxy:
    """One client session. ``handle`` is synchronous and thread-safe."""

    def __init__(
        self,
        backends: dict[str, Backend],
        *,
        level: str = levels.DEFAULT_LEVEL,
        results: bool = levels.DEFAULT_RESULTS,
        session: str | None = None,
        log: events.EventLog | None = None,
        state_root: Path | None = None,
        record: Callable[[str, str], bool] = mcp_server.record_restore,
        expand: Callable[[str], str | None] = mcp_server.load_restore,
        persist: bool = True,
    ) -> None:
        if level not in levels.LEVELS:
            raise ValueError(f"unknown level {level!r}; choose one of {', '.join(levels.LEVELS)}")
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = backends
        self.level = level
        self.results = results
        self.session = session or events.new_session_id()
        self.log = log or events.EventLog(self.session)
        self.state_root = state_root
        self.record = record
        self.expand = expand
        self.persist = persist  # False: no catalog snapshots, no learned state (the bench)
        self.emit: Callable[[dict[str, Any]], None] = lambda msg: None
        self.surfaces: dict[str, levels.Surface] = {}
        self.caps: dict[str, dict[str, Any]] = {}
        self.list_changes = 0
        self._raw_owner: dict[str, str] = {}
        self._unlocked: dict[str, list[str]] = {}
        self._server_requests: dict[str, tuple[str, Any]] = {}
        self._sreq_ids = itertools.count(1)
        self._lock = threading.RLock()
        self._states = {n: events.ServerState(n, state_root) for n in backends}
        for name, backend in backends.items():
            backend.on_message = self._relay_from(name)

    def _relay_from(self, server: str) -> Callable[[dict[str, Any]], None]:
        return lambda msg: self.on_backend_message(server, msg)

    # -- client -> proxy ----------------------------------------------------

    def handle(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Handle one client message; return what to send back, in order."""
        method = msg.get("method")
        msg_id = msg.get("id")
        if method is None:
            self._route_client_response(msg)
            return []
        if msg_id is None:
            for backend in self.backends.values():
                backend.notify(method, msg.get("params"))
            return []
        params = msg.get("params")
        try:
            if method == "initialize":
                return [self._initialize(msg_id, params)]
            if method == "ping":
                return [_result(msg_id, {})]
            if method == "tools/list":
                return [_result(msg_id, {"tools": self._compressed_list()})]
            if method == "tools/call":
                return self._tools_call(msg_id, params if isinstance(params, dict) else {})
            return [self._relay(msg_id, method, params)]
        except BackendError as exc:
            return [_error(msg_id, -32603, str(exc))]
        except Exception as exc:  # noqa: BLE001 — fail open: the backend's own answer
            self.log.emit("error", "*", err=type(exc).__name__)
            try:
                return [self._raw(msg_id, method, params)]
            except BackendError as inner:
                return [_error(msg_id, -32603, str(inner))]

    def _initialize(self, msg_id: Any, params: Any) -> dict[str, Any]:
        first: dict[str, Any] | None = None
        instructions: list[str] = []
        caps: dict[str, Any] = {}
        for name, backend in list(self.backends.items()):
            resp = backend.request("initialize", params)
            if "error" in resp:
                if len(self.backends) == 1:
                    return {"jsonrpc": "2.0", "id": msg_id, "error": resp["error"]}
                self.log.emit("error", name, err="initialize")
                del self.backends[name]
                continue
            res = resp.get("result") or {}
            first = first or res
            self.caps[name] = res.get("capabilities") or {}
            for k, v in self.caps[name].items():
                caps.setdefault(k, v)
            if isinstance(res.get("instructions"), str):
                instructions.append(res["instructions"])
        if first is None:
            return _error(msg_id, -32603, "no MCP server behind distil mcp initialized")
        caps["tools"] = {**(caps.get("tools") or {}), "listChanged": True}
        from .. import __version__

        out: dict[str, Any] = {
            "protocolVersion": first.get("protocolVersion") or mcp_server.DEFAULT_PROTOCOL,
            "capabilities": caps,
            "serverInfo": {
                "name": f"distil-mcp[{','.join(self.backends)}]",
                "version": __version__,
            },
        }
        if instructions:
            out["instructions"] = "\n\n".join(instructions)
        return _result(msg_id, out)

    # -- tools/list ----------------------------------------------------------

    def _fetch_tools(self, backend: Backend) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = None
        for _ in range(100):  # a server paging forever must not hang the session
            resp = backend.request("tools/list", {"cursor": cursor} if cursor else {})
            if "error" in resp:
                raise BackendError(f"{backend.name}: tools/list failed: {resp['error']}")
            res = resp.get("result") or {}
            tools += [t for t in res.get("tools") or [] if isinstance(t, dict)]
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return tools

    def _surfaces(self) -> dict[str, levels.Surface]:
        with self._lock:
            if all(n in self.surfaces for n in self.backends):
                return self.surfaces
            taken: set[str] = set()
            for name, surf in self.surfaces.items():
                taken |= {surf.prefix + n for n in surf.by_name}
            for name, backend in self.backends.items():
                if name in self.surfaces:
                    continue
                tools = self._fetch_tools(backend)
                names = [str(t.get("name")) for t in tools]
                prefix = f"{levels.safe_server(name)}_" if taken & set(names) else ""
                state = self._states[name]
                pins = (
                    levels.choose_pins(state.usage(), names) if self.level == "L3" else frozenset()
                )
                unlocked = self._unlocked.get(name) or state.unlocked(self.session)
                surf = levels.Surface(
                    name,
                    tools,
                    self.level,
                    self.results,
                    pinned=pins,
                    prefix=prefix,
                    unlocked=unlocked,
                )
                self.surfaces[name] = surf
                for n in names:
                    self._raw_owner.setdefault(n, name)
                taken |= {prefix + n for n in names}
                if self.persist:
                    self._snapshot(surf)
            return self.surfaces

    def _snapshot(self, surf: levels.Surface) -> None:
        """Write the before/after catalog the webdash diff view reads."""
        rows = []
        for name, tool in surf.by_name.items():
            before = levels.model_view(tool)
            after = levels.model_view(surf.definition(name))
            lazy_line = levels.index_line(tool) if surf.lazy else None
            rows.append(
                {
                    "name": name,
                    "before": before,
                    "after": after,
                    "tokens_before": levels.tokens(before),
                    "tokens_after": levels.tokens(after),
                    "tokens_lazy": levels.tokens(lazy_line) if lazy_line else None,
                    "dropped": levels.dropped_paths(before, after),
                    "pinned": name in surf.pinned,
                }
            )
        events.write_catalog(
            surf.server,
            {
                "server": surf.server,
                "level": surf.level,
                "requested": surf.requested,
                "results": surf.results,
                "session": self.session,
                "ts": round(time.time(), 3),
                "meta": [levels.model_view(t) for t in surf.meta_tools()],
                "tools": rows,
            },
            self.state_root,
        )

    def _compressed_list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for surf in self._surfaces().values():
            listed = surf.tools_list()
            if not self.log.enabled:
                out += listed
                continue
            before = sum(levels.definition_tokens(t) for t in surf.tools)
            after = sum(levels.definition_tokens(t) for t in listed)
            self.log.emit(
                "list",
                surf.server,
                level=surf.level,
                tokens_before=before,
                tokens_after=after,
                n=len(listed),
            )
            out += listed
        return out

    # -- tools/call ----------------------------------------------------------

    def _tools_call(self, msg_id: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        name = str(params.get("name") or "")
        args = params.get("arguments")
        args = args if isinstance(args, dict) else {}
        for server, surf in self._surfaces().items():
            kind, tool = surf.resolve(name)
            if kind != "unknown":
                break
        else:
            if len(self.backends) == 1:  # the backend is the authority on its own tools
                return [self._raw(msg_id, "tools/call", params)]
            return [_error(msg_id, -32602, f"unknown tool: {name!r}")]

        if kind == "expand":
            handle = args.get("handle")
            text = self.expand(handle) if isinstance(handle, str) else None
            self.log.emit("expand", server, n=1 if text is not None else 0)
            if text is None:
                return [
                    _result(
                        msg_id,
                        _text_result(f"error: no original found for handle {handle!r}", True),
                    )
                ]
            return [_result(msg_id, _text_result(text))]

        if kind in ("schema", "invoke"):
            tool = args.get("tool_name")
            if not isinstance(tool, str) or tool not in surf.by_name:
                hint = surf.suggest(tool) if isinstance(tool, str) else []
                more = f" Did you mean: {', '.join(hint)}?" if hint else ""
                return [
                    _result(
                        msg_id, _text_result(f"error: no tool {tool!r} on '{server}'.{more}", True)
                    )
                ]
        assert tool is not None
        notes = self._unlock(surf, tool)
        if kind == "schema":
            self.log.emit("schema_fetch", server, tool=tool, level=surf.level)
            return [_result(msg_id, _text_result(surf.schema_text(tool))), *notes]
        call_args = args.get("arguments") if kind == "invoke" else args
        forward = {
            **params,
            "name": tool,
            "arguments": call_args if isinstance(call_args, dict) else {},
        }
        return [self._call(msg_id, server, surf, tool, forward, via=kind), *notes]

    def _unlock(self, surf: levels.Surface, tool: str) -> list[dict[str, Any]]:
        with self._lock:
            changed = surf.unlock(tool)
            if not changed:
                return []
            self._unlocked[surf.server] = list(surf.unlocked)
            self.list_changes += 1
        if self.persist:
            self._states[surf.server].update(
                self.session, unlocked=list(surf.unlocked), list_changed=True
            )
        self.log.emit("unlock", surf.server, tool=tool, n=len(surf.unlocked))
        return [dict(_LIST_CHANGED)]

    def _call(
        self,
        msg_id: Any,
        server: str,
        surf: levels.Surface,
        tool: str,
        params: dict[str, Any],
        via: str,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        resp = self.backends[server].request("tools/call", params, id=msg_id)
        ms = round((time.monotonic() - t0) * 1000, 1)
        if self.persist:
            self._states[server].update(self.session, used=tool)
        if "error" in resp or not self.results:
            self.log.emit(
                "call", server, tool=tool, ms=ms, skipped="error" if "error" in resp else None
            )
            return {**resp, "id": msg_id}
        result = resp.get("result")
        try:
            result, info = levels.compress_result(tool, result, record=self.record)
        except Exception as exc:  # noqa: BLE001 — fail open: the uncompressed result
            self.log.emit("error", server, tool=tool, err=type(exc).__name__)
            return {**resp, "id": msg_id}
        self.log.emit(
            "call",
            server,
            tool=tool,
            ms=ms,
            tokens_before=info.tokens_before,
            tokens_after=info.tokens_after,
            skipped=info.skipped,
            via=via,
        )
        return _result(msg_id, result)

    # -- relays --------------------------------------------------------------

    def _owner_for(self, method: str) -> Backend | None:
        if len(self.backends) == 1:
            return next(iter(self.backends.values()))
        family = _FAMILY.get(method.split("/", 1)[0])
        for name, backend in self.backends.items():
            if family and family in self.caps.get(name, {}):
                return backend
        return None

    def _relay(self, msg_id: Any, method: str, params: Any) -> dict[str, Any]:
        """Methods the proxy does not compress. List methods merge across servers."""
        if len(self.backends) > 1 and method in (
            "resources/list",
            "resources/templates/list",
            "prompts/list",
        ):
            key = {
                "resources/list": "resources",
                "resources/templates/list": "resourceTemplates",
                "prompts/list": "prompts",
            }[method]
            family = method.split("/", 1)[0]
            merged: list[Any] = []
            for name, b in self.backends.items():
                if family in self.caps.get(name, {}):
                    resp = b.request(method, params)
                    merged += (resp.get("result") or {}).get(key) or []
            return _result(msg_id, {key: merged})
        backend = self._owner_for(method)
        if backend is None:
            return _error(msg_id, -32601, f"method not found: {method!r}")
        return {**backend.request(method, params, id=msg_id), "id": msg_id}

    def _raw(self, msg_id: Any, method: str, params: Any) -> dict[str, Any]:
        """The fail-open path: what the client would have got without distil."""
        if method == "tools/list":
            tools: list[dict[str, Any]] = []
            for b in self.backends.values():
                tools += self._fetch_tools(b)
            return _result(msg_id, {"tools": tools})
        if method == "tools/call" and isinstance(params, dict):
            owner = self._raw_owner.get(str(params.get("name")))
            backend = self.backends.get(owner) if owner else self._owner_for(method)
            if backend is None:
                return _error(msg_id, -32602, f"unknown tool: {params.get('name')!r}")
            return {**backend.request(method, params, id=msg_id), "id": msg_id}
        backend = self._owner_for(method)
        if backend is None:
            return _error(msg_id, -32601, f"method not found: {method!r}")
        return {**backend.request(method, params, id=msg_id), "id": msg_id}

    # -- server -> client ----------------------------------------------------

    def on_backend_message(self, server: str, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        if "id" in msg and method is not None:  # a request for the client (roots, sampling…)
            new_id = f"distil-mcp-s{next(self._sreq_ids)}"
            with self._lock:
                self._server_requests[new_id] = (server, msg["id"])
            self.emit({**msg, "id": new_id})
            return
        if method == "notifications/tools/list_changed":
            with self._lock:
                old = self.surfaces.pop(server, None)
                if old is not None:
                    self._unlocked[server] = list(old.unlocked)
                self.list_changes += 1
            self.log.emit("list_changed", server)
        self.emit(msg)

    def _route_client_response(self, msg: dict[str, Any]) -> None:
        with self._lock:
            route = self._server_requests.pop(str(msg.get("id")), None)
        if route is None:
            return
        server, original = route
        backend = self.backends.get(server)
        if backend is not None:
            backend.send({**msg, "id": original})

    def close(self) -> None:
        for backend in self.backends.values():
            backend.close()


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------


def build(
    specs: list[ServerSpec], *, level: str, results: bool, session: str | None = None
) -> Proxy:
    backends: dict[str, Backend] = {}
    for spec in specs:
        backend = StdioBackend(spec)
        backend.start()
        backends[spec.name] = backend
    return Proxy(backends, level=level, results=results, session=session)


def serve(
    proxy: Proxy, stdin: IO[bytes] | None = None, stdout: IO[bytes] | None = None, workers: int = 8
) -> None:
    """Run the proxy over newline-delimited JSON-RPC until the client closes stdin."""
    src = stdin if stdin is not None else sys.stdin.buffer
    dst = stdout if stdout is not None else sys.stdout.buffer
    wlock = threading.Lock()

    def write(msg: dict[str, Any]) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with wlock:
            try:
                dst.write(data)
                dst.flush()
            except (OSError, ValueError):
                pass  # the client went away; nothing left to tell it

    def run(msg: dict[str, Any]) -> None:
        try:
            outs = proxy.handle(msg)
        except Exception as exc:  # noqa: BLE001 — never let one message end the session
            outs = (
                [_error(msg.get("id"), -32603, f"distil mcp: {type(exc).__name__}")]
                if msg.get("id") is not None
                else []
            )
        for out in outs:
            write(out)

    proxy.emit = write
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="distil-mcp")
    try:
        for raw in src:
            try:
                parsed = json.loads(raw)
            except ValueError:
                continue
            batch = parsed if isinstance(parsed, list) else [parsed]
            for msg in batch:
                if not isinstance(msg, dict):
                    continue
                # initialize and notifications run inline so ordering is preserved
                # (`notifications/initialized` must reach a backend after initialize).
                if (
                    msg.get("method") == "initialize"
                    or msg.get("id") is None
                    or "method" not in msg
                ):
                    run(msg)
                else:
                    pool.submit(run, msg)
    finally:
        pool.shutdown(wait=True)
        proxy.close()
