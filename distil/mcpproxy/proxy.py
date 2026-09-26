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

import contextlib
import itertools
import json
import re
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
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
    safe = [levels.safe_server(sp.name) for sp in specs]
    dup = sorted({n for n in safe if safe.count(n) > 1})
    if dup:
        raise ConfigError(f"server names collide once sanitised to [A-Za-z0-9-]: {', '.join(dup)}")
    clash = levels.exposed_names_clash([sp.name for sp in specs], len(specs) > 1)
    if clash:
        raise ConfigError(f"server names would share tool names: {', '.join(sorted(set(clash)))}")
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
#: ``_meta`` key naming the server a relayed server-to-client request came from.
ORIGIN_KEY = "io.distil/server"
#: Bounds on the id maps a misbehaving client or server could otherwise grow forever.
MAX_SERVER_REQUESTS = 256
MAX_INFLIGHT = 1024
MAX_HANDLES = 4096  # per server: originals its expand tool may return this session
#: Pages followed per server when a list is merged server-side.
MAX_PAGES = 100
# A resource template routes only if its literal prefix names at least a scheme.
_ROUTABLE_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_AMBIGUOUS = "\0ambiguous"


class _Bounded(OrderedDict):  # type: ignore[type-arg]
    """An insertion-ordered dict that evicts its oldest entry past ``cap``."""

    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        while len(self) > self.cap:
            self.popitem(last=False)


def _params(msg: dict[str, Any]) -> dict[str, Any]:
    raw = msg.get("params")
    return raw if isinstance(raw, dict) else {}


def _label_sampling(params: dict[str, Any], server: str) -> dict[str, Any]:
    """Prefix the first text a sampling request would show with the server it came from."""
    msgs = params.get("messages")
    if not isinstance(msgs, list):
        return params
    out = list(msgs)
    for i, m in enumerate(out):
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, dict) and content.get("type") == "text":
            label = f"[sampling request from MCP server '{server}'] "
            out[i] = {**m, "content": {**content, "text": label + str(content.get("text", ""))}}
            break
    return {**params, "messages": out}


class Proxy:
    """One client session. ``handle`` is synchronous and thread-safe.

    With more than one backend every tool, meta tool and prompt is namespaced
    ``<server>__<name>`` (``levels.safe_server`` has no ``_``, so the first ``__`` always
    ends the server part), and calls are routed through ONE explicit name→server table
    built in config order. No backend can claim another's names, and a
    ``list_changed`` never reorders who owns what. A single backend keeps its tools'
    real names: there is nothing for it to shadow.
    """

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
        safe = [levels.safe_server(n) for n in backends]
        if len(set(safe)) != len(safe):
            raise ValueError(f"server names collide once sanitised: {', '.join(backends)}")
        if levels.exposed_names_clash(backends, len(backends) > 1):
            raise ValueError("server names would share meta-tool names")
        self.backends = backends
        self.order = list(backends)  # config order: the only routing priority, ever
        self.multi = len(backends) > 1
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
        self.routes: dict[str, tuple[str, str, str | None]] = {}
        self.caps: dict[str, dict[str, Any]] = {}
        self.list_changes = 0
        self._unlocked: dict[str, list[str]] = {}
        self._handles: dict[str, _Bounded] = {n: _Bounded(MAX_HANDLES) for n in backends}
        self._progress: _Bounded = _Bounded(MAX_INFLIGHT)  # progressToken -> server
        self._server_requests: _Bounded = _Bounded(MAX_SERVER_REQUESTS)
        self._inflight: _Bounded = _Bounded(MAX_INFLIGHT)
        self._res_owner: dict[str, str] = {}
        self._tmpl_owner: dict[str, str] = {}
        self._res_known = False  # has resources/list been merged from every server?
        self._tmpl_known = False  # has resources/templates/list?
        self._sreq_ids = itertools.count(1)
        self._ids = itertools.count(1)
        # Guards surfaces/routes. Never held across backend I/O: a backend's reader
        # thread takes it (list_changed) while a caller may be waiting on that backend.
        self._lock = threading.RLock()
        self._maplock = threading.Lock()  # the id maps only; never held across I/O
        self._tl = threading.local()
        self._states = {n: events.ServerState(n, state_root) for n in backends}
        for name, backend in backends.items():
            backend.on_message = self._relay_from(name)

    def _relay_from(self, server: str) -> Callable[[dict[str, Any]], None]:
        return lambda msg: self.on_backend_message(server, msg)

    def _ns(self, server: str) -> str:
        return f"{levels.safe_server(server)}{levels.NAMESPACE_SEP}" if self.multi else ""

    def _request(
        self, server: str, method: str, params: Any, client_id: Any = None
    ) -> dict[str, Any]:
        """Send to a backend under an INTERNAL id; remember it for cancellation."""
        rid = f"distil-mcp-c{next(self._ids)}"
        meta = params.get("_meta") if isinstance(params, dict) else None
        token = meta.get("progressToken") if isinstance(meta, dict) else None
        with self._maplock:
            if client_id is not None:
                # A list per client id: a client reusing an id concurrently must not make
                # one call's entry overwrite (or its cleanup delete) the other's.
                routes = self._inflight.get(client_id) or []
                self._inflight[client_id] = [*routes, (server, rid)]
            if isinstance(token, (str, int)):
                self._progress[token] = server
        try:
            return self.backends[server].request(method, params, id=rid)
        finally:
            with self._maplock:
                if client_id is not None:
                    left = [r for r in self._inflight.get(client_id) or [] if r[1] != rid]
                    if left:
                        self._inflight[client_id] = left
                    else:
                        self._inflight.pop(client_id, None)
                if isinstance(token, (str, int)):
                    self._progress.pop(token, None)

    # -- client -> proxy ----------------------------------------------------

    def handle(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Handle one client message; return what to send back, in order."""
        method = msg.get("method")
        msg_id = msg.get("id")
        if method is None:
            self._route_client_response(msg)
            return []
        if msg_id is None:
            self._notify(method, msg.get("params"))
            return []
        params = msg.get("params")
        self._tl.sent = False
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
        except Exception as exc:  # noqa: BLE001 — fail open, but never twice
            self.log.emit("error", "*", err=type(exc).__name__)
            if getattr(self._tl, "sent", False):
                # The call already reached a server. Re-sending it could run a
                # non-idempotent tool twice, so this is an error, not a retry.
                return [
                    _error(
                        msg_id,
                        -32603,
                        "distil mcp: failed after the server was called; not retried",
                    )
                ]
            try:
                return [self._raw(msg_id, method, params)]
            except BackendError as inner:
                return [_error(msg_id, -32603, str(inner))]

    def _notify(self, method: str, params: Any) -> None:
        if method == "notifications/cancelled" and isinstance(params, dict):
            with self._maplock:
                routes = list(self._inflight.get(params.get("requestId")) or [])
            for server, rid in routes:  # only the server(s) running it, under their ids
                self.backends[server].notify(method, {**params, "requestId": rid})
            return
        for backend in self.backends.values():
            backend.notify(method, params)

    def _initialize(self, msg_id: Any, params: Any) -> dict[str, Any]:
        first: dict[str, Any] | None = None
        instructions: list[str] = []
        caps: dict[str, Any] = {}
        for name in list(self.order):
            resp = self._request(name, "initialize", params)
            if "error" in resp:
                if len(self.backends) == 1:
                    return {"jsonrpc": "2.0", "id": msg_id, "error": resp["error"]}
                self.log.emit("error", name, err="initialize")
                del self.backends[name]
                self.order.remove(name)
                continue
            res = resp.get("result") or {}
            first = first or res
            self.caps[name] = res.get("capabilities") or {}
            for k, v in self.caps[name].items():
                caps.setdefault(k, v)
            if isinstance(res.get("instructions"), str):
                text = res["instructions"]
                instructions.append(f"[{name}] {text}" if self.multi else text)
        if first is None:
            return _error(msg_id, -32603, "no MCP server behind distil mcp initialized")
        caps["tools"] = {**(caps.get("tools") or {}), "listChanged": True}
        from .. import __version__

        out: dict[str, Any] = {
            "protocolVersion": first.get("protocolVersion") or mcp_server.DEFAULT_PROTOCOL,
            "capabilities": caps,
            "serverInfo": {
                "name": f"distil-mcp[{','.join(self.order)}]",
                "version": __version__,
            },
        }
        if instructions:
            out["instructions"] = "\n\n".join(instructions)
        return _result(msg_id, out)

    # -- tools/list ----------------------------------------------------------

    def _fetch_tools(self, server: str) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = None
        for _ in range(100):  # a server paging forever must not hang the session
            resp = self._request(server, "tools/list", {"cursor": cursor} if cursor else {})
            if "error" in resp:
                raise BackendError(f"{server}: tools/list failed: {resp['error']}")
            res = resp.get("result") or {}
            tools += [t for t in res.get("tools") or [] if isinstance(t, dict)]
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return tools

    def _surfaces(self) -> dict[str, levels.Surface]:
        with self._lock:
            missing = [n for n in self.order if n not in self.surfaces]
            if not missing:
                return dict(self.surfaces)
        # Backend I/O happens OUTSIDE the lock (see ``_lock``).
        fetched = {n: self._fetch_tools(n) for n in missing}
        states = {n: self._states[n] for n in missing}
        usage = {n: states[n].usage() for n in missing} if self.level == "L3" else {}
        saved = {n: states[n].unlocked(self.session) for n in missing}
        built: list[levels.Surface] = []
        with self._lock:
            for name in missing:
                if name in self.surfaces:
                    continue  # another thread built it meanwhile
                tools = fetched[name]
                names = [str(t.get("name")) for t in tools]
                pins = levels.choose_pins(usage[name], names) if self.level == "L3" else frozenset()
                surf = levels.Surface(
                    name,
                    tools,
                    self.level,
                    self.results,
                    pinned=pins,
                    prefix=self._ns(name),
                    unlocked=self._unlocked.get(name) or saved[name],
                    sep=levels.NAMESPACE_SEP if self.multi else "_",
                )
                self.surfaces[name] = surf
                built.append(surf)
            shadowed = self._build_routes()
            out = dict(self.surfaces)
        for server, tool in shadowed:
            self.log.emit("shadowed", server, tool=tool)
        if self.persist:
            for surf in built:
                self._snapshot(surf)
        return out

    def _build_routes(self) -> list[tuple[str, str]]:
        """The name→server table, in config order. Returns every refused name."""
        routes: dict[str, tuple[str, str, str | None]] = {}
        refused: list[tuple[str, str]] = []
        for name in self.order:
            surf = self.surfaces.get(name)
            if surf is None:
                continue
            refused += [(name, n) for n in surf.shadowed if (name, n) not in refused]
            for exposed, kind, tool in surf.exposure():
                if exposed in routes:
                    refused.append((name, tool or exposed))
                    if tool is not None:
                        surf.drop(tool)
                    continue
                routes[exposed] = (name, kind, tool)
        self.routes = routes
        return refused

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
        surfaces = self._surfaces()
        for name in self.order:
            surf = surfaces.get(name)
            if surf is None:
                continue
            listed = surf.tools_list()
            out += listed
            if self.log.enabled:
                before = sum(levels.definition_tokens(t) for t in surf.tools)
                after = sum(levels.definition_tokens(t) for t in listed)
                self.log.emit(
                    "list",
                    name,
                    level=surf.level,
                    tokens_before=before,
                    tokens_after=after,
                    n=len(listed),
                )
        return out

    # -- tools/call ----------------------------------------------------------

    def _tools_call(self, msg_id: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        name = str(params.get("name") or "")
        args = params.get("arguments")
        args = args if isinstance(args, dict) else {}
        surfaces = self._surfaces()
        with self._lock:
            route = self.routes.get(name)
        if route is None:
            if not self.multi:  # the one backend is the authority on its own tools
                return [self._raw(msg_id, "tools/call", params)]
            return [_error(msg_id, -32602, f"unknown tool: {name!r}")]
        server, kind, tool = route
        surf = surfaces[server]

        if kind == "expand":
            handle = args.get("handle")
            with self._maplock:
                known = isinstance(handle, str) and handle in self._handles[server]
            text = self.expand(handle) if known and isinstance(handle, str) else None
            self.log.emit("expand", server, n=1 if text is not None else 0)
            if text is None:
                return [
                    _result(
                        msg_id,
                        _text_result(
                            f"error: no original for handle {handle!r} from '{server}' in this session",
                            True,
                        ),
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
            unlocked = list(surf.unlocked)
        if self.persist:
            self._states[surf.server].update(self.session, unlocked=unlocked, list_changed=True)
        self.log.emit("unlock", surf.server, tool=tool, n=len(unlocked))
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
        self._tl.sent = True  # from here on, a failure must not re-send this call
        resp = self._request(server, "tools/call", params, client_id=msg_id)
        raw = {**resp, "id": msg_id}
        try:
            ms = round((time.monotonic() - t0) * 1000, 1)
            if self.persist:
                self._states[server].update(self.session, used=tool)
            if "error" in resp or not self.results:
                self.log.emit(
                    "call", server, tool=tool, ms=ms, skipped="error" if "error" in resp else None
                )
                return raw

            def record(handle: str, original: str) -> bool:
                ok = self.record(handle, original)
                if ok:
                    with self._maplock:
                        self._handles[server][handle] = True
                return ok

            result, info = levels.compress_result(
                tool, resp.get("result"), record=record, tool_def=surf.by_name.get(tool)
            )
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
        except Exception as exc:  # noqa: BLE001 — fail open to the server's own answer
            with contextlib.suppress(Exception):
                self.log.emit("error", server, tool=tool, err=type(exc).__name__)
            return raw

    # -- relays --------------------------------------------------------------

    def _relay(self, msg_id: Any, method: str, params: Any) -> dict[str, Any]:
        """Methods the proxy does not compress, routed to the server that OWNS the thing."""
        if not self.multi:
            server = self.order[0]
            return {**self._request(server, method, params, client_id=msg_id), "id": msg_id}
        p = params if isinstance(params, dict) else {}
        if method in ("resources/list", "resources/templates/list", "prompts/list"):
            return _result(msg_id, self._merged_list(method, params))
        if method == "logging/setLevel":
            for name in self.order:
                if "logging" in self.caps.get(name, {}):
                    self._request(name, method, params)
            return _result(msg_id, {})
        if method in ("resources/read", "resources/subscribe", "resources/unsubscribe"):
            owner = self._resource_owner(str(p.get("uri", "")))
            if owner is None:
                return _error(msg_id, -32602, f"no single server owns resource {p.get('uri')!r}")
            return {**self._request(owner, method, params, client_id=msg_id), "id": msg_id}
        if method == "prompts/get":
            split = self._split(str(p.get("name", "")))
            if split is None:
                return _error(msg_id, -32602, f"unknown prompt: {p.get('name')!r}")
            server, prompt = split
            return {
                **self._request(server, method, {**p, "name": prompt}, client_id=msg_id),
                "id": msg_id,
            }
        if method == "completion/complete":
            raw_ref = p.get("ref")
            ref: dict[str, Any] = raw_ref if isinstance(raw_ref, dict) else {}
            if ref.get("type") == "ref/prompt":
                split = self._split(str(ref.get("name", "")))
                if split is None:
                    return _error(msg_id, -32602, f"unknown prompt: {ref.get('name')!r}")
                server, prompt = split
                fwd = {**p, "ref": {**ref, "name": prompt}}
                return {**self._request(server, method, fwd, client_id=msg_id), "id": msg_id}
            owner = self._resource_owner(str(ref.get("uri", "")))
            if owner is None:
                return _error(msg_id, -32602, "no single server owns that completion target")
            return {**self._request(owner, method, params, client_id=msg_id), "id": msg_id}
        return _error(msg_id, -32601, f"method not found: {method!r}")

    def _split(self, namespaced: str) -> tuple[str, str] | None:
        """``<server>__<name>`` → (server, name), for a server this proxy fronts."""
        head, sep, rest = namespaced.partition(levels.NAMESPACE_SEP)
        if not sep:
            return None
        for name in self.order:
            if levels.safe_server(name) == head:
                return name, rest
        return None

    def _merged_list(self, method: str, params: Any) -> dict[str, Any]:
        """Every capable server's COMPLETE list, merged here (all pages followed).

        The merge is server-side, so the client never gets a cursor from this proxy; a
        client cursor is therefore not one of ours, and gets an empty page rather than
        being forwarded to servers it means nothing to.
        """
        key = {
            "resources/list": "resources",
            "resources/templates/list": "resourceTemplates",
            "prompts/list": "prompts",
        }[method]
        if isinstance(params, dict) and params.get("cursor"):
            return {key: []}
        family = method.split("/", 1)[0]
        merged: list[Any] = []
        owners: dict[str, str] = {}
        for name in self.order:
            if family not in self.caps.get(name, {}):
                continue
            cursor = None
            for _ in range(MAX_PAGES):
                page = self._request(name, method, {"cursor": cursor} if cursor else {})
                res = page.get("result") or {}
                for item in res.get(key) or []:
                    if not isinstance(item, dict):
                        continue
                    if key == "prompts":
                        item = {**item, "name": self._ns(name) + str(item.get("name", ""))}
                    elif key == "resourceTemplates":
                        prefix = str(item.get("uriTemplate") or "").split("{", 1)[0]
                        if not _ROUTABLE_PREFIX.match(prefix):
                            # A catch-all template ("{uri}") would claim every URI.
                            self.log.emit("shadowed", name, tool="resource-template")
                            continue
                        owners[prefix] = _AMBIGUOUS if owners.get(prefix, name) != name else name
                    else:
                        uri = str(item.get("uri") or "")
                        owners[uri] = _AMBIGUOUS if owners.get(uri, name) != name else name
                    merged.append(item)
                cursor = res.get("nextCursor")
                if not cursor:
                    break
        with self._maplock:
            if key == "resources":
                self._res_owner = owners
                self._res_known = True
            elif key == "resourceTemplates":
                self._tmpl_owner = owners
                self._tmpl_known = True
        return {key: merged}

    def _resource_owner(self, uri: str) -> str | None:
        """The ONE server that can own *uri*, else None.

        Every claim counts — an exact listing AND every template whose literal prefix
        matches. No kind of claim outranks another: if two servers claim the URI in any
        combination, it is ambiguous. (An exact listing used to win outright, which let
        one server list another's file URI and receive the reads.)
        """
        # BOTH kinds of claim, from ALL servers, before deciding. Populated-ness is tracked
        # per list type: a client that asked for templates first must not leave exact
        # listings unknown (a broad template would then capture a URI another server
        # listed explicitly), and an empty listing is still a fetched one.
        with self._maplock:
            need_res, need_tmpl = not self._res_known, not self._tmpl_known
        if need_res:
            self._merged_list("resources/list", {})
        if need_tmpl:
            self._merged_list("resources/templates/list", {})
        with self._maplock:
            claims = {s for p, s in self._tmpl_owner.items() if uri.startswith(p)}
            if uri in self._res_owner:
                claims.add(self._res_owner[uri])
        if _AMBIGUOUS in claims or len(claims) != 1:
            return None
        return claims.pop()

    def _raw(self, msg_id: Any, method: str, params: Any) -> dict[str, Any]:
        """The fail-open path, for requests that have NOT reached a server yet."""
        if method == "tools/list":
            tools: list[dict[str, Any]] = []
            for name in self.order:
                ns = self._ns(name)
                tools += [{**t, "name": ns + str(t.get("name"))} for t in self._fetch_tools(name)]
            return _result(msg_id, {"tools": tools})
        if method == "tools/call" and isinstance(params, dict):
            name = str(params.get("name"))
            if self.multi:
                split = self._split(name)
                if split is None:
                    return _error(msg_id, -32602, f"unknown tool: {name!r}")
                server, tool = split
                params = {**params, "name": tool}
            else:
                server = self.order[0]
            self._tl.sent = True
            return {**self._request(server, method, params, client_id=msg_id), "id": msg_id}
        if not self.multi:
            return {**self._request(self.order[0], method, params, client_id=msg_id), "id": msg_id}
        return _error(msg_id, -32603, f"distil mcp could not route {method!r}")

    # -- server -> client ----------------------------------------------------

    def on_backend_message(self, server: str, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        if "id" in msg and method is not None:  # a request for the client (roots, sampling…)
            new_id = f"distil-mcp-s{next(self._sreq_ids)}"
            with self._maplock:
                self._server_requests[new_id] = (server, msg["id"])
            raw_params = msg.get("params")
            params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
            raw_meta = params.get("_meta")
            meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
            params = {**params, "_meta": {**meta, ORIGIN_KEY: server}}
            if method == "elicitation/create" and isinstance(params.get("message"), str):
                params["message"] = f"[{server}] {params['message']}"
            if method == "sampling/createMessage":
                params = _label_sampling(params, server)
            self.emit({**msg, "id": new_id, "params": params})
            return
        if method == "notifications/cancelled":
            # A server may only cancel ITS OWN request to the client, under our id for it.
            p = _params(msg)
            with self._maplock:
                ours = next(
                    (
                        k
                        for k, v in self._server_requests.items()
                        if v == (server, p.get("requestId"))
                    ),
                    None,
                )
            if ours is not None:
                self.emit({**msg, "params": {**p, "requestId": ours}})
            return
        if method == "notifications/progress":
            p = _params(msg)
            with self._maplock:
                owner = self._progress.get(p.get("progressToken"))
            if owner == server:  # progress for somebody else's call is dropped
                self.emit(msg)
            return
        if method == "notifications/resources/list_changed":
            with self._maplock:  # re-derive every owner on the next read
                self._res_owner = {}
                self._tmpl_owner = {}
                self._res_known = False
                self._tmpl_known = False
        if method == "notifications/tools/list_changed":
            with self._lock:
                old = self.surfaces.pop(server, None)
                if old is not None:
                    self._unlocked[server] = list(old.unlocked)
                self.list_changes += 1
            self.log.emit("list_changed", server)
        self.emit(msg)

    def _route_client_response(self, msg: dict[str, Any]) -> None:
        with self._maplock:
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

    def write(msg: Any) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with wlock:
            try:
                dst.write(data)
                dst.flush()
            except (OSError, ValueError):
                pass  # the client went away; nothing left to tell it

    def outcome(msg: Any) -> list[dict[str, Any]]:
        if not isinstance(msg, dict):
            return [_error(None, -32600, "Invalid Request")]
        try:
            return proxy.handle(msg)
        except Exception as exc:  # noqa: BLE001 — never let one message end the session
            if msg.get("id") is None:
                return []
            return [_error(msg.get("id"), -32603, f"distil mcp: {type(exc).__name__}")]

    def run(msg: Any) -> None:
        for out in outcome(msg):
            write(out)

    def run_batch(batch: list[Any]) -> None:
        """JSON-RPC 2.0 batch: ONE array holding every response, in request order.

        Notifications (and client responses) produce no entry; an all-notification batch
        produces no reply at all. Server notifications a call triggered (``list_changed``)
        are not responses, so they follow the array as ordinary lines.
        """
        replies: list[dict[str, Any]] = []
        extra: list[dict[str, Any]] = []
        for msg in batch:
            for out in outcome(msg):
                (replies if "method" not in out else extra).append(out)
        if replies:
            write(replies)
        for out in extra:
            write(out)

    proxy.emit = write
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="distil-mcp")
    try:
        for raw in src:
            if not raw.strip():
                continue
            try:
                parsed = json.loads(raw)
            except ValueError:
                write(_error(None, -32700, "Parse error"))
                continue
            if isinstance(parsed, list):
                if not parsed:  # the spec's answer to an empty batch
                    write(_error(None, -32600, "Invalid Request"))
                elif any(isinstance(m, dict) and m.get("method") == "initialize" for m in parsed):
                    run_batch(parsed)
                else:
                    pool.submit(run_batch, parsed)
                continue
            msg = parsed
            # initialize and notifications run inline so ordering is preserved
            # (`notifications/initialized` must reach a backend after initialize).
            if not isinstance(msg, dict) or (
                msg.get("method") == "initialize" or msg.get("id") is None or "method" not in msg
            ):
                run(msg)
            else:
                pool.submit(run, msg)
    finally:
        pool.shutdown(wait=True)
        proxy.close()
