"""Preserved thinking, end to end: a stub upstream that VERIFIES thinking signatures.

The stub models the API's binding (platform.claude.com/docs/en/build-with-claude/
preserved-thinking): each thinking block it produces is recorded with the prefix it was
produced after — ``system``, ``tools`` and every message content block before it, thinking
blocks excluded, ``cache_control`` and key order ignored — and with the thinking block that
preceded it. A later request is accepted only if every thinking block it carries is one the
stub produced, its prefix is unchanged, and the kept blocks form an unbroken run of the
original sequence (removing the oldest or the newest is fine, a gap is not). Anything else
is the 400 a current account gets.

A simulated client then runs multi-turn sessions through a real distil proxy, appending
every assistant turn exactly as returned, across each way distil used to rewrite a prefix:
recency exit, intent change, drift-guard hold engage/lift, output shaping toggling with it,
a fail-open turn, a restart, a history over prefix replay's byte cap, and a hidden expand
round (buffered and streamed). The acceptance bar is zero 400s.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from distil import coldpoint, prefixreplay, thinkguard

MODEL = "claude-opus-5-5"
SYSTEM = [
    {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}
]
_HANDLE = re.compile(r"handle=([0-9a-f]{8})")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    coldpoint.reset()
    prefixreplay.reset()
    thinkguard.reset()
    monkeypatch.setenv("DISTIL_COLD_POINT", "0")  # cold-point has its own suite
    yield
    coldpoint.reset()
    prefixreplay.reset()
    thinkguard.reset()


# --------------------------------------------------------------------------- the API model


def _strip(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k != "cache_control"}
    if isinstance(node, list):
        return [_strip(x) for x in node]
    return node


def _text(b: Any) -> Any:
    if isinstance(b, str):
        return {"type": "text", "text": b}
    return _strip(b)


def _is_think(b: Any) -> bool:
    return isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")


def _prefix(body: dict[str, Any], i: int, j: int) -> str:
    """system + tools + messages[:i] + messages[i].content[:j], thinking excluded."""
    sysv = body.get("system")
    parts: list[Any] = [
        [_text(b) for b in sysv] if isinstance(sysv, list) else [_text(sysv)] if sysv else [],
        _strip(body.get("tools") or []),
    ]
    msgs = body.get("messages") or []
    for k, m in enumerate(msgs[: i + 1]):
        c = m["content"] if isinstance(m["content"], list) else [m["content"]]
        if k == i:
            c = c[:j]
        parts.append([m["role"], [_text(b) for b in c if not _is_think(b)]])
    return json.dumps(parts, sort_keys=True)


class Api:
    """What the stub upstream knows: every block it produced, and its own counters."""

    def __init__(self) -> None:
        self.produced: dict[str, tuple[str, str | None]] = {}  # sig -> (prefix, pred)
        self.errors: list[str] = []
        self.requests = 0
        self.expand_next = False  # answer the next request with a distil_expand call
        self.n = 0
        self.lock = threading.Lock()

    def check(self, body: dict[str, Any]) -> str | None:
        sent: list[str] = []
        for i, m in enumerate(body.get("messages") or []):
            c = m["content"] if isinstance(m["content"], list) else []
            for j, b in enumerate(c):
                if not _is_think(b):
                    continue
                sig = b.get("signature") or ""
                rec = self.produced.get(sig)
                if rec is None:
                    return f"messages.{i}.content.{j}: unknown signature"
                if rec[0] != _prefix(body, i, j):
                    return f"messages.{i}.content.{j}: prefix_binding_mismatch"
                if sent and sent[-1] != rec[1]:
                    return f"messages.{i}.content.{j}: thinking chain broken"
                sent.append(sig)
        return None

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.n += 1
            sig = f"sig-{self.n}"
            msgs = body["messages"]
            pred = None
            for m in msgs:
                for b in m["content"] if isinstance(m["content"], list) else []:
                    if _is_think(b):
                        pred = b["signature"]
            nxt = {**body, "messages": [*msgs, {"role": "assistant", "content": []}]}
            self.produced[sig] = (_prefix(nxt, len(msgs), 0), pred)
            think = {"type": "thinking", "thinking": "", "signature": sig}
            last = msgs[-1]["content"]
            last = last if isinstance(last, list) else []
            handles = _HANDLE.findall(json.dumps(msgs[:-1]))
            if self.expand_next and handles:
                self.expand_next = False
                blocks = [
                    think,
                    {"type": "text", "text": "let me look closer"},
                    {
                        "type": "tool_use",
                        "id": f"toolu_x{self.n}",
                        "name": "distil_expand",
                        "input": {"handle": handles[0]},
                    },
                ]
                return {"content": blocks, "stop_reason": "tool_use"}
            if any(b.get("type") == "tool_result" for b in last if isinstance(b, dict)):
                rounds = sum(1 for m in msgs if m["role"] == "user")
                if rounds % 3 == 0:
                    return {
                        "content": [think, {"type": "text", "text": "done"}],
                        "stop_reason": "end_turn",
                    }
            use = {
                "type": "tool_use",
                "id": f"toolu_{self.n}",
                "name": "Bash",
                "input": {"n": self.n},
            }
            return {"content": [think, use], "stop_reason": "tool_use"}


def _upstream(api: Api) -> type[BaseHTTPRequestHandler]:
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            with api.lock:
                api.requests += 1
                err = api.check(body)
            if err is not None:
                api.errors.append(err)
                return self._json(
                    400,
                    {"type": "error", "error": {"type": "invalid_request_error", "message": err}},
                )
            out = api.answer(body)
            msg = {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": out["content"],
                "stop_reason": out["stop_reason"],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            if body.get("stream"):
                return self._sse(msg)
            return self._json(200, msg)

        def _json(self, status: int, obj: Any) -> None:
            raw = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _sse(self, msg: dict[str, Any]) -> None:
            ev: list[dict[str, Any]] = [
                {"type": "message_start", "message": {**msg, "content": []}}
            ]
            for i, b in enumerate(msg["content"]):
                if b["type"] == "thinking":
                    ev += [
                        {
                            "type": "content_block_start",
                            "index": i,
                            "content_block": {**b, "signature": ""},
                        },
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "signature_delta", "signature": b["signature"]},
                        },
                    ]
                elif b["type"] == "text":
                    ev += [
                        {
                            "type": "content_block_start",
                            "index": i,
                            "content_block": {"type": "text", "text": ""},
                        },
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "text_delta", "text": b["text"]},
                        },
                    ]
                else:
                    ev += [
                        {
                            "type": "content_block_start",
                            "index": i,
                            "content_block": {**b, "input": {}},
                        },
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(b["input"]),
                            },
                        },
                    ]
                ev.append({"type": "content_block_stop", "index": i})
            ev += [
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": msg["stop_reason"]},
                    "usage": {"output_tokens": 5},
                },
                {"type": "message_stop"},
            ]
            raw = b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in ev)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    return H


def _read_sse(raw: bytes) -> dict[str, Any]:
    blocks: dict[int, dict[str, Any]] = {}
    inputs: dict[int, str] = {}
    stop = None
    for frame in raw.split(b"\n\n"):
        data = [ln[5:].strip() for ln in frame.split(b"\n") if ln.startswith(b"data:")]
        if not data:
            continue
        e = json.loads(b"".join(data))
        if e["type"] == "content_block_start":
            blocks[e["index"]] = dict(e["content_block"])
        elif e["type"] == "content_block_delta":
            d, b = e["delta"], blocks[e["index"]]
            if d["type"] == "signature_delta":
                b["signature"] = b.get("signature", "") + d["signature"]
            elif d["type"] == "text_delta":
                b["text"] = b.get("text", "") + d["text"]
            elif d["type"] == "input_json_delta":
                inputs[e["index"]] = inputs.get(e["index"], "") + d["partial_json"]
        elif e["type"] == "message_delta":
            stop = e["delta"].get("stop_reason")
    for i, s in inputs.items():
        blocks[i]["input"] = json.loads(s)
    return {"content": [blocks[i] for i in sorted(blocks)], "stop_reason": stop}


# --------------------------------------------------------------------------- the client


def _log(tag: str, n: int = 60) -> str:
    return "\n".join(
        f"{tag} line {i}: status=ok elapsed={i * 7}ms worker=w{i % 5} path=/srv/{tag}/m{i}.py"
        for i in range(n)
    )


class Client:
    """Appends every assistant turn exactly as returned, like Claude Code."""

    def __init__(self, port: int, *, marker: bool | str = False, stream: bool = False) -> None:
        self.port, self.marker, self.stream = port, marker, stream
        self.messages: list[dict[str, Any]] = []
        self.statuses: list[int] = []
        self.headers: list[dict[str, str]] = []

    def say(self, text: str) -> None:
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        self._run()

    def _body(self) -> dict[str, Any]:
        msgs = json.loads(json.dumps(self.messages))
        if self.marker == "lag":
            # A client whose breakpoint stays put: everything after it is uncached.
            msgs[min(2, len(msgs) - 1)]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        elif self.marker:
            msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        body = {
            "model": MODEL,
            "max_tokens": 256,
            "thinking": {"type": "adaptive"},
            "system": SYSTEM,
            "messages": msgs,
        }
        if self.stream:
            body["stream"] = True
        return body

    def _run(self) -> None:
        for _ in range(6):
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/v1/messages",
                data=json.dumps(self._body()).encode(),
                headers={"content-type": "application/json", "x-api-key": "sk-test"},
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    raw = r.read()
                    self.statuses.append(r.status)
                    self.headers.append({k.lower(): v for k, v in r.headers.items()})
            except urllib.error.HTTPError as e:
                self.statuses.append(e.code)
                return
            msg = _read_sse(raw) if self.stream else json.loads(raw)
            self.messages.append({"role": "assistant", "content": msg["content"]})
            uses = [b for b in msg["content"] if b.get("type") == "tool_use"]
            if msg["stop_reason"] != "tool_use" or not uses:
                return
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": u["id"], "content": _log(u["id"])}
                        for u in uses
                    ],
                }
            )


_SEEN: list[bytes] = []
_SERVERS: list[Any] = []  # [stub upstream] for the async-proxy test


def _count_thinking(msgs: list[dict[str, Any]]) -> int:
    return sum(
        1 for m in msgs if isinstance(m["content"], list) for b in m["content"] if _is_think(b)
    )


@pytest.fixture
def stack():
    api = Api()
    handler = _upstream(api)
    orig = handler.do_POST

    def spy(self):  # keep the forwarded bytes for the assertions
        n = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(n)
        _SEEN.append(raw)
        import io

        self.rfile = io.BytesIO(raw)
        return orig(self)

    handler.do_POST = spy  # type: ignore[method-assign]
    up = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    servers = [up]
    _SERVERS[:] = [up]
    _SEEN.clear()

    def proxy(**kw: Any) -> int:
        from distil.proxy import build_handler

        px = ThreadingHTTPServer(
            ("127.0.0.1", 0), build_handler(f"http://127.0.0.1:{up.server_address[1]}", **kw)
        )
        threading.Thread(target=px.serve_forever, daemon=True).start()
        servers.append(px)
        return px.server_address[1]

    yield api, proxy
    for s in servers:
        s.shutdown()


def _session(client: Client, turns: int, start: int = 0, *, vary: bool = False) -> None:
    for t in range(start, start + turns):
        # vary: each turn asks about a different file, so query-aware intent moves.
        client.say(f"now look at /srv/toolu_{t}/m{t}.py and fix it" if vary else f"continue {t}")


def _ok(api: Api, client: Client) -> None:
    assert api.errors == [], f"upstream rejected: {api.errors[:3]}"
    assert all(s == 200 for s in client.statuses), client.statuses


# --------------------------------------------------------------------------- scenarios


@pytest.mark.parametrize("marker", [False, True], ids=["no-marker", "marker"])
def test_recency_exit_and_redigest(stack, marker) -> None:
    """(2/6) a block leaves the recency window and would be re-digested."""
    api, proxy = stack
    c = Client(proxy(), marker=marker)
    _session(c, 6)
    _ok(api, c)
    assert _count_thinking(json.loads(_SEEN[-1])["messages"]) >= 10, "thinking was not kept"


def test_intent_change(stack) -> None:
    """(3) each turn's new intent would re-digest older uncached blocks."""
    api, proxy = stack
    c = Client(proxy(), marker="lag")
    _session(c, 6, vary=True)
    _ok(api, c)
    assert _count_thinking(json.loads(_SEEN[-1])["messages"]) >= 10


def _fake_guard(monkeypatch) -> list[bool]:
    from distil import drift

    flag = [False]

    class G:
        held = True
        uncertified = ""

        @property
        def engaged(self) -> bool:
            return flag[0]

    monkeypatch.setattr(drift.DriftGuard, "start", classmethod(lambda cls, **kw: G()))
    return flag


@pytest.mark.parametrize("shape", ["off", "light"])
def test_hold_engage_and_lift(stack, monkeypatch, shape) -> None:
    """(11) a hold flips every digest to verbatim and back; (10) with shaping on, the
    hold also toggled the directive appended to ``system``."""
    flag = _fake_guard(monkeypatch)
    api, proxy = stack
    c = Client(proxy(shape_output=shape), marker=True)
    _session(c, 3)
    flag[0] = True
    _session(c, 2, start=3)
    flag[0] = False
    _session(c, 2, start=5)
    _ok(api, c)
    assert _count_thinking(json.loads(_SEEN[-1])["messages"]) >= 10
    # (10) one system string for the whole lineage: shaping never toggles it.
    assert len({json.dumps(json.loads(r).get("system")) for r in _SEEN}) == 1


def test_fail_open_turn(stack, monkeypatch) -> None:
    """(12) a compressor that raises forwards the turn uncompressed after compressed ones."""
    from distil.adapters import anthropic as A

    api, proxy = stack
    c = Client(proxy(), marker=True)
    _session(c, 4)
    real = A.compress_messages

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("boom")

    import distil.serve_core as sc

    monkeypatch.setattr(sc, "compress_messages", boom)
    _session(c, 1, start=4)
    monkeypatch.setattr(sc, "compress_messages", real)
    _session(c, 2, start=5)
    _ok(api, c)
    assert _count_thinking(json.loads(_SEEN[-1])["messages"]) >= 10


def test_restart(stack) -> None:
    """(4) a restart or hot-swap loses prefix replay's in-memory state."""
    api, proxy = stack
    c = Client(proxy())
    _session(c, 4)
    prefixreplay.reset()
    thinkguard.reset()  # memory only; the ledger on disk survives, like a new process
    c.port = proxy()
    _session(c, 3, start=4)
    _ok(api, c)


def test_history_over_the_replay_cap(stack, monkeypatch) -> None:
    """(4) prefix replay forgets a lineage over its byte cap."""
    monkeypatch.setattr(prefixreplay, "_MAX_STATE_BYTES", 20_000)
    api, proxy = stack
    c = Client(proxy())
    _session(c, 6)
    _ok(api, c)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_expand_round(stack, stream) -> None:
    """(9) a hidden assistant[thinking, distil_expand] + user[tool_result] round: the
    continuation's thinking is signed over a prefix the client never sends.

    Prefix replay is off here so that history is digested at all (a frozen lineage keeps
    every block as first sent, verbatim, and so carries no handle to expand)."""
    api, proxy = stack
    c = Client(proxy(prefix_replay=False), stream=stream)
    _session(c, 3)
    api.expand_next = True
    _session(c, 1, start=3)
    assert not api.expand_next, "the scenario never produced an expand call"
    hidden = [
        s
        for s, (p, _) in api.produced.items()
        if '"name": "distil_expand", "type": "tool_use"' in p
    ]
    assert hidden, "no thinking was produced over a hidden round"
    _session(c, 3, start=4)
    _ok(api, c)
    for raw in _SEEN[-6:]:
        sent = {b.get("signature") for m in json.loads(raw)["messages"] for b in _blocks(m)}
        assert not sent & set(hidden), "thinking signed over the hidden round went back"


def _blocks(m: dict[str, Any]) -> list[Any]:
    return m["content"] if isinstance(m["content"], list) else []


def test_quote_guard_widening(stack, monkeypatch) -> None:
    """(8) a later Edit quoting an earlier read widens the exact-quote guard, which used to
    flip that already-digested read back to verbatim under thinking signed over the stub.

    The session is resumed with a long history the model produced without thinking, so its
    first request digests the old reads at once (first emission), and the cache marker
    sits on the newest block so nothing else moves."""
    from distil.adapters import anthropic as A

    api, proxy = stack
    c = Client(proxy(), marker=True)
    c.messages = [{"role": "user", "content": [{"type": "text", "text": "resume"}]}]
    for k in range(6):
        c.messages += [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"toolu_r{k}", "name": "Bash", "input": {"k": k}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"toolu_r{k}", "content": _log(f"r{k}")}
                ],
            },
        ]
    c.messages[-1]["content"].append({"type": "text", "text": "go on"})
    c._run()
    assert "handle=" in _SEEN[0].decode(), "the resumed history was not digested"
    real = A.exact_quote_tool_use_ids
    monkeypatch.setattr(
        A,
        "exact_quote_tool_use_ids",
        lambda m, **kw: {**real(m, **kw), **{f"toolu_r{k}": "Edit" for k in range(6)}},
    )
    _session(c, 3)
    _ok(api, c)
    assert _count_thinking(json.loads(_SEEN[-1])["messages"]) >= 4


def _aproxy(upstream_port: int) -> tuple[int, Any]:
    """The async proxy on a thread of its own, same wire as the threaded one."""
    import asyncio

    from aiohttp import web

    from distil.aproxy import make_app

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    box: dict[str, Any] = {}

    # Built here: aiohttp's AppKey reads the creating frame's module, absent on a thread.
    app = make_app(f"http://127.0.0.1:{upstream_port}", shape_output="light")

    async def start() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        box["port"] = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        box["runner"] = runner
        ready.set()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start())
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    assert ready.wait(10)

    def stop() -> None:
        asyncio.run_coroutine_threadsafe(box["runner"].cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)

    return box["port"], stop


def test_async_proxy_recency_and_hold(stack, monkeypatch) -> None:
    """The async proxy carries the same freeze and guard (no expand loop there)."""
    pytest.importorskip("aiohttp")
    flag = _fake_guard(monkeypatch)
    api, proxy = stack
    del proxy  # this one runs the async proxy instead
    port, stop = _aproxy(_SERVERS[0].server_address[1])
    try:
        c = Client(port)
        _session(c, 3)
        flag[0] = True
        _session(c, 2, start=3)
        flag[0] = False
        _session(c, 2, start=5)
    finally:
        stop()
    _ok(api, c)
    assert len({json.dumps(json.loads(r).get("system")) for r in _SEEN}) == 1
