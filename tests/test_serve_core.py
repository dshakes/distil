"""One compress-or-forward path for proxy and gateway (ADR 0023).

Three properties, each through real servers against a recording upstream:

* parity — the SAME request sequence through ``proxy.build_handler`` and
  ``gateway.build_gateway_handler`` forwards byte-identical bodies, with cold-point
  recompression and cache-delta coding both exercised;
* fail-open — a compressor that raises inside the shared path still forwards the
  request, on both servers;
* tenant isolation — one gateway tenant can never expand another tenant's handle, nor
  share its cache-delta session or cold-point lineage.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from distil import cachedelta, coldpoint, serve_core
from distil.expand import EXPAND_TOOL_NAME
from distil.gateway import GatewayState, build_gateway_handler
from distil.pricing import get as pricing_get
from distil.proxy import build_handler

_HANDLE = re.compile(r"handle=([0-9a-f]{8})")


class _Upstream(BaseHTTPRequestHandler):
    """Records every forwarded body. Answers ``end_turn`` — unless the request carries
    ``x-test-expand: <handle>`` and has not yet been answered, in which case it asks for
    that handle via distil_expand; the follow-up turn's answer echoes what came back."""

    seen: list[bytes] = []

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).seen.append(raw)
        body = json.loads(raw)
        last = (body.get("messages") or [{}])[-1]
        content: list[dict[str, Any]]
        want = self.headers.get("x-test-expand")
        answered = [
            b
            for b in (last.get("content") if isinstance(last.get("content"), list) else [])
            if isinstance(b, dict) and b.get("tool_use_id") == "tu_expand"
        ]
        if answered:
            content = [{"type": "text", "text": str(answered[0].get("content"))}]
        elif want:
            content = [
                {
                    "type": "tool_use",
                    "id": "tu_expand",
                    "name": EXPAND_TOOL_NAME,
                    "input": {"handle": want},
                }
            ]
        else:
            content = [{"type": "text", "text": "ok"}]
        payload = json.dumps(
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "content": content,
                "model": "claude-test",
                "stop_reason": "tool_use" if content[0]["type"] == "tool_use" else "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a: object) -> None:
        pass


@pytest.fixture
def servers():
    """``make("proxy"|"gateway", **kw) -> port``, all against one recording upstream."""
    _Upstream.seen = []
    coldpoint.reset()
    cachedelta.reset_sessions()
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    running = [up]
    url = f"http://127.0.0.1:{up.server_address[1]}"

    def make(kind: str, **kw: Any) -> int:
        if kind == "proxy":
            # Output shaping is a proxy-only lever (the gateway shapes nothing).
            handler = build_handler(url, shape_output="off", **kw)
        else:
            # Parity is about the opted-in gateway: digest is `--digest` (ADR 0023).
            kw.setdefault("digest", True)
            price = pricing_get("claude-opus-4-8")
            handler = build_gateway_handler(
                url, GatewayState(price), price, trust_tenant_header=True, **kw
            )
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        running.append(srv)
        return srv.server_address[1]

    yield make
    for s in running:
        s.shutdown()
    coldpoint.reset()
    cachedelta.reset_sessions()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(coldpoint, "_clock", lambda: now[0])
    return now


def _settle() -> None:
    """Wait for the handler's deferred ``coldpoint.end()`` (it runs after the relay)."""
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with coldpoint._LOCK:
            if all(st.inflight <= 0 for st in coldpoint._STATES.values()):
                return
        time.sleep(0.005)


def _post(
    port: int,
    body: dict[str, Any],
    *,
    path: str = "/v1/messages",
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": "sk-test", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        resp = json.loads(r.read())
        hdrs = {k.lower(): v for k, v in r.headers.items()}
    _settle()
    return hdrs, resp


def _log(tag: str, n: int = 40) -> str:
    return "\n".join(
        f"{tag} line {i}: status=ok elapsed={i * 7}ms worker=w{i % 5} path=/srv/app/mod_{i}.py"
        for i in range(n)
    )


def _conv(rounds: int, *, head: str = "fix the build", tag: str = "run") -> dict[str, Any]:
    """A Claude-Code-shaped session: one Bash round per turn, moving breakpoint."""
    msgs: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": head}]}]
    for k in range(rounds):
        msgs += [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"toolu_{k}", "name": "Bash", "input": {"k": k}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{k}",
                        "content": _log(f"{tag}{k}"),
                    }
                ],
            },
        ]
    msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return {
        "model": "claude-test",
        "max_tokens": 64,
        "system": [
            {
                "type": "text",
                "text": "You are a coding agent.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": msgs,
    }


def _reread(body: dict[str, Any]) -> dict[str, Any]:
    """*body* plus one more Bash round that re-reads the first round's output."""
    msgs = json.loads(json.dumps(body["messages"]))
    msgs[-1]["content"][-1].pop("cache_control", None)
    msgs += [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_rr", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_rr",
                    "content": _log("run0"),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]
    return {**body, "messages": msgs}


def _chat(rounds: int) -> dict[str, Any]:
    msgs: list[dict[str, Any]] = [{"role": "user", "content": "fix the build"}]
    for k in range(rounds):
        msgs += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{k}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call_{k}", "content": _log(f"chat{k}")},
        ]
    return {"model": "gpt-5.2", "messages": msgs}


# --------------------------------------------------------------------------- parity


def _run(make, kind: str, clock: list[float], sequence: list[tuple[float, str, dict]], **kw):
    port = make(kind, **kw)
    start = len(_Upstream.seen)
    heads = []
    for advance, path, body in sequence:
        clock[0] += advance
        h, _ = _post(port, body, path=path, headers={"x-distil-tenant": "t1"})
        heads.append(h)
    return _Upstream.seen[start:], heads


def test_cold_point_parity(servers, clock) -> None:
    """Warm turns, a cold gap, then warm again: both servers evict the same blocks on
    the same turn and forward byte-identical bodies on every turn."""
    seq = [(0, "/v1/messages", _conv(3)), (120, "/v1/messages", _conv(4))]
    seq += [(20 * 60, "/v1/messages", _conv(5)), (60, "/v1/messages", _conv(6))]
    seq += [(0, "/v1/chat/completions", _chat(4))]

    proxy_bodies, proxy_heads = _run(servers, "proxy", clock, seq)
    coldpoint.reset()
    gw_bodies, gw_heads = _run(servers, "gateway", clock, seq)

    assert [h.get("x-distil-cold") for h in proxy_heads[:4]] == [
        "first-seen",
        "warm",
        "cold",
        "warm",
    ]
    assert [h.get("x-distil-cold") for h in gw_heads] == [
        h.get("x-distil-cold") for h in proxy_heads
    ]
    assert "distil evicted" in proxy_bodies[2].decode(), "fixture no longer exercises cold-point"
    assert gw_bodies == proxy_bodies
    for key in ("x-distil-tokens-saved", "x-distil-compressible-tokens", "x-distil-mode"):
        assert [h[key] for h in gw_heads] == [h[key] for h in proxy_heads], key


@pytest.mark.parametrize("refetch", ["0", "1"])
def test_cache_delta_parity(servers, clock, monkeypatch, refetch) -> None:
    """--session-delta: a re-read turn is delta-coded identically on both servers. The
    re-read repeats a folded block, so with re-fetch verbatim on (ADR 0025) it is kept
    whole instead of referenced — identically on both servers too."""
    monkeypatch.setenv("DISTIL_REFETCH_VERBATIM", refetch)
    seq = [(0, "/v1/messages", _conv(3)), (5, "/v1/messages", _reread(_conv(3)))]
    seq += [(5, "/v1/chat/completions", _chat(3)), (5, "/v1/chat/completions", _chat(3))]

    proxy_bodies, proxy_heads = _run(servers, "proxy", clock, seq, session_delta=True)
    cachedelta.reset_sessions()
    gw_bodies, gw_heads = _run(servers, "gateway", clock, seq, session_delta=True)

    assert all("x-distil-cache-refs" in h for h in proxy_heads), "delta did not run"
    if refetch == "0":
        assert int(proxy_heads[1]["x-distil-cache-refs"]) > 0, "fixture no longer dedups"
    else:
        assert proxy_heads[1]["x-distil-cache-refs"] == "0", "a re-fetch was referenced"
        last = json.loads(proxy_bodies[1])["messages"][-1]["content"][-1]["content"]
        assert last == _log("run0"), "the re-fetch did not go out whole"
    assert gw_bodies == proxy_bodies
    for key in ("x-distil-cache-refs", "x-distil-cache-delta", "x-distil-cache-prefix-msgs"):
        assert [h[key] for h in gw_heads] == [h[key] for h in proxy_heads], key


@pytest.mark.parametrize("kind", ["proxy", "gateway"])
def test_fail_open_from_the_shared_path(servers, monkeypatch, kind) -> None:
    """Every compressor and planner raising: the request is still served, forwarding
    the client's own messages."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("compressor exploded")

    monkeypatch.setattr(serve_core, "compress_messages", boom)
    monkeypatch.setattr(coldpoint, "plan", boom)
    body = _conv(4)
    h, resp = _post(servers(kind), body, headers={"x-distil-tenant": "t1"})
    assert resp["stop_reason"] == "end_turn"
    assert json.loads(_Upstream.seen[-1])["messages"] == body["messages"]
    assert h["x-distil-tokens-saved"] == "0"


# --------------------------------------------------------------------------- tenants


def test_a_tenant_cannot_expand_another_tenants_handle(servers) -> None:
    port = servers("gateway")
    secret = _conv(4, head="tenant A's private repo")
    _post(port, secret, headers={"x-distil-tenant": "alice"})
    stubbed = _Upstream.seen[-1].decode()
    handles = _HANDLE.findall(stubbed)
    assert handles, "fixture no longer digests"
    h = handles[0]

    # Bob forges a stub naming Alice's handle in his own history and the model asks
    # for it: the answer is the miss placeholder, never Alice's bytes.
    forged = _conv(1, head=f"<< +40 lines, handle={h} >>", tag="bob")
    _, resp = _post(port, forged, headers={"x-distil-tenant": "bob", "x-test-expand": h})
    text = resp["content"][0]["text"]
    assert "no original found" in text
    assert "run0 line" not in text

    # Positive control: Alice's own expand of the same handle recovers it.
    _, resp = _post(port, secret, headers={"x-distil-tenant": "alice", "x-test-expand": h})
    assert "status=ok" in resp["content"][0]["text"]
    # ...and her expansion shapes only HER future cold-point evictions.
    assert h in coldpoint.expanded("alice\0")
    assert h not in coldpoint.expanded("bob\0")


def test_tenants_never_share_delta_sessions_or_cold_lineages(servers, clock) -> None:
    port = servers("gateway", session_delta=True)
    _post(port, _conv(3), headers={"x-distil-tenant": "alice"})
    h2, _ = _post(port, _reread(_conv(3)), headers={"x-distil-tenant": "alice"})
    # Alice's second turn continues HER session: its prefix is byte-stable vs turn one.
    assert int(h2["x-distil-cache-prefix-msgs"]) > 0, "fixture no longer exercises delta"
    # The same turn from Bob opens a fresh session — nothing of Alice's is "previous".
    h3, _ = _post(port, _reread(_conv(3)), headers={"x-distil-tenant": "bob"})
    assert h3["x-distil-cache-prefix-msgs"] == "0", "bob's request continued alice's session"

    cold_port = servers("gateway")
    _post(cold_port, _conv(3), headers={"x-distil-tenant": "alice"})
    clock[0] += 60
    hb, _ = _post(cold_port, _conv(4), headers={"x-distil-tenant": "bob"})
    assert hb["x-distil-cold"] == "first-seen", "bob inherited alice's lineage"


# --------------------------------------------------------------------------- unit


def test_admit_rejection_leaves_no_state() -> None:
    coldpoint.reset()
    out = serve_core.compress_or_forward(
        _conv(4),
        "/v1/messages",
        count=lambda m: 1,
        verbatim=False,
        expand=True,
        mode="digest",
        cold=True,
        scope="t\0",
        admit=lambda n: False,
    )
    assert out is None
    assert not coldpoint._STATES


def test_a_raise_after_planning_closes_the_lineage(monkeypatch) -> None:
    coldpoint.reset()

    def boom(body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("injection exploded")

    monkeypatch.setattr("distil.expand.inject_expand_tool", boom)
    with pytest.raises(RuntimeError):
        serve_core.compress_or_forward(
            _conv(4),
            "/v1/messages",
            count=lambda m: 1,
            verbatim=False,
            expand=True,
            mode="digest",
            cold=True,
            scope="t\0",
        )
    assert coldpoint._STATES and all(st.inflight == 0 for st in coldpoint._STATES.values())


def test_tenant_view_refuses_ungranted_handles() -> None:
    from distil.adapters.anthropic import RestoreStore

    grants = serve_core.TenantHandles()
    store = RestoreStore(persist=False)
    store._record("deadbeef", "alice's bytes")
    alice = grants.view("alice", store)
    assert alice.expand("deadbeef") == "alice's bytes"
    with pytest.raises(KeyError):
        grants.view("bob", None).expand("deadbeef")
    with pytest.raises(KeyError):
        grants.view("bob", RestoreStore(persist=False)).expand("deadbeef")
