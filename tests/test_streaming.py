"""Streaming pass-through — the proxy must deliver SSE bytes as they arrive.

Time-to-first-token is the product experience of an interactive agent; a proxy
that buffers a whole generation turns TTFT into time-to-last-token. These tests
run a dribbling upstream (chunk, pause, chunk) and assert the first chunk
reaches the client while the upstream is still sleeping — for the sync proxy,
the gateway, and the async proxy.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from distil.proxy import build_handler

_DELAY = 3.0  # upstream pause between chunks; first byte must beat this
# 0.6s previously: on a loaded shared CI runner (observed on windows-latest) the
# whole first-chunk round trip — thread spin-up, socket connect, HTTP framing —
# can itself take over a second with no buffering bug involved, which left only
# ~450ms of slack and flaked. 3.0s keeps the assertion meaningful (a genuinely
# buffered response still needs the full upstream pause before any byte arrives)
# while giving the streamed path enough absolute slack to absorb that jitter.
# Realistic Anthropic SSE deltas (with the `type` fields real streams carry) so the
# decision signature reconstructs to a real "text" decision, not "none" — shadow
# recording deliberately skips "none" signatures (transient/unparseable responses).
_CHUNK1 = (
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}\n\n'
)
_CHUNK2 = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


def _start_sse_upstream(delay: float = _DELAY) -> ThreadingHTTPServer:
    """An upstream that streams two SSE chunks with a pause in between
    (close-delimited, like a real SSE endpoint without Content-Length)."""

    class SSE(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(_CHUNK1)
            self.wfile.flush()
            time.sleep(delay)
            self.wfile.write(_CHUNK2)
            self.wfile.flush()

        def log_message(self, *a):  # noqa: ANN002
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), SSE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _stream_request(port: int, payload: bytes, path: str = "/v1/messages"):
    """POST via http.client and return (t_first_chunk, full_body, headers) with
    t_first_chunk measured from just after the request is sent."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    resp = conn.getresponse()
    first = b""
    t_first = None
    body = bytearray()
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        if t_first is None:
            t_first = time.monotonic() - t0
            first = chunk
        body += chunk
    conn.close()
    return t_first, bytes(body), dict(resp.headers), first


_DIGESTIBLE = "\n".join(f"log line {i}: benign filler output" for i in range(40))


def _payload(stream: bool = True) -> bytes:
    # The tool_result sits two turns back so the recency exemption does not keep
    # it verbatim — the request genuinely compresses and savings are non-zero
    # (0-savings flush windows write no ledger row by design).
    return json.dumps(
        {
            "model": "claude-opus-4-8",
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": _DIGESTIBLE}
                    ],
                },
                {"role": "user", "content": "next"},
                {"role": "user", "content": "hi"},
            ],
        }
    ).encode()


def test_sync_proxy_streams_sse_incrementally(tmp_path):
    upstream = _start_sse_upstream()
    handler = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        t_first, body, headers, first = _stream_request(proxy.server_address[1], _payload())
        # First chunk must arrive while the upstream is still sleeping — the
        # whole point. Generous margin so slow CI never flakes.
        assert t_first is not None and t_first < _DELAY * 0.75, (
            f"first chunk took {t_first:.2f}s — response was buffered, not streamed"
        )
        # This request digests, so it streams through the expand interceptor
        # (streamexpand), which reconstructs/normalizes SSE JSON framing to be able to
        # splice a distil_expand recovery — so assert on CONTENT, not exact bytes. The
        # streaming property itself (first chunk before the pause) is asserted above.
        assert b"hello" in body and b"message_stop" in body  # nothing lost
        assert b"message_stop" not in first  # ...and it genuinely arrived in two pieces
        assert headers.get("x-distil-compressed") == "1"  # compression still ran
    finally:
        proxy.shutdown()
        upstream.shutdown()


def test_sync_proxy_nonstream_response_keeps_content_length(tmp_path):
    """Regression: buffered (non-stream) responses still carry an exact
    Content-Length — HTTP/1.1 keep-alive framing must stay correct."""

    class Echo(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # noqa: ANN002
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    handler = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
        conn.request(
            "POST",
            "/v1/messages",
            body=_payload(stream=False),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 200
        assert resp.headers.get("Content-Length") == str(len(raw))
        conn.close()
    finally:
        proxy.shutdown()
        upstream.shutdown()


def test_sync_proxy_streaming_records_savings_and_shadow(tmp_path, monkeypatch):
    """Savings and shadow-mode still work on the streaming path (accounting is
    tee'd from the streamed bytes, never by re-buffering the response)."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil.runtime import RuntimeSavings

    upstream = _start_sse_upstream(delay=0.05)
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=tmp_path / "savings.jsonl")
    handler = build_handler(
        f"http://127.0.0.1:{upstream.server_address[1]}",
        savings=rs,
        flush_every=1,
        shadow_rate=1.0,
    )
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        long = "\n".join(f"verbose tool output line {i}" for i in range(40))
        payload = json.dumps(
            {
                "model": "claude-opus-4-8",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "user", "content": [{"type": "tool_result", "content": long}]},
                    # Two later turns keep the tool_result out of the recency-exempt
                    # window so it digests — 0-savings windows write no ledger row.
                    {"role": "user", "content": "next"},
                    {"role": "user", "content": "hi"},
                ],
            }
        ).encode()
        _t, body, headers, _f = _stream_request(proxy.server_address[1], payload)
        assert (
            b"message_stop" in body
        )  # digests → streams through streamexpand (normalized framing)
        assert headers.get("x-distil-shadow") == "sampled"
        # shadow thread records asynchronously — the replay itself re-drives the
        # SSE upstream (~2s of chunk delays), so a loaded CI runner needs far
        # more headroom than a fast local box; the poll exits early when healthy
        from distil.shadow import ShadowLedger

        deadline = time.monotonic() + 30
        led = ShadowLedger.load()
        # rc4 split sampling: 1/3 of draws replay A/A (ledger.aa_samples), the
        # rest A/B (ledger.samples) — a streamed request must produce a row of
        # EITHER kind; asserting on .samples alone fails on every A/A draw
        while led.samples + led.aa_samples == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
            led = ShadowLedger.load()
        assert led.samples + led.aa_samples >= 1  # streamed request produced a shadow row
        assert (tmp_path / "savings.jsonl").exists()  # savings flushed
    finally:
        proxy.shutdown()
        upstream.shutdown()


def test_gateway_streams_sse_incrementally(tmp_path):
    from distil.gateway import GatewayState, build_gateway_handler
    from distil.pricing import get as pricing_get

    upstream = _start_sse_upstream()
    price = pricing_get("claude-opus-4-8")
    handler = build_gateway_handler(
        f"http://127.0.0.1:{upstream.server_address[1]}",
        GatewayState(price),
        price,
    )
    gw = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=gw.serve_forever, daemon=True).start()
    try:
        t_first, body, _headers, _f = _stream_request(gw.server_address[1], _payload())
        assert t_first is not None and t_first < _DELAY * 0.75
        assert _CHUNK1 in body and _CHUNK2 in body
    finally:
        gw.shutdown()
        upstream.shutdown()


def test_aproxy_streams_sse_incrementally(tmp_path):
    aiohttp = __import__("pytest").importorskip("aiohttp")  # noqa: F841
    import asyncio

    from aiohttp import web

    from distil.aproxy import make_app

    async def _sse(request: web.Request) -> web.StreamResponse:
        sr = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await sr.prepare(request)
        await sr.write(_CHUNK1)
        await asyncio.sleep(_DELAY)
        await sr.write(_CHUNK2)
        await sr.write_eof()
        return sr

    async def _body() -> None:
        from aiohttp.test_utils import TestClient, TestServer

        fake = web.Application()
        fake.router.add_post("/v1/messages", _sse)
        async with TestServer(fake) as upstream_server:
            app = make_app(str(upstream_server.make_url("/")).rstrip("/"))
            async with TestClient(TestServer(app)) as client:
                t0 = time.monotonic()
                async with client.post(
                    "/v1/messages",
                    data=_payload(),
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    t_first = None
                    body = bytearray()
                    async for chunk in resp.content.iter_any():
                        if t_first is None:
                            t_first = time.monotonic() - t0
                        body += chunk
                assert t_first is not None and t_first < _DELAY * 0.75, (
                    f"first chunk took {t_first:.2f}s — aproxy buffered the stream"
                )
                assert _CHUNK1 in body and _CHUNK2 in body

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_body())
    finally:
        loop.close()


# --- D: streaming distil_expand interception, end to end -----------------------
def _sse(*events) -> bytes:
    out = b""
    for e in events:
        out += b"event: " + e["type"].encode() + b"\ndata: " + json.dumps(e).encode() + b"\n\n"
    return out


def _expand_turn(handle: str) -> bytes:
    return _sse(
        {
            "type": "message_start",
            "message": {
                "id": "m1",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 50, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu", "name": "distil_expand", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps({"handle": handle})},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    )


def _answer_turn(text: str) -> bytes:
    return _sse(
        {
            "type": "message_start",
            "message": {
                "id": "m2",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 60, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    )


def test_streaming_expand_intercept_end_to_end():
    """A streamed request that digests (→ a recoverable stub in context) must, when the
    model calls distil_expand, be intercepted and spliced: the client streams the
    re-query answer, never sees distil's internal tool, and the upstream is hit twice."""
    calls = {"n": 0}

    class SSE(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            calls["n"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                _expand_turn("deadbeef") if calls["n"] == 1 else _answer_turn("SPLICED-ANSWER")
            )
            self.wfile.flush()

        def log_message(self, *a):  # noqa: ANN002
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), SSE)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    handler = build_handler(f"http://127.0.0.1:{up.server_address[1]}", expand=True)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        _, body, _headers, _first = _stream_request(proxy.server_address[1], _payload())
        assert b"SPLICED-ANSWER" in body  # the client received the re-query's answer
        assert b"distil_expand" not in body  # never saw distil's internal recovery tool
        assert b"input_json_delta" not in body
        assert calls["n"] == 2  # first turn (expand call) + re-query (answer)
        assert body.count(b'"type": "message_start"') == 1  # one coherent message
    finally:
        proxy.shutdown()
        up.shutdown()
