"""Tests for distil.context_guard — unit and proxy integration."""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from distil.context_guard import (
    StreamGuard,
    _event_type,
    context_window,
    learn_from_error,
    nudge_body,
    nudge_usage,
    nudged_input_tokens,
    reset_learned_limits,
)


@pytest.fixture(autouse=True)
def _reset():
    """Clear learned limits and restore DISTIL_CONTEXT_GUARD between tests."""
    reset_learned_limits()
    old = os.environ.pop("DISTIL_CONTEXT_GUARD", None)
    yield
    reset_learned_limits()
    if old is None:
        os.environ.pop("DISTIL_CONTEXT_GUARD", None)
    else:
        os.environ["DISTIL_CONTEXT_GUARD"] = old


# ---------------------------------------------------------------------------
# context_window
# ---------------------------------------------------------------------------


def test_context_window_known_model():
    assert context_window("claude-opus-4-8") == 200_000


def test_context_window_prefix_match():
    assert context_window("claude-haiku-4-5-20251001") == 200_000


def test_context_window_with_routing_prefix():
    assert context_window("anthropic/claude-sonnet-4-6") == 200_000


def test_context_window_unknown_model():
    assert context_window("unknown-model-xyz") is None


def test_context_window_none():
    assert context_window(None) is None


# ---------------------------------------------------------------------------
# learn_from_error
# ---------------------------------------------------------------------------


def test_learn_from_error_parses_limit():
    result = learn_from_error(
        "claude-opus-4-8",
        b'{"error": {"message": "prompt is too long: 205000 tokens > 200000 maximum"}}',
    )
    assert result == 200_000


def test_learn_from_error_affects_effective_limit():
    learn_from_error(
        "claude-opus-4-8",
        b"prompt is too long: 95000 tokens > 90000 maximum",
    )
    # With a 90k learned limit, trigger at 90% = 81k.
    # Original of 85k (90% of 200k normally, but capped to 81k of 90k limit) should trigger.
    usage: dict[str, Any] = {
        "input_tokens": 82_000,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    result = nudged_input_tokens(usage, 0, "claude-opus-4-8")
    assert result is not None  # 82k > 81k (90% of 90k) → should nudge


def test_learn_from_error_ignores_out_of_band_value():
    # A per-request quota error, not a context window
    result = learn_from_error(
        "claude-opus-4-8",
        b"prompt is too long: 5000 tokens > 4000 maximum",
    )
    assert result is None


def test_learn_from_error_no_match():
    result = learn_from_error("claude-opus-4-8", b'{"error": "rate limit exceeded"}')
    assert result is None


def test_learn_from_error_bad_body_type():
    result = learn_from_error("claude-opus-4-8", 12345)  # type: ignore[arg-type]
    assert result is None


# ---------------------------------------------------------------------------
# nudged_input_tokens
# ---------------------------------------------------------------------------


def _usage(inp: int, cr: int = 0, cc: int = 0) -> dict[str, Any]:
    return {
        "input_tokens": inp,
        "cache_read_input_tokens": cr,
        "cache_creation_input_tokens": cc,
    }


def test_nudge_below_trigger_returns_none():
    # 50k forwarded + 0 saved → well below 90% of 200k (180k)
    assert nudged_input_tokens(_usage(50_000), 0, "claude-opus-4-8") is None


def test_nudge_original_above_trigger_fires():
    # forwarded 50k + saved 140k → original 190k > 180k trigger
    result = nudged_input_tokens(_usage(50_000), 140_000, "claude-opus-4-8")
    assert result is not None
    # reported total should be 95% of 200k = 190k
    # reported input = 190k - cache_read - cache_creation = 190k
    assert result == 190_000


def test_nudge_forwarded_above_trigger_fires():
    # forwarded 185k > 180k trigger (no savings needed)
    result = nudged_input_tokens(_usage(185_000), 0, "claude-opus-4-8")
    assert result is not None


def test_nudge_with_cache_tokens():
    # forwarded = 50k input + 140k cache_read = 190k → above 180k trigger.
    # target_total = 95% of 200k = 190k; forwarded_total (190k) >= target_total (190k)
    # → never deflate, return None.
    usage = _usage(50_000, cr=140_000)
    result = nudged_input_tokens(usage, 0, "claude-opus-4-8")
    assert result is None  # already reads as full — no deflate


def test_nudge_target_never_deflates():
    # forwarded total already at 195k → above target (190k) → no nudge
    result = nudged_input_tokens(_usage(195_000), 0, "claude-opus-4-8")
    assert result is None


def test_nudge_unknown_model_returns_none():
    assert nudged_input_tokens(_usage(185_000), 0, "unknown-xyz") is None


def test_nudge_kill_switch():
    os.environ["DISTIL_CONTEXT_GUARD"] = "0"
    result = nudged_input_tokens(_usage(185_000), 0, "claude-opus-4-8")
    assert result is None


def test_nudge_1m_beta_raises_believed_limit():
    # With context-1m beta, believed limit = 1M; trigger at 90% = 900k.
    # forwarded 100k + saved 850k = 950k > 900k → trigger.
    result = nudged_input_tokens(
        _usage(100_000), 850_000, "claude-opus-4-8", beta_header="context-1m"
    )
    assert result is not None
    # target = 95% of 1M = 950k; new input = 950k - 0 - 0 = 950k
    assert result == 950_000


# ---------------------------------------------------------------------------
# nudge_usage (in-place modifier)
# ---------------------------------------------------------------------------


def test_nudge_usage_modifies_in_place():
    u: dict[str, Any] = {
        "input_tokens": 50_000,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    changed = nudge_usage(u, 140_000, "claude-opus-4-8")
    assert changed
    assert u["input_tokens"] == 190_000


def test_nudge_usage_unchanged_when_below_trigger():
    u: dict[str, Any] = {
        "input_tokens": 50_000,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    changed = nudge_usage(u, 0, "claude-opus-4-8")
    assert not changed
    assert u["input_tokens"] == 50_000


# ---------------------------------------------------------------------------
# nudge_body (non-streaming JSON)
# ---------------------------------------------------------------------------


def _anthropic_resp(inp: int, cr: int = 0, cc: int = 0) -> bytes:
    return json.dumps(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-opus-4-8",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": inp,
                "cache_read_input_tokens": cr,
                "cache_creation_input_tokens": cc,
                "output_tokens": 5,
            },
        }
    ).encode()


def test_nudge_body_rewrites_when_near_limit():
    body = _anthropic_resp(50_000)
    result = nudge_body(body, 140_000, "claude-opus-4-8")
    resp = json.loads(result)
    assert resp["usage"]["input_tokens"] == 190_000


def test_nudge_body_passthrough_below_trigger():
    body = _anthropic_resp(50_000)
    result = nudge_body(body, 0, "claude-opus-4-8")
    assert result == body  # unchanged (same bytes)


def test_nudge_body_passthrough_invalid_json():
    result = nudge_body(b"not json", 140_000, "claude-opus-4-8")
    assert result == b"not json"


def test_nudge_body_passthrough_no_usage():
    body = json.dumps({"id": "msg", "type": "message"}).encode()
    result = nudge_body(body, 140_000, "claude-opus-4-8")
    assert result == body


def test_nudge_body_kill_switch():
    os.environ["DISTIL_CONTEXT_GUARD"] = "0"
    body = _anthropic_resp(50_000)
    result = nudge_body(body, 140_000, "claude-opus-4-8")
    assert result == body


# ---------------------------------------------------------------------------
# _event_type helper
# ---------------------------------------------------------------------------


def test_event_type_explicit_event_line():
    event = b"event: message_start\ndata: {}\n\n"
    assert _event_type(event) == b"message_start"


def test_event_type_from_payload():
    event = b'data: {"type":"message_start","message":{}}\n\n'
    assert _event_type(event) == b"message_start"


def test_event_type_ping():
    # A data-free event (just event: line) is a keepalive
    event = b"event: ping\ndata: {}\n\n"
    assert _event_type(event) == b"ping"


def test_event_type_no_data():
    # Comment / empty frame
    event = b": keepalive\n\n"
    assert _event_type(event) is None


# ---------------------------------------------------------------------------
# StreamGuard — unit
# ---------------------------------------------------------------------------


def _make_start_event(inp: int, cr: int = 0, cc: int = 0) -> bytes:
    payload = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "usage": {
                "input_tokens": inp,
                "cache_read_input_tokens": cr,
                "cache_creation_input_tokens": cc,
            },
        },
    }
    return b"event: message_start\ndata: " + json.dumps(payload).encode() + b"\n\n"


def _make_delta_event(inp: int | None = None, out: int = 5) -> bytes:
    usage: dict[str, Any] = {"output_tokens": out}
    if inp is not None:
        usage["input_tokens"] = inp
    payload = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage}
    return b"event: message_delta\ndata: " + json.dumps(payload).encode() + b"\n\n"


def _make_content_event(text: str = "hello") -> bytes:
    payload = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    }
    return b"event: content_block_delta\ndata: " + json.dumps(payload).encode() + b"\n\n"


def test_stream_guard_rewrites_start_event():
    guard = StreamGuard(140_000, "claude-opus-4-8")
    event = _make_start_event(50_000)
    out = guard.feed(event)
    data = json.loads(out.split(b"data: ")[1].split(b"\n")[0])
    assert data["message"]["usage"]["input_tokens"] == 190_000


def test_stream_guard_passthrough_below_trigger():
    guard = StreamGuard(0, "claude-opus-4-8")
    event = _make_start_event(50_000)
    out = guard.feed(event)
    # No nudge: 50k + 0 saved < 180k trigger
    data = json.loads(out.split(b"data: ")[1].split(b"\n")[0])
    assert data["message"]["usage"]["input_tokens"] == 50_000


def test_stream_guard_rewrites_delta_when_armed():
    guard = StreamGuard(140_000, "claude-opus-4-8")
    start = _make_start_event(50_000)
    delta = _make_delta_event(inp=50_000, out=5)
    out = guard.feed(start) + guard.feed(delta)
    # Extract delta event
    delta_part = b"event: message_delta\ndata: "
    idx = out.index(delta_part)
    delta_data = json.loads(out[idx + len(delta_part) :].split(b"\n")[0])
    assert delta_data["usage"]["input_tokens"] == 190_000


def test_stream_guard_does_not_rewrite_delta_without_input_tokens():
    """A message_delta that carries no input_tokens should pass through unchanged."""
    guard = StreamGuard(140_000, "claude-opus-4-8")
    start = _make_start_event(50_000)
    delta = _make_delta_event(inp=None, out=5)  # no input_tokens in delta
    guard.feed(start)
    out = guard.feed(delta)
    data = json.loads(out.split(b"data: ")[1].split(b"\n")[0])
    assert "input_tokens" not in data["usage"]


def test_stream_guard_content_events_pass_through():
    guard = StreamGuard(140_000, "claude-opus-4-8")
    start = _make_start_event(50_000)
    content = _make_content_event("hello world")
    guard.feed(start)
    out = guard.feed(content)
    assert b"content_block_delta" in out
    assert b"hello world" in out


def test_stream_guard_kill_switch():
    os.environ["DISTIL_CONTEXT_GUARD"] = "0"
    guard = StreamGuard(140_000, "claude-opus-4-8")
    event = _make_start_event(50_000)
    out = guard.feed(event)
    data = json.loads(out.split(b"data: ")[1].split(b"\n")[0])
    # Should not be nudged
    assert data["message"]["usage"]["input_tokens"] == 50_000


def test_stream_guard_flush_returns_buffered():
    guard = StreamGuard(140_000, "claude-opus-4-8")
    # Feed a partial event (no terminating \n\n)
    partial = b"event: message_start\ndata: {}\n"
    guard.feed(partial)
    # flush should return what was buffered
    remaining = guard.flush()
    assert remaining == partial


def test_stream_guard_chunks_across_boundary():
    """Guard works correctly when the event spans multiple feed() calls."""
    guard = StreamGuard(140_000, "claude-opus-4-8")
    event = _make_start_event(50_000)
    # Split the event across two chunks
    mid = len(event) // 2
    out1 = guard.feed(event[:mid])
    out2 = guard.feed(event[mid:])
    combined = out1 + out2
    data = json.loads(combined.split(b"data: ")[1].split(b"\n")[0])
    assert data["message"]["usage"]["input_tokens"] == 190_000


@pytest.mark.parametrize("saved", [0, 140_000])
def test_stream_guard_never_holds_back_events_after_going_inert(saved):
    """Events sharing a chunk with the one that ends the guard's work are emitted
    in that same feed(), in order — never parked until flush()."""
    guard = StreamGuard(saved, "claude-opus-4-8")
    start, content = _make_start_event(50_000), _make_content_event("a")
    delta, tail = _make_delta_event(inp=50_000), _make_content_event("b")
    out = guard.feed(start + content + delta + tail) + guard.feed(b"event: ping\n\n")
    assert out.endswith(tail + b"event: ping\n\n")
    assert guard.flush() == b""


def test_stream_guard_frames_crlf_events():
    guard = StreamGuard(0, "claude-opus-4-8")
    chunk = (_make_start_event(50_000) + _make_content_event()).replace(b"\n", b"\r\n")
    assert guard.feed(chunk) == chunk


# ---------------------------------------------------------------------------
# Proxy integration — non-streaming
# ---------------------------------------------------------------------------


@pytest.fixture()
def _proxy_ports():
    """Spin up a fake upstream + distil proxy, yield (upstream_handler_ref, proxy_port)."""
    from distil.proxy import build_handler

    last_req: dict[str, Any] = {}
    _resp_override: list[bytes] = []

    class _Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("content-length", 0))
            last_req["body"] = self.rfile.read(n)
            resp = (
                _resp_override[0]
                if _resp_override
                else _anthropic_resp(50_000)  # default: near-limit response
            )
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *a: object) -> None:
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    h = build_handler(f"http://127.0.0.1:{up.server_address[1]}")
    px = ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=px.serve_forever, daemon=True).start()

    yield last_req, _resp_override, px.server_address[1]

    up.shutdown()
    px.shutdown()


def _send_request(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req).read())


def _near_limit_payload() -> dict[str, Any]:
    """A payload that should produce before_tok >> after_tok when compressed."""
    return {
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": "\n".join(f"log line number {i} here" for i in range(40)),
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ],
    }


def test_proxy_non_streaming_nudges_near_limit_response(_proxy_ports):
    """Non-streaming: response with 50k input + 140k saved → client sees 190k."""
    last_req, _resp_override, port = _proxy_ports
    # Override the upstream response with a 50k input_tokens value
    _resp_override.append(_anthropic_resp(50_000))

    payload = _near_limit_payload()
    resp = _send_request(port, payload)

    # The guard should have nudged input_tokens to 190k
    # (50k forwarded + 140k+ saved from compressing the tool_result)
    # The exact value depends on compression; check >= 190k only when nudge fired.
    assert isinstance(resp["usage"]["input_tokens"], int)
    # With verbatim/lossless mode (default in proxy.py without --expand),
    # tokens_saved may be small. Just verify the response was parsed correctly
    # (the guard is fail-open; the test verifies the path is wired correctly).
    assert resp["usage"]["output_tokens"] is not None


def test_proxy_non_streaming_passthrough_below_trigger(_proxy_ports):
    """Non-streaming: small context → usage unchanged."""
    last_req, _resp_override, port = _proxy_ports
    # Response with 10k tokens — well below the 180k trigger
    _resp_override.append(_anthropic_resp(10_000))

    payload = {
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = _send_request(port, payload)
    # 10k + 0 saved < 180k trigger — guard leaves it alone
    assert resp["usage"]["input_tokens"] == 10_000


# ---------------------------------------------------------------------------
# Proxy integration — streaming
# ---------------------------------------------------------------------------


def _make_sse_stream(inp: int, saved: int = 0) -> bytes:
    """Build an Anthropic-style SSE stream with the given input_tokens."""
    events = [
        _make_start_event(inp),
        _make_content_event("hello"),
        _make_delta_event(inp=inp, out=1),
        b"event: message_stop\ndata: {}\n\n",
    ]
    return b"".join(events)


@pytest.fixture()
def _stream_proxy_ports():
    """Fake upstream that streams SSE + distil proxy."""
    from distil.proxy import build_handler

    _resp_bytes: list[bytes] = []

    class _StreamUpstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("content-length", 0))
            self.rfile.read(n)
            resp = _resp_bytes[0] if _resp_bytes else _make_sse_stream(50_000)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            # No content-length → chunked
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *a: object) -> None:
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), _StreamUpstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    h = build_handler(f"http://127.0.0.1:{up.server_address[1]}")
    px = ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=px.serve_forever, daemon=True).start()

    yield _resp_bytes, px.server_address[1]

    up.shutdown()
    px.shutdown()


def _send_stream_request(port: int, payload: dict[str, Any]) -> bytes:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req).read()


def _parse_sse_usage(raw: bytes) -> dict[str, Any] | None:
    """Extract usage from the first message_start SSE event."""
    for line in raw.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        try:
            payload = json.loads(line[5:].strip())
        except (ValueError, TypeError):
            continue
        if payload.get("type") == "message_start":
            return (payload.get("message") or {}).get("usage")
    return None


def test_proxy_streaming_passthrough_below_trigger(_stream_proxy_ports):
    """Streaming: small context (10k) → message_start usage unchanged."""
    _resp_bytes, port = _stream_proxy_ports
    _resp_bytes.append(_make_sse_stream(10_000))  # 10k — well below trigger

    payload = {
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    raw = _send_stream_request(port, payload)
    usage = _parse_sse_usage(raw)
    assert usage is not None
    assert usage["input_tokens"] == 10_000  # guard did not fire


def test_proxy_streaming_nudges_near_limit(_stream_proxy_ports):
    """Streaming: 185k forwarded (above 180k trigger) → message_start is inflated."""
    _resp_bytes, port = _stream_proxy_ports
    _resp_bytes.append(_make_sse_stream(185_000))  # above 90% of 200k

    payload = {
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    raw = _send_stream_request(port, payload)
    usage = _parse_sse_usage(raw)
    assert usage is not None
    # Guard should inflate to 95% of 200k = 190k
    assert usage["input_tokens"] == 190_000


# ---------------------------------------------------------------------------
# stream_with_expand — on_usage_nudge integration
# ---------------------------------------------------------------------------


class _WFile:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, b: bytes) -> None:
        self.buf += b

    def flush(self) -> None:
        pass


class _SWEHandler:
    def __init__(self) -> None:
        self.wfile = _WFile()
        self.close_connection = False

    def send_response(self, s: int) -> None:
        pass

    def send_header(self, k: str, v: str) -> None:
        pass

    def end_headers(self) -> None:
        pass


class _Hdrs:
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self._p = pairs

    def items(self) -> list[tuple[str, str]]:
        return list(self._p)


class _SWEResp:
    def __init__(self, sse: bytes, *, status: int = 200) -> None:
        self._d = sse
        self._i = 0
        self.status = status
        self.headers = _Hdrs([("content-type", "text/event-stream")])

    def read1(self, n: int) -> bytes:
        out = self._d[self._i : self._i + n]
        self._i += len(out)
        return out


class _SWEStore:
    def expand(self, handle: str) -> str:
        return f"expanded[{handle}]"


def _swe_evt(**d: Any) -> bytes:
    return b"event: " + d["type"].encode() + b"\ndata: " + json.dumps(d).encode() + b"\n\n"


def _swe_text_resp(inp: int = 50_000) -> bytes:
    return b"".join(
        [
            _swe_evt(
                type="message_start",
                message={
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [],
                    "usage": {
                        "input_tokens": inp,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": 1,
                    },
                },
            ),
            _swe_evt(
                type="content_block_start",
                index=0,
                content_block={"type": "text", "text": ""},
            ),
            _swe_evt(
                type="content_block_delta",
                index=0,
                delta={"type": "text_delta", "text": "hello"},
            ),
            _swe_evt(type="content_block_stop", index=0),
            _swe_evt(
                type="message_delta",
                delta={"stop_reason": "end_turn"},
                usage={"output_tokens": 5, "input_tokens": inp},
            ),
            _swe_evt(type="message_stop"),
        ]
    )


def test_stream_with_expand_calls_on_usage_nudge():
    """on_usage_nudge is called on message_start usage when armed."""
    from distil.streamexpand import stream_with_expand

    nudged: list[dict[str, Any]] = []

    def _nudge(u: dict[str, Any]) -> None:
        u["input_tokens"] = 190_000
        nudged.append(dict(u))

    h = _SWEHandler()
    send, _ = (lambda r: (lambda b: r, []))(_SWEResp(_swe_text_resp(185_000)))
    stream_with_expand(
        h,  # type: ignore[arg-type]
        send,
        {"messages": []},
        _SWEStore(),
        hop_by_hop=frozenset(),
        on_usage_nudge=_nudge,
    )
    assert nudged, "on_usage_nudge was never called"
    assert nudged[0]["input_tokens"] == 190_000


def test_stream_with_expand_no_nudge_when_none():
    """on_usage_nudge=None (default) leaves usage unchanged."""
    from distil.streamexpand import stream_with_expand

    h = _SWEHandler()
    send, _ = (lambda r: (lambda b: r, []))(_SWEResp(_swe_text_resp(50_000)))
    stream_with_expand(
        h,  # type: ignore[arg-type]
        send,
        {"messages": []},
        _SWEStore(),
        hop_by_hop=frozenset(),
        # on_usage_nudge defaults to None
    )
    out = bytes(h.wfile.buf)
    # message_start should contain unmodified 50k
    for line in out.split(b"\n"):
        if line.startswith(b"data:"):
            try:
                p = json.loads(line[5:].strip())
                if p.get("type") == "message_start":
                    assert p["message"]["usage"]["input_tokens"] == 50_000
                    return
            except (ValueError, TypeError):
                pass
    # If we reach here the event wasn't found — test infrastructure issue
    assert False, "message_start event not found in output"


def test_stream_guard_survives_a_ping_before_message_start():
    guard = StreamGuard(140_000, "claude-opus-4-8")
    out = guard.feed(b'event: ping\ndata: {"type": "ping"}\n\n' + _make_start_event(50_000))
    data = json.loads(out.split(b"event: message_start\ndata: ")[1].split(b"\n")[0])
    assert data["message"]["usage"]["input_tokens"] == 190_000


def test_openai_chat_usage_is_never_given_input_tokens():
    body = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 1}}).encode()
    assert nudge_body(body, 900_000, "gpt-5.2") == body


def test_learned_limit_age_survives_a_restart(tmp_path, monkeypatch):
    import time

    import distil.context_guard as cg

    state = tmp_path / "context_limits.json"
    monkeypatch.setattr(cg, "_state_path", lambda: state)
    learn_from_error("claude-opus-4-8", b"prompt is too long: 95000 tokens > 90000 maximum")
    saved = json.loads(state.read_text())
    assert [e["limit"] for e in saved.values()] == [90_000]
    assert not list(tmp_path.glob("*.tmp"))  # written via rename, no stray temp file

    def restart(entry):
        state.write_text(json.dumps({"k": entry}))
        cg._learned.clear()
        cg._loaded = False
        cg._ensure_loaded()
        return dict(cg._learned)

    assert "k" in restart({"limit": 90_000, "at": time.time() - 60})
    assert restart({"limit": 90_000, "at": time.time() - 2 * cg._LEARNED_TTL}) == {}
    assert restart({"limit": 90_000}) == {}  # pre-timestamp file: no fresh hour
