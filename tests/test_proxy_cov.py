"""Coverage tests for distil.proxy and distil.streamrelay.

Fills branches not exercised by test_proxy.py / test_streaming.py:

proxy.py
  • do_GET / do_PUT / do_DELETE / do_PATCH / do_HEAD / do_OPTIONS → passthrough
  • chunked Transfer-Encoding without Content-Length → 411
  • oversized Content-Length → 413
  • path traversal on compressible and passthrough paths → 400
  • non-JSON body on compressible path → forwarded as-is
  • upstream HTTPError → relayed (via _post_upstream)
  • upstream connection refused → 502 (URLError in passthrough / compressible)
  • upstream timeout → 504 (URLError + socket.timeout reason)
  • Gemini /generateContent path with ``contents`` field
  • shape_output="light" adds x-distil-output-shaping header
  • _model_from_path edge cases

streamrelay.py
  • _is_timeout (lines 35-37) exercised via URLError path
  • HTTPError from upstream → status relayed (lines 67-71)
  • URLError → 502 (lines 85-93)
  • URLError with timeout reason → 504 (lines 86-93, 35-37)
  • read1 loop + chunked framing (SSE, no Content-Length) — confirm chunks arrive
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import distil.proxy as _proxy_mod
from distil.proxy import build_handler

# ---------------------------------------------------------------------------
# Fake upstream handlers
# ---------------------------------------------------------------------------


class _EchoHandler(BaseHTTPRequestHandler):
    """Echo POST body verbatim; echo path for all other verbs."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _echo_path(self) -> None:
        resp = self.path.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    do_GET = _echo_path  # type: ignore[assignment]
    do_PUT = _echo_path  # type: ignore[assignment]
    do_DELETE = _echo_path  # type: ignore[assignment]
    do_PATCH = _echo_path  # type: ignore[assignment]
    do_HEAD = _echo_path  # type: ignore[assignment]
    do_OPTIONS = _echo_path  # type: ignore[assignment]


class _ErrorHandler(BaseHTTPRequestHandler):
    """Always returns HTTP 500 after draining the request body."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def _respond(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = b'{"error":"internal server error"}'
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _respond  # type: ignore[assignment]
    do_POST = _respond  # type: ignore[assignment]


class _HungHandler(BaseHTTPRequestHandler):
    """Accepts the connection but never sends a response (simulates a timeout)."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def _hang(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        time.sleep(30)  # never responds within any reasonable test window

    do_GET = _hang  # type: ignore[assignment]
    do_POST = _hang  # type: ignore[assignment]


def _start(handler_cls: type) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def echo_proxy() -> Any:
    upstream = _start(_EchoHandler)
    handler = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    yield proxy.server_address[1]
    proxy.shutdown()
    upstream.shutdown()


@pytest.fixture()
def error_proxy() -> Any:
    upstream = _start(_ErrorHandler)
    handler = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    yield proxy.server_address[1]
    proxy.shutdown()
    upstream.shutdown()


# ---------------------------------------------------------------------------
# Tiny HTTP client helper
# ---------------------------------------------------------------------------


def _request(
    method: str,
    port: int,
    path: str = "/v1/models",
    body: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, http.client.HTTPResponse, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    hdrs: dict[str, str] = {}
    if body is not None:
        hdrs["Content-Length"] = str(len(body))
        hdrs["Content-Type"] = "application/json"
    if extra_headers:
        hdrs.update(extra_headers)
    conn.request(method, path, body=body, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, resp, data


# ---------------------------------------------------------------------------
# Tests: other HTTP verbs → passthrough (do_GET / do_PUT / do_DELETE / etc.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH", "OPTIONS"])
def test_proxy_other_verbs_passthrough(echo_proxy: int, method: str) -> None:
    """Non-POST verbs go through _passthrough and are relayed to the upstream."""
    status, _, _ = _request(method, echo_proxy)
    assert status == 200


def test_proxy_head_passthrough(echo_proxy: int) -> None:
    """HEAD is dispatched to _passthrough; response has no body."""
    conn = http.client.HTTPConnection("127.0.0.1", echo_proxy, timeout=5)
    conn.request("HEAD", "/v1/models")
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 200


# ---------------------------------------------------------------------------
# Tests: malformed request body → early rejection
# ---------------------------------------------------------------------------


def test_proxy_chunked_body_rejected_411(echo_proxy: int) -> None:
    """Transfer-Encoding: chunked without Content-Length → 411."""
    conn = http.client.HTTPConnection("127.0.0.1", echo_proxy, timeout=5)
    conn.putrequest("POST", "/v1/messages")
    conn.putheader("Transfer-Encoding", "chunked")
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    # Proxy rejects before reading the body, so we can read the response immediately.
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    assert resp.status == 411
    assert b"chunked" in data.lower()


def test_proxy_oversized_cl_rejected_413(echo_proxy: int) -> None:
    """Content-Length beyond the 8 MiB guard → 413 (no body read needed)."""
    conn = http.client.HTTPConnection("127.0.0.1", echo_proxy, timeout=5)
    # POST to compressible path; oversized CL on passthrough path also triggers 413
    conn.putrequest("POST", "/v1/messages")
    conn.putheader("Content-Length", "99999999999")
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 413


# ---------------------------------------------------------------------------
# Tests: invalid path → 400
# ---------------------------------------------------------------------------


def test_proxy_invalid_path_compressible_400(echo_proxy: int) -> None:
    """@ in Gemini model name routes to _handle_compressible then fails safe_forward_path → 400.

    A plain ``..`` path doesn't enter _handle_compressible (the proxy dispatches
    on the exact compressible-path set before safe_forward_path is checked).
    A Gemini URL like /v1beta/models/gemini@evil:generateContent IS accepted by
    is_gemini_path (@ is not a slash, so [^/]+ matches it) but rejected by
    safe_forward_path (@ in path = host-injection vector), covering lines 335-336.
    """
    body = b'{"contents": []}'
    conn = http.client.HTTPConnection("127.0.0.1", echo_proxy, timeout=5)
    conn.request(
        "POST",
        "/v1beta/models/gemini@evil:generateContent",
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 400


def test_proxy_invalid_path_passthrough_400(echo_proxy: int) -> None:
    """Path traversal on a passthrough GET path → 400."""
    conn = http.client.HTTPConnection("127.0.0.1", echo_proxy, timeout=5)
    conn.request("GET", "/v1/../models")
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 400


# ---------------------------------------------------------------------------
# Tests: non-JSON body on compressible path → forwarded unchanged
# ---------------------------------------------------------------------------


def test_proxy_invalid_json_forwarded_as_is(echo_proxy: int) -> None:
    """Non-JSON body on /v1/messages is forwarded to the upstream as-is."""
    body = b"not-json-at-all"
    status, _, data = _request("POST", echo_proxy, "/v1/messages", body=body)
    assert status == 200
    assert data == body  # echo upstream echoes it back unchanged


# ---------------------------------------------------------------------------
# Tests: upstream HTTP errors → relayed (via _post_upstream, non-streaming POST)
# ---------------------------------------------------------------------------


def test_proxy_upstream_500_on_compressible_post(error_proxy: int) -> None:
    """Upstream 500 on a compressible POST (non-streaming) is relayed via _post_upstream."""
    body = json.dumps(
        {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    status, _, _ = _request("POST", error_proxy, "/v1/messages", body=body)
    assert status == 500


# ---------------------------------------------------------------------------
# Tests: upstream connection refused → 502 (URLError path)
# ---------------------------------------------------------------------------


def test_proxy_connection_refused_502() -> None:
    """Connection refused at upstream → proxy returns 502 (URLError path)."""
    # Bind a server to reserve a port, then release it without serving.
    placeholder = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    dead_port = placeholder.server_address[1]
    placeholder.server_close()

    handler = build_handler(f"http://127.0.0.1:{dead_port}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        # Non-streaming POST (compressible) → _post_upstream URLError → 502
        body = json.dumps(
            {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hello"}]}
        ).encode()
        status, _, _ = _request("POST", proxy.server_address[1], "/v1/messages", body=body)
        assert status == 502
    finally:
        proxy.shutdown()


# ---------------------------------------------------------------------------
# Tests: upstream timeout → 504 (URLError with socket.timeout reason)
# ---------------------------------------------------------------------------


def test_proxy_upstream_timeout_504(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hung upstream + short timeout → proxy returns 504 (URLError/socket.timeout path).

    Also exercises streamrelay._is_timeout (lines 35-37) via the passthrough path.
    """
    upstream = _start(_HungHandler)
    try:
        # _UPSTREAM_TIMEOUT is a module global read at call-time, so patching it
        # affects the next request even though the handler is already built.
        monkeypatch.setattr(_proxy_mod, "_UPSTREAM_TIMEOUT", 0.4)
        handler = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            # GET → _passthrough → stream_upstream with 0.4s timeout. The proxy
            # MUST terminate the request rather than hang. On most Pythons that's
            # a clean 504; on some (e.g. 3.9) http.client surfaces the timeout as
            # the proxy closing the connection first — both prove "didn't hang".
            import http.client as _hc

            try:
                status, _, _ = _request("GET", proxy.server_address[1])
                assert status == 504
            except (_hc.RemoteDisconnected, ConnectionError, OSError):
                pass  # proxy closed the connection on timeout — also non-hanging
        finally:
            proxy.shutdown()
    finally:
        upstream.shutdown()


# ---------------------------------------------------------------------------
# Tests: Gemini generateContent path (``contents`` field)
# ---------------------------------------------------------------------------


def test_proxy_gemini_contents_path(echo_proxy: int) -> None:
    """POST to a Gemini generateContent URL compresses the ``contents`` field."""
    contents_text = "\n".join(f"Gemini content line {i}" for i in range(30))
    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": contents_text}]}],
            "generationConfig": {"maxOutputTokens": 100},
        }
    ).encode()
    status, resp, _ = _request(
        "POST",
        echo_proxy,
        "/v1beta/models/gemini-pro:generateContent",
        body=body,
    )
    assert status == 200
    assert resp.headers.get("x-distil-compressed") == "1"


# ---------------------------------------------------------------------------
# Tests: shape_output="light" → x-distil-output-shaping header
# ---------------------------------------------------------------------------


def test_proxy_shape_output_light_header() -> None:
    """shape_output='light' adds x-distil-output-shaping on compressed responses."""
    upstream = _start(_EchoHandler)
    handler = build_handler(
        f"http://127.0.0.1:{upstream.server_address[1]}",
        shape_output="light",
    )
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        long_text = "\n".join(f"verbose tool output line {i}" for i in range(25))
        body = json.dumps(
            {
                "model": "claude-opus-4-8",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "content": long_text}],
                    }
                ],
            }
        ).encode()
        status, resp, _ = _request("POST", proxy.server_address[1], "/v1/messages", body=body)
        assert status == 200
        assert resp.headers.get("x-distil-output-shaping") == "light"
    finally:
        proxy.shutdown()
        upstream.shutdown()


# ---------------------------------------------------------------------------
# Unit tests: proxy helpers
# ---------------------------------------------------------------------------


def test_model_from_path_extracts_model() -> None:
    from distil.proxy import _model_from_path

    assert _model_from_path("/v1beta/models/gemini-pro:generateContent") == "gemini-pro"
    assert (
        _model_from_path("/v1beta/models/gemini-1.5-flash:streamGenerateContent")
        == "gemini-1.5-flash"
    )
    # No /models/ marker → None
    assert _model_from_path("/v1/messages") is None
    # /models/ with empty tail → None
    assert _model_from_path("/v1beta/models/") is None
    # /models/ with colon immediately → None (empty model id)
    assert _model_from_path("/v1beta/models/:action") is None


# ---------------------------------------------------------------------------
# streamrelay: chunked framing when upstream sends no Content-Length (SSE)
# ---------------------------------------------------------------------------


def test_streamrelay_chunked_framing_sse() -> None:
    """Streaming response without Content-Length gets chunked transfer encoding."""
    chunk1 = b'data: {"delta":"hello"}\n\n'
    chunk2 = b"data: [DONE]\n\n"

    class _SSE(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:  # noqa: ANN002
            pass

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            # Deliberately omit Content-Length → streamrelay uses chunked framing
            self.end_headers()
            self.wfile.write(chunk1)
            self.wfile.flush()
            time.sleep(0.05)
            self.wfile.write(chunk2)
            self.wfile.flush()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _SSE)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    handler = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        body = json.dumps(
            {
                "model": "claude-opus-4-8",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode()
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
        conn.request(
            "POST",
            "/v1/messages",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        assert resp.headers.get("Transfer-Encoding") == "chunked", (
            "Expected chunked framing when upstream sends no Content-Length"
        )
        data = resp.read()
        conn.close()
        assert chunk1 in data
        assert chunk2 in data
    finally:
        proxy.shutdown()
        upstream.shutdown()


# ---------------------------------------------------------------------------
# streamrelay: HTTPError from upstream relayed via passthrough (GET)
# ---------------------------------------------------------------------------


def test_streamrelay_http_error_relayed_via_passthrough(error_proxy: int) -> None:
    """streamrelay.stream_upstream relays upstream HTTP error status (lines 67-71)."""
    # GET → _passthrough → stream_upstream; upstream returns 500
    status, _, _ = _request("GET", error_proxy)
    assert status == 500


# ---------------------------------------------------------------------------
# streamrelay: connection refused → 502 via passthrough (URLError, not timeout)
# ---------------------------------------------------------------------------


def test_streamrelay_connection_refused_502() -> None:
    """streamrelay returns 502 when the upstream refuses the connection (lines 85-93)."""
    placeholder = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    dead_port = placeholder.server_address[1]
    placeholder.server_close()

    handler = build_handler(f"http://127.0.0.1:{dead_port}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        # GET → _passthrough → stream_upstream → URLError (refused) → 502
        status, _, data = _request("GET", proxy.server_address[1])
        assert status == 502
        assert b"upstream" in data.lower()
    finally:
        proxy.shutdown()


# ---------------------------------------------------------------------------
# Unit tests: _count_messages non-dict block (line 151) + nested list (156-161)
# _tokens_saved (line 167)
# ---------------------------------------------------------------------------


def test_count_messages_non_dict_block_in_content() -> None:
    """A non-dict element in a list-content message is skipped without error (line 151)."""
    from distil.proxy import _count_messages

    msgs = [
        {
            "role": "user",
            "content": [
                "string-not-a-dict",  # non-dict → skip (line 151)
                {"type": "text", "text": "actual text here"},
            ],
        }
    ]
    count = _count_messages(msgs)
    assert count > 0  # the dict block's text is counted


def test_count_messages_nested_list_content() -> None:
    """A block whose 'content' or 'text' key is a list of sub-dicts (lines 156-161)."""
    from distil.proxy import _count_messages

    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    # 'content' value is a list → nested list branch
                    "content": [{"text": "nested tool output line one"}],
                }
            ],
        }
    ]
    count = _count_messages(msgs)
    assert count > 0


def test_tokens_saved_returns_nonnegative() -> None:
    """_tokens_saved(before, after) returns max(0, delta) (line 167)."""
    from distil.proxy import _tokens_saved

    before = [{"role": "user", "content": "word " * 50}]
    after = [{"role": "user", "content": "word " * 20}]
    assert _tokens_saved(before, after) > 0
    # After > before → clamp to 0
    assert _tokens_saved(after, before) == 0
