"""Coverage tests for distil.gateway — fills gaps in test_gateway.py.

Covers:
  • do_GET for non-stats/non-dashboard paths → _passthrough (line ~483)
  • do_PUT / do_DELETE / do_PATCH / do_HEAD / do_OPTIONS → _passthrough
  • _handle_compressible: invalid path → 400 (lines 555-556)
  • _handle_compressible: oversized Content-Length → 413 (lines 559-560, 671)
  • _handle_compressible: non-JSON body → forwarded (lines 566-570)
  • _handle_compressible: Gemini ``contents`` path (lines 595-605)
  • _handle_compressible: stream=True → streamrelay (lines 631-632)
  • _passthrough: URLError → 502 (lines 650-660)
  • _post_upstream: HTTPError relayed (lines ~707+)
  • anon tenant (x-api-key) not echoed in response headers
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from distil.gateway import GatewayState, build_gateway_handler
from distil.pricing import get as pricing_get

# ---------------------------------------------------------------------------
# Fake upstream handlers
# ---------------------------------------------------------------------------


class _EchoHandler(BaseHTTPRequestHandler):
    """Echo POST body back; echo path for other verbs."""

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
    """Always returns 500."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def _err(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = b'{"error":"server error"}'
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _err  # type: ignore[assignment]
    do_POST = _err  # type: ignore[assignment]


def _start(handler_cls: type) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _make_gateway(upstream_port: int, **kwargs: Any) -> tuple[ThreadingHTTPServer, GatewayState]:
    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    handler = build_gateway_handler(
        f"http://127.0.0.1:{upstream_port}",
        state,
        price,
        trust_tenant_header=True,
        **kwargs,
    )
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gw() -> Any:
    """(gateway_port, state) pair backed by an echo upstream; torn down after."""
    upstream = _start(_EchoHandler)
    srv, state = _make_gateway(upstream.server_address[1])
    yield srv.server_address[1], state
    srv.shutdown()
    upstream.shutdown()


@pytest.fixture()
def error_gw() -> Any:
    """Gateway backed by an error (500) upstream."""
    upstream = _start(_ErrorHandler)
    srv, state = _make_gateway(upstream.server_address[1])
    yield srv.server_address[1], state
    srv.shutdown()
    upstream.shutdown()


# ---------------------------------------------------------------------------
# Tiny request helper
# ---------------------------------------------------------------------------


def _req(
    method: str,
    port: int,
    path: str = "/v1/models",
    body: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, http.client.HTTPResponse, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    hdrs: dict[str, str] = {}
    if body is not None:
        hdrs["Content-Type"] = "application/json"
        hdrs["Content-Length"] = str(len(body))
    if extra_headers:
        hdrs.update(extra_headers)
    conn.request(method, path, body=body, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, resp, data


# ---------------------------------------------------------------------------
# Tests: other HTTP verbs on gateway → _passthrough
# ---------------------------------------------------------------------------


def test_gateway_get_non_admin_passthrough(gw: Any) -> None:
    """GET to a normal path (not /distil/*) is forwarded transparently."""
    port, _ = gw
    status, _, _ = _req("GET", port, "/v1/models")
    assert status == 200


def test_gateway_passthrough_invalid_path_400(gw: Any) -> None:
    """GET with @ in path routes to _passthrough then hits safe_forward_path → 400 (lines 631-632)."""
    port, _ = gw
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/@injected/v1/models")
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 400


def test_gateway_dashboard_empty_state() -> None:
    """Dashboard with no recorded requests renders the empty-row placeholder (line 241)."""
    from distil.gateway import GatewayState, _dashboard_html
    from distil.pricing import get as pricing_get

    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    html = _dashboard_html(state.snapshot())
    assert "No requests recorded yet" in html


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS"])
def test_gateway_other_verbs_passthrough(gw: Any, method: str) -> None:
    """PUT/DELETE/PATCH/OPTIONS are all dispatched to _passthrough."""
    port, _ = gw
    status, _, _ = _req(method, port)
    assert status == 200


def test_gateway_head_passthrough(gw: Any) -> None:
    """HEAD is dispatched to _passthrough (no body expected)."""
    port, _ = gw
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("HEAD", "/v1/models")
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 200


# ---------------------------------------------------------------------------
# Tests: _handle_compressible invalid path → 400
# ---------------------------------------------------------------------------


def test_gateway_compressible_invalid_path_400(gw: Any) -> None:
    """@ in Gemini model name routes to _handle_compressible but fails safe_forward_path → 400.

    ``..`` paths don't reach _handle_compressible because the gateway dispatches
    on the exact set before safe_forward_path runs.  A Gemini URL with @ in the
    model name is accepted by is_gemini_path but rejected by safe_forward_path
    (@ = host-injection vector), covering lines 555-556.
    """
    port, _ = gw
    body = b'{"contents": []}'
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
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


# ---------------------------------------------------------------------------
# Tests: _read_body / oversized CL → 413 (lines 559-560, 635-636, 671)
# ---------------------------------------------------------------------------


def test_gateway_compressible_oversized_cl_413(gw: Any) -> None:
    """Content-Length beyond 8 MiB limit on compressible path → 413 (lines 559-560, 671)."""
    port, _ = gw
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("POST", "/v1/messages")
    conn.putheader("Content-Length", "99999999999")
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 413


def test_gateway_passthrough_oversized_cl_413(gw: Any) -> None:
    """Oversized Content-Length on a passthrough GET path → 413 (lines 635-636)."""
    port, _ = gw
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("GET", "/v1/models")
    conn.putheader("Content-Length", "99999999999")
    conn.endheaders()
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 413


# ---------------------------------------------------------------------------
# Tests: non-JSON body on compressible path → forwarded (lines 566-570)
# ---------------------------------------------------------------------------


def test_gateway_compressible_non_json_forwarded(gw: Any) -> None:
    """Non-JSON body on /v1/messages is forwarded to the upstream as-is."""
    port, _ = gw
    body = b"raw-binary-body"
    status, _, data = _req("POST", port, "/v1/messages", body=body)
    assert status == 200
    assert data == body  # echo upstream echoes it back unchanged


# ---------------------------------------------------------------------------
# Tests: Gemini ``contents`` path (lines 595-605)
# ---------------------------------------------------------------------------


def test_gateway_gemini_contents_compressed(gw: Any) -> None:
    """Gateway compresses Gemini generateContent payloads (``contents`` field)."""
    port, _ = gw
    contents_text = "\n".join(f"Gemini response line {i}" for i in range(30))
    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": contents_text}]}],
            "generationConfig": {"maxOutputTokens": 100},
        }
    ).encode()
    status, resp, _ = _req("POST", port, "/v1beta/models/gemini-pro:generateContent", body=body)
    assert status == 200
    # gateway.py sets x-distil-tokens-saved (not x-distil-compressed) for the Gemini path
    assert resp.headers.get("x-distil-tokens-saved") is not None


# ---------------------------------------------------------------------------
# Tests: stream=True → streamrelay path (lines 631-632)
# ---------------------------------------------------------------------------


def test_gateway_streaming_request_uses_streamrelay() -> None:
    """stream=True on a compressible path goes through streamrelay (chunked)."""
    chunk1 = b'data: {"delta":"hi"}\n\n'
    chunk2 = b"data: [DONE]\n\n"

    class _SSE(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:  # noqa: ANN002
            pass

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            # No Content-Length → chunked framing
            self.end_headers()
            self.wfile.write(chunk1)
            self.wfile.flush()
            time.sleep(0.03)
            self.wfile.write(chunk2)
            self.wfile.flush()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _SSE)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    srv, _ = _make_gateway(upstream.server_address[1])
    try:
        body = json.dumps(
            {
                "model": "claude-opus-4-8",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode()
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        conn.request(
            "POST",
            "/v1/messages",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        assert resp.headers.get("Transfer-Encoding") == "chunked"
        data = resp.read()
        conn.close()
        assert chunk1 in data and chunk2 in data
    finally:
        srv.shutdown()
        upstream.shutdown()


# ---------------------------------------------------------------------------
# Tests: passthrough URLError → 502 (lines 650-660)
# ---------------------------------------------------------------------------


def test_gateway_passthrough_connection_refused_502() -> None:
    """Gateway _passthrough: refused upstream connection → 502."""
    # Reserve the port by binding without listening (see test_proxy_cov.py's
    # test_proxy_connection_refused_502 for why): a bound-but-not-listening socket
    # gets an immediate RST on any connect attempt while we hold it, so nothing else
    # on a shared CI host can steal the port out from under us mid-test.
    placeholder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    placeholder.bind(("127.0.0.1", 0))
    dead_port = placeholder.getsockname()[1]
    try:
        srv, _ = _make_gateway(dead_port)
        try:
            # GET to non-admin path → _passthrough → URLError → 502
            status, _, data = _req("GET", srv.server_address[1], "/v1/models")
            assert status == 502
        finally:
            srv.shutdown()
    finally:
        placeholder.close()


# ---------------------------------------------------------------------------
# Tests: passthrough upstream 500 relayed (HTTPError in _passthrough)
# ---------------------------------------------------------------------------


def test_gateway_passthrough_upstream_500_relayed(error_gw: Any) -> None:
    """Gateway _passthrough: upstream 500 is relayed via the HTTPError handler."""
    port, _ = error_gw
    status, _, _ = _req("GET", port, "/v1/models")
    assert status == 500


# ---------------------------------------------------------------------------
# Tests: anon tenant (via x-api-key) is NOT echoed in response headers
# ---------------------------------------------------------------------------


def test_gateway_anon_tenant_not_echoed_in_response(gw: Any) -> None:
    """Credential-derived anon- tenant id must not appear in response headers."""
    port, _ = gw
    long_text = "\n".join(f"tool result line {i}" for i in range(20))
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
    # x-api-key → anon-<hash> tenant, which must not be echoed back
    status, resp, _ = _req(
        "POST",
        port,
        "/v1/messages",
        body=body,
        extra_headers={"x-api-key": "sk-test-key-1234"},
    )
    assert status == 200
    assert resp.headers.get("x-distil-tenant") is None, (
        "anon- tenant id must never be echoed in response headers"
    )


# ---------------------------------------------------------------------------
# Tests: explicit tenant label IS echoed in response (trust_tenant_header=True)
# ---------------------------------------------------------------------------


def test_gateway_explicit_tenant_echoed_in_response(gw: Any) -> None:
    """With trust_tenant_header=True, an explicit label is echoed back."""
    port, _ = gw
    long_text = "\n".join(f"tool result line {i}" for i in range(20))
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
    status, resp, _ = _req(
        "POST",
        port,
        "/v1/messages",
        body=body,
        extra_headers={"x-distil-tenant": "myteam"},
    )
    assert status == 200
    assert resp.headers.get("x-distil-tenant") == "myteam"


# ---------------------------------------------------------------------------
# Health endpoint + crash-safety checkpoint
# ---------------------------------------------------------------------------


def test_gateway_health_unauthenticated(gw: Any) -> None:
    gw_port, _state = gw
    status, _resp, data = _req("GET", gw_port, "/distil/health")
    assert status == 200
    assert json.loads(data) == {"status": "ok"}


def test_state_record_checkpoints_periodically(tmp_path: Any, monkeypatch: Any) -> None:
    """record() persists to disk once the checkpoint interval has elapsed —
    a kill -9 must not zero more than _CHECKPOINT_SECS of tenant accounting."""
    import distil.gateway as gwmod

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    state._last_save -= gwmod._CHECKPOINT_SECS + 1  # pretend the interval elapsed
    state.record("tenant-a", 100, 40)
    fresh = GatewayState(price)
    fresh.load()  # reads what record() checkpointed, no explicit save()
    assert fresh.snapshot()["tenants"][0]["requests"] == 1
