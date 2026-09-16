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

import base64
import hashlib
import hmac
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


class _HeaderEchoHandler(BaseHTTPRequestHandler):
    """Echo the request headers back as JSON — what the UPSTREAM actually saw."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        if n:
            self.rfile.read(n)
        seen = json.dumps({k.lower(): v for k, v in self.headers.items()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(seen)))
        self.end_headers()
        self.wfile.write(seen)


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
    # Release the placeholder as late as possible (see test_proxy_cov.py's
    # test_proxy_connection_refused_502 for why, and why bind-without-listen isn't
    # used instead — it isn't portable, macOS runners don't RST it the way Linux does).
    placeholder = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    dead_port = placeholder.server_address[1]
    try:
        srv, _ = _make_gateway(dead_port)
        try:
            placeholder.server_close()
            # GET to non-admin path → _passthrough → URLError → 502
            status, _, data = _req("GET", srv.server_address[1], "/v1/models")
            assert status == 502
        finally:
            srv.shutdown()
    finally:
        placeholder.server_close()


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


# ---------------------------------------------------------------------------
# Request framing: a body the gateway cannot read must not become the next request
# ---------------------------------------------------------------------------


_CHUNKED_REJECTION = b'{"error": "chunked request bodies are not supported; send Content-Length"}'


def _raw_exchange(port: int, payload: bytes, *, timeout: float = 3.0) -> bytes:
    """Send *payload* as one write and read until the server closes (or stalls).

    Deliberately raw: http.client would frame the request for us, and the bug
    under test is entirely about framing.
    """
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    try:
        sock.sendall(payload)
        out = b""
        while True:
            try:
                chunk = sock.recv(65536)
            except (TimeoutError, OSError):
                break  # keep-alive with nothing more to send — that is the bug
            if not chunk:
                break
            out += chunk
        return out
    finally:
        sock.close()


_OVERSIZED_REJECTION = b'{"error": "request body too large or malformed Content-Length"}'


def test_chunked_body_is_refused_and_cannot_smuggle_a_second_request(gw: Any) -> None:
    """A Transfer-Encoding body reads as empty, so whatever follows it on the
    socket used to be parsed as a separate, separately-authorized request."""
    gw_port, _state = gw
    smuggled = (
        b"POST /v1/messages HTTP/1.1\r\n"
        b"Host: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nhello\r\n0\r\n\r\n"
        b"GET /distil/health HTTP/1.1\r\nHost: x\r\n\r\n"
    )
    raw = _raw_exchange(gw_port, smuggled)
    assert raw.startswith(b"HTTP/1.1 411 "), raw[:80]
    # The rejection body is the last byte on the wire: the smuggled GET was never
    # served, and the connection closed rather than keeping the body queued.
    # (A malformed leftover comes back as an HTTP/0.9 body with no status line,
    # so counting status lines would miss it — compare the whole body.)
    _head, _, body = raw.partition(b"\r\n\r\n")
    assert body == _CHUNKED_REJECTION, body


def test_content_length_and_transfer_encoding_together_are_refused(gw: Any) -> None:
    """Two framings on one request is the TE.CL desync pair stated outright."""
    gw_port, _state = gw
    body = b'{"model":"claude-opus-4-8","messages":[]}'
    raw = _raw_exchange(
        gw_port,
        b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n" + body,
    )
    assert raw.startswith(b"HTTP/1.1 400 "), raw[:80]
    _head, _, body = raw.partition(b"\r\n\r\n")
    assert b"conflicting" in body and body.endswith(b"}"), body


def test_oversized_content_length_cannot_smuggle_a_second_request(gw: Any) -> None:
    """The 413 answers from the headers alone, so the body the client already
    sent is still queued — the same desync the chunked case has, reached through
    a different rejection. Declare far more than the 8 MiB guard allows, send a
    short body, and trail a GET that must never be served."""
    gw_port, _state = gw
    raw = _raw_exchange(
        gw_port,
        b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 99999999999\r\n\r\n"
        b"hello"
        b"GET /distil/health HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    assert raw.startswith(b"HTTP/1.1 413 "), raw[:80]
    # One message on the wire. /distil/health is unauthenticated and always 200s,
    # so if the connection had stayed open its body would be sitting right here.
    _head, _, body = raw.partition(b"\r\n\r\n")
    assert body == _OVERSIZED_REJECTION, body
    assert b"status" not in raw, raw


def test_duplicate_content_length_cannot_smuggle_a_second_request(gw: Any) -> None:
    """CL.CL. headers.get() hands back the first value and hides the rest, so a
    request declaring both 5 and 0 is read as 5 here and as 0 by any front-end
    that takes the last — those five bytes then head the next request."""
    gw_port, _state = gw
    raw = _raw_exchange(
        gw_port,
        b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 5\r\n"
        b"Content-Length: 0\r\n\r\n"
        b"hello"
        b"GET /distil/health HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    assert raw.startswith(b"HTTP/1.1 400 "), raw[:80]
    _head, _, payload = raw.partition(b"\r\n\r\n")
    assert payload == b'{"error": "conflicting Content-Length headers"}', payload
    assert b'"status"' not in raw, raw


def test_identical_duplicate_content_length_is_still_served(gw: Any) -> None:
    """RFC 9112 §6.3 allows one value sent twice. Refusing it would be a new
    outage dressed up as a fix, so the guard has to tell the two cases apart."""
    gw_port, _state = gw
    body = b'{"model":"claude-opus-4-8","messages":[]}'
    n = str(len(body)).encode()
    raw = _raw_exchange(
        gw_port,
        b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + n + b"\r\n"
        b"Content-Length: " + n + b"\r\n\r\n" + body,
    )
    assert not raw.startswith(b"HTTP/1.1 400 "), raw[:200]


def test_rate_limited_429_cannot_smuggle_a_second_request(tmp_path: Any, monkeypatch: Any) -> None:
    """The third door into the same desync: the RPM 429 in _check_inbound_auth.

    It fires before _read_body like every other rejection, but it used to write
    itself through _relay instead of _reject — so it kept the connection alive
    and left the body queued, on the one path that by definition is being hit
    repeatedly. Exhaust a per-key limit of 1, then send body + trailing GET.
    """
    from distil.gateway_keys import GatewayKeyStore

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    store = GatewayKeyStore()
    raw_key, _rec = store.issue(tenant="acme", rpm=1)

    upstream = _start(_EchoHandler)
    srv, _state = _make_gateway(upstream.server_address[1], key_store=store, require_keys=True)
    port = srv.server_address[1]
    try:
        body = b'{"model":"claude-opus-4-8","messages":[]}'
        head = (
            b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: application/json\r\n"
            b"x-distil-key: " + raw_key.encode() + b"\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
        )
        # Burn the single allowed request for this tenant.
        first = _raw_exchange(port, head + body)
        assert not first.startswith(b"HTTP/1.1 429 "), first[:80]

        # Now the limit is spent: this one is rejected on headers alone.
        raw = _raw_exchange(
            port,
            head + body + b"GET /distil/health HTTP/1.1\r\nHost: x\r\n\r\n",
        )
    finally:
        srv.shutdown()
        upstream.shutdown()

    assert raw.startswith(b"HTTP/1.1 429 "), raw[:80]
    assert b"Retry-After: 60" in raw, raw[:400]  # the header survived the move to _reject
    _head, _, payload = raw.partition(b"\r\n\r\n")
    assert payload == b'{"error": "rate limit exceeded"}', payload
    assert b'"status"' not in raw, raw  # the smuggled health GET was never served


def test_passthrough_verbs_refuse_a_chunked_body_too(gw: Any) -> None:
    """The guard lives in _read_body, so every verb gets it, not just POST."""
    gw_port, _state = gw
    raw = _raw_exchange(
        gw_port,
        b"PUT /v1/models HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
    )
    assert raw.startswith(b"HTTP/1.1 411 "), raw[:80]


def test_gateway_handler_sets_the_proxy_client_timeout() -> None:
    """Slowloris: with timeout=None a peer that connects and dribbles pins a
    server thread forever. The gateway is the exposed component; it gets the
    same socket timeout the proxy has always had."""
    from distil.proxy import _CLIENT_TIMEOUT

    price = pricing_get("claude-opus-4-8")
    handler = build_gateway_handler("http://127.0.0.1:1", GatewayState(price), price)
    assert handler.timeout == _CLIENT_TIMEOUT


# ---------------------------------------------------------------------------
# OIDC: configuring an issuer is configuring authentication
# ---------------------------------------------------------------------------


_OIDC_SECRET = "correct horse battery staple"
_OIDC_ISSUER = "https://idp.example"


def _jwt(claims: dict[str, Any]) -> str:
    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps(claims).encode())
    sig = hmac.new(_OIDC_SECRET.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}.{b64(sig)}"


@pytest.fixture()
def oidc_gw(tmp_path: Any, monkeypatch: Any) -> Any:
    """Gateway with OIDC configured, no keys issued, bound non-loopback."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_OIDC_ISSUER", _OIDC_ISSUER)
    monkeypatch.setenv("DISTIL_OIDC_HS256_SECRET", _OIDC_SECRET)
    upstream = _start(_EchoHandler)
    srv, _state = _make_gateway(upstream.server_address[1], loopback=False)
    yield srv.server_address[1]
    srv.shutdown()
    upstream.shutdown()


@pytest.fixture()
def oidc_gw_seen(tmp_path: Any, monkeypatch: Any) -> Any:
    """OIDC gateway whose upstream reports the headers it received."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_OIDC_ISSUER", _OIDC_ISSUER)
    monkeypatch.setenv("DISTIL_OIDC_HS256_SECRET", _OIDC_SECRET)
    upstream = _start(_HeaderEchoHandler)
    srv, _state = _make_gateway(upstream.server_address[1], loopback=False)
    yield srv.server_address[1]
    srv.shutdown()
    upstream.shutdown()


def _operator_token() -> str:
    return _jwt(
        {
            "sub": "u1",
            "iss": _OIDC_ISSUER,
            "tenant": "acme",
            "role": "operator",
            "exp": time.time() + 3600,
        }
    )


def test_oidc_configured_with_no_keys_still_requires_a_credential(oidc_gw: Any) -> None:
    """The auth gate used to key on issued keys alone, so an operator who wired
    up an IdP and issued no dsk- key ran a fully open gateway."""
    status, _resp, data = _req(
        "POST",
        oidc_gw,
        "/v1/messages",
        body=json.dumps({"model": "claude-opus-4-8", "messages": []}).encode(),
    )
    assert status == 401, data


def test_a_valid_oidc_token_is_a_complete_credential(oidc_gw: Any) -> None:
    """Closing the gate must not close the door: with no key store at all, a
    verified token is still enough to proxy. Sent on x-distil-token, the carrier
    that cannot be confused with the provider credential."""
    status, resp, data = _req(
        "POST",
        oidc_gw,
        "/v1/messages",
        body=json.dumps({"model": "claude-opus-4-8", "messages": []}).encode(),
        extra_headers={"x-distil-token": _operator_token()},
    )
    assert status == 200, data
    assert resp.headers.get("x-distil-tenant") == "acme"


def test_a_bearer_jwt_alongside_x_api_key_is_still_accepted(oidc_gw: Any) -> None:
    """The Anthropic shape, which has always sent the JWT on Authorization: the
    provider credential rides on x-api-key, so consuming the bearer costs the
    upstream nothing and this keeps working unchanged."""
    status, resp, data = _req(
        "POST",
        oidc_gw,
        "/v1/messages",
        body=json.dumps({"model": "claude-opus-4-8", "messages": []}).encode(),
        extra_headers={
            "Authorization": f"Bearer {_operator_token()}",
            "x-api-key": "sk-ant-provider",
        },
    )
    assert status == 200, data
    assert resp.headers.get("x-distil-tenant") == "acme"


def test_oidc_only_deployment_can_reach_a_bearer_auth_upstream(oidc_gw_seen: Any) -> None:
    """The regression this header exists for. For OpenAI, Azure and Gemini-bearer,
    `Authorization` IS the provider credential, and this gateway injects none of
    its own — it forwards the client's. With the JWT on x-distil-token the bearer
    rides through untouched, byte-identical, and the IdP token never leaks."""
    provider = "Bearer sk-proj-abc123.DEF-456_xyz"
    status, _resp, data = _req(
        "POST",
        oidc_gw_seen,
        "/v1/messages",
        body=json.dumps({"model": "claude-opus-4-8", "messages": []}).encode(),
        extra_headers={"x-distil-token": _operator_token(), "Authorization": provider},
    )
    assert status == 200, data
    seen = json.loads(data)
    assert seen.get("authorization") == provider, seen
    assert "x-distil-token" not in seen, seen
    assert _operator_token()[:20] not in json.dumps(seen), seen


def test_the_idp_token_never_reaches_the_upstream_on_the_bearer_carrier(
    oidc_gw_seen: Any,
) -> None:
    """The Anthropic shape: the JWT is consumed here and x-api-key goes on."""
    status, _resp, data = _req(
        "POST",
        oidc_gw_seen,
        "/v1/messages",
        body=json.dumps({"model": "claude-opus-4-8", "messages": []}).encode(),
        extra_headers={
            "Authorization": f"Bearer {_operator_token()}",
            "x-api-key": "sk-ant-provider",
        },
    )
    assert status == 200, data
    seen = json.loads(data)
    assert seen.get("x-api-key") == "sk-ant-provider", seen
    assert "authorization" not in seen, seen


def test_a_bearer_jwt_with_no_provider_credential_is_refused_with_guidance(
    oidc_gw: Any,
) -> None:
    """Consuming this bearer would strip the only credential the upstream was
    going to see, and the gateway has none of its own to inject. Refuse with the
    fix in the message rather than forward a request certain to 401 upstream."""
    status, _resp, data = _req(
        "POST",
        oidc_gw,
        "/v1/messages",
        body=json.dumps({"model": "claude-opus-4-8", "messages": []}).encode(),
        extra_headers={"Authorization": f"Bearer {_operator_token()}"},
    )
    assert status == 401, data
    assert b"x-distil-token" in data, data


def test_an_unverified_authorization_header_is_never_stripped(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """A bearer that is not our JWT is the provider's credential. Authenticate
    with a dsk- key so the OIDC carrier is never consulted, and the Authorization
    header must reach the upstream exactly as sent."""
    from distil.gateway_keys import GatewayKeyStore

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_OIDC_ISSUER", _OIDC_ISSUER)
    monkeypatch.setenv("DISTIL_OIDC_HS256_SECRET", _OIDC_SECRET)
    store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    raw_key, _rec = store.issue(tenant="acme")
    upstream = _start(_HeaderEchoHandler)
    srv, _state = _make_gateway(upstream.server_address[1], key_store=store, loopback=False)
    provider = "Bearer sk-proj-not-a-jwt"
    try:
        status, _resp, data = _req(
            "POST",
            srv.server_address[1],
            "/v1/messages",
            body=json.dumps({"model": "claude-opus-4-8", "messages": []}).encode(),
            extra_headers={"x-distil-key": raw_key, "Authorization": provider},
        )
    finally:
        srv.shutdown()
        upstream.shutdown()
    assert status == 200, data
    seen = json.loads(data)
    assert seen.get("authorization") == provider, seen
    assert "x-distil-key" not in seen, seen


def test_a_crlf_tenant_claim_never_reaches_a_response_header(oidc_gw: Any) -> None:
    """send_header performs no CRLF validation, and the OIDC path skipped the
    tenant validator the client-header path has always had."""
    tok = _jwt(
        {
            "sub": "u1",
            "iss": _OIDC_ISSUER,
            "tenant": "acme\r\nX-Injected: yes",
            "role": "operator",
            "exp": time.time() + 3600,
        }
    )
    status, resp, data = _req(
        "POST",
        oidc_gw,
        "/v1/messages",
        body=json.dumps({"model": "claude-opus-4-8", "messages": []}).encode(),
        extra_headers={"x-distil-token": tok},
    )
    assert status == 200, data
    assert resp.headers.get("X-Injected") is None
    assert resp.headers.get("x-distil-tenant", "").startswith("oidc-")
