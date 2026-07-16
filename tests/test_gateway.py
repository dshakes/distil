"""Integration tests for distil.gateway — no external network required.

Architecture
------------
* A fake upstream ``ThreadingHTTPServer`` binds to an ephemeral port on
  127.0.0.1.  For POST requests it reads the body and echoes it back 200 so
  the gateway can forward something real.
* The distil gateway is started (also on an ephemeral port) pointed at the
  fake upstream.
* Tests use ``urllib.request`` as the HTTP client — stdlib only, no network.
* Both servers are shut down cleanly via a pytest fixture.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from distil.gateway import GatewayState, build_gateway_handler, tenant_of
from distil.pricing import get as pricing_get

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# A large multi-line tool_result — well above the 6-line digest threshold
_LONG_TOOL_RESULT = "\n".join(
    [
        "Result from bash tool execution on the remote host:",
        "total disk usage: 48 GB across 12 partitions",
        "filesystem /dev/sda1: 32 GB used of 100 GB available",
        "filesystem /dev/sdb1: 16 GB used of 200 GB available",
        "warning: /tmp is 89% full — consider cleaning up old build artefacts",
        "warning: inode count on /var/log approaching limit (91% used)",
        "no errors detected in kernel ring buffer",
        "last boot: 2026-06-20T03:14:22Z (uptime 18h 42m)",
        "load averages: 0.23 0.31 0.29 (1m/5m/15m)",
        "memory: 14.2 GB used / 31.9 GB total, 0 GB swap",
        "top process: python3 pid=8821 cpu=4.1% mem=2.3%",
        "all health checks passed",
    ]
)  # 12 lines — well above the 6-line digest threshold


def _messages_payload(tool_result_text: str = _LONG_TOOL_RESULT) -> dict[str, Any]:
    return {
        "model": "claude-opus-4-8",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": tool_result_text,
                    }
                ],
            },
            # Later turns keep the tool_result out of the recency-exempt window
            # (the adapter keeps the most recent turns verbatim) so it still digests.
            {"role": "user", "content": "next"},
            {"role": "user", "content": "next"},
        ],
    }


# ---------------------------------------------------------------------------
# Fake upstream server
# ---------------------------------------------------------------------------


class _EchoHandler(BaseHTTPRequestHandler):
    """Fake upstream: echo POST body verbatim; 200 for everything else."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        resp = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


# ---------------------------------------------------------------------------
# Pytest fixture: both servers, torn down after each test module session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gw_servers() -> Any:
    """Yield (gateway_port, upstream_port); shut both down after the module."""
    # 1. Fake upstream on ephemeral port
    upstream_server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    upstream_port = upstream_server.server_address[1]
    upstream_thread = threading.Thread(target=upstream_server.serve_forever, daemon=True)
    upstream_thread.start()

    # 2. Gateway pointed at fake upstream, also on ephemeral port
    upstream_url = f"http://127.0.0.1:{upstream_port}"
    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    # trust_tenant_header=True: these tests exercise multi-tenant accounting via
    # explicit labels (the operator-opt-in mode); identity-derivation is tested
    # separately in test_tenant_of_*.
    handler_cls = build_gateway_handler(upstream_url, state, price, trust_tenant_header=True)
    gw_server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    gw_port = gw_server.server_address[1]
    gw_thread = threading.Thread(target=gw_server.serve_forever, daemon=True)
    gw_thread.start()

    yield gw_port, state

    gw_server.shutdown()
    upstream_server.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post(
    port: int, path: str, payload: dict[str, Any], extra_headers: dict[str, str] | None = None
) -> urllib.request.Request:
    body = json.dumps(payload).encode()
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    if extra_headers:
        headers.update(extra_headers)
    return urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers=headers,
        method="POST",
    )


def _get(port: int, path: str) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    with urllib.request.urlopen(req) as resp:
        return resp.status, dict(resp.headers), resp.read()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tenant_a_two_requests_and_tenant_b_one(gw_servers: Any) -> None:
    """
    Send two /v1/messages POSTs from tenant 'acme' and one from 'globex'.
    Assert:
    - Each response is 200 with x-distil-tenant echoed and positive x-distil-tokens-saved.
    - GET /distil/stats shows acme.requests==2, globex.requests==1, tokens_saved>0 for both.
    - GET /distil/dashboard returns 200 HTML containing both tenant ids and a "$" figure.
    """
    gw_port, state = gw_servers

    # --- Two requests from acme ---
    for i in range(2):
        req = _post(
            gw_port,
            "/v1/messages",
            _messages_payload(),
            extra_headers={"x-distil-tenant": "acme"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200, f"acme request {i} got {resp.status}"
            tenant_hdr = resp.headers.get("x-distil-tenant")
            tokens_saved_hdr = resp.headers.get("x-distil-tokens-saved")
            assert tenant_hdr == "acme", f"x-distil-tenant wrong: {tenant_hdr!r}"
            assert tokens_saved_hdr is not None, "x-distil-tokens-saved missing"
            assert int(tokens_saved_hdr) > 0, (
                f"x-distil-tokens-saved should be positive, got {tokens_saved_hdr!r}"
            )

    # --- One request from globex ---
    req = _post(
        gw_port,
        "/v1/messages",
        _messages_payload(),
        extra_headers={"x-distil-tenant": "globex"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"globex request got {resp.status}"
        tenant_hdr = resp.headers.get("x-distil-tenant")
        tokens_saved_hdr = resp.headers.get("x-distil-tokens-saved")
        assert tenant_hdr == "globex", f"x-distil-tenant wrong: {tenant_hdr!r}"
        assert tokens_saved_hdr is not None, "x-distil-tokens-saved missing"
        assert int(tokens_saved_hdr) > 0, (
            f"x-distil-tokens-saved should be positive, got {tokens_saved_hdr!r}"
        )

    # --- Check /distil/stats ---
    status, _, body = _get(gw_port, "/distil/stats")
    assert status == 200, f"/distil/stats returned {status}"
    snap = json.loads(body)

    tenants_by_id = {t["tenant"]: t for t in snap["tenants"]}

    assert "acme" in tenants_by_id, f"acme not in stats: {list(tenants_by_id)}"
    assert "globex" in tenants_by_id, f"globex not in stats: {list(tenants_by_id)}"

    acme = tenants_by_id["acme"]
    globex = tenants_by_id["globex"]

    assert acme["requests"] == 2, f"acme.requests expected 2, got {acme['requests']}"
    assert globex["requests"] == 1, f"globex.requests expected 1, got {globex['requests']}"

    assert acme["tokens_saved"] > 0, f"acme.tokens_saved should be >0, got {acme['tokens_saved']}"
    assert globex["tokens_saved"] > 0, (
        f"globex.tokens_saved should be >0, got {globex['tokens_saved']}"
    )

    # totals should include both
    totals = snap["totals"]
    assert totals["requests"] == 3, f"totals.requests expected 3, got {totals['requests']}"
    assert totals["tokens_saved"] > 0

    # --- Check /distil/dashboard ---
    status, hdrs, html_body = _get(gw_port, "/distil/dashboard")
    assert status == 200, f"/distil/dashboard returned {status}"
    content_type = hdrs.get("Content-Type", "")
    assert "text/html" in content_type, f"dashboard Content-Type wrong: {content_type!r}"

    html = html_body.decode()
    assert "acme" in html, "dashboard HTML missing 'acme'"
    assert "globex" in html, "dashboard HTML missing 'globex'"
    assert "$" in html, "dashboard HTML missing '$' figure"


def test_stats_empty_before_requests() -> None:
    """A fresh GatewayState snapshot is well-formed with empty tenant list."""
    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    snap = state.snapshot()
    assert snap["tenants"] == []
    assert snap["totals"]["requests"] == 0
    assert snap["totals"]["tokens_saved"] == 0
    assert snap["totals"]["dollars_saved"] == 0.0


def test_tenant_of_explicit_header() -> None:
    """x-distil-tenant is honored ONLY under operator opt-in — by default the
    client-writable header must never enter accounting (impersonation)."""

    class _FakeHeaders:
        def get(self, key: str) -> str | None:
            return {"x-distil-tenant": "myco"}.get(key.lower())

    assert tenant_of(_FakeHeaders()) == "default"  # untrusted by default
    assert tenant_of(_FakeHeaders(), trust_tenant_header=True) == "myco"


def test_tenant_of_api_key_anonymised() -> None:
    """Without x-distil-tenant, x-api-key produces an anon- prefixed id."""

    class _FakeHeaders:
        def get(self, key: str) -> str | None:
            return {"x-api-key": "sk-secret-key-12345"}.get(key.lower())

    result = tenant_of(_FakeHeaders())
    assert result.startswith("anon-"), f"Expected anon- prefix, got {result!r}"
    assert len(result) == len("anon-") + 8, f"Expected 8 hex chars after prefix, got {result!r}"


def test_tenant_of_default_fallback() -> None:
    """No credentials → 'default'."""

    class _FakeHeaders:
        def get(self, key: str) -> str | None:
            return None

    assert tenant_of(_FakeHeaders()) == "default"


def test_dashboard_html_contains_dark_bg() -> None:
    """Dashboard uses the project's dark background colour."""
    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    # Seed some data so the leaderboard row is rendered
    state.record("widget-co", 1000, 700)

    from distil.gateway import _dashboard_html

    html = _dashboard_html(state.snapshot())
    assert "#06070a" in html, "dark bg colour missing from dashboard"
    assert "widget-co" in html, "tenant not rendered in dashboard"
    assert "$" in html, "dollar sign missing from dashboard"
    assert 'content="5"' in html, "auto-refresh meta tag missing"


def test_gateway_passthrough_non_compressible(gw_servers: Any) -> None:
    """Non-compressible paths are forwarded transparently (no distil headers added)."""
    gw_port, _ = gw_servers

    payload = {"key": "value"}
    req = _post(gw_port, "/v1/models", payload)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        # No distil compression header
        assert resp.headers.get("x-distil-tokens-saved") is None


def test_management_endpoints_gated_off_loopback() -> None:
    """/distil/* must not leak per-tenant usage to anyone on the network:
    non-loopback binds refuse without a token; a token gates with Bearer auth."""
    import http.client

    upstream_server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=upstream_server.serve_forever, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream_server.server_address[1]}"
    price = pricing_get("claude-opus-4-8")

    def _get(port: int, headers: dict | None = None) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/distil/stats", headers=headers or {})
        status = conn.getresponse().status
        conn.close()
        return status

    # Non-loopback bind, no token → refused
    h1 = build_gateway_handler(upstream_url, GatewayState(price), price, loopback=False)
    s1 = ThreadingHTTPServer(("127.0.0.1", 0), h1)
    threading.Thread(target=s1.serve_forever, daemon=True).start()
    assert _get(s1.server_address[1]) == 403

    # Token configured → 401 without, 200 with the right Bearer
    h2 = build_gateway_handler(
        upstream_url, GatewayState(price), price, loopback=False, admin_token="sekrit"
    )
    s2 = ThreadingHTTPServer(("127.0.0.1", 0), h2)
    threading.Thread(target=s2.serve_forever, daemon=True).start()
    assert _get(s2.server_address[1]) == 401
    assert _get(s2.server_address[1], {"Authorization": "Bearer sekrit"}) == 200

    for srv in (s1, s2, upstream_server):
        srv.shutdown()


# ---------------------------------------------------------------------------
# TenantStats.pct_saved() with zero baseline (line 100)
# ---------------------------------------------------------------------------


def test_tenant_stats_pct_saved_zero_baseline() -> None:
    """pct_saved() returns 0.0 when tokens_baseline is 0 (avoids ZeroDivisionError)."""
    from distil.gateway import TenantStats

    s = TenantStats()
    assert s.pct_saved() == 0.0


# ---------------------------------------------------------------------------
# GatewayState.save() OSError (lines 231-232)
# ---------------------------------------------------------------------------


def test_gateway_state_save_oserror(tmp_path: Any) -> None:
    """A failed save is silently ignored — never raises (crash-safety guarantee)."""

    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    state.record("tenant-a", 1000, 600)

    # Make path.parent an existing FILE so mkdir() raises (FileExistsError → OSError)
    blocker = tmp_path / "notadir"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    bad_path = blocker / "state.json"
    state.save(path=bad_path)  # must not raise


# ---------------------------------------------------------------------------
# _count_tokens: nested list content (lines 346-356)
# ---------------------------------------------------------------------------


def test_count_tokens_nested_list_content() -> None:
    """_count_tokens handles non-dict blocks (line 346) and nested list values (lines 352-356)."""
    from distil.gateway import _count_tokens

    msgs = [
        {
            "role": "user",
            "content": [
                "not-a-dict",  # non-dict block → continue (line 346)
                {
                    # block with 'content' key that is a list of sub-dicts
                    "type": "tool_result",
                    "content": [{"type": "text", "text": "line one\nline two"}],
                },
            ],
        }
    ]
    count = _count_tokens(msgs)
    assert count > 0


# ---------------------------------------------------------------------------
# Gateway: Gemini daily-token quota exceeded (lines 954-961)
# ---------------------------------------------------------------------------


def test_gateway_gemini_daily_quota_exceeded() -> None:
    """POST to a Gemini path that exceeds the per-day token cap → 429 quota error."""
    upstream_server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=upstream_server.serve_forever, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream_server.server_address[1]}"
    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    # daily_tokens=1 so any request with > 1 token triggers the quota on first hit
    handler = build_gateway_handler(upstream_url, state, price, default_daily_tokens=1)
    gw = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=gw.serve_forever, daemon=True).start()
    port = gw.server_address[1]

    gemini_path = "/v1beta/models/gemini-1.5-pro:generateContent"
    body = json.dumps(
        {"contents": [{"role": "user", "parts": [{"text": "hello world this is a test prompt"}]}]}
    ).encode()

    statuses = []
    for _ in range(2):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{gemini_path}",
            data=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                statuses.append(resp.status)
        except urllib.error.HTTPError as exc:
            statuses.append(exc.code)

    gw.shutdown()
    upstream_server.shutdown()
    assert 429 in statuses, f"expected a 429 from daily quota gate, got {statuses}"


# ---------------------------------------------------------------------------
# GatewayState.load: cap oversized state file at _MAX_TENANTS (line 254)
# ---------------------------------------------------------------------------


def test_gateway_state_load_caps_at_max_tenants(tmp_path: Any) -> None:
    """load() enforces _MAX_TENANTS even when the persisted file has more entries."""
    from distil.gateway import _MAX_TENANTS

    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)

    # Write a state file with _MAX_TENANTS + 5 tenants
    excess = 5
    data = {
        "tenants": {
            f"t{i}": {"requests": 1, "tokens_baseline": 100, "tokens_compressed": 80}
            for i in range(_MAX_TENANTS + excess)
        }
    }
    state_file = tmp_path / "gateway_state.json"
    state_file.write_text(json.dumps(data), encoding="utf-8")
    state.load(path=state_file)

    snap = state.snapshot()
    assert len(snap["tenants"]) == _MAX_TENANTS
