"""Tests for the gateway key auth, rate limiting, and key management CLI round-trip.

Architecture mirrors test_gateway.py: a fake echo upstream + a gateway server,
both on ephemeral ports, using stdlib urllib — zero external deps.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from distil.gateway import GatewayState, _RateLimiter, build_gateway_handler
from distil.gateway_keys import GatewayKeyStore
from distil.pricing import get as pricing_get


# ---------------------------------------------------------------------------
# Fake upstream (echoes POST bodies; 200 for everything)
# ---------------------------------------------------------------------------


class _EchoHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Echo the received Authorization + x-distil-key headers back so tests
        # can verify the gateway stripped or passed them through.
        for hdr in ("Authorization", "x-distil-key", "x-api-key"):
            val = self.headers.get(hdr)
            if val is not None:
                self.send_header(f"x-echo-{hdr.lower()}", val)
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
# Helpers: start a server, build a gateway, make requests
# ---------------------------------------------------------------------------


def _start(handler_cls: type) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _make_gateway(
    upstream_port: int,
    key_store: GatewayKeyStore | None = None,
    require_keys: bool = False,
    default_rpm: int = 0,
    default_daily_tokens: int = 0,
) -> tuple[ThreadingHTTPServer, GatewayState]:
    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    handler = build_gateway_handler(
        f"http://127.0.0.1:{upstream_port}",
        state,
        price,
        key_store=key_store,
        require_keys=require_keys,
        default_rpm=default_rpm,
        default_daily_tokens=default_daily_tokens,
    )
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def _req(
    method: str,
    port: int,
    path: str = "/v1/messages",
    body: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, http.client.HTTPMessage, bytes]:
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
    return resp.status, resp.msg, data  # type: ignore[return-value]


def _post_messages(
    port: int, extra_headers: dict[str, str] | None = None
) -> tuple[int, Any, bytes]:
    payload = json.dumps(
        {
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    hdrs: dict[str, str] = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
    }
    if extra_headers:
        hdrs.update(extra_headers)
    conn.request("POST", "/v1/messages", body=payload, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    headers = resp
    conn.close()
    return resp.status, headers, data


# ---------------------------------------------------------------------------
# GatewayKeyStore unit tests (no HTTP server needed)
# ---------------------------------------------------------------------------


def test_key_store_issue_and_lookup(tmp_path: Path) -> None:
    """Issue a key, look it up with the raw token, verify it is NOT stored plaintext."""
    store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    raw, rec = store.issue("acme", rpm=60, daily_tokens=10_000)

    assert raw.startswith("dsk-"), f"key must start with dsk-, got {raw!r}"
    assert rec.tenant == "acme"
    assert rec.rpm == 60
    assert rec.daily_tokens == 10_000
    assert rec.is_active

    # Lookup with the correct raw key
    found = store.lookup(raw)
    assert found is not None
    assert found.tenant == "acme"

    # Confirm the raw key is NOT in the key file
    key_file = (tmp_path / "gateway_keys.json").read_text(encoding="utf-8")
    assert raw not in key_file, "raw key must never be stored on disk"
    assert "dsk-" not in key_file, "dsk- prefix must not appear in the stored data"


def test_key_store_wrong_key_returns_none(tmp_path: Path) -> None:
    store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    store.issue("acme")
    assert store.lookup("dsk-wrongkey") is None
    assert store.lookup("not-a-dsk-key") is None


def test_key_store_revoke_blocks_lookup(tmp_path: Path) -> None:
    store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    raw, rec = store.issue("acme")

    assert store.lookup(raw) is not None  # active

    updated = store.revoke(rec.id)
    assert updated is not None
    assert not updated.is_active
    assert updated.revoked is not None

    assert store.lookup(raw) is None  # revoked → None


def test_key_store_revoke_unknown_id_returns_none(tmp_path: Path) -> None:
    store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    assert store.revoke("gk_does_not_exist") is None


def test_key_store_list_all_records(tmp_path: Path) -> None:
    store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    _, r1 = store.issue("acme")
    _, r2 = store.issue("globex")
    store.revoke(r2.id)

    keys = store.list_keys()
    assert len(keys) == 2
    ids = {k.id for k in keys}
    assert r1.id in ids and r2.id in ids


def test_key_store_has_active_keys(tmp_path: Path) -> None:
    store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    assert not store.has_active_keys()

    _, rec = store.issue("acme")
    assert store.has_active_keys()

    store.revoke(rec.id)
    assert not store.has_active_keys()


def test_key_store_cross_process_reload(tmp_path: Path) -> None:
    """A second GatewayKeyStore pointed at the same file sees keys issued by the first."""
    path = tmp_path / "gateway_keys.json"
    store1 = GatewayKeyStore(path)
    raw, _ = store1.issue("acme")

    store2 = GatewayKeyStore(path)
    # Force cache miss so it reads from disk
    store2._last_load = 0.0  # type: ignore[attr-defined]
    assert store2.lookup(raw) is not None


def test_key_store_file_chmod(tmp_path: Path) -> None:
    """Key file must be chmod 0600 (owner-only)."""
    store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    store.issue("acme")
    mode = (tmp_path / "gateway_keys.json").stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


# ---------------------------------------------------------------------------
# Rate limiter unit tests
# ---------------------------------------------------------------------------


def test_rate_limiter_rpm_allows_then_blocks() -> None:
    rl = _RateLimiter()
    # limit=2: first two pass, third blocked
    assert rl.check_rpm("t1", 2) is True
    assert rl.check_rpm("t1", 2) is True
    assert rl.check_rpm("t1", 2) is False


def test_rate_limiter_rpm_zero_unlimited() -> None:
    rl = _RateLimiter()
    for _ in range(100):
        assert rl.check_rpm("t1", 0) is True


def test_rate_limiter_rpm_window_reset(monkeypatch: Any) -> None:
    """After 60s the window resets and requests are allowed again."""
    rl = _RateLimiter()
    assert rl.check_rpm("t1", 1) is True
    assert rl.check_rpm("t1", 1) is False  # blocked

    # Advance monotonic time by 61s
    original = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original() + 61)
    assert rl.check_rpm("t1", 1) is True  # new window


def test_rate_limiter_daily_tokens_allows_then_blocks() -> None:
    rl = _RateLimiter()
    # 80 tokens used first; then 30 more would put us at 110 > 100 → blocked
    assert rl.check_daily_tokens("t1", 100, 80) is True
    assert rl.check_daily_tokens("t1", 100, 30) is False  # 80+30=110 > 100


def test_rate_limiter_daily_tokens_zero_unlimited() -> None:
    rl = _RateLimiter()
    for _ in range(100):
        assert rl.check_daily_tokens("t1", 0, 1_000_000) is True


def test_rate_limiter_daily_tokens_day_rollover(monkeypatch: Any) -> None:
    rl = _RateLimiter()
    assert rl.check_daily_tokens("t1", 10, 10) is True
    assert rl.check_daily_tokens("t1", 10, 1) is False  # full

    # Advance wall time by 25h so _today() changes
    original_time = time.time
    monkeypatch.setattr(time, "time", lambda: original_time() + 90_000)
    assert rl.check_daily_tokens("t1", 10, 5) is True  # new day


# ---------------------------------------------------------------------------
# Gateway HTTP integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def echo_upstream() -> Any:
    srv = _start(_EchoHandler)
    yield srv
    srv.shutdown()


@pytest.fixture()
def gw_with_keys(echo_upstream: Any, tmp_path: Path) -> Any:
    """Gateway with a key store; yields (port, state, key_store, raw_key)."""
    key_store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    raw, _ = key_store.issue("acme")
    srv, state = _make_gateway(echo_upstream.server_address[1], key_store=key_store)
    yield srv.server_address[1], state, key_store, raw
    srv.shutdown()


def test_no_keys_no_auth_default_path(echo_upstream: Any) -> None:
    """No keys issued + no require_keys → requests pass without auth (backward compat)."""
    srv, _ = _make_gateway(echo_upstream.server_address[1])
    try:
        status, _, _ = _post_messages(srv.server_address[1])
        assert status == 200
    finally:
        srv.shutdown()


def test_require_keys_no_store_returns_401(echo_upstream: Any) -> None:
    """require_keys=True but key_store=None → 401 with helpful message."""
    srv, _ = _make_gateway(echo_upstream.server_address[1], key_store=None, require_keys=True)
    try:
        status, _, data = _post_messages(srv.server_address[1])
        assert status == 401
        err = json.loads(data)
        assert "key store" in err["error"]
    finally:
        srv.shutdown()


def test_missing_key_returns_401(gw_with_keys: Any) -> None:
    """Request without any key header → 401."""
    port, _, _, _ = gw_with_keys
    status, _, data = _post_messages(port)
    assert status == 401
    err = json.loads(data)
    assert "gateway key" in err["error"].lower() or "required" in err["error"].lower()


def test_wrong_key_returns_401(gw_with_keys: Any) -> None:
    """Request with an invalid dsk- key → 401."""
    port, _, _, _ = gw_with_keys
    status, _, data = _post_messages(port, {"Authorization": "Bearer dsk-badkey12345"})
    assert status == 401
    err = json.loads(data)
    assert "invalid" in err["error"].lower() or "revoked" in err["error"].lower()


def test_valid_key_via_authorization_header_passes(gw_with_keys: Any) -> None:
    """Valid dsk- key via Authorization: Bearer → 200."""
    port, _, _, raw = gw_with_keys
    status, _, _ = _post_messages(port, {"Authorization": f"Bearer {raw}"})
    assert status == 200


def test_valid_key_via_x_distil_key_header_passes(gw_with_keys: Any) -> None:
    """Valid dsk- key via x-distil-key header → 200."""
    port, _, _, raw = gw_with_keys
    status, _, _ = _post_messages(port, {"x-distil-key": raw})
    assert status == 200


def test_revoked_key_returns_401(gw_with_keys: Any) -> None:
    """Revoked key → 401 (not 200)."""
    port, _, key_store, raw = gw_with_keys
    keys = key_store.list_keys()
    active = next(k for k in keys if k.is_active)
    key_store.revoke(active.id)
    # Force cache expiry
    key_store._last_load = 0.0  # type: ignore[attr-defined]

    status, _, data = _post_messages(port, {"Authorization": f"Bearer {raw}"})
    assert status == 401


def test_upstream_never_sees_distil_key_via_authorization(gw_with_keys: Any) -> None:
    """Gateway strips the Authorization: Bearer dsk-… before forwarding.

    The echo upstream reflects back any Authorization header it received as
    x-echo-authorization.  When a dsk- key is used, that header must be absent
    from the response (or empty) — the provider must never see the gateway key.
    """
    port, _, _, raw = gw_with_keys
    payload = json.dumps(
        {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    conn.request(
        "POST",
        "/v1/messages",
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "Authorization": f"Bearer {raw}",
            "x-api-key": "sk-provider-key",
        },
    )
    resp = conn.getresponse()
    resp.read()
    echoed_auth = resp.getheader("x-echo-authorization")
    conn.close()
    assert resp.status == 200
    # The dsk- key must NOT have been forwarded to the upstream
    assert echoed_auth is None or "dsk-" not in echoed_auth, (
        f"upstream saw the gateway key: {echoed_auth!r}"
    )


def test_upstream_never_sees_x_distil_key_header(gw_with_keys: Any) -> None:
    """x-distil-key header is stripped before forwarding."""
    port, _, _, raw = gw_with_keys
    payload = json.dumps(
        {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    conn.request(
        "POST",
        "/v1/messages",
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "x-distil-key": raw,
            "x-api-key": "sk-provider-key",
        },
    )
    resp = conn.getresponse()
    resp.read()
    echoed = resp.getheader("x-echo-x-distil-key")
    conn.close()
    assert resp.status == 200
    assert echoed is None, f"upstream saw x-distil-key: {echoed!r}"


def test_provider_credential_passes_through(gw_with_keys: Any) -> None:
    """The provider credential (x-api-key) is forwarded unchanged alongside the dsk- key."""
    port, _, _, raw = gw_with_keys
    payload = json.dumps(
        {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    conn.request(
        "POST",
        "/v1/messages",
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "Authorization": f"Bearer {raw}",
            "x-api-key": "sk-my-provider-key",
        },
    )
    resp = conn.getresponse()
    resp.read()
    echoed_apikey = resp.getheader("x-echo-x-api-key")
    conn.close()
    assert resp.status == 200
    assert echoed_apikey == "sk-my-provider-key", (
        f"provider x-api-key not forwarded: {echoed_apikey!r}"
    )


def test_key_attributed_to_correct_tenant(gw_with_keys: Any) -> None:
    """Accounting records the key's tenant label, not an anon hash."""
    port, state, key_store, raw = gw_with_keys
    # Issue a second key for 'globex' to make sure attribution is per-key
    raw2, _ = key_store.issue("globex")

    _post_messages(port, {"Authorization": f"Bearer {raw}"})  # acme
    _post_messages(port, {"Authorization": f"Bearer {raw2}"})  # globex
    _post_messages(port, {"Authorization": f"Bearer {raw}"})  # acme again

    snap = state.snapshot()
    by_tenant = {t["tenant"]: t for t in snap["tenants"]}
    assert "acme" in by_tenant, f"acme missing: {list(by_tenant)}"
    assert "globex" in by_tenant, f"globex missing: {list(by_tenant)}"
    assert by_tenant["acme"]["requests"] == 2
    assert by_tenant["globex"]["requests"] == 1


def test_rpm_rate_limit_429(echo_upstream: Any, tmp_path: Path) -> None:
    """Gateway-wide RPM=1: second request in the same window → 429."""
    key_store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    raw, _ = key_store.issue("acme")
    srv, _ = _make_gateway(echo_upstream.server_address[1], key_store=key_store, default_rpm=1)
    try:
        s1, _, _ = _post_messages(srv.server_address[1], {"Authorization": f"Bearer {raw}"})
        s2, resp2, data2 = _post_messages(srv.server_address[1], {"Authorization": f"Bearer {raw}"})
        assert s1 == 200, f"first request should pass, got {s1}"
        assert s2 == 429, f"second request should be rate-limited, got {s2}"
        body2 = json.loads(data2)
        assert "rate" in body2["error"].lower() or "limit" in body2["error"].lower()
    finally:
        srv.shutdown()


def test_daily_token_quota_429(echo_upstream: Any, tmp_path: Path) -> None:
    """Per-tenant daily token quota: exceeded → 429."""
    key_store = GatewayKeyStore(tmp_path / "gateway_keys.json")
    raw, _ = key_store.issue("acme")
    # Set an absurdly low daily-token quota so the first real request hits it
    srv, _ = _make_gateway(
        echo_upstream.server_address[1],
        key_store=key_store,
        default_daily_tokens=1,  # 1 token/day — any real request exceeds this
    )
    try:
        s1, _, _ = _post_messages(srv.server_address[1], {"Authorization": f"Bearer {raw}"})
        s2, _, data2 = _post_messages(srv.server_address[1], {"Authorization": f"Bearer {raw}"})
        # One of the two requests must be 429 (quota=1 token, any message exceeds that)
        assert 429 in (s1, s2), f"expected a 429, got {s1} and {s2}"
        if s2 == 429:
            body2 = json.loads(data2)
            assert "quota" in body2["error"].lower() or "daily" in body2["error"].lower()
    finally:
        srv.shutdown()


def test_health_unauthenticated_when_keys_active(gw_with_keys: Any) -> None:
    """/distil/health remains unauthenticated even when key auth is enforced."""
    port, _, _, _ = gw_with_keys
    status, _, data = _req("GET", port, "/distil/health")
    assert status == 200
    assert json.loads(data) == {"status": "ok"}
