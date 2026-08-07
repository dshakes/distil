"""RBAC + OIDC. Tested adversarially: the interesting cases are the attacks.

Every classic JWT vulnerability gets an explicit test here, because "we didn't
implement that mistake" is only credible if something fails when you do.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

from distil.authz import (
    ROLE_ORDER,
    AuthzError,
    Identity,
    identity_from_claims,
    parse_role,
    verify_jwt,
)

SECRET = "correct horse battery staple"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _make(claims: dict, *, alg: str = "HS256", secret: str = SECRET, sig: bytes | None = None):
    h = _b64(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    p = _b64(json.dumps(claims).encode())
    signing_input = f"{h}.{p}".encode()
    if sig is None:
        sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64(sig)}"


def _claims(**over):
    base = {"sub": "u1", "tenant": "acme", "role": "operator", "exp": time.time() + 3600}
    base.update(over)
    return base


# --- the attacks -------------------------------------------------------------


def test_alg_none_is_refused():
    """The canonical JWT bypass: a token declaring it needs no signature."""
    tok = _make(_claims(), alg="none", sig=b"")
    with pytest.raises(AuthzError, match="unsupported or unsafe alg"):
        verify_jwt(tok, secret=SECRET)


def test_tampered_payload_is_refused():
    """Escalate role in the payload, keep the old signature."""
    good = _make(_claims(role="viewer"))
    h, _p, s = good.split(".")
    forged_payload = _b64(json.dumps(_claims(role="admin")).encode())
    with pytest.raises(AuthzError, match="signature verification failed"):
        verify_jwt(f"{h}.{forged_payload}.{s}", secret=SECRET)


def test_wrong_secret_is_refused():
    with pytest.raises(AuthzError, match="signature verification failed"):
        verify_jwt(_make(_claims()), secret="not the secret")


def test_rs256_without_a_configured_key_is_refused_not_bypassed():
    """An attacker switching alg to RS256 must not skip verification."""
    tok = _make(_claims(), alg="RS256", sig=b"whatever")
    with pytest.raises(AuthzError, match="no public key is configured"):
        verify_jwt(tok, secret=SECRET)


def test_hs256_without_a_secret_is_refused():
    with pytest.raises(AuthzError, match="no shared secret"):
        verify_jwt(_make(_claims()), secret="")


def test_expired_token_is_refused():
    with pytest.raises(AuthzError, match="expired"):
        verify_jwt(_make(_claims(exp=time.time() - 7200)), secret=SECRET)


def test_not_yet_valid_token_is_refused():
    with pytest.raises(AuthzError, match="not yet valid"):
        verify_jwt(_make(_claims(nbf=time.time() + 7200)), secret=SECRET)


def test_issuer_and_audience_are_enforced_when_configured():
    tok = _make(_claims(iss="https://idp.example", aud="distil"))
    verify_jwt(tok, secret=SECRET, issuer="https://idp.example", audience="distil")
    with pytest.raises(AuthzError, match="issuer mismatch"):
        verify_jwt(tok, secret=SECRET, issuer="https://evil.example")
    with pytest.raises(AuthzError, match="audience mismatch"):
        verify_jwt(tok, secret=SECRET, audience="other-service")


def test_audience_array_form_is_accepted():
    tok = _make(_claims(aud=["other", "distil"]))
    assert verify_jwt(tok, secret=SECRET, audience="distil")["sub"] == "u1"


@pytest.mark.parametrize("tok", ["", "a.b", "a.b.c.d", "not-a-token", "...", "!!!.???.***"])
def test_malformed_tokens_raise_rather_than_crash(tok):
    with pytest.raises(AuthzError):
        verify_jwt(tok, secret=SECRET)


def test_non_object_payload_is_refused():
    h = _b64(json.dumps({"alg": "HS256"}).encode())
    p = _b64(json.dumps(["not", "an", "object"]).encode())
    sig = hmac.new(SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    with pytest.raises(AuthzError, match="must be objects"):
        verify_jwt(f"{h}.{p}.{_b64(sig)}", secret=SECRET)


# --- roles -------------------------------------------------------------------


def test_role_ordering_is_transitive():
    admin = Identity("u", "t", "admin", "oidc")
    operator = Identity("u", "t", "operator", "oidc")
    viewer = Identity("u", "t", "viewer", "oidc")

    assert admin.can("viewer") and admin.can("operator") and admin.can("admin")
    assert operator.can("viewer") and operator.can("operator")
    assert not operator.can("admin")
    assert viewer.can("viewer")
    assert not viewer.can("operator") and not viewer.can("admin")


def test_require_raises_for_insufficient_role():
    with pytest.raises(AuthzError, match="insufficient"):
        Identity("u", "t", "viewer", "oidc").require("admin")


def test_unknown_role_never_escalates():
    """An IdP group distil has never seen must not become admin."""
    assert parse_role("superuser") == "operator"
    assert parse_role(None) == "operator"
    assert parse_role("") == "operator"
    assert not Identity("u", "t", "wat", "oidc").can("viewer"), "unknown role denies"


def test_role_is_case_insensitive():
    assert parse_role("ADMIN") == "admin"


def test_highest_role_wins_in_a_groups_array():
    ident = identity_from_claims(_claims(role=["viewer", "admin", "operator"]))
    assert ident.role == "admin"


def test_unknown_roles_in_an_array_are_ignored():
    ident = identity_from_claims(_claims(role=["wheel", "superuser"]))
    assert ident.role == "operator", "no known role present => default, never admin"


def test_tenant_falls_back_to_subject():
    ident = identity_from_claims({"sub": "u9"})
    assert ident.tenant == "u9" and ident.source == "oidc"


def test_identity_expiry_is_exposed():
    past = identity_from_claims(_claims(exp=time.time() - 10))
    assert past.is_expired
    assert not identity_from_claims(_claims()).is_expired


def test_role_order_constant_is_ascending():
    assert ROLE_ORDER == ("viewer", "operator", "admin")


# --- end to end through a live gateway ---------------------------------------
# The module-level tests above prove the crypto. These prove it is actually WIRED:
# an auth library nothing calls protects nothing.


def _live_gateway(monkeypatch, tmp_path, **oidc):
    import threading
    from http.server import ThreadingHTTPServer

    from distil.gateway import GatewayState, build_gateway_handler
    from distil.gateway_keys import GatewayKeyStore
    from distil.pricing import get as pricing_get
    from tests.test_gateway_keys import _EchoHandler, _start

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    for k, v in oidc.items():
        monkeypatch.setenv(k, v)

    up = _start(_EchoHandler)
    price = pricing_get("claude-opus-4-8")
    state = GatewayState(price)
    store = GatewayKeyStore(tmp_path / "keys.json")
    handler = build_gateway_handler(
        f"http://127.0.0.1:{up.server_address[1]}",
        state,
        price,
        key_store=store,
        require_keys=True,
    )
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, up, state


def _call(srv, token: str | None):
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    conn.request(
        "POST",
        "/v1/messages",
        json.dumps({"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}),
        headers,
    )
    r = conn.getresponse()
    r.read()
    conn.close()
    return r.status


def test_e2e_valid_oidc_token_is_accepted(tmp_path, monkeypatch):
    srv, up, state = _live_gateway(
        monkeypatch,
        tmp_path,
        DISTIL_OIDC_ISSUER="https://idp.example",
        DISTIL_OIDC_HS256_SECRET=SECRET,
    )
    try:
        tok = _make(_claims(iss="https://idp.example", role="operator", tenant="acme"))
        assert _call(srv, tok) == 200
        # accounting must attribute the request to the token's tenant
        tenants = {r["tenant"] for r in state.snapshot()["tenants"]}
        assert "acme" in tenants
    finally:
        srv.shutdown()
        up.shutdown()


def test_e2e_forged_token_is_rejected_with_401(tmp_path, monkeypatch):
    srv, up, _ = _live_gateway(
        monkeypatch,
        tmp_path,
        DISTIL_OIDC_ISSUER="https://idp.example",
        DISTIL_OIDC_HS256_SECRET=SECRET,
    )
    try:
        forged = _make(_claims(iss="https://idp.example"), secret="wrong secret")
        assert _call(srv, forged) == 401
    finally:
        srv.shutdown()
        up.shutdown()


def test_e2e_viewer_role_cannot_proxy_requests(tmp_path, monkeypatch):
    """Least privilege actually enforced: read-only means read-only."""
    srv, up, _ = _live_gateway(
        monkeypatch,
        tmp_path,
        DISTIL_OIDC_ISSUER="https://idp.example",
        DISTIL_OIDC_HS256_SECRET=SECRET,
    )
    try:
        tok = _make(_claims(iss="https://idp.example", role="viewer"))
        assert _call(srv, tok) == 403
    finally:
        srv.shutdown()
        up.shutdown()


def test_e2e_oidc_disabled_means_dsk_keys_still_required(tmp_path, monkeypatch):
    """Enabling nothing changes nothing — OIDC is additive, never a bypass."""
    srv, up, _ = _live_gateway(monkeypatch, tmp_path)  # no DISTIL_OIDC_ISSUER
    try:
        tok = _make(_claims())
        assert _call(srv, tok) == 401, "a JWT must not authenticate when OIDC is off"
        assert _call(srv, None) == 401
    finally:
        srv.shutdown()
        up.shutdown()
