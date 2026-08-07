"""Role-based access control and OIDC identity for the gateway.

Design constraints, in priority order:

1. **No new runtime dependency.** distil's core is stdlib-only, and a security
   component is the worst place to break that — a JWT library is a supply-chain
   surface sitting directly in the auth path. JWS verification here is written
   against ``hmac``/``hashlib`` for HS256 and, for RS256, against the ``crypto``
   module only when the optional ``[oidc]`` extra is installed. If asymmetric
   verification is unavailable, distil **refuses the token** rather than falling
   back to an unverified decode.
2. **Fail closed.** Every path that cannot prove a claim denies it. There is no
   "if we can't check the signature, trust the payload" branch — that is the
   single most common JWT vulnerability and it is absent by construction.
3. **Additive.** ``dsk-`` bearer keys keep working exactly as before. OIDC is a
   second way to present an identity, not a replacement, so enabling it cannot
   lock out an existing deployment.

Roles
-----
Three, deliberately few — an RBAC model with twenty verbs nobody can reason about
is worse than one with three that everybody can:

``viewer``    read stats and metrics; cannot proxy requests or change anything
``operator``  everything a viewer can, plus proxy requests through the gateway
``admin``     everything, plus key issue/revoke and the admin dashboard

Roles are ordered: an ``admin`` satisfies any requirement an ``operator`` does.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Identity",
    "Role",
    "AuthzError",
    "ROLE_ORDER",
    "parse_role",
    "verify_jwt",
    "identity_from_claims",
]


class AuthzError(Exception):
    """Authentication or authorization failed. The message is safe to log."""


# Ascending privilege. Index comparison is the whole authorization model.
ROLE_ORDER: tuple[str, ...] = ("viewer", "operator", "admin")

Role = str


def parse_role(value: str | None, *, default: str = "operator") -> Role:
    """Normalise a role string, falling back to *default* for unknown values.

    Unknown roles degrade to the default rather than raising: an identity
    provider adding a group distil has never heard of must not take the gateway
    down. It must also never silently escalate, which is why the default is
    ``operator`` and never ``admin``.
    """
    v = (value or "").strip().lower()
    return v if v in ROLE_ORDER else default


@dataclass(frozen=True)
class Identity:
    """Who is making this request, and what they may do."""

    subject: str
    tenant: str
    role: Role
    source: str  # "key" | "oidc"
    expires: float | None = None

    def can(self, required: Role) -> bool:
        """True when this identity's role is at least *required*."""
        try:
            return ROLE_ORDER.index(self.role) >= ROLE_ORDER.index(required)
        except ValueError:  # unknown role: deny rather than guess
            return False

    def require(self, required: Role) -> None:
        if not self.can(required):
            raise AuthzError(f"role '{self.role}' is insufficient; '{required}' required")

    @property
    def is_expired(self) -> bool:
        return self.expires is not None and time.time() >= self.expires


def _b64url_decode(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def _verify_hs256(signing_input: bytes, sig: bytes, secret: str) -> bool:
    expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return hmac.compare_digest(expected, sig)  # constant time — not `==`


def _verify_rs256(signing_input: bytes, sig: bytes, public_key_pem: str) -> bool:
    """RS256 via the optional [oidc] extra. Absent extra => refuse, never bypass."""
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise AuthzError(
            "RS256 token presented but asymmetric verification is unavailable; "
            "install the [oidc] extra. Refusing rather than skipping verification."
        ) from exc
    try:
        key = serialization.load_pem_public_key(public_key_pem.encode())
        key.verify(sig, signing_input, padding.PKCS1v15(), hashes.SHA256())  # type: ignore[union-attr]
        return True
    except Exception:  # noqa: BLE001 — any verification failure is a failure
        return False


def verify_jwt(
    token: str,
    *,
    secret: str = "",
    public_key_pem: str = "",
    issuer: str = "",
    audience: str = "",
    leeway: float = 60.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify a compact JWS and return its claims, or raise :class:`AuthzError`.

    Checks, in order: structure, algorithm allow-list, signature, ``exp``,
    ``nbf``, ``iss``, ``aud``. Every one is mandatory when configured; none can be
    skipped by anything in the token itself.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthzError("malformed token: expected three dot-separated segments")
    h_seg, p_seg, s_seg = parts
    try:
        header = json.loads(_b64url_decode(h_seg))
        claims = json.loads(_b64url_decode(p_seg))
        signature = _b64url_decode(s_seg)
    except (ValueError, TypeError) as exc:
        raise AuthzError("malformed token: undecodable segment") from exc
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise AuthzError("malformed token: header and payload must be objects")

    alg = str(header.get("alg", ""))
    # Allow-list, and `none` can never appear in it. A token that names its own
    # algorithm is choosing its own verification, so the SERVER decides.
    if alg not in ("HS256", "RS256"):
        raise AuthzError(f"unsupported or unsafe alg: {alg!r}")

    signing_input = f"{h_seg}.{p_seg}".encode()
    if alg == "HS256":
        if not secret:
            raise AuthzError("HS256 token presented but no shared secret is configured")
        ok = _verify_hs256(signing_input, signature, secret)
    else:
        if not public_key_pem:
            raise AuthzError("RS256 token presented but no public key is configured")
        ok = _verify_rs256(signing_input, signature, public_key_pem)
    if not ok:
        raise AuthzError("signature verification failed")

    t = time.time() if now is None else now
    exp = claims.get("exp")
    if exp is not None:
        try:
            if t >= float(exp) + leeway:
                raise AuthzError("token expired")
        except (TypeError, ValueError) as exc:
            raise AuthzError("invalid exp claim") from exc
    nbf = claims.get("nbf")
    if nbf is not None:
        try:
            if t + leeway < float(nbf):
                raise AuthzError("token not yet valid")
        except (TypeError, ValueError) as exc:
            raise AuthzError("invalid nbf claim") from exc
    if issuer and claims.get("iss") != issuer:
        raise AuthzError("issuer mismatch")
    if audience:
        aud = claims.get("aud")
        auds = aud if isinstance(aud, list) else [aud]
        if audience not in auds:
            raise AuthzError("audience mismatch")
    return claims


def identity_from_claims(
    claims: dict[str, Any],
    *,
    role_claim: str = "role",
    tenant_claim: str = "tenant",
    default_role: str = "operator",
) -> Identity:
    """Map verified claims onto an :class:`Identity`.

    The role may arrive as a scalar or inside a groups/roles array (the common
    Okta/Entra shape). The **highest** role present wins, which is the least
    surprising reading of "this user is in both groups".
    """
    raw = claims.get(role_claim)
    role = parse_role(None, default=default_role)
    if isinstance(raw, str):
        role = parse_role(raw, default=default_role)
    elif isinstance(raw, (list, tuple)):
        found = [parse_role(r, default="") for r in raw if isinstance(r, str)]
        found = [r for r in found if r in ROLE_ORDER]
        if found:
            role = max(found, key=ROLE_ORDER.index)

    subject = str(claims.get("sub") or "unknown")
    tenant = str(claims.get(tenant_claim) or subject)
    exp = claims.get("exp")
    try:
        expires = float(exp) if exp is not None else None
    except (TypeError, ValueError):
        expires = None
    return Identity(subject=subject, tenant=tenant, role=role, source="oidc", expires=expires)


def oidc_config_from_env() -> dict[str, str]:
    """Read OIDC settings from the environment. Empty issuer disables OIDC."""
    return {
        "issuer": os.environ.get("DISTIL_OIDC_ISSUER", ""),
        "audience": os.environ.get("DISTIL_OIDC_AUDIENCE", ""),
        "secret": os.environ.get("DISTIL_OIDC_HS256_SECRET", ""),
        "public_key_pem": os.environ.get("DISTIL_OIDC_PUBLIC_KEY", ""),
        "role_claim": os.environ.get("DISTIL_OIDC_ROLE_CLAIM", "role"),
        "tenant_claim": os.environ.get("DISTIL_OIDC_TENANT_CLAIM", "tenant"),
    }
