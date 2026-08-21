"""Tests for distil.audit — the gateway security audit trail.

Covers: content-free records, 0600 permissions, concurrent append integrity,
tolerance of a truncated tail, and the fail-open contract (an audit write must
never break the request it describes).
"""

from __future__ import annotations

import json
import os
import sys
import threading

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    return tmp_path


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod 0600 is a no-op on Windows (mode reads back 0o666); the owner-only guarantee is POSIX-only",
)
def test_records_are_written_0600_and_readable_back(tmp_path) -> None:
    """The trail names tenants and key ids, so it is not world-readable."""
    from distil import audit

    audit.record(audit.AUTH_OK, key_id="gk_1", tenant="acme", remote="10.0.0.1")
    audit.record(audit.AUTH_FAIL, reason="unknown key", remote="10.0.0.9")

    events = audit.read_events()
    assert [e["event"] for e in events] == [audit.AUTH_OK, audit.AUTH_FAIL]
    assert events[0]["tenant"] == "acme"
    assert all("ts" in e for e in events)
    assert oct(os.stat(audit.audit_path()).st_mode & 0o777) == "0o600"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="line-atomic appends rely on flock, which is POSIX-only; without it concurrent writes can splice",
)
def test_concurrent_appends_never_interleave(tmp_path) -> None:
    """The gateway serves requests concurrently; JSONL lines must stay whole.

    Without the flock, two records larger than the atomic-append size can splice
    into one another and corrupt the trail exactly when it matters most.
    """
    from distil import audit

    def worker(t: int) -> None:
        for i in range(40):
            audit.record(audit.AUTH_OK, key_id=f"gk_{t}", tenant=f"t{t}", seq=i)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    raw = audit.audit_path().read_text().splitlines()
    assert len(raw) == 16 * 40
    for line in raw:  # every line must independently parse
        assert isinstance(json.loads(line), dict)
    assert len(audit.read_events()) == 16 * 40


def test_truncated_tail_does_not_break_the_reader(tmp_path) -> None:
    """A process killed mid-write leaves a partial last line — normal for an
    append-only log, and it must not make the whole trail unreadable."""
    from distil import audit

    audit.record(audit.AUTH_OK, key_id="gk_1", tenant="acme")
    with audit.audit_path().open("a", encoding="utf-8") as fh:
        fh.write('{"event":"truncated"')  # no closing brace, no newline

    events = audit.read_events()
    assert len(events) == 1 and events[0]["key_id"] == "gk_1"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod 0400 does not make a file unwritable on Windows, so the failure path cannot be provoked",
)
def test_write_failures_never_raise(tmp_path) -> None:
    """Fail-open: an unwritable trail must not take the gateway down."""
    from distil import audit

    audit.record(audit.AUTH_OK, key_id="gk_1")
    os.chmod(audit.audit_path(), 0o400)
    try:
        audit.record(audit.AUTH_OK, key_id="gk_2")  # must not raise
    finally:
        os.chmod(audit.audit_path(), 0o600)

    audit.record(audit.AUTH_OK, unserialisable=object())  # must not raise


def test_limit_returns_the_most_recent_events(tmp_path) -> None:
    from distil import audit

    for i in range(10):
        audit.record(audit.AUTH_OK, key_id=f"gk_{i}")

    tail = audit.read_events(limit=3)
    assert [e["key_id"] for e in tail] == ["gk_7", "gk_8", "gk_9"]


def test_gateway_auth_events_reach_the_trail(tmp_path) -> None:
    """End-to-end: a real gateway must record auth failures, successes and
    rate-limit rejections — the three questions an enterprise audit asks.

    Unit-level calls to `audit.record` prove the writer works; only a live request
    proves the gateway is actually wired to it.
    """
    import socket
    import time
    import urllib.error
    import urllib.request

    from distil import audit, gateway
    from distil.gateway_keys import GatewayKeyStore

    good, _ = GatewayKeyStore().issue(tenant="acme", rpm=2)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    threading.Thread(
        target=gateway.serve_gateway,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "upstream": "http://127.0.0.1:1",  # unreachable: we only assert on auth
            "require_keys": True,
        },
        daemon=True,
    ).start()

    deadline = time.time() + 10
    while time.time() < deadline:  # wait for the listener rather than sleeping blind
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)

    def call(key: str | None) -> None:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/messages",
            data=b'{"model":"m","messages":[]}',
            headers=headers,
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except (urllib.error.HTTPError, OSError):
            pass  # upstream is intentionally dead; auth already happened

    call("dsk-totally-wrong")
    for _ in range(4):  # rpm=2, so the last two must be rate limited
        call(good)

    events = [e["event"] for e in audit.read_events()]
    assert audit.AUTH_FAIL in events, "a rejected key must be auditable"
    assert audit.AUTH_OK in events, "a successful auth must be auditable"
    assert audit.RATE_LIMITED in events, "a rate-limit rejection must be auditable"

    # Content-free: no request body, no raw key, ever.
    blob = audit.audit_path().read_text()
    assert good not in blob, "the raw key must never be written to the audit trail"
    assert "messages" not in blob, "request content must never reach the audit trail"


def test_reader_handles_blank_lines_and_a_missing_file(tmp_path) -> None:
    """Blank lines are skipped, and no trail at all reads as no events.

    An operator running `distil gateway audit` before the gateway has ever
    authenticated anyone must get an empty list, not a traceback.
    """
    from distil import audit

    assert audit.read_events() == []  # file does not exist yet

    audit.record(audit.AUTH_OK, key_id="gk_1")
    with audit.audit_path().open("a", encoding="utf-8") as fh:
        fh.write("\n   \n")  # blank and whitespace-only lines

    assert [e["key_id"] for e in audit.read_events()] == ["gk_1"]


def test_unreadable_trail_reads_as_empty(tmp_path, monkeypatch) -> None:
    """A trail that cannot be opened must not raise into the CLI.

    Simulated via a PermissionError from Path.open rather than chmod 0o000:
    chmod is a no-op for root and on Windows, which would silently drop this
    coverage in exactly the containers/CI this test exists to guard.
    """
    from pathlib import Path
    from typing import Any

    from distil import audit

    audit.record(audit.AUTH_OK, key_id="gk_1")

    real_open = Path.open

    def _unreadable(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == audit.audit_path():
            raise PermissionError("simulated: audit trail is unreadable")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _unreadable)
    assert audit.read_events() == []


def test_gateway_audit_cli_renders_filters_and_json(tmp_path, capsys) -> None:
    """`distil gateway audit` is how an operator actually reads the trail.

    Covers the empty case, the human table, both filters, and --json, because a
    trail nobody can read closes no compliance gap.
    """
    from distil.cli import main

    from distil import audit

    assert main(["gateway", "audit"]) == 0
    assert "No audit events yet" in capsys.readouterr().out

    audit.record(audit.KEY_ISSUED, key_id="gk_a", tenant="acme", expires=1_800_000_000.0)
    audit.record(audit.AUTH_FAIL, reason="invalid or revoked gateway key", remote="10.0.0.9")
    audit.record(audit.AUTH_OK, key_id="gk_b", tenant="beta", remote="10.0.0.1")

    assert main(["gateway", "audit"]) == 0
    out = capsys.readouterr().out
    assert "key.issued" in out and "auth.fail" in out and "auth.ok" in out
    assert "2027-01-15" in out, "an expiry timestamp must render as a date, not a raw float"

    assert main(["gateway", "audit", "--event", "auth.fail"]) == 0
    out = capsys.readouterr().out
    assert "auth.fail" in out and "auth.ok" not in out

    assert main(["gateway", "audit", "--tenant", "beta"]) == 0
    out = capsys.readouterr().out
    assert "gk_b" in out and "gk_a" not in out

    assert main(["gateway", "audit", "--json", "-n", "1"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 1 and json.loads(lines[0])["event"] == audit.AUTH_OK


def test_gateway_keys_cli_reports_expiry_and_status(tmp_path, capsys) -> None:
    """`keys issue --expires-in-days` and the list status column.

    `list` must distinguish expired from revoked: an operator debugging a sudden
    401 should not go hunting for a revocation that never happened.
    """
    from distil.cli import main

    assert main(["gateway", "keys", "issue", "--tenant", "acme", "--expires-in-days", "30"]) == 0
    assert "expires:" in capsys.readouterr().out

    assert main(["gateway", "keys", "issue", "--tenant", "stale", "--expires-in-days", "-1"]) == 0
    capsys.readouterr()
    assert main(["gateway", "keys", "issue", "--tenant", "forever"]) == 0
    capsys.readouterr()

    assert main(["gateway", "keys", "list"]) == 0
    rows = capsys.readouterr().out
    assert "expired" in rows, "an expired key must be labelled expired, not revoked"
    assert "active" in rows


def _start_gateway(tmp_path, **kwargs):
    """Boot a gateway on an ephemeral port with a dead upstream; return its port."""
    import socket
    import time

    from distil import gateway

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    threading.Thread(
        target=gateway.serve_gateway,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "upstream": "http://127.0.0.1:1",  # dead on purpose: auth happens first
            "require_keys": True,
            **kwargs,
        },
        daemon=True,
    ).start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return port
        except OSError:
            time.sleep(0.05)
    raise AssertionError("gateway did not start")


def _post(port: int, token: str | None = None) -> None:
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=b'{"model":"m","messages":[]}',
        headers=headers,
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except (urllib.error.HTTPError, OSError):
        pass


def test_every_refusal_path_is_audited(tmp_path) -> None:
    """A refusal the trail does not record is a compliance gap, not a detail.

    The first cut only audited the `dsk-` happy path plus invalid-key, which left
    "no credential presented at all" — the single most common refusal — invisible.
    """
    from distil import audit
    from distil.gateway_keys import GatewayKeyStore

    good, _ = GatewayKeyStore().issue(tenant="acme", rpm=2)
    port = _start_gateway(tmp_path)

    _post(port)  # no credential
    _post(port, "dsk-bogus")  # invalid key
    for _ in range(4):  # rpm=2 → two allowed, two limited
        _post(port, good)

    events = audit.read_events()
    reasons = {e.get("reason") for e in events if e.get("reason")}
    assert "no gateway key presented" in reasons, "an unauthenticated request must be auditable"
    assert "invalid or revoked gateway key" in reasons
    assert audit.AUTH_OK in {e["event"] for e in events}
    assert audit.RATE_LIMITED in {e["event"] for e in events}


def test_oidc_paths_are_audited(tmp_path, monkeypatch) -> None:
    """OIDC rejection, RBAC denial and OIDC success must all reach the trail.

    These bypassed auditing entirely in the first cut: a deployment on OIDC got an
    empty audit log no matter how much traffic it refused.
    """
    import base64
    import hashlib
    import hmac
    import time

    from distil import audit

    monkeypatch.setenv("DISTIL_OIDC_ISSUER", "https://idp.example")
    monkeypatch.setenv("DISTIL_OIDC_HS256_SECRET", "topsecret")
    monkeypatch.setenv("DISTIL_OIDC_AUDIENCE", "distil")

    def _b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    def _jwt(**claims) -> str:
        header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        payload = _b64(json.dumps(claims).encode())
        sig = hmac.new(b"topsecret", f"{header}.{payload}".encode(), hashlib.sha256).digest()
        return f"{header}.{payload}.{_b64(sig)}"

    from distil.gateway_keys import GatewayKeyStore

    GatewayKeyStore().issue(tenant="seed")  # key store must exist for require_keys
    port = _start_gateway(tmp_path)

    now = time.time()
    base = {"sub": "u", "aud": "distil", "iss": "https://idp.example", "tenant": "acme"}
    _post(port, _jwt(**base, exp=now - 9999, role="operator"))  # expired
    _post(port, _jwt(**base, exp=now + 3600, role="viewer"))  # role too low → 403
    _post(port, _jwt(**base, exp=now + 3600, role="operator"))  # success

    events = audit.read_events()
    reasons = " ".join(e.get("reason", "") for e in events)
    assert "token expired" in reasons, "an expired OIDC token must be auditable"
    assert "role check failed" in reasons, "an RBAC denial must be auditable"
    assert any(e.get("auth") == "oidc" and e["event"] == audit.AUTH_OK for e in events)
