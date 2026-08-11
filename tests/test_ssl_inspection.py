"""Corporate SSL inspection: the failure that looks like distil being broken.

Behind an inspecting proxy (Zscaler, Netskope, a company MITM box) every outbound
TLS connection is re-signed by an internal CA. curl and the browser read the OS
trust store; Python does not. So distil's upstream call fails certificate
verification while everything else on the machine works, and the user reasonably
concludes distil is at fault.

distil honours the variables such an environment has ALREADY exported for curl and
requests, rather than inventing a distil-only name nobody has set.
"""

from __future__ import annotations

import ssl

import pytest

from distil import proxy

VARS = ("DISTIL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)


def _bundle(tmp_path):
    p = tmp_path / "corp-ca.pem"
    p.write_text("-----BEGIN CERTIFICATE-----\nnot a real cert\n-----END CERTIFICATE-----\n")
    return p


# ---------------------------------------------------------------------------
# Which bundle wins
# ---------------------------------------------------------------------------


def test_no_variables_means_the_system_store(monkeypatch):
    assert proxy.ca_bundle_path() is None


@pytest.mark.parametrize("var", VARS)
def test_each_supported_variable_is_honoured(var, monkeypatch, tmp_path):
    monkeypatch.setenv(var, str(_bundle(tmp_path)))
    assert proxy.ca_bundle_path() == str(_bundle(tmp_path))


def test_precedence_is_most_specific_first(monkeypatch, tmp_path):
    """A machine can have several of these exported; only one takes effect, and
    doctor names which — otherwise the user edits the wrong one and nothing
    changes."""
    specific = tmp_path / "distil.pem"
    specific.write_text("x")
    monkeypatch.setenv("DISTIL_CA_BUNDLE", str(specific))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(_bundle(tmp_path)))
    assert proxy.ca_bundle_path() == str(specific)
    assert proxy._which_ca_var() == "DISTIL_CA_BUNDLE"


def test_a_path_that_does_not_exist_is_ignored(monkeypatch, tmp_path):
    """A stale export in a shell profile must not take the proxy down at startup.

    Passing a missing file to OpenSSL raises while building the context, which
    would turn someone else's leftover environment variable into "distil will not
    start" — a worse failure than the one this feature exists to fix.
    """
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "gone.pem"))
    assert proxy.ca_bundle_path() is None
    proxy._build_opener()  # must not raise


def test_a_real_bundle_produces_a_verifying_context(monkeypatch, tmp_path):
    """The bundle is actually installed on the opener, and it still VERIFIES.

    There is no distil switch to disable verification, deliberately: "it works
    with verification off" is how a proxy holding every prompt on the machine
    ends up trusting anything on the wire.
    """
    import urllib.request
    from pathlib import Path

    default = ssl.get_default_verify_paths().cafile
    if not default:
        pytest.skip("no system CA file to copy on this platform")
    real = tmp_path / "real.pem"
    real.write_bytes(Path(default).read_bytes())  # a guaranteed-valid PEM

    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(real))
    handlers = [
        h for h in proxy._build_opener().handlers if isinstance(h, urllib.request.HTTPSHandler)
    ]
    assert handlers, "no HTTPS handler installed, so the bundle would be silently ignored"
    ctx = handlers[0]._context
    assert ctx.verify_mode == ssl.CERT_REQUIRED, "verification was weakened"
    assert ctx.check_hostname is True


# ---------------------------------------------------------------------------
# The message, which is the actual deliverable
# ---------------------------------------------------------------------------


def test_a_verification_failure_explains_itself():
    hint = proxy.tls_failure_hint(
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
        "unable to get local issuer certificate"
    )
    assert hint is not None
    assert "REQUESTS_CA_BUNDLE" in hint, "the fix must be named, not implied"
    assert "corporate" in hint.lower() or "inspect" in hint.lower(), "the cause must be named"


def test_a_failure_with_a_bundle_already_set_says_the_bundle_is_wrong(monkeypatch, tmp_path):
    """Different problem, different sentence: the bundle exists but lacks the issuer.
    Repeating "set REQUESTS_CA_BUNDLE" at someone who already did is useless."""
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(_bundle(tmp_path)))
    hint = proxy.tls_failure_hint("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    assert hint is not None and "corp-ca.pem" in hint
    assert "full corporate root chain" in hint


def test_other_connection_errors_get_no_certificate_hint():
    """A certificate hint on a DNS or refused-connection failure sends the reader
    somewhere the problem is not, which is worse than saying nothing."""
    assert proxy.tls_failure_hint("[Errno 61] Connection refused") is None
    assert proxy.tls_failure_hint("[Errno 8] nodename nor servname provided") is None
    assert proxy.tls_failure_hint("timed out") is None


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_names_the_active_bundle(monkeypatch, tmp_path):
    from distil.doctor import _check_tls_trust

    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(_bundle(tmp_path)))
    c = _check_tls_trust()
    assert c.status == "ok" and "corp-ca.pem" in c.detail
    assert "REQUESTS_CA_BUNDLE" in c.hint


def test_doctor_warns_about_a_variable_pointing_nowhere(monkeypatch, tmp_path):
    """Silently ignored is the worst outcome: the user believes they configured it."""
    from distil.doctor import _check_tls_trust

    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    c = _check_tls_trust()
    assert c.status == "warn" and "SSL_CERT_FILE" in c.detail


def test_doctor_is_quiet_on_a_normal_machine():
    from distil.doctor import _check_tls_trust

    c = _check_tls_trust()
    assert c.status == "ok" and "system trust store" in c.detail
