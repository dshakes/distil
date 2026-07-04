"""`distil doctor` — setup diagnosis. Checks must never crash, and the proxy
self-test must round-trip a request through an in-process upstream."""

from __future__ import annotations

from distil import doctor


def test_diagnose_runs_every_check_without_crashing() -> None:
    checks = doctor.diagnose()
    assert checks
    names = {c.name for c in checks}
    assert "distil" in names
    assert "proxy self-test" in names
    for c in checks:
        assert c.status in (doctor.OK, doctor.WARN, doctor.INFO, doctor.FAIL)
        assert c.detail  # every check explains itself


def test_proxy_selftest_round_trips() -> None:
    # The headline check: a request must route through the distil proxy to an
    # in-process fake upstream and back — no network, fully self-contained.
    c = doctor._check_proxy_selftest()
    assert c.status == doctor.OK, c.detail


def test_version_check_ok() -> None:
    c = doctor._check_version()
    assert c.status == doctor.OK  # we run on a supported Python


def test_subscription_mode_env_override(monkeypatch) -> None:
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    assert doctor.subscription_mode() is True
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    assert doctor.subscription_mode() is False


def test_subscription_mode_metered_key_means_real_dollars(monkeypatch) -> None:
    # A metered API key set, no explicit override → dollars are real, not notional.
    monkeypatch.delenv("DISTIL_SUBSCRIPTION", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert doctor.subscription_mode() is False


def test_mode_check_warns_on_verbatim_service(tmp_path, monkeypatch):
    """A verbatim always-on service must be flagged — it caps savings ~0."""
    import platform

    from distil import doctor

    svc = tmp_path / "Library" / "LaunchAgents" / "com.distil.proxy.plist"
    svc.parent.mkdir(parents=True)
    svc.write_text("<string>distil</string><string>proxy</string><string>--verbatim</string>")
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    ch = doctor._check_mode()
    assert ch.status == doctor.WARN
    assert "VERBATIM" in ch.detail
    # lossless-only is healthy
    svc.write_text("<string>proxy</string><string>--lossless-only</string>")
    assert doctor._check_mode().status == doctor.OK


def test_shadowed_install_warns(monkeypatch):
    """Two distil on PATH (brew active, pipx shadowed) must be flagged."""
    from distil import doctor

    monkeypatch.setattr(
        doctor, "_find_all_distil",
        lambda: ["/usr/local/bin/distil", "/Users/x/.local/bin/distil"],
    )
    ch = doctor._check_shadowed_install()
    assert ch.status == doctor.WARN
    assert "homebrew" in ch.detail and "pipx" in ch.detail
    assert "ACTIVE: /usr/local/bin/distil" in ch.detail
    # single install is fine
    monkeypatch.setattr(doctor, "_find_all_distil", lambda: ["/usr/local/bin/distil"])
    assert doctor._check_shadowed_install().status == doctor.OK


def test_live_routing_warns_on_bypass(monkeypatch):
    """wrap running + stale ledger → WARN 'bypassing distil'."""
    import subprocess
    import time as _t
    from distil import doctor, ledger

    class _P:
        returncode = 0
        stdout = "user 123 distil wrap --lossless-only -- claude\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _P())
    monkeypatch.setattr(ledger, "latest_session", lambda: ("s1", _t.time() - 600))  # 10m stale
    ch = doctor._check_live_routing()
    assert ch.status == doctor.WARN and "bypassing" in ch.hint

    # fresh traffic → OK
    monkeypatch.setattr(ledger, "latest_session", lambda: ("s1", _t.time() - 60))
    assert doctor._check_live_routing().status == doctor.OK

    # no wrap running → INFO
    class _P2:
        returncode = 0
        stdout = "user 123 some-other-process\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _P2())
    assert doctor._check_live_routing().status == doctor.INFO
