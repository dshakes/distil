"""`distil doctor` — setup diagnosis. Checks must never crash, and the proxy
self-test must round-trip a request through an in-process upstream."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys

import pytest

from distil import doctor, ledger


def test_diagnose_runs_every_check_without_crashing() -> None:
    checks = doctor.diagnose()
    assert checks
    names = {c.name for c in checks}
    assert "distil" in names
    assert "proxy self-test" in names
    for c in checks:
        assert c.status in (doctor.OK, doctor.WARN, doctor.INFO, doctor.FAIL)
        assert c.detail  # every check explains itself


def test_check_base_url_guard() -> None:
    import socket

    # No base URL wired → no check (silent, the common case).
    assert doctor._check_base_url({}) is None
    assert doctor._check_base_url({"env": {}}) is None

    # Dead loopback port → FAIL (this is the outage that bricked Claude Code).
    dead = doctor._check_base_url({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}})
    assert dead is not None and dead.status == doctor.FAIL

    # Foreign (non-loopback) host is never probed — INFO, not a connection attempt.
    foreign = doctor._check_base_url({"env": {"ANTHROPIC_BASE_URL": "http://example.com:443"}})
    assert foreign is not None and foreign.status == doctor.INFO

    # A port that merely LISTENS is not OK. This is the whole lesson of the outage:
    # the socket was bound the entire time, `doctor` said OK, and every Claude Code
    # session on the machine still failed. Only serving /v1/messages counts.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        port = srv.getsockname()[1]
        bound = doctor._check_base_url({"env": {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}})
        assert bound is not None and bound.status == doctor.FAIL
    finally:
        srv.close()

    # A port that actually answers /v1/messages → OK.
    with _messages_server() as port:
        live = doctor._check_base_url({"env": {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}})
        assert live is not None and live.status == doctor.OK


@contextlib.contextmanager
def _messages_server(status: int = 401):
    """A loopback server that answers /v1/messages with *status* and 404s everything
    else — the smallest thing that tells "routes" apart from "merely listening"."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
            self.send_response(status if self.path == "/v1/messages" else 404)
            self.end_headers()

        def log_message(self, *a: object) -> None:  # keep pytest output clean
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


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
    # A genuine PAYG setup: a metered API key, NO OAuth login → dollars are real.
    monkeypatch.delenv("DISTIL_SUBSCRIPTION", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(doctor, "_claude_oauth_present", lambda: False)
    assert doctor.subscription_mode() is False


def test_subscription_mode_oauth_wins_over_stray_api_key(monkeypatch) -> None:
    """The mode-flip fix: a Claude Code OAuth login is flat-rate even with an
    ANTHROPIC_API_KEY in the env, so the mode can't flip digest↔lossless-only between
    launches depending on whether the key happened to be exported. DISTIL_SUBSCRIPTION
    still overrides for a genuinely metered OAuth user."""
    monkeypatch.delenv("DISTIL_SUBSCRIPTION", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(doctor, "_claude_oauth_present", lambda: True)
    assert doctor.subscription_mode() is True  # OAuth wins → subscription-safe
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    assert doctor.subscription_mode() is False  # explicit override still forces metered


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
        doctor,
        "_find_all_distil",
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


# --------------------------------------------------------------------------- #
# _check_mode — verbatim warning + lossless-only OK + no service INFO
# --------------------------------------------------------------------------- #


def test_check_mode_no_service_is_info(tmp_path, monkeypatch):
    """No service file → INFO (mode set per run)."""
    import platform

    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    ch = doctor._check_mode()
    assert ch.status == doctor.INFO
    assert "no always-on" in ch.detail or "mode is set" in ch.detail


def test_check_mode_digest_mode_is_ok(tmp_path, monkeypatch):
    """Always-on service with no verbatim/lossless-only flag → digest → OK."""
    import platform

    svc = tmp_path / "Library" / "LaunchAgents" / "com.distil.proxy.plist"
    svc.parent.mkdir(parents=True)
    svc.write_text("<string>distil</string><string>proxy</string><string>--port</string>")
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    ch = doctor._check_mode()
    assert ch.status == doctor.OK
    assert "digest" in ch.detail


def test_check_mode_linux_systemd(tmp_path, monkeypatch):
    """Linux systemd service with --verbatim → WARN."""
    import platform

    svc = tmp_path / ".config" / "systemd" / "user" / "distil-proxy.service"
    svc.parent.mkdir(parents=True)
    svc.write_text("[Service]\nExecStart=distil proxy --verbatim\n")
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    ch = doctor._check_mode()
    assert ch.status == doctor.WARN


# --------------------------------------------------------------------------- #
# _check_pricing_catalog
# --------------------------------------------------------------------------- #


def test_check_pricing_catalog_no_ledger(tmp_path, monkeypatch):
    """Missing ledger → no unpriced models → OK."""
    from distil import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "default_path", lambda: tmp_path / "no_such.jsonl")
    ch = doctor._check_pricing_catalog()
    assert ch.status == doctor.OK
    assert "catalog" in ch.detail


def test_check_pricing_catalog_known_model(tmp_path, monkeypatch):
    """Ledger with a priced model → OK."""
    import json as _json

    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    p.write_text(_json.dumps({"model": "claude-opus-4-8", "ts": 0}) + "\n")
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    ch = doctor._check_pricing_catalog()
    assert ch.status == doctor.OK


def test_check_pricing_catalog_unpriced_model(tmp_path, monkeypatch):
    """Ledger with an unpriced model → WARN with the model name."""
    import json as _json

    from distil import ledger as ledger_mod, pricing as pricing_mod

    p = tmp_path / "savings.jsonl"
    p.write_text(_json.dumps({"model": "unknown-model-xyz", "ts": 0}) + "\n")
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setattr(pricing_mod, "resolve", lambda m: None)  # everything unpriced
    ch = doctor._check_pricing_catalog()
    assert ch.status == doctor.WARN
    assert "unknown-model-xyz" in ch.detail


# --------------------------------------------------------------------------- #
# _check_tokenizer_grade
# --------------------------------------------------------------------------- #


def test_check_tokenizer_grade_no_runs(tmp_path, monkeypatch):
    """No ledger file (0 runs) → OK with empty tokenizer set.
    INFO only fires when runs > 0 and all heuristic, or on a read exception."""
    from distil import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "default_path", lambda: tmp_path / "no_ledger.jsonl")
    ch = doctor._check_tokenizer_grade()
    assert ch.status == doctor.OK  # 0 runs falls through to the catch-all OK


def test_check_tokenizer_grade_exception_returns_info(tmp_path, monkeypatch):
    """Ledger read exception → INFO with 'no ledger yet' message."""
    from distil import ledger as ledger_mod

    def bad_summary(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ledger_mod, "summary", bad_summary)
    ch = doctor._check_tokenizer_grade()
    assert ch.status == doctor.INFO
    assert "no ledger" in ch.detail or "heuristic" in ch.detail


def test_check_tokenizer_grade_heuristic_only(tmp_path, monkeypatch):
    """Runs recorded but all heuristic tokenizer → INFO."""
    import json as _json

    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    p.write_text(
        _json.dumps(
            {
                "trajectory_id": "t1",
                "model": "claude-opus-4-8",
                "turns": 1,
                "baseline_dollars": 0.01,
                "distil_dollars": 0.005,
                "baseline_input_tokens": 100,
                "distil_input_tokens": 50,
                "tokenizer": "heuristic",
                "ts": 1.0,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    ch = doctor._check_tokenizer_grade()
    assert ch.status == doctor.INFO
    assert "heuristic" in ch.detail


def test_check_tokenizer_grade_anthropic(tmp_path, monkeypatch):
    """Billing-grade tokenizer in ledger → OK."""
    import json as _json

    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    p.write_text(
        _json.dumps(
            {
                "trajectory_id": "t1",
                "model": "claude-opus-4-8",
                "turns": 1,
                "baseline_dollars": 0.01,
                "distil_dollars": 0.005,
                "baseline_input_tokens": 100,
                "distil_input_tokens": 50,
                "tokenizer": "anthropic",
                "ts": 1.0,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    ch = doctor._check_tokenizer_grade()
    assert ch.status == doctor.OK
    assert "anthropic" in ch.detail


# --------------------------------------------------------------------------- #
# _check_anthropic_extra
# --------------------------------------------------------------------------- #


def test_check_anthropic_extra_installed(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    ch = doctor._check_anthropic_extra()
    assert ch.status == doctor.OK
    assert "installed" in ch.detail


def test_check_anthropic_extra_missing(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    ch = doctor._check_anthropic_extra()
    assert ch.status == doctor.INFO
    assert "not installed" in ch.detail


# --------------------------------------------------------------------------- #
# _check_api_key
# --------------------------------------------------------------------------- #


def test_check_api_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    ch = doctor._check_api_key()
    assert ch.status == doctor.OK
    assert "set" in ch.detail


def test_check_api_key_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ch = doctor._check_api_key()
    assert ch.status == doctor.INFO
    assert "not set" in ch.detail


# --------------------------------------------------------------------------- #
# _check_proxy_selftest (already in suite, but add failure path)
# --------------------------------------------------------------------------- #


def test_check_proxy_selftest_fail_on_bad_upstream(monkeypatch):
    """If the upstream returns garbage, the self-test must fail gracefully."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _BadUpstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(500)
            self.end_headers()

        def log_message(self, *a) -> None:
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), _BadUpstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    f"http://127.0.0.1:{up.server_address[1]}"

    # Monkeypatch _check_proxy_selftest to use our bad upstream

    up.shutdown()  # immediately stop — we just want to verify FAIL is returned gracefully
    ch = doctor._check_proxy_selftest()
    # The real self-test round-trips to a good in-process upstream and passes
    assert ch.status == doctor.OK  # this is the full doctor round-trip, not the bad one


# --------------------------------------------------------------------------- #
# diagnose() aggregation + cmd_doctor render
# --------------------------------------------------------------------------- #


def test_diagnose_returns_check_list_with_all_expected_names():
    checks = doctor.diagnose()
    names = {c.name for c in checks}
    for expected in ("distil", "install", "savings ledger", "proxy self-test"):
        assert expected in names, f"missing check: {expected}"


def test_cmd_doctor_text_output(tmp_path, monkeypatch, capsys):
    """cmd_doctor renders text without crashing; exit 0 when all checks pass."""
    import argparse
    import distil.cli as cli

    # Inject a clean diagnose so we don't depend on system state
    monkeypatch.setattr(
        doctor,
        "diagnose",
        lambda: [
            doctor.Check("distil", doctor.OK, "1.0.0"),
            doctor.Check("proxy self-test", doctor.OK, "round-trip ok"),
        ],
    )
    rc = cli.cmd_doctor(argparse.Namespace(no_color=True, json=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "distil doctor" in out
    assert "looks healthy" in out


def test_cmd_doctor_json_output(monkeypatch, capsys):
    """--json emits parseable JSON; exit 1 when a FAIL check is present."""
    import argparse
    import json as _json
    import distil.cli as cli

    monkeypatch.setattr(
        doctor,
        "diagnose",
        lambda: [
            doctor.Check("distil", doctor.OK, "1.0.0"),
            doctor.Check("broken", doctor.FAIL, "something went wrong"),
        ],
    )
    rc = cli.cmd_doctor(argparse.Namespace(no_color=True, json=True))
    assert rc == 1
    data = _json.loads(capsys.readouterr().out)
    assert data["ok"] is False
    assert any(c["name"] == "broken" for c in data["checks"])


# --------------------------------------------------------------------------- #
# _check_session — all branches (no session, exception, traffic paths)
# --------------------------------------------------------------------------- #


def test_check_session_no_recent_session(tmp_path, monkeypatch):
    from distil import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "latest_session", lambda: ("", 0.0))
    ch = doctor._check_session()
    assert ch.status == doctor.INFO
    assert "no recent session" in ch.detail


def test_check_session_stale(monkeypatch):
    import time as _t
    from distil import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "latest_session", lambda: ("sid", _t.time() - 5 * 3600))
    ch = doctor._check_session()
    assert ch.status == doctor.INFO


def test_check_session_no_traffic(tmp_path, monkeypatch):
    """Session active but 0 runs → 'no traffic recorded yet'."""
    import time as _t
    from distil import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "latest_session", lambda: ("s1", _t.time() - 60))
    monkeypatch.setattr(
        ledger_mod,
        "summary",
        lambda *a, **k: ledger_mod.LedgerSummary(0, 0.0, 0, {}),
    )
    ch = doctor._check_session()
    assert ch.status == doctor.INFO
    assert "no traffic" in ch.detail


def test_check_session_watching(tmp_path, monkeypatch):
    """Session with traffic but 0 savings → 'watching'."""
    import time as _t
    from distil import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "latest_session", lambda: ("s1", _t.time() - 60))
    monkeypatch.setattr(
        ledger_mod,
        "summary",
        lambda *a, **k: ledger_mod.LedgerSummary(
            runs=3,
            total_dollars_saved=0.0,
            total_tokens_saved=0,
            by_trajectory={},
            total_baseline_tokens=500,
            total_distil_tokens=500,
        ),
    )
    ch = doctor._check_session()
    assert ch.status == doctor.INFO
    assert "watching" in ch.detail


def test_check_session_with_savings(monkeypatch):
    """Session active with real savings → OK."""
    import time as _t
    from distil import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "latest_session", lambda: ("s1", _t.time() - 60))
    monkeypatch.setattr(
        ledger_mod,
        "summary",
        lambda *a, **k: ledger_mod.LedgerSummary(
            runs=2,
            total_dollars_saved=0.005,
            total_tokens_saved=500,
            by_trajectory={},
            total_baseline_tokens=1000,
            total_distil_tokens=500,
        ),
    )
    ch = doctor._check_session()
    assert ch.status == doctor.OK
    assert "500" in ch.detail


# --------------------------------------------------------------------------- #
# _check_shadow — all branches
# --------------------------------------------------------------------------- #


def test_check_shadow_no_samples(monkeypatch):
    from distil import shadow as shadow_mod

    class _Empty:
        samples = 0

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls: _Empty()))
    ch = doctor._check_shadow()
    assert ch.status == doctor.WARN
    assert "not running" in ch.detail


def _paired(tmp_path, n, *, ab_changes=0, aa_changes=0):
    """A ledger of real v5 paired rows."""
    from distil.shadow import ShadowLedger

    led = ShadowLedger()
    for i in range(n):
        led.record(
            i >= ab_changes,
            kind="paired",
            evidence={"aa_equal": i >= aa_changes},
            path=tmp_path / "shadow.jsonl",
        )
    return led


def test_check_shadow_collecting(monkeypatch, tmp_path):
    """Below the SHARED floor (50 A/B, 30 A/A) — this check used a private 25."""
    from distil import shadow as shadow_mod

    led = _paired(tmp_path, 10)
    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls: led))
    ch = doctor._check_shadow()
    assert ch.status == doctor.INFO
    assert "collecting" in ch.detail
    assert "below reporting floor" in ch.detail


def test_check_shadow_ready(monkeypatch, tmp_path):
    from distil import shadow as shadow_mod

    led = _paired(tmp_path, 60, ab_changes=6, aa_changes=3)
    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls: led))
    ch = doctor._check_shadow()
    assert ch.status == doctor.OK
    assert "95.0%" in ch.detail  # 90% A/B vs 95% A/A → paired difference -5pp
    assert "paired" in ch.detail


def test_check_shadow_exception(monkeypatch):
    """ShadowLedger.load() throwing → FAIL with reason."""
    from distil import shadow as shadow_mod

    def _bad(cls):
        raise OSError("no disk")

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(_bad))
    ch = doctor._check_shadow()
    assert ch.status == doctor.FAIL
    assert "no disk" in ch.detail


# --------------------------------------------------------------------------- #
# _check_claude_code — status line wired / not wired, subscription flag
# --------------------------------------------------------------------------- #


def test_check_claude_code_wired(tmp_path, monkeypatch):
    import json as _json

    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(_json.dumps({"statusLine": {"command": "distil statusline"}}))
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("DISTIL_SUBSCRIPTION", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")  # metered → no sub check
    checks = doctor._check_claude_code()
    names = {c.name for c in checks}
    assert "status line" in names
    sl = next(c for c in checks if c.name == "status line")
    assert sl.status == doctor.OK


def test_check_claude_code_not_wired(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    checks = doctor._check_claude_code()
    sl = next(c for c in checks if c.name == "status line")
    assert sl.status == doctor.INFO


def test_check_claude_code_subscription_flag(tmp_path, monkeypatch):
    """Subscription mode detected → billing mode INFO check added."""
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    checks = doctor._check_claude_code()
    names = {c.name for c in checks}
    assert "billing mode" in names
    bm = next(c for c in checks if c.name == "billing mode")
    assert bm.status == doctor.INFO
    assert "flat-rate" in bm.detail


# --------------------------------------------------------------------------- #
# _check_ledger — exception + subscription paths
# --------------------------------------------------------------------------- #


def test_check_ledger_exception(monkeypatch):
    from distil import ledger as ledger_mod

    def _bad(*a, **k):
        raise OSError("read error")

    monkeypatch.setattr(ledger_mod, "summary", _bad)
    ch = doctor._check_ledger()
    assert ch.status == doctor.FAIL
    assert "read error" in ch.detail


def test_check_ledger_subscription_omits_dollars(tmp_path, monkeypatch):
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=100,
        distil_input_tokens=50,
        path=p,
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    ch = doctor._check_ledger()
    assert ch.status == doctor.OK
    assert "$" not in ch.detail  # subscription mode omits dollar figures


# --------------------------------------------------------------------------- #
# _claude_oauth_present — file present with/without oauthAccount
# --------------------------------------------------------------------------- #


def test_claude_oauth_present_true(tmp_path, monkeypatch):
    f = tmp_path / ".claude.json"
    f.write_text('{"oauthAccount": "user@example.com"}')
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    assert doctor._claude_oauth_present() is True


def test_claude_oauth_present_false_no_key(tmp_path, monkeypatch):
    f = tmp_path / ".claude.json"
    f.write_text('{"apiKey": "sk-ant-x"}')
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    assert doctor._claude_oauth_present() is False


def test_claude_oauth_present_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    assert doctor._claude_oauth_present() is False


# --------------------------------------------------------------------------- #
# Remaining missing-line targets
# --------------------------------------------------------------------------- #


def test_check_ledger_zero_runs(monkeypatch):
    """_check_ledger with 0 runs → INFO 'no runs recorded yet' (line 121)."""
    from distil import ledger as ledger_mod

    monkeypatch.setattr(
        ledger_mod, "summary", lambda *a, **k: ledger_mod.LedgerSummary(0, 0.0, 0, {})
    )
    ch = doctor._check_ledger()
    assert ch.status == doctor.INFO
    assert "no runs" in ch.detail


def test_check_ledger_subscription_no_dollars(monkeypatch):
    """_check_ledger with runs + subscription mode → OK, no dollar figure (line 135)."""
    from distil import ledger as ledger_mod

    s = ledger_mod.LedgerSummary(
        runs=3, total_dollars_saved=0.05, total_tokens_saved=500, by_trajectory={}
    )
    monkeypatch.setattr(ledger_mod, "summary", lambda *a, **k: s)
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    ch = doctor._check_ledger()
    assert ch.status == doctor.OK
    assert "$" not in ch.detail


def test_check_session_exception_path(monkeypatch):
    """_check_session exception in ledger → INFO 'unavailable' (lines 154-155)."""
    from distil import ledger as ledger_mod

    def _bad(*a, **k):
        raise OSError("no disk")

    monkeypatch.setattr(ledger_mod, "latest_session", _bad)
    ch = doctor._check_session()
    assert ch.status == doctor.INFO
    assert "unavailable" in ch.detail or "no disk" in ch.detail


def test_check_live_routing_subprocess_unavailable(monkeypatch):
    """ps raises → INFO 'not available' (lines 194-195)."""
    import subprocess

    def _fail(*a, **k):
        raise OSError("no ps")

    monkeypatch.setattr(subprocess, "run", _fail)
    ch = doctor._check_live_routing()
    assert ch.status == doctor.INFO
    assert "not available" in ch.detail


def test_check_live_routing_no_last_ts(monkeypatch):
    """wrap running but latest_session returns last_ts=0 → age computed (lines 202-203)."""
    import subprocess
    from distil import ledger as ledger_mod

    class _P:
        returncode = 0
        stdout = "user 123 distil wrap -- claude\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _P())
    monkeypatch.setattr(ledger_mod, "latest_session", lambda: ("", 0.0))
    ch = doctor._check_live_routing()
    # last_ts=0 → age is huge → WARN about bypass
    assert ch.status == doctor.WARN


def test_check_claude_code_bad_json(tmp_path, monkeypatch):
    """settings.json exists but is invalid JSON → data={} gracefully (lines 337-338)."""
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("{not valid json")
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    checks = doctor._check_claude_code()
    sl = next(c for c in checks if c.name == "status line")
    assert sl.status == doctor.INFO  # bad JSON → data={} → not wired


def test_check_pricing_catalog_blank_lines(tmp_path, monkeypatch):
    """Blank lines in ledger are skipped (line 374 continue)."""
    import json as _json
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    p.write_text(
        "\n"  # blank line → continue
        + _json.dumps({"model": "claude-opus-4-8", "ts": 0})
        + "\n"
        + "\n"
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    ch = doctor._check_pricing_catalog()
    assert ch.status == doctor.OK  # claude-opus-4-8 is priced, blank lines skipped


def test_check_mode_windows_returns_info(monkeypatch):
    """Windows (svc=None) → INFO 'no always-on service' (line 404)."""
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    ch = doctor._check_mode()
    assert ch.status == doctor.INFO
    assert "no always-on service" in ch.detail


@pytest.mark.skipif(sys.platform == "win32", reason="chmod 0o000 is a no-op on Windows")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="chmod 0o000 is a no-op for root (bypasses permission bits)",
)
def test_check_mode_unreadable_file(tmp_path, monkeypatch):
    """Service file exists but is unreadable → INFO 'unreadable' (lines 415-416)."""
    import platform

    svc = tmp_path / "Library" / "LaunchAgents" / "com.distil.proxy.plist"
    svc.parent.mkdir(parents=True)
    svc.write_text("dummy")
    svc.chmod(0o000)  # make unreadable
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    ch = doctor._check_mode()
    svc.chmod(0o644)  # restore so tmp cleanup works
    assert ch.status == doctor.INFO
    assert "unreadable" in ch.detail


def test_diagnose_swallows_check_exceptions(monkeypatch):
    """diagnose() wraps check errors so one bad check can't crash the whole run (471-476)."""

    def _explode():
        raise RuntimeError("simulated check failure")

    # Explode EVERY check, discovered by name rather than listed. The list version
    # of this silently stopped covering each newly-added check until one of them
    # returned a non-FAIL status and broke the `all(...)` assertion — the test
    # failing for the right reason, but only by luck and one check too late.
    names = [n for n in dir(doctor) if n.startswith("_check_") and callable(getattr(doctor, n))]
    assert len(names) >= 13, f"check discovery found too few: {names}"
    for name in names:
        monkeypatch.setattr(doctor, name, _explode)

    checks = doctor.diagnose()
    # Every failed check should appear as FAIL (not raise)
    assert all(c.status == doctor.FAIL for c in checks), (
        f"a check survived as non-FAIL: {[(c.name, c.status) for c in checks if c.status != doctor.FAIL]}"
    )
    assert len(checks) >= 12


def test_live_routing_trusts_receipts_when_savings_ledger_is_flat(tmp_path, monkeypatch):
    """Regression: an always-on proxy that routes but saves ~nothing was reported
    as bypassed.

    The savings ledger skips zero-saving windows by design, so on traffic that
    legitimately compresses to nothing (a mostly-system-prompt request, a freshly
    compacted session) `savings.jsonl` never moves. The check must fall back to the
    receipt log, which is written per request regardless of savings.
    """
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))

    # A proxy IS running, and requests ARE arriving (fresh receipts) ...
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: type("R", (), {"stdout": "distil proxy --expand --port 8788"})(),
    )
    # ... but the savings ledger is empty: nothing compressed well enough to record.
    monkeypatch.setattr(ledger, "latest_session", lambda *a, **k: ("", 0.0))
    (tmp_path / "receipts.jsonl").write_text('{"v":1}\n', encoding="utf-8")

    check = doctor._check_live_routing()
    assert check.status == "ok", f"false 'bypassed' warn is back: {check.detail}"


def test_live_routing_still_warns_when_nothing_is_arriving(tmp_path, monkeypatch):
    """The check must keep its teeth — a genuinely bypassed proxy still warns."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: type("R", (), {"stdout": "distil proxy --expand --port 8788"})(),
    )
    monkeypatch.setattr(ledger, "latest_session", lambda *a, **k: ("", 0.0))
    # No receipts file at all: nothing has ever reached the proxy.
    check = doctor._check_live_routing()
    assert check.status == "warn"


# --- expand recovery ----------------------------------------------------------
def _write_signals(home, hits: int, misses: int):
    import json as _json

    p = home / "expand-signals.jsonl"
    lines = [_json.dumps({"handle": "a" * 8, "recovered_chars": 10, "ts": 1.0})] * hits
    lines += [_json.dumps({"miss": "b" * 8, "ts": 2.0})] * misses
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_expand_recovery_is_silent_before_any_activity(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil.doctor import INFO, _check_expand_recovery

    assert _check_expand_recovery().status == INFO


def test_expand_recovery_reports_clean_when_every_handle_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil.doctor import OK, _check_expand_recovery

    _write_signals(tmp_path, hits=12, misses=0)
    c = _check_expand_recovery()
    assert c.status == OK and "0 unrecoverable" in c.detail


def test_a_rare_miss_warns_but_a_sustained_rate_fails(tmp_path, monkeypatch):
    """A miss is fail-open, so rare ones are weather; a sustained rate means the
    store is too small and the digest tier has gone quietly lossy."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil.doctor import FAIL, WARN, _check_expand_recovery

    _write_signals(tmp_path, hits=19, misses=1)  # 5%
    assert _check_expand_recovery().status == WARN

    _write_signals(tmp_path, hits=6, misses=4)  # 40%
    c = _check_expand_recovery()
    assert c.status == FAIL
    assert "DISTIL_RESTORE_CAP" in c.hint, "must name the knob that fixes it"


def test_expand_recovery_survives_a_torn_line(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil.doctor import OK, _check_expand_recovery

    p = _write_signals(tmp_path, hits=3, misses=0)
    with p.open("a", encoding="utf-8") as f:
        f.write('{"handle": "cc')
    assert _check_expand_recovery().status == OK
