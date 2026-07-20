"""Census guarantees — the promises TELEMETRY.md makes, executable.

Every test isolates state via DISTIL_HOME=tmp_path and replaces the socket
layer with a tripwire: if urlopen is reached when the rules say it must not
be, the test fails loudly.
"""

from __future__ import annotations

import json

import pytest

from distil import census


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("DISTIL_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DISTIL_CENSUS_ENDPOINT", raising=False)


def _arm_network_tripwire(monkeypatch, calls: list):
    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data.decode()))

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    monkeypatch.setattr(census.urllib.request, "urlopen", fake_urlopen)


def test_default_is_silent(monkeypatch):
    """No consent → no network, even with an endpoint configured."""
    calls: list = []
    _arm_network_tripwire(monkeypatch, calls)
    monkeypatch.setenv("DISTIL_CENSUS_ENDPOINT", "http://127.0.0.1:1/ping")
    assert census.enabled() is False
    assert census.maybe_ping() is False
    assert calls == []


def test_do_not_track_beats_stored_consent(monkeypatch):
    calls: list = []
    _arm_network_tripwire(monkeypatch, calls)
    census.opt_in()
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert census.enabled() is False
    assert census.maybe_ping() is False
    assert calls == []


def test_distil_no_telemetry_beats_stored_consent(monkeypatch):
    calls: list = []
    _arm_network_tripwire(monkeypatch, calls)
    census.opt_in()
    monkeypatch.setenv("DISTIL_NO_TELEMETRY", "1")
    assert census.maybe_ping() is False
    assert calls == []


def test_payload_schema_frozen():
    """The census may contain EXACTLY these keys — widening it must edit this
    test and TELEMETRY.md together (that's the point)."""
    census.opt_in()
    payload = census.build_payload()
    assert set(payload) == {
        "schema",
        "install_id",
        "version",
        "os",
        "arch",
        "python",
        "runs",
        "tokens_saved",
        "dollars_saved",
        "billing",
        "by_model",
        "agents",
        "ts",
    }
    assert payload["schema"] == 2
    assert payload["billing"] in ("subscription", "metered")
    assert isinstance(payload["by_model"], dict) and len(payload["by_model"]) <= 5
    assert all(a in ("claude", "codex", "gemini", "aider", "other") for a in payload["agents"])
    # Numbers and short platform strings only — nothing that can carry content.
    for key, value in payload.items():
        assert isinstance(value, (int, float, str, dict, list)), key
        if isinstance(value, str):
            assert len(value) < 128, key
            assert "/" not in value or key == "version", key  # no paths


def test_opt_in_sends_and_throttles(monkeypatch):
    calls: list = []
    _arm_network_tripwire(monkeypatch, calls)
    census.opt_in()
    assert census.maybe_ping() is True
    assert len(calls) == 1
    assert calls[0]["schema"] == 2
    # Second call inside 24h: throttled, no second request.
    assert census.maybe_ping() is False
    assert len(calls) == 1


def test_opt_out_deletes_install_id():
    iid = census.opt_in()
    assert census.status()["install_id"] == iid
    census.opt_out()
    st = census.status()
    assert st["install_id"] is None
    assert st["consent"] == "off"
    assert census.enabled() is False


def test_send_failure_never_raises(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("endpoint down")

    monkeypatch.setattr(census.urllib.request, "urlopen", boom)
    census.opt_in()
    assert census.maybe_ping() is True  # attempted, swallowed


def test_install_id_is_stable_and_random():
    a = census.opt_in()
    b = census.install_id()
    assert a == b and len(a) == 32
    census.opt_out()
    c = census.opt_in()
    assert c != a  # re-consent mints a fresh identity


def test_subscription_reports_dollars_and_billing(monkeypatch):
    """Subscription installs are NOT excluded: dollars are reported and the
    billing field lets the rollup bucket them as notional (never real $)."""
    census.opt_in()
    monkeypatch.setattr("distil.doctor.subscription_mode", lambda: True)
    p = census.build_payload()
    assert p["billing"] == "subscription"
    assert p["dollars_saved"] >= 0.0  # present, bucketed server-side


def test_calibration_factor_applied(monkeypatch, tmp_path):
    """Census totals wear the same heuristic→billed correction as the proof ledger."""
    census.opt_in()
    from distil.ledger import LedgerSummary

    fake = LedgerSummary(
        runs=1,
        total_dollars_saved=10.0,
        total_tokens_saved=1000,
        by_trajectory={},
        total_baseline_tokens=2000,
        total_distil_tokens=1000,
        total_baseline_dollars=20.0,
        total_distil_dollars=10.0,
    )
    monkeypatch.setattr("distil.ledger.summary", lambda: fake)
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (0.8, 99))
    monkeypatch.setattr("distil.doctor.subscription_mode", lambda: False)
    p = census.build_payload()
    assert p["tokens_saved"] == 800  # 1000 * 0.8 — never more than billed
    assert p["dollars_saved"] == 8.0
