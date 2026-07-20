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
        "ts",
    }
    # Numbers and short platform strings only — nothing that can carry content.
    for key, value in payload.items():
        assert isinstance(value, (int, float, str)), key
        if isinstance(value, str):
            assert len(value) < 128, key
            assert "/" not in value or key == "version", key  # no paths


def test_opt_in_sends_and_throttles(monkeypatch):
    calls: list = []
    _arm_network_tripwire(monkeypatch, calls)
    census.opt_in()
    assert census.maybe_ping() is True
    assert len(calls) == 1
    assert calls[0]["schema"] == 1
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
