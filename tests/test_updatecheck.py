"""Update-notice guarantees: throttled, opt-out, fail-open, notice only when behind."""

from __future__ import annotations

import time

import pytest

from distil import updatecheck


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_NO_UPDATE_CHECK", raising=False)


def _run_sync(monkeypatch):
    """Make maybe_notify synchronous so tests can assert on its output."""

    class _T:
        def __init__(self, target, **kw):
            self._t = target

        def start(self):
            self._t()

    monkeypatch.setattr(updatecheck.threading, "Thread", _T)


def test_notice_when_outdated(monkeypatch, capsys):
    _run_sync(monkeypatch)
    monkeypatch.setattr("distil.onboard.latest_pypi_version", lambda: "999.0.0")
    assert updatecheck.maybe_notify() is True
    assert "999.0.0" in capsys.readouterr().err


def test_silent_when_current(monkeypatch, capsys):
    _run_sync(monkeypatch)
    from distil import __version__

    monkeypatch.setattr("distil.onboard.latest_pypi_version", lambda: __version__)
    assert updatecheck.maybe_notify() is True
    assert capsys.readouterr().err == ""


def test_throttled_to_daily(monkeypatch):
    _run_sync(monkeypatch)
    monkeypatch.setattr("distil.onboard.latest_pypi_version", lambda: "999.0.0")
    assert updatecheck.maybe_notify() is True
    assert updatecheck.maybe_notify() is False  # inside 24h → no second check


def test_opt_out(monkeypatch):
    monkeypatch.setenv("DISTIL_NO_UPDATE_CHECK", "1")
    assert updatecheck.maybe_notify() is False


def test_pypi_failure_is_swallowed(monkeypatch, capsys):
    _run_sync(monkeypatch)

    def boom():
        raise OSError("pypi down")

    monkeypatch.setattr("distil.onboard.latest_pypi_version", boom)
    assert updatecheck.maybe_notify() is True  # started, failed silently
    assert capsys.readouterr().err == ""


def test_stale_marker_reenables(monkeypatch):
    _run_sync(monkeypatch)
    monkeypatch.setattr("distil.onboard.latest_pypi_version", lambda: "999.0.0")
    updatecheck.maybe_notify()
    updatecheck._last_path().write_text(str(time.time() - 100_000), encoding="utf-8")
    assert updatecheck.maybe_notify() is True
