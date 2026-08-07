"""The settings.json precedence trap.

Regression suite for a total-outage class observed three times in one day: a base
URL pinned in `~/.claude/settings.json` outranks the environment `distil wrap`
exports. When that pinned port is dead, EVERY session fails with
`ConnectionRefused` — an error that names the provider, not distil.

The contract these tests pin:
  * a dead pinned URL is fatal (wrap must refuse, not start into a broken config)
  * a live pinned URL is a warning (wrap works, but its proxy sees no traffic)
  * a pin that names the wrap's own URL is not a conflict at all
  * project-scoped settings outrank user-scoped ones
"""

from __future__ import annotations

import json
import socket

import pytest

from distil.precedence import check_claude_settings, settings_candidates


@pytest.fixture
def live_port():
    """A real listening socket, so 'reachable' is measured rather than mocked.

    No accept loop is needed: a connect() succeeds into the listen backlog, which
    is exactly the signal `_reachable` is asking for ("is anything bound here").
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    try:
        yield srv.getsockname()[1]
    finally:
        srv.close()


def _dead_port() -> int:
    """A port with nothing on it — bind then immediately release."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _write_settings(path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": url}}), encoding="utf-8")


def test_dead_pinned_url_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_IGNORE_SETTINGS_PRECEDENCE", raising=False)
    url = f"http://127.0.0.1:{_dead_port()}"
    _write_settings(tmp_path / ".claude" / "settings.json", url)

    c = check_claude_settings("ANTHROPIC_BASE_URL", cwd=tmp_path)
    assert c is not None
    assert c.fatal, "a pin at a dead port is a guaranteed outage, not a warning"
    assert "NOT accepting connections" in c.message()
    assert "distil offboard" in c.message(), "must name the fix, not just the fault"


def test_live_pinned_url_warns_but_is_not_fatal(tmp_path, monkeypatch, live_port):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_settings(tmp_path / ".claude" / "settings.json", f"http://127.0.0.1:{live_port}")

    c = check_claude_settings("ANTHROPIC_BASE_URL", cwd=tmp_path)
    assert c is not None and not c.fatal
    assert "no traffic" in c.message()


def test_pin_naming_our_own_wrap_url_is_not_a_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    url = "http://127.0.0.1:61956"
    _write_settings(tmp_path / ".claude" / "settings.json", url)
    assert check_claude_settings("ANTHROPIC_BASE_URL", url, cwd=tmp_path) is None
    # trailing-slash difference must not read as a different endpoint
    assert check_claude_settings("ANTHROPIC_BASE_URL", url + "/", cwd=tmp_path) is None


def test_no_settings_file_means_no_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert check_claude_settings("ANTHROPIC_BASE_URL", cwd=tmp_path) is None


def test_project_settings_outrank_user_settings(tmp_path, monkeypatch):
    """The effective value is the highest-precedence one, and that is what we report."""
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "proj"
    _write_settings(tmp_path / ".claude" / "settings.json", "http://127.0.0.1:8788")
    _write_settings(project / ".claude" / "settings.local.json", "http://127.0.0.1:9999")

    c = check_claude_settings("ANTHROPIC_BASE_URL", cwd=project)
    assert c is not None
    assert c.value == "http://127.0.0.1:9999", "project-local must win"
    assert c.path.name == "settings.local.json"


def test_escape_hatch_disables_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_IGNORE_SETTINGS_PRECEDENCE", "1")
    _write_settings(tmp_path / ".claude" / "settings.json", f"http://127.0.0.1:{_dead_port()}")
    assert check_claude_settings("ANTHROPIC_BASE_URL", cwd=tmp_path) is None


def test_remote_urls_are_never_probed(tmp_path, monkeypatch):
    """Probing an arbitrary remote at every wrap start is latency and looks like scanning."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_settings(tmp_path / ".claude" / "settings.json", "https://gateway.corp.example:9443")
    c = check_claude_settings("ANTHROPIC_BASE_URL", cwd=tmp_path)
    assert c is not None and not c.fatal, "non-loopback is assumed reachable"


def test_malformed_settings_never_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = tmp_path / ".claude" / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    for junk in ("{not json", "[]", '{"env": "a string"}', '{"env": {"OTHER": "x"}}'):
        p.write_text(junk, encoding="utf-8")
        assert check_claude_settings("ANTHROPIC_BASE_URL", cwd=tmp_path) is None


def test_candidate_order_is_ascending_precedence(tmp_path):
    paths = settings_candidates(tmp_path)
    assert paths[-1].name == "settings.local.json"
    assert ".claude" in str(paths[-1])
