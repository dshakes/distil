"""The preflight that stands between a broken proxy and a machine-wide outage.

Written after `distil default --always-on` wired ANTHROPIC_BASE_URL into
~/.claude/settings.json, reported success from `launchctl load`'s exit code, and left
every Claude Code session on the machine failing. The proxy was up the whole time. It
was pointed at an upstream that does not serve /v1/messages, so it answered 404 to
everything — and because Claude Code skips model-name validation whenever a base URL is
set, the failure surfaced as "there's an issue with the selected model".

The load-bearing test is `test_listening_but_404_is_not_routing`: a socket you can
connect to was the old bar, and it is exactly the signal that was already true while
everything was broken.
"""

from __future__ import annotations

import argparse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from distil import cli
from distil.setup import probe_routing


def _serve(status: int, headers: dict[str, str] | None = None):
    """A loopback server answering every POST with *status*. Returns (port, shutdown)."""

    class H(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's contract
            self.send_response(status)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a: object) -> None:  # keep pytest output readable
            return

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1], srv.shutdown


def test_listening_but_404_is_not_routing() -> None:
    """The exact production failure: the port answers, and answers 404 to everything."""
    port, stop = _serve(404)
    try:
        ok, detail = probe_routing("127.0.0.1", port, deadline=2.0)
    finally:
        stop()
    assert ok is False
    assert "404" in detail and "not routing" in detail


def test_401_proves_routing_without_credentials() -> None:
    """We send no API key on purpose — a 401 means a real messages handler saw it."""
    port, stop = _serve(401)
    try:
        ok, detail = probe_routing("127.0.0.1", port, deadline=2.0)
    finally:
        stop()
    assert ok is True
    assert "401" in detail


def test_distil_header_upgrades_the_verdict() -> None:
    """`a proxy` answers; `distil` answers *and* says so."""
    port, stop = _serve(401, {"x-distil-mode": "digest"})
    try:
        _, with_hdr = probe_routing("127.0.0.1", port, deadline=2.0)
    finally:
        stop()
    port2, stop2 = _serve(401)
    try:
        _, without = probe_routing("127.0.0.1", port2, deadline=2.0)
    finally:
        stop2()
    assert with_hdr.startswith("distil on")
    assert without.startswith("a proxy on")


def test_nothing_listening_fails_within_the_deadline() -> None:
    port, stop = _serve(401)
    stop()  # free the port, then probe it — nothing is there
    ok, detail = probe_routing("127.0.0.1", port, deadline=1.0)
    assert ok is False
    assert "no response" in detail


def test_404_does_not_burn_the_deadline() -> None:
    """A 404 is an answer, not a race — retrying it would stall setup for 10s."""
    import time

    port, stop = _serve(404)
    try:
        t0 = time.monotonic()
        probe_routing("127.0.0.1", port, deadline=30.0)
        assert time.monotonic() - t0 < 5.0
    finally:
        stop()


# --------------------------------------------------------------------------- #
# The wiring itself: a failed preflight must leave the machine untouched.
# --------------------------------------------------------------------------- #


def _always_on_ns(rc_file, port: int) -> argparse.Namespace:
    return argparse.Namespace(
        undo=False,
        always_on=True,
        rc=str(rc_file),
        port=port,
        agent="claude",
        mode="expand",
        no_start=True,
        force=False,
    )


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """cmd_default with every real side effect redirected into tmp_path."""
    import distil.setup as setup_mod
    from distil import onboard

    monkeypatch.setattr(
        onboard,
        "detect",
        lambda: onboard.Env(
            os_name="Darwin",
            agents=[("claude", "Claude Code")],
            installed_version="1.0.0",
            method="pipx",
            managers=["pipx"],
        ),
    )
    rc_file = tmp_path / ".zshrc"
    rc_file.write_text("")
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: settings)
    monkeypatch.setattr(
        setup_mod, "service_spec", lambda *a, **k: (tmp_path / "svc.plist", "content", None)
    )
    return rc_file, settings


def test_failed_preflight_writes_no_config(wired, capsys) -> None:
    """The whole point: a proxy that does not route must not become the default."""
    rc_file, settings = wired
    port, stop = _serve(404)
    try:
        rc = cli.cmd_default(_always_on_ns(rc_file, port))
    finally:
        stop()
    out = capsys.readouterr().out
    assert rc == 1
    assert "preflight failed" in out
    assert "Nothing was wired" in out
    assert not settings.exists(), "settings.json was written despite a failed preflight"
    assert "ANTHROPIC_BASE_URL" not in rc_file.read_text()


def test_passing_preflight_writes_config(wired, capsys) -> None:
    """And the happy path still wires — a gate that never opens is not a gate."""
    rc_file, settings = wired
    port, stop = _serve(401, {"x-distil-mode": "digest"})
    try:
        rc = cli.cmd_default(_always_on_ns(rc_file, port))
    finally:
        stop()
    out = capsys.readouterr().out
    assert rc == 0
    assert "preflight: distil on" in out
    assert "ANTHROPIC_BASE_URL" in settings.read_text()


# --------------------------------------------------------------------------- #
# doctor: the same tightening, so it names the real fault instead of "listening"
# --------------------------------------------------------------------------- #


def test_doctor_fails_a_listening_but_unrouted_base_url() -> None:
    from distil.doctor import FAIL, OK, _check_base_url

    port, stop = _serve(404)
    try:
        chk = _check_base_url({"env": {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}})
    finally:
        stop()
    assert chk is not None
    assert chk.status == FAIL
    assert chk.status != OK, "a listening socket used to be enough — it never was"
    assert "selected model" in (chk.hint or ""), "the hint must name the misleading symptom"


def test_doctor_passes_a_routing_base_url() -> None:
    from distil.doctor import OK, _check_base_url

    port, stop = _serve(401, {"x-distil-mode": "digest"})
    try:
        chk = _check_base_url({"env": {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}})
    finally:
        stop()
    assert chk is not None
    assert chk.status == OK
    assert "routes /v1/messages" in chk.detail


# ── The second failure: an uninstall that reported success and left the machine broken ──
#
# The preflight above stops distil writing a base URL that does not work. It does nothing
# about one already written. `distil default --undo` was supposed to be that escape hatch
# and could not reach the entry that mattered, for two independent reasons:
#
#   1. it looked only in ~/.claude/settings.json, and Claude Code also merges *project*
#      settings — which take PRECEDENCE over the home file; and
#   2. it matched the value against `--port` from the current invocation, so an entry
#      wired on another port was classified "foreign" and deliberately left in place.
#
# Both were true at once for one real machine: `.claude/settings.local.json` in a repo
# held 127.0.0.1:8788, nothing listened there, and every session started in that
# directory died with ConnectionRefused — after distil had been uninstalled entirely.


def test_undo_removes_a_base_url_wired_on_a_different_port(tmp_path, monkeypatch) -> None:
    """The load-bearing one. Undo must clean up by *shape*, not by this run's --port."""
    from distil import setup

    settings = tmp_path / "settings.json"
    settings.write_text('{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788"}}\n')
    monkeypatch.setattr(setup, "claude_settings_files", lambda cwd=None: [settings])
    monkeypatch.setattr(setup, "detect_shell", lambda: ("zsh", tmp_path / ".zshrc"))

    # --port defaults to 8788 in real use; pass a DIFFERENT one, which is what happens
    # whenever someone types plain `distil default --undo` after wiring a custom port.
    rc = cli.cmd_default(
        argparse.Namespace(
            undo=True,
            always_on=False,
            rc=str(tmp_path / ".zshrc"),
            port=9999,
            agent="claude",
            mode="lossless-only",
            no_start=True,
            force=False,
        )
    )
    assert rc == 0
    assert "ANTHROPIC_BASE_URL" not in settings.read_text(), (
        "undo left the entry that was breaking every session — the exact uninstall bug"
    )


def test_unwire_base_url_spares_a_real_gateway(tmp_path) -> None:
    """Cleaning by shape must not mean cleaning by force: a non-loopback base URL is
    somebody's actual gateway (LiteLLM, Bedrock proxy, a corporate egress) and distil
    never wrote it. Deleting it would trade our outage for theirs."""
    from distil.setup import unwire_base_url

    settings = tmp_path / "settings.json"
    settings.write_text('{"env": {"ANTHROPIC_BASE_URL": "https://gateway.corp.example"}}\n')
    status, msg = unwire_base_url(settings)
    assert status == "foreign"
    assert "gateway.corp.example" in settings.read_text()
    assert "left as-is" in msg


def test_unwire_base_url_preserves_neighbouring_settings(tmp_path) -> None:
    from distil.setup import unwire_base_url

    settings = tmp_path / "settings.json"
    settings.write_text(
        '{"permissions": {"allow": ["Bash(ls)"]}, '
        '"env": {"ANTHROPIC_BASE_URL": "http://localhost:8788", "FOO": "bar"}}\n'
    )
    assert unwire_base_url(settings)[0] == "ok"
    data = __import__("json").loads(settings.read_text())
    assert data["env"] == {"FOO": "bar"}, "only distil's key comes out"
    assert data["permissions"]["allow"] == ["Bash(ls)"]
    assert (tmp_path / "settings.json.bak").exists(), "an undo that edits must be reversible"


def test_project_settings_outrank_home_settings(tmp_path, real_claude_settings_files) -> None:
    """Precedence is the reason the outage was invisible: doctor read the home file and
    reported clean while a project file overrode it."""
    home = tmp_path / "home"
    proj = home / "work" / "repo"
    proj.mkdir(parents=True)
    (home / ".claude").mkdir()

    order = real_claude_settings_files(cwd=proj)
    idx = {p: i for i, p in enumerate(order)}
    assert idx[proj / ".claude" / "settings.local.json"] < idx[proj / ".claude" / "settings.json"]
    assert idx[proj / ".claude" / "settings.json"] < idx[home / ".claude" / "settings.json"]
    assert order[0].name == "managed-settings.json", "managed policy outranks everything"


def test_doctor_reports_a_shadowed_base_url_as_shadowed(tmp_path, monkeypatch) -> None:
    """Two files, both set. Only the winner is diagnosed; naming the loser as a live
    fault would send the user editing a file that changes nothing."""
    from distil import doctor, setup

    win = tmp_path / "project.json"
    lose = tmp_path / "home.json"
    win.write_text('{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}}\n')
    lose.write_text('{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:2"}}\n')
    monkeypatch.setattr(setup, "claude_settings_files", lambda cwd=None: [win, lose])

    checks = doctor._base_url_checks()
    assert len(checks) == 2
    assert checks[0].status == doctor.FAIL and "project.json" in checks[0].name
    assert checks[1].status == doctor.INFO and "shadowed" in checks[1].detail


# ── `distil offboard`, the command a leaving user actually runs ────────────────
#
# `default --undo` was only half the escape hatch. `offboard` is the documented way
# out, and it had the same two defects plus a third of its own: it prompted off
# `~/.claude/settings.json` existing, so on a machine without that file it asked
# nothing and cleaned nothing — in any file. It then reported success and printed the
# uninstall command, which is how a base URL outlived distil itself.


def _offboard_args(**over):
    base = dict(yes=True, no_interactive=True, purge=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_offboard_clears_a_project_scoped_base_url(tmp_path, monkeypatch, capsys) -> None:
    """The user's actual machine: the dead entry lived in a repo's settings.local.json,
    on a port offboard never matched, while ~/.claude/settings.json did not exist."""
    from distil import setup

    proj = tmp_path / "repo" / ".claude"
    proj.mkdir(parents=True)
    local = proj / "settings.local.json"
    local.write_text('{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788"}}\n')
    home_settings = tmp_path / "home" / ".claude" / "settings.json"  # deliberately absent

    monkeypatch.setattr(setup, "claude_settings_files", lambda cwd=None: [local, home_settings])
    monkeypatch.setattr(setup, "default_settings_path", lambda: home_settings)
    monkeypatch.setattr(setup, "detect_shell", lambda: ("zsh", tmp_path / ".zshrc"))
    monkeypatch.setattr(setup, "service_spec", lambda *a, **k: (tmp_path / "absent.plist", "", ""))

    assert cli.cmd_offboard(_offboard_args()) == 0
    assert "ANTHROPIC_BASE_URL" not in local.read_text(), (
        "offboard reported success and left the entry that was killing every session"
    )


def test_offboard_is_silent_when_there_is_nothing_to_unwire(tmp_path, monkeypatch, capsys) -> None:
    """No base URL anywhere → say so once. The old code asked the user to confirm
    removing something that was not there, then reported 'absent' after they said yes."""
    from distil import setup

    empty = tmp_path / "settings.json"
    empty.write_text('{"permissions": {"allow": []}}\n')
    monkeypatch.setattr(setup, "claude_settings_files", lambda cwd=None: [empty])
    monkeypatch.setattr(setup, "default_settings_path", lambda: empty)
    monkeypatch.setattr(setup, "detect_shell", lambda: ("zsh", tmp_path / ".zshrc"))
    monkeypatch.setattr(setup, "service_spec", lambda *a, **k: (tmp_path / "absent.plist", "", ""))

    assert cli.cmd_offboard(_offboard_args()) == 0
    out = capsys.readouterr().out
    assert "no ANTHROPIC_BASE_URL wired" in out
    assert "Unwire ANTHROPIC_BASE_URL" not in out, "never prompt about nothing"


def test_offboard_spares_a_real_gateway(tmp_path, monkeypatch) -> None:
    """--yes must not become a licence to delete somebody's actual gateway."""
    from distil import setup

    settings = tmp_path / "settings.json"
    settings.write_text('{"env": {"ANTHROPIC_BASE_URL": "https://gateway.corp.example"}}\n')
    monkeypatch.setattr(setup, "claude_settings_files", lambda cwd=None: [settings])
    monkeypatch.setattr(setup, "default_settings_path", lambda: settings)
    monkeypatch.setattr(setup, "detect_shell", lambda: ("zsh", tmp_path / ".zshrc"))
    monkeypatch.setattr(setup, "service_spec", lambda *a, **k: (tmp_path / "absent.plist", "", ""))

    assert cli.cmd_offboard(_offboard_args()) == 0
    assert "gateway.corp.example" in settings.read_text()


def test_loopback_base_url_reads_without_writing(tmp_path) -> None:
    from distil.setup import loopback_base_url

    settings = tmp_path / "settings.json"
    settings.write_text('{"env": {"ANTHROPIC_BASE_URL": "http://localhost:9999"}}\n')
    before = settings.read_text()
    assert loopback_base_url(settings) == "http://localhost:9999"
    assert settings.read_text() == before, "the probe that decides whether to prompt must not edit"
    assert loopback_base_url(tmp_path / "nope.json") is None
    settings.write_text("{ not json\n")
    assert loopback_base_url(settings) is None
