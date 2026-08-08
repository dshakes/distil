"""Socket activation: the listener must outlive the worker.

These tests exist because of a field outage, not a hypothesis. With
``ANTHROPIC_BASE_URL`` pinned in ``~/.claude/settings.json``, every API request
on the machine goes through one local process and the client has no fallback: a
proxy that is down for one second is a session that dies with
``ConnectionRefused``. ``KeepAlive`` restarts the process but cannot rescue the
requests that arrive in the gap.

The fix is that the supervisor owns the listening socket. What must hold:

1. Unsupervised, we still bind for ourselves — the fix must never make ``distil
   proxy`` in a terminal fail to start.
2. When a socket IS handed down, we serve on it and do NOT bind a second one
   (binding would raise EADDRINUSE against the supervisor's own socket and turn
   a fault-tolerance feature into a crash loop — the exact failure that made
   ``launchctl bootstrap`` report "5: Input/output error").
3. The generated plist actually asks launchd for the socket.
"""

from __future__ import annotations

import os
import plistlib
import socket
import threading
import urllib.request

import pytest

from distil import activation
from distil import setup
from distil.proxy import _listen
from distil.setup import service_spec


def test_unsupervised_start_is_unaffected(monkeypatch):
    """No supervisor -> None, so the caller binds as it always has."""
    for var in ("LISTEN_FDS", "LISTEN_PID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(activation, "_from_launchd", lambda: None)
    assert activation.inherited_listener() is None


def test_a_supervisor_we_cannot_talk_to_does_not_prevent_startup(monkeypatch):
    """A broken activation path must degrade to self-binding, never to exiting.

    Failing closed here would convert "your supervisor is unusual" into "distil
    does not start", which is a worse outage than the one we are fixing.
    """
    monkeypatch.delenv("LISTEN_FDS", raising=False)

    def boom():
        raise OSError("no launchd here")

    monkeypatch.setattr(activation, "_from_launchd", boom)
    assert activation.inherited_listener() is None


def test_systemd_hands_down_its_listener(monkeypatch):
    """LISTEN_FDS protocol: fd 3 is the listening socket."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    dup = os.dup(listener.fileno())
    try:
        os.dup2(dup, activation._SD_LISTEN_FDS_START)
        monkeypatch.setenv("LISTEN_FDS", "1")
        monkeypatch.setenv("LISTEN_PID", str(os.getpid()))
        got = activation._from_systemd()
        assert got is not None
        assert got.getsockname() == listener.getsockname()
        got.detach()
    finally:
        os.close(dup)
        listener.close()


def test_systemd_vars_inherited_by_the_wrong_process_are_ignored(monkeypatch):
    """LISTEN_PID guards against a child inheriting the variables across exec."""
    monkeypatch.setenv("LISTEN_FDS", "1")
    monkeypatch.setenv("LISTEN_PID", str(os.getpid() + 1))
    assert activation._from_systemd() is None


def test_activated_proxy_serves_without_binding_a_second_socket(monkeypatch):
    """The whole point: serve on the handed-down socket, bind nothing.

    If `_listen` bound its own socket here it would raise EADDRINUSE, because
    the port is already held by the "supervisor" socket below.
    """
    supervisor = socket.socket()
    supervisor.bind(("127.0.0.1", 0))
    supervisor.listen(16)
    host, port = supervisor.getsockname()

    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib signature
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"served")

        def log_message(self, *a):  # noqa: ANN002 - silence test output
            pass

    monkeypatch.setattr(activation, "inherited_listener", lambda: supervisor)
    server, activated = _listen(host, port, H)
    assert activated is True
    assert server.socket.fileno() == supervisor.fileno()
    assert server.server_port == port

    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=5) as r:
            assert r.read() == b"served"
    finally:
        server.shutdown()
        server.server_close()


def test_unactivated_proxy_still_binds_its_own_port(monkeypatch):
    """The ordinary path must keep working exactly as before."""
    monkeypatch.setattr(activation, "inherited_listener", lambda: None)

    from http.server import BaseHTTPRequestHandler

    server, activated = _listen("127.0.0.1", 0, BaseHTTPRequestHandler)
    try:
        assert activated is False
        assert server.socket.getsockname()[1] > 0
    finally:
        server.server_close()


def test_launchd_probe_is_safe_to_call_unsupervised():
    """Exercises the real ctypes path — not a mock of it.

    ``launch_activate_socket`` returns non-zero when the caller is not a launchd
    job (ESRV_NOT_MANAGED) or its plist has no ``Sockets`` key (ESRV_NO_SOCKET).
    Both are the ordinary case for a developer running pytest, and both must come
    back as "no socket handed down" rather than an exception.
    """
    if os.uname().sysname != "Darwin":
        assert activation._from_launchd() is None
        return
    assert activation._from_launchd() is None  # pytest is not a launchd job


def test_missing_libsystem_is_not_fatal(monkeypatch):
    monkeypatch.setattr("ctypes.util.find_library", lambda name: None)
    assert activation._from_launchd() is None


def test_a_libsystem_without_the_symbol_is_not_fatal(monkeypatch):
    """Older or unusual platforms may not export launch_activate_socket."""

    class _Lib:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr("ctypes.util.find_library", lambda name: "libSystem.dylib")
    monkeypatch.setattr("ctypes.CDLL", lambda *a, **k: _Lib())
    assert activation._from_launchd() is None


def test_launchd_hands_down_its_listener(monkeypatch):
    """The success path, faked at the ctypes boundary.

    On a developer machine pytest is never a launchd job, so the only way to
    exercise the branch that actually builds the socket is to stand in for
    ``launch_activate_socket`` itself — filling the out-parameters the way libc
    does. Verified for real against a live launchd job during the incident this
    module was written for; this keeps the wiring honest in CI.
    """
    import ctypes

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    handed = os.dup(listener.fileno())

    class _FakeLib:
        def __init__(self):
            self._keepalive = None

        @property
        def launch_activate_socket(self):
            def fn(name, fds_ref, cnt_ref):
                assert name == activation._LAUNCHD_SOCKET_NAME
                arr = (ctypes.c_int * 1)(handed)
                self._keepalive = arr  # outlive the call; ctypes would GC it
                fds_ref._obj.contents = ctypes.cast(arr, ctypes.POINTER(ctypes.c_int)).contents
                cnt_ref._obj.value = 1
                return 0

            fn.argtypes = fn.restype = None
            return fn

        def free(self, _ptr):  # libc's free — a no-op on a Python-owned array
            return None

    monkeypatch.setattr("ctypes.util.find_library", lambda name: "libSystem.dylib")
    monkeypatch.setattr("ctypes.CDLL", lambda *a, **k: _FakeLib())

    got = activation._from_launchd()
    try:
        assert got is not None
        assert got.getsockname() == listener.getsockname()
    finally:
        if got is not None:
            got.detach()
        os.close(handed)
        listener.close()


def test_launchd_reporting_zero_sockets_is_not_a_listener(monkeypatch):

    class _FakeLib:
        @property
        def launch_activate_socket(self):
            def fn(name, fds_ref, cnt_ref):
                cnt_ref._obj.value = 0
                return 0

            fn.argtypes = fn.restype = None
            return fn

        def free(self, _ptr):
            return None

    monkeypatch.setattr("ctypes.util.find_library", lambda name: "libSystem.dylib")
    monkeypatch.setattr("ctypes.CDLL", lambda *a, **k: _FakeLib())
    assert activation._from_launchd() is None


def test_a_nonnumeric_listen_fds_is_ignored(monkeypatch):
    monkeypatch.setenv("LISTEN_FDS", "not-a-number")
    monkeypatch.delenv("LISTEN_PID", raising=False)
    assert activation._from_systemd() is None


def test_inherited_listener_returns_the_first_source_that_has_one(monkeypatch):
    """Covers the success return, which the sandboxed environment never reaches."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    monkeypatch.setattr(activation, "_from_systemd", lambda: None)
    monkeypatch.setattr(activation, "_from_launchd", lambda: sock)
    try:
        assert activation.inherited_listener() is sock
    finally:
        sock.close()


class _Res:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_registered_job_with_no_process_is_not_running(monkeypatch):
    """`launchctl print` succeeding is not proof of a running proxy.

    This is the state the machine was repeatedly left in: a job launchd knows
    about but has no process for. Reporting it as healthy is what turned a failed
    start into an outage discovered twenty minutes later.
    """
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _Res(0, "gui/501/com.distil.proxy = {\n\tstate = not running\n}"),
    )
    ok, why = setup.service_is_running(8788)
    assert ok is False
    assert "no running process" in why


def test_unregistered_job_is_not_running(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Res(1, "", "Could not find service"))
    ok, why = setup.service_is_running(8788)
    assert ok is False
    assert "no launchd job registered" in why


def test_running_job_reports_its_pid(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Res(0, "\tpid = 13260\n\truns = 1\n"))
    ok, why = setup.service_is_running(8788)
    assert ok is True
    assert "13260" in why


def test_reload_fails_loudly_when_bootstrap_fails(monkeypatch, real_service_api):
    """The bug: bootstrap failing was invisible because the legacy `load` exits 0.

    A failed bootstrap means NOTHING will restart the proxy. That has to be an
    error the caller sees, not a value it infers from a port probe that an
    orphaned process can satisfy.
    """
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(setup, "_port_free", lambda port, **k: True)

    def fake_run(cmd, *a, **k):
        if "bootstrap" in cmd:
            return _Res(5, "", "Bootstrap failed: 5: Input/output error")
        return _Res(0)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("time.sleep", lambda *_: None)  # assert the verdict, not the backoff
    ok, why = real_service_api["service_reload"](8788)
    assert ok is False
    assert "NOT registered" in why


def test_reload_refuses_to_bootstrap_onto_a_held_port(monkeypatch, real_service_api):
    """Bootstrapping while the old proxy still holds the port is the race itself.

    `launchctl unload` is asynchronous. Starting the replacement before the port
    is released is what made it die on EADDRINUSE and get discarded by launchd.
    """
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(setup, "_port_free", lambda port, **k: False)
    called = []
    monkeypatch.setattr("subprocess.run", lambda cmd, *a, **k: (called.append(cmd), _Res(0))[1])
    ok, why = real_service_api["service_reload"](8788)
    assert ok is False
    assert "still held" in why
    assert not any("bootstrap" in c for c in called), "must not bootstrap onto a held port"


def test_systemd_unit_state_decides_running(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Res(0, "active\n"))
    assert setup.service_is_running(8788)[0] is True
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Res(3, "failed\n"))
    ok, why = setup.service_is_running(8788)
    assert ok is False and "failed" in why


def test_a_supervisor_query_that_errors_is_not_running(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    def boom(*a, **k):
        raise OSError("launchctl missing")

    monkeypatch.setattr("subprocess.run", boom)
    ok, why = setup.service_is_running(8788)
    assert ok is False and "could not query" in why


def test_reload_on_an_unsupported_platform_says_so(monkeypatch, real_service_api):
    monkeypatch.setattr("platform.system", lambda: "Plan9")
    ok, why = real_service_api["service_reload"](8788)
    assert ok is False and "no service supervisor" in why


def test_reload_reports_a_failed_systemctl(monkeypatch, real_service_api):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("subprocess.run", lambda cmd, *a, **k: _Res(1, "", "unit not found"))
    ok, why = real_service_api["service_reload"](8788)
    assert ok is False and "unit not found" in why


def test_darwin_reload_waits_for_the_record_to_clear_then_bootstraps(monkeypatch, real_service_api):
    """The full launchd sequence, in order, with the wait that was missing.

    bootout → poll until the job record is GONE (not merely until the port frees:
    the port is released the instant the process dies, while launchd keeps the
    record in a `SIGTERMed` state, and bootstrapping into that returns EIO) →
    bootstrap → confirm a pid.
    """
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(setup, "_port_free", lambda port, **k: True)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls: list[str] = []
    # Two "still tearing down" prints, then the record is gone, then a live pid.
    prints = iter([_Res(0, "\tstate = SIGTERMed\n"), _Res(0, "\tstate = SIGTERMed\n"), _Res(1, "")])

    def fake_run(cmd, *a, **k):
        calls.append(cmd[1])
        if cmd[1] == "print":
            return next(prints, _Res(0, "\tpid = 4242\n"))
        return _Res(0)

    monkeypatch.setattr("subprocess.run", fake_run)
    ok, why = real_service_api["service_reload"](8788)
    assert ok is True and "4242" in why
    assert calls[0] == "bootout", "must stop the old job first"
    assert "bootstrap" in calls, "must register the new job"
    assert calls.index("bootout") < calls.index("bootstrap")


def test_reload_succeeds_once_the_unit_reports_active(monkeypatch, real_service_api):
    """The happy path: restart, then poll until the supervisor confirms a process."""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    states = iter(["activating\n", "activating\n", "active\n"])
    monkeypatch.setattr("subprocess.run", lambda cmd, *a, **k: _Res(0, next(states, "active\n")))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ok, why = real_service_api["service_reload"](8788)
    assert ok is True
    assert "active" in why


def test_reload_gives_up_rather_than_hanging(monkeypatch, real_service_api):
    """A unit that restarts but never becomes active must return, not block.

    `distil default` calls this synchronously; an unbounded wait here would hang
    the install instead of reporting a failure the user can act on.
    """
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("subprocess.run", lambda cmd, *a, **k: _Res(0, "activating\n"))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    clock = iter([0.0] + [i * 2.0 for i in range(1, 40)])
    monkeypatch.setattr("time.monotonic", lambda: next(clock, 999.0))
    ok, why = real_service_api["service_reload"](8788)
    assert ok is False
    assert "activating" in why or "did not come up" in why


def test_port_free_reports_a_port_that_never_frees():
    """The wait must terminate rather than block a CLI forever."""
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    try:
        assert setup._port_free(held.getsockname()[1], deadline=0.6) is False
    finally:
        held.close()


def test_port_free_returns_as_soon_as_nothing_listens():
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    assert setup._port_free(port, deadline=2.0) is True


def test_always_on_does_not_wire_the_pin_when_the_service_did_not_start(
    tmp_path, monkeypatch, capsys
):
    """The regression test for the actual outage.

    Previously `distil default --always-on` printed ✓ and wrote
    ANTHROPIC_BASE_URL into settings.json on a machine where the service had
    failed to register, because a leftover process answered the probe. The pin is
    durable and the job was not, so every later session died with
    ConnectionRefused. If the service did not start, nothing may be wired.
    """
    import argparse

    import distil.setup as setup_mod
    from distil import cli, onboard

    monkeypatch.setattr(
        onboard,
        "detect",
        lambda: onboard.Env(os_name="Darwin", agents=[("claude", "Claude Code")]),
    )
    rc_file = tmp_path / ".zshrc"
    rc_file.write_text("")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    monkeypatch.setattr(
        setup_mod, "service_spec", lambda *a, **k: (tmp_path / "s.plist", "content", "load-cmd")
    )
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: settings)
    monkeypatch.setattr(setup_mod, "log_dir", lambda: tmp_path)
    monkeypatch.setattr(setup_mod, "service_reload", lambda port: (False, "bootstrap failed"))
    # If the guard regresses, these are what would silently strand the machine.
    monkeypatch.setattr(
        setup_mod,
        "wire_settings_env",
        lambda *a, **k: pytest.fail("wired the pin despite a dead service"),
    )
    monkeypatch.setattr(
        setup_mod,
        "probe_routing",
        lambda *a, **k: pytest.fail("probed routing after the service failed to start"),
    )

    rc = cli.cmd_default(
        argparse.Namespace(
            undo=False,
            always_on=True,
            rc=str(rc_file),
            port=34121,
            agent="claude",
            mode="lossless-only",
            no_start=False,
            force=False,
        )
    )
    assert rc == 1
    assert not settings.exists(), "settings.json must be untouched when the service is down"
    out = capsys.readouterr().out
    assert "did not start" in out
    assert "Nothing was wired" in out


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="launchd plist is macOS-only")
def test_plist_asks_launchd_to_hold_the_socket():
    """Without this key launchd never creates a socket and activation is dead.

    The code path would still work — `inherited_listener()` returns None and we
    bind — so nothing fails loudly. The regression would be invisible until the
    next crash took the machine's API traffic down again, which is precisely why
    it is asserted here.
    """
    _path, content, _load = service_spec(8788, "lossless-only")
    plist = plistlib.loads(content.encode())
    listeners = plist["Sockets"]["Listeners"]
    assert listeners["SockServiceName"] == "8788"
    assert listeners["SockNodeName"] == "127.0.0.1"
    assert listeners["SockType"] == "stream"
    # Interactive keeps a small idle proxy off the top of the jetsam kill list.
    assert plist["ProcessType"] == "Interactive"
    # With the socket held by launchd, a client is already queued and waiting:
    # throttle is latency on a live request, not a crash-loop guard.
    assert plist["ThrottleInterval"] == 1
    # A service that is mandatory for the machine's API traffic must be able to
    # say why it stopped.
    assert plist["StandardErrorPath"].endswith("proxy.err")
