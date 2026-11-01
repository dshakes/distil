"""`distil wrap` — transparent process wrapper. Real e2e, no mocks."""

from __future__ import annotations

import sys

import pytest
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from distil import ledger, proxy


def _await_file(path, timeout=60.0):
    """Readiness sync for slow CI runners — a fixed sleep loses the race there.

    The wrap installs its signal handlers before spawning the child, so a marker
    written by the child proves the whole stack is armed before we signal it."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"ready marker {path} not written within {timeout}s")


# Child program: reads ANTHROPIC_BASE_URL from env, drives a real request through
# the wrapped proxy, and exits 0 only if Distil actually saved tokens on it.
_CHILD = r"""
import os, json, urllib.request
base = os.environ["ANTHROPIC_BASE_URL"]
tr = "get_logs()\n" + "\n".join("info: verbose log line %d" % i for i in range(600)) + "\nDECISION: act"  # over the 2k-token lossy floor
body = json.dumps({"model": "claude-opus-4-8", "messages": [
    {"role": "user", "content": "go"},
    {"role": "user", "content": [{"type": "tool_result", "content": tr}]},
    {"role": "user", "content": "next"},
    {"role": "user", "content": "next"},
]}).encode()
req = urllib.request.Request(base + "/v1/messages", data=body,
                            headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=5) as r:
    saved = int(r.headers.get("x-distil-tokens-saved", "0"))
raise SystemExit(0 if saved > 0 else 7)
"""


def _start_echo_upstream() -> ThreadingHTTPServer:
    class Echo(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0))
            b = self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):  # noqa: ANN002
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_wrap_runs_child_through_proxy_and_records_savings(tmp_path, monkeypatch):
    upstream = _start_echo_upstream()
    up_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    led = tmp_path / "savings.jsonl"
    # RuntimeSavings flushes to ledger.default_path(), which honors DISTIL_HOME
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    try:
        code = proxy.wrap_run([sys.executable, "-c", _CHILD], upstream=up_url, record=True)
        assert code == 0  # child saw ANTHROPIC_BASE_URL, routed through, saw savings>0
        s = ledger.summary(led)
        assert s.by_trajectory.get("live-proxy", 0.0) > 0  # genuine savings persisted
        assert s.total_tokens_saved > 0
    finally:
        upstream.shutdown()


def test_wrap_missing_command_returns_127():
    code = proxy.wrap_run(["distil-no-such-binary-xyz"], record=False)
    assert code == 127


def test_wrap_propagates_child_exit_code(tmp_path):
    code = proxy.wrap_run([sys.executable, "-c", "raise SystemExit(3)"], record=False)
    assert code == 3


def test_cmd_wrap_strips_separator_and_rejects_empty():
    import argparse

    from distil.cli import cmd_wrap

    ns = argparse.Namespace(
        command=["--"],
        host="127.0.0.1",
        upstream="x",
        env_var="A",
        lossless_only=False,
        shape_output="off",
        no_record=True,
        pricing="claude-opus-4-8",
    )
    assert cmd_wrap(ns) == 2  # only the separator → nothing to run


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_wrap_survives_ctrl_c_while_child_lives(tmp_path):
    """A terminal Ctrl+C hits the whole foreground group. Agents like Claude Code
    swallow the first SIGINT (it cancels the turn, not the app) — the wrap must
    keep the proxy alive underneath them instead of exiting 130 and leaving the
    agent pointed at a dead port. A rapid BURST of presses must survive too:
    catching KeyboardInterrupt only around proc.wait() loses the race when a
    second press lands inside the except clause (the v1.11.2 escape path)."""
    import os
    import signal
    import subprocess
    import time

    # Child mimics claude: ignores SIGINT, then proves the proxy still answers.
    child = tmp_path / "child.py"
    ready = tmp_path / "ready"
    child.write_text(
        "import os, signal, time, urllib.request\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        f"open({str(ready)!r}, 'w').write('r')\n"  # armed — test may fire now
        "time.sleep(3)\n"  # outlive the SIGINT burst sent at ready
        "urllib.request.urlopen(os.environ['ANTHROPIC_BASE_URL'] + '/', timeout=2)\n",
        encoding="utf-8",
    )
    wrap = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "distil.cli",
            "wrap",
            "--no-record",
            "--",
            sys.executable,
            str(child),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # own group, like a terminal foreground job
    )
    _await_file(ready)  # child armed ⇒ wrap handlers installed (they precede spawn)
    for _ in range(5):  # rapid repeated presses, like a user mashing Ctrl+C
        os.killpg(os.getpgid(wrap.pid), signal.SIGINT)
        time.sleep(0.05)
    out, _ = wrap.communicate(timeout=60)
    # Child got HTTP 404 from the still-alive proxy root (any response beats
    # connection-refused); it exits nonzero on urllib.HTTPError — accept that,
    # reject only the pre-fix signature: wrap exit 130 with the child orphaned.
    assert wrap.returncode != 130, f"wrap died on Ctrl+C while child lived:\n{out}"
    assert "Connection refused" not in out and "URLError" not in out, out


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_wrap_still_exits_on_sigterm(tmp_path):
    """SIGTERM keeps its meaning: terminate the child, flush, exit."""
    import os
    import signal
    import subprocess

    ready = tmp_path / "ready"
    wrap = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "distil.cli",
            "wrap",
            "--no-record",
            "--",
            sys.executable,
            "-c",
            f"import time; open({str(ready)!r}, 'w').write('r'); time.sleep(30)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _await_file(ready)  # child armed ⇒ wrap's SIGTERM handler installed
    os.kill(wrap.pid, signal.SIGTERM)
    assert wrap.wait(timeout=30) == 130


@pytest.mark.skipif(sys.platform == "win32", reason="termios is POSIX-only")
def test_wrap_run_restores_terminal_on_child_exit(monkeypatch):
    """FIX 3: wrap_run restores the tty mode after the child exits, so an agent that
    dies in raw mode never leaves the user's shell wedged."""
    import termios

    sentinel = ["saved-termios-state"]
    calls: list = []

    class _FakeStdin:
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 0

    monkeypatch.setattr(sys, "stdin", _FakeStdin())
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: sentinel)
    monkeypatch.setattr(
        termios, "tcsetattr", lambda fd, when, attrs: calls.append((fd, when, attrs))
    )

    rc = proxy.wrap_run(["true"], record=False)
    assert rc == 0
    assert calls == [(0, termios.TCSADRAIN, sentinel)]  # restored once, with saved state


def test_wrap_marker_flips_to_one_when_traffic_flows(tmp_path, monkeypatch):
    """wrap_run writes the session marker "0"; the child's request through the
    proxy flips it to "1" — the signal the statusline's bypass check reads."""
    upstream = _start_echo_upstream()
    up_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    try:
        code = proxy.wrap_run([sys.executable, "-c", _CHILD], upstream=up_url, record=False)
        assert code == 0
        mp = ledger.session_marker_path()  # env still carries the wrap-minted sid
        assert mp is not None
        assert mp.read_text(encoding="utf-8") == "1"
    finally:
        upstream.shutdown()


def test_wrap_marker_stays_zero_when_child_never_calls(tmp_path):
    """A child that bypasses the proxy leaves the marker at "0" — that is the
    bypass signature, not an error."""
    code = proxy.wrap_run([sys.executable, "-c", "pass"], record=False)
    assert code == 0
    mp = ledger.session_marker_path()
    assert mp is not None
    assert mp.read_text(encoding="utf-8") == "0"


def test_wrap_records_child_exit_code(tmp_path):
    """The wrap is the only witness to how the agent died — a silent quit is
    undiagnosable without this breadcrumb. Clean exit and signal death both."""
    code = proxy.wrap_run([sys.executable, "-c", "raise SystemExit(3)"], record=False)
    assert code == 3
    mp = ledger.session_marker_path()
    exit_file = mp.with_name(mp.name + ".exit")
    assert "exit code 3" in exit_file.read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal death semantics")
def test_wrap_records_child_signal_death(tmp_path, monkeypatch):
    monkeypatch.delenv("DISTIL_SESSION", raising=False)  # fresh sid, fresh files
    child = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    code = proxy.wrap_run([sys.executable, "-c", child], record=False)
    assert code == -9
    mp = ledger.session_marker_path()
    exit_file = mp.with_name(mp.name + ".exit")
    assert "signal SIGKILL" in exit_file.read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGHUP is POSIX")
def test_wrap_breadcrumbs_sighup_group_kill(tmp_path):
    """Terminal tab close = SIGHUP to the group: the wrap dies WITH the child,
    so the wrap itself must breadcrumb the signal — it's the only trace."""
    import os
    import signal
    import subprocess

    ready = tmp_path / "ready"
    wrap = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "distil.cli",
            "wrap",
            "--no-record",
            "--",
            sys.executable,
            "-c",
            f"import time; open({str(ready)!r}, 'w').write('r'); time.sleep(30)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _await_file(ready)  # child armed ⇒ handlers installed and marker written
    os.kill(wrap.pid, signal.SIGHUP)
    wrap.wait(timeout=30)
    sessions = os.path.join(os.environ["DISTIL_HOME"], "sessions")
    exits = [f for f in os.listdir(sessions) if f.endswith(".exit")]
    assert exits, os.listdir(sessions)
    with open(os.path.join(sessions, exits[0]), encoding="utf-8") as f:
        assert "wrap received SIGHUP" in f.read()
