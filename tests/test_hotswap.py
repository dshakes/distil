"""Hot-swap — supervisor/worker handover. Real subprocesses, no mocks, no sleeps
as synchronization: every wait is on an explicit readiness signal (READY line,
marker file, output line) with a deadline, per the CI-deflake lesson."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from distil.hotswap import _READY_PREFIX, WorkerConfig

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="hot-swap is POSIX-only (listener-FD inheritance)"
)

_TOOL_RESULT = (
    "get_logs()\n"
    + "\n".join("info: verbose log line %d" % i for i in range(40))
    + "\nDECISION: act"
)


def _payload(stream: bool = False) -> bytes:
    # tool_result two turns back so the recency exemption doesn't keep it
    # verbatim — the request genuinely compresses (same shape as test_streaming)
    return json.dumps(
        {
            "model": "claude-opus-4-8",
            "max_tokens": 64,
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": _TOOL_RESULT}
                    ],
                },
                {"role": "assistant", "content": "checking"},
                {"role": "user", "content": "and now?"},
            ],
        }
    ).encode()


def _start_upstream(*, sse_chunks: int = 0, delay: float = 0.0) -> ThreadingHTTPServer:
    """Fake provider. Plain JSON by default; slow SSE when sse_chunks > 0."""

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — http.server API
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if sse_chunks:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(sse_chunks):
                    self.wfile.write(
                        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"c%d"}}\n\n'
                        % i
                    )
                    self.wfile.flush()
                    time.sleep(delay)
                self.wfile.write(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
            else:
                body = b'{"id":"m1","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":2}}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *a):  # noqa: D102 — quiet
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _read_ready(proc: subprocess.Popen, timeout: float = 30.0) -> str:
    got: list[str] = []

    def _r() -> None:
        line = proc.stdout.readline().decode("utf-8", "replace").strip()
        if line.startswith(_READY_PREFIX):
            got.append(line)

    t = threading.Thread(target=_r, daemon=True)
    t.start()
    t.join(timeout)
    assert got, f"worker never reported READY (exit={proc.poll()})"
    return got[0]


def _stop_upstream(server: ThreadingHTTPServer) -> None:
    """Stop the fake provider AND close its listening socket.

    shutdown() only stops the accept loop — the listener stays open, so a worker
    that connects afterwards completes its handshake into the backlog, sends its
    POST, and waits for a reply nobody will ever send. That turns "upstream is
    gone" into a 600s _UPSTREAM_TIMEOUT black hole instead of an instant
    ECONNREFUSED, and a worker draining such a handler looks hung. Closing the
    listener is what actually makes the upstream gone.
    """
    server.shutdown()
    server.server_close()


def _spawn_worker(tmp_path, upstream_port: int, **cfg_kw):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    os.set_inheritable(listener.fileno(), True)
    cfg = WorkerConfig(upstream=f"http://127.0.0.1:{upstream_port}", **cfg_kw)
    env = dict(os.environ)
    env["DISTIL_WORKER_CONFIG"] = cfg.to_env()
    env["DISTIL_WORKER_FD"] = str(listener.fileno())
    env["DISTIL_HOME"] = str(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "distil.cli", "proxy-worker"],
        pass_fds=(listener.fileno(),),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    _read_ready(proc)
    return proc, listener.getsockname()[1], listener


def _post(port: int, body: bytes, timeout: float = 30.0):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _await_file(path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"marker {path} not written within {timeout}s")


def test_worker_config_roundtrip(monkeypatch):
    cfg = WorkerConfig(upstream="https://x", shadow_rate=0.02, expand=True)
    monkeypatch.setenv("DISTIL_WORKER_CONFIG", cfg.to_env())
    assert WorkerConfig.from_env() == cfg


def test_worker_serves_on_inherited_fd_and_drains_clean(tmp_path):
    """The worker adopts the supervisor's listener (same port, no rebind) and a
    SIGTERM drain flushes and exits 0."""
    upstream = _start_upstream()
    proc, port, _listener = _spawn_worker(tmp_path, upstream.server_address[1], record=False)
    try:
        r = _post(port, _payload())
        assert r.status == 200
        assert b'"text":"ok"' in r.read()
    finally:
        proc.terminate()
        _stop_upstream(upstream)
    assert proc.wait(timeout=30) == 0


def test_drain_is_bounded_when_a_handler_cannot_finish(tmp_path, monkeypatch):
    """A SIGTERM'd worker must still exit when a handler can never complete.

    server_close() joins the non-daemon handler threads — that join is *how*
    in-flight streams drain, but unbounded it hands one wedged handler the power
    to pin the worker for the whole upstream timeout (600s), or forever if a
    client simply stops reading. Regression for the intermittent macOS CI hang
    where a straggling handler outlived SIGTERM by >30s.

    The wedge here is an upstream that accepts and then never answers — the same
    shape as a provider that black-holes a connection.
    """
    black_hole = socket.socket()
    black_hole.bind(("127.0.0.1", 0))
    black_hole.listen(8)
    monkeypatch.setenv("DISTIL_DRAIN_BUDGET_S", "5")
    monkeypatch.setenv("DISTIL_UPSTREAM_TIMEOUT", "600")  # the wedge, if unbounded
    proc, port, _listener = _spawn_worker(tmp_path, black_hole.getsockname()[1], record=False)
    try:
        threading.Thread(
            target=_post, args=(port, _payload()), kwargs={"timeout": 90}, daemon=True
        ).start()
        black_hole.settimeout(30)
        conn, _ = black_hole.accept()  # the handler is now parked on the upstream read
        proc.terminate()
        # 5s budget; without the bound this waits out the 600s upstream timeout.
        assert proc.wait(timeout=60) == 0
    finally:
        try:
            conn.close()
        except (OSError, NameError, UnboundLocalError):
            pass
        black_hole.close()
        if proc.poll() is None:
            proc.kill()


@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason="no SIGPIPE on this platform")
def test_worker_survives_abrupt_client_disconnect(tmp_path):
    """Regression for `proxy worker died (exit=-13)` on long sessions.

    main() restores SIGPIPE to SIG_DFL (right for CLI filters piped to head).
    A write to a client socket that hung up would then kill the whole worker
    with signal 13 instead of raising a catchable BrokenPipeError. The worker
    must reinstate SIG_IGN so a broken client only aborts that one request.
    """
    import struct

    upstream = _start_upstream()
    proc, port, _listener = _spawn_worker(tmp_path, upstream.server_address[1], record=False)
    payload = _payload()
    req = (
        b"POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\nContent-Length: "
        + str(len(payload)).encode()
        + b"\r\nConnection: close\r\n\r\n"
        + payload
    )
    try:
        # Send a valid request, then slam the socket shut with an RST (SO_LINGER 0)
        # before reading the response — so the worker's reply lands on a dead peer.
        # Without the fix, one of these delivers SIGPIPE and kills the worker.
        for _ in range(30):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", port))
            s.sendall(req)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            s.close()  # RST: peer is gone before the worker writes its response
        # The worker must still be alive and serving a normal request.
        assert proc.poll() is None, f"worker died (exit={proc.poll()}) on abrupt client disconnect"
        r = _post(port, payload)
        assert r.status == 200
        assert b'"text":"ok"' in r.read()
    finally:
        proc.terminate()
        _stop_upstream(upstream)
    assert proc.wait(timeout=30) == 0


def test_worker_drain_completes_inflight_stream(tmp_path):
    """The seamless property: a SIGTERM mid-stream must NOT cut the response.
    Non-daemon handler threads mean the draining worker finishes the stream,
    then exits."""
    n_chunks = 15
    upstream = _start_upstream(sse_chunks=n_chunks, delay=0.15)  # ~2.3s stream
    proc, port, _listener = _spawn_worker(tmp_path, upstream.server_address[1], record=False)
    got: dict = {}
    first_bytes = threading.Event()

    def _client() -> None:
        r = _post(port, _payload(stream=True), timeout=60)
        chunks = []
        while True:
            b = r.read(256)
            if not b:
                break
            first_bytes.set()
            chunks.append(b)
        got["body"] = b"".join(chunks)

    t = threading.Thread(target=_client)
    t.start()
    assert first_bytes.wait(timeout=30), "stream never started"
    proc.terminate()  # drain begins while the stream is mid-flight
    t.join(timeout=60)
    assert not t.is_alive(), "client never finished reading"
    body = got["body"]
    # each SSE event names the type twice (event: line + data: json)
    assert body.count(b"event: content_block_delta") == n_chunks  # nothing cut short
    assert b"message_stop" in body
    assert proc.wait(timeout=60) == 0
    _stop_upstream(upstream)


def _wrap_session(tmp_path, upstream_port: int, extra_env: dict | None = None):
    """Start a full `distil wrap` around a child that: makes a request, drops
    marker r1, waits for marker go, makes a second request, exits by result."""
    payload_file = tmp_path / "payload.json"
    payload_file.write_bytes(_payload())
    r1 = tmp_path / "r1"
    go = tmp_path / "go"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, time, urllib.request\n"
        f"body = open({str(payload_file)!r}, 'rb').read()\n"
        "def req():\n"
        "    r = urllib.request.urlopen(urllib.request.Request(\n"
        "        os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages', data=body,\n"
        "        headers={'Content-Type': 'application/json'}), timeout=30)\n"
        "    assert r.status == 200, r.status\n"
        "req()\n"
        f"open({str(r1)!r}, 'w').write('x')\n"
        "deadline = time.monotonic() + 60\n"
        f"while not os.path.exists({str(go)!r}):\n"
        "    assert time.monotonic() < deadline, 'go marker never arrived'\n"
        "    time.sleep(0.05)\n"
        "req()\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["DISTIL_HOME"] = str(tmp_path)
    env.update(extra_env or {})
    wrap = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "distil.cli",
            "wrap",
            "--no-record",
            "--upstream",
            f"http://127.0.0.1:{upstream_port}",
            "--",
            sys.executable,
            str(child),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return wrap, r1, go


def test_wrap_handover_on_sigusr1_keeps_session(tmp_path):
    """Full e2e: request → forced hot-swap → request, one uninterrupted session.
    SIGUSR1 stands in for 'a new version landed on disk' — same code path."""
    upstream = _start_upstream()
    wrap, r1, go = _wrap_session(tmp_path, upstream.server_address[1])
    try:
        _await_file(r1)  # first request done through worker #1
        os.kill(wrap.pid, signal.SIGUSR1)
        # handover completion is announced on wrap stdout; wait for the line
        # by watching the process output via the swap's own announcement file-
        # free signal: poll the second request's success (child unblocks on go).
        # We must not touch `go` until the swap happened — read wrap stdout.
        deadline = time.monotonic() + 60
        out_lines: list[str] = []

        def _pump() -> None:
            for line in wrap.stdout:  # type: ignore[union-attr]
                out_lines.append(line)

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        while time.monotonic() < deadline:
            if any("hot-swapped" in ln for ln in out_lines):
                break
            time.sleep(0.05)
        assert any("hot-swapped" in ln for ln in out_lines), "".join(out_lines)
        go.write_text("x")  # unblock the child: request #2 through worker #2
        code = wrap.wait(timeout=60)
        assert code == 0, "".join(out_lines)
    finally:
        _stop_upstream(upstream)
        if wrap.poll() is None:
            wrap.kill()


def test_failed_upgrade_rolls_back_and_session_survives(tmp_path):
    """A replacement worker that never reports READY is discarded: the old
    worker keeps serving and the session finishes clean."""
    upstream = _start_upstream()
    fail_marker = tmp_path / "break-the-new-worker"
    wrap, r1, go = _wrap_session(
        tmp_path,
        upstream.server_address[1],
        extra_env={
            "DISTIL_HOTSWAP_TEST_FAIL_READY": str(fail_marker),
            "DISTIL_WORKER_READY_TIMEOUT": "5",
        },
    )
    try:
        _await_file(r1)  # worker #1 came up fine (marker didn't exist yet)
        fail_marker.write_text("x")  # every worker spawned from now on dies
        os.kill(wrap.pid, signal.SIGUSR1)
        deadline = time.monotonic() + 60
        out_lines: list[str] = []

        def _pump() -> None:
            for line in wrap.stdout:  # type: ignore[union-attr]
                out_lines.append(line)

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        while time.monotonic() < deadline:
            if any("hot-swap to new version failed" in ln for ln in out_lines):
                break
            time.sleep(0.05)
        assert any("hot-swap to new version failed" in ln for ln in out_lines), "".join(out_lines)
        go.write_text("x")  # request #2 must still succeed via the OLD worker
        code = wrap.wait(timeout=60)
        assert code == 0, "".join(out_lines)
    finally:
        _stop_upstream(upstream)
        if wrap.poll() is None:
            wrap.kill()


def test_dead_worker_respawns_and_session_survives(tmp_path):
    """The self-heal contract, hot-swap edition: a worker that dies mid-session
    (crash/OOM) is respawned — the agent must not see connection-refused for
    the rest of the session. SIGUSR1 wakes the watch thread immediately so the
    test doesn't ride the 30s poll."""
    upstream = _start_upstream()
    wrap, r1, go = _wrap_session(tmp_path, upstream.server_address[1])
    try:
        _await_file(r1)  # worker #1 served request #1
        pgrep = subprocess.run(
            ["pgrep", "-P", str(wrap.pid), "-f", "proxy-worker"],
            capture_output=True,
            text=True,
        )
        old_pid = int(pgrep.stdout.split()[0])
        os.kill(old_pid, signal.SIGKILL)  # simulate a worker crash/OOM
        os.kill(wrap.pid, signal.SIGUSR1)  # wake the watch (dead-check runs first)
        deadline = time.monotonic() + 60
        new_pid = None
        while time.monotonic() < deadline:
            pgrep = subprocess.run(
                ["pgrep", "-P", str(wrap.pid), "-f", "proxy-worker"],
                capture_output=True,
                text=True,
            )
            pids = [int(p) for p in pgrep.stdout.split()]
            live = [p for p in pids if p != old_pid]
            if live:
                new_pid = live[0]
                break
            time.sleep(0.05)
        assert new_pid is not None, "worker was never respawned"
        go.write_text("x")  # request #2 must succeed via the respawned worker
        assert wrap.wait(timeout=60) == 0
    finally:
        _stop_upstream(upstream)
        if wrap.poll() is None:
            wrap.kill()


# ---------------------------------------------------------------------------
# In-process supervisor tests: the e2e tests above run the supervisor inside a
# wrap *subprocess*, which coverage can't see. These drive ProxySupervisor
# directly in this process — same real workers, deterministic branch coverage
# for handover, rollback, and drain-reaping.
# ---------------------------------------------------------------------------


def _inproc_supervisor(tmp_path, monkeypatch, upstream_port: int):
    from distil.hotswap import ProxySupervisor

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    return ProxySupervisor(WorkerConfig(upstream=f"http://127.0.0.1:{upstream_port}", record=False))


def test_supervisor_inprocess_handover_and_reap(tmp_path, monkeypatch):
    upstream = _start_upstream()
    sup = _inproc_supervisor(tmp_path, monkeypatch, upstream.server_address[1])
    try:
        sup.start()
        v1 = sup.worker_version
        assert _post(sup.port, _payload()).status == 200
        old_worker = sup._worker
        sup._handover(reason="test")  # direct call: no watch-thread wait
        assert sup.worker_version == v1  # same code on disk, fresh worker
        assert sup._worker is not old_worker
        assert _post(sup.port, _payload()).status == 200  # new worker serves
        assert old_worker.wait(timeout=30) == 0  # old worker drained clean
        sup._reap_drained()
        assert sup._draining == []
    finally:
        sup.shutdown()
        _stop_upstream(upstream)
    assert sup._worker is not None and sup._worker.poll() is not None


def test_supervisor_inprocess_rollback_keeps_old_worker(tmp_path, monkeypatch):
    upstream = _start_upstream()
    fail_marker = tmp_path / "break-replacements"
    monkeypatch.setenv("DISTIL_HOTSWAP_TEST_FAIL_READY", str(fail_marker))
    monkeypatch.setenv("DISTIL_WORKER_READY_TIMEOUT", "5")
    sup = _inproc_supervisor(tmp_path, monkeypatch, upstream.server_address[1])
    try:
        sup.start()  # marker absent: worker #1 healthy
        survivor = sup._worker
        fail_marker.write_text("x")  # every worker from now on dies pre-READY
        sup._handover(reason="test")  # must roll back, not raise
        assert sup._worker is survivor  # old worker untouched
        assert _post(sup.port, _payload()).status == 200  # and still serving
    finally:
        sup.shutdown()
        _stop_upstream(upstream)


def test_await_ready_times_out_and_kills(tmp_path):
    from distil.hotswap import ProxySupervisor

    proc = subprocess.Popen(  # never prints READY
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
    )
    with pytest.raises(RuntimeError, match="did not report ready"):
        ProxySupervisor._await_ready(proc, timeout=0.5)
    assert proc.poll() is not None  # the impostor was killed, not leaked


def test_watch_thread_handles_manual_trigger_and_respawn(tmp_path, monkeypatch):
    """Drive the watch thread itself (not _handover directly): request_handover
    wakes it for a manual swap; a SIGKILLed worker is respawned on the next
    wake. Waits are on observable state changes, not sleeps."""
    upstream = _start_upstream()
    sup = _inproc_supervisor(tmp_path, monkeypatch, upstream.server_address[1])
    try:
        sup.start()
        first = sup._worker
        sup.request_handover()  # manual: watch thread wakes and swaps
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and sup._worker is first:
            time.sleep(0.05)
        assert sup._worker is not first, "watch thread never performed the swap"
        assert _post(sup.port, _payload()).status == 200

        crashed = sup._worker
        crashed.kill()  # simulate worker crash/OOM
        crashed.wait(timeout=10)
        sup.request_handover()  # wake the watch: dead-check runs before swap
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and (
            sup._worker is crashed or sup._worker.poll() is not None
        ):
            time.sleep(0.05)
        assert sup._worker is not crashed and sup._worker.poll() is None
        assert _post(sup.port, _payload()).status == 200  # respawned and serving
    finally:
        sup.shutdown()
        _stop_upstream(upstream)


def test_respawn_during_upgrade_window_is_quiet(tmp_path, monkeypatch, caplog):
    """A worker that dies while the package is mid-reinstall (installed_version()
    momentarily None, e.g. pip/uv --force-reinstall) is respawned WITHOUT a scary
    WARNING and without tight-looping — the non-atomic reinstall thrash guard."""
    import logging
    import distil.hotswap as hs

    upstream = _start_upstream()
    sup = _inproc_supervisor(tmp_path, monkeypatch, upstream.server_address[1])
    try:
        sup.start()
        # Simulate the reinstall window: package version unreadable, settle fast.
        monkeypatch.setattr(hs, "installed_version", lambda: None)
        monkeypatch.setattr(hs, "_UPGRADE_SETTLE_S", 0.05)
        crashed = sup._worker
        crashed.kill()
        crashed.wait(timeout=10)
        with caplog.at_level(logging.INFO, logger="distil"):
            sup.request_handover()  # wake the watch: dead-check runs
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and (
                sup._worker is crashed or sup._worker.poll() is not None
            ):
                time.sleep(0.05)
        assert sup._worker is not crashed and sup._worker.poll() is None
        assert _post(sup.port, _payload()).status == 200  # respawned and serving
        died = [r for r in caplog.records if "died" in r.getMessage()]
        assert died, "no death was logged"
        # every death log during the window is INFO, never WARNING+
        assert all(r.levelno == logging.INFO for r in died), [
            (r.levelno, r.getMessage()) for r in died
        ]
    finally:
        sup.shutdown()
        _stop_upstream(upstream)


def test_reap_kills_a_wedged_drain(tmp_path, monkeypatch):
    """A draining worker stuck past the cap gets SIGKILLed, not leaked."""
    upstream = _start_upstream()
    sup = _inproc_supervisor(tmp_path, monkeypatch, upstream.server_address[1])
    wedged = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        wedged._drain_t0 = time.monotonic() - 10_000  # long past _DRAIN_CAP_S
        sup._draining.append(wedged)
        sup._reap_drained()
        assert wedged.wait(timeout=10) != 0  # killed
        assert sup._draining == []
    finally:
        if wedged.poll() is None:
            wedged.kill()
        sup.shutdown()
        _stop_upstream(upstream)


def test_memory_evidence_is_one_parseable_line():
    from distil.hotswap import memory_evidence

    line = memory_evidence()
    assert "\n" not in line
    assert "wrap_maxrss_mb=" in line  # rusage works on every POSIX box
    for token in line.split():
        assert "=" in token  # strictly key=value, greppable by soak-report


def test_supervisor_writes_heartbeat_immediately(tmp_path, monkeypatch):
    """A session that dies to SIGKILL leaves no exit breadcrumb — the heartbeat
    is the posthumous witness, so it must exist within seconds of start."""
    upstream = _start_upstream()
    sid = f"s999-{os.getpid()}"
    monkeypatch.setenv("DISTIL_SESSION", sid)
    sup = _inproc_supervisor(tmp_path, monkeypatch, upstream.server_address[1])
    try:
        sup.start()
        hb = tmp_path / "sessions" / f"{sid}.hb"
        _await_file(hb, timeout=10)  # first beat is immediate, not 30s away
        content = hb.read_text()
        assert content.startswith("alive ")
        assert "wrap_maxrss_mb=" in content
    finally:
        sup.shutdown()
        _stop_upstream(upstream)


def test_watch_pulses_census_heartbeat(monkeypatch):
    """The supervisor's watch loop is the ONLY census beat in the hot-swap path
    (the worker subprocess never touches census). If this wiring breaks, a
    long-lived `distil wrap` session sends no beat until exit and the live
    counter reads 'no machines active' the whole time it runs."""
    from distil.hotswap import ProxySupervisor

    calls = []
    import distil.census as census

    monkeypatch.setattr(census, "maybe_heartbeat", lambda: calls.append(1))
    # _census_beat uses no instance state — call the unbound method w/ dummy self.
    ProxySupervisor._census_beat(object())
    assert calls == [1], "watch loop must pulse the community heartbeat"

    # Fail-open: a census fault must never escape into the serving path.
    def _boom() -> bool:
        raise RuntimeError("upstash down")

    monkeypatch.setattr(census, "maybe_heartbeat", _boom)
    ProxySupervisor._census_beat(object())  # must not raise
