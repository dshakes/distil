"""Seamless proxy hot-swap: live wrap sessions pick up upgrades without a restart.

The problem: ``distil wrap`` historically ran the proxy as a *thread inside the
wrap process*. A ``pipx upgrade`` replaces the code on disk, but a running
interpreter can't reload itself — so picking up a new version meant restarting
the wrap, which kills the agent session with it.

The design (nginx-style zero-downtime handover):

  wrap (supervisor)                 proxy worker (subprocess)
  ───────────────────              ─────────────────────────
  owns the LISTENING socket  ──►   inherits the listener FD, accepts on it
  spawns / supervises              serves every request (same code path as the
  polls installed version           in-thread proxy: build_handler + server)
  on upgrade: spawn NEW worker ─►  new worker = fresh interpreter = NEW code,
  (same inherited FD)               accepts on the SAME socket — no port change,
  health-check via READY line       no bind race, no dropped connections
  drain old worker (SIGTERM)  ──►  stops accepting, finishes in-flight requests
                                    (non-daemon handler threads), flushes
                                    savings + shadow ledger, exits 0

Why this is safe and cheap:

* **Zero request-path cost.** The agent talks to the same localhost socket it
  always did; supervision is entirely out-of-band. The upgrade poll is one
  ``importlib.metadata`` read every 30 s in a daemon thread.
* **No handover gap.** Both workers hold the *same* listener FD during the
  overlap; the kernel delivers each new connection to exactly one ``accept()``.
  There is no unbind/rebind window and the port never changes, so the child's
  ``ANTHROPIC_BASE_URL`` stays valid for the whole session.
* **Fail-safe.** If the new worker does not report READY in time, it is killed
  and the old worker keeps serving — a broken upgrade downgrades to the old
  "restart when convenient" behavior, never to a dead session. If the
  supervisor itself cannot start, ``wrap_run`` falls back to the historical
  in-thread proxy.
* **State is on disk.** Savings ledger, shadow ledger, learn stats, restore
  store, gist cache — all persisted with advisory locks (POSIX). The new worker
  simply picks them up; nothing is handed over in memory.

POSIX-only: FD inheritance uses ``Popen(pass_fds=...)``. On Windows (or with
``DISTIL_HOT_SWAP=0``) the wrap keeps the historical in-thread proxy and the
existing version-skew warning.

Manual trigger: ``kill -USR1 <wrap pid>`` forces a handover now (used by the
test suite and handy after an upgrade if you don't want to wait for the poll).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields
from typing import Any

from ._log import log

_CONFIG_ENV = "DISTIL_WORKER_CONFIG"
_FD_ENV = "DISTIL_WORKER_FD"
_READY_PREFIX = "DISTIL-WORKER-READY "
# LLM streams legitimately run for many minutes; a draining worker must be
# allowed to finish them. ponytail: fixed ceiling, make env-tunable if a real
# workload ever streams longer.
_DRAIN_CAP_S = 15 * 60.0
# Worker-side twin of the cap above. server_close() joins non-daemon handler
# threads — that join is *how* in-flight streams finish, but unbounded it lets a
# single wedged handler pin a SIGTERM'd worker indefinitely (a client that never
# reads, or an upstream that accepts and never answers: 600s of _UPSTREAM_TIMEOUT
# each). Same reasoning as the finite upstream timeout and _drain_shadow's budget.
# Well under _DRAIN_CAP_S so the worker leaves on its own before the supervisor
# has to SIGKILL it.
_DRAIN_BUDGET_S = float(os.environ.get("DISTIL_DRAIN_BUDGET_S", "300"))
_POLL_INTERVAL_S = 30.0
# ponytail: a non-atomic reinstall window lasts ~1s; wait a beat, don't tight-loop.
_UPGRADE_SETTLE_S = 2.0


@dataclass
class WorkerConfig:
    """Everything a worker needs to rebuild the proxy handler.

    Serialized as JSON through the environment (not argv: keeps ``ps`` output
    clean and sidesteps argv quoting). Fields mirror ``wrap_run``'s parameters.
    """

    upstream: str
    lossless_only: bool = False
    verbatim: bool = False
    shape_output: str = "off"
    record: bool = True
    pricing_model: str = "claude-opus-4-8"
    expand: bool = False
    session_delta: bool = False
    shadow_rate: float = 0.0
    retention_rate: float = 0.0

    def to_env(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_env(cls) -> WorkerConfig:
        # Tolerate unknown keys: a supervisor from a NEWER build may spawn a worker
        # from an older one mid-upgrade (that is the whole point of hot-swap), and an
        # unrecognised config field must not take the session down.
        raw = json.loads(os.environ[_CONFIG_ENV])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def memory_evidence() -> str:
    """One line of memory context: swap + this process's peak RSS.

    Motivated by a real soak day: agent sessions died SIGKILL-style under swap
    exhaustion and the exit breadcrumbs couldn't say why. This line rides the
    breadcrumb and the heartbeat so the next silent kill is self-diagnosing."""
    parts = []
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kilobytes on Linux
        parts.append(f"wrap_maxrss_mb={maxrss >> (20 if sys.platform == 'darwin' else 10)}")
    except Exception:  # noqa: BLE001 — evidence is best-effort
        pass
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5
            ).stdout
            # "total = 11264.00M  used = 10968.75M  free = 295.25M ..."
            for k in ("used", "free"):
                if f"{k} = " in out:
                    parts.append(f"swap_{k}_mb={out.split(f'{k} = ')[1].split('M')[0].strip()}")
        else:
            with open("/proc/meminfo", encoding="ascii") as f:
                mi = dict(line.split(":", 1) for line in f if ":" in line)
            for k, label in (("SwapFree", "swap_free_mb"), ("MemAvailable", "mem_avail_mb")):
                if k in mi:
                    parts.append(f"{label}={int(mi[k].split()[0]) >> 10}")
    except Exception:  # noqa: BLE001 — evidence is best-effort
        pass
    return " ".join(parts)


def installed_version() -> str | None:
    """The distil version currently on disk (not the one this interpreter runs).

    ``importlib.metadata`` re-discovers the dist-info on every call, so a
    pipx/pip upgrade under a running process is visible here."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("distil-llm")
    except Exception:  # noqa: BLE001 — a version probe must never break anything
        return None


# ---------------------------------------------------------------------------
# Worker side — runs in the subprocess (`distil proxy-worker`, internal)
# ---------------------------------------------------------------------------


def worker_main() -> int:  # pragma: no cover — subprocess entry point: exercised
    # end-to-end by every test in tests/test_hotswap.py (READY handshake, serve,
    # mid-stream drain, rollback exit), but always in a child process where
    # in-process coverage cannot trace it.
    """Entry point for the proxy worker subprocess.

    Rebuilds the exact proxy the in-thread path would have built, on the
    listener FD inherited from the supervisor, then serves until told to drain.
    """
    from .proxy import QuietHTTPServer, _drain_shadow, build_handler

    cfg = WorkerConfig.from_env()
    fd = int(os.environ[_FD_ENV])

    # Test hook: if the named file exists, die before READY — lets the suite
    # break only the *replacement* worker (touch the file after the first spawn)
    # and prove the supervisor rolls back instead of killing the session.
    _fail_marker = os.environ.get("DISTIL_HOTSWAP_TEST_FAIL_READY")
    if _fail_marker and os.path.exists(_fail_marker):
        return 3

    savings = None
    if cfg.record:
        from .runtime import RuntimeSavings

        savings = RuntimeSavings(model=cfg.pricing_model)
    handler = build_handler(
        cfg.upstream,
        lossless_only=cfg.lossless_only,
        verbatim=cfg.verbatim,
        shape_output=cfg.shape_output,
        savings=savings,
        expand=cfg.expand,
        session_delta=cfg.session_delta,
        shadow_rate=cfg.shadow_rate,
        retention_rate=cfg.retention_rate,
    )

    class _WorkerServer(QuietHTTPServer):
        # Non-daemon handler threads: on drain, server_close() joins them, so
        # in-flight requests (including long LLM streams) finish before exit.
        daemon_threads = False

    server = _WorkerServer(("127.0.0.1", 0), handler, bind_and_activate=False)
    # Adopt the supervisor's listener instead of binding our own: same socket,
    # same port, zero-gap handover. The dup() via fileno keeps FD lifetimes
    # independent between supervisor and worker.
    server.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, fileno=fd)
    server.server_address = server.socket.getsockname()

    stopping = threading.Event()

    def _on_term(*_: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        # shutdown() blocks until the accept loop exits — never call it from
        # the signal frame itself.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _on_term)
    # The wrap's foreground group gets the terminal's Ctrl+C; the agent owns
    # that gesture, the worker must ignore it (same rule as the in-thread proxy).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # main() restored SIGPIPE to SIG_DFL (right for CLI filters piped to head).
    # A long-lived server must NOT die on it: a write to a client/upstream socket
    # that hung up should raise a catchable BrokenPipeError, not kill the worker
    # with exit=-13 mid-session. Restore the interpreter's server-safe default.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (ValueError, AttributeError):
        pass  # not main thread, or no SIGPIPE (Windows) — harmless to skip

    from . import __version__ as running_version

    print(f"{_READY_PREFIX}{running_version} {os.getpid()}", flush=True)

    try:
        while not stopping.is_set():
            try:
                server.serve_forever(poll_interval=0.25)
            except Exception:  # noqa: BLE001 — keep the session alive
                if stopping.is_set():
                    break
                log.warning("worker accept loop crashed; restarting", exc_info=True)
    finally:
        # Drain in-flight requests, but on a deadline. server_close() joins the
        # non-daemon handler threads; run it on a helper thread so the join is
        # *bounded* — an unbounded one hands any single wedged handler the power
        # to keep a SIGTERM'd worker alive forever.
        drain = threading.Thread(target=server.server_close, daemon=True)
        drain.start()
        drain.join(_DRAIN_BUDGET_S)
        _drain_shadow(handler)
        if savings is not None:
            savings.flush()  # SIGTERM lands here too — no savings are ever dropped
        if drain.is_alive():
            # Bounding the join alone changes nothing: returning from main() runs
            # interpreter shutdown, which re-joins those same non-daemon threads
            # and blocks all over again. Everything durable (shadow, savings) is
            # already flushed above, so leaving now costs nothing but the wedge.
            log.warning(
                "drain exceeded %.0fs — exiting with a handler still in flight",
                _DRAIN_BUDGET_S,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    return 0


# ---------------------------------------------------------------------------
# Supervisor side — runs inside the wrap process
# ---------------------------------------------------------------------------


class ProxySupervisor:
    """Owns the listener socket and the current proxy worker; swaps workers
    when the on-disk distil version changes (or on SIGUSR1)."""

    def __init__(self, cfg: WorkerConfig, *, host: str = "127.0.0.1") -> None:
        self._cfg = cfg
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, 0))
        self._listener.listen(128)
        os.set_inheritable(self._listener.fileno(), True)
        self.port: int = self._listener.getsockname()[1]
        self._worker: subprocess.Popen[bytes] | None = None
        self.worker_version: str | None = None
        self._handover_asked = threading.Event()
        self._stopping = threading.Event()
        self._lock = threading.Lock()  # ponytail: one handover at a time is plenty
        self._failed_version: str | None = None
        self._draining: list[subprocess.Popen[bytes]] = []

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Spawn the first worker and the upgrade-watch thread. Raises if the
        first worker fails readiness — caller falls back to in-thread proxy."""
        self._worker, self.worker_version = self._spawn_worker()
        threading.Thread(target=self._watch, daemon=True).start()

    def request_handover(self) -> None:
        """Signal-handler-safe manual trigger (SIGUSR1)."""
        self._handover_asked.set()

    def shutdown(self) -> None:
        """Session over: drain the current worker so its savings/shadow flush."""
        self._stopping.set()
        with self._lock:
            procs = [p for p in [self._worker, *self._draining] if p is not None]
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            self._listener.close()
        except OSError:
            pass

    # -- internals ----------------------------------------------------------

    def _spawn_worker(self) -> tuple[subprocess.Popen[bytes], str]:
        fd = self._listener.fileno()
        env = dict(os.environ)
        env[_CONFIG_ENV] = self._cfg.to_env()
        env[_FD_ENV] = str(fd)
        proc = subprocess.Popen(
            [sys.executable, "-m", "distil.cli", "proxy-worker"],
            pass_fds=(fd,),
            env=env,
            stdout=subprocess.PIPE,
            stderr=None,  # worker stderr shares the wrap's tty: DISTIL_DEBUG works
        )
        timeout = float(os.environ.get("DISTIL_WORKER_READY_TIMEOUT", "30"))
        version = self._await_ready(proc, timeout)
        # Keep the pipe drained forever after READY so a chatty worker can
        # never block on a full pipe; it should not print, but must not hang.
        assert proc.stdout is not None
        threading.Thread(target=proc.stdout.read, daemon=True).start()
        return proc, version

    @staticmethod
    def _await_ready(proc: subprocess.Popen[bytes], timeout: float) -> str:
        """Block until the worker prints its READY line; kill it and raise on
        anything else. Event-driven — the readline returns the moment the
        worker is serving, there is no polling sleep."""
        assert proc.stdout is not None
        result: list[str] = []

        def _read() -> None:
            line = proc.stdout.readline().decode("utf-8", "replace").strip()  # type: ignore[union-attr]
            if line.startswith(_READY_PREFIX):
                result.append(line[len(_READY_PREFIX) :].split()[0])

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        if not result:
            proc.kill()
            proc.wait(timeout=5)
            raise RuntimeError(
                f"proxy worker did not report ready within {timeout:.0f}s (exit={proc.poll()})"
            )
        return result[0]

    def _watch(self) -> None:
        """Upgrade poll + manual-trigger wait + dead-worker respawn + heartbeat."""
        self._heartbeat()  # first beat immediately: short sessions get one too
        self._census_beat()
        while not self._stopping.is_set():
            self._handover_asked.wait(_POLL_INTERVAL_S)
            if self._stopping.is_set():
                return
            manual = self._handover_asked.is_set()
            self._handover_asked.clear()
            self._heartbeat()
            self._census_beat()
            self._reap_drained()
            try:
                if self._worker is not None and self._worker.poll() is not None:
                    # Worker died underneath us (crash/OOM): the agent would get
                    # connection-refused for the rest of the session. Respawn —
                    # same self-heal contract as the old in-thread accept loop.
                    exit_code = self._worker.poll()
                    # A non-atomic reinstall (pip/uv --force-reinstall) deletes the
                    # package files before rewriting them; a worker spawned in that
                    # window dies importing half-gone code. Don't cry wolf or tight-
                    # loop: if the version is momentarily unreadable, let the install
                    # settle, then respawn from the (now complete) code on disk.
                    if installed_version() is None:
                        log.info(
                            "proxy worker died (exit=%s) during a package upgrade "
                            "window; waiting for it to settle",
                            exit_code,
                        )
                        self._stopping.wait(_UPGRADE_SETTLE_S)
                    else:
                        log.warning("proxy worker died (exit=%s); respawning", exit_code)
                    self._worker, self.worker_version = self._spawn_worker()
                    continue
                disk = installed_version()
                if manual or (
                    disk and disk != self.worker_version and disk != self._failed_version
                ):
                    self._handover(reason="manual" if manual else f"upgrade to {disk}")
            except Exception:  # noqa: BLE001 — the watch thread must survive anything
                log.warning("hot-swap watch iteration failed", exc_info=True)

    def _handover(self, *, reason: str) -> None:
        with self._lock:
            old, old_version = self._worker, self.worker_version
            try:
                self._worker, self.worker_version = self._spawn_worker()
            except Exception:  # noqa: BLE001 — a broken upgrade must not kill the session
                self._failed_version = installed_version()
                log.warning(
                    "hot-swap aborted (%s): new worker failed readiness; keeping v%s",
                    reason,
                    old_version,
                    exc_info=True,
                )
                print(
                    f"distil wrap: hot-swap to new version failed — staying on "
                    f"v{old_version} (restart the session to retry)",
                    file=sys.stderr,
                    flush=True,
                )
                return
            self._failed_version = None
            if old is not None and old.poll() is None:
                old.terminate()  # SIGTERM → drain: finish in-flight, flush, exit
                old._drain_t0 = time.monotonic()  # type: ignore[attr-defined]
                self._draining.append(old)
            print(
                f"distil wrap: proxy hot-swapped v{old_version} → "
                f"v{self.worker_version} ({reason}) — session uninterrupted",
                flush=True,  # announced from a thread; stdout may be a pipe
            )

    def _heartbeat(self) -> None:
        """Overwrite sessions/<sid>.hb with a timestamped memory snapshot.

        A SIGKILL (kernel swap-exhaustion kill) leaves no exit breadcrumb —
        nothing *can* be written at death. The heartbeat is the posthumous
        witness: a session with no .exit and a stale .hb died silently, and
        the .hb says what the machine looked like ≤30s before."""
        try:
            from .ledger import session_marker_path

            mp = session_marker_path()
            if mp is None:
                return
            mp.parent.mkdir(parents=True, exist_ok=True)
            hb = mp.with_name(mp.name + ".hb")
            tmp = hb.with_name(hb.name + ".tmp")
            tmp.write_text(
                f"alive {time.strftime('%Y-%m-%d %H:%M:%S')} {memory_evidence()}\n",
                encoding="utf-8",
            )
            os.replace(tmp, hb)
        except Exception:  # noqa: BLE001 — a heartbeat must never hurt the session
            pass

    def _census_beat(self) -> None:
        """Pulse the near-real-time community heartbeat while the session runs.

        The legacy in-thread proxy did this via _start_heartbeat_timer(); the
        hot-swap path serves in a worker subprocess that never touches census,
        so a long-lived `distil wrap` session would otherwise send NO beat until
        exit — the live counter reads 'no machines active' the whole time it's
        running. The supervisor outlives every worker swap, so it owns the beat.
        census self-throttles to ≤1/5min and only sends when tokens grew, so a
        30s poll tick is a cheap no-op most of the time. Fail-open."""
        try:
            from . import census as _census

            _census.maybe_heartbeat()
        except Exception:  # noqa: BLE001 — census must never touch the session
            pass

    def _reap_drained(self) -> None:
        now = time.monotonic()
        still = []
        for proc in self._draining:
            if proc.poll() is None:
                # A wedged drain must not linger forever next to a live session.
                # _drain_t0 is stamped once at handover; a plain-attribute read
                # here on purpose — an eager default would restart the clock
                # every poll and the cap would never fire.
                if now - getattr(proc, "_drain_t0", now) > _DRAIN_CAP_S:
                    proc.kill()
                else:
                    still.append(proc)
        self._draining = still
