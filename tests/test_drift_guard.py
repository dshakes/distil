"""The drift alarm that acts: a breach holds the proxy at lossless-only.

Every test here drives a real proxy over a real socket against a stub upstream, because
the property is about what reaches the wire — a guard that trips in memory but still
lets a digest through is exactly the alarm-that-only-prints this replaced.
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from distil import drift
from distil import receipts as R
from distil.proxy import build_handler

_FULL_MARK = b"quietly the harbor lamps were counted"


def _log() -> str:
    """Bland, varied lines: Tier-0 keeps every one (nothing to fold), the digest elides
    them — so the marker's presence on the wire tells the two modes apart."""
    rng = random.Random(7)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo".split()
    lines = [" ".join(rng.choice(words) for _ in range(rng.randint(3, 9))) for _ in range(40)]
    lines[10] = _FULL_MARK.decode()
    return "\n".join(lines)


def _digestible() -> dict[str, Any]:
    return {
        "model": "claude-test",
        "max_tokens": 64,
        "system": "You are a test agent. " * 30,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": _log(),
                    }
                ],
            },
            {"role": "user", "content": "continue"},
            {"role": "user", "content": "and again"},
        ],
    }


class _Up(BaseHTTPRequestHandler):
    """Records the last forwarded body. Answers with a tool call whose name depends on
    whether the full log reached it — so compression changes the decision, and the
    model's self-replay (full vs full) never does. That is a paired d = -1 per sample."""

    last: dict[str, bytes] = {}

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        _Up.last["raw"] = raw
        name = "Read" if _FULL_MARK in raw else "Grep"
        b = json.dumps(
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "x", "name": name, "input": {"p": 1}}],
                "model": "claude-test",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 100, "output_tokens": 10},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a: object) -> None:
        pass


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv(drift.GUARD_OPT_OUT, raising=False)
    return tmp_path


@pytest.fixture()
def proxy():  # type: ignore[no-untyped-def]
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    servers = [up]

    def make(**kw: Any) -> int:
        h = build_handler(f"http://127.0.0.1:{up.server_address[1]}", **kw)
        px = ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=px.serve_forever, daemon=True).start()
        servers.append(px)
        return int(px.server_address[1])

    yield make
    for s in servers:
        s.shutdown()


def _post(port: int) -> tuple[dict[str, str], bytes]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(_digestible()).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return {k.lower(): v for k, v in r.headers.items()}, r.read()


def _capture_guard(monkeypatch: pytest.MonkeyPatch) -> list[drift.DriftGuard]:
    """Keep a handle on the guard the proxy builds, so a test can feed it directly."""
    made: list[drift.DriftGuard] = []
    real = drift.DriftGuard.start

    def start(**kw: Any) -> drift.DriftGuard:
        g = real(**kw)
        made.append(g)
        return g

    monkeypatch.setattr(drift.DriftGuard, "start", staticmethod(start))
    return made


def _trip(g: drift.DriftGuard) -> None:
    for _ in range(200):
        g.observe(-1)
        if g.engaged:
            return
    raise AssertionError("sustained harm never tripped the guard")


def _release() -> None:
    import argparse

    from distil.cli import cmd_reset

    cmd_reset(argparse.Namespace(shadow=False, drift_guard=True))


def _assert_digest(h: dict[str, str]) -> None:
    assert h["x-distil-mode"] == "digest"
    assert _FULL_MARK not in _Up.last["raw"], "fixture must actually digest, or nothing is tested"
    assert h.get("x-distil-output-shaping") == "aggressive"
    assert "x-distil-drift-guard" not in h


def _assert_held(h: dict[str, str]) -> None:
    assert h["x-distil-mode"] == "lossless-only"
    assert h["x-distil-drift-guard"] == "held"
    assert _FULL_MARK in _Up.last["raw"], "a held request must reach the wire undigested"
    assert "x-distil-output-shaping" not in h


# ---------------------------------------------------------------------------


def test_no_breach_leaves_compression_unchanged(home, proxy, monkeypatch):
    guards = _capture_guard(monkeypatch)
    port = proxy(shape_output="aggressive")
    for _ in range(3):
        _assert_digest(_post(port)[0])
    for _ in range(40):
        guards[0].observe(0)  # neutral evidence never trips
    _assert_digest(_post(port)[0])
    assert not drift.held_now()


def test_breach_holds_the_very_next_request_lossless_only(home, proxy, monkeypatch):
    guards = _capture_guard(monkeypatch)
    port = proxy(shape_output="aggressive")
    _assert_digest(_post(port)[0])

    _trip(guards[0])
    _assert_held(_post(port)[0])

    # ...recorded where it can be audited: the state file and one receipt on the chain.
    state = drift.LiveDrift.load()
    assert state.monitor.tripped and state.monitor.evalue >= 20 and state.tripped_ts > 0
    rows = [r for r in R.read() if r.mode == "drift-trip"]
    assert len(rows) == 1 and "lossless-only" in rows[0].certificate
    assert rows[0].reversible is False and rows[0].tokens_original == 0
    assert R.verify().ok


def test_held_requests_are_receipted_as_byte_reversible(home, proxy, monkeypatch):
    """Held = Tier-0, which round-trips byte-exact; the receipt must say so."""
    guards = _capture_guard(monkeypatch)
    port = proxy(shape_output="aggressive")
    _trip(guards[0])
    _post(port)
    last = [r for r in R.read() if r.mode == "lossless-only"][-1]
    assert last.reversible is True and last.handles == []


def test_the_hold_survives_a_proxy_restart_until_released(home, proxy, monkeypatch):
    guards = _capture_guard(monkeypatch)
    proxy(shape_output="aggressive")
    _trip(guards[0])

    restarted = proxy(shape_output="aggressive")  # a fresh process reads drift.json
    _assert_held(_post(restarted)[0])

    _release()
    assert not drift.held_now()
    _assert_digest(_post(proxy(shape_output="aggressive"))[0])


def test_release_touches_only_the_drift_state(home, capsys):
    """A false trip must be releasable without wiping the statusline/leaderboard totals."""
    (home / "savings.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    (home / "shadow.jsonl").write_text("", encoding="utf-8")
    drift.fold([-1] * 200)
    _release()
    assert (home / "savings.jsonl").read_text(encoding="utf-8") == '{"x": 1}\n'
    assert (home / "shadow.jsonl").exists()
    assert not drift.held_now()
    assert "no restart needed" in capsys.readouterr().out


def test_a_running_proxy_notices_another_process_trip_and_release(home, proxy, monkeypatch):
    """A long-lived launch-agent proxy must not keep digesting after another process
    trips, nor stay held after a release — its watcher stats the state file."""
    monkeypatch.setattr(drift.DriftGuard, "POLL_S", 0.05)
    port = proxy(shape_output="aggressive")
    _assert_digest(_post(port)[0])

    drift.fold([-1] * 200)  # another proxy's verdicts trip the shared e-process
    deadline = time.time() + 10
    while time.time() < deadline:
        h = _post(port)[0]
        if h.get("x-distil-drift-guard") == "held":
            break
        time.sleep(0.05)
    _assert_held(h)

    _release()
    deadline = time.time() + 10
    while time.time() < deadline:
        h = _post(port)[0]
        if "x-distil-drift-guard" not in h:
            break
        time.sleep(0.05)
    _assert_digest(h)


def test_two_processes_crossing_together_write_one_receipt(home):
    """The trip and its receipt are one critical section — no double-trip."""
    a = drift.DriftGuard.start(watch=False)
    b = drift.DriftGuard.start(watch=False)
    for _ in range(200):
        a.observe(-1)
        b.observe(-1)
    assert a.engaged and b.engaged
    assert len([r for r in R.read() if r.mode == "drift-trip"]) == 1


def test_live_shadow_verdicts_trip_the_guard_end_to_end(home, proxy, monkeypatch):
    """No test hooks: shadow replays the request, the stub model changes its tool call
    when the log is digested, and the paired rows alone must trip the hold."""
    guards = _capture_guard(monkeypatch)
    port = proxy(shape_output="aggressive", shadow_rate=1.0)
    deadline = time.time() + 60
    while not guards[0].engaged and time.time() < deadline:
        _post(port)
    assert guards[0].engaged, "paired shadow harm never reached the guard"
    _assert_held(_post(port)[0])


def test_the_status_line_shows_the_hold_and_its_release_command(home, capsys):
    import argparse

    from distil.cli import cmd_statusline

    args = argparse.Namespace(no_color=True)
    cmd_statusline(args)
    assert "drift hold" not in capsys.readouterr().out
    drift.fold([-1] * 200)
    cmd_statusline(args)
    assert "⚠ drift hold · distil reset --drift-guard" in capsys.readouterr().out


def test_opt_out_keeps_compressing_and_says_so(home, proxy, monkeypatch):
    drift.fold([-1] * 200)
    monkeypatch.setenv(drift.GUARD_OPT_OUT, "1")
    _assert_digest(_post(proxy(shape_output="aggressive"))[0])
    assert not drift.DriftGuard.start(watch=False).engaged
    assert not drift.held_now()
    assert "compression was NOT held" in drift.LiveDrift.load().line()


# --- fail-open ---------------------------------------------------------------


def test_a_guard_that_raises_never_breaks_a_request(home, proxy, monkeypatch):
    class _Broken:
        @property
        def engaged(self) -> bool:
            raise RuntimeError("alarm exploded")

        def observe(self, diff: int) -> None:
            raise RuntimeError("alarm exploded")

    monkeypatch.setattr(drift.DriftGuard, "start", staticmethod(lambda **kw: _Broken()))
    port = proxy(shape_output="aggressive", shadow_rate=1.0)
    for _ in range(3):  # and again after the shadow thread's observe() has raised
        h, body = _post(port)
        assert b"tool_use" in body
        # Served exactly as configured. Headers, not the stub's last body: shadow's
        # replays of the ORIGINAL request race onto the same upstream.
        assert h["x-distil-mode"] == "digest" and "x-distil-drift-guard" not in h


def test_observe_and_refresh_swallow_their_own_errors(home, monkeypatch):
    g = drift.DriftGuard.start(watch=False)
    monkeypatch.setattr(drift, "fold", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(drift.LiveDrift, "load", classmethod(lambda cls, path=None: 1 / 0))
    g.observe(-1)  # must not raise
    g._seen = (0, 0)
    g.refresh()  # must not raise
    assert not g.engaged


def test_a_start_that_cannot_load_state_starts_unheld(home, monkeypatch):
    monkeypatch.setattr(drift, "_bootstrap", lambda *a, **k: 1 / 0)
    assert not drift.DriftGuard.start(watch=False).engaged
