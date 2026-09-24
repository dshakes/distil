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

    def start() -> drift.DriftGuard:
        g = real()
        made.append(g)
        return g

    monkeypatch.setattr(drift.DriftGuard, "start", staticmethod(start))
    return made


def _trip(g: drift.DriftGuard) -> None:
    for _ in range(200):
        g.observe(-1)
        if g.trip is not None:
            return
    raise AssertionError("sustained harm never tripped the guard")


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
    assert drift.read_trip() is None


def test_breach_holds_the_very_next_request_lossless_only(home, proxy, monkeypatch):
    guards = _capture_guard(monkeypatch)
    port = proxy(shape_output="aggressive")
    _assert_digest(_post(port)[0])

    _trip(guards[0])
    _assert_held(_post(port)[0])

    # ...recorded where it can be audited: a trip file and one receipt on the chain.
    trip = drift.read_trip()
    assert trip is not None and trip["source"] == "proxy" and trip["evalue"] >= 20
    rows = [r for r in R.read() if r.mode == "drift-trip"]
    assert len(rows) == 1 and "lossless-only" in rows[0].certificate
    assert R.verify().ok


def test_the_hold_survives_a_proxy_restart_until_reset(home, proxy, monkeypatch):
    guards = _capture_guard(monkeypatch)
    proxy(shape_output="aggressive")
    _trip(guards[0])

    restarted = proxy(shape_output="aggressive")  # a fresh process reads the trip file
    _assert_held(_post(restarted)[0])

    import argparse

    from distil.cli import cmd_reset

    cmd_reset(argparse.Namespace(shadow=True))
    assert drift.read_trip() is None
    _assert_digest(_post(proxy(shape_output="aggressive"))[0])


def test_live_shadow_verdicts_trip_the_guard_end_to_end(home, proxy):
    """No test hooks: shadow replays the request, the stub model changes its tool call
    when the log is digested, and the paired rows alone must trip the hold."""
    port = proxy(shape_output="aggressive", shadow_rate=1.0)
    deadline = time.time() + 60
    while drift.read_trip() is None and time.time() < deadline:
        _post(port)
        time.sleep(0.05)
    assert drift.read_trip() is not None, "paired shadow harm never reached the guard"
    _assert_held(_post(port)[0])


def test_a_breach_found_at_exit_arms_the_next_session(home):
    """The wrap-exit fold (distil stats / proof ledger) must arm the guard too, or a
    breach discovered after the proxy stopped would be resumed on the next start."""
    from distil.proof_ledger import proof_lines
    from distil.shadow import SIG_VERSION

    with (home / "shadow.jsonl").open("w", encoding="utf-8") as f:
        for _ in range(300):
            row = {"equivalent": False, "aa_equal": True, "ts": time.time()}
            f.write(json.dumps({**row, "kind": "paired", "sig": SIG_VERSION}) + "\n")
    assert "BREACHED" in dict(proof_lines())["budget"]
    assert (drift.read_trip() or {}).get("source") == "ledger"
    assert drift.DriftGuard.start().engaged


def test_opt_out_keeps_compressing_and_says_so(home, proxy, monkeypatch):
    drift.arm(99.0, 60, source="test")
    monkeypatch.setenv(drift.GUARD_OPT_OUT, "1")
    _assert_digest(_post(proxy(shape_output="aggressive"))[0])
    assert not drift.DriftGuard.start().engaged
    line = drift.LiveDrift.load().line(None, drift.read_trip())
    assert "compression was NOT held" in line


def test_an_unreadable_trip_file_holds_rather_than_resumes(home):
    (home / "drift-trip.json").write_text("{torn", encoding="utf-8")
    assert drift.DriftGuard.start().engaged


# --- fail-open ---------------------------------------------------------------


def test_a_guard_that_raises_never_breaks_a_request(home, proxy, monkeypatch):
    class _Broken:
        @property
        def engaged(self) -> bool:
            raise RuntimeError("alarm exploded")

        def observe(self, diff: int) -> None:
            raise RuntimeError("alarm exploded")

    monkeypatch.setattr(drift.DriftGuard, "start", staticmethod(lambda: _Broken()))
    port = proxy(shape_output="aggressive", shadow_rate=1.0)
    for _ in range(3):  # and again after the shadow thread's observe() has raised
        h, body = _post(port)
        assert b"tool_use" in body
        # Served exactly as configured. Headers, not the stub's last body: shadow's
        # replays of the ORIGINAL request race onto the same upstream.
        assert h["x-distil-mode"] == "digest" and "x-distil-drift-guard" not in h


def test_observe_swallows_its_own_errors(home, monkeypatch):
    g = drift.DriftGuard.start()
    monkeypatch.setattr(
        drift.DriftMonitor, "update", lambda self, x: (_ for _ in ()).throw(ValueError("x"))
    )
    g.observe(-1)  # must not raise
    assert not g.engaged


def test_a_start_that_cannot_load_state_starts_fresh(home, monkeypatch):
    monkeypatch.setattr(drift.LiveDrift, "load", classmethod(lambda cls, path=None: 1 / 0))
    g = drift.DriftGuard.start()
    assert not g.engaged and g.monitor.n == 0


def test_an_unwritable_home_still_holds_this_session(home, monkeypatch):
    """Persistence is best-effort; the in-memory hold is not."""
    monkeypatch.setattr(drift, "arm", lambda *a, **k: None)
    g = drift.DriftGuard.start()
    _trip(g)
    assert g.engaged
