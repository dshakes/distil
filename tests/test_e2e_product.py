"""End-to-end product tests — the whole chain, not a unit at a time.

Unit tests kept passing while real traffic broke, three separate times in one day.
Every test here therefore exercises a *user-visible path* through real sockets and
real files, and asserts the thing the user would actually notice:

  1. a request through a live proxy is compressed, billed, receipted, and the
     elided bytes come back through the public library API in another "process"
  2. the four savings surfaces never show a broken-looking 0%
  3. a settings.json pin at a dead port is refused before it can cause an outage
  4. the doctor agrees with reality in both directions

Reuses the echo-upstream harness from test_runtime rather than inventing a second
one — a divergent second harness is how the first one stops being trusted.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from distil import ledger
from distil.runtime import RuntimeSavings
from tests.test_runtime import _start_echo_upstream, build_handler

BIG_TOOL_OUTPUT = "DECISION: keep\n" + "\n".join(
    f"2026-08-07T12:00:{i:02d}Z INFO worker heartbeat seq={i} latency_ms=12" for i in range(300)
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))


@pytest.fixture
def proxy(tmp_path):
    """A real distil proxy in front of a real echo upstream."""
    upstream = _start_echo_upstream()
    up_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=tmp_path / "savings.jsonl")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(up_url, savings=rs, flush_every=1))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        upstream.shutdown()


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url + "/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers)


def _conversation() -> dict:
    return {
        "model": "claude-opus-4-8",
        "messages": [
            {"role": "user", "content": "why did the worker pool die?"},
            {"role": "user", "content": [{"type": "tool_result", "content": BIG_TOOL_OUTPUT}]},
            {"role": "user", "content": "next"},
            {"role": "user", "content": "next"},
        ],
    }


def test_request_is_compressed_billed_and_receipted(proxy, tmp_path):
    """One real request must move every surface a user checks."""
    status, headers = _post(proxy, _conversation())
    assert status == 200
    assert int(headers.get("x-distil-tokens-saved", "0")) > 0, "header must report savings"

    summary = ledger.summary(tmp_path / "savings.jsonl")
    assert summary.total_tokens_saved > 0, "savings ledger must record genuine savings"

    from distil import receipts

    verdict = receipts.verify()
    assert verdict.ok, f"receipt chain must verify: {verdict.statement}"
    assert verdict.total >= 1, "every request writes a receipt, regardless of savings"


def test_digested_bytes_survive_into_a_separate_process(proxy, tmp_path):
    """The reversibility claim, tested the way a user would experience it.

    Compress via the proxy, then recover the original from a *different* Python
    process through the public API. In-process recovery would not prove the
    on-disk restore store works, which is the property the promise rests on.
    """
    _post(proxy, _conversation())

    handles = (
        [p.name for p in (tmp_path / "restore").iterdir()]
        if (tmp_path / "restore").exists()
        else []
    )
    assert handles, "a 300-line tool result must leave at least one restore handle"

    code = (
        "import sys; from distil import expand_handle;"
        "out = expand_handle(sys.argv[1]);"
        "sys.exit(0 if out and 'worker heartbeat' in out else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, handles[0]],
        capture_output=True,
        text=True,
        env={"DISTIL_HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        f"a handle written by the proxy did not expand in a fresh process: {result.stderr}"
    )


def test_wrap_refuses_a_config_that_would_take_the_agent_down(tmp_path, monkeypatch):
    """The three-outages-in-a-day regression, end to end through the real CLI."""
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()

    proj = tmp_path / "proj"
    (proj / ".claude").mkdir(parents=True)
    (proj / ".claude" / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{dead_port}"}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "distil.cli", "wrap", "--", "claude", "-p", "hi"],
        cwd=proj,
        capture_output=True,
        text=True,
        env={"DISTIL_HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode == 1, "wrap must refuse, not start into a guaranteed outage"
    assert "NOT accepting connections" in result.stderr
    assert "distil offboard" in result.stderr, "must name the fix, not just the fault"


def test_no_savings_surface_can_render_a_bare_zero():
    """A real number next to a 0% is the single most likely uninstall trigger."""
    from distil.ledger import pct_smaller_label

    for fraction in (0.0001, 0.004, 0.009):
        label = pct_smaller_label(fraction)
        assert label == "<1% smaller", f"{fraction} rendered as {label!r}"
    assert pct_smaller_label(0.0) == "0% smaller", "genuinely zero may say zero"


def test_doctor_agrees_with_reality_in_both_directions(tmp_path, monkeypatch):
    """Doctor must not cry wolf, and must not stay silent when routing is dead."""
    import subprocess as sp

    from distil import doctor, ledger as _ledger

    monkeypatch.setattr(
        sp, "run", lambda *a, **k: type("R", (), {"stdout": "distil proxy --port 8788"})()
    )
    monkeypatch.setattr(_ledger, "latest_session", lambda *a, **k: ("", 0.0))

    # Requests are arriving (fresh receipts) but nothing compressed well enough to
    # be ledgered — the always-on false alarm.
    (tmp_path / "receipts.jsonl").write_text('{"v":1}\n', encoding="utf-8")
    assert doctor._check_live_routing().status == "ok"

    # Nothing arriving at all — the check must keep its teeth.
    (tmp_path / "receipts.jsonl").unlink()
    assert doctor._check_live_routing().status == "warn"


def test_sigpipe_disposition_does_not_leak_out_of_a_cli_call():
    """`cli.main()` sets SIGPIPE to SIG_DFL process-wide; the suite must not inherit it.

    Without the conftest guard, the first test that runs a CLI entry point turns
    every later broken pipe into a fatal signal, and pytest dies with exit 141 —
    no failing test, no traceback, just a run that stops. It reproduced only on
    some Python versions because it depends on execution order.
    """
    import signal

    sigpipe = getattr(signal, "SIGPIPE", None)
    if sigpipe is None:
        pytest.skip("no SIGPIPE on this platform")
    assert signal.getsignal(sigpipe) != signal.SIG_DFL, (
        "SIGPIPE is SIG_DFL inside the test process — a broken pipe will kill pytest"
    )
