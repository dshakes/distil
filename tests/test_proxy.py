"""Integration tests for distil.proxy — no external network required.

Architecture
------------
* A fake upstream ``ThreadingHTTPServer`` binds to an ephemeral port on
  127.0.0.1.  For POST requests it reads the JSON body and echoes it back
  verbatim so tests can inspect exactly what the proxy forwarded.  For other
  methods it echoes the path back as plain text.
* The distil proxy is started (also on an ephemeral port) pointing at the
  fake upstream.
* Tests use ``urllib.request`` as the HTTP client — stdlib only, no network.
* Both servers are shut down cleanly in teardown.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from distil.proxy import build_handler

# ---------------------------------------------------------------------------
# Fake upstream server
# ---------------------------------------------------------------------------

_LONG_TOOL_RESULT = "\n".join(
    [
        "Result from bash tool execution on the remote host:",
        "total disk usage: 48 GB across 12 partitions",
        "filesystem /dev/sda1: 32 GB used of 100 GB available",
        "filesystem /dev/sdb1: 16 GB used of 200 GB available",
        "warning: /tmp is 89% full — consider cleaning up old build artefacts",
        "warning: inode count on /var/log approaching limit (91% used)",
        "no errors detected in kernel ring buffer",
        "last boot: 2026-06-20T03:14:22Z (uptime 18h 42m)",
        "load averages: 0.23 0.31 0.29 (1m/5m/15m)",
        "memory: 14.2 GB used / 31.9 GB total, 0 GB swap",
        "top process: python3 pid=8821 cpu=4.1% mem=2.3%",
        "all health checks passed",
    ]
)  # 12 lines — well above the 6-line digest threshold


class _EchoHandler(BaseHTTPRequestHandler):
    """Fake upstream: echo POST body as JSON response; echo path for other verbs."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _echo_path(self) -> None:
        resp = self.path.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    do_GET = _echo_path  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pytest fixture: both servers, torn down after each test
# ---------------------------------------------------------------------------


@pytest.fixture()
def servers() -> Any:
    """Yield (proxy_port, upstream_port); shut both down after the test."""
    # 1. Fake upstream on ephemeral port
    upstream_server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    upstream_port = upstream_server.server_address[1]
    upstream_thread = threading.Thread(target=upstream_server.serve_forever, daemon=True)
    upstream_thread.start()

    # 2. Proxy pointed at fake upstream, also on ephemeral port
    upstream_url = f"http://127.0.0.1:{upstream_port}"
    handler_cls = build_handler(upstream_url)
    proxy_server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    proxy_port = proxy_server.server_address[1]
    proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    proxy_thread.start()

    yield proxy_port, upstream_port

    proxy_server.shutdown()
    upstream_server.shutdown()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _post(port: int, path: str, payload: dict[str, Any]) -> urllib.request.Request:
    """Return an opened urllib response for a POST to proxy."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        method="POST",
    )
    return req


# ---------------------------------------------------------------------------
# Test 1: compressible path — tool_result digested, headers set
# ---------------------------------------------------------------------------


def test_compressible_path_digests_tool_result(servers: Any) -> None:
    proxy_port, _ = servers

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": _LONG_TOOL_RESULT,
                }
            ],
        },
        # Two later turns keep the tool_result out of the recency-exempt window
        # (adapter keeps the most recent K turns verbatim) so it still digests.
        {"role": "user", "content": "next"},
        {"role": "user", "content": "next"},
    ]
    payload = {"model": "claude-opus-4-5", "max_tokens": 256, "messages": messages}

    req = _post(proxy_port, "/v1/messages", payload)
    with urllib.request.urlopen(req) as resp:
        status = resp.status
        compressed_header = resp.headers.get("x-distil-compressed")
        tokens_saved_header = resp.headers.get("x-distil-tokens-saved")
        echoed: dict[str, Any] = json.loads(resp.read())

    # Proxy returned 200
    assert status == 200, f"Expected 200, got {status}"

    # distil headers present
    assert compressed_header == "1", f"x-distil-compressed missing or wrong: {compressed_header!r}"
    assert tokens_saved_header is not None, "x-distil-tokens-saved header missing"
    assert int(tokens_saved_header) > 0, (
        f"x-distil-tokens-saved should be positive, got {tokens_saved_header!r}"
    )

    # The forwarded body shows the tool_result was digested
    forwarded_content = echoed["messages"][0]["content"][0]["content"]
    assert "handle=" in forwarded_content, (
        f"Expected digest marker in forwarded content, got: {forwarded_content!r}"
    )
    assert len(forwarded_content) < len(_LONG_TOOL_RESULT), (
        "Forwarded tool_result should be shorter than original after digest"
    )


# ---------------------------------------------------------------------------
# Test 2: non-compressible path — body forwarded unchanged
# ---------------------------------------------------------------------------


def test_non_compressible_path_forwarded_unchanged(servers: Any) -> None:
    proxy_port, _ = servers

    # /v1/models is not in _COMPRESSIBLE_PATHS — should pass through as-is.
    payload = {"some_key": "some_value", "messages": [{"role": "user", "content": "hi"}]}

    req = _post(proxy_port, "/v1/models", payload)
    with urllib.request.urlopen(req) as resp:
        status = resp.status
        compressed_header = resp.headers.get("x-distil-compressed")
        echoed: dict[str, Any] = json.loads(resp.read())

    assert status == 200
    # No distil compression headers on non-compressible paths
    assert compressed_header is None, (
        f"x-distil-compressed should be absent for non-compressible path, got {compressed_header!r}"
    )
    # Body forwarded byte-for-byte (the fake upstream echoes exactly what it received)
    assert echoed == payload, f"Body should be unchanged. Got: {echoed!r}"


def test_drain_shadow_waits_for_inflight_thread() -> None:
    """The shadow-mode fix: a sampled comparison runs in a daemon thread; teardown
    must drain it so the sample isn't lost when the process exits (e.g. claude -p)."""
    import threading
    import time

    from distil.proxy import _drain_shadow

    done: list[int] = []

    def work() -> None:
        time.sleep(0.2)
        done.append(1)

    t = threading.Thread(target=work, daemon=True)

    class _H:
        shadow_threads = [t]

    t.start()
    _drain_shadow(_H, budget=3.0)
    assert done == [1]  # drain waited for the in-flight comparison to record


def test_drain_shadow_is_bounded_by_budget() -> None:
    """A hung upstream must not block teardown: drain returns within ~budget."""
    import threading
    import time

    from distil.proxy import _drain_shadow

    t = threading.Thread(target=lambda: time.sleep(5), daemon=True)

    class _H:
        shadow_threads = [t]

    t.start()
    start = time.monotonic()
    _drain_shadow(_H, budget=0.3)
    assert time.monotonic() - start < 2.0  # did not wait the full 5s


def test_quiet_server_swallows_client_disconnects(capsys) -> None:
    """Client resets (agent cancels a stream) must not traceback-spam stderr."""
    from distil.proxy import QuietHTTPServer

    srv = QuietHTTPServer.__new__(QuietHTTPServer)  # no bind needed for handle_error
    try:
        raise ConnectionResetError(54, "Connection reset by peer")
    except ConnectionResetError:
        srv.handle_error(None, ("127.0.0.1", 12345))
    assert capsys.readouterr().err == ""

    try:
        raise ValueError("real bug")
    except ValueError:
        srv.handle_error(None, ("127.0.0.1", 12345))
    assert "ValueError" in capsys.readouterr().err  # real errors still surface


def test_build_handler_eager_loads_request_path_modules() -> None:
    """FIX 4a: request-path modules imported lazily per request must be warmed at
    server setup, so an in-place upgrade never loads a post-upgrade file mid-serve."""
    import sys

    for m in ("distil.streamrelay", "distil.compress.guideline"):
        sys.modules.pop(m, None)
    build_handler("http://127.0.0.1:1")
    assert "distil.streamrelay" in sys.modules
    assert "distil.compress.guideline" in sys.modules


# ---------------------------------------------------------------------------
# Health endpoint: answers locally, never hits upstream
# ---------------------------------------------------------------------------


def test_health_endpoint_answers_locally(servers: Any) -> None:
    proxy_port, _ = servers
    with urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/distil/health") as resp:
        assert resp.status == 200
        assert json.loads(resp.read()) == {"status": "ok"}
    # A non-health GET still passes through to the upstream (echoes its path).
    with urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/v1/models") as resp:
        assert resp.read() == b"/v1/models"


# --- byte-faithful passthrough ------------------------------------------------
def test_unchanged_body_forwards_the_original_bytes():
    """An untouched body must reach upstream byte-identical.

    The prompt cache matches on exact bytes. ``json.dumps`` is not a byte-faithful
    round-trip — it rewrites separators and escapes non-ASCII — so re-encoding an
    unmodified body rewrites the cached prefix and converts cheap cache reads into
    expensive cache writes while saving nothing. Measured externally as 1.56x
    baseline cache-creation tokens at 0.0% savings in lossless-only mode.
    """
    import json as _json

    from distil.proxy import _serialize_if_changed

    raw = (
        b'{"model": "m", "max_tokens": 1024, '
        b'"messages": [{"role": "user", "content": "caf\xc3\xa9 \xe2\x80\x94 na\xc3\xafve"}]}'
    )
    body = _json.loads(raw)
    assert _serialize_if_changed(raw, body) is raw or _serialize_if_changed(raw, body) == raw


def test_changed_body_is_reserialized_compactly_and_unescaped():
    """When a transform did change the body, re-serialize without \\uXXXX escapes."""
    import json as _json

    from distil.proxy import _serialize_if_changed

    raw = b'{"messages": [{"role": "user", "content": "caf\xc3\xa9"}]}'
    body = _json.loads(raw)
    body["messages"][0]["content"] = "café compressed"
    out = _serialize_if_changed(raw, body)
    assert out != raw
    assert b"\\u" not in out, "non-ASCII must stay raw UTF-8, not be escaped"
    assert b", " not in out.replace(b'", "', b""), "compact separators"
    assert _json.loads(out)["messages"][0]["content"] == "café compressed"


def test_unparseable_original_still_serializes():
    """A body we could not re-parse must not crash the request path."""
    from distil.proxy import _serialize_if_changed

    out = _serialize_if_changed(b"not json at all", {"a": 1})
    assert out == b'{"a":1}'


def test_expand_tool_is_injected_before_any_handle_exists():
    """The tools array must not change shape mid-session.

    Anthropic caches the tools array at the very FRONT of the prefix — ahead of the
    system prompt and all history. Injecting distil_expand only once a handle exists
    means the array gains an entry on the turn compression first fires, invalidating
    the whole cached entry at exactly the moment savings begin. Worse, distil's own
    drift report then blames "a tool list whose order varies" upstream — which was
    ours. Session-sticky injection trades one tool definition on early turns for a
    byte-stable prefix all session.
    """
    from distil.expand import EXPAND_TOOL_NAME, inject_expand_tool

    # An empty store + no stubs: the old gate would NOT have injected here.
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": []}
    out = inject_expand_tool(body)
    assert any(t["name"] == EXPAND_TOOL_NAME for t in out["tools"])

    # And it must stay idempotent, so a later turn produces the same array.
    again = inject_expand_tool(out)
    assert sum(t["name"] == EXPAND_TOOL_NAME for t in again["tools"]) == 1
    assert again["tools"] == out["tools"], "the tools array must be byte-stable across turns"


# --- session survival ---------------------------------------------------------
# A proxy sits in the request path of a live agent session. Anything it does badly
# ends someone's work, so the contract is narrow: distil may fail to COMPRESS, but
# it must never fail to SERVE. These drive the real handler over a real socket, so
# they fail if any layer between the client and upstream drops the request.
def test_a_crash_inside_compression_still_serves_the_request(
    servers: Any, monkeypatch: Any
) -> None:
    """If distil's own compressor raises, the turn must still complete uncompressed.

    This is the difference between "we saved fewer tokens today" and "the user's
    session died". A guarded try/except in the source is a claim; this drives the
    real request path over a socket to make it a measurement.
    """
    import distil.adapters.anthropic as an

    def _boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("synthetic compressor failure")

    # Fault-inject inside the adapter the proxy actually calls. Patching the proxy's
    # module-level alias would not reach the handler, which resolved it at import.
    monkeypatch.setattr(an, "_compress_tool_result_text", _boom)

    proxy_port, _ = servers
    payload = {
        "model": "claude-opus-4-5",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": _LONG_TOOL_RESULT}
                ],
            },
            {"role": "user", "content": "next"},
            {"role": "user", "content": "next"},
        ],
    }
    with urllib.request.urlopen(_post(proxy_port, "/v1/messages", payload)) as resp:
        assert resp.status == 200, "a compressor crash must not fail the request"
        echoed = json.loads(resp.read())
    # The full conversation must reach upstream. Whether this particular turn ended
    # up compressed is not the point — no message may be lost, and the turn must
    # complete. Asserting byte-verbatim content would over-specify: other digest
    # paths remain live, and any of them producing a RECOVERABLE digest is fine.
    assert len(echoed["messages"]) == 3, "no message may be dropped by a failure"
    assert echoed["messages"][1]["content"] == "next"
    assert echoed["messages"][2]["content"] == "next"


def test_a_crash_in_token_counting_still_serves_the_request(servers: Any, monkeypatch: Any) -> None:
    """Accounting is bookkeeping. It must never be load-bearing for the response."""
    import distil.proxy as px

    def _boom(*_a: Any, **_kw: Any) -> int:
        raise RuntimeError("synthetic counter failure")

    monkeypatch.setattr(px, "_count_messages", _boom, raising=True)

    proxy_port, _ = servers
    payload = {
        "model": "claude-opus-4-5",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    with urllib.request.urlopen(_post(proxy_port, "/v1/messages", payload)) as resp:
        assert resp.status == 200
        assert json.loads(resp.read())["messages"][0]["content"] == "hello"


def test_a_malformed_body_is_forwarded_rather_than_rejected(servers: Any) -> None:
    """distil is not the API's validator. A body it cannot parse goes upstream as-is,
    so the provider's own error reaches the agent instead of distil inventing one."""
    proxy_port, _ = servers
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}/v1/messages",
        data=b"{not valid json at all",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, "the echo upstream accepted it — distil did not block"
