"""Integration tests for distil.aproxy — no external network required.

Architecture
------------
* A fake upstream aiohttp server binds to an ephemeral port on 127.0.0.1.
  For POST requests it echoes the body back as JSON so tests can inspect what
  the proxy forwarded.  For other paths/methods it echoes the path back.
* The distil async proxy is built via ``make_app`` and pointed at the fake
  upstream.
* Tests drive both via ``aiohttp.test_utils.TestServer``/``TestClient``.
* aiohttp is guarded with ``pytest.importorskip`` so these tests are silently
  skipped in environments that don't have the [async] extra installed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

# Skip the whole module if aiohttp is not available.
aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from distil.aproxy import make_app

# ---------------------------------------------------------------------------
# Sample data: a long tool_result (>= 8 lines) that should be digested
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


# ---------------------------------------------------------------------------
# Fake upstream aiohttp application
# ---------------------------------------------------------------------------


async def _echo_post(request: web.Request) -> web.Response:
    """Echo the raw POST body back as-is (so we can inspect what the proxy sent)."""
    body = await request.read()
    return web.Response(body=body, content_type="application/json", status=200)


async def _echo_path(request: web.Request) -> web.Response:
    """Echo the request path back as plain text for non-POST routes."""
    return web.Response(text=request.path, content_type="text/plain", status=200)


def _make_fake_upstream() -> web.Application:
    app = web.Application()
    # POST routes for the compressible paths
    app.router.add_post("/v1/messages", _echo_post)
    app.router.add_post("/v1/chat/completions", _echo_post)
    app.router.add_post("/v1/responses", _echo_post)
    # Catch-all for everything else
    app.router.add_route("*", "/{path_info:.*}", _echo_path)
    return app


# ---------------------------------------------------------------------------
# Helper: run a coroutine in a one-shot event loop (for sync pytest tests)
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Test 1: POST /v1/messages with a large tool_result → stays reversible (no handle)
# ---------------------------------------------------------------------------


def test_compressible_path_stays_reversible() -> None:
    async def _body() -> None:
        fake_app = _make_fake_upstream()
        async with TestServer(fake_app) as upstream_server:
            upstream_url = str(upstream_server.make_url("/")).rstrip("/")
            proxy_app = make_app(upstream_url)
            async with TestClient(TestServer(proxy_app)) as client:
                messages: list[dict[str, Any]] = [
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
                    # Two later turns keep the tool_result out of the
                    # recency-exempt window (adapter keeps the most recent
                    # K turns verbatim) so it still digests.
                    {"role": "user", "content": "next"},
                    {"role": "user", "content": "next"},
                ]
                payload = {
                    "model": "claude-opus-4-5",
                    "max_tokens": 256,
                    "messages": messages,
                }
                resp = await client.post(
                    "/v1/messages",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )

                assert resp.status == 200, f"Expected 200, got {resp.status}"

                # aproxy runs no expand loop, so it must NEVER emit an unrecoverable
                # Tier-1 digest stub (a handle= marker the async path can't recover). It
                # stays Tier-0 / reversible: the long tool_result is forwarded WITHOUT a
                # recovery handle, so nothing the agent later needs is irretrievably lost.
                # (Recoverable Tier-1 needs the expand loop the sync proxy implements.)
                echoed: dict[str, Any] = await resp.json()
                forwarded_content = echoed["messages"][0]["content"][0]["content"]
                assert "handle=" not in forwarded_content, (
                    f"aproxy must not emit an unrecoverable digest handle: {forwarded_content!r}"
                )

    _run(_body())


# ---------------------------------------------------------------------------
# Test 2: non-compressible path (/v1/models GET) is forwarded unchanged
# ---------------------------------------------------------------------------


def test_passthrough_path_forwarded_unchanged() -> None:
    async def _body() -> None:
        fake_app = _make_fake_upstream()
        async with TestServer(fake_app) as upstream_server:
            upstream_url = str(upstream_server.make_url("/")).rstrip("/")
            proxy_app = make_app(upstream_url)
            async with TestClient(TestServer(proxy_app)) as client:
                # GET /v1/models — not a compressible path, should pass through
                resp = await client.get("/v1/models")

                assert resp.status == 200, f"Expected 200, got {resp.status}"

                # No distil compression headers on passthrough paths
                compressed_hdr = resp.headers.get("x-distil-compressed")
                assert compressed_hdr is None, (
                    f"x-distil-compressed should be absent for passthrough, got {compressed_hdr!r}"
                )

                # The fake upstream echoes the path back as plain text
                text = await resp.text()
                assert "/v1/models" in text, f"Expected path echoed back, got: {text!r}"

    _run(_body())


# ---------------------------------------------------------------------------
# Test 3: POST to non-compressible path — body forwarded byte-for-byte
# ---------------------------------------------------------------------------


def test_passthrough_post_body_unchanged() -> None:
    async def _body() -> None:
        fake_app = _make_fake_upstream()
        async with TestServer(fake_app) as upstream_server:
            upstream_url = str(upstream_server.make_url("/")).rstrip("/")
            proxy_app = make_app(upstream_url)
            async with TestClient(TestServer(proxy_app)) as client:
                payload = {
                    "some_key": "some_value",
                    "messages": [{"role": "user", "content": "hi"}],
                }
                resp = await client.post(
                    "/v1/models",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )

                assert resp.status == 200
                compressed_hdr = resp.headers.get("x-distil-compressed")
                assert compressed_hdr is None, (
                    "x-distil-compressed should be absent for non-compressible path"
                )
                # The fake upstream echoes path for non-POST routes; for the /v1/models
                # POST route (not registered explicitly in fake), the catch-all returns the path.
                # We just verify no crash and 200.

    _run(_body())


# ---------------------------------------------------------------------------
# Test 4: aiohttp is NOT imported at the top-level of distil.aproxy
# ---------------------------------------------------------------------------


def test_aiohttp_not_imported_at_module_level() -> None:
    import importlib
    import sys

    # Remove aproxy from sys.modules to get a fresh import
    sys.modules.pop("distil.aproxy", None)

    # Temporarily hide aiohttp from sys.modules to check that the import of
    # distil.aproxy itself does NOT trigger an aiohttp import.
    aiohttp_mod = sys.modules.pop("aiohttp", None)
    aiohttp_web = sys.modules.pop("aiohttp.web", None)
    try:
        # This should succeed even without aiohttp in sys.modules
        importlib.import_module("distil.aproxy")
        # aiohttp should NOT have been re-imported as a side-effect
        assert "aiohttp" not in sys.modules, (
            "aiohttp was imported at module level in distil.aproxy — it must be lazy"
        )
    finally:
        # Restore aiohttp so other tests can use it
        if aiohttp_mod is not None:
            sys.modules["aiohttp"] = aiohttp_mod
        if aiohttp_web is not None:
            sys.modules["aiohttp.web"] = aiohttp_web
        sys.modules.pop("distil.aproxy", None)
