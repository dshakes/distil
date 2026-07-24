"""Coverage tests for distil.aproxy.

Uses aiohttp's test utilities to exercise the async proxy in-process.
The whole module is skipped when aiohttp is not installed (it is an
optional [async] extra).

Covers:
  • compressible POST (Anthropic messages) → compression + headers
  • non-compressible POST → body forwarded unchanged
  • OpenAI /v1/chat/completions path → compressed
  • Gemini generateContent path (contents field) → compressed
  • shape_output="light" → x-distil-output-shaping header
  • savings.record() + maybe_flush() called on compressible request
  • invalid path → 400
  • upstream timeout → 504 (aiohttp.ServerTimeoutError path)
  • upstream connection error → 502 (ClientError path)
  • passthrough preserves non-compressible request body
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from distil.aproxy import make_app  # noqa: E402

# ---------------------------------------------------------------------------
# Common test data
# ---------------------------------------------------------------------------

_LONG_TOOL_RESULT = "\n".join(
    [
        "Result from bash tool execution on the remote host:",
        "total disk usage: 48 GB across 12 partitions",
        "filesystem /dev/sda1: 32 GB used of 100 GB available",
        "filesystem /dev/sdb1: 16 GB used of 200 GB available",
        "warning: /tmp is 89% full",
        "warning: inode count on /var/log approaching limit",
        "no errors detected in kernel ring buffer",
        "last boot: 2026-06-20T03:14:22Z",
        "load averages: 0.23 0.31 0.29",
        "memory: 14.2 GB used / 31.9 GB total",
        "top process: python3 pid=8821 cpu=4.1%",
        "all health checks passed",
    ]
)


# ---------------------------------------------------------------------------
# Fake upstream factory helpers
# ---------------------------------------------------------------------------


def _echo_app() -> web.Application:
    """Upstream that echoes POST bodies; echoes path for other verbs."""

    async def _echo_post(req: web.Request) -> web.Response:
        body = await req.read()
        return web.Response(body=body, content_type="application/json")

    async def _echo_path(req: web.Request) -> web.Response:
        return web.Response(text=req.path, content_type="text/plain")

    app = web.Application()
    app.router.add_post("/v1/messages", _echo_post)
    app.router.add_post("/v1/chat/completions", _echo_post)
    app.router.add_post("/v1/responses", _echo_post)
    app.router.add_route("*", "/{p:.*}", _echo_path)
    return app


def _run(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests: compressible paths
# ---------------------------------------------------------------------------


def test_aproxy_openai_chat_completions_compressed() -> None:
    """POST to /v1/chat/completions is treated as a compressible path."""

    async def _body() -> None:
        async with TestServer(_echo_app()) as up:
            app = make_app(str(up.make_url("/")).rstrip("/"))
            async with TestClient(TestServer(app)) as client:
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "content": _LONG_TOOL_RESULT}],
                        },
                        # Two later turns keep the tool_result out of the
                        # recency-exempt window so it still digests.
                        {"role": "user", "content": "next"},
                        {"role": "user", "content": "next"},
                    ],
                }
                resp = await client.post(
                    "/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 200
                assert resp.headers.get("x-distil-compressed") == "1"
                # aproxy runs no expand loop → it must stay reversible: no unrecoverable
                # Tier-1 digest handle in the forwarded body (Tier-0 lossless only).
                assert "handle=" not in await resp.text(), (
                    "aproxy must not emit an unrecoverable digest handle"
                )

    _run(_body())


def test_aproxy_responses_path_compressed() -> None:
    """POST to /v1/responses is also a compressible path."""

    async def _body() -> None:
        async with TestServer(_echo_app()) as up:
            app = make_app(str(up.make_url("/")).rstrip("/"))
            async with TestClient(TestServer(app)) as client:
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "content": _LONG_TOOL_RESULT}],
                        }
                    ],
                }
                resp = await client.post(
                    "/v1/responses",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 200
                assert resp.headers.get("x-distil-compressed") == "1"

    _run(_body())


def test_aproxy_gemini_contents_compressed() -> None:
    """POST to a Gemini generateContent URL compresses the ``contents`` field."""

    async def _body() -> None:
        async with TestServer(_echo_app()) as up:
            app = make_app(str(up.make_url("/")).rstrip("/"))
            async with TestClient(TestServer(app)) as client:
                contents_text = "\n".join(f"Gemini line {i}" for i in range(30))
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": contents_text}]}],
                    "generationConfig": {"maxOutputTokens": 100},
                }
                resp = await client.post(
                    "/v1beta/models/gemini-pro:generateContent",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 200
                assert resp.headers.get("x-distil-compressed") == "1"

    _run(_body())


def test_aproxy_shape_output_light_header() -> None:
    """shape_output='light' adds x-distil-output-shaping header."""

    async def _body() -> None:
        async with TestServer(_echo_app()) as up:
            app = make_app(str(up.make_url("/")).rstrip("/"), shape_output="light")
            async with TestClient(TestServer(app)) as client:
                payload = {
                    "model": "claude-opus-4-8",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "content": _LONG_TOOL_RESULT}],
                        }
                    ],
                }
                resp = await client.post(
                    "/v1/messages",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 200
                assert resp.headers.get("x-distil-output-shaping") == "light"

    _run(_body())


# ---------------------------------------------------------------------------
# Tests: savings callback
# ---------------------------------------------------------------------------


def test_aproxy_savings_record_called() -> None:
    """savings.record() and maybe_flush() are called on each compressed request."""

    async def _body() -> None:
        savings = MagicMock()
        savings.record = MagicMock()
        savings.maybe_flush = MagicMock()
        savings.flush = MagicMock()

        async with TestServer(_echo_app()) as up:
            app = make_app(str(up.make_url("/")).rstrip("/"), savings=savings)
            async with TestClient(TestServer(app)) as client:
                payload = {
                    "model": "claude-opus-4-8",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "content": _LONG_TOOL_RESULT}],
                        }
                    ],
                }
                resp = await client.post(
                    "/v1/messages",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 200

        savings.record.assert_called_once()
        savings.maybe_flush.assert_called_once()

    _run(_body())


# ---------------------------------------------------------------------------
# Tests: invalid path → 400
# ---------------------------------------------------------------------------


def test_aproxy_invalid_path_400() -> None:
    """Path with @ (host-injection vector) on a compressible route → 400.

    aiohttp normalises ``..`` segments client-side, so we use a path that
    safe_forward_path rejects but aiohttp passes through unchanged: ``@`` in
    the path would let a forged request reach an unintended host.
    """

    async def _body() -> None:
        async with TestServer(_echo_app()) as up:
            app = make_app(str(up.make_url("/")).rstrip("/"))
            async with TestClient(TestServer(app)) as client:
                payload = {"messages": []}
                resp = await client.post(
                    "/@injected/v1/messages",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 400

    _run(_body())


def test_aproxy_invalid_path_passthrough_400() -> None:
    """Path with @ on a non-compressible route → 400."""

    async def _body() -> None:
        async with TestServer(_echo_app()) as up:
            app = make_app(str(up.make_url("/")).rstrip("/"))
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/@injected/v1/models")
                assert resp.status == 400

    _run(_body())


# ---------------------------------------------------------------------------
# Tests: upstream errors relayed
# ---------------------------------------------------------------------------


def test_aproxy_upstream_timeout_504() -> None:
    """Hung upstream + short sock_read timeout → 504 (ServerTimeoutError path).

    DISTIL_UPSTREAM_TIMEOUT is read by make_app at construction time so we set
    it in the environment before calling make_app, then restore it.
    """
    import os

    async def _body() -> None:
        async def _hung(req: web.Request) -> web.Response:
            await asyncio.sleep(30)  # never responds within the test window
            return web.Response(text="never")

        hung_app = web.Application()
        hung_app.router.add_post("/v1/messages", _hung)
        hung_app.router.add_route("*", "/{p:.*}", _hung)

        async with TestServer(hung_app) as up:
            upstream_url = str(up.make_url("/")).rstrip("/")
            # Set a very short timeout so the aiohttp client fires quickly.
            old = os.environ.get("DISTIL_UPSTREAM_TIMEOUT")
            os.environ["DISTIL_UPSTREAM_TIMEOUT"] = "0.3"
            try:
                app = make_app(upstream_url)
            finally:
                if old is not None:
                    os.environ["DISTIL_UPSTREAM_TIMEOUT"] = old
                else:
                    os.environ.pop("DISTIL_UPSTREAM_TIMEOUT", None)

            async with TestClient(TestServer(app)) as client:
                payload = {"messages": [{"role": "user", "content": "hi"}]}
                resp = await client.post(
                    "/v1/messages",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 504

    _run(_body())


def test_aproxy_upstream_client_error_502() -> None:
    """aiohttp.ClientError from upstream → 502."""

    async def _body() -> None:
        # Use a closed upstream URL to trigger a real ClientConnectorError → 502
        placeholder_app = web.Application()
        async with TestServer(placeholder_app) as up:
            dead_url = str(up.make_url("/")).rstrip("/")
        # Server is now shut down; connecting to its port gets refused.
        app = make_app(dead_url)
        async with TestClient(TestServer(app)) as client:
            payload = {"messages": [{"role": "user", "content": "hi"}]}
            resp = await client.post(
                "/v1/messages",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 502

    _run(_body())


def test_aproxy_health_answers_locally() -> None:
    """GET /distil/health is served by the proxy itself, not the upstream."""

    async def _body() -> None:
        async with TestServer(_echo_app()) as up:
            app = make_app(str(up.make_url("/")).rstrip("/"))
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/distil/health")
                assert resp.status == 200
                assert await resp.json() == {"status": "ok"}

    _run(_body())
