"""Async high-concurrency proxy using aiohttp (optional [async] extra).

Handles thousands of concurrent streaming sessions that the threaded proxy.py
cannot sustain under high fan-out. aiohttp is lazy-imported so the stdlib core
remains dependency-free.

Usage
-----
::

    from distil.aproxy import serve
    serve(host="127.0.0.1", port=8788, upstream="https://api.anthropic.com")

Or build the app yourself::

    from distil.aproxy import make_app
    app = make_app("https://api.anthropic.com", lossless_only=False, shape_output="off")
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from ._log import log
from .adapters.anthropic import compress_messages
from .adapters.gemini import compress_generate_request
from .adapters.gemini import count_tokens as _gemini_count
from .adapters.gemini import is_gemini_path
from .httpguard import MAX_BODY_BYTES, safe_forward_path
from .tokenizer import DEFAULT as _tokenizer

# ---------------------------------------------------------------------------
# Paths that carry a ``messages`` payload worth compressing
# ---------------------------------------------------------------------------

_COMPRESSIBLE_PATHS = frozenset({"/v1/messages", "/v1/chat/completions", "/v1/responses"})

# Hop-by-hop headers must not be forwarded — they are connection-specific.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailers",
        "upgrade",
    }
)


# ---------------------------------------------------------------------------
# Token-saving estimator (mirrors proxy.py._tokens_saved)
# ---------------------------------------------------------------------------


def _count_msgs(msgs: list[dict[str, Any]]) -> int:
    """Heuristic token count of a messages list (mirrors proxy._count_messages)."""
    total = 0
    for msg in msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _tokenizer.count(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                for key in ("text", "content"):
                    val = block.get(key)
                    if isinstance(val, str):
                        total += _tokenizer.count(val)
                    elif isinstance(val, list):
                        for sub in val:
                            if isinstance(sub, dict):
                                sv = sub.get("text", "")
                                if isinstance(sv, str):
                                    total += _tokenizer.count(sv)
    return total


def _tokens_saved(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int:
    """Rough estimate of tokens saved via the default heuristic tokeniser."""
    return max(0, _count_msgs(before) - _count_msgs(after))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_headers(headers: Any) -> dict[str, str]:
    """Strip hop-by-hop headers from a mapping, returning a plain dict."""
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def make_app(
    upstream: str,
    *,
    lossless_only: bool = False,
    verbatim: bool = False,
    shape_output: str = "off",
    savings: Any = None,
) -> Any:
    """Build and return an ``aiohttp.web.Application``.

    aiohttp is lazy-imported here; if it is not installed you will get a clear
    ImportError pointing to ``pip install 'distil-llm[async]'``.

    Parameters
    ----------
    upstream:
        Base URL of the real LLM API, e.g. ``"https://api.anthropic.com"``.
        Must not have a trailing slash.
    lossless_only:
        When *True* only Tier-0 lossless transforms are applied.
    shape_output:
        Output-compression level (``"off"``/``"light"``/``"aggressive"``). When
        not ``"off"`` and lossy compression is permitted, a verbosity-control
        directive is appended so the model emits fewer tokens.
    """
    try:
        from aiohttp import web
        import aiohttp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "aiohttp is required for the async proxy. "
            "Install it with: pip install 'distil-llm[async]'"
        ) from exc

    _upstream = upstream.rstrip("/")

    # lossless-only implies Tier-0-only: without an injected expand tool the agent
    # cannot recover a Tier-1 digest stub, so a stub there would be irreversibly
    # lossy. Fold it into verbatim (the flag that already disables Tier-1 digests).
    from .policy import AuthMode, may_compress_lossy

    # Route the lossy-allowed decision through policy as the single source of truth:
    # subscription / OAuth sessions are lossless-only. Forces Tier-0-only (verbatim)
    # and gates output shaping below.
    _auth_mode = AuthMode.SUBSCRIPTION if lossless_only else AuthMode.PAYG
    _lossy_ok = may_compress_lossy(_auth_mode)
    # aproxy injects no distil_expand tool and runs no expand loop, so a Tier-1 digest
    # stub here can NEVER be recovered — irreversibly lossy on PAYG exactly as on a
    # subscription. The note above says to fold unrecoverable digest into verbatim; the
    # previous `not _lossy_ok` only did so for subscription, leaving PAYG async sessions
    # emitting irreversible stubs. Fold it for EVERY session: the async path stays Tier-0
    # (reversible lossless transforms — e.g. the #24 columnar fold — still apply, so real
    # savings remain). Recoverable Tier-1 needs the expand loop the sync proxy implements;
    # until aproxy grows a streaming equivalent it must not emit an unrecoverable stub.
    # `_lossy_ok` still gates response shaping (opt-in) below.
    verbatim = True

    # Eager-load the request-path module the handler otherwise imports lazily, so
    # an in-place upgrade never loads a post-upgrade .py mid-serve against the
    # running interpreter (version skew). Warmed here at server setup; the
    # per-request `from .output import ...` is then a module-cache hit.
    from .output import shape_request as _shape_request  # noqa: F401

    # Generous but finite: connect fails fast; sock_read is an INACTIVITY timeout
    # (resets on every chunk), so long generations stream freely while a wedged
    # upstream can never hold a request open forever.
    _timeout = aiohttp.ClientTimeout(
        total=None, connect=30, sock_read=float(os.environ.get("DISTIL_UPSTREAM_TIMEOUT", "600"))
    )

    from . import __version__ as _running_version
    from .proxy import _warn_if_version_skew

    # Version-skew guard: warn once if distil was upgraded on disk while this
    # long-lived async proxy keeps running its old in-memory code (see proxy.py).
    _version_state: dict[str, Any] = {"running": _running_version}

    # Typed key avoids NotAppKeyWarning and provides a stable reference for handlers.
    _client_key: web.AppKey[aiohttp.ClientSession] = web.AppKey("client", aiohttp.ClientSession)

    # -----------------------------------------------------------------------
    # Startup / cleanup: one shared ClientSession for the app lifetime
    # -----------------------------------------------------------------------

    async def _on_startup(app: web.Application) -> None:
        # Bound concurrent upstream connections so inbound load can't fan out into
        # an unbounded socket count (backpressure + self-DoS protection).
        connector = aiohttp.TCPConnector(limit=100)
        app[_client_key] = aiohttp.ClientSession(connector=connector)

    async def _on_cleanup(app: web.Application) -> None:
        await app[_client_key].close()
        if savings is not None:
            savings.flush()  # persist remaining genuine savings on shutdown

    # -----------------------------------------------------------------------
    # Incremental relay: stream upstream bytes to the client as they arrive
    # (time-to-first-token preserved; never buffer a whole SSE generation).
    # -----------------------------------------------------------------------

    async def _relay_streaming(
        request: web.Request,
        method: str,
        url: str,
        data: bytes | None,
        headers: dict[str, str],
        extras: dict[str, str] | None = None,
        on_success: Callable[[], None] | None = None,
    ) -> web.StreamResponse:
        client: aiohttp.ClientSession = request.app[_client_key]
        try:
            # allow_redirects=False: relay 3xx instead of re-sending the client's
            # credentials to whatever host the upstream redirect names.
            resp_cm = client.request(
                method, url, data=data, headers=headers, timeout=_timeout, allow_redirects=False
            )
            resp = await resp_cm.__aenter__()
        except aiohttp.ServerTimeoutError:
            return web.json_response({"error": "upstream timed out"}, status=504)
        except (aiohttp.ClientError, TimeoutError) as exc:
            return web.json_response(
                {"error": "upstream connection failed", "detail": str(exc)[:200]}, status=502
            )
        try:
            resp_headers = _filter_headers(resp.headers)
            if extras:
                resp_headers.update(extras)
            sr = web.StreamResponse(status=resp.status, headers=resp_headers)
            await sr.prepare(request)
            try:
                async for chunk in resp.content.iter_any():
                    await sr.write(chunk)
            except (aiohttp.ClientError, TimeoutError, ConnectionResetError):
                pass  # mid-stream failure: headers are out; close what we have
            await sr.write_eof()
            # Book savings only after a fully-relayed 2xx (P0-1): failed or
            # SDK-retried upstream calls must not be counted as savings.
            if on_success is not None and 200 <= resp.status < 300:
                on_success()
            return sr
        finally:
            await resp_cm.__aexit__(None, None, None)

    # -----------------------------------------------------------------------
    # Compression handler
    # -----------------------------------------------------------------------

    async def _handle_compressible(request: web.Request) -> web.Response:
        _warn_if_version_skew(_version_state)
        if safe_forward_path(request.path) is None:
            return web.json_response({"error": "invalid request path"}, status=400)
        raw = await request.read()
        fwd_headers = _filter_headers(request.headers)
        extras: dict[str, str] = {}
        # Savings are booked only after a confirmed 2xx (P0-1): (before, after, model).
        _pending_savings: tuple[int, int, str | None] | None = None

        try:
            body: dict[str, Any] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            body_bytes = raw
        else:
            if "messages" in body and isinstance(body["messages"], list):
                original: list[dict[str, Any]] = body["messages"]
                try:
                    compressed, _store = compress_messages(original, verbatim=verbatim)
                except Exception:  # noqa: BLE001 — compression must never break a request
                    log.debug("compress_messages failed; forwarding uncompressed", exc_info=True)
                    compressed = original
                saved = _tokens_saved(original, compressed)
                body = {**body, "messages": compressed}
                extras = {
                    "x-distil-compressed": "1",
                    "x-distil-tokens-saved": str(saved),
                }
                if shape_output != "off" and _lossy_ok:
                    from .output import shape_request

                    _shape = "anthropic" if request.path == "/v1/messages" else "openai"
                    body = shape_request(body, level=shape_output, allow=True, shape=_shape)
                    extras["x-distil-output-shaping"] = shape_output
                if savings is not None:
                    before = _count_msgs(original)
                    _pending_savings = (before, before - saved, body.get("model"))

            elif "contents" in body and isinstance(body["contents"], list):
                # Gemini generateContent shape (reversible content compression).
                before_tok = _gemini_count(body)
                try:
                    body, _store = compress_generate_request(body, verbatim=verbatim)
                except Exception:  # noqa: BLE001 — compression must never break a request
                    log.debug("gemini compression failed; forwarding uncompressed", exc_info=True)
                saved = max(0, before_tok - _gemini_count(body))
                extras = {
                    "x-distil-compressed": "1",
                    "x-distil-tokens-saved": str(saved),
                }
                if savings is not None:
                    _pending_savings = (before_tok, before_tok - saved, None)

            body_bytes = json.dumps(body).encode()

        url = _upstream + request.path
        if request.query_string:
            url = f"{url}?{request.query_string}"

        def _book() -> None:
            if savings is not None and _pending_savings is not None:
                b, a, m = _pending_savings
                savings.record(b, a, model=m)
                savings.maybe_flush()

        return await _relay_streaming(
            request,
            "POST",
            url,
            body_bytes,
            {**fwd_headers, "content-length": str(len(body_bytes))},
            extras=extras,
            on_success=_book,
        )

    # -----------------------------------------------------------------------
    # Transparent passthrough (all other paths / methods)
    # -----------------------------------------------------------------------

    async def _passthrough(request: web.Request) -> web.Response:
        if safe_forward_path(request.path) is None:
            return web.json_response({"error": "invalid request path"}, status=400)
        raw = await request.read()
        fwd_headers = _filter_headers(request.headers)
        if raw:
            fwd_headers["content-length"] = str(len(raw))

        url = _upstream + request.path
        if request.query_string:
            url = f"{url}?{request.query_string}"

        return await _relay_streaming(request, request.method, url, raw or None, fwd_headers)

    # -----------------------------------------------------------------------
    # Router
    # -----------------------------------------------------------------------

    async def _route_post(request: web.Request) -> web.Response:
        if request.path in _COMPRESSIBLE_PATHS:
            return await _handle_compressible(request)
        return await _passthrough(request)

    async def _route_any(request: web.Request) -> web.Response:
        # Wildcard dispatcher: Gemini's path is dynamic (model name in the URL), so
        # it can't be a fixed route — match it here and compress; else passthrough.
        if request.method == "POST" and (
            request.path in _COMPRESSIBLE_PATHS or is_gemini_path(request.path)
        ):
            return await _handle_compressible(request)
        return await _passthrough(request)

    # Cap inbound body size so a giant POST can't exhaust memory (aiohttp returns
    # 413 automatically past this); matches the sync servers' guard.
    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    async def _health(request: web.Request) -> web.Response:  # noqa: ARG001
        # Liveness probe: answers locally, never touches the billed upstream.
        return web.json_response({"status": "ok"})

    app.router.add_get("/distil/health", _health)

    # Register compressible POST paths explicitly; everything else hits the wildcard.
    for path in _COMPRESSIBLE_PATHS:
        app.router.add_post(path, _route_post)

    # Wildcard for all other paths/methods (aiohttp doesn't have a true catch-all,
    # so we register explicit wildcards for the common verbs plus a fallback).
    app.router.add_route("*", "/{path_info:.*}", _route_any)

    return app


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------


def serve(
    host: str = "127.0.0.1",
    port: int = 8788,
    upstream: str = "https://api.anthropic.com",
    *,
    lossless_only: bool = False,
    verbatim: bool = False,
    shape_output: str = "off",
    record: bool = True,
    pricing_model: str = "claude-opus-4-8",
) -> None:
    """Run an async aiohttp proxy server.

    Parameters
    ----------
    host:       Interface to bind on.
    port:       Port to listen on.
    upstream:   Real LLM API base URL (no trailing slash).
    lossless_only:
        Policy mode: no lossy output-shaping, no tool injection (digest still runs).
    verbatim:
        When *True*, skip the Tier-1 digest (Tier-0 only) — interactive-safe.
    shape_output:
        Output-compression level: ``"off"``/``"light"``/``"aggressive"``.
    """
    try:
        from aiohttp import web
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "aiohttp is required for the async proxy. "
            "Install it with: pip install 'distil-llm[async]'"
        ) from exc

    savings = None
    if record:
        from .runtime import RuntimeSavings

        savings = RuntimeSavings(model=pricing_model)
    app = make_app(
        upstream,
        lossless_only=lossless_only,
        verbatim=verbatim,
        shape_output=shape_output,
        savings=savings,
    )
    print(f"distil async proxy listening on http://{host}:{port}")
    print(f"  → upstream: {upstream}")
    if shape_output != "off":
        if lossless_only:
            print(
                "  ⚠ --shape-output requested but SUPPRESSED: lossless-only never modifies "
                "the response. No shaping will happen. Drop --lossless-only to enable it."
            )
        else:
            print(f"  → output shaping: {shape_output}")
    if savings is not None:
        print("  → recording genuine savings → distil leaderboard")
    web.run_app(app, host=host, port=port, print=None)
