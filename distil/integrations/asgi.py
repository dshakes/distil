"""Generic ASGI middleware — compress provider-shaped POST bodies in-process.

For apps that host their own LLM-facing endpoint — a FastAPI/Starlette/Litestar
backend that builds a request and forwards it to Anthropic/OpenAI/Gemini itself,
rather than going through ``distil proxy`` as a sidecar. Wrap the ASGI app once
and its own outbound call sees a compressed body, with no other code change::

    from distil.integrations.asgi import DistilMiddleware

    app = DistilMiddleware(app)                 # ASGI 3 callable, any framework
    app = DistilMiddleware(app, verbatim=True)  # Tier-0 lossless only

**Pure ASGI** — no Starlette/FastAPI import — so it works under any ASGI 3 server
or framework. Reuses the exact body-shape detection and reversible compression
:mod:`distil.aproxy` uses for the sidecar proxy rather than re-implementing it:
``aproxy._COMPRESSIBLE_PATHS`` for the Anthropic/OpenAI-shaped routes,
``adapters.gemini.is_gemini_path`` for Gemini's dynamic ``generateContent`` path,
and ``adapters.anthropic.compress_messages`` / ``adapters.gemini.compress_generate_request``
for the transform itself — so a request compressed here is compressed identically
to one that went through the sidecar proxy, sharing the same on-disk restore store.

Fail-open: a non-JSON body, an unrecognized shape, or a compression error all pass
the original bytes through unchanged rather than breaking the request.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from ..adapters.anthropic import compress_messages
from ..adapters.gemini import compress_generate_request, is_gemini_path
from ..aproxy import _COMPRESSIBLE_PATHS
from ..httpguard import MAX_BODY_BYTES

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
App = Callable[[Scope, Receive, Send], Awaitable[None]]

__all__ = ["DistilMiddleware"]


def _compressible(path: str) -> bool:
    return path in _COMPRESSIBLE_PATHS or is_gemini_path(path)


async def _read_body(receive: Receive) -> bytes:
    """Drain the ``http.request`` events into one buffer."""
    body = bytearray()
    more = True
    while more:
        message = await receive()
        if message.get("type") == "http.disconnect":
            break
        body += message.get("body", b"")
        more = bool(message.get("more_body", False))
    return bytes(body)


def _compress_body(body: bytes, *, verbatim: bool) -> bytes:
    """Return a compressed body, or *body* unchanged on any unrecognized shape/error."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body
    if not isinstance(payload, dict):
        return body
    try:
        if isinstance(payload.get("messages"), list):
            compressed, _store = compress_messages(payload["messages"], verbatim=verbatim)
            payload = {**payload, "messages": compressed}
        elif isinstance(payload.get("contents"), list):
            payload, _store = compress_generate_request(payload, verbatim=verbatim)
        else:
            return body
    except Exception:  # noqa: BLE001 — compression must never break a request
        return body
    return json.dumps(payload).encode()


def _with_content_length(
    headers: list[tuple[bytes, bytes]], new_len: int
) -> list[tuple[bytes, bytes]]:
    """Return *headers* with ``content-length`` fixed up to *new_len*."""
    out = [(k, v) for k, v in headers if k.lower() != b"content-length"]
    out.append((b"content-length", str(new_len).encode()))
    return out


class DistilMiddleware:
    """ASGI middleware that compresses the JSON body of provider-shaped POST requests.

    ``app`` is the wrapped ASGI application — typically the app whose own route
    handler forwards the (now-compressed) body upstream. Everything that is not a
    ``POST`` to a compressible path (``/v1/messages``, ``/v1/chat/completions``,
    ``/v1/responses``, or a Gemini ``generateContent`` route) passes through with
    the original ``receive`` untouched, at zero cost.

    ponytail: buffers the whole request body before compressing, the same ceiling
    every body-inspecting ASGI middleware has (e.g. Starlette's ``Request.body()``);
    skips compression above ``MAX_BODY_BYTES`` rather than risk memory blowup on a
    hostile payload — add a streaming chunk cap if that ever measures as a real
    concern for this deployment.
    """

    def __init__(self, app: App, *, verbatim: bool = False) -> None:
        self.app = app
        self.verbatim = verbatim

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and _compressible(str(scope.get("path", "")))
        ):
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        new_body = (
            body if len(body) > MAX_BODY_BYTES else _compress_body(body, verbatim=self.verbatim)
        )

        new_scope = dict(scope)
        new_scope["headers"] = _with_content_length(list(scope.get("headers", [])), len(new_body))

        sent = False

        async def _replay() -> Message:
            nonlocal sent
            if sent:
                return await receive()  # e.g. a later http.disconnect — the real channel
            sent = True
            return {"type": "http.request", "body": new_body, "more_body": False}

        await self.app(new_scope, _replay, send)
