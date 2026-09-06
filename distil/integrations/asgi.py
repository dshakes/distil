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
the sidecar proxy uses rather than re-implementing it: ``httpguard.is_compressible_path``
for the Anthropic/OpenAI-shaped routes (the one path matcher all three servers
share), ``adapters.gemini.is_gemini_path`` for Gemini's dynamic ``generateContent``
path, and ``adapters.anthropic.compress_messages`` / ``adapters.gemini.compress_generate_request``
for the transform itself — so a request compressed here is compressed identically
to one that went through the sidecar proxy, sharing the same on-disk restore store.

Only the Anthropic Messages and Gemini shapes (``messages`` / ``contents`` lists)
are transformed here — a path this middleware recognizes but whose body it does
not (OpenAI Responses' ``input`` list) fails open, unchanged, same as malformed
JSON or an unknown shape.

Fail-open: a non-JSON body, an unrecognized shape, or a compression error all pass
the original bytes through unchanged rather than breaking the request.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from ..adapters.anthropic import compress_messages
from ..adapters.gemini import compress_generate_request, is_gemini_path
from ..httpguard import MAX_BODY_BYTES, is_compressible_path

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
App = Callable[[Scope, Receive, Send], Awaitable[None]]

__all__ = ["DistilMiddleware"]


def _compressible(path: str) -> bool:
    return is_compressible_path(path) or is_gemini_path(path)


class _BodyReader:
    """Drains ``http.request`` events up to ``MAX_BODY_BYTES``, remembering every
    message consumed so a caller that gives up on compression can replay the
    exact same event sequence rather than a lossy reconstruction.
    """

    def __init__(self, receive: Receive) -> None:
        self._receive = receive
        self.consumed: list[Message] = []

    async def read(self) -> bytes | None:
        """Return the full body if it arrived within the cap with no disconnect.

        Returns ``None`` the moment the cap is exceeded or the client
        disconnects mid-body — at that point ``self.consumed`` holds every
        message read so far (the chunk that crossed the cap included, and
        nothing past it), for the caller to replay verbatim.
        """
        total = 0
        while True:
            message = await self._receive()
            self.consumed.append(message)
            if message.get("type") == "http.disconnect":
                return None
            total += len(message.get("body", b""))
            if total > MAX_BODY_BYTES:
                return None
            if not message.get("more_body", False):
                return b"".join(m.get("body", b"") for m in self.consumed)


def _verbatim_receiver(consumed: list[Message], receive: Receive) -> Receive:
    """Return a ``receive`` that replays *consumed* first, then falls through to *receive*."""
    queue = list(consumed)

    async def _replay() -> Message:
        if queue:
            return queue.pop(0)
        return await receive()

    return _replay


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

    ponytail: buffers the whole request body before compressing (up to
    ``MAX_BODY_BYTES``), the same ceiling every body-inspecting ASGI middleware
    has (e.g. Starlette's ``Request.body()``) — add a streaming *compressor* if
    that ever measures as a real concern for this deployment. The cap itself is
    enforced while reading, not after, so a hostile payload never sits fully
    buffered first.
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

        reader = _BodyReader(receive)
        body = await reader.read()

        if body is None:
            # Oversized (stopped reading the instant the cap was crossed — never
            # holding more than that one chunk beyond the cap) or the client
            # disconnected mid-body. Either way, downstream must see exactly the
            # event sequence it would have without this middleware: replay what
            # was already consumed (a disconnect included, if that is why we
            # stopped), then hand any further receive() straight to the real
            # channel.
            await self.app(scope, _verbatim_receiver(reader.consumed, receive), send)
            return

        new_body = _compress_body(body, verbatim=self.verbatim)

        if new_body == body:
            # Fail-open (malformed JSON, unrecognized shape, or a compression error):
            # replay the exact original event sequence and headers, not a synthesized
            # single chunk — an app that inspects chunking must see no difference.
            await self.app(scope, _verbatim_receiver(reader.consumed, receive), send)
            return

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
