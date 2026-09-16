"""Request-path safety helpers shared by the proxy, async proxy, and gateway.

These servers forward to a single configured upstream and (for the gateway) are
meant to be exposed. The helpers here defend the forwarding path against hostile
but well-formed input: upstream-host injection (SSRF), malformed/oversized bodies,
and credential-leaking path tricks. Kept dependency-free and side-effect-free so
every server can apply them identically.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Default maximum request body. Agent contexts are large but bounded; anything
# past this is almost certainly abuse, and reading it would be a memory-DoS.
MAX_BODY_BYTES = 8 * 1024 * 1024  # 8 MiB

# Azure OpenAI serves the SAME two request bodies distil already compresses, under a
# different path prefix — the deployment name (classic data plane) or a ``v1`` segment
# (the v1 preview API) sits where OpenAI has nothing:
#   POST {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=…
#   POST {endpoint}/openai/v1/chat/completions?api-version=preview
#   POST {endpoint}/openai/responses?api-version=…
#   POST {endpoint}/openai/v1/responses?api-version=preview
# Source: Azure OpenAI REST API reference (data plane, authoring + inference).
# Matched here rather than in each server so all three agree on one definition —
# three hand-maintained frozensets is how the Responses API ended up compressible
# in one server and forwarded raw by the other two.
_AZURE = r"/openai/(?:deployments/[^/]+/|v1/)?"
_CHAT_RE = re.compile(rf"^(?:/v1/chat/completions|{_AZURE}chat/completions)$")
_RESPONSES_RE = re.compile(rf"^(?:/v1/responses|{_AZURE}responses)$")
_MESSAGES_PATH = "/v1/messages"


def is_messages_path(target: str) -> bool:
    """True for the Anthropic Messages endpoint (``/v1/messages``)."""
    return strip_query(target) == _MESSAGES_PATH


def is_chat_completions_path(target: str) -> bool:
    """True for an OpenAI (or Azure OpenAI) Chat Completions endpoint."""
    return bool(_CHAT_RE.match(strip_query(target)))


def is_responses_path(target: str) -> bool:
    """True for an OpenAI (or Azure OpenAI) Responses API endpoint."""
    return bool(_RESPONSES_RE.match(strip_query(target)))


def is_compressible_path(target: str) -> bool:
    """True if *target* carries a request body distil knows how to compress.

    Covers Anthropic Messages, OpenAI Chat Completions and OpenAI Responses,
    including their Azure OpenAI path forms. Gemini's endpoint is matched
    separately (``adapters.gemini.is_gemini_path``) because the model name is
    embedded in the path and that adapter owns the pattern.
    """
    p = strip_query(target)
    return p == _MESSAGES_PATH or bool(_CHAT_RE.match(p)) or bool(_RESPONSES_RE.match(p))


def safe_forward_path(target: str) -> str | None:
    """Validate a request target before concatenating it onto the upstream base URL.

    Returns the (unchanged) target if it is a safe origin-form path, else ``None``.
    Blocks the host-injection / credential-leak vectors that raw ``base + path``
    string concat enables: ``@`` userinfo (``base@evil.com``), protocol-relative
    ``//evil.com``, scheme injection, ``..`` traversal, and control characters.
    Only the path portion (before ``?``) is constrained; the query string is
    forwarded as-is since it cannot change the upstream host.
    """
    if not isinstance(target, str) or not target:
        return None
    if any(ord(c) < 0x20 for c in target) or "\\" in target:
        return None
    path = target.split("?", 1)[0].split("#", 1)[0]
    if not path.startswith("/") or path.startswith("//"):
        return None
    if "@" in path or "://" in path:
        return None
    if any(seg == ".." for seg in path.split("/")):
        return None
    return target


def strip_query(target: str) -> str:
    """The path without query/fragment — for matching compressible routes."""
    return target.split("?", 1)[0].split("#", 1)[0]


def framing_rejection(
    content_lengths: Sequence[str] | None, transfer_encoding: str | None
) -> tuple[int, str] | None:
    """Reject a request whose body framing these servers cannot read.

    Returns ``(status, message)`` when the request must be refused, else ``None``.

    Takes *every* ``Content-Length`` value, not the header dict's first one: a
    repeated ``Content-Length`` is its own desync (CL.CL) and the first value is
    exactly what hides it. ``email.message.Message.get`` returns value one and
    keeps the rest in ``get_all``, so ``Content-Length: 5`` followed by
    ``Content-Length: 0`` reads as 5 here and may read as 0 to a front-end that
    picks the last — five bytes then stay queued as the head of the next request.
    RFC 9112 §6.3 permits identical duplicates (one value, sent twice), so those
    are allowed and anything differing is refused.

    Both stdlib-server entry points size the body from ``Content-Length`` alone.
    A request framed with ``Transfer-Encoding`` instead would read as *empty* and
    its bytes would stay queued on the socket — on a keep-alive HTTP/1.1
    connection the next parse then treats the undrained body as a second request,
    which is request smuggling as soon as any front-end that DOES honour
    ``Transfer-Encoding`` sits in front (TE.CL desync). Both headers together is
    the same disagreement stated outright. LLM SDKs always send a length, so
    refusing is free; the caller must also close the connection so nothing queued
    behind a rejected request is ever parsed.
    """
    lengths = [v.strip() for v in (content_lengths or [])]
    te = (transfer_encoding or "").strip()
    if te:
        if any(lengths):
            return (400, "conflicting Content-Length and Transfer-Encoding headers")
        return (411, "chunked request bodies are not supported; send Content-Length")
    # A single header may itself carry a comma list ("5, 0") — same disagreement,
    # one header line. Flatten before comparing so that shape cannot slip past.
    values = [part.strip() for v in lengths for part in v.split(",")]
    if len(values) > 1 and len(set(values)) > 1:
        return (400, "conflicting Content-Length headers")
    return None


def parse_content_length(raw: object, *, max_bytes: int = MAX_BODY_BYTES) -> int | None:
    """Defensively parse a ``Content-Length`` header value.

    Returns a non-negative byte count, or ``None`` if the header is missing,
    non-numeric, negative, or exceeds ``max_bytes`` (caller should reject with
    400/413). Prevents ``int()`` crashes, negative-length read hangs, and
    unbounded-body memory exhaustion.
    """
    if raw is None:
        return 0
    if not isinstance(raw, (str, int)):
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n < 0 or n > max_bytes:
        return None
    return n
