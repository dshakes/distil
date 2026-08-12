"""Recency carve-out — the shared constant, and the statement of how the certificate
transfers to serving.

An agent must see its freshest tool outputs byte-exact to choose its next action (and
the in-context path may not be able to expand a stub there), so the live adapter never
digests tool outputs in the last ``RECENCY_KEEP_TURNS`` tool-bearing turns.

That carve-out was once justified here as "strictly cache-safe: the cached prefix never
contains recent message history". That is false, and it was expensive. A window counted
back from the END slides forward as the conversation grows, so every block is protected
while fresh and digested one turn later — rewriting a message that the client has by
then committed to its cached prefix. Measured against the real API: zero cache reads on
every turn and 2x the cost of sending nothing compressed at all, because the whole
prefix is re-written at 1.25x instead of re-read at 0.1x.

So recency is anchored to the client's own ``cache_control`` breakpoint instead of to
the end of the list (``cached_prefix_end``). Content the client has marked cacheable is
already committed and must reach the wire in its final form; only the uncached tail
after the breakpoint is exempt. Where a client caches everything, that means digesting
the freshest observation too — which is exactly what the certified strategy below
already does, so the transfer direction stays safe.

The certified ``distil`` strategy deliberately does NOT apply this carve-out: it digests
every volatile block, including the freshest observation. That makes certification
*harsher* than serving, which is the safe transfer direction — if decisions survive with
the freshest output digested, they survive a-fortiori when serving keeps it verbatim.
The invariant that serving's digest-set is a SUBSET of certification's digest-set is
pinned by ``tests/test_live_certified_equivalence.py``; the constant lives here so both
sides of that invariant are reviewed in one place.
"""

from __future__ import annotations

from typing import Any

RECENCY_KEEP_TURNS = 2


def exempt_indices(idxs: list[int], k: int, cached_through: int | None) -> set[int]:
    """Which of the tool-bearing turns at *idxs* may stay verbatim.

    The rule across every adapter: **exempt only content the provider will not
    have cached.** Anything the provider has cached must reach the wire in its
    final form forever, because rewriting it later invalidates the cache entry
    for the whole prefix — which costs far more than the digest saves.

    ``cached_through`` is the index the provider has cached through:

    * ``None`` — no prefix caching in play (e.g. an Anthropic request with no
      ``cache_control``). Nothing to invalidate, so the plain last-*k* window
      applies and the freshest tool output stays byte-exact.
    * ``-1`` — caching is available but nothing is committed yet.
    * ``>= 0`` — only turns strictly after it are exempt.

    Providers that cache **implicitly** (OpenAI, Gemini: automatic prefix
    caching, no client marker) commit everything they are sent, so they pass the
    last index and get no carve-out at all. That digests the freshest
    observation, which is exactly what the certified strategy does — see the
    module docstring on why that transfer direction is the safe one.
    """
    if k <= 0:
        return set()
    window = set(idxs[-k:])
    if cached_through is None:
        return window
    return {i for i in window if i > cached_through}


def cached_prefix_end(messages: list[Any]) -> int:
    """Index of the last message the client marked cacheable, or -1 if none.

    A ``cache_control`` marker is the client stating that everything up to and
    including that block is a stable prefix it intends to re-send byte-for-byte.
    Rewriting anything at or before it invalidates the provider's cache entry for
    the entire prefix, so those messages must be emitted in their final form.
    """
    last = -1
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):
            if any(isinstance(b, dict) and b.get("cache_control") for b in content):
                last = i
        elif isinstance(m.get("cache_control"), dict):
            last = i
    return last
