"""Cold-point recompression — evict old tool output on the turn the cache is already gone.

ADR 0014. Anthropic's prompt cache lives for a fixed TTL after the last request that
touched it (5 minutes by default, 1 hour when the client asks for ``ttl: "1h"``). A turn
that arrives after that has to re-write the whole prefix whatever distil does, so it is
the one moment old bytes can change for free. On such a turn distil replaces OLDER
tool_results with recoverable stubs, and from then on forwards those stubs on every turn,
so the smaller prefix is what the provider caches for the rest of the session.

This module is the state half, and deliberately knows nothing about message content
beyond one hash of one message. It answers a single question per request — *which
tool_use ids does this lineage evict?* — and the adapter renders them
(``adapters.anthropic.compress_messages(evict=...)``).

**Cache safety is the whole design.** A new eviction is decided only when expiry is
certain from distil's OWN observation:

* the lineage is known (first-seen, a restarted proxy, or an LRU-dropped lineage does
  nothing — distil cannot know when the provider last saw that prefix);
* nothing of this lineage is in flight (a streaming response, an expand re-query or a
  shadow replay can refresh the cache after the request that started it);
* the time since distil last *finished* forwarding it exceeds the longest TTL the
  lineage ever asked for plus a margin, on a monotonic clock;
* the conversation is unambiguously one conversation: every request must extend the
  previous one (same message count or more, and the previous last message unchanged).
  Anything else — two conversations sharing a lineage key, a rewind, a fork — marks the
  lineage ambiguous for good and no new eviction is ever decided on it.

Merging two conversations under one key can only make distil evict LESS: ``last`` is
the latest touch by either, so the gap it measures is never longer than either one's own.

Once evicted, an id stays evicted for the lineage's life (the set only grows) and its stub
is a pure function of the block's content, so every later turn forwards the same bytes.
State is content-free (ids, one 16-hex hash, counters) and bounded by an LRU.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .prefixreplay import canonical

log = logging.getLogger("distil.coldpoint")

TTL_DEFAULT_S = 300.0  # Anthropic's default ephemeral TTL
_TTLS = {None: TTL_DEFAULT_S, "5m": TTL_DEFAULT_S, "1h": 3600.0}
# ponytail: a fixed margin over the TTL. It covers network latency between distil's clock
# and the provider's, which is seconds; the paths that refresh the cache LATER than the
# forward (streaming, expand re-queries, shadow replays) are tracked as in-flight instead.
MARGIN_S = 60.0

_MAX_LINEAGES = 256
# The clock, as a module attribute so tests can drive it without patching the GLOBAL
# `time.monotonic` (which every socket timeout in the process also reads).
_clock = time.monotonic
_MAX_EXPANDED = 4096


@dataclass
class _State:
    last: float  # monotonic: when distil last forwarded (or finished forwarding) this lineage
    n: int  # messages in the last request
    tail: str  # hash of that request's last message, canonicalised
    ttl: float  # the longest TTL this lineage has asked for; inf = one we cannot parse
    inflight: int = 0
    evicted: frozenset[str] = frozenset()
    ambiguous: bool = False


@dataclass(frozen=True)
class Plan:
    """What this request does. ``evict`` is applied every turn; ``fresh`` counts the ids
    first evicted on THIS turn (non-zero only on a cold turn)."""

    evict: frozenset[str]
    reason: str  # first-seen | warm | cold | ambiguous | inflight | unknown-ttl | held
    fresh: int = 0


_STATES: "OrderedDict[str, _State]" = OrderedDict()
_EXPANDED: "OrderedDict[str, None]" = OrderedDict()
_LOCK = threading.Lock()


def reset() -> None:
    """Drop all state (tests)."""
    with _LOCK:
        _STATES.clear()
        _EXPANDED.clear()


def _markers(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every ``cache_control`` dict in the parts of *body* the provider caches."""

    def blocks(v: Any) -> Iterator[Any]:
        if isinstance(v, list):
            yield from v

    for node in [*blocks(body.get("system")), *blocks(body.get("tools"))]:
        if isinstance(node, dict) and isinstance(node.get("cache_control"), dict):
            yield node["cache_control"]
    for msg in blocks(body.get("messages")):
        if not isinstance(msg, dict):
            continue
        if isinstance(msg.get("cache_control"), dict):
            yield msg["cache_control"]
        for b in blocks(msg.get("content")):
            if isinstance(b, dict) and isinstance(b.get("cache_control"), dict):
                yield b["cache_control"]


def request_ttl(body: dict[str, Any]) -> float:
    """The longest cache TTL *body* asks for, in seconds. An unrecognised ``ttl`` is
    ``inf``: distil will not guess how long a cache entry it cannot parse lives."""
    ttl = TTL_DEFAULT_S
    for cc in _markers(body):
        t = cc.get("ttl")
        ttl = max(ttl, _TTLS.get(t, math.inf) if t is None or isinstance(t, str) else math.inf)
    return ttl


def _sig(msg: Any) -> str:
    return hashlib.sha256(canonical(msg).encode("utf-8", "replace")).hexdigest()[:16]


def plan(
    key: str,
    body: dict[str, Any],
    messages: list[Any],
    candidates: Callable[[], frozenset[str]],
    *,
    held: bool = False,
    now: float | None = None,
) -> Plan:
    """Decide this request's eviction set and mark the lineage in flight.

    Every call must be paired with :func:`end` once the request (and anything it spawned
    that talks to the provider) is finished. *candidates* is called only on a cold turn,
    outside the lock. *held* is the drift guard's hook: a held lineage keeps applying the
    set it has but decides nothing new.
    """
    now = _clock() if now is None else now
    ttl = request_ttl(body)
    with _LOCK:
        st = _STATES.get(key)
        if st is None:
            _STATES[key] = _State(
                last=now,
                n=len(messages),
                tail=_sig(messages[-1]) if messages else "",
                ttl=ttl,
                inflight=1,
            )
            while len(_STATES) > _MAX_LINEAGES:
                _STATES.popitem(last=False)
            return Plan(frozenset(), "first-seen")
        _STATES.move_to_end(key)
        extends = 0 < st.n <= len(messages) and _sig(messages[st.n - 1]) == st.tail
        st.ambiguous = st.ambiguous or not extends
        st.ttl = max(st.ttl, ttl)
        if st.ambiguous:
            reason = "ambiguous"
        elif st.inflight:
            reason = "inflight"
        elif held:
            reason = "held"
        elif math.isinf(st.ttl):
            reason = "unknown-ttl"
        elif now - st.last > st.ttl + MARGIN_S:
            reason = "cold"
        else:
            reason = "warm"
        st.last = max(st.last, now)
        st.n = len(messages)
        st.tail = _sig(messages[-1]) if messages else ""
        st.inflight += 1
        if reason != "cold":
            return Plan(st.evicted, reason)
        before = st.evicted

    try:
        fresh = candidates() - before
    except Exception:  # noqa: BLE001 — a failed choice evicts nothing new, never breaks
        log.debug("cold-point candidate selection failed", exc_info=True)
        fresh = frozenset()
    with _LOCK:
        st.evicted = st.evicted | fresh
        return Plan(st.evicted, "cold", len(fresh))


def _touch(key: str, delta: int, now: float | None) -> None:
    now = _clock() if now is None else now
    with _LOCK:
        st = _STATES.get(key)
        if st is not None:
            st.inflight = max(0, st.inflight + delta)
            st.last = max(st.last, now)


def begin(key: str, *, now: float | None = None) -> None:
    """Something else is about to send this lineage's prefix upstream (a shadow replay)."""
    _touch(key, +1, now)


def end(key: str, *, now: float | None = None) -> None:
    """A request (or replay) of this lineage finished talking to the provider."""
    _touch(key, -1, now)


def note_expanded(handle: str) -> None:
    """The model asked for this block back — never choose it for eviction."""
    with _LOCK:
        _EXPANDED[handle] = None
        _EXPANDED.move_to_end(handle)
        while len(_EXPANDED) > _MAX_EXPANDED:
            _EXPANDED.popitem(last=False)


def expanded() -> frozenset[str]:
    with _LOCK:
        return frozenset(_EXPANDED)
