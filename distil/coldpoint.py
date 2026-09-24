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

import functools
import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .prefixreplay import canonical

log = logging.getLogger("distil.coldpoint")

TTL_DEFAULT_S = 300.0  # Anthropic's default ephemeral TTL
_TTLS = {None: TTL_DEFAULT_S, "5m": TTL_DEFAULT_S, "1h": 3600.0}
# The ONE safety margin over the TTL, in seconds. It covers the gap between distil's clock
# and the provider's (network latency, seconds); the paths that refresh the cache LATER
# than the forward (streaming, expand re-queries, shadow replays) are tracked as in-flight
# instead. ADR 0014's soak gate says when to raise it: a `cold` turn that still reads
# more than the static prefix from cache means the entry was alive.
MARGIN_S = 60.0

_MAX_LINEAGES = 256
_MAX_EXPANDED = 4096
# Lineages whose evicted-id set is persisted (see `_persist`). LRU, oldest dropped.
_MAX_PERSISTED = 512


def _pick_clock(platform: str = sys.platform) -> Callable[[], float]:
    """A monotonic clock that keeps counting while the machine sleeps.

    The provider's TTL runs on wall time, so a laptop that slept through an idle hour has
    a cold cache. Python's ``time.monotonic`` on macOS is ``mach_absolute_time``, which
    STOPS during sleep; that errs safe (distil would just miss the cold point), but it
    misses the commonest one. ``clock_gettime(CLOCK_MONOTONIC)`` on darwin "will continue
    to increment while the system is asleep" (clock_gettime(3)); on Linux the equivalent
    is ``CLOCK_BOOTTIME``. Anything else falls back to ``time.monotonic``.
    """
    cid = time.CLOCK_MONOTONIC if platform == "darwin" else getattr(time, "CLOCK_BOOTTIME", None)
    if cid is None or not hasattr(time, "clock_gettime"):
        return time.monotonic
    try:
        time.clock_gettime(cid)
    except OSError:
        return time.monotonic
    return functools.partial(time.clock_gettime, cid)


# The clock, as a module attribute so tests can drive it without patching the GLOBAL
# `time` functions (which every socket timeout in the process also reads).
_clock = _pick_clock()


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
_EXPANDED: "OrderedDict[tuple[str, str], None]" = OrderedDict()  # (scope, handle)
_LOCK = threading.Lock()


def reset() -> None:
    """Drop all in-memory state (tests; what a restart looks like from inside)."""
    with _LOCK:
        _STATES.clear()
        _EXPANDED.clear()


# Credential headers whose value is a long-lived key. `Authorization` is deliberately NOT
# here: Claude Code's OAuth bearer token refreshes mid-session, and keying on it forked
# the lineage and un-evicted a warm prefix at every refresh.
_STATIC_KEY_HEADERS = ("x-api-key", "api-key", "x-goog-api-key")


def account_scope(headers: Mapping[str, str]) -> str:
    """Lineage scope that survives a token refresh: a hash of the static API key, or "".

    Coarser than ``prefixreplay.credential_scope`` on purpose, and safe to be: this state
    holds no content — timing, counts, random tool_use ids — and every stub is rendered
    from the request's OWN content. Two bearer-token callers of one proxy that send the
    same conversation head share a lineage; interleaved, they fail the extension check
    and go ambiguous (no new eviction), so the worst case is a missed saving or one
    rewrite, never another caller's bytes. The gateway, which is where tenants are, does
    not run this (ADR 0014 scope).
    """
    creds = sorted(
        (k.lower(), v)
        for k, v in headers.items()
        if k.lower() in _STATIC_KEY_HEADERS and isinstance(v, str) and v
    )
    if not creds:
        return ""
    blob = "\0".join(f"{k}={v}" for k, v in creds)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:12] + "\0"


def _persist_path() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "coldpoint.json"


def _read_persisted(path: Path) -> "OrderedDict[str, frozenset[str]]":
    """``{lineage key: evicted ids}`` from disk, oldest first. Empty on any problem."""
    out: "OrderedDict[str, frozenset[str]]" = OrderedDict()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return out
    except (OSError, ValueError):
        log.debug("cold-point state unreadable; starting empty", exc_info=True)
        return out
    lineages = raw.get("lineages") if isinstance(raw, dict) else None
    if isinstance(lineages, dict):
        for k, v in lineages.items():
            if isinstance(k, str) and isinstance(v, list):
                out[k] = frozenset(x for x in v if isinstance(x, str))
    return out


def _load(key: str) -> frozenset[str]:
    """The evicted set a previous process persisted for *key* (first-seen only)."""
    return _read_persisted(_persist_path()).get(key, frozenset())


def _persist(key: str, ids: frozenset[str]) -> None:
    """Merge *ids* into *key*'s persisted set. Called only when a set grows.

    Why it exists: the set on the wire must outlive the process. A hot-swap (every
    upgrade), a restart or an LRU drop would otherwise forward the un-evicted form on the
    next turn — one full rewrite of a prefix that was still warm. Read-merge-write under
    a file lock so an old worker draining during a hot-swap and the new one cannot drop
    each other's ids; written with mkstemp (0600 at creation) → fsync → os.replace.
    Content-free: hashed lineage keys and the provider's random tool_use ids.
    """
    try:
        from . import _filelock
        from .config_wrap import _atomic_write_secure

        path = _persist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _filelock.locked(path):
            data = _read_persisted(path)
            data[key] = data.get(key, frozenset()) | ids
            data.move_to_end(key)
            while len(data) > _MAX_PERSISTED:
                data.popitem(last=False)
            blob = {"version": 1, "lineages": {k: sorted(v) for k, v in data.items()}}
            _atomic_write_secure(path, json.dumps(blob, separators=(",", ":")).encode())
    except Exception:  # noqa: BLE001 — persistence is an optimisation; never break a request
        log.debug("cold-point state not persisted", exc_info=True)


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
    # A lineage new to THIS process may still have a set on the wire from the previous
    # one. Read outside the lock; only first-seen lineages ever touch the disk here.
    persisted = _load(key) if key not in _STATES else frozenset()
    with _LOCK:
        st = _STATES.get(key)
        if st is None:
            _STATES[key] = _State(
                last=now,
                n=len(messages),
                tail=_sig(messages[-1]) if messages else "",
                ttl=ttl,
                inflight=1,
                evicted=persisted,
            )
            while len(_STATES) > _MAX_LINEAGES:
                _STATES.popitem(last=False)
            # Re-apply what is on the wire, decide nothing new: distil cannot know when
            # the provider last saw this prefix.
            return Plan(persisted, "first-seen")
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
        evict = st.evicted
    if fresh:
        _persist(key, evict)
    return Plan(evict, "cold", len(fresh))


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


def note_expanded(scope: str, handle: str) -> None:
    """The model asked for this block back — never choose it for eviction. Scoped by
    :func:`account_scope`, so one caller's expansions do not shape another's evictions."""
    with _LOCK:
        _EXPANDED[(scope, handle)] = None
        _EXPANDED.move_to_end((scope, handle))
        while len(_EXPANDED) > _MAX_EXPANDED:
            _EXPANDED.popitem(last=False)


def expanded(scope: str) -> frozenset[str]:
    with _LOCK:
        return frozenset(h for s, h in _EXPANDED if s == scope)
