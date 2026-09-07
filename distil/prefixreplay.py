"""Forwarded-bytes prefix replay — hold the provider's cache when the CLIENT moves.

ADR 0011. ADR 0008 states distil's half of the cache contract: *when the client
re-sends a message byte-identical, distil forwards it byte-identical.* Clause (d)
then says what happens when the client does not — "a client that rewrites its own
history gets no promise" — and real agentic clients rewrite their history on **every
single turn**, non-semantically:

* the ``cache_control`` breakpoint moves to the newest block (Claude Code's shape),
* SDK shims add positional ``index`` fields to blocks that had none,
* string content becomes a single ``{"type":"text"}`` block, or back again.

None of that changes a token the model reads, and all of it changes the bytes distil
forwards, because distil forwards what it receives. The provider then misses on a
prefix that is semantically identical to the one it already holds, and the whole
prefix is re-billed at the write rate. The failure is silent in exactly the way ADR
0008 describes: every request succeeds, only the bill moves.

So distil remembers, per conversation lineage, the ``(original, forwarded)`` pair from
the previous turn. On the next turn it walks the new originals against the old ones
with a **canonical** comparison that ignores only SDK bookkeeping, and for the longest
canonically-equal leading prefix it forwards **the bytes it forwarded last time**
rather than the bytes it would produce now. The client's current breakpoint markers are
re-placed at the client's current positions, because the marker is not content — moving
it does not invalidate the span it delimits (``prefix._flatten`` has taken the same
position since 1.41).

Prior art, named plainly: Headroom's ``PrefixCacheTracker.overlay_cached_prefix`` does
the same thing, with the same comparison-key/forwarded-bytes separation and the same
conversation-lineage scoping. It is the right mechanism and distil is not going to
pretend otherwise. What differs is the guard below.

**The guard.** Replay only ever restores bytes; it must never restore a *decision*.
distil's compressor is not a pure function of one message — the exact-quote guarantee
(``compress.provenance``) keeps a tool result verbatim because of an ``Edit`` that
arrives LATER in the list, so a block digested at turn N can legitimately need to be
verbatim at turn N+1. Overlaying turn N's stub there would break the agent's next edit
to buy a cache hit. So an item is replayed only when the previous turn's forwarded form
and this turn's forwarded form are **themselves canonically equal**: replay is then a
pure byte restoration, and "prefix replay never changes semantic content" is not a
claim but the loop condition. `distil validate` asserts it as an invariant anyway,
because the loop condition is the kind of thing a later refactor optimises away.

State is in-memory and bounded, and deliberately not persisted: it holds message bytes,
and distil has exactly one place content is allowed to rest on disk (the TTL'd restore
store). A hot-swap therefore starts a lineage cold and rebuilds it from the next turn,
at a cost of one cache write.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Fields an SDK adds, moves, or renumbers without changing what the model reads.
# Kept deliberately short: every name here is a promise that two items differing only
# in it are the same input, and a wrong entry forwards stale content. `citations`,
# `annotations` and friends are NOT here — they may carry meaning, and the cost of
# excluding them is a missed hit, not a wrong prefix.
_NON_SEMANTIC = frozenset({"cache_control", "index"})

# Keys whose values are the agent's own tool payloads — the arguments going out and the
# result coming back. Opaque: compared as they are, never structurally rewritten, because
# the rules above are about MESSAGE structure and neither holds one level down:
#
# * a payload may carry a key called "content" holding a string, and applying the
#   string/text-block sugar there would declare two different tool calls equal;
# * a payload may carry a key called "index" that is DATA. Gemini's
#   `functionResponse.response` is arbitrary tool output, so stripping `index` inside it
#   would call `{"index":1,"value":"A"}` and `{"index":2,"value":"A"}` the same result and
#   forward the first one's bytes for the second — a stale tool result, which is the one
#   failure this module must never produce.
_OPAQUE = frozenset({"input", "arguments", "args", "response", "output"})

# The spellings of "one block of text". Anthropic accepts a bare string or
# ``[{"type":"text",...}]``; the Responses API spells the same block ``input_text`` /
# ``output_text`` and also accepts the bare string. SDK round-trips flip between them.
# The type is dropped from the comparison key only for a SINGLE block that carries
# nothing but a type and a text — never from a multi-block list, and never from a block
# with any other field — so an image, a tool_result or a signed thinking block cannot be
# folded into this. Within one lineage the provider is fixed, so replaying whichever
# spelling that provider already accepted is spelling it the same way twice.
_TEXT_TYPES = frozenset({"text", "input_text", "output_text"})

# Lineages tracked at once. An agent runs one conversation per session; a handful of
# subagents fan out from it. 16 covers that with room to spare and bounds the state at
# roughly one context window each.
_MAX_LINEAGES = 16
# ponytail: a byte ceiling on retained state rather than a real cache. A session whose
# history exceeds this simply stops being replayed (and keeps working); if long-context
# sessions ever need it, spill the prefix to the restore store instead of growing this.
_MAX_STATE_BYTES = 8 * 1024 * 1024

log = logging.getLogger("distil.prefixreplay")


def _text_sugar(value: Any) -> Any:
    """One text block in one spelling, or ``None`` if *value* is not that shape."""
    if isinstance(value, str):
        return [{"text": value}]
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
        and set(value[0]) <= {"type", "text"}
        and value[0].get("type") in _TEXT_TYPES
        and isinstance(value[0].get("text"), str)
    ):
        return [{"text": value[0]["text"]}]
    return None


def _canon(node: Any) -> Any:
    """Structural form of *node* with SDK bookkeeping removed.

    Not for forwarding — only ever for comparison. The result is serialized with
    ``sort_keys=True`` below, so key ORDER is normalised too: re-ordering a JSON object
    does not change what the model reads, and replaying the previous order is precisely
    the repair.
    """
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k in _NON_SEMANTIC:
                continue
            if k in _OPAQUE:
                out[k] = v
            elif k == "content":
                # Canonicalise the blocks FIRST, then look for the sugar. The other
                # way round, a text block carrying a `cache_control` marker has three
                # keys instead of two, fails the shape test, and canonicalises
                # differently from the same block once the marker advances off it —
                # which is the single most common rewrite there is.
                cv = v if isinstance(v, str) else _canon(v)
                sugar = _text_sugar(cv)
                out[k] = cv if sugar is None else sugar
            else:
                out[k] = _canon(v)
        return out
    if isinstance(node, list):
        return [_canon(x) for x in node]
    return node


def canonical(item: Any) -> str:
    """Comparison key for one message/item: what the model reads, nothing else."""
    return json.dumps(_canon(item), sort_keys=True, separators=(",", ":"), default=str)


def _wire(item: Any) -> str:
    """The item as ``proxy._serialize_if_changed`` would encode it. Byte truth, so the
    ``restored`` counter measures repairs that actually happened rather than value
    equality — a pure key re-ordering is a real cache bust and a real repair."""
    return json.dumps(item, separators=(",", ":"), ensure_ascii=False, default=str)


def _reput(node: Dict[str, Any], marker: Any) -> Dict[str, Any]:
    """*node* carrying *marker* as its ``cache_control``, or none if *marker* is not one.

    Returns the node UNTOUCHED when it already carries exactly that marker, and that is
    load-bearing rather than an optimisation: rebuilding the dict moves ``cache_control``
    to the end of the key order, and JSON key order is part of the bytes the provider
    hashes. A block whose marker never moved would otherwise be re-ordered the first
    time it was replayed and bust the very prefix this module exists to hold.
    """
    if node.get("cache_control") == marker:
        return node
    out = {k: v for k, v in node.items() if k != "cache_control"}
    if isinstance(marker, dict):
        out["cache_control"] = marker
    return out


def _remark(replayed: Any, client: Any) -> Any:
    """Previously-forwarded *replayed* item, carrying *client*'s CURRENT breakpoints.

    ``cache_control`` marks where the provider should cut the cached span; it is not
    part of the span's content. Replaying last turn's marker position would pin the
    breakpoint behind the conversation and defeat the point of an agent that advances
    it, so the markers follow the client, block index for block index.
    """
    if not isinstance(replayed, dict) or not isinstance(client, dict):
        return replayed
    out = _reput(replayed, client.get("cache_control"))
    blocks = out.get("content")
    if not isinstance(blocks, list):
        return out
    cc = client.get("content")
    marks: List[Any] = (
        [b.get("cache_control") if isinstance(b, dict) else None for b in cc]
        if isinstance(cc, list)
        else []
    )
    new_blocks: List[Any] = []
    for j, b in enumerate(blocks):
        if not isinstance(b, dict):
            new_blocks.append(b)
            continue
        new_blocks.append(_reput(b, marks[j] if j < len(marks) else None))
    if out is replayed:
        # Copy on write. `replayed` is the PREVIOUS turn's stored item, shared with the
        # lineage and — in a threaded server — with any concurrent request on it, and
        # `_reput` hands it straight back when the message-level marker did not move.
        # Assigning `content` here would write this turn's marker placement into state
        # another request is about to read. Rebuild rather than mutate, preserving key
        # ORDER, because the order is part of the bytes the provider hashes.
        return {k: (new_blocks if k == "content" else v) for k, v in replayed.items()}
    out["content"] = new_blocks
    return out


@dataclass
class ReplayStats:
    """Content-free: three counts, no text.

    ``hits`` without ``restored`` is the healthy steady state — the client re-sent the
    prefix byte-identical and there was nothing to repair. ``restored`` is the only one
    that means money changed hands, so it is reported separately rather than folded in.
    """

    hits: int = 0  # leading items forwarded as previously sent
    misses: int = 0  # items past the divergence point, compressed fresh
    restored: int = 0  # of the hits, how many differed from what we'd have sent now


@dataclass
class _Lineage:
    canon: List[str] = field(default_factory=list)  # per original item
    forwarded: List[Any] = field(default_factory=list)  # what actually went on the wire
    fwd_canon: List[str] = field(default_factory=list)  # lazily filled, index-aligned


_LINEAGES: "OrderedDict[str, _Lineage]" = OrderedDict()
_LOCK = threading.Lock()


def reset() -> None:
    """Drop all lineage state (tests, and `distil doctor`-style maintenance)."""
    with _LOCK:
        _LINEAGES.clear()


def lineage_key(body: Mapping[str, Any], items: List[Any]) -> str:
    """A content-free id for "the same conversation, same model, same tools".

    ADR 0008's boundary is per-provider but its *scope* is not: a cached prefix belongs
    to one leading system-run + model + tool set. Change any of them and the provider
    holds a different entry, so replaying across that change would forward the wrong
    history. The conversation's first item pins the lineage the way
    ``cachedelta.session_key`` already does; it is canonicalised here so that a marker
    landing on the head message does not fork the lineage every turn.
    """
    seed = {
        "model": body.get("model"),
        "system": body.get("system") or body.get("instructions") or body.get("systemInstruction"),
        "tools": body.get("tools") or body.get("toolConfig"),
        "head": canonical(items[0]) if items else None,
        "sid": os.environ.get("DISTIL_SESSION", ""),
    }
    blob = json.dumps(seed, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]


def _get(key: str) -> Optional[_Lineage]:
    with _LOCK:
        lin = _LINEAGES.get(key)
        if lin is not None:
            _LINEAGES.move_to_end(key)
        return lin


def _put(key: str, lin: _Lineage) -> None:
    with _LOCK:
        _LINEAGES[key] = lin
        _LINEAGES.move_to_end(key)
        while len(_LINEAGES) > _MAX_LINEAGES:
            _LINEAGES.popitem(last=False)


def replay(key: str, original: List[Any], forwarded: List[Any]) -> Tuple[List[Any], ReplayStats]:
    """Forward the previous turn's bytes for the canonically-equal leading prefix.

    Returns ``(items_to_forward, stats)`` and records this turn as the new state. The
    walk stops at the FIRST divergence and never resumes: a cached prefix is a byte
    prefix, so an item restored after a break repairs nothing and only risks pairing
    old bytes with a history the client has since edited (contract clause (d) — that
    divergence is the client's rewrite and is forwarded exactly as it arrived).
    """
    stats = ReplayStats(misses=len(forwarded))
    if not isinstance(original, list) or not isinstance(forwarded, list):
        return forwarded, stats

    cur_canon = [canonical(m) for m in original]
    if sum(len(c) for c in cur_canon) > _MAX_STATE_BYTES:
        # Too big to remember. Forward as compressed and forget the lineage rather
        # than hold a context window per conversation forever.
        with _LOCK:
            _LINEAGES.pop(key, None)
        return forwarded, stats

    prev = _get(key)
    out = list(forwarded)
    # Index-aligned with `out`, and only as long as the replayed prefix: past the
    # divergence these keys are never consulted, so computing them would be work spent
    # on every turn of every session to answer a question nobody asks.
    fwd_canon: List[str] = []
    if prev is not None:
        prev_fwd_canon = list(prev.fwd_canon)  # local: another thread may hold `prev`
        limit = min(len(cur_canon), len(prev.canon), len(forwarded), len(prev.forwarded))
        for i in range(limit):
            if cur_canon[i] != prev.canon[i]:
                break  # the client changed this item — stop, and stay stopped
            if i >= len(prev_fwd_canon):
                prev_fwd_canon.append(canonical(prev.forwarded[i]))
            cur_key = canonical(forwarded[i])
            if cur_key != prev_fwd_canon[i]:
                # We would send something semantically different this turn (an
                # exact-quote exemption came into range, say). Replay restores bytes,
                # never decisions — so this is a divergence like any other.
                break
            item = _remark(prev.forwarded[i], original[i])
            if _wire(item) != _wire(forwarded[i]):
                stats.restored += 1
            out[i] = item
            fwd_canon.append(cur_key)
            stats.hits += 1
        stats.misses = len(forwarded) - stats.hits

    _put(key, _Lineage(canon=cur_canon, forwarded=out, fwd_canon=fwd_canon))
    return out, stats


def apply(
    body: Dict[str, Any],
    list_key: str,
    original: List[Any],
    *,
    scope: str = "",
    extras: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Replay the prefix of ``body[list_key]`` and record the counters in *extras*.

    The whole integration, in one place. All three servers — the threaded proxy, the
    async proxy, and the multi-tenant gateway — call this at the same point: on the
    final forwarded body, after every transform, immediately before serialization.
    Splitting the fail-open across three copies is how one of them ends up without it.

    *scope* prefixes the lineage key. The gateway passes the tenant, so two tenants
    posting the same conversation never share replay state; a cached prefix is the
    provider's, per credential, and crossing that boundary would forward one tenant's
    bytes into another's request.
    """
    try:
        forwarded = body.get(list_key)
        if not isinstance(forwarded, list):
            return body
        out, stats = replay(scope + lineage_key(body, original), original, forwarded)
        if extras is not None:
            extras["x-distil-replay-hits"] = str(stats.hits)
            extras["x-distil-replay-misses"] = str(stats.misses)
            extras["x-distil-replay-restored"] = str(stats.restored)
        # Only rebuild the body when bytes actually changed, so an unmodified request
        # keeps whatever fast path its server has for forwarding the original bytes.
        return {**body, list_key: out} if stats.restored else body
    except Exception:  # noqa: BLE001 — never break a request for a cache hit
        log.debug("prefix replay failed; forwarding as compressed", exc_info=True)
        return body
