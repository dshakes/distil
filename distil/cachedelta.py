"""Cache-delta context coding — cross-VERSION delta, decision-equivalence-anchored.

The coding-agent hot path is **read → edit → RE-READ**. The re-read file is *not*
byte-identical to the first read (a hunk changed), so exact-duplicate dedup — which
is the state of the art elsewhere (e.g. Headroom's ``dedup_identical_items``) —
*misses it* and re-sends the whole file as fresh tokens. Cache-delta coding instead
sends only the **diff** against the previously-delivered version, referencing the
rest.

Why this is decision-equivalent (the motto): the prior version is still present
earlier in the (cached) conversation, so *prior-version + diff* carries exactly the
information the agent needs for its next decision — its chosen action is unchanged.
It is **reversible** (the full new content is kept locally; ``distil_expand``
recovers it byte-exact) and **measurable** (shadow mode records the live
decision-change rate), so the equivalence is proven, not asserted.

Two cache-safe wins, both confined to the **volatile suffix** — the stable,
already-cached prefix is never mutated (*cache-monotonicity*), so prompt-cache hits
are preserved:

* exact re-send   → a compact reference to the prior handle (table stakes).
* near-duplicate  → a reference + a unified diff of what changed (the invention).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from .adapters.anthropic import RestoreStore
from .pricing import Pricing
from .tokenizer import DEFAULT as _tok

# Only blocks at least this large are worth referencing/delta-coding.
_MIN_CHARS = 400
# A candidate counts as a near-duplicate of a prior block at/above this similarity.
_NEAR_DUP_RATIO = 0.5
# How many recent large blocks to consider as delta bases (bounds latency).
_MAX_BASES = 12


def _handle(text: str) -> str:
    """8-hex content handle — mirrors the adapter / tier1 so expand is uniform."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _msg_hash(msg: Any) -> str:
    try:
        return hashlib.sha256(
            json.dumps(msg, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
    except (TypeError, ValueError):
        return hashlib.sha256(str(msg).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Per-session state (process-level; the proxy is otherwise stateless)
# ---------------------------------------------------------------------------


@dataclass
class DeltaSession:
    """Per-session memory of the prior turn's message-hash prefix (content-free)."""

    prev_msg_hashes: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


_SESSIONS: OrderedDict[str, DeltaSession] = OrderedDict()
_SESSIONS_LOCK = threading.Lock()
_MAX_SESSIONS = 512


def session_key(messages: list[Any]) -> str:
    """A stable per-session key derived from the conversation's seed (first turn).

    The first message is byte-stable for the life of a session and distinct across
    sessions, so it identifies the session without any client cooperation.
    """
    seed = ""
    for m in messages[:1]:
        seed = json.dumps(m, sort_keys=True, default=str) if isinstance(m, (dict, list)) else str(m)
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def get_session(key: str) -> DeltaSession:
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(key)
        if sess is None:
            sess = DeltaSession()
            _SESSIONS[key] = sess
        _SESSIONS.move_to_end(key)
        while len(_SESSIONS) > _MAX_SESSIONS:
            _SESSIONS.popitem(last=False)
        return sess


def reset_sessions() -> None:
    """Clear all session state (test/maintenance helper)."""
    with _SESSIONS_LOCK:
        _SESSIONS.clear()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    prefix_msgs: int = 0  # cache-stable messages left byte-identical
    exact_refs: int = 0  # blocks replaced by an exact back-reference
    delta_refs: int = 0  # blocks replaced by a cross-version diff
    tokens_saved: int = 0  # volatile (fresh-billed) tokens removed

    def dollars_saved(self, pricing: Pricing) -> float:
        # Deduped/delta'd content would have been billed as fresh input tokens.
        return self.tokens_saved * pricing.input


# ---------------------------------------------------------------------------
# Reference markers (decision-equivalent, expand-recoverable)
# ---------------------------------------------------------------------------


def _exact_marker(handle: str, nlines: int) -> str:
    return (
        f"«distil-ref handle={handle}» identical content was already provided "
        f"earlier in this session ({nlines} lines). Call distil_expand with this handle "
        f"to recover it verbatim."
    )


def _delta_marker(base_handle: str, new_handle: str, diff: str, nlines: int) -> str:
    return (
        f"«distil-delta base={base_handle} handle={new_handle}» this is the content "
        f"you saw as {base_handle} ({nlines} lines) with the following changes:\n{diff}\n"
        f"Call distil_expand with handle={new_handle} for the full current version."
    )


# ---------------------------------------------------------------------------
# Core: encode one text block against session memory
# ---------------------------------------------------------------------------


def _best_base(text: str, bases: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Return the (handle, text) of the most similar prior block, or None.

    Cheap prefilter (length ratio + difflib quick_ratio) before the full ratio, so
    the near-duplicate search stays inexpensive even with many candidates.
    """
    best: tuple[float, str, str] | None = None
    for h, btext in bases:
        if not btext:
            continue
        lr = min(len(text), len(btext)) / max(len(text), len(btext))
        if lr < _NEAR_DUP_RATIO:
            continue
        sm = difflib.SequenceMatcher(None, btext, text)
        if sm.quick_ratio() < _NEAR_DUP_RATIO:
            continue
        ratio = sm.ratio()
        if ratio >= _NEAR_DUP_RATIO and (best is None or ratio > best[0]):
            best = (ratio, h, btext)
    return (best[1], best[2]) if best else None


def _encode_block(
    text: str,
    *,
    prior: dict[str, str],
    bases: list[tuple[str, str]],
    store: RestoreStore,
    stats: CacheStats,
) -> str:
    """Replace *text* with a reference/delta if it repeats prior content; else keep it.

    ``prior`` is the snapshot of blocks delivered *before this turn* (exact match);
    ``bases`` is the same as (handle, text) pairs for near-duplicate search.
    """
    if len(text) < _MIN_CHARS:
        return text

    h = _handle(text)
    nlines = text.count("\n") + 1

    # 1) Exact re-send → back-reference.
    if h in prior:
        store._record(h, text)
        exact = _exact_marker(h, nlines)
        if len(exact) < len(text):
            stats.exact_refs += 1
            stats.tokens_saved += max(0, _tok.count(text) - _tok.count(exact))
            return exact
        return text

    # 2) Near-duplicate (e.g. a file re-read after an edit) → reference + delta.
    base = _best_base(text, bases)
    if base is not None:
        base_h, base_text = base
        # Prefer an AST-structural delta (formatting/comment/import-order invariant);
        # fall back to a textual unified diff for non-Python / unparseable source.
        from .astdelta import structural_delta

        marker = structural_delta(base_text, text, base_h, h)
        if marker is None:
            diff = "".join(
                difflib.unified_diff(
                    base_text.splitlines(keepends=True),
                    text.splitlines(keepends=True),
                    fromfile=base_h,
                    tofile="current",
                    n=2,
                )
            )
            cand = _delta_marker(base_h, h, diff, nlines)
            marker = cand if len(cand) < len(text) else None
        if marker is not None and len(marker) < len(text):
            store._record(h, text)  # full current version recoverable by expand
            stats.delta_refs += 1
            stats.tokens_saved += max(0, _tok.count(text) - _tok.count(marker))
            return marker

    return text


# ---------------------------------------------------------------------------
# Message-content walking
# ---------------------------------------------------------------------------


def _rewrite_tool_texts(
    msg: Any, transform: Callable[[str], str], keep_ids: frozenset[str] = frozenset()
) -> Any:
    """Apply *transform* to every large tool_result text in *msg* (non-mutating).

    Mirrors the adapter's block model: string tool/user content and ``tool_result``
    blocks (string or list-of-text). Returns the same object when nothing changed.

    Blocks whose tool id is in *keep_ids* are skipped: those are the results the agent
    must be able to quote back byte-exact (see ``compress.provenance``). A delta
    reference is not a byte-exact copy, so encoding one there voids the same guarantee
    the compressor is about to honour — which is exactly what ``--session-delta`` did.
    """
    if not isinstance(msg, dict):
        return msg
    role = msg.get("role", "")
    content = msg.get("content")

    if isinstance(content, str):
        if role in ("tool", "user") and msg.get("tool_call_id") not in keep_ids:
            new = transform(content)
            if new != content:
                return {**msg, "content": new}
        return msg

    if isinstance(content, list):
        new_list: list[Any] = []
        changed = False
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") not in keep_ids
            ):
                bc = block.get("content")
                if isinstance(bc, str):
                    nb = transform(bc)
                    if nb != bc:
                        new_list.append({**block, "content": nb})
                        changed = True
                        continue
                elif isinstance(bc, list):
                    sub_list: list[Any] = []
                    sub_changed = False
                    for sub in bc:
                        if (
                            isinstance(sub, dict)
                            and sub.get("type") == "text"
                            and isinstance(sub.get("text"), str)
                        ):
                            nt = transform(sub["text"])
                            if nt != sub["text"]:
                                sub_list.append({**sub, "text": nt})
                                sub_changed = True
                                continue
                        sub_list.append(sub)
                    if sub_changed:
                        new_list.append({**block, "content": sub_list})
                        changed = True
                        continue
            new_list.append(block)
        if changed:
            return {**msg, "content": new_list}
    return msg


def _collect_texts(msg: Any) -> list[str]:
    """All large tool_result texts in a message (for registering into session memory)."""
    out: list[str] = []

    def _grab(t: str) -> str:
        if len(t) >= _MIN_CHARS:
            out.append(t)
        return t

    _rewrite_tool_texts(msg, _grab)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def delta_encode(
    messages: list[Any],
    *,
    session: DeltaSession | None = None,
    store: RestoreStore | None = None,
    keep_ids: frozenset[str] = frozenset(),
) -> tuple[list[Any], RestoreStore, CacheStats]:
    """Cross-turn delta-encode *messages* against *session* memory (non-mutating).

    Cache-monotonic: the maximal stable prefix (identical to the previous turn) is
    left byte-identical, so prompt-cache hits survive; only the volatile suffix is
    touched. Returns ``(new_messages, store, stats)``; ``store`` gains a handle for
    every reference/delta so ``distil_expand`` can recover the original.

    *keep_ids* names the tool results the exact-quote guarantee will keep verbatim; they
    are left untouched here so the two mechanisms cannot contradict each other. They are
    still registered as delta *bases*, so later blocks keep deduping against them.
    """
    store = store if store is not None else RestoreStore()
    stats = CacheStats()
    if not isinstance(messages, list):
        return messages, store, stats

    cur_hashes = [_msg_hash(m) for m in messages]
    # Prefix stat (informational): longest common prefix with the previous turn.
    if session is not None:
        with session.lock:
            prev = session.prev_msg_hashes
            lcp = 0
            limit = min(len(cur_hashes), len(prev))
            while lcp < limit and cur_hashes[lcp] == prev[lcp]:
                lcp += 1
            stats.prefix_msgs = lcp
            session.prev_msg_hashes = cur_hashes

    # Encoding is a PURE per-call walk over the cumulative messages: message i is
    # deduped/delta'd only against blocks in messages[0..i-1]. Because the cached
    # prefix is byte-identical across turns, each prefix message encodes to the SAME
    # bytes every turn — so cache-delta is cache-monotonic BY CONSTRUCTION (no
    # marker<->original flip), and the store is fully repopulated here so expand
    # always resolves. The cumulative conversation is the memory; no cross-call
    # state is needed for correctness.
    delivered: dict[str, str] = {}
    new_messages: list[Any] = []
    for msg in messages:
        bases = list(delivered.items())[-_MAX_BASES:]

        def _transform(text: str, _prior: dict[str, str] = delivered, _bases: Any = bases) -> str:
            return _encode_block(text, prior=_prior, bases=_bases, store=store, stats=stats)

        new_messages.append(_rewrite_tool_texts(msg, _transform, keep_ids))
        for t in _collect_texts(msg):  # register AFTER, so a block can't dedup itself
            delivered[_handle(t)] = t

    return new_messages, store, stats
