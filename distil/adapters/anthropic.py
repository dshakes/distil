"""Anthropic Messages API runtime adapter — Phase 3 of the distil roadmap.

Compresses an in-flight Messages API request with no caller code change:

  client = anthropic.Anthropic(...)
  client = distil.adapters.anthropic.wrap(client)
  # all subsequent client.messages.create(...) calls are transparently compressed

Design decisions
----------------
* No `anthropic` import at module level — the adapter is duck-typed so it works
  even if the `anthropic` SDK is not installed (e.g. in test environments).
* Only `tool_result` blocks with >= 6 lines are digested (Tier 1 / reversible).
  Plain `text` blocks get Tier 0 transforms (minify_json / collapse_runs).
  `tool_use`, `image`, and assistant text blocks are passed through unchanged.
* `RestoreStore` keeps originals keyed by the 8-hex handle that `tier1.digest`
  embeds in its marker lines, so callers can always recover the full content.
* Cache-control placement: marking the last stable system block (or the system
  string itself) as `{"cache_control": {"type": "ephemeral"}}` pins a cacheable
  prefix. Reads of that prefix are billed at ~0.1x vs. a full write, so every
  repeated call after the first amortises the system-prompt tokens cheaply.
  The prefix must be *stable* across turns (same bytes) to get a cache hit —
  hence we mark the *last* system block rather than anything in the volatile
  message history.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from ..compress.tier0 import collapse_runs, minify_json
from ..mcp_server import load_restore as _load_restore
from ..mcp_server import record_restore as _record_restore
from ..compress.intent import extract_intent
from ..compress import vision as _vision
from ..compress.tier1 import digest as _tier1_digest
from ..tokenizer import DEFAULT as _tokenizer

# Minimum line count for a tool_result to be digested (matches Tier1Reversible default).
_MIN_LINES = 6

# Recency exemption: tool_result blocks in the last K user/tool turns are NEVER
# digested — an agent must always see its most recent tool outputs byte-exact to
# choose its next action, and a Tier-1 stub it may not be able to expand there
# would break that. The rule itself lives in compress.recency and is shared with
# the certified strategy, so both paths make the same keep/digest decisions.
from ..compress.recency import RECENCY_KEEP_TURNS as _RECENCY_KEEP_TURNS  # noqa: E402

# Thread-local learned "keep byte-exact" predicate, scoped per compress_messages call
# (ThreadingHTTPServer handles requests on separate threads, so this must be per-thread).
import threading as _threading  # noqa: E402

_keep_tls = _threading.local()


def _active_keep(text: str) -> bool:
    fn = getattr(_keep_tls, "fn", None)
    return bool(fn and fn(text))


# Query-aware salience: the request's intent terms, set once per compress_messages call
# (it holds the whole conversation) and read by the tool_result digester. Thread-local so
# the per-block compressor need not thread a new arg through its recursion.
_intent_tls = _threading.local()


def _active_intent() -> frozenset[str]:
    return getattr(_intent_tls, "terms", frozenset())


# Vision duplicate-elision state for this request (ADR 0003). Thread-local for the
# same reason as the intent terms: the per-block compressor recurses, and threading
# a new arg through every call site would touch code that has nothing to do with
# images. None = disabled, which is the default until `vision` is certified.
_vision_tls = _threading.local()


def _active_vision() -> Any:
    return getattr(_vision_tls, "dedup", None)


def _recent_verbatim_indices(messages: list[dict[str, Any]], k: int) -> set[int]:
    """Indices of the last *k* tool-output-bearing turns (role ``user``/``tool``),
    whose tool_result blocks must stay verbatim. See ``_RECENCY_KEEP_TURNS``."""
    if k <= 0:
        return set()
    idxs = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") in ("user", "tool")
    ]
    return set(idxs[-k:])


# ---------------------------------------------------------------------------
# RestoreStore
# ---------------------------------------------------------------------------


class RestoreStore:
    """Maps 8-hex handles -> original text so callers can reverse any digest.

    The store is populated by `compress_messages` and is local — it is never
    sent to the model, so it costs zero tokens.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal (used by compress_messages)
    # ------------------------------------------------------------------

    def _record(self, handle: str, original: str) -> bool:
        """Record ``handle -> original``. Returns ``False`` on an 8-hex COLLISION —
        the handle already maps to *different* text. The caller must then NOT emit a
        digest stub for this block: its handle would resolve to the other block's
        content, so an expand would return the wrong bytes. Keeping the block verbatim
        is always safe. Idempotent re-records of the same text return ``True``.
        """
        existing = self._store.get(handle)
        if existing is not None and existing != original:
            return False
        self._store[handle] = original
        _record_restore(handle, original)  # survive restarts; expandable cross-process
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def handles(self) -> frozenset[str]:
        """All handles currently registered in this store."""
        return frozenset(self._store)

    def expand(self, handle: str) -> str:
        """Return the original text for *handle*.

        Raises KeyError if the handle is not in this store.
        """
        try:
            return self._store[handle]
        except KeyError:
            original = _load_restore(handle)  # disk fallback: pre-restart handles
            if original is None:
                raise
            self._store[handle] = original
            return original


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _handle(text: str) -> str:
    """8-hex SHA-256 prefix — mirrors tier1._handle exactly."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _apply_tier0(text: str) -> str:
    """Apply lossless Tier-0 transforms: JSON minification then run collapse.

    Run-collapse is reject-if-bigger **by tokens** (what we bill): collapsing a run
    of near-free whitespace/blank lines into a ``<<x N>>`` count marker can cost
    *more* tokens than it removes, so we only keep the collapse when it actually
    reduces the token count — Tier-0 must never inflate.
    """
    mj = minify_json(text)
    base = mj if mj is not None else text
    collapsed = collapse_runs(base)
    if collapsed != base and _tokenizer.count(collapsed) <= _tokenizer.count(base):
        return collapsed
    return base


def _lossless_fold(text: str) -> str | None:
    """In-context-lossless structured compaction with NO recovery handle: a
    self-describing columnar table (JSON array of flat records) or templated run.
    Every value stays inline and the model reads it directly, so it is safe on the
    lossless/subscription path (no expand tool injected). Returns None when the
    content isn't tabular/repetitive. Inherits fold's decision-equivalence — the
    table is byte-identical to the handle-bearing fold minus a metadata token."""
    from ..compress.structured import (
        fold,
        fold_records,
        template_fold,
    )  # local: avoid load-time cycle

    return (
        fold(text, emit_handle=False)
        or fold_records(text, emit_handle=False)
        or template_fold(text, emit_handle=False)
    )


def _compress_text_content(text: str, store: RestoreStore, verbatim: bool) -> str:
    """Apply only Tier-0 lossless transforms to a plain text block."""
    return _apply_tier0(text)


def _compress_tool_result_text(
    text: str, store: RestoreStore, verbatim: bool = False, is_recent: bool = False
) -> str:
    """Digest a large tool_result string and record the original in *store*.

    In ``verbatim`` mode only *in-context-lossless* Tier-0 transforms are applied:
    the model sees semantically identical content (minified JSON, collapsed exact-
    duplicate runs), never a Tier-1 digest stub. Use it for interactive sessions or
    out-of-distribution traffic — anywhere the model must reason over the real
    content rather than recover it via a tool. The default (digest) is reversible
    and decision-equivalent by the certificate; ``verbatim`` trades that for
    byte-in-context fidelity at lower savings.

    ``is_recent`` marks a recency-exempt block (the last few tool turns): the agent
    must see its most recent output byte-exact to choose its next action, so those
    stay verbatim with NO fold — even a lossless columnar fold changes the bytes.
    """
    # Learned policy: if your agents keep expanding this kind of content, keep it
    # byte-exact (strictly safer — only ever reduces savings, never equivalence).
    # Hoisted above the HTML step so both digest paths honour it, and evaluated once.
    keep_byte_exact = _active_keep(text)

    # HTML from a fetch/browser tool. Handled BEFORE the _MIN_LINES gate because
    # minified markup is a single enormous line, so that gate would fall through to
    # Tier-0-only and save nothing (measured: 0.0% on a realistic 8.3k-token page).
    # Requires a handle to stay recoverable, so it is skipped when verbatim (no expand
    # tool available) and when recent (must stay byte-exact for the next decision).
    if not verbatim and not is_recent and not keep_byte_exact:
        from ..compress.htmlx import extract as _html_extract

        stripped = _html_extract(text)
        if stripped is not None:
            h = _handle(text)
            if store._record(h, text):
                return f"{stripped}\n<< html chrome elided, handle={h} >>"

    lines = text.splitlines()
    if len(lines) < _MIN_LINES:
        # Too short to digest — lossless Tier-0 transforms only.
        return _apply_tier0(text)

    # Verbatim + not recent (a subscription/lossless OLDER block): a self-describing
    # columnar/template fold is in-context-lossless (all data inline, no recovery handle
    # to invite an unavailable distil_expand), so it is safe and saves far more than
    # tier-0 alone on the tabular tool output subscription users see. Recent blocks stay
    # byte-exact (no fold) so the agent's latest output is unchanged.
    if verbatim:
        folded = None if is_recent else _lossless_fold(text)
        return folded or _apply_tier0(text)

    if keep_byte_exact:
        return _apply_tier0(text)

    digested, changed = _tier1_digest(text, intent=_active_intent())
    if changed:
        h = _handle(text)
        if store._record(h, text):
            return digested
        # 8-hex collision with a different block — a stub here would expand to the
        # wrong content, so keep this block byte-exact (lossless transforms only).
    return _apply_tier0(text)


def _compress_image_block(
    item: dict[str, Any], store: RestoreStore, verbatim: bool, is_recent: bool
) -> dict[str, Any]:
    """Elide a REPEATED image, byte-reversibly. See distil/compress/vision.py.

    A first occurrence is always sent verbatim — the model has to actually see
    the image. Only a byte-identical repeat becomes a reference stub, and only
    when the vision content type has been certified (ADR 0003).

    Skipped entirely in verbatim mode (its contract is that the model sees
    semantically identical content, and a reference is not that) and on recent
    turns (the recency rule: never make the agent reason blind over its freshest
    input). Both cases still *note* the payload, so a later duplicate is still
    recognized as a repeat rather than mistaken for a first sighting.
    """
    dedup = _active_vision()
    if dedup is None:
        return item
    source = item.get("source")
    if not isinstance(source, dict):
        return item  # malformed/unknown shape — pass through untouched

    if verbatim or is_recent:
        dedup.note(source)
        return item

    verdict = dedup.elide(source)
    if verdict is None:
        # Not a repeat, so the model has not seen these pixels yet and the block
        # must carry an actual image. It may still be oversized — see
        # _downscale_image_block, which is off unless separately certified.
        return _downscale_image_block(item, source, store, dedup)
    handle, original, tokens = verdict
    if not store._record(handle, original):
        # 8-hex collision with different content — a stub would expand to the
        # wrong image. Keeping the block verbatim is always safe (same rule the
        # text digester follows).
        return item
    return {"type": "text", "text": _vision.reference_text(handle, tokens)}


def _downscale_image_block(
    item: dict[str, Any], source: dict[str, Any], store: RestoreStore, dedup: Any
) -> Any:
    """Send a first-occurrence image at reduced resolution, original recoverable.

    Returns the block unchanged, or a PAIR — the downscaled image followed by a
    note stating what changed and the handle that returns the original. The note
    is not decoration: without it the transform is silently lossy, and silent
    loss is the one thing distil does not do.

    Off unless `distil certify --strategy vision-downscale` has passed locally
    AND a codec is installed. Both, independently — an enabled feature with no
    codec must not half-work. See distil/compress/vision_scale.py for why this
    one does not inherit the bundled certificate.
    """
    import base64 as _b64
    import json as _json

    from ..compress import vision_scale as _scale

    if not _scale.active():
        return item
    decided = _scale.plan(source)
    if decided is None:
        return item
    new_raw, after, saved = decided
    before = _vision.image_dims(_vision._decode_b64(source.get("data")) or b"") or (0, 0)

    original = _json.dumps(source, sort_keys=True)
    handle = _vision._handle(original)
    if not store._record(handle, original):
        # 8-hex collision with different content: expand would return the wrong
        # image. Leaving the block verbatim is always safe.
        return item
    # Record the saving through the same counter the elision path uses, so
    # `dissect`/savings see one consistent image-tokens number.
    dedup.tokens_saved += saved

    scaled = {
        **item,
        "source": {**source, "data": _b64.b64encode(new_raw).decode("ascii")},
    }
    return [scaled, {"type": "text", "text": _scale.note_text(handle, before, after, saved)}]


def _compress_content_item(
    item: dict[str, Any], store: RestoreStore, role: str, verbatim: bool, is_recent: bool = False
) -> dict[str, Any]:
    """Return a (possibly new) content block after compression.

    Rules:
    - tool_use / image: pass through unchanged.
    - assistant text: pass through unchanged.
    - user text: Tier-0 lossless transforms.
    - tool_result (any role): digest large string content; recurse into list content.
    """
    btype = item.get("type", "")

    # Never touch tool_use blocks.
    if btype == "tool_use":
        return item

    if btype == "image":
        return _compress_image_block(item, store, verbatim, is_recent)

    if btype == "text":
        if role == "assistant":
            # Never rewrite the assistant's own words.
            return item
        text = item.get("text")
        if not isinstance(text, str):
            return item  # malformed/absent text — pass through untouched
        new_text = _compress_text_content(text, store, verbatim)
        if new_text == text:
            return item
        return {**item, "text": new_text}

    if btype == "tool_result":
        content = item.get("content")
        if content is None:
            return item

        if isinstance(content, str):
            new_content = _compress_tool_result_text(content, store, verbatim, is_recent)
            if new_content == content:
                return item
            return {**item, "content": new_content}

        if isinstance(content, list):
            new_list: list[Any] = []
            changed = False
            for sub in content:
                if (
                    isinstance(sub, dict)
                    and sub.get("type") == "text"
                    and isinstance(sub.get("text"), str)
                ):
                    new_text = _compress_tool_result_text(sub["text"], store, verbatim, is_recent)
                    if new_text != sub["text"]:
                        new_list.append({**sub, "text": new_text})
                        changed = True
                    else:
                        new_list.append(sub)
                elif isinstance(sub, dict) and sub.get("type") == "image":
                    # Where computer-use / browser screenshots actually live: a
                    # tool_result whose content list carries the image block.
                    new_sub = _compress_image_block(sub, store, verbatim, is_recent)
                    new_list.append(new_sub)
                    changed = changed or new_sub is not sub
                else:
                    new_list.append(sub)
            if not changed:
                return item
            return {**item, "content": new_list}

    # Unknown block type — leave untouched.
    return item


def _compress_message(
    msg: dict[str, Any], store: RestoreStore, verbatim: bool, is_recent: bool = False
) -> dict[str, Any]:
    """Return a (possibly new) message dict after compressing its content."""
    role = msg.get("role", "")
    content = msg.get("content")

    if isinstance(content, str):
        if role == "assistant":
            return msg
        # OpenAI tool-result messages ({"role":"tool","content":"…"}) get the same
        # decision-aware reversible digest as Anthropic tool_result blocks; other
        # string content gets Tier-0 lossless transforms.
        if role == "tool":
            new_text = _compress_tool_result_text(content, store, verbatim, is_recent)
        else:
            new_text = _compress_text_content(content, store, verbatim)
        if new_text == content:
            return msg
        return {**msg, "content": new_text}

    if isinstance(content, list):
        new_blocks: list[Any] = []
        changed = False
        for item in content:
            if isinstance(item, dict):
                new_item = _compress_content_item(item, store, role, verbatim, is_recent)
                if isinstance(new_item, list):
                    # A block may expand into a PAIR (a downscaled image plus the
                    # note carrying its recovery handle). Splice rather than nest:
                    # a list inside `content` is not a shape the provider accepts.
                    new_blocks.extend(new_item)
                    changed = True
                else:
                    new_blocks.append(new_item)
                    if new_item is not item:
                        changed = True
            else:
                new_blocks.append(item)
        if not changed:
            return msg
        return {**msg, "content": new_blocks}

    return msg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compress_messages(
    messages: list[dict[str, Any]],
    *,
    verbatim: bool = False,
    keep: Any = None,
) -> tuple[list[dict[str, Any]], RestoreStore]:
    """Compress an Anthropic Messages API messages list in place (non-mutating).

    Parameters
    ----------
    messages:
        The ``messages`` kwarg value as passed to ``client.messages.create``.
        Each element is ``{"role": ..., "content": str | list[block]}``.
    verbatim:
        When *True*, only *in-context-lossless* Tier-0 transforms are applied — the
        model sees semantically identical content, never a Tier-1 digest stub. The
        right mode for interactive (human-in-the-loop) sessions, out-of-distribution
        traffic, or anywhere recovery (the ``distil_expand`` tool) is unavailable.
        When *False* (the default), large tool results are replaced by reversible
        Tier-1 digests — decision-equivalent by the certificate, recoverable via the
        RestoreStore / ``distil_expand`` — for far higher savings.

    Returns
    -------
    (new_messages, store)
        ``new_messages`` is a new list (input is not mutated).
        ``store`` maps every 8-hex handle embedded in digest markers back to the
        original text; call ``store.expand(handle)`` to recover it.
    """
    _keep_tls.fn = keep  # learned keep-byte-exact policy for this call (per-thread)
    _intent_tls.terms = frozenset() if verbatim else extract_intent(messages)
    # Vision duplicate elision (ADR 0003) — None unless the content type has been
    # certified, so the default path is byte-for-byte what it was before.
    _vision_tls.dedup = _vision.ImageDedup() if (not verbatim and _vision.enabled()) else None
    try:
        store = RestoreStore()
        new_messages: list[dict[str, Any]] = []
        recent = _recent_verbatim_indices(messages, _RECENCY_KEEP_TURNS)
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                new_messages.append(msg)  # malformed entry — pass through untouched
                continue
            # Force verbatim for the most recent turns so their tool_results are
            # never replaced by a digest stub the agent must reason over blind.
            msg_verbatim = verbatim or idx in recent
            new_messages.append(
                _compress_message(msg, store, msg_verbatim, is_recent=idx in recent)
            )
        return new_messages, store
    finally:
        _keep_tls.fn = None
        _intent_tls.terms = frozenset()
        _vision_tls.dedup = None


def place_cache_control(
    system: list[dict[str, Any]] | str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a kwargs dict that pins the cacheable prefix via cache_control.

    Anthropic's prompt caching works by marking the *boundary* of the stable
    prefix with ``{"cache_control": {"type": "ephemeral"}}``.  A cache *hit*
    costs ~0.1x compared to a full write, so even a single repeated call pays
    back the marking overhead.

    The stable prefix must be byte-identical across calls to get a hit.
    System blocks are a natural boundary — they rarely change between turns —
    so we mark the **last** system block.  Message history is volatile (each
    turn adds a new assistant/user pair), so it is intentionally left outside
    the cached prefix.

    Parameters
    ----------
    system:
        The ``system`` kwarg: either a plain string or a list of content blocks
        (``{"type": "text", "text": ..., ...}``).
    messages:
        The (already compressed) ``messages`` list.

    Returns
    -------
    A dict ready to be spread into ``client.messages.create(**kwargs)``:
    ``{"system": <marked_system>, "messages": messages}``.
    """
    _cc: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}

    if isinstance(system, str):
        # Promote the bare string to a single cacheable block.
        marked_system: list[dict[str, Any]] | str = [{"type": "text", "text": system, **_cc}]
    elif isinstance(system, list) and system:
        # Deep-copy so we do not mutate the caller's list, then mark the last block.
        marked_system = copy.deepcopy(system)
        marked_system[-1].update(_cc)  # type: ignore[union-attr]
    else:
        marked_system = system

    return {"system": marked_system, "messages": messages}


# ---------------------------------------------------------------------------
# Proxy wrapper
# ---------------------------------------------------------------------------


class _MessagesProxy:
    """Thin proxy for ``client.messages`` that compresses before delegating.

    Compresses in **verbatim** mode: only in-context-lossless Tier-0 transforms
    (minified JSON, collapsed duplicate runs) are applied — never a Tier-1 digest stub.

    Why verbatim and not the higher-savings digest: recovering a digest stub needs the
    server-side ``distil_expand`` loop (see :mod:`distil.expand`), which lives in the
    proxy/gateway that own the full request→response→re-query cycle. This in-process SDK
    wrapper delegates a single ``create`` call to the real client and hands the response
    straight back to the caller's own loop; it never sees the model's follow-up turns, so
    it cannot resolve an expand call. Emitting a stub here would leave the model looking at
    content it cannot pull back — silent loss. Verbatim keeps the one-liner honest:
    identical meaning in context, real (if smaller) savings, nothing unrecoverable. For
    full digest savings, route through the proxy/gateway instead.
    """

    def __init__(self, real_messages: Any) -> None:
        self._real = real_messages

    def create(self, **kwargs: Any) -> Any:
        # Compress messages if present (verbatim: no unrecoverable stubs — see class doc).
        if "messages" in kwargs:
            compressed, _store = compress_messages(kwargs["messages"], verbatim=True)
            kwargs = {**kwargs, "messages": compressed}

        # Apply cache_control only when messages are also present: place_cache_control
        # indexes the messages list, so guard against it being absent (the API requires
        # messages anyway — let it raise the real error rather than a KeyError here).
        if "system" in kwargs and "messages" in kwargs:
            cache_kwargs = place_cache_control(kwargs["system"], kwargs["messages"])
            kwargs = {**kwargs, **cache_kwargs}

        return self._real.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _ClientProxy:
    """Thin proxy for an Anthropic client that injects compression transparently."""

    def __init__(self, real_client: Any) -> None:
        self._real = real_client
        self.messages = _MessagesProxy(real_client.messages)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap(client: Any) -> _ClientProxy:
    """Wrap an Anthropic client so every ``messages.create`` call is compressed.

    The wrapper is a pure structural proxy — it imports nothing from the
    ``anthropic`` SDK and works with any duck-typed object that exposes a
    ``messages.create(**kwargs)`` method.

    Compression is **verbatim** (in-context-lossless Tier-0 only): the model always sees
    semantically identical content and never an unrecoverable digest stub, because the
    in-process wrapper has no server-side ``distil_expand`` loop. Route through the
    proxy/gateway for the higher-savings reversible digest. See :class:`_MessagesProxy`.

    Example
    -------
    ::

        import anthropic
        import distil.adapters.anthropic as distil_anthropic

        client = distil_anthropic.wrap(anthropic.Anthropic())
        # All subsequent calls are transparently compressed:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    """
    return _ClientProxy(client)
