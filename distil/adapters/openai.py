"""OpenAI Chat Completions and Responses API runtime adapter.

Compresses in-flight OpenAI requests using the same Tier-0/Tier-1 machinery as
the Anthropic adapter (``distil.adapters.anthropic``). Only the *shape walking*
differs here — the compression guarantees, RestoreStore, recency carve-out, and
learned keep-byte-exact policy are identical.

Chat Completions (``/v1/chat/completions``)
-------------------------------------------
The Chat Completions schema maps closely to Anthropic's Messages shape, but with
two OpenAI-specific twists worth calling out:

* ``role:"tool"`` messages carry the tool output as a **bare string** *or* as a
  **list of** ``{"type": "text", "text": "..."}`` **parts**.  Both are treated as
  tool results and digested with the Tier-1 reversible digest — the Anthropic
  adapter already handles string-content tool messages (it was written to be
  shape-agnostic), but the list-content variant needed a dedicated code path here.

* ``role:"assistant"`` messages may carry a ``tool_calls`` list (model output).
  These pass through unchanged — we never rewrite the model's own words.

Responses API (``/v1/responses``)
----------------------------------
The body's ``input`` field is an array of heterogeneous typed items::

    {"type": "message",              "role": "user",      "content": [{"type": "input_text",  "text": "..."}]}
    {"type": "message",              "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}
    {"type": "function_call",        "id": "...", ...}      # model output — passthrough
    {"type": "function_call_output", "call_id": "...", "output": "..."}  # tool result — Tier-1
    {"type": "reasoning",  "id": "...", "encrypted_content": "..."}  # opaque — passthrough, censused
    {"type": "compaction", "id": "...", "encrypted_content": "..."}  # opaque — passthrough, censused

A ``reasoning`` item's ``encrypted_content`` (stateless mode / Zero Data Retention) and
a ``compaction`` item (what ``POST /v1/responses/compact`` returns, and what
``context_management: [{"type": "compaction", ...}]`` appends inline) are opaque,
provider-owned bytes the client must return unaltered on the next turn — see
``tests/test_openai_opaque_passthrough.py`` for the contract this pins. They pass
through byte-identical (same object) and are censused under ``reasoning_billed`` /
``compaction_billed`` so a cost distil cannot reduce is not also one it hides from
the eligibility census, mirroring the Anthropic adapter's ``thinking``/``compaction``
content-block handling.

The top-level ``instructions`` field (the system prompt) passes through unchanged.

All three seams are now wired end-to-end in the proxy:

* **Expand-tool injection**: ``inject_expand_tool_responses`` injects the
  ``distil_expand`` tool using the flat Responses function-tool schema
  (``{"type":"function","name":...,"parameters":...}``), which differs from the
  Chat Completions nested ``{"type":"function","function":{...}}`` form.  The
  proxy's ``_expand_should_intercept`` guard and ``run_expand_loop_responses``
  intercept loop mirror the messages-path semantics exactly — same caps, same
  fail-open behaviour, same PAYG/--expand gating.

* **Output shaping**: ``shape_request(body, shape="responses")`` appends the
  verbosity-control directive to the top-level ``instructions`` field (append,
  never replace), wired under the same ``shape_output != "off" and _lossy_ok``
  guard as the other paths.

* **Intent extraction**: ``_extract_responses_intent`` pulls salient terms from
  the latest user ``input_text`` parts and any ``function_call`` argument values,
  feeding the same ``_intent_tls`` mechanism as the messages path so query-aware
  salience works for Responses requests.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Mapping

from ..compress.intent import terms_of
from ..compress.recency import RECENCY_KEEP_TURNS as _RECENCY_KEEP_TURNS
from ..compress.recency import exempt_indices as _exempt_indices
from ..compress import provenance as _provenance
from ..compress import vision as _vision
from .anthropic import (
    _active_vision,
    _census,
    _census_tls,
    _census_tokens,
    _census_tool_result,
    _hazard_tls,
    _widen_rescued,
    RestoreStore,
    _compress_text_content,
    _compress_tool_result_text,
    _intent_tls,
    _keep_tls,
    _vision_tls,
)


def _extract_responses_intent(items: list[dict[str, Any]]) -> frozenset[str]:
    """Intent terms for a Responses API ``input`` array.

    Mirrors ``extract_intent`` for the messages path: the latest user message's
    ``input_text`` parts provide the query; ``function_call`` argument values name
    the needle. Content-free — only token names are extracted, never content itself.
    """
    intent: set[str] = set()
    # Latest user message — stop at the first found when working backwards.
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message" and item.get("role") == "user":
            content = item.get("content") or []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_text":
                        intent |= terms_of(str(part.get("text") or ""))
            break
    # function_call items — model's prior calls name the needle.
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            intent |= terms_of(str(item.get("name") or ""))
            try:
                args = json.loads(item.get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            intent |= terms_of(json.dumps(args))
    return frozenset(intent)


# ---------------------------------------------------------------------------
# Chat Completions adapter
# ---------------------------------------------------------------------------


def _chat_tool_calls(messages: list[dict[str, Any]]) -> list[_provenance.ToolCall]:
    """Normalised tool calls from an assistant message's ``tool_calls`` list."""
    calls: list[_provenance.ToolCall] = []
    for idx, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or ():
            if not isinstance(tc, dict):
                continue
            cid = tc.get("id")
            fn = tc.get("function")
            if not isinstance(cid, str) or not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append(
                _provenance.ToolCall(
                    id=cid,
                    name=str(fn.get("name", "")),
                    command=_provenance.command_text(args),
                    pos=idx,
                )
            )
    return calls


def exact_quote_tool_call_ids(messages: list[dict[str, Any]]) -> dict[str, str]:
    """``tool_call_id``s whose ``role:"tool"`` result must stay byte-exact.

    Same guarantee and same classifier as the Messages path — a file the agent read,
    however it read it, has to survive verbatim or its next literal-match edit cannot
    apply. This provider caches prefixes implicitly and commits everything it is sent,
    so nothing may be demoted once sent: ``cached_through`` is the last index.
    """
    return _provenance.exact_quote_ids(_chat_tool_calls(messages), cached_through=len(messages) - 1)


def _compress_openai_message(
    msg: dict[str, Any],
    store: RestoreStore,
    verbatim: bool,
    is_recent: bool = False,
    exact_ids: Mapping[str, str] = MappingProxyType({}),
) -> dict[str, Any]:
    """Return a (possibly new) Chat Completions message dict after compressing.

    The logic mirrors ``_compress_message`` in the Anthropic adapter for all
    roles except one: when ``role:"tool"`` carries a *list* of content parts
    (each ``{"type": "text", "text": "..."}``), every text part is treated as
    tool output and digested with Tier-1 — not Tier-0 as plain user text would
    be. The Anthropic adapter handles the parallel ``type:"tool_result"`` block
    via its own ``_compress_content_item``; this handles the OpenAI equivalent.
    """
    role = msg.get("role", "")
    content = msg.get("content")

    bucket = exact_ids.get(str(msg.get("tool_call_id") or "")) if role == "tool" else None
    if bucket:
        # File content the agent must quote back verbatim to edit it. Censused through the
        # shared helper because a tool message's content is a string OR a list of parts,
        # and counting only the string form leaves list-shaped reads out of a census whose
        # whole value is that it accounts for the entire payload.
        _census_tool_result(bucket, content)
        return msg

    # --- bare string content ---
    if isinstance(content, str):
        if role == "assistant":
            # Never rewrite the model's own words.
            _census("assistant_text", content)
            return msg
        if role == "tool":
            # Tool outputs: Tier-1 reversible digest.
            new_text = _compress_tool_result_text(content, store, verbatim, is_recent)
        else:
            # user / system — Tier-0 lossless transforms only.
            _census("user_text", content)
            new_text = _compress_text_content(content, store, verbatim)
        if new_text == content:
            return msg
        return {**msg, "content": new_text}

    # --- list of content parts ---
    if isinstance(content, list):
        if role == "assistant":
            # Never touch assistant output (includes tool_calls).
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    _census("assistant_text", part["text"])
            return msg
        new_parts: list[Any] = []
        changed = False
        for part in content:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                # Repeated-image elision (ADR 0003) — same certificate gate, same
                # census reasons, same RestoreStore handle contract as the
                # Anthropic path. See compress.vision.elide_or_keep.
                replacement = _vision.elide_or_keep(
                    _active_vision(), part["image_url"], store, _census_tokens, verbatim, is_recent
                )
                if replacement is not None:
                    new_parts.append({"type": "text", "text": replacement})
                    changed = True
                    continue
                new_parts.append(part)
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                if role == "tool":
                    # OpenAI tool messages with list content: every text part is a
                    # tool result fragment → Tier-1, same as the string-content path.
                    new_text = _compress_tool_result_text(part["text"], store, verbatim, is_recent)
                else:
                    _census("user_text", part["text"])
                    new_text = _compress_text_content(part["text"], store, verbatim)
                if new_text != part["text"]:
                    new_parts.append({**part, "text": new_text})
                    changed = True
                    continue
            new_parts.append(part)
        if not changed:
            return msg
        return {**msg, "content": new_parts}

    return msg


def _recent_chat_verbatim_indices(messages: list[dict[str, Any]], k: int) -> set[int]:
    """Indices of the last *k* tool-output-bearing turns in a Chat Completions list.

    OpenAI caches prefixes automatically, with no client marker to anchor to, so
    everything sent is committed the moment it is sent. A window counted back
    from the end would therefore digest, one turn later, a message the provider
    has already cached — invalidating the entry for the whole prefix. So this
    returns the empty set: see ``compress.recency.exempt_indices``.
    """
    idxs = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") in ("user", "tool")
    ]
    return _exempt_indices(idxs, k, len(messages) - 1)


def compress_chat_completions(
    messages: list[dict[str, Any]],
    *,
    verbatim: bool = False,
    keep: Any = None,
) -> tuple[list[dict[str, Any]], RestoreStore]:
    """Compress an OpenAI Chat Completions messages list.

    API-compatible with ``distil.adapters.anthropic.compress_messages``: same
    signature, same return shape.  The compression machinery (tier1, RestoreStore,
    recency carve-out, learned keep policy) is identical; only the schema walking
    differs (``role:"tool"`` list content, content-part type names).

    Parameters
    ----------
    messages:
        The ``messages`` list from a Chat Completions request body.
    verbatim:
        When *True*, apply only in-context-lossless Tier-0 transforms — never a
        Tier-1 digest stub.  Use for interactive sessions or lossless-only policy.
    keep:
        Optional learned keep-byte-exact predicate (same role as in the Anthropic
        adapter — blocks whose content the agent historically expands are never
        digested).

    Returns
    -------
    (new_messages, store)
        ``new_messages`` is a new list (input is not mutated).
        ``store`` maps every 8-hex handle embedded in digest markers back to the
        original text; call ``store.expand(handle)`` to recover it.
    """
    _keep_tls.fn = keep
    # Same contract as the Anthropic entry point: open a fresh census so a thread that
    # previously served Anthropic traffic cannot leak its counts into this request.
    _census_tls.counts = {}
    # The quote-hazard counter is Messages-path only (that is where Edit/MultiEdit
    # live), so it is CLEARED here rather than left alone: the proxy reads it per
    # request off the same thread, and a stale count from an earlier Anthropic request
    # would be reported against this one.
    _hazard_tls.counts = None
    # Empty by design, not an oversight: this provider caches prefixes
    # implicitly and commits everything it is sent, so every block is cached
    # content by the next request. Intent terms change every turn, so letting
    # them choose which lines survive would rewrite the cached prefix on every
    # question. Same reason there is no recency carve-out here.
    # See compress.recency.exempt_indices.
    _intent_tls.terms = frozenset()
    try:
        # ADR 0003 — None unless the content type has been certified, so the
        # default path is byte-for-byte what it was before. Reset every call so a
        # thread that previously served a different request cannot leak "already
        # seen" state into this one.
        _vision_tls.dedup = _vision.ImageDedup() if (not verbatim and _vision.enabled()) else None
        store = RestoreStore()
        new_messages: list[dict[str, Any]] = []
        recent = _recent_chat_verbatim_indices(messages, _RECENCY_KEEP_TURNS)
        # Results the agent must quote back byte-exact to edit — provenance, not position.
        exact_ids = exact_quote_tool_call_ids(messages)
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                new_messages.append(msg)  # malformed entry — pass through untouched
                continue
            msg_verbatim = verbatim or idx in recent
            new_messages.append(
                _compress_openai_message(
                    msg, store, msg_verbatim, is_recent=idx in recent, exact_ids=exact_ids
                )
            )
        return new_messages, store
    finally:
        _keep_tls.fn = None
        _intent_tls.terms = frozenset()
        _vision_tls.dedup = None


# ---------------------------------------------------------------------------
# Responses API adapter
# ---------------------------------------------------------------------------


def _recent_response_verbatim_indices(items: list[dict[str, Any]], k: int) -> set[int]:
    """Indices of the last *k* ``function_call_output`` items in a Responses input list.

    Same automatic-prefix-caching reasoning as ``_recent_chat_verbatim_indices``
    above: nothing can be exempt without invalidating a cached prefix one turn
    later, so this is empty.
    """
    idxs = [
        i
        for i, item in enumerate(items)
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    return _exempt_indices(idxs, k, len(items) - 1)


def _response_tool_calls(items: list[dict[str, Any]]) -> list[_provenance.ToolCall]:
    """Normalised tool calls from ``function_call`` items in a Responses input array."""
    calls: list[_provenance.ToolCall] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        cid = item.get("call_id") or item.get("id")
        if not isinstance(cid, str):
            continue
        args = item.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        calls.append(
            _provenance.ToolCall(
                id=cid,
                name=str(item.get("name", "")),
                command=_provenance.command_text(args),
                pos=idx,
            )
        )
    return calls


def exact_quote_call_ids(items: list[dict[str, Any]], *, widen: bool = False) -> dict[str, str]:
    """``call_id``s whose ``function_call_output`` must stay byte-exact. See
    :func:`exact_quote_tool_call_ids` — same rule, Responses shape.

    ``widen`` drops supersession, the reaction to an observed quote miss (see
    :func:`_guard_response_quotes`)."""
    return _provenance.exact_quote_ids(
        _response_tool_calls(items), cached_through=len(items) - 1, widen=widen
    )


def _census_opaque_response_item(bucket: str, item: dict[str, Any]) -> None:
    """Attribute a Responses API opaque item's billed text to *bucket*.

    Mirrors ``anthropic._census_opaque_block`` for this provider's item shape.
    ``encrypted_content`` is the field both ``reasoning`` and ``compaction`` items
    carry; ``summary`` is a reasoning item's plaintext parts (``[{"type":
    "summary_text", "text": ...}]``) when ``include`` asks for one instead of/
    alongside the encrypted form. Not guessing at further field names beyond
    that: an opaque item's cost is billed whether or not distil can name every
    field it might carry, so under-counting here only hides the very thing this
    census exists to show.

    Approximate by construction: ``encrypted_content`` is base64 ciphertext, and
    this counts it through the same tokenizer heuristic used on plaintext — a
    proxy for the provider's real billed reasoning-token count, not that count
    itself. Reported to the user as approximate (see ``dissect._ELIGIBILITY_LABEL``)
    for the same reason.
    """
    for key in ("encrypted_content", "text", "content"):
        val = item.get(key)
        if isinstance(val, str):
            _census(bucket, val)
    summary = item.get("summary")
    if isinstance(summary, list):
        for part in summary:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                _census(bucket, part["text"])


def _compress_response_item(
    item: dict[str, Any],
    store: RestoreStore,
    verbatim: bool,
    is_recent: bool = False,
    exact_ids: Mapping[str, str] = MappingProxyType({}),
) -> dict[str, Any]:
    """Return a (possibly new) Responses API input item after compression.

    Handling by ``type``:

    * ``function_call_output`` — the ``output`` string is the tool result; compress
      with the Tier-1 reversible digest (Tier-0 in verbatim / recency-exempt mode).
    * ``message``              — walk ``content`` parts by the message ``role``:
      user ``input_text`` parts get Tier-0; assistant ``output_text`` passes through.
    * ``function_call``        — model output; passthrough unchanged.
    * anything else            — passthrough (computer_call, file_search_call, …).
    """
    itype = item.get("type", "")

    if itype == "function_call_output":
        output = item.get("output")
        if not isinstance(output, str):
            return item  # non-string output field — pass through
        bucket = exact_ids.get(str(item.get("call_id") or ""))
        if bucket:
            # File content the agent must quote back verbatim to edit it.
            _census(bucket, output)
            return item
        new_output = _compress_tool_result_text(output, store, verbatim, is_recent)
        if new_output == output:
            return item
        return {**item, "output": new_output}

    if itype in ("reasoning", "compaction") or (
        "encrypted_content" in item
        and itype not in ("message", "function_call_output", "function_call")
    ):
        # Opaque, provider-signed items: a `reasoning` item's `encrypted_content`
        # (stateless mode / ZDR) lets the provider re-derive the model's reasoning on
        # the next turn, and a `compaction` item is exactly what `POST
        # /v1/responses/compact` returns — OpenAI's own docs say "do not prune
        # /responses/compact output... pass it into your next /responses call as-is".
        # Editing either (even a lossless re-encode) risks the provider rejecting the
        # next request or silently losing state it cannot recover. `item` is returned
        # unchanged (same object), only censused — same contract as the Anthropic
        # adapter's `thinking`/`compaction` blocks, generalised on the presence of
        # `encrypted_content` rather than an allowlist of type strings, so a future
        # opaque item type is safe by construction. The three known compressible
        # types are excluded from that generalisation so a stray top-level
        # `encrypted_content` key on one of them (e.g. a malformed/future `message`)
        # can never shadow its own handling below.
        _census_opaque_response_item(
            "reasoning_billed"
            if itype == "reasoning"
            else "compaction_billed"
            if itype == "compaction"
            else "signed_item_billed",
            item,
        )
        return item

    if itype == "message":
        role = item.get("role", "")
        content = item.get("content")
        if not isinstance(content, list):
            return item
        if role == "assistant":
            # Never rewrite the model's own words.
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    _census("assistant_text", part["text"])
            return item
        # User (and any other non-assistant) message: compress text content parts.
        new_parts: list[Any] = []
        changed = False
        for part in content:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            if part.get("type") == "input_image" and isinstance(part.get("image_url"), str):
                # Repeated-image elision (ADR 0003). Responses carries the URL/data-URI
                # as a bare string (unlike Chat Completions' nested image_url object),
                # so it is wrapped in a source dict for the shared helper.
                source = {"url": part["image_url"]}
                replacement = _vision.elide_or_keep(
                    _active_vision(), source, store, _census_tokens, verbatim, is_recent
                )
                if replacement is not None:
                    new_parts.append({"type": "input_text", "text": replacement})
                    changed = True
                    continue
                new_parts.append(part)
                continue
            # input_text → user content (Tier-0 lossless); output_text → model content (skip)
            if part.get("type") == "input_text" and isinstance(part.get("text"), str):
                _census("user_text", part["text"])
                new_text = _compress_text_content(part["text"], store, verbatim)
                if new_text != part["text"]:
                    new_parts.append({**part, "text": new_text})
                    changed = True
                    continue
            new_parts.append(part)
        if not changed:
            return item
        return {**item, "content": new_parts}

    # function_call, computer_call, file_search_call, etc. — all passthrough.
    return item


def _guard_response_quotes(
    items: list[dict[str, Any]],
    compressed: list[dict[str, Any]],
    store: RestoreStore,
    walk: Any,
    verbatim: bool,
) -> tuple[list[dict[str, Any]], RestoreStore]:
    """The Messages path's quote guard, for the shape Codex actually speaks.

    1.51 shipped the exact-quote exemption on all three adapters but the *measurement* on
    the Messages path only, on the reasoning that ``Edit``/``MultiEdit`` live there. Codex
    does the same edits through different names: ``apply_patch``, whose argument on GPT-5
    models is a freeform patch body rather than JSON, and ``str_replace_editor``. Its
    pre-image — the context and removed lines of each hunk — has to be found byte-exact in
    the file just as an ``old_string`` does, so the same guarantee applies and, until now,
    the same guarantee went unmeasured. ``distil dissect`` reads one counter for every
    provider; wiring this one means Codex traffic stops being reported as "no edit here".

    Same reaction as the Messages path: on a miss that widening repairs, supersession is
    dropped for the rest of the session (the history only grows, so the miss is re-detected
    every turn). A miss widening cannot repair keeps the narrow pass — see ``_widen_rescued``.
    """
    _hazard_tls.counts = None
    quotes = _provenance.response_edit_quotes(items)
    if not quotes or verbatim:
        return compressed, store
    missing = _provenance.missing_quotes(quotes, _provenance.observed_view(compressed))
    if missing:
        wide, wide_store = walk(exact_quote_call_ids(items, widen=True))
        w_missing = _provenance.missing_quotes(quotes, _provenance.observed_view(wide))
        if _widen_rescued(missing, w_missing):
            compressed, store, missing = wide, wide_store, w_missing
    _hazard_tls.counts = {"survived": len(quotes) - len(missing), "lost": len(missing)}
    return compressed, store


def compress_responses_input(
    items: list[dict[str, Any]],
    *,
    verbatim: bool = False,
    keep: Any = None,
) -> tuple[list[dict[str, Any]], RestoreStore]:
    """Compress an OpenAI Responses API ``input`` array.

    Handles ``function_call_output`` items (Tier-1 reversible digest on their
    ``output`` field) and ``message`` items (Tier-0 on user ``input_text`` parts).
    All other item types (``function_call``, ``computer_call``, …) pass through
    unchanged — they are model outputs or structured metadata.

    Recency carve-out: the last ``RECENCY_KEEP_TURNS`` ``function_call_output`` items
    stay byte-exact, same guarantee as the Chat Completions and Anthropic adapters.

    Query-aware salience: ``_extract_responses_intent`` extracts terms from the latest
    user ``input_text`` parts and any ``function_call`` argument values, feeding the
    same ``_intent_tls`` mechanism used by the messages path.

    Parameters
    ----------
    items:
        The ``input`` array from a Responses API request body.
    verbatim:
        When *True*, apply only in-context-lossless Tier-0 transforms.
    keep:
        Optional learned keep-byte-exact predicate.

    Returns
    -------
    (new_items, store)
        ``new_items`` is a new list (input is not mutated).
        ``store`` maps handles back to originals for RestoreStore round-trips.
    """
    _keep_tls.fn = keep
    _census_tls.counts = {}
    # Cleared, then recomputed below: the proxy reads this counter per request off the
    # same thread, so a stale count from an earlier Anthropic request must never be
    # reported against this one.
    _hazard_tls.counts = None
    # Empty by design, not an oversight: this provider caches prefixes
    # implicitly and commits everything it is sent, so every block is cached
    # content by the next request. Intent terms change every turn, so letting
    # them choose which lines survive would rewrite the cached prefix on every
    # question. Same reason there is no recency carve-out here.
    # See compress.recency.exempt_indices.
    _intent_tls.terms = frozenset()
    try:
        recent = _recent_response_verbatim_indices(items, _RECENCY_KEEP_TURNS)

        def _walk(exact_ids: Mapping[str, str]) -> tuple[list[dict[str, Any]], RestoreStore]:
            _census_tls.counts = {}
            # Reset per attempt, same reason the Anthropic path resets inside its own
            # _walk: a quote-hazard retry must not treat images the FIRST pass already
            # elided as "already seen" — that would elide every image on the retry and
            # the model would receive none. ADR 0003 — None unless certified.
            _vision_tls.dedup = (
                _vision.ImageDedup() if (not verbatim and _vision.enabled()) else None
            )
            store = RestoreStore()
            new_items: list[dict[str, Any]] = []
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    new_items.append(item)  # malformed entry — pass through
                    continue
                item_verbatim = verbatim or idx in recent
                new_items.append(
                    _compress_response_item(
                        item, store, item_verbatim, is_recent=idx in recent, exact_ids=exact_ids
                    )
                )
            return new_items, store

        # Results the agent must quote back byte-exact to edit — provenance, not position.
        new_items, store = _walk(exact_quote_call_ids(items))
        return _guard_response_quotes(items, new_items, store, _walk, verbatim)
    finally:
        _keep_tls.fn = None
        _intent_tls.terms = frozenset()
        _vision_tls.dedup = None


def count_responses_tokens(items: list[dict[str, Any]]) -> int:
    """Heuristic token count for a Responses API ``input`` array.

    Counts the *compressible* content — ``function_call_output`` output strings,
    user message ``input_text`` parts, and ``input_image`` parts (billed by pixel
    area via ``vision.source_tokens``, same as the eligibility census, so an
    elided repeat's before/after diff lands on the same scale it was censused
    on). Passthrough items (``function_call``, assistant messages, etc.) are
    excluded, matching the ``x-distil-compressible-tokens`` semantics in the
    proxy's messages path.

    Deliberately asymmetric with the Anthropic adapter's baseline
    (``proxy._count_messages``), which *does* count ``thinking``/``redacted_thinking``
    tokens even though it never rewrites them, specifically so a signature-pinned
    block is still visible in the before/after diff. This baseline does not extend
    that same inclusion to ``reasoning``/``compaction`` items: doing so would put a
    provider-signed OpenAI item on the same "content distil is allowed to touch"
    baseline as the compressible content this function measures, which the census
    (``reasoning_billed``/``compaction_billed``/``signed_item_billed``, see
    ``_census_opaque_response_item``) already exists to keep separate and visible on
    its own axis. The practical consequence: on a reasoning/compaction-heavy
    Responses session the eligibility census total can legitimately exceed
    ``x-distil-compressible-tokens`` — that is the signal, not a bug, that a real
    cost was billed on content this baseline was never claiming to cover. See
    ``tests/test_openai_opaque_passthrough.py::test_count_responses_tokens_excludes_opaque_items_by_design``
    and cache-contract.html clause (g).
    """
    from ..tokenizer import DEFAULT as _tokenizer

    total = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "")
        if itype == "function_call_output":
            v = item.get("output", "")
            if isinstance(v, str):
                total += _tokenizer.count(v)
        elif itype == "message" and item.get("role") != "assistant":
            content = item.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "input_text":
                        v = part.get("text", "")
                        if isinstance(v, str):
                            total += _tokenizer.count(v)
                    elif part.get("type") == "input_image" and isinstance(
                        part.get("image_url"), str
                    ):
                        total += _vision.source_tokens({"url": part["image_url"]})
    return total
