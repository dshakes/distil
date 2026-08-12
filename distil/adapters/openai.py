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
from typing import Any

from ..compress.intent import extract_intent, terms_of
from ..compress.recency import RECENCY_KEEP_TURNS as _RECENCY_KEEP_TURNS
from ..compress.recency import exempt_indices as _exempt_indices
from .anthropic import (
    _census,
    _census_tls,
    RestoreStore,
    _compress_text_content,
    _compress_tool_result_text,
    _intent_tls,
    _keep_tls,
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


def _compress_openai_message(
    msg: dict[str, Any],
    store: RestoreStore,
    verbatim: bool,
    is_recent: bool = False,
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
    _intent_tls.terms = frozenset() if verbatim else extract_intent(messages)
    try:
        store = RestoreStore()
        new_messages: list[dict[str, Any]] = []
        recent = _recent_chat_verbatim_indices(messages, _RECENCY_KEEP_TURNS)
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                new_messages.append(msg)  # malformed entry — pass through untouched
                continue
            msg_verbatim = verbatim or idx in recent
            new_messages.append(
                _compress_openai_message(msg, store, msg_verbatim, is_recent=idx in recent)
            )
        return new_messages, store
    finally:
        _keep_tls.fn = None
        _intent_tls.terms = frozenset()


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


def _compress_response_item(
    item: dict[str, Any],
    store: RestoreStore,
    verbatim: bool,
    is_recent: bool = False,
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
        new_output = _compress_tool_result_text(output, store, verbatim, is_recent)
        if new_output == output:
            return item
        return {**item, "output": new_output}

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
    _intent_tls.terms = frozenset() if verbatim else _extract_responses_intent(items)
    try:
        store = RestoreStore()
        new_items: list[dict[str, Any]] = []
        recent = _recent_response_verbatim_indices(items, _RECENCY_KEEP_TURNS)
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                new_items.append(item)  # malformed entry — pass through
                continue
            item_verbatim = verbatim or idx in recent
            new_items.append(
                _compress_response_item(item, store, item_verbatim, is_recent=idx in recent)
            )
        return new_items, store
    finally:
        _keep_tls.fn = None
        _intent_tls.terms = frozenset()


def count_responses_tokens(items: list[dict[str, Any]]) -> int:
    """Heuristic token count for a Responses API ``input`` array.

    Counts only the *compressible* content — ``function_call_output`` output strings
    and user message ``input_text`` parts.  Passthrough items (``function_call``,
    assistant messages, etc.) are excluded, matching the ``x-distil-compressible-tokens``
    semantics in the proxy's messages path.
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
                    if isinstance(part, dict) and part.get("type") == "input_text":
                        v = part.get("text", "")
                        if isinstance(v, str):
                            total += _tokenizer.count(v)
    return total
