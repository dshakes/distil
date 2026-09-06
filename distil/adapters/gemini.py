"""Google Gemini ``generateContent`` API runtime adapter — Phase 2 of the roadmap.

Compresses an in-flight Gemini request with no caller code change, mirroring the
Anthropic/OpenAI adapter. Gemini's request shape differs from the Messages API::

    {"contents": [{"role": "user"|"model",
                   "parts": [{"text": ...} | {"functionCall": ...} | {"functionResponse": ...}]}],
     "systemInstruction": {"parts": [{"text": ...}]},
     "tools": [...]}

What we compress (reversibly — the original is kept in the ``RestoreStore`` and is
never sent to the model, so it costs zero tokens):

* ``text`` parts (non-model role) -> Tier-0 lossless (``minify_json`` + ``collapse_runs``).
* ``functionResponse`` parts      -> large string values inside ``response`` are
  digested with the Tier-1 *reversible* digest; the object structure is preserved
  so the request stays valid.

Passed through untouched (the decision-bearing or non-textual parts):
``functionCall``, ``inlineData``, ``fileData``, ``executableCode``, and
**model-authored text** (we never rewrite the model's own words). The
``systemInstruction`` is left byte-exact, matching how the proxy treats the
Anthropic ``system`` field.

Faithful by reuse: the tier logic, the ``RestoreStore``, the recency carve-out
(``RECENCY_KEEP_TURNS``), query-aware intent (``_intent_tls``), and the learned
keep-byte-exact policy all come from the Anthropic adapter — only the *shape*
walking is new here, so Gemini gets the exact same compression guarantees.

Parity with the Anthropic/OpenAI adapters (all now wired):

* **Recency carve-out** — the last ``RECENCY_KEEP_TURNS`` role:"user" turns stay
  verbatim; Gemini puts both human text and ``functionResponse`` (tool results)
  under role:"user", so this preserves the agent's freshest tool outputs
  byte-exact.
* **Query-aware intent** — ``_intent_tls`` is seeded from the latest user text
  parts + every ``functionCall`` name/args; the tier-1 digester uses it to pin
  lines relevant to the agent's query.
* **Output verbosity shaping** — ``distil.output.shape_request`` now accepts
  ``shape="gemini"`` and injects a conciseness directive into ``systemInstruction``
  (PAYG-only, gated identically to the Anthropic/OpenAI paths). Wired in
  ``proxy.py`` under the ``shape_output != "off" and _lossy_ok`` guard.

Expand-tool (closed seam):

* ``distil_expand`` is now wired for Gemini: :func:`distil.expand.inject_expand_tool_gemini`
  injects it as a ``functionDeclarations`` entry (``type:"OBJECT"``/``"STRING"`` —
  Gemini's uppercase enum strings) and :func:`distil.expand.run_expand_loop_gemini`
  intercepts ``candidates[0].content.parts[*].functionCall`` responses, resolves the
  handle, and re-queries with the ``role:"user"`` ``functionResponse`` appended to
  ``contents`` — same round cap, same fail-open, same PAYG/``--expand`` gating as the
  messages path. See ``distil.proxy`` (Gemini branch) for the wiring.

Gemini context caching (``cachedContent``):

* When present, ``cachedContent`` is a server-side resource name; the early turns are
  NOT in ``contents`` — only the new incremental turns are. So neither compression nor
  the expand loop ever touches the cached region (it is simply not there). Appending new
  ``functionResponse`` turns to ``contents`` for the expand loop is always valid
  regardless of ``cachedContent``. No guard needed.

Shadow-mode live decision-equivalence works for Gemini (see
``distil.shadow.decision_signature``).
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..compress.recency import RECENCY_KEEP_TURNS as _RECENCY_KEEP_TURNS
from ..compress.recency import exempt_indices as _exempt_indices
from ..compress import provenance as _provenance
from ..httpguard import strip_query
from ..tokenizer import DEFAULT as _tokenizer
from .anthropic import (
    RestoreStore,
    _hazard_tls,
    _compress_text_content,
    _compress_tool_result_text,
    _intent_tls,
    _keep_tls,
)

# /v1beta/models/{model}:generateContent  — also :streamGenerateContent and the /v1 host.
_GENERATE_RE = re.compile(r"^/v1(?:beta)?/models/[^/:]+:(?:stream)?[Gg]enerateContent$")


def is_gemini_path(path: str) -> bool:
    """True if *path* is a Gemini ``(stream)generateContent`` endpoint."""
    return bool(_GENERATE_RE.match(strip_query(path)))


# ---------------------------------------------------------------------------
# Intent extraction (Gemini shape)
# ---------------------------------------------------------------------------


def _extract_gemini_intent(contents: list[Any]) -> frozenset[str]:
    """Intent terms for a Gemini ``contents`` array.

    Mirrors ``extract_intent`` (``compress.intent``) for the Gemini parts/contents
    shape: the latest role:"user" text parts name what the agent is asking for,
    and every ``functionCall`` name + args name what it looked up. Used to seed
    ``_intent_tls`` so the tier-1 digester can pin query-relevant lines.

    Degrades gracefully (empty frozenset) on any unexpected shape.
    """
    from ..compress.intent import terms_of  # local: avoid load-time cycle

    terms: set[str] = set()
    # Latest user text turn's parts name the needle.
    for content in reversed(contents):
        if not isinstance(content, dict) or content.get("role") != "user":
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        found = False
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                terms |= terms_of(part["text"])
                found = True
        if found:
            break
    # Every functionCall's name + args also name what was looked up.
    for content in contents:
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            fc = part.get("functionCall")
            if isinstance(fc, dict):
                terms |= terms_of(str(fc.get("name", "")))
                args = fc.get("args")
                if args is not None:
                    terms |= terms_of(json.dumps(args, default=str))
    return frozenset(terms)


# ---------------------------------------------------------------------------
# Recency carve-out (Gemini shape)
# ---------------------------------------------------------------------------


def _recent_gemini_verbatim_indices(contents: list[Any], k: int) -> set[int]:
    """Indices of the last *k* role:"user" turns in a Gemini ``contents`` list.

    Gemini places both human text and ``functionResponse`` (tool results) in
    role:"user" turns. Like OpenAI it caches prefixes implicitly, with no client
    marker to anchor to, so a window counted back from the end would digest one
    turn later a turn the provider has already cached — invalidating the whole
    prefix. Empty for that reason: see ``compress.recency.exempt_indices``.
    """
    idxs = [i for i, c in enumerate(contents) if isinstance(c, dict) and c.get("role") == "user"]
    return _exempt_indices(idxs, k, len(contents) - 1)


# ---------------------------------------------------------------------------
# Exact-quote exemption (Gemini shape)
# ---------------------------------------------------------------------------


def _exact_quote_positions(contents: list[Any]) -> set[tuple[int, int]]:
    """``(content index, part index)`` of every ``functionResponse`` to keep byte-exact.

    Gemini gives a ``functionCall`` no id — the response is matched to its call by
    ``name`` alone. So calls and responses are paired positionally: the k-th response
    named *N* answers the k-th call named *N*, which is the order the API guarantees.
    Synthetic ids (``name#k``) then feed the shared classifier, so Gemini gets exactly the
    rule the other two adapters get.

    This provider caches prefixes implicitly and commits everything it is sent, so a read
    is never demoted once sent: ``cached_through`` is the last index.
    """
    calls: list[_provenance.ToolCall] = []
    seen: dict[str, int] = {}
    positions: dict[str, tuple[int, int]] = {}
    answered: dict[str, int] = {}
    for ci, content in enumerate(contents):
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for pi, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            fc = part.get("functionCall")
            if isinstance(fc, dict):
                name = str(fc.get("name", ""))
                k = seen[name] = seen.get(name, 0) + 1
                args = fc.get("args") if isinstance(fc.get("args"), dict) else {}
                calls.append(
                    _provenance.ToolCall(
                        id=f"{name}#{k}",
                        name=name,
                        command=_provenance.command_text(args),
                        pos=ci,
                    )
                )
                continue
            fr = part.get("functionResponse")
            if isinstance(fr, dict):
                name = str(fr.get("name", ""))
                k = answered[name] = answered.get(name, 0) + 1
                positions[f"{name}#{k}"] = (ci, pi)
    keep = _provenance.exact_quote_ids(calls, cached_through=len(contents) - 1)
    return {positions[i] for i in keep if i in positions}


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _compress_json_value(
    val: Any, store: RestoreStore, verbatim: bool, is_recent: bool = False
) -> Any:
    """Recursively compress string values inside a ``functionResponse.response``.

    Strings are digested (large) or Tier-0 transformed (small) via the shared
    tool-result path; structure (dicts/lists) is walked but preserved, so the
    object the Gemini API requires stays intact and the request remains valid.
    Returns the *same object* when nothing changed, so callers can use identity
    to detect a no-op. ``verbatim`` restricts to in-context-lossless Tier-0;
    ``is_recent`` additionally blocks the lossless fold so recent bytes stay exact.
    """
    if isinstance(val, str):
        new = _compress_tool_result_text(val, store, verbatim, is_recent)
        return new if new != val else val
    if isinstance(val, dict):
        out: dict[str, Any] = {}
        changed = False
        for k, v in val.items():
            nv = _compress_json_value(v, store, verbatim, is_recent)
            out[k] = nv
            if nv is not v:
                changed = True
        return out if changed else val
    if isinstance(val, list):
        out_list: list[Any] = []
        changed = False
        for v in val:
            nv = _compress_json_value(v, store, verbatim, is_recent)
            out_list.append(nv)
            if nv is not v:
                changed = True
        return out_list if changed else val
    return val


def _compress_part(
    part: Any,
    store: RestoreStore,
    role: str,
    verbatim: bool,
    is_recent: bool = False,
    exact_quote: bool = False,
) -> Any:
    """Compress a single Gemini ``part``; returns the same object when unchanged."""
    if not isinstance(part, dict):
        return part
    if exact_quote:
        # File content the agent must quote back verbatim to edit it.
        return part

    text = part.get("text")
    if isinstance(text, str):
        if role == "model":
            return part  # never rewrite the model's own words
        new_text = _compress_text_content(text, store, verbatim)
        return part if new_text == text else {**part, "text": new_text}

    fr = part.get("functionResponse")
    if isinstance(fr, dict) and "response" in fr:
        resp = fr.get("response")
        new_resp = _compress_json_value(resp, store, verbatim, is_recent)
        if new_resp is resp:
            return part
        return {**part, "functionResponse": {**fr, "response": new_resp}}

    # functionCall / inlineData / fileData / executableCode / unknown — untouched.
    return part


def compress_generate_request(
    body: dict[str, Any],
    *,
    verbatim: bool = False,
    keep: Any = None,
) -> tuple[dict[str, Any], RestoreStore]:
    """Compress a Gemini ``generateContent`` request body (non-mutating).

    Parameters mirror :func:`distil.adapters.anthropic.compress_messages`.
    Returns ``(new_body, store)``; ``new_body`` is a shallow copy with a
    compressed ``contents`` list, ``store`` maps every digest handle back to the
    original text via ``store.expand(handle)``.

    Recency carve-out: the last ``RECENCY_KEEP_TURNS`` role:"user" turns are kept
    verbatim so the agent always sees its freshest tool outputs byte-exact.

    Query-aware intent: ``_intent_tls`` is seeded once per call from the latest
    user text parts + every ``functionCall`` name/args, then read by the tier-1
    digester to pin lines the agent is actively looking for.
    """
    _keep_tls.fn = keep  # learned keep-byte-exact policy for this call (per-thread)
    # The quote-hazard counter is Messages-path only (that is where Edit/MultiEdit
    # live), so it is CLEARED here rather than left alone: the proxy reads it per
    # request off the same thread, and a stale count from an earlier Anthropic request
    # would be reported against this one.
    _hazard_tls.counts = None
    contents = body.get("contents")
    # Empty by design, not an oversight: this provider caches prefixes
    # implicitly and commits everything it is sent, so every block is cached
    # content by the next request. Intent terms change every turn, so letting
    # them choose which lines survive would rewrite the cached prefix on every
    # question. Same reason there is no recency carve-out here.
    # See compress.recency.exempt_indices.
    _intent_tls.terms = frozenset()
    try:
        store = RestoreStore()
        if not isinstance(contents, list):
            return body, store

        recent = _recent_gemini_verbatim_indices(contents, _RECENCY_KEEP_TURNS)
        # Results the agent must quote back byte-exact to edit — provenance, not position.
        exact = _exact_quote_positions(contents)
        new_contents: list[Any] = []
        changed = False
        for idx, content in enumerate(contents):
            if not isinstance(content, dict):
                new_contents.append(content)
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                new_contents.append(content)
                continue
            role = content.get("role", "")
            is_recent = idx in recent
            content_verbatim = verbatim or is_recent
            new_parts = [
                _compress_part(
                    p, store, role, content_verbatim, is_recent, exact_quote=(idx, pi) in exact
                )
                for pi, p in enumerate(parts)
            ]
            if any(np is not p for np, p in zip(new_parts, parts)):
                new_contents.append({**content, "parts": new_parts})
                changed = True
            else:
                new_contents.append(content)

        if not changed:
            return body, store
        return {**body, "contents": new_contents}, store
    finally:
        _keep_tls.fn = None
        _intent_tls.terms = frozenset()


# ---------------------------------------------------------------------------
# Token accounting (heuristic — same tokeniser as the messages path)
# ---------------------------------------------------------------------------


def _part_tokens(part: Any) -> int:
    if not isinstance(part, dict):
        return 0
    total = 0
    text = part.get("text")
    if isinstance(text, str):
        total += _tokenizer.count(text)
    fr = part.get("functionResponse")
    if isinstance(fr, dict):
        total += _tokenizer.count(json.dumps(fr.get("response"), default=str, sort_keys=True))
    fc = part.get("functionCall")
    if isinstance(fc, dict):
        total += _tokenizer.count(json.dumps(fc.get("args"), default=str, sort_keys=True))
    return total


def count_tokens(body: dict[str, Any]) -> int:
    """Heuristic token count of a Gemini request's ``contents``."""
    total = 0
    contents = body.get("contents")
    if isinstance(contents, list):
        for content in contents:
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if isinstance(parts, list):
                for part in parts:
                    total += _part_tokens(part)
    return total
