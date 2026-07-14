"""Query-aware salience — keep the tool-output lines the agent is actually asking about.

distil is a proxy, so at compress time it holds the agent's intent in the *same request*
as the output being compressed: the ``tool_use`` call that produced a ``tool_result`` (its
name + arguments literally name the needle — a grep pattern, a path, a symbol) and the
latest user turn. No post-hoc compressor (a shell filter, a content router, a vector finder)
has that pairing.

This module turns that intent into a set of salient terms and reports which output lines are
relevant. The keep is strictly **additive** in ``tier1.digest`` — query-relevant lines join
the never-dropped answer tier — so reversibility (the full block is always in RestoreStore)
and the decision-equivalence certificate are only ever *widened*, never weakened. A term that
matches most lines is not a needle, so it is dropped (selectivity guard); that preserves
compression when intent does not actually narrow anything.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# A term matching more than this fraction of a block's lines is not discriminating
# (e.g. the agent grepped a token that appears everywhere) — ignore it.
SELECTIVITY_CAP = 0.4

_TOKEN_RE = re.compile(r"[A-Za-z_][\w./\-]{2,}")

# Terms too common to be a needle. Small on purpose — the selectivity guard is the real
# defense; this just avoids wasting it on obvious noise.
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "was",
        "are",
        "not",
        "you",
        "your",
        "use",
        "using",
        "run",
        "get",
        "set",
        "all",
        "any",
        "out",
        "into",
        "then",
        "when",
        "true",
        "false",
        "null",
        "none",
        "return",
        "const",
        "let",
        "var",
        "def",
        "class",
        "import",
        "export",
        "value",
        "result",
        "name",
        "type",
        "http",
        "https",
        "www",
        "com",
        "org",
    }
)


def terms_of(text: str) -> set[str]:
    """Salient lowercased tokens (identifiers, paths, dotted names) in *text*."""
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 3} - _STOP


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flatten(v) for v in value)
    return "" if value is None else str(value)


def _texts(content: Any):
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                yield b["text"]


def extract_intent(messages: list[dict[str, Any]]) -> frozenset[str]:
    """Intent terms for this request: the latest user turn + every ``tool_use`` call's
    name and input values. These name what the agent is looking for."""
    terms: set[str] = set()
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            for t in _texts(m.get("content")):
                terms |= terms_of(t)
            break
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    terms |= terms_of(str(b.get("name", "")))
                    terms |= terms_of(_flatten(b.get("input")))
    return frozenset(terms)


def relevant_lines(lines: list[str], intent: frozenset[str], *, bridge: bool = False) -> set[int]:
    """Indices of lines relevant to *intent*, after dropping non-selective terms.
    A line is relevant if it shares a salient token with the discriminating intent set.

    With ``bridge=True`` the match is widened by the semantic bridge (`compress.lexicon`):
    morphology, curated technical synonyms, char-trigram fuzz, and distributional vectors —
    so an answer line like ``max_attempts = 5`` is pinned for the query "the retry limit?"
    even with no shared word. The bridge is strictly *additive* (it only ever adds indices),
    so reversibility and the decision-equivalence certificate are untouched — a false bridge
    can only keep an extra line (a little wasted compression), never drop a needed one."""
    if not intent:
        return set()
    line_terms = [terms_of(ln) for ln in lines]
    cap = max(1, int((len(lines) or 1) * SELECTIVITY_CAP))
    freq: Counter[str] = Counter()
    for lt in line_terms:
        freq.update(lt & intent)
    keep_terms = frozenset(t for t in intent if freq[t] <= cap)
    if not keep_terms:
        return set()
    out = {i for i, lt in enumerate(line_terms) if lt & keep_terms}
    if bridge:
        from .lexicon import bridge_line_terms

        out |= {
            i
            for i, lt in enumerate(line_terms)
            if i not in out and bridge_line_terms(lt, keep_terms)
        }
    return out
