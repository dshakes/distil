"""codebench's corpus under the client shape that actually bills.

``benchmarks/codebench.py`` builds its sessions with **no** ``cache_control`` marker, and
then prices them with a cache: :func:`benchmarks.codebench._session_dollars` bills the
longest prefix identical to the previous turn at the cache-read rate. Those two
assumptions do not describe any real Anthropic client. Anthropic caches **only what the
client marks**, so an unmarked request has no cached prefix to preserve — and a
transform whose rendering legitimately changes one turn later (every recency-anchored
carve-out distil has) is charged there for busting a cache that was never created.

That artefact is invisible while the digest row is 0.0%, which is what it was before the
re-read delta. It is not invisible after.

So this runner replays the same corpus twice: as codebench ships it, and with the newest
turn pinned — what Claude Code does, and the configuration ADR 0008 calls out as the one
that bills. The second is the number to read for anything recency-anchored; the first is
kept beside it because hiding a row that looks bad is how benchmarks stop being evidence.

Run:  PYTHONPATH=. python benchmarks/codebench_marked.py
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from benchmarks.codebench import (
    _m_distil,
    _m_none,
    _session_dollars,
    _session_tokens,
    make_corpus,
)
from distil.pricing import get as get_pricing


def pin_newest(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the newest turn cacheable — the Claude Code shape, whole history cached."""
    out = [dict(m) for m in msgs]
    last = out[-1]
    content = last.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        head = {**content[0], "cache_control": {"type": "ephemeral"}}
        out[-1] = {**last, "content": [head, *content[1:]]}
    return out


def measure(
    n_sessions: int, mark: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
) -> tuple[float, float]:
    """``(token_saving, cache_aware_dollar_saving)`` for distil's digest path."""
    pricing = get_pricing("claude-opus-4-8")
    base_tok = base_dol = sent_tok = sent_dol = 0.0
    for session in make_corpus(n_sessions):
        marked = [mark(copy.deepcopy(turn)) for turn in session]
        plain = [_m_none(turn) for turn in marked]
        distil = [_m_distil(turn) for turn in marked]
        base_tok += _session_tokens(plain)
        base_dol += _session_dollars(plain, pricing)
        sent_tok += _session_tokens(distil)
        sent_dol += _session_dollars(distil, pricing)
    return 1 - sent_tok / base_tok, 1 - sent_dol / base_dol


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    shapes: list[tuple[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]]] = [
        ("no marker (codebench as shipped)", lambda m: m),
        ("newest turn pinned (Claude Code)", pin_newest),
    ]
    print(f"distil (PAYG digest) on the codebench corpus, by client cache shape ({n} sessions)")
    print()
    print(f"{'client shape':<36}{'tok save':>10}{'$ save (cache)':>17}")
    print("-" * 63)
    for label, mark in shapes:
        tok, dol = measure(n, mark)
        print(f"{label:<36}{tok * 100:>9.1f}%{dol * 100:>16.1f}%")
    print("-" * 63)
    print(
        "An unmarked Anthropic request is not cached at all, so the first row prices a "
        "cache\nthat does not exist for it. The second row is the configuration that bills."
    )
