"""Expand-aware grading — measure the reversible tier *with* its recovery loop.

distil's premise is digest-behind-handle + recover-on-demand: a tool output is
folded to a marker ``<< +N lines, handle=XXXXXXXX >>`` and the model can pull the
original back with ``distil_expand`` when it needs detail. Grading the digest with
that loop *disabled* (the default offline path) measures a conservative LOWER BOUND
on decision-equivalence — it counts a flip even when the model would simply have
recovered the content. This wrapper closes that gap: it gives the grader the same
recover-on-demand ability, so the reversible tier is measured the way it deploys.

It is **runner-agnostic**: it drives any base runner that exposes ``_raw(system,
user) -> str`` (Anthropic, OpenAI/vLLM, claude-cli, smoke) through a text protocol —
the model either asks to ``{"expand": [...]}`` or commits to ``{"action","target"}``.
Resolved handles are spliced back from a content-addressed restore map and the model
is re-queried, bounded by ``max_iters``. Truncation levels carry no handles, so they
are unaffected (irrecoverable, correctly) — only the reversible digest benefits.

The restore map is derived directly from the ORIGINAL turn blocks: distil's handle is
``sha256(original_block_text)[:8]`` for each digestible (tool-output / retrieved)
block, so ``build_restore`` reproduces handle→original without re-running compression.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

from ..trajectory import Block, Kind
from . import prompts

_DIGESTIBLE = {Kind.TOOL_OUTPUT, Kind.RETRIEVED}
_HANDLE_IN_TEXT = re.compile(r"handle=([0-9a-f]{8})")


def _handle(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def build_restore(original_blocks: list[Block]) -> dict[str, str]:
    """handle → original block text, for every digestible block in the uncompressed
    turn. Mirrors distil.compress.tier1._handle so the markers in compressed blocks
    resolve to their byte-exact originals."""
    return {_handle(b.text): b.text for b in original_blocks if b.kind in _DIGESTIBLE}


def _expand_blocks(blocks: list[Block], handles: list[str], restore: dict[str, str]) -> list[Block]:
    """Replace any block that carries one of ``handles`` with its restored original."""
    out: list[Block] = []
    for b in blocks:
        present = set(_HANDLE_IN_TEXT.findall(b.text))
        hit = next((h for h in handles if h in present), None)
        if hit is not None and hit in restore:
            out.append(b.copy_with(restore[hit]))
        else:
            out.append(b)
    return out


class ExpandAwareRunner:
    """Wrap a base runner (with ``_raw``) to grade with the distil_expand recovery loop."""

    def __init__(self, base, *, samples: int = 1, max_iters: int = 3):
        self.base = base
        self.name = f"{getattr(base, 'name', 'runner')}+expand"
        self.evidential = getattr(base, "evidential", True)
        self.samples = max(1, samples)
        self.max_iters = max_iters

    def decide(self, blocks: list[Block], restore: dict[str, str] | None = None) -> str:
        restore = restore or {}
        if self.samples == 1:
            return self._one(blocks, restore)
        votes = Counter(self._one(blocks, restore) for _ in range(self.samples))
        return votes.most_common(1)[0][0]

    def _one(self, blocks: list[Block], restore: dict[str, str]) -> str:
        cur = blocks
        structured = getattr(self.base, "structured_decision", False)
        for _ in range(self.max_iters):
            has_handle = any(_HANDLE_IN_TEXT.search(b.text) for b in cur)
            if not has_handle:
                break  # nothing left to recover — decide directly
            system, user = prompts.expand_prompt(cur)
            text = self.base._raw(system, user)
            want = prompts.parse_expand(text)
            if want:
                resolvable = [h for h in want if h in restore]
                if resolvable:
                    cur = _expand_blocks(cur, resolvable, restore)
                    continue  # recovered something → look again
            fp = prompts.parse_fingerprint(text)
            if fp != "<no-decision>" and not structured:
                return fp  # model committed to an action in the same step
            break  # unparseable / asked for nothing resolvable → clean decide below
        if structured:
            # Commit through the base runner's constrained decision tool — the same
            # format the uncompressed arm is graded in. A free-text commit differs
            # from it in format alone, and that flipped the action on 21% of turns
            # on claude-opus-4-8 (benchmarks/model_migration_eval.py, format_consist).
            return self.base.decide(cur)
        # always finish with a clean, constrained decision query on the current context
        system, user = prompts.decision_prompt(cur)
        return prompts.parse_fingerprint(self.base._raw(system, user))
