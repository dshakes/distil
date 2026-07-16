"""Per-turn compression strategies, as (blocks, turn_index) -> blocks.

These are what the cache simulator runs. The contrast between `distil` and
`naive` is the whole point of technique #1: both shrink tokens, but `naive`
rewrites the cacheable prefix every turn and so destroys the 10x cache-read
discount, while `distil` keeps the prefix byte-stable and compresses only the
volatile tail.
"""

from __future__ import annotations

from typing import Callable

from ..trajectory import Block, Stability
from .stabilize import stabilize_blocks
from .tier0 import Tier0Lossless
from .tier1 import Tier1Reversible

Strategy = Callable[[list[Block], int], list[Block]]

_T0 = Tier0Lossless()
_T1 = Tier1Reversible()


def none(blocks: list[Block], turn: int) -> list[Block]:
    return blocks


def _no_bigger(originals: list[Block], compressed: list[Block]) -> list[Block]:
    """Reject-if-bigger invariant: never emit a block larger than its original."""
    by_id = {b.id: b for b in originals}
    out: list[Block] = []
    for c in compressed:
        o = by_id.get(c.id)
        out.append(o if o is not None and len(c.text) > len(o.text) else c)
    return out


def distil(blocks: list[Block], turn: int) -> list[Block]:
    """Lossless pipeline: stabilize the cacheable prefix (lift volatile fields so
    it stays byte-identical across turns), then Tier-1/0 the VOLATILE tail only.
    Stable prefix is otherwise untouched; reject-if-bigger guards every block.

    DELIBERATELY HARSHER than serving: this digests every volatile block including
    the turn's freshest tool output, which the live adapter exempts under the
    recency rule (`compress.recency`). Certifying non-inferiority under the harsher
    compression and serving the strictly gentler one is the safe transfer
    direction — the invariant (served digest-set is a SUBSET of the certified
    digest-set) is pinned by tests/test_live_certified_equivalence.py."""
    blocks = stabilize_blocks(blocks)
    stable = [b for b in blocks if b.stability is not Stability.VOLATILE]
    volatile = [b for b in blocks if b.stability is Stability.VOLATILE]
    compressed = _T0.compress(_T1.compress(volatile).blocks).blocks
    return stable + _no_bigger(volatile, compressed)


def naive(blocks: list[Block], turn: int) -> list[Block]:
    """Compress everything, but re-run the compressor over the whole prompt each
    turn so the prefix text changes turn-to-turn (a per-turn re-summarization
    tag). Fewer tokens than baseline, yet every turn is a cache miss."""
    blocks = _T1.compress(blocks).blocks
    blocks = _T0.compress(blocks).blocks
    out: list[Block] = []
    for b in blocks:
        if b.stability is not Stability.VOLATILE:
            out.append(b.copy_with(f"{b.text}\n<<recompressed@t{turn}>>"))
        else:
            out.append(b)
    return out


def aggressive(blocks: list[Block], turn: int) -> list[Block]:
    """Lossy truncation that ignores decision-relevance. Kept only so the
    certification gate has something it MUST reject."""
    return [b.copy_with(b.text[:120]) for b in blocks]


REGISTRY: dict[str, Strategy] = {
    "none": none,
    "distil": distil,
    "naive": naive,
    "aggressive": aggressive,
}
