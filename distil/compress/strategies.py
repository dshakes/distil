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


def vision(blocks: list[Block], turn: int) -> list[Block]:
    """Vision duplicate elision, as a certifiable strategy (ADR 0003).

    The first appearance of an image is left alone; a later BYTE-IDENTICAL copy
    has its media dropped and a reference line appended to the block's text. The
    live runner renders media as real image content blocks, so certifying this
    compares the model's next action when it sees N images against when it sees
    the distinct subset — which is the actual claim, rather than a claim about
    text that mentions images.

    Identity is the exact base64 payload. A url source is never treated as a
    duplicate: two occurrences of one URL are not evidence of the same pixels
    (signed URLs, dashboards, cache-busted screenshots), and eliding on that
    basis would assert an identity nobody verified.

    Text is untouched, so this composes with the text strategies rather than
    competing with them: certifying `vision` isolates the image transform.
    """
    seen: set[str] = set()
    out: list[Block] = []
    for b in blocks:
        if not b.media:
            out.append(b)
            continue
        kept: list[dict] = []
        elided = 0
        for item in b.media:
            data = item.get("source", {}).get("data") if isinstance(item, dict) else None
            if not isinstance(data, str) or not data:
                kept.append(item)  # url or unrecognized shape — never elided
                continue
            if data in seen:
                elided += 1
                continue
            seen.add(data)
            kept.append(item)
        if not elided:
            out.append(b)
            continue
        note = (
            f"\n<< distil:image — {elided} image(s) identical to one shown earlier in this "
            "conversation, not repeated here; the originals remain recoverable >>"
        )
        nb = b.copy_with(b.text + note)
        nb.media = kept or None
        out.append(nb)
    return out


REGISTRY: dict[str, Strategy] = {
    "none": none,
    "distil": distil,
    "naive": naive,
    "aggressive": aggressive,
    "vision": vision,
}
