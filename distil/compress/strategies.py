"""Per-turn compression strategies, as (blocks, turn_index) -> blocks.

These are what the cache simulator runs. The contrast between `distil` and
`naive` is the whole point of technique #1: both shrink tokens, but `naive`
rewrites the cacheable prefix every turn and so destroys the 10x cache-read
discount, while `distil` keeps the prefix byte-stable and compresses only the
volatile tail.
"""

from __future__ import annotations

from typing import Callable

from ..trajectory import Block, Kind, Stability
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


def _tool_use(prev: Block | None, declared: list[str]) -> tuple[str, dict]:
    """(name, input) of the tool_use a TOOL_OUTPUT block answers, as far as the
    trajectory says. The adapter's exact-quote rule keys on this — a file view named
    ``open``, a ``bash`` ``cat`` — so a name is only given when it is knowable: the
    fenced command in the agent message right before the output, and only a name the
    turn's TOOLS block declares (or ``bash`` for any other command, when declared).
    Anything else is the neutral ``tool`` with no input: guessing ``read`` would exempt
    what serving would have digested and certify something gentler than is served."""
    from ..replay.realtrace import swe_fenced_command

    if prev is None or prev.kind is not Kind.HISTORY:
        return "tool", {}
    cmd = swe_fenced_command(prev.text)
    verb = cmd.split(" ", 1)[0]
    if verb and verb in declared:
        return verb, {"command": cmd}
    if cmd and "bash" in declared:
        return "bash", {"command": cmd}
    return "tool", {}


def served(blocks: list[Block], turn: int) -> list[Block]:
    """What the SERVING adapter sends for this turn, as a certifiable strategy.

    `distil` above digests only VOLATILE blocks, so it certifies a tiny slice of what
    ships: on 120 real SWE-agent trajectories it saves 2.6% of tokens, while
    ``adapters.anthropic.compress_messages`` — what a caching client is actually
    served — digests every earlier tool_result and saves 52.6%. This runs that real
    adapter on the turn, so the served path is what gets certified:

    * the turn becomes the Messages request a caching tool-use client would send —
      stable SYSTEM/TOOLS stay out of it (the system prompt, never rewritten), each
      TOOL_OUTPUT becomes a tool_result answering a paired assistant tool_use (named
      per :func:`_tool_use`), HISTORY becomes assistant text, everything else user text;
    * the client's cache breakpoint sits on the newest message, so everything is
      committed prefix: no recency carve-out, the freshest output digests too, and
      every block's served bytes depend only on its own text — the same block is
      byte-identical on every turn that carries it (the cache contract);
    * ``compress_messages`` runs, and its texts are mapped back onto the blocks, with
      reject-if-bigger on top.

    Handles are the adapter's ``sha256(original_tool_result)[:8]``, and the
    tool_result text IS the block text, so ``expand_runner.build_restore`` of the
    original turn resolves them. A ``<distil:keep>`` span or a re-read delta carries a
    handle for a SUB-span that build_restore does not map; the expand arm then fails to
    recover it, which errs harsher, the safe direction.

    What it does NOT certify: a client with no or an earlier cache breakpoint (its
    newest turns stay verbatim — gentler — but its uncached blocks get query-aware
    salience and superseded shell reads may digest, neither exercised here); the
    exact-quote exemption beyond the tool names the trajectory reveals; images (media
    passes through untouched — `vision` certifies that); verbatim/subscription mode;
    cold-point eviction. Unlike serving it never touches the on-disk restore store
    (``persist=False``): a certify run must not evict a live session's blobs, nor let
    what is already on disk change its output.
    """
    from ..adapters.anthropic import compress_messages
    from ..replay.prompts import available_actions

    declared = available_actions(blocks)
    msgs: list[dict] = []
    where: dict[int, tuple[int, int]] = {}

    def put(role: str, item: dict) -> tuple[int, int]:
        if not msgs or msgs[-1]["role"] != role:
            msgs.append({"role": role, "content": []})
        msgs[-1]["content"].append(item)
        return len(msgs) - 1, len(msgs[-1]["content"]) - 1

    prev: Block | None = None
    for i, b in enumerate(blocks):
        if b.stability is Stability.STABLE and b.kind in (Kind.SYSTEM, Kind.TOOLS):
            continue
        if b.kind is Kind.TOOL_OUTPUT:
            name, inp = _tool_use(prev, declared)
            tid = f"toolu_{i:04d}"
            put("assistant", {"type": "tool_use", "id": tid, "name": name, "input": inp})
            where[i] = put("user", {"type": "tool_result", "tool_use_id": tid, "content": b.text})
        elif b.kind is Kind.HISTORY:
            where[i] = put("assistant", {"type": "text", "text": b.text})
        else:
            where[i] = put("user", {"type": "text", "text": b.text})
        prev = b
    if not msgs:
        return blocks
    msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    out, _ = compress_messages(msgs, persist=False)

    new: list[Block] = []
    for i, b in enumerate(blocks):
        if i not in where:
            new.append(b)
            continue
        mi, ci = where[i]
        item = out[mi]["content"][ci]
        text = item["text"] if item.get("type") == "text" else item["content"]
        if not isinstance(text, str):  # the adapter keeps a string content a string
            text = "".join(x.get("text", "") for x in text if isinstance(x, dict))
        new.append(b.copy_with(text))
    return _no_bigger(blocks, new)


REGISTRY: dict[str, Strategy] = {
    "none": none,
    "distil": distil,
    "naive": naive,
    "aggressive": aggressive,
    "vision": vision,
    "served": served,
}
