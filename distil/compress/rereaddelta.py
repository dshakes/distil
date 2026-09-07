"""Re-read delta — the second read of a file need not re-send the lines the first one did.

The coding hot path is read → edit → **re-read**, and the re-read is almost never a
byte-identical resend. Measured over 2,489 local Claude Code sessions: 51.4% of
``Read``/``cat`` results are a re-read of a path already read in the same session, but
only 12.8% are byte-identical — the rest are a *different line range* of the same file, or
the same file with one hunk changed. That shape defeats both mechanisms distil already
had. Exact-duplicate dedup needs identical bytes. ``cachedelta``'s near-duplicate gate
compares whole blocks with a 0.5 ``difflib`` ratio, and 608 of 726 changed re-reads fail
it, because two disjoint windows onto one file are not 50% similar as blocks even when
every line they share is identical. It fires on 5.2% of re-reads and removes 0.23% of
tool-result mass.

The line is the right unit. **50.6% of re-read tokens are lines that were already
delivered verbatim earlier in the same conversation.** So this module matches on line
content, not on block similarity: the longest run of lines in the new read that appears
contiguously in an earlier read of the same path is replaced by a reference stub, and
every line outside that run is kept verbatim.

Why this does not weaken the exact-quote guarantee
--------------------------------------------------
:mod:`distil.compress.provenance` promises that file content the agent must reproduce
character-for-character in an ``Edit(old_string=...)`` still occurs byte-exact somewhere
in the payload distil forwards. The promise is about the *conversation*, not about any
one block — an agent quoting from what it saw can quote from either copy.

Three rules keep it true:

* **The base must be a read that is exempt unconditionally**, i.e. one matched by the
  tool-NAME table (``Read``, ``view``, ``read_file``, …). Those are never superseded and
  never digested at any age, so the referenced lines cannot disappear from the
  conversation later. Shell reads (``cat``, ``sed -n``) are deliberately *not* eligible as
  bases: their exemption is conditional on not being superseded, and a whole-file shell
  re-read supersedes exactly the block it would want to reference. Pinning the base to
  stop that would cost the base's whole digest to save the same bytes on the copy — a
  wash at best. They remain eligible as *targets*.
* **A block that has been elided is never itself a base.** References never chain, so
  every stub points at literal bytes.
* **Runs are trimmed by :data:`EDGE_MARGIN` lines at any cut that is internal to the
  block.** A quote lying entirely inside the elided run survives in the base; a quote
  lying entirely outside survives here. Only a quote *straddling* a cut would be in
  neither copy contiguously, so the cut is pulled back far enough that a straddling quote
  has to be longer than the margin on one side to break — and an ``old_string`` that long
  is rarer than the saving is worth. A run that reaches the block's own first or last line
  needs no margin there: the agent never saw anything beyond it in this block.

Reversibility is the same mechanism every other distil stub uses: the elided lines are
recorded in the ``RestoreStore`` under a content-addressed handle and ``distil_expand``
returns them byte-exact.

Cache safety (ADR 0008, ADR 0010)
---------------------------------
:func:`plan` is a pure function of the message *prefix* — block *i*'s elision depends only
on reads before it — so a given block encodes to the same bytes on every turn of a growing
conversation. That is the same construction ``cachedelta`` relies on, and it is what the
contract actually requires: no message the client re-sends byte-identical is ever
forwarded differently. There is deliberately **no** volatile-suffix gate. Under the client
shape that bills (Claude Code pins its newest turn, so the whole history is cached) such a
gate would leave nothing to compress, and it would introduce the one thing the contract
forbids — a rendering that changes as the boundary moves past a block.
"""

from __future__ import annotations

import difflib
from typing import Iterable, NamedTuple

__all__ = [
    "EDGE_MARGIN",
    "Elision",
    "MIN_RUN",
    "ReadBlock",
    "plan",
    "stub_text",
]

# A run shorter than this is not worth a stub, and short coincidental matches (a run of
# blank lines, a repeated `    return None`) are not evidence that the agent is looking at
# the same region of the same file twice.
MIN_RUN = 8

# Lines held back on either side of a cut that lands inside the block, so an
# ``old_string`` straddling the boundary still occurs contiguously in one copy or the
# other. See the module docstring.
EDGE_MARGIN = 20

# Ceiling on the line-matcher's work per (block, base) pair. ``difflib`` is O(n*m) in the
# worst case and a read can be thousands of lines.
# ponytail: a flat product cap, not a smarter matcher. Raise it or switch to a
# hash-index-based longest-common-run if real files start exceeding it.
_MAX_CELLS = 4_000_000

# How many earlier verbatim reads of one path to try as bases. The newest is almost always
# the best match; the extras cost little and catch an alternating read pattern.
_MAX_BASES = 3


class ReadBlock(NamedTuple):
    """One tool result carrying a verbatim slice of a file.

    ``base_ok`` is whether this block may be *referenced* by a later one — true only for
    reads exempted by tool name, which are never superseded (see the module docstring).
    """

    id: str
    path: str
    text: str
    base_ok: bool


class Elision(NamedTuple):
    """A contiguous line run of one block that is byte-identical to a run of an earlier one.

    Indices are 0-based and half-open over ``text.splitlines()``.
    """

    path: str
    start: int
    end: int
    base_start: int
    base_end: int


def _longest_common_run(base: list[str], lines: list[str]) -> Elision | None:
    """The longest run of *lines* occurring contiguously in *base*, trimmed for safety.

    Returns None when nothing survives the minimum-length and edge-margin rules.
    """
    if len(base) < MIN_RUN or len(lines) < MIN_RUN:
        return None
    if len(base) * len(lines) > _MAX_CELLS:
        return None
    # autojunk would treat any line recurring in >1% of a 200+ line sequence as noise —
    # in source code that is every blank line, every `    pass`, every closing brace, and
    # it fragments exactly the runs this is looking for.
    match = difflib.SequenceMatcher(None, base, lines, autojunk=False).find_longest_match(
        0, len(base), 0, len(lines)
    )
    if match.size < MIN_RUN:
        return None
    b_start, b_end = match.b, match.b + match.size
    a_start, a_end = match.a, match.a + match.size
    # Only cuts INSIDE this block can split a quote; a run reaching the block's own first
    # or last line has nothing beyond it here for a quote to straddle into.
    if b_start > 0:
        b_start += EDGE_MARGIN
        a_start += EDGE_MARGIN
    if b_end < len(lines):
        b_end -= EDGE_MARGIN
        a_end -= EDGE_MARGIN
    if b_end - b_start < MIN_RUN:
        return None
    return Elision("", b_start, b_end, a_start, a_end)


def plan(blocks: Iterable[ReadBlock]) -> dict[str, Elision]:
    """Which blocks may drop which line run, keyed by tool-call id.

    Walks the reads in conversation order. A block is elided against the most similar of
    the last few *verbatim* reads of the same path seen before it; a block that is elided
    does not itself become a base, so references never chain.
    """
    out: dict[str, Elision] = {}
    bases: dict[str, list[list[str]]] = {}
    for block in blocks:
        lines = block.text.splitlines()
        best: Elision | None = None
        for base in reversed(bases.get(block.path, ())):
            found = _longest_common_run(base, lines)
            if found is not None and (
                best is None or found.end - found.start > best.end - best.start
            ):
                best = found
        if best is not None:
            out[block.id] = best._replace(path=block.path)
            continue
        if block.base_ok:
            per_path = bases.setdefault(block.path, [])
            per_path.append(lines)
            del per_path[:-_MAX_BASES]
    return out


def stub_text(elision: Elision, handle: str) -> str:
    """The reference that replaces the elided run. Line numbers are 1-based and inclusive."""
    return (
        f"«distil-reread handle={handle}» lines {elision.start + 1}-{elision.end} of this "
        f"result are byte-identical to lines {elision.base_start + 1}-{elision.base_end} of "
        f"the earlier read of {elision.path}, which is still in this conversation verbatim. "
        f"Call distil_expand with this handle to recover them here."
    )
