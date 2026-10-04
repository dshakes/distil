"""Re-fetch verbatim — a tool result the agent fetched again because distil digested it.

On 300 SWE-bench Lite tasks served through distil, agents called ``distil_expand`` five
times. When an earlier result had been digested and the agent needed what was folded, it
re-ran the command or re-read the file instead — and, under a client that caches through
its newest turn, the re-fetch was digested on first sight too. A byte-identical re-run
hashes to the same handle and comes back as the *same stub*: the agent asked again and was
shown exactly what had just failed it. Three, four reads of one window were observed.

This module recognises the re-fetch by its **content**, not its command. A result is a
re-fetch when most of its lines already occurred in an earlier tool result (as the client
sent it) but not in what distil *forwarded* for those results — i.e. the agent is asking for
lines it was only ever shown folded. Such a result is forwarded verbatim. Content, because
the commands vary freely (``cat f | sed -n 1,80p`` then ``sed -n 55,100p f`` then a
``python`` heredoc printing the same lines) while the lines do not.

What it does not do: re-inflate the *earlier* block. That block is in the provider's cached
prefix, and rewriting it costs a 1.25x write of the whole prefix (ADR 0008). The earlier
digest stays; the new block — which has never been sent, so has no cached rendering to
break — is the one kept whole.

Stateless and prefix-deterministic (ADR 0008, ADR 0010): the verdict for a block depends
only on the blocks before it and on how they were forwarded, which are themselves functions
of their own prefixes. So a block encodes to the same bytes on every turn that carries it,
and the adapter recomputes it from the history on each request with no session state.

Monotone toward verbatim: the rule can only *stop* a digest, never cause one. An adversary
who controls an earlier tool result can at most make a later block cost its full tokens
(denial of savings, ADR 0009), never hide a line.
"""

from __future__ import annotations

import re

__all__ = ["MIN_LINES", "THRESHOLD", "Tracker", "line_keys"]

# A line-number prefix the reader added, not the file: ``cat -n``/``nl`` (`   12\t`), the
# edit tool's ``view`` (`    12\t`), ``grep -n`` (`12:`/`12-`) and ``grep -rn``
# (`path:12:`). Stripped so a numbered and an unnumbered view of one line compare equal.
_NUMBERED = re.compile(r"^\s*(?:[^\s:]+[:-])?\d+[\t:-]")

# Lines shorter than this (after stripping) carry no identity — `)`, `}`, `else:` — and a
# match on them is coincidence, not a re-fetch.
_MIN_CHARS = 4

# A block needs this many distinct keyed lines before its coverage means anything.
MIN_LINES = 3

# Share of a block's keyed lines that must have been seen before, and NOT forwarded
# verbatim, for it to count as a re-fetch. The same 0.8 the offline replay uses to call a
# call redundant (benchmarks/reinflate_replay.py).
THRESHOLD = 0.8


def line_keys(text: str) -> frozenset[str]:
    """The comparable lines of *text*: line-number prefix and surrounding blanks dropped."""
    out = set()
    for line in text.splitlines():
        key = _NUMBERED.sub("", line, count=1).strip()
        if len(key) >= _MIN_CHARS:
            out.add(key)
    return frozenset(out)


class Tracker:
    """Lines seen so far in one walk over the history: as sent, and as forwarded.

    One per compression pass, fed every tool result in conversation order. Holds nothing
    between requests — the history *is* the state.
    """

    def __init__(self) -> None:
        self._sent: set[str] = set()
        self._forwarded: set[str] = set()

    def is_refetch(self, text: str) -> bool:
        """Whether *text* is mostly lines the agent has already been sent but only seen folded."""
        keys = line_keys(text)
        if len(keys) < MIN_LINES:
            return False
        need = THRESHOLD * len(keys)
        return len(keys & self._sent) >= need and len(keys & self._forwarded) < need

    def observe(self, sent: str, forwarded: str) -> None:
        """Record one tool result: its text as the client sent it and as distil forwarded it."""
        self._sent |= line_keys(sent)
        self._forwarded |= line_keys(forwarded)
