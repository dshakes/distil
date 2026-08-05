"""Continuation — can the agent still resume the task?

Recall asks "is the fact there". Decision-equivalence asks "is the next action the
same". Neither asks the question a long-horizon agent fails on first: *does it still
know what it was doing and what is left to do*.

Factory.ai grades this as its own probe axis alongside recall, artifact and decision
(https://factory.ai/news/evaluating-compression), and it is separable from all three.
A compressed context can preserve every fact and still lose the plan.

The asymmetry that makes this measurable is the same one that runs through the rest
of distil's fidelity work:

  * dropping a **done** item is cheap — at worst the agent redoes finished work.
  * dropping a **pending** item is silent — the agent never does it, reports success,
    and nothing in the transcript says a step was skipped.
  * flipping a status is worst of all — `done -> pending` wastes a turn, and
    `pending -> done` means work is dropped while the agent believes it complete.

So ``pending_recall`` is the headline and ``flips`` is reported separately, rather
than averaging everything into one number that hides which way the errors go.

Content-free: obligation text is compared and discarded in-call; only counts escape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Status(str, Enum):
    PENDING = "pending"
    DONE = "done"


# Markers that open an obligation. Ordered by specificity; first hit wins per line.
# `\b` belongs ONLY on the word-shaped alternatives. After `[x]` it can never match:
# `]` and the following space are both non-word characters, so there is no boundary
# between them, and the whole pattern silently never fires. That bug made every
# checkbox list invisible to this probe — a metric reporting perfect continuation on
# text it had failed to parse.
_PENDING_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\[\s\]|(?:TODO|FIXME|NEXT|REMAINING|STILL\s+(?:need|to)|PENDING)\b)"
    r"[:.\s-]*(?P<t>.+)",
    re.I,
)
_DONE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\[[xX]\]|(?:DONE|COMPLETED?|FIXED)\b|✓|✔)[:.\s-]*(?P<t>.+)",
    re.I,
)
# A stated goal is an obligation that is never "done" until the task ends.
_GOAL_RE = re.compile(
    r"\b(?:your task is(?:\s+to)?|the goal is(?:\s+to)?|objective:|you must)\b[:.\s-]*(?P<t>.+)",
    re.I,
)

_MIN_OBLIGATION = 8
_KEY_WORDS = 6  # obligations are matched on their first N normalised words


@dataclass(frozen=True)
class Obligation:
    """One unit of remaining or completed work, keyed on its opening words.

    Keying on a prefix rather than the full string is deliberate: compression legitimately
    truncates and re-wraps list items, and an exact-string match would report that as lost
    work. The prefix is long enough to distinguish two real obligations and short enough
    to survive reflow.
    """

    key: str
    status: Status


def _key(text: str) -> str:
    words = re.sub(r"[^\w\s]+", " ", text.lower()).split()
    return " ".join(words[:_KEY_WORDS])


def extract_obligations(text: str) -> set[Obligation]:
    """Pending and completed work items stated anywhere in a block."""
    if not text:
        return set()
    out: set[Obligation] = set()
    for line in text.splitlines():
        for pattern, status in ((_DONE_RE, Status.DONE), (_PENDING_RE, Status.PENDING)):
            m = pattern.match(line)
            if m:
                body = m.group("t").strip()
                if len(body) >= _MIN_OBLIGATION:
                    out.add(Obligation(_key(body), status))
                break
    for m in _GOAL_RE.finditer(text):
        body = m.group("t").strip().split("\n")[0]
        if len(body) >= _MIN_OBLIGATION:
            out.add(Obligation(_key(body), Status.PENDING))
    return out


@dataclass
class ContinuationProbe:
    """Obligation outcomes for one comparison."""

    pending_total: int = 0
    pending_kept: int = 0
    done_total: int = 0
    done_kept: int = 0
    flips: int = 0

    @property
    def pending_recall(self) -> float:
        """The headline: share of remaining work still visible after compression."""
        return self.pending_kept / self.pending_total if self.pending_total else 1.0

    @property
    def done_recall(self) -> float:
        return self.done_kept / self.done_total if self.done_total else 1.0

    @property
    def dropped_work(self) -> int:
        """Pending items the agent can no longer see — the silent-skip count."""
        return self.pending_total - self.pending_kept

    def add(self, other: ContinuationProbe) -> None:
        self.pending_total += other.pending_total
        self.pending_kept += other.pending_kept
        self.done_total += other.done_total
        self.done_kept += other.done_kept
        self.flips += other.flips

    def to_dict(self) -> dict[str, float | int]:
        return {
            "pending_total": self.pending_total,
            "pending_kept": self.pending_kept,
            "dropped_work": self.dropped_work,
            "done_total": self.done_total,
            "done_kept": self.done_kept,
            "flips": self.flips,
            "pending_recall": round(self.pending_recall, 4),
            "done_recall": round(self.done_recall, 4),
        }


def _fold(blocks: Iterable[str]) -> dict[str, Status]:
    """Final status per obligation, folding blocks in order.

    A plan legitimately evolves: `- [ ] wire it up` at turn 1 becomes `- [x] wire it
    up` at turn 3. Unioning every turn's obligations into a set would hold that item
    under BOTH statuses and score the completion as a status flip — reporting the
    agent's own progress as compression damage. Later blocks supersede earlier ones,
    exactly as :class:`distil.artifacts.Ledger` does for file state.
    """
    final: dict[str, Status] = {}
    for block in blocks:
        for o in extract_obligations(block):
            final[o.key] = o.status
    return final


def score(original: Iterable[str], compressed: Iterable[str]) -> ContinuationProbe:
    """Grade whether the plan survived, not just the facts."""
    truth_status = _fold(original)
    seen_status = _fold(compressed)
    truth = {Obligation(key, status) for key, status in truth_status.items()}

    probe = ContinuationProbe()
    for o in truth:
        if o.status is Status.PENDING:
            probe.pending_total += 1
        else:
            probe.done_total += 1
        got = seen_status.get(o.key)
        if got is None:
            continue
        if got is o.status:
            if o.status is Status.PENDING:
                probe.pending_kept += 1
            else:
                probe.done_kept += 1
        else:
            probe.flips += 1
    return probe


def score_texts(original: str, compressed: str) -> ContinuationProbe:
    return score([original], [compressed])


def format_probe(probe: ContinuationProbe) -> str:
    if not (probe.pending_total or probe.done_total):
        return "continuation: no stated obligations found — nothing to grade."
    lines = [
        f"pending-work recall       {probe.pending_recall:6.1%}"
        f"  ({probe.pending_kept}/{probe.pending_total} remaining items still visible)",
    ]
    if probe.dropped_work:
        lines.append(
            f"  dropped work            {probe.dropped_work:>6}       "
            "  <- agent cannot see it, will report success without doing it"
        )
    if probe.done_total:
        lines.append(
            f"  completed-work recall   {probe.done_recall:6.1%}  ({probe.done_kept}/{probe.done_total})"
        )
    if probe.flips:
        lines.append(f"  status flips            {probe.flips:>6}         <- done/pending inverted")
    return "\n".join(lines)
