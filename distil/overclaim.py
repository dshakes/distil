"""Overclaim — did compression make the context more certain than the source?

:mod:`distil.retention` measures whether a fact was *lost*. This measures whether a
fact was *distorted*: the value survives, but the uncertainty attached to it does not.

    original    "the timeout is approximately 4000 ms"
    compressed  "the timeout is 4000 ms"

Every retention metric scores that as perfect. The number is there, byte-identical.
But the agent has been handed a precise figure where the source offered an estimate,
and an agent that trusts precision will act differently — it will stop measuring, or
it will assert the figure downstream as fact. The hedge *was* the information.

The same shape covers modality ("may fail" -> "will fail"), attribution ("reportedly
deprecated" -> "deprecated") and negation-scope loss. Work on information fidelity in
compressed financial analysis (arXiv:2606.29251) formalises this as *overclaim risk*
and finds it changes downstream decisions independently of factual recall.

Direction matters and is not symmetric. Adding hedges that were not in the source
(*underclaim*) makes an agent over-cautious — wasteful, rarely dangerous. Dropping
them makes it over-confident — that is the failure worth a metric. Both are counted;
only overclaim is gated.

Content-free: hedge spans are compared and discarded in-call; only counts escape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Hedge classes. Grouping synonyms means a legitimate reshaping ("approximately" ->
# "about") is not scored as an overclaim — only the *disappearance* of hedging is.
_HEDGE_CLASSES: dict[str, tuple[str, ...]] = {
    "approx": ("approximately", "approx", "about", "around", "roughly", "circa", "~", "±", "or so"),
    "modal": ("may", "might", "could", "possibly", "perhaps", "potentially", "likely", "probably"),
    "seem": ("appears", "appear", "seems", "seem", "suggests", "suggest", "indicates", "indicate"),
    "attrib": ("reportedly", "allegedly", "according to", "claimed", "purportedly", "said to"),
    # Bounds are split by DIRECTION. Sharing one class made "at least 3 retries" ->
    # "at most 3 retries" score as preserved hedging, because both terms mapped to the
    # same label — a floor silently inverted into a ceiling, which is the strongest
    # possible claim change short of dropping the hedge outright. Grouping exists to
    # forgive synonyms, not antonyms.
    "bound_lower": ("at least", "no fewer than", "over"),
    "bound_upper": ("at most", "up to", "no more than", "under"),
    "partial": (
        "some",
        "several",
        "many",
        "most",
        "few",
        "often",
        "sometimes",
        "usually",
        "typically",
    ),
    "unsure": ("unclear", "unknown", "uncertain", "not sure", "unconfirmed", "tbd", "estimated"),
}

# Classes whose meanings are opposites rather than synonyms. Substituting one for the
# other is a reversal, not a reshaping, and is scored as such.
_OPPOSED: dict[str, str] = {"bound_lower": "bound_upper", "bound_upper": "bound_lower"}

# Longest-first so "at least" wins over "least", and "not sure" over "sure".
_ALL_HEDGES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((term, cls) for cls, terms in _HEDGE_CLASSES.items() for term in terms),
        key=lambda pair: -len(pair[0]),
    )
)

# A claim anchor: a number, or a capitalised/quoted token worth hedging.
#
# The delimiters are load-bearing. Without them the pattern matched number FRAGMENTS —
# `8` inside the JSON array `[88.1,89.0]`, `2023` inside the URL path `.../202313` —
# and every such fragment became a "claim" whose hedging could then appear to change.
# Measured on the bundled corpus, that produced 24 phantom overclaims against a
# compressor whose output was byte-identical to its input.
_ANCHOR = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?[A-Za-z%]*(?![\w.])"
    r"|\"[^\"]{2,40}\""
    r"|(?<!\w)[A-Z][\w.-]{2,}(?!\w)"
)

# How far from the anchor a hedge still counts as attached to it. Wide enough for
# "approximately 4000" and "4000 or so", tight enough that an unrelated hedge in the
# next sentence is not credited.
_WINDOW = 48


@dataclass(frozen=True)
class Claim:
    """One hedged assertion: an anchor plus the class of hedge attached to it."""

    anchor: str
    hedge_class: str


@dataclass
class OverclaimProbe:
    """Hedge outcomes for one comparison.

    ``preserved + overclaimed + inverted == total`` — every hedged claim in the source
    either kept a hedge of its class, lost its hedging, or had it reversed.
    ``underclaimed`` is counted separately because it is not a subset of the source's
    claims.
    """

    total: int = 0
    preserved: int = 0
    overclaimed: int = 0
    underclaimed: int = 0
    inverted: int = 0

    @property
    def fidelity(self) -> float:
        """Fraction of hedged claims that kept their hedging."""
        return self.preserved / self.total if self.total else 1.0

    @property
    def overclaim_rate(self) -> float:
        """Fraction stripped of hedging — the gated number."""
        return self.overclaimed / self.total if self.total else 0.0

    def add(self, other: OverclaimProbe) -> None:
        self.total += other.total
        self.preserved += other.preserved
        self.overclaimed += other.overclaimed
        self.underclaimed += other.underclaimed
        self.inverted += other.inverted

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "preserved": self.preserved,
            "overclaimed": self.overclaimed,
            "underclaimed": self.underclaimed,
            "inverted": self.inverted,
            "fidelity": round(self.fidelity, 4),
            "overclaim_rate": round(self.overclaim_rate, 4),
        }


# Structural punctuation density above which a line is data, not prose.
_STRUCT_CHARS = set('{}[]":,=|')
_STRUCT_RATIO = 0.12


def _is_structured(line: str) -> bool:
    """Is this line structured data rather than prose?

    Hedging is a natural-language act. A JSON tool result is not making a claim when
    it contains the token `most` in a field name and the number `6` in a value — but
    a proximity rule cannot tell the difference, and on the bundled corpus that
    mismatch produced every remaining phantom overclaim: `"steps_evaluated":6` bound
    to an unrelated hedge word elsewhere on an 800-character JSON line.

    Skipping these costs nothing real. A model reading structured output takes the
    values at face value; there is no hedge there to preserve or drop.
    """
    stripped = line.strip()
    if not stripped:
        return True
    if stripped[0] in "{[" or stripped.startswith(("- {", "| ")):
        return True
    dense = sum(1 for ch in stripped if ch in _STRUCT_CHARS) / len(stripped)
    return dense >= _STRUCT_RATIO


def _find_hedges(text: str) -> list[tuple[int, int, str]]:
    """(start, end, class) for every hedge term, non-overlapping, longest-first."""
    lowered = text.lower()
    taken: list[tuple[int, int, str]] = []
    for term, cls in _ALL_HEDGES:
        pattern = re.escape(term) if not term[0].isalpha() else rf"(?<!\w){re.escape(term)}(?!\w)"
        for m in re.finditer(pattern, lowered):
            if any(s < m.end() and m.start() < e for s, e, _ in taken):
                continue
            taken.append((m.start(), m.end(), cls))
    return taken


def extract_claims(text: str) -> set[Claim]:
    """Hedged claims in a block: an anchor with a hedge of some class nearby.

    An anchor with no hedge within the window is not a claim here — it is an
    unqualified fact, which is :mod:`distil.retention`'s job, not this module's.
    """
    if not text:
        return set()
    claims: set[Claim] = set()
    # Binding is per LINE, not across the whole block. Character distance within a
    # block is not stable under compression: removing an unrelated span shifts every
    # offset after it, so a hedge and an anchor that were 60 chars apart can end up 40
    # apart and a claim appears out of nowhere — or vice versa. A line is the smallest
    # unit that survives reflow, which makes the binding reproducible rather than
    # dependent on what else happened to be in the block.
    for line in text.splitlines():
        if _is_structured(line):
            continue
        hedges = _find_hedges(line)
        if not hedges:
            continue
        for m in _ANCHOR.finditer(line):
            a_start, a_end = m.span()
            for h_start, h_end, cls in hedges:
                if h_end <= a_start and a_start - h_end <= _WINDOW:
                    claims.add(Claim(m.group(0), cls))
                elif a_end <= h_start and h_start - a_end <= _WINDOW:
                    claims.add(Claim(m.group(0), cls))
    return claims


def _anchor_survives(anchor: str, compressed: str) -> bool:
    """Did the VALUE survive somewhere a claim could still be read from?

    A raw `anchor in compressed` substring test is far too weak. On the bundled
    corpus it credited survival for a one-character anchor (`8`) that had been
    digested away with its whole prose sentence, purely because the digit reappeared
    inside an unrelated JSON array — and the missing hedge was then reported as an
    overclaim. The value must survive the same way it was found: as a delimited
    token, on a line that is prose rather than structured data.
    """
    for line in compressed.splitlines():
        if _is_structured(line):
            continue
        for m in _ANCHOR.finditer(line):
            if m.group(0) == anchor:
                return True
    return False


def score(original: str, compressed: str) -> OverclaimProbe:
    """Grade hedge preservation between an original block and its compression."""
    truth = extract_claims(original)
    seen = extract_claims(compressed)
    probe = OverclaimProbe(total=len(truth))
    seen_by_anchor: dict[str, set[str]] = {}
    for c in seen:
        seen_by_anchor.setdefault(c.anchor, set()).add(c.hedge_class)
    for claim in truth:
        classes = seen_by_anchor.get(claim.anchor)
        if classes and claim.hedge_class in classes:
            probe.preserved += 1
        elif _anchor_survives(claim.anchor, compressed) and not classes:
            # the value survived, its hedging did not — the overclaim case
            probe.overclaimed += 1
        elif classes and _OPPOSED.get(claim.hedge_class) in classes:
            # Hedged, but in the OPPOSITE direction: "at least 3 retries" became
            # "at most 3 retries". Treating that as "still hedged" scored a reversed
            # claim as faithful — worse than dropping the hedge, because a floor read
            # as a ceiling is a confident wrong bound rather than a visible gap.
            # Counted separately for the same reason `continuation` reports status
            # flips apart from recall: averaging it away hides which way the error went.
            probe.inverted += 1
        elif classes:
            # hedged, but with a different class. Still hedged: not an overclaim.
            probe.preserved += 1
        else:
            # the anchor itself is gone; that is plain loss, retention's business,
            # not a distortion. Do not double-count it here.
            probe.total -= 1
    truth_anchors = {c.anchor for c in truth}
    probe.underclaimed = sum(
        1 for c in seen if c.anchor not in truth_anchors and c.anchor in original
    )
    return probe


def format_probe(probe: OverclaimProbe) -> str:
    if not probe.total:
        return "overclaim: no hedged claims found — nothing to grade."
    lines = [
        f"hedge fidelity            {probe.fidelity:6.1%}  ({probe.preserved}/{probe.total} claims kept their hedging)",
        f"  overclaimed             {probe.overclaim_rate:6.1%}  ({probe.overclaimed})"
        "  <- value kept, uncertainty dropped",
    ]
    if probe.inverted:
        lines.append(
            f"  inverted bounds         {probe.inverted:>6}       "
            "  <- floor read as ceiling; worse than a dropped hedge"
        )
    if probe.underclaimed:
        lines.append(
            f"  underclaimed            {probe.underclaimed:>6}       "
            "  <- hedging added; cautious, not dangerous"
        )
    return "\n".join(lines)
