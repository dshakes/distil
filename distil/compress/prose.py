"""Reversible prose condensing — sentence-level, for large Markdown/text reads.

The line digest (tier1) folds repeated or low-salience *lines*; prose has neither, so a
large Markdown read used to pass through almost whole (4.8% off on the Headroom 0.42
head-to-head, where Headroom's Kompress model takes ~35% by dropping tokens).

This drops whole *sentences* instead of tokens, so everything the agent does see still
reads as written. Kept: headings, code fences, tables, blank structure, the lead sentence
of every paragraph and list item, and every sentence naming something specific (a
number, path, URL, hash, ``key=value``, inline code, an error word, or a term from the
current request). Each elided run becomes ``[…]``; the caller records the original
behind one handle. Pure function of (text, intent), so a block renders the same bytes on
every turn.

Used only above the adapter's lossy floor (``_DIGEST_MIN_TOKENS``): under it the agent
reads prose whole, for the same reason it reads any fresh output whole.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from .keep_policy import _GENERIC_RE, _IDENT_RE, _SUMMARY_RE, ContentKind, must_keep

#: The ``cat -n`` / Read-tool line prefix (``   12\t`` or ``12→``), kept on output.
_NUM_PREFIX = re.compile(r"^(\s*\d+(?:\t|→))")
_HEADING = re.compile(r"^\s{0,3}(#{1,6}\s|={3,}\s*$|-{3,}\s*$)")
_LIST = re.compile(r"^(\s*)([-*+]|\d{1,3}[.)])\s+")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SPECIFIC = re.compile(r"`[^`]+`|\d{2,}|\d\.\d")
# A sentence ends at . ! ? (optionally closed by a quote/bracket) before whitespace and an
# opening capital, digit, quote or bracket. Abbreviations split wrongly at worst: that only
# makes the unit smaller, never drops a kept fact.
_SENTENCE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z]{2,}")

#: A block is prose when at least this share of its non-blank lines read as sentences.
PROSE_SHARE = 0.6
#: Share of the sentence text kept (leads first, then the most specific sentences).
KEEP_SHARE = 0.6
ELIDED = "[…]"


def _body(line: str) -> tuple[str, str]:
    m = _NUM_PREFIX.match(line)
    return (m.group(1), line[m.end() :]) if m else ("", line)


def _is_sentence_line(body: str) -> bool:
    words = _WORD.findall(body)
    if len(words) < 6:
        return False
    if len({w.lower() for w in words}) < len(words) / 2:
        return False  # repetitive filler or a templated row, not a sentence
    letters = sum(len(w) for w in words)
    return letters / max(1, len(body.strip())) >= 0.6 and not body.rstrip().endswith(("{", ";"))


def is_prose(text: str) -> bool:
    """Mostly natural-language sentences (Markdown, docs, a fetched article), not code/logs."""
    bodies = [_body(ln)[1] for ln in text.splitlines()]
    body_lines = [b for b in bodies if b.strip() and not _FENCE.match(b)]
    if len(body_lines) < 8:
        return False
    prose = sum(1 for b in body_lines if _is_sentence_line(b) or _HEADING.match(b))
    return prose / len(body_lines) >= PROSE_SHARE


def _pinned(line: str) -> bool:
    """What the line digest never folds (DECISION:, errors, verdicts) stays whole here too."""
    return _SUMMARY_RE.search(line) is not None or must_keep(line, ContentKind.GENERIC)


def _score(sentence: str, intent: Collection[str]) -> float:
    """Specific things named per character: what makes a sentence worth its tokens."""
    hits = len(_IDENT_RE.findall(sentence)) + len(_SPECIFIC.findall(sentence))
    hits += len(_GENERIC_RE.findall(sentence))
    low = sentence.lower()
    hits += 2 * sum(1 for t in intent if t in low)
    return hits / (len(sentence) + 1)


def condense(
    text: str, intent: Collection[str] = frozenset(), keep_share: float = KEEP_SHARE
) -> str | None:
    """The condensed text, or None when nothing would be elided.

    Lead sentences always stay; the rest are kept by `_score`, best first, until they
    fill `keep_share` of the sentence text. A sentence wrapped across lines is one unit."""
    # Pass 1: lines -> segments, each segment tied to a sentence id.
    lines: list[tuple[str, str, list[tuple[int, str]]] | str] = []  # (prefix, head, segs) | raw
    sentences: list[list[str]] = []  # sentence id -> its segments
    lead_ids: set[int] = set()
    fenced, lead, open_sid = False, True, None
    for line in text.splitlines():
        prefix, body = _body(line)
        if _FENCE.match(body):
            fenced = not fenced
            lines.append(line)
            continue
        if fenced or not body.strip() or _HEADING.match(body) or body.lstrip().startswith("|"):
            lines.append(line)
            lead, open_sid = True, None
            continue
        item = _LIST.match(body)
        if item:
            lead, open_sid = True, None
        if not _is_sentence_line(body) or _pinned(body):
            lines.append(line)
            lead, open_sid = False, None
            continue
        head = item.group(0) if item else body[: len(body) - len(body.lstrip())]
        parts = _SENTENCE.split(body[len(head) :])
        segs: list[tuple[int, str]] = []
        for i, part in enumerate(parts):
            if i == 0 and open_sid is not None:
                sid = open_sid  # the tail of a sentence begun on the line above
                sentences[sid].append(part)
            else:
                sid = len(sentences)
                sentences.append([part])
                if lead:
                    lead_ids.add(sid)
                lead = False
            segs.append((sid, part))
        last = parts[-1].rstrip()
        open_sid = None if last.endswith((".", "!", "?", ":")) else segs[-1][0]
        lines.append((prefix, head, segs))
    if len(sentences) <= len(lead_ids):
        return None

    # Pass 2: choose. Ties break on position, so the result is a pure function of input.
    full = [" ".join(seg) for seg in sentences]
    budget = keep_share * sum(len(f) for f in full) - sum(len(full[i]) for i in lead_ids)
    keep = set(lead_ids)
    ranked = sorted(
        (i for i in range(len(full)) if i not in lead_ids),
        key=lambda i: (-_score(full[i], intent), i),
    )
    for i in ranked:
        if _score(full[i], intent) <= 0 or len(full[i]) > budget:
            continue
        keep.add(i)
        budget -= len(full[i])
    if len(keep) == len(sentences):
        return None

    # Pass 3: render, one ELIDED per dropped run.
    out: list[str] = []
    for entry in lines:
        if isinstance(entry, str):
            out.append(entry)
            continue
        prefix, head, segs = entry
        kept: list[str] = []
        gap = False
        for sid, part in segs:
            if sid in keep:
                if gap:
                    kept.append(ELIDED)
                kept.append(part)
                gap = False
            else:
                gap = True
        if gap:
            kept.append(ELIDED)
        if kept != [ELIDED]:
            out.append(prefix + head + " ".join(kept))
        elif not out or not out[-1].endswith(ELIDED):
            out.append(prefix + head + ELIDED)
    return "\n".join(out)
