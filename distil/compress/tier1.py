"""Tier 1 — reversible digest behind a retrieval handle.

Large tool outputs / retrieved docs are replaced by a compact, *decision-aware*
digest plus a content handle. The full original is kept locally and can be
re-expanded on demand (`expand(handle)`), so this is lossless in effect.

The codec is decision-aware: any line the runtime cannot prove irrelevant is
preserved verbatim. The keep rules delegate to ``keep_policy``: the generic net
keeps DECISION: markers and error/failure lines, and each content kind adds its
own load-bearing lines (a test log's pass/fail summary, a traceback's frames, a
diff's hunk headers). The dropped span is folded behind a retrieval handle so
the full original is always recoverable.

On top of the per-kind keep decision, an outcome-aware dedup layer collapses
near-identical error/warn lines that differ only in an index/id: on a GREEN run
those errors did not cause the failure, so one sample per shape is enough signal.
"""

from __future__ import annotations

import hashlib
import re

from ..trajectory import Block, Kind
from .base import CompressResult
from . import query_relevance as _query_relevance
from .intent import relevant_lines
from .keep_policy import ContentKind, classify, must_keep
from .keep_policy import _SUMMARY_RE  # re-exported: keep_model imports it from here (1.14.1 compat)

_DIGESTIBLE = {Kind.TOOL_OUTPUT, Kind.RETRIEVED}


def _handle(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _must_keep(line: str) -> bool:
    """Compatibility shim for tests; superseded by keep_policy.must_keep + _SUMMARY_RE."""
    return _SUMMARY_RE.search(line) is not None or must_keep(line, ContentKind.GENERIC)


# Noise dedup — a run that logs the same ERROR on purpose in a loop produces
# hundreds of near-identical lines that differ only in an index/id/timestamp.
# Keeping them all floods the digest with noise (the second half of the
# "kept the noise, folded the answer" inversion). Normalize away the varying
# numerics to get the line's *shape*; the first few occurrences of a shape are
# kept as the signal "this error happens", the rest fold into the existing
# handle markers (recoverable like any folded line). DECISION: and verdict
# lines are exempt — they are answers, not noise.
_NUM_RE = re.compile(r"0x[0-9a-fA-F]+|\d+")


def _shape(line: str) -> str:
    return _NUM_RE.sub("#", " ".join(line.split())).lower()


# Outcome-aware routing — the first content-type profile. When the log's own
# verdict says GREEN (tests passed / build succeeded, nothing failed), the
# ERROR/WARN stdout is noise the SUT logged on purpose: by definition it did
# not fail the run, so one sample per shape is enough signal. A red or unknown
# outcome keeps the cautious default — there those errors may BE the answer.
# Detection reads only verdict lines (the trustworthy part of the log), and
# everything folded stays recoverable behind the handle.
_RED_RE = re.compile(
    r"""
      \b[1-9]\d*\ +(?:failed|failing|errors?)\b    # non-zero fail/error counts
    | ^\s*---\ +FAIL:                              # go subtest failure
    | ^\s*FAIL\b                                   # go package FAIL / bare FAIL
    | \bBUILD\ FAIL(?:ED|URE)\b                    # gradle/maven
    | \bexit\ (?:code|status)\ +[1-9]              # non-zero exit
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _outcome(lines: list[str]) -> str:
    """'green' | 'red' | 'unknown', judged from verdict lines only."""
    saw_verdict = False
    for ln in lines:
        if _SUMMARY_RE.search(ln) is None:
            continue
        saw_verdict = True
        if _RED_RE.search(ln) is not None:
            return "red"
    return "green" if saw_verdict else "unknown"


def digest(
    text: str,
    head: int = 3,
    tail: int = 1,
    max_repeats: int | None = None,
    intent: frozenset[str] = frozenset(),
) -> tuple[str, bool]:
    """Return (digest_text, changed). Keeps head/tail context + every must-keep
    line, replacing the dropped middle with a single handle marker. Near-identical
    error/warn repeats beyond `max_repeats` per shape fold with the rest; the
    default routes by the log's own outcome (green run -> 1 sample per shape,
    red/unknown -> 2).

    `intent` (query-aware salience): salient terms naming what the agent is looking
    for (its tool_use args + latest ask). Lines matching a *discriminating* intent
    term are additively pinned — this only ever widens the keep set, so the full
    original stays recoverable and the certificate is unaffected."""
    lines = text.splitlines()
    if len(lines) <= head + tail + 1:
        return text, False
    if max_repeats is None:
        max_repeats = 1 if _outcome(lines) == "green" else 2

    kind = classify(text)
    keep_idx = set(range(head)) | set(range(len(lines) - tail, len(lines)))
    if intent:
        # phase 1 + semantic bridge: lexical keeps, widened by morphology / synonyms / fuzzy /
        # vectors so answer lines with no shared word are pinned too. Additive — only widens.
        keep_idx |= relevant_lines(lines, intent, bridge=True)
        # phase 2 (query-aware salience): a learned model pins *semantically* relevant lines
        # a fixed lexical rule can't ("retry limit" ↔ `max_attempts = 5`). Additive union —
        # only ever widens the keep set. Returns None (→ exactly phase 1) until a certified
        # model is promoted; loaded once, so this is a dot-product per line when active.
        _qmodel = _query_relevance.get_model()
        if _qmodel is not None:
            keep_idx |= _qmodel.relevant_lines(lines, intent, kind)
    shape_seen: dict[str, int] = {}
    for i, ln in enumerate(lines):
        if "DECISION:" in ln or _SUMMARY_RE.search(ln) is not None:
            keep_idx.add(i)  # answers — never deduped
        elif must_keep(ln, kind):
            s = _shape(ln)
            shape_seen[s] = shape_seen.get(s, 0) + 1
            if shape_seen[s] <= max_repeats:
                keep_idx.add(i)

    # Phase-2 dark collection: record the *dropped* lines' numeric query-features so a future
    # retrain can learn semantic relevance from real expands. Content-free, gated + sampled,
    # fail-open — a no-op unless the live proxy enabled it (offline/cert/tests untouched).
    if intent:
        _dropped = set(range(len(lines))) - keep_idx
        if _dropped:
            from distil import query_flywheel

            query_flywheel.maybe_record(_handle(text), intent, lines, kind, _dropped)

    out: list[str] = []
    dropped = 0
    emitted = False  # did we actually elide anything?
    i = 0
    n = len(lines)
    while i < n:
        if i in keep_idx:
            if dropped:
                out.append(f"<< +{dropped} lines, handle={_handle(text)} >>")
                emitted = True
                dropped = 0
            out.append(lines[i])
        else:
            dropped += 1
        i += 1
    if dropped:
        out.append(f"<< +{dropped} lines, handle={_handle(text)} >>")
        emitted = True
    # `changed` must mean the output actually differs. When every line is must-keep —
    # a 400-line test log where the verdict policy pins each PASS line — nothing is
    # dropped, no marker is emitted, and the output is byte-identical to the input.
    # Returning True there made callers store an original for a block they never
    # digested: `RestoreStore._record` persisted plaintext to ~/.distil/restore for
    # content that can never need recovery, consuming the FIFO cap and widening the
    # at-rest surface for nothing. It also made the MCP server hand back a handle for
    # text it had not compressed.
    return "\n".join(out), emitted


class Tier1Reversible:
    tier = 1
    name = "tier1-reversible"

    def __init__(self, min_lines: int = 6) -> None:
        self.min_lines = min_lines

    def compress(self, blocks: list[Block]) -> CompressResult:
        from .structured import (
            fold,
            fold_records,
            template_fold,
        )  # local: avoids formatter stripping

        out: list[Block] = []
        restore: dict[str, str] = {}
        for b in blocks:
            if b.kind in _DIGESTIBLE:
                # 1) reversible structured compaction: flat columnar fold, then the
                #    nested-record fold (tool outputs with nested fields), then template mining
                compact = fold(b.text) or fold_records(b.text) or template_fold(b.text)
                if compact is not None:
                    restore[_handle(b.text)] = b.text  # byte-exact original, expandable
                    out.append(b.copy_with(compact))
                    continue
                # 1b) code skeleton: keep signatures + structure, elide bodies behind
                #     the expand handle. Python via ast, other languages via the
                #     zero-dep brace skeleton. API path only (this restore dict IS the
                #     expand source) — never on the lossless/subscription fold, where
                #     an elided body has no handle to recover it.
                from ..skeleton import code_skeleton, generic_code_skeleton

                sk = code_skeleton(b.text) or generic_code_skeleton(b.text)
                if sk is not None:
                    h = _handle(b.text)
                    restore[h] = b.text
                    out.append(b.copy_with(f"{sk}\n<< elided bodies, handle={h} >>"))
                    continue
                # 2) otherwise, decision-aware reversible digest for verbose blocks
                if b.text.count("\n") + 1 >= self.min_lines:
                    dtext, changed = digest(b.text)
                    if changed:
                        restore[_handle(b.text)] = b.text
                        out.append(b.copy_with(dtext))
                        continue
            out.append(b)
        return CompressResult(out, restore)
