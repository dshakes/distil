"""Per-content-type keep policy for the Tier-1 reversible digest.

The digest folds a verbose block's droppable middle behind a handle. WHICH lines
are droppable depends on the content: a passing test run's value is its pass/fail
summary, a traceback's is its frames, a diff's is its hunk headers. This module
classifies a block and returns a per-kind keep decision so tier1.digest stays
content-aware.
"""

from __future__ import annotations

import re
from enum import Enum


class ContentKind(str, Enum):
    GENERIC = "generic"
    LOG = "log"
    TRACEBACK = "traceback"
    DIFF = "diff"


# Generic salience net — every content kind inherits this.
_GENERIC_RE = re.compile(r"error|exception|traceback|fail|warn|panic|fatal", re.IGNORECASE)

# URLs are decision-relevant artifacts like SHAs and paths: live opus grading
# (docs/EVALUATION.md §2.1) caught a digested report URL turning a direct
# `fetchurl` into a `websearch` detour. URL-bearing lines route through
# must_keep, so tier1's per-shape repeat cap still bounds URL-spam logs
# (ponytail: same-shape distinct URLs beyond the cap fold; per-URL keeps if
# that ever bites a real trace).
_URL_RE = re.compile(r"https?://")

# Result/verdict line net — pins the outcome line of a command (test run, build,
# script). Superset of tier1's shipped _SUMMARY_RE + PR additions:
#   - "errored" count variant (added)
#   - "passing"/"failing" merged into the count arm (was separate)
#   - pytest === summary rule (added)
# ponytail: regex covers vitest/jest/pytest/mocha/go/cargo/gradle/maven + exit
# status — add a runner here if one slips through; keep _RED_RE in sync for
# non-zero fail patterns.
_SUMMARY_RE = re.compile(
    r"""
      \b\d+\ +(?:passed|passing|failed|failing|skipped|pending|todo|errors?|errored)\b
    | \btest\ result:                                            # cargo
    | ^\s*(?:ok|FAIL|PASS)\b                                     # go package ok/FAIL
    | ^\s*---\ +(?:FAIL|PASS|SKIP):                              # go subtest verdicts
    | \bBUILD\ (?:SUCCESS(?:FUL)?|FAIL(?:ED|URE))\b              # gradle/maven
    | \bexit\ (?:code|status)\ +\d+                              # command exit status
    | =+.*\b(?:passed|failed|error)\b.*=+                        # pytest === summary
    """,
    re.IGNORECASE | re.VERBOSE,
)

# TRACEBACK: keep every stack frame so the call site survives, not just the message.
_FRAME_RE = re.compile(
    r'Traceback \(most recent call last\)|^\s*File ".*", line \d+',
    re.MULTILINE,
)

# DIFF: keep file and hunk headers (the positional lines that give context).
_DIFF_KEEP_RE = re.compile(r"^@@|^diff --git|^---|^\+\+\+")


def classify(text: str) -> ContentKind:
    """Best-effort content kind for a block.

    A failing test log often also contains a traceback, so LOG wins over
    TRACEBACK: its policy still catches frames via the generic error net while
    also pinning the run summary.
    """
    if re.search(r"^diff --git|^@@", text, re.MULTILINE):
        return ContentKind.DIFF
    if _SUMMARY_RE.search(text):
        return ContentKind.LOG
    if _FRAME_RE.search(text):
        return ContentKind.TRACEBACK
    return ContentKind.GENERIC


def must_keep(line: str, kind: ContentKind) -> bool:
    """True if this line must be kept verbatim for this content kind.

    Every kind inherits the generic net (DECISION markers + error/failure lines).
    Additional load-bearing lines are pinned per kind: LOG pins result-summary
    lines, TRACEBACK pins stack frames, DIFF pins file and hunk headers.
    """
    if "DECISION:" in line or _GENERIC_RE.search(line) or _URL_RE.search(line):
        return True
    if kind is ContentKind.LOG:
        return _SUMMARY_RE.search(line) is not None
    if kind is ContentKind.TRACEBACK:
        return bool(_FRAME_RE.search(line))
    if kind is ContentKind.DIFF:
        return bool(_DIFF_KEEP_RE.search(line))
    return False


# ---- marker breakdown helpers (available for callers; not used in digest) ----

_ERR_RE = re.compile(r"error|exception|traceback|fatal|panic", re.IGNORECASE)
_WARN_RE = re.compile(r"warn", re.IGNORECASE)


def _flag(line: str) -> str:
    if _ERR_RE.search(line):
        return "error"
    if _WARN_RE.search(line):
        return "warn"
    return "other"


def summarize_dropped(dropped: list[str]) -> str:
    """Human-readable breakdown of dropped lines, e.g. ``2 error, 1 warn, 918 other``.

    Returns ``""`` when nothing flagged (no error or warn) was folded: the keep
    policy already surfaces flagged lines verbatim, so a fold with no flagged lines
    needs no annotation.
    """
    counts: dict[str, int] = {"error": 0, "warn": 0, "other": 0}
    for ln in dropped:
        counts[_flag(ln)] += 1
    if not counts["error"] and not counts["warn"]:
        return ""
    return ", ".join(f"{n} {name}" for name in ("error", "warn", "other") if (n := counts[name]))
