"""Salience protection — keep the needle, compress the haystack (model-free).

The decision-aware counterpart to entity protection: before a *lossy* strategy
crushes a block, we mark the spans that are most likely to carry the decision and
guarantee they survive — so aggressive compression stops dropping load-bearing
identifiers and directives.

Unlike syntactic entity protectors, salience here is a **model-free blend of three
signals**, and it operates at *line* granularity so the decision unit survives
together (e.g. ``notify_customer(PAY-12345)`` keeps the verb AND the target):

  1. PATTERN  — known identifier shapes (UUID, git sha, ``PREFIX-NNN``, email, IP, semver).
  2. ENTROPY  — high-information mixed-alnum tokens, so novel ID formats that no
                fixed regex anticipates are still caught (Shannon entropy + diversity).
  3. REFERENCE — tokens that recur across blocks are anchors the agent navigates by;
                a token referenced in two places is load-bearing, not noise.

This is not a competing heuristic to the certificate — it is a *frontier shifter*.
Protecting salient lines lowers the decision-change rate, so the conformal
certificate (:mod:`distil.conformal`) can certify a **more aggressive** level. The
guarantee still comes from the gate; this just moves where the gate says "safe".

No model, no network — a few hundred microseconds, so it stays in the request path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from ..trajectory import Block

__all__ = [
    "salient_tokens",
    "salient_lines",
    "surprise_lines",
    "reference_index",
    "protect",
]

# --- Signal 1: identifier shapes ------------------------------------------- #
_PATTERNS = re.compile(
    r"""(
        [0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}  # UUID
      | \b(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b      # hex hash / git sha (mixed)
      | \b[A-Z][A-Z0-9]{1,}-[A-Za-z0-9]{1,}\b                       # PAY-12345, NODE-77, GHSA-xv9
      | \b[\w.+-]+@[\w-]+\.[\w.-]+\b                                # email
      | \b(?:\d{1,3}\.){3}\d{1,3}\b                                 # IPv4
      | \bv?\d+\.\d+\.\d+(?:-[\w.]+)?\b                             # semver
      | \bhttps?://[^\s<>"')\]]+                                    # URL (live-grading find: a
                                                                    # digested URL turned fetchurl
                                                                    # into a websearch detour)
      | (?:~|\.{1,2})?/?(?:[\w.\-]+/){2,}[\w.\-]+                   # file path (>=2 separators)
    )""",
    re.VERBOSE,
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]{3,}")

# --- Signal 4: surprise ------------------------------------------------------ #
# Under a tight budget, compressors drop INCONGRUENT details first — yet in an
# agent trajectory the anomaly (the error line, the unexpected state, the failed
# assertion) is precisely what decides the next action ("lost if surprise",
# arXiv 2412.17483). These lines are over-retained regardless of token signals.
_SURPRISE = re.compile(
    r"""(?xi)
      \b(?:error|exception|traceback|fail(?:ed|ure|ing)?|fatal|panic|abort(?:ed)?
        |denied|refused|rejected|timed\s+out|timeout|unexpected|mismatch
        |assert(?:ion)?(?:\s*(?:error|failed))?|segfault|core\s+dumped
        |not\s+found|no\s+such|cannot|unable\s+to|permission\s+denied)\b
      | \bexit(?:\s+code|ed)?\s+[1-9]\d*\b
      | \b[EW]\d{3,}\b
    """,
)


def surprise_lines(text: str) -> list[str]:
    """Lines carrying an anomaly/incongruence signal (errors, failures,
    unexpected states, diff changes) — the load-bearing surprises a lossy
    compressor would otherwise drop first."""
    out: list[str] = []
    for ln in text.splitlines():
        if _SURPRISE.search(ln):
            out.append(ln)
        elif ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")) and ln.strip("+- "):
            out.append(ln)  # unified-diff changed line: the change IS the content
    return out


def _entropy(tok: str) -> float:
    """Shannon entropy (bits/char) of a token — high for random-looking strings."""
    if not tok:
        return 0.0
    counts = Counter(tok)
    n = len(tok)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _high_entropy(tok: str, min_entropy: float, min_len: int) -> bool:
    """A novel identifier the patterns miss: long, mixed letters+digits, high entropy."""
    if len(tok) < min_len:
        return False
    has_alpha = any(c.isalpha() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    return has_alpha and has_digit and _entropy(tok) >= min_entropy


def reference_index(blocks: list[Block]) -> Counter:
    """Count token occurrences across DISTINCT blocks — a token in >=2 blocks is an
    anchor the agent cross-references (a tool name in the schema and a tool output)."""
    idx: Counter = Counter()
    for b in blocks:
        for tok in set(_TOKEN.findall(b.text)):
            idx[tok] += 1
    return idx


def salient_tokens(
    text: str,
    *,
    ref_index: Counter | None = None,
    min_entropy: float = 3.2,
    min_len: int = 10,
    scorer: Callable[[str], Any] | None = None,
) -> set[str]:
    """The set of tokens in ``text`` worth protecting, from the three model-free
    signals (pattern + entropy + cross-reference).

    ``scorer`` is an optional pluggable seam: a callable ``(text) -> Iterable[str]``
    returning extra spans/tokens to protect (e.g. a semantic / NER / embedding
    scorer for free-form prose, where the model-free signals are weaker). It stays
    **off by default** so the runtime is model-free and zero-dependency; whatever it
    returns is unioned in and then judged by the same certificate — the seam adds
    coverage, never an unverified guarantee."""
    out: set[str] = set(m.group(0) for m in _PATTERNS.finditer(text))
    for tok in _TOKEN.findall(text):
        if tok in out:
            continue
        if _high_entropy(tok, min_entropy, min_len):
            out.add(tok)
        elif (
            ref_index is not None and ref_index.get(tok, 0) >= 2 and any(ch.isdigit() for ch in tok)
        ):
            out.add(tok)  # cross-referenced identifier-like anchor
    if scorer is not None:
        try:
            for span in scorer(text) or ():
                if isinstance(span, str) and span:
                    out.add(span)
        except Exception:  # noqa: BLE001 — a bad scorer must never break compression
            pass
    return out


def salient_lines(text: str, *, ref_index: Counter | None = None, **kw) -> list[str]:
    """The lines carrying a salient token OR a surprise signal — the decision
    *unit* (verb + target), not an isolated token. Preserved verbatim so the
    agent reads them intact."""
    toks = salient_tokens(text, ref_index=ref_index, **kw)
    surprises = set(surprise_lines(text))
    if not toks and not surprises:
        return []
    return [ln for ln in text.splitlines() if ln in surprises or any(t in ln for t in toks)]


# --- The composable protector ---------------------------------------------- #
Strategy = Callable[[list[Block], int], list[Block]]


def protect(strategy: Strategy, **salience_kw) -> Strategy:
    """Wrap any (possibly lossy) strategy so no salient *line* is ever dropped.

    For each block, if compression removed a salient line that was in the original,
    the line is re-appended verbatim under a compact ``⟦keep⟧`` marker — the needle
    survives even when the haystack is crushed. Never emits a block larger than the
    original (reject-if-bigger), so protection can only *reduce* savings, never invert
    them. Lossless strategies already keep everything, so this is a no-op there; its
    value is making the aggressive operating points decision-safe.
    """

    def wrapped(blocks: list[Block], turn: int) -> list[Block]:
        ref = reference_index(blocks)
        out = strategy(blocks, turn)
        by_id = {b.id: b for b in blocks}
        fixed: list[Block] = []
        for c in out:
            orig = by_id.get(c.id)
            if orig is None or orig.text == c.text:
                fixed.append(c)
                continue
            want = salient_lines(orig.text, ref_index=ref, **salience_kw)
            c_lines = set(c.text.splitlines())  # exact line membership, not substring
            missing = [ln for ln in want if ln.strip() and ln not in c_lines]
            if not missing:
                fixed.append(c)
                continue
            # Re-inject each needle on its OWN line, verbatim. Indentation is semantic
            # in code/YAML, and line structure carries meaning too — joining needles
            # onto one line with separators would flatten multi-line context into a
            # single misleading line. A single ⟦keep⟧ header marks the recovered region;
            # `missing` already excludes blank lines, so nothing is stripped.
            patched = c.text + "\n⟦keep⟧\n" + "\n".join(missing)
            # Guarantee the needle survives: if the patched form would exceed the
            # original (so re-injection saves nothing), fall back to the ORIGINAL
            # block byte-exact — never to ``c``, which has the salient line removed.
            fixed.append(orig.copy_with(patched) if len(patched) < len(orig.text) else orig)
        return fixed

    wrapped.__name__ = f"protect({getattr(strategy, '__name__', 'strategy')})"
    return wrapped
