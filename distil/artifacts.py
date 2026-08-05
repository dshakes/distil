"""Artifact-state fidelity — does the agent still know the state of the files it touched?

The ``artifacts`` dimension in :mod:`distil.retention` asks whether a path *string*
survived compression. That is a weaker question than the one that matters, and the
gap between them is where agents actually break.

A trajectory can preserve every path token and still leave the agent with a false
model of its workspace. Keep "created ``config.yaml``" from turn 2, drop "deleted
``config.yaml``" from turn 9, and string recall reads 100% while the agent now
believes a file exists that does not. It will confidently plan around it.

Factory.ai's probe study over 36,611 production engineering messages found this
unsolved: every compression method they tested scored **2.19–2.45 out of 5.0** on
knowing which files were created, modified or examined
(https://factory.ai/news/evaluating-compression). String-presence metrics cannot
see the failure, because the string is present.

This module measures the thing directly, and separates two outcomes that string
recall conflates:

  * **lost**  — the path is gone. The agent knows that it does not know.
  * **stale** — the path survives carrying the WRONG final state. The agent does
                not know that it does not know, and acts on the false belief.

`stale` is strictly worse than `lost`, so it is reported as its own rate rather
than folded into a single accuracy number. A compressor that drops a whole file
history is safer than one that preserves half of it.

Content-free by construction: the ledger is built, compared and discarded inside a
single call. What leaves this module is counts — never a path, never file content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class Op(str, Enum):
    """A file-state transition, ordered by how strongly it asserts existence.

    Ordering matters for the fold: when one turn's text mentions the same path under
    two verbs (``"read config.yaml, then deleted config.yaml"``), the *strongest*
    assertion within that turn wins, so a delete is never masked by an incidental
    read on the same line.
    """

    READ = "read"
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


# Strength order for intra-turn conflicts. DELETE dominates: it is the transition
# whose loss produces the phantom-file failure this module exists to catch.
_STRENGTH = {Op.READ: 0, Op.CREATE: 1, Op.MODIFY: 2, Op.DELETE: 3}

# A path-ish token. Deliberately stricter than retention's _PATH_RE: that one is tuned
# for recall over any path-shaped string, while a state ledger keyed on false positives
# would report phantom churn. Requires a file extension or a leading ./ ~/ or /.
_PATH = r"(?:(?:~|\.{1,2})?/)?(?:[\w.-]+/)*[\w-]+\.[A-Za-z][\w]{0,9}"

# Verb -> Op. Matched against the text of tool calls and their narration. Each entry is
# (regex, op); the path is group "p".
_PATTERNS: tuple[tuple[re.Pattern[str], Op], ...] = (
    # Structured agent tool calls, e.g. Write(file_path="src/a.py")
    (re.compile(rf'\bWrite\s*\(\s*file_path\s*=\s*["\']?(?P<p>{_PATH})', re.I), Op.CREATE),
    (re.compile(rf'\bEdit\s*\(\s*file_path\s*=\s*["\']?(?P<p>{_PATH})', re.I), Op.MODIFY),
    (re.compile(rf'\bRead\s*\(\s*file_path\s*=\s*["\']?(?P<p>{_PATH})', re.I), Op.READ),
    # Shell
    (re.compile(rf"\brm\s+(?:-[rf]+\s+)*(?P<p>{_PATH})"), Op.DELETE),
    (re.compile(rf"\bgit\s+rm\s+(?:-[rf]+\s+)*(?P<p>{_PATH})"), Op.DELETE),
    (re.compile(rf"\b(?:cat|head|tail|less)\s+(?P<p>{_PATH})"), Op.READ),
    (re.compile(rf"\btouch\s+(?P<p>{_PATH})"), Op.CREATE),
    # Narration — how a model reports what it did.
    (re.compile(rf"\b(?:created|wrote|added|generated)\s+(?P<p>{_PATH})", re.I), Op.CREATE),
    (
        re.compile(rf"\b(?:modified|edited|updated|patched|changed)\s+(?P<p>{_PATH})", re.I),
        Op.MODIFY,
    ),
    (re.compile(rf"\b(?:deleted|removed|dropped|unlinked)\s+(?P<p>{_PATH})", re.I), Op.DELETE),
    (re.compile(rf"\b(?:read|opened|examined|inspected|viewed)\s+(?P<p>{_PATH})", re.I), Op.READ),
    # Diff headers — the most reliable modify signal there is.
    (re.compile(rf"^\+\+\+ b/(?P<p>{_PATH})", re.M), Op.MODIFY),
    (re.compile(rf"^diff --git a/{_PATH} b/(?P<p>{_PATH})", re.M), Op.MODIFY),
)


@dataclass
class Ledger:
    """Final state per path, as folded from a sequence of turns."""

    state: dict[str, Op] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.state)

    def apply(self, path: str, op: Op) -> None:
        """Later turns overwrite earlier ones; within a turn the strongest op wins."""
        prev = self.state.get(path)
        if prev is None or _STRENGTH[op] >= _STRENGTH[prev]:
            self.state[path] = op

    def apply_turn(self, path: str, op: Op) -> None:
        """Unconditional overwrite — used when folding across turn boundaries."""
        self.state[path] = op


def extract_ops(text: str) -> list[tuple[str, Op]]:
    """Every (path, op) asserted by one block of text, in no particular order."""
    if not text:
        return []
    found: list[tuple[str, Op]] = []
    for pattern, op in _PATTERNS:
        for m in pattern.finditer(text):
            path = _canonical(m.group("p"))
            if path:
                found.append((path, op))
    return found


def _canonical(path: str) -> str:
    """Normalise so ``./src/a.py``, ``src/a.py`` and ``/src/a.py`` are one artifact.

    Without this the ledger double-counts the same file under spelling variants and
    reports phantom state changes that never happened.
    """
    p = path.strip().strip("\"'`,;:")
    for prefix in ("./", "~/"):
        if p.startswith(prefix):
            p = p[len(prefix) :]
    p = p.lstrip("/")
    return p


def build_ledger(turns: Iterable[str]) -> Ledger:
    """Fold a trajectory's turns into a final file-state ledger.

    Within a turn the strongest assertion wins (so a delete is not masked by a read on
    the same line); across turns, later simply replaces earlier — that is what "final
    state" means.
    """
    ledger = Ledger()
    for text in turns:
        per_turn: dict[str, Op] = {}
        for path, op in extract_ops(text):
            prev = per_turn.get(path)
            if prev is None or _STRENGTH[op] >= _STRENGTH[prev]:
                per_turn[path] = op
        for path, op in per_turn.items():
            ledger.apply_turn(path, op)
    return ledger


@dataclass
class StateProbe:
    """Artifact-state outcome for one trajectory.

    ``exact + stale + lost == total`` always; the three are mutually exclusive.
    """

    total: int = 0
    exact: int = 0
    stale: int = 0
    lost: int = 0

    @property
    def fidelity(self) -> float:
        """Fraction of artifacts whose final state survived intact."""
        return self.exact / self.total if self.total else 1.0

    @property
    def stale_rate(self) -> float:
        """Fraction carrying a WRONG final state — the actively-misleading class."""
        return self.stale / self.total if self.total else 0.0

    @property
    def silent_failure_share(self) -> float:
        """Of everything that went wrong, how much fails silently?

        1.0 means every failure is a confident falsehood; 0.0 means every failure is
        an honest gap. This is the number to watch when tuning a compressor: driving
        total error down while pushing this up is a bad trade.
        """
        wrong = self.stale + self.lost
        return self.stale / wrong if wrong else 0.0

    def add(self, other: StateProbe) -> None:
        self.total += other.total
        self.exact += other.exact
        self.stale += other.stale
        self.lost += other.lost

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "exact": self.exact,
            "stale": self.stale,
            "lost": self.lost,
            "fidelity": round(self.fidelity, 4),
            "stale_rate": round(self.stale_rate, 4),
            "silent_failure_share": round(self.silent_failure_share, 4),
        }


def score(original_turns: Iterable[str], compressed_turns: Iterable[str]) -> StateProbe:
    """Grade a compressed trajectory's artifact state against the original's.

    The ground truth is the original's own ledger — not an external answer key — so
    this works on any trajectory without annotation, which is what makes it runnable
    on live traffic as well as the corpus.
    """
    truth = build_ledger(original_turns)
    seen = build_ledger(compressed_turns)
    probe = StateProbe(total=len(truth))
    for path, op in truth.state.items():
        got = seen.state.get(path)
        if got is None:
            probe.lost += 1
        elif got == op:
            probe.exact += 1
        else:
            probe.stale += 1
    return probe


def score_texts(original: str, compressed: str) -> StateProbe:
    """Single-block convenience wrapper — the live-proxy entry point."""
    return score([original], [compressed])


def format_probe(probe: StateProbe) -> str:
    """One-screen summary. Leads with stale, because that is the number that hurts."""
    if not probe.total:
        return "artifact-state: no file operations found — nothing to grade."
    lines = [
        f"artifact-state fidelity   {probe.fidelity:6.1%}  ({probe.exact}/{probe.total} artifacts intact)",
        f"  stale (wrong state)     {probe.stale_rate:6.1%}  ({probe.stale})  <- agent acts on a false belief",
        f"  lost  (absent)          {probe.lost / probe.total:6.1%}  ({probe.lost})  <- agent knows it does not know",
    ]
    if probe.stale + probe.lost:
        lines.append(
            f"  silent-failure share    {probe.silent_failure_share:6.1%}"
            "  <- of all errors, the share that fail confidently"
        )
    return "\n".join(lines)


def _iter_tool_texts(messages: Any) -> list[str]:
    """Tool calls AND results from a provider payload, oldest first.

    Both halves matter: the call says what was attempted, the result says whether it
    happened. Reading only results misses a delete whose output was empty.
    """
    out: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            out.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                # the call: name plus its arguments, rendered so _PATTERNS can see it
                name = str(block.get("name", ""))
                args = block.get("input")
                if isinstance(args, dict):
                    rendered = ", ".join(
                        f'{k}="{v}"' for k, v in args.items() if isinstance(v, (str, int))
                    )
                    out.append(f"{name}({rendered})")
            elif btype in ("tool_result", "text"):
                out.append(
                    _flatten(block.get("content") if btype == "tool_result" else block.get("text"))
                )
    return [t for t in out if t]


def _flatten(node: Any) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if node.get("type") == "text":
            return str(node.get("text", ""))
        return "\n".join(_flatten(v) for v in node.values())
    if isinstance(node, list):
        return "\n".join(_flatten(item) for item in node)
    return ""


def measure_live(original: Any, compressed: Any) -> StateProbe:
    """Score one real request. Returns counts only — never a path."""
    return score(_iter_tool_texts(original), _iter_tool_texts(compressed))
