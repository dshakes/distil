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
# would report phantom churn.
#
# Three shapes, because requiring an extension made the most-edited files in a real
# repository invisible to the gate — `Dockerfile`, `Makefile`, `LICENSE`, `.env`,
# `.gitignore` all matched nothing:
#   1. anything with an extension            src/app.py, a/b/c.tar.gz
#   2. a dotfile                             .env, .gitignore
#   3. a Capitalised extensionless filename   Dockerfile, Makefile, LICENSE, README
# Shape 3 is capitalised on purpose: lowercase bare words would swallow ordinary prose
# ("created something"), and a ledger full of English is worse than one missing a file.
# `(?:\.ext)+` — one or more extension segments, so multi-dot names are ONE artifact.
# With a single segment, `a/b/c.tar.gz` extracted as `a/b/c.tar` and `x.test.tsx` as
# `x.test`, which both truncates real filenames and can merge two distinct files into
# one ledger entry. Archives and `*.test.tsx` / `*.spec.ts` are common enough that the
# comment above already claimed this worked.
_WITH_EXT = r"(?:[\w.-]+/)*[\w-]+(?:\.[A-Za-z][\w]{0,9})+"
_DOTFILE = r"(?:[\w.-]+/)*\.[A-Za-z][\w.-]{1,20}"
# `(?-i:...)` is load-bearing. Several patterns below compile with re.I, and under
# IGNORECASE the class `[A-Z]` matches lowercase too — which turned every bare word
# into an artifact: "created something and removed nothing" produced two phantom file
# operations. The scoped flag pins case-sensitivity to this class regardless of how
# the enclosing pattern is compiled.
_BARE_CAP = r"(?:[\w.-]+/)*(?-i:[A-Z][A-Za-z]{2,19})"
_PATH = rf"(?:(?:~|\.{{1,2}})?/)?(?:{_WITH_EXT}|{_DOTFILE}|{_BARE_CAP})"
# `_PATH` without the bare-capitalised-word form, for contexts where the surrounding
# syntax is too weak to tell a path from an ordinary word. `>` is the case that forced
# this: it is a shell redirect, but it is also the comparison operator, and prose is
# full of `Rate > Burst`. Allowing `_BARE_CAP` after it put 19 phantom artifacts
# ("Accept", "Error", "The") into a 7-artifact corpus — every one of them scored as a
# perfectly preserved file, which is the silent-and-green failure this module exists to
# detect. A redirect target is a real filename; `Makefile > x` is not a thing.
_PATH_FILE = rf"(?:(?:~|\.{{1,2}})?/)?(?:{_WITH_EXT}|{_DOTFILE})"
# An optional opening quote before a path. Agents quote paths in shell commands as a
# matter of habit — `rm "src/app.py"`, `cat 'config.yaml'` — and without this every one
# of those operations matched nothing and was silently absent from the ledger. The
# closing quote needs no rule: it is not a path character, so the path class stops
# there on its own, and `_canonical` strips any that survives.
_QP = r"[\"\']?"

# Markers that a tool call did NOT do what it said. Without these the ledger records
# ATTEMPTS rather than state: `rm missing.py` followed by `exit: 1` was scored as a
# successful DELETE, so a compressor could be penalised for dropping a deletion that
# never happened — or credited for preserving one.
_FAILED_RE = re.compile(
    r'"exit"\s*:\s*[1-9]'
    r"|\bexit(?:\s+code)?[ =:]+[1-9]"
    r"|\bNo such file or directory\b"
    r"|\bPermission denied\b"
    r'|"ok"\s*:\s*false'
    r"|\b(?:command|operation) failed\b",
    re.I,
)

# Verb -> Op. Matched against the text of tool calls and their narration. Each entry is
# (regex, op); the path is group "p".
_PATTERNS: tuple[tuple[re.Pattern[str], Op], ...] = (
    # Structured agent tool calls, e.g. Write(file_path="src/a.py")
    (re.compile(rf'\bWrite\s*\(\s*file_path\s*=\s*["\']?(?P<p>{_PATH})', re.I), Op.CREATE),
    (re.compile(rf'\bEdit\s*\(\s*file_path\s*=\s*["\']?(?P<p>{_PATH})', re.I), Op.MODIFY),
    (re.compile(rf'\bRead\s*\(\s*file_path\s*=\s*["\']?(?P<p>{_PATH})', re.I), Op.READ),
    # Shell
    (re.compile(rf"\brm\s+(?:-[rf]+\s+)*{_QP}(?P<p>{_PATH})"), Op.DELETE),
    (re.compile(rf"\bgit\s+rm\s+(?:-[rf]+\s+)*{_QP}(?P<p>{_PATH})"), Op.DELETE),
    (re.compile(rf"\b(?:cat|head|tail|less)\s+{_QP}(?P<p>{_PATH})"), Op.READ),
    (re.compile(rf"\btouch\s+{_QP}(?P<p>{_PATH})"), Op.CREATE),
    # Narration — how a model reports what it did.
    (re.compile(rf"\b(?:created|wrote|added|generated)\s+{_QP}(?P<p>{_PATH})", re.I), Op.CREATE),
    (
        re.compile(rf"\b(?:modified|edited|updated|patched|changed)\s+{_QP}(?P<p>{_PATH})", re.I),
        Op.MODIFY,
    ),
    (re.compile(rf"\b(?:deleted|removed|dropped|unlinked)\s+{_QP}(?P<p>{_PATH})", re.I), Op.DELETE),
    (
        re.compile(rf"\b(?:read|opened|examined|inspected|viewed)\s+{_QP}(?P<p>{_PATH})", re.I),
        Op.READ,
    ),
    # Diff headers — the most reliable modify signal there is.
    (re.compile(rf"^\+\+\+ b/{_QP}(?P<p>{_PATH})", re.M), Op.MODIFY),
    (re.compile(rf"^diff --git a/{_PATH} b/{_QP}(?P<p>{_PATH})", re.M), Op.MODIFY),
    # `apply_patch` envelopes — the edit format used by Codex and several agent
    # harnesses. Absent these, an entire agent family's file operations were invisible
    # and the probe scored its trajectories as a workspace it had never touched.
    (re.compile(rf"^\*\*\*\s*Add File:\s*{_QP}(?P<p>{_PATH})", re.M | re.I), Op.CREATE),
    (re.compile(rf"^\*\*\*\s*Delete File:\s*{_QP}(?P<p>{_PATH})", re.M | re.I), Op.DELETE),
    (re.compile(rf"^\*\*\*\s*Update File:\s*{_QP}(?P<p>{_PATH})", re.M | re.I), Op.MODIFY),
    # In-place edit and shell redirects. `>` is the hardest character in this file: it
    # is a redirect, the comparison operator, a markdown quote marker, an HTML tag
    # close, and a CSS child combinator, and agent transcripts contain all five. Each
    # guard here bought back a phantom artifact measured on the real corpus:
    #
    #   `[^\s>-]`   a redirect has a command before it on the line; a blockquote does
    #               not (`> Some note` created a file named "Some"). Excluding `-`
    #               keeps a type hint's `->` from redirecting into its return type;
    #               excluding `>` keeps `>>` out of the single-redirect rule.
    #   `\s+`       CSS `>.c-0` and HTML `>Rate limits` have no space after the
    #               operator; shell redirects conventionally do.
    #   `_PATH_FILE` a redirect target is a filename, not a bare word — this alone
    #               removed 18 phantoms ("Accept", "Error", "The") from prose
    #               comparisons like `Rate > Burst`.
    #
    # Together these are clean on the corpus. The deliberate cost is `cmd >out.txt`
    # with no space, which goes unrecorded: a missed operation is a visible gap, and a
    # phantom one is a silent wrong state. This module exists because those are not
    # the same failure.
    (re.compile(rf"\bsed\s+-i\S*\s+(?:'[^']*'\s+|\"[^\"]*\"\s+)?{_QP}(?P<p>{_PATH})"), Op.MODIFY),
    (re.compile(rf"[^\s>-]\s*>>\s+{_QP}(?P<p>{_PATH_FILE})", re.M), Op.MODIFY),
    (re.compile(rf"[^\s>-]\s*>(?!>)\s+{_QP}(?P<p>{_PATH_FILE})", re.M), Op.CREATE),
)

# Verbs that name TWO paths and assign each a different op. These cannot live in
# `_PATTERNS`, which yields one path per match: `mv old.py new.py` is a delete *and* a
# create, and scoring it as either alone leaves the ledger describing a workspace that
# never existed. `mv` also matches inside `git mv`, so that needs no separate entry.
_PAIR_PATTERNS: tuple[tuple[re.Pattern[str], Op, Op], ...] = (
    (
        re.compile(rf"\bmv\s+(?:-\S+\s+)*{_QP}(?P<s>{_PATH}){_QP}\s+{_QP}(?P<d>{_PATH})"),
        Op.DELETE,
        Op.CREATE,
    ),
    # `cp` leaves the source in place, so the source is only ever a read.
    (
        re.compile(rf"\bcp\s+(?:-\S+\s+)*{_QP}(?P<s>{_PATH}){_QP}\s+{_QP}(?P<d>{_PATH})"),
        Op.READ,
        Op.CREATE,
    ),
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
    """Every (path, op) *successfully* asserted by one block of text.

    A tool call that failed asserts nothing about workspace state, so text carrying a
    failure marker is skipped rather than parsed: the ledger records what happened,
    not what was attempted.

    That inference only applies to UNSTRUCTURED text, where success and failure are
    tangled in the same prose ("rm failed: No such file"). A structured call whose
    result `_iter_tool_texts` has already adjudicated arrives pre-marked, and is not
    re-scanned. Re-deriving a verdict that was already computed correctly is what
    caused two separate false-skip bugs — see :data:`_ADJUDICATED`.

    Adjudication is PER OPERATION, not per block. A coding-agent block routinely
    carries several tool calls, and rejecting the whole block on one failure marker
    dropped every successful operation beside the failed one: a block creating `a.py`
    and failing to create `b.py` recorded neither, so `a.py` was never graded and a
    compressor could lose it for free. Each operation is judged on the text from
    where it is asserted up to where the next one is — the span that actually reports
    its outcome.
    """
    scan = not text.startswith(_ADJUDICATED)
    if not scan:
        text = text[len(_ADJUDICATED) :]
    if not text:
        return []
    found: list[tuple[int, str, Op]] = []
    for pattern, op in _PATTERNS:
        for m in pattern.finditer(text):
            path = _canonical(m.group("p"))
            if path:
                found.append((m.start(), path, op))
    # Two-path verbs: the source and destination take DIFFERENT ops from one match.
    for pattern, src_op, dst_op in _PAIR_PATTERNS:
        for m in pattern.finditer(text):
            for group, op in (("s", src_op), ("d", dst_op)):
                path = _canonical(m.group(group))
                if path:
                    found.append((m.start(), path, op))
    if not scan:
        return [(path, op) for _, path, op in found]
    found.sort(key=lambda f: f[0])
    kept: list[tuple[str, Op]] = []
    for i, (start, path, op) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(text)
        if _FAILED_RE.search(text, start, end):
            continue
        kept.append((path, op))
    return kept


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


# Sentinels marking where a call/result sat in the document, so the pair can be
# joined without losing its position. Chosen to be unmatchable by any real content.
_CALL_SLOT = "\x00call\x00"
_RESULT_SLOT = "\x00result\x00"
# Marks a tool call whose result has ALREADY been checked for failure, against the
# result alone — the only text that is evidence about success.
#
# The verdict has to cross this boundary as a flag, because re-deriving it from the
# emitted string is not possible without guessing, and both guesses were wrong:
#
#   * joining the pair as `call -> result` and scanning the whole thing skipped a
#     successful `Write(file_path="t.py", content="No such file or directory")` — an
#     agent writing an error fixture vanished from the ledger;
#   * scanning only after the last `->` fixed that but made the delimiter ambiguous,
#     because written Python source contains `->` in every type hint. A successful
#     write of `def run() -> int: ... sys.exit(1)` had its own body read as a failed
#     outcome and was skipped too.
#
# Both are the same mistake: a verdict computed correctly, thrown away, and recovered
# by string heuristic from text that legitimately contains the delimiter. Control
# characters cannot occur in tool arguments or source, so the flag is unambiguous.
_ADJUDICATED = "\x00ok\x00"


def _iter_tool_texts(messages: Any) -> list[str]:
    """Tool calls AND results from a provider payload, oldest first.

    Both halves matter, and they must be JOINED: the call says what was attempted,
    the result says whether it happened. Reading only results misses a delete whose
    output was empty; reading them as independent strings records every failed
    attempt as a completed state change.
    """
    out: list[str] = []
    calls: dict[str, str] = {}
    results: dict[str, str] = {}
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        # A `role:"tool"` message IS a result and is consumed at its call's position
        # below. Emitting its string content here as ordinary text too parsed it
        # TWICE: a `Write(file_path="a.py")` whose result text read "deleted a.py"
        # produced both the joined pair AND a standalone "deleted a.py", folding the
        # ledger to DELETE when the operation was a write. Order and pairing were
        # both already fixed; this was the third way the same text could be
        # mis-counted.
        # Only the ASSISTANT's own narration reports what happened. A user message
        # ("Please inspect: cat src/app.py") is a REQUEST, not an observation — but
        # it was parsed as a completed read, so compressing the request away
        # reported a lost artifact when no workspace state had ever changed. System
        # prompts have the same problem: they routinely name files as instructions.
        if (
            isinstance(content, str)
            and content
            and msg.get("role") not in ("tool", "user", "system")
        ):
            out.append(content)
        # --- OpenAI Chat Completions ------------------------------------------
        # `assistant.tool_calls` sits BESIDE `content`, not inside it, so a purely
        # structured OpenAI turn produced no text at all and the probe reported
        # total=0, fidelity=1.0 — a perfect score on a workspace it never looked at.
        # That is the worst possible failure for this metric: silent, and green.
        # `distil.retention` already normalises OpenAI shapes; this now matches.
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            name = str(fn.get("name", ""))
            calls[str(call.get("id", ""))] = f"{name}({_args_text(fn.get('arguments'))})"
            out.append(_CALL_SLOT + str(call.get("id", "")))
        if msg.get("role") == "tool":
            rid = str(msg.get("tool_call_id", ""))
            results[rid] = content if isinstance(content, str) else _flatten(content)
            out.append(_RESULT_SLOT + rid)
            continue

        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            # --- OpenAI Responses API ------------------------------------------
            if btype == "function_call":
                cid = str(block.get("call_id", block.get("id", "")))
                calls[cid] = f"{block.get('name', '')}({_args_text(block.get('arguments'))})"
                out.append(_CALL_SLOT + cid)
                continue
            if btype == "function_call_output":
                cid = str(block.get("call_id", ""))
                results[cid] = _flatten(block.get("output"))
                out.append(_RESULT_SLOT + cid)
                continue
            if btype == "tool_use":
                # the call: name plus its arguments, rendered so _PATTERNS can see it
                name = str(block.get("name", ""))
                args = block.get("input")
                if isinstance(args, dict):
                    rendered = ", ".join(
                        f'{k}="{v}"' for k, v in args.items() if isinstance(v, (str, int))
                    )
                    # Keyed by tool_use_id so the RESULT can be joined to this CALL
                    # below. Emitting them as two independent strings meant the
                    # failure check never saw them together: a failed
                    # `Write(file_path="a.py")` whose "Permission denied" arrived in a
                    # separate tool_result block was still recorded as a successful
                    # create, which is the "ledger of attempts" bug all over again in
                    # the live path.
                    # Placeholder holds the CALL'S POSITION in the document; the
                    # result is spliced in below. Buffering calls and appending them
                    # after the text ran the ledger out of order: `Write(a.py)` then
                    # a later `deleted a.py` folded as delete-then-create, so the
                    # final state read EXISTS when the transcript ends with it
                    # deleted. A ledger of final state that ignores order is not a
                    # ledger of final state.
                    calls[str(block.get("id", ""))] = f"{name}({rendered})"
                    out.append(_CALL_SLOT + str(block.get("id", "")))
            elif btype == "tool_result":
                results[str(block.get("tool_use_id", ""))] = _flatten(block.get("content"))
                out.append(_RESULT_SLOT + str(block.get("tool_use_id", "")))
            elif btype == "text":
                # Same rule as above: only the assistant narrates outcomes.
                if msg.get("role") not in ("user", "system"):
                    out.append(_flatten(block.get("text")))

    resolved: list[str] = []
    for item in out:
        if item.startswith(_CALL_SLOT):
            call_id = item[len(_CALL_SLOT) :]
            # Call joined to its outcome so `extract_ops` can reject the pair when the
            # outcome says it failed. Emitted at the CALL's position, which is where
            # the operation happened.
            result = results.get(call_id, "")
            call_text = calls.get(call_id, "")
            # The CALL states which operation was attempted; the RESULT states only
            # whether it succeeded. So the failure check runs HERE, against the result
            # alone, and the result text is then dropped: joining them let result
            # narration override the call, and because the strongest op wins within a
            # block, a Write whose result mentioned "deleted a.py" folded the ledger
            # to DELETE.
            #
            # A failed call asserts nothing about workspace state, so it is dropped
            # outright — identical to what `extract_ops` did with it, minus a string
            # round-trip that could not survive contact with real content.
            if not call_text:
                continue
            if result and _FAILED_RE.search(result):
                continue
            resolved.append(_ADJUDICATED + call_text)
        elif item.startswith(_RESULT_SLOT):
            # Already consumed at its call's position; emit only if orphaned, since an
            # orphan result still describes state ("deleted old/x.json").
            rid = item[len(_RESULT_SLOT) :]
            if rid not in calls:
                resolved.append(results.get(rid, ""))
        else:
            resolved.append(item)
    return [t for t in resolved if t]


def _args_text(arguments: Any) -> str:
    """Render a tool call's arguments so `_PATTERNS` can see the paths in them.

    OpenAI passes them as a JSON *string*; Anthropic as a dict. Both end up in the
    same `k="v"` shape the extraction patterns already match.
    """
    if isinstance(arguments, str):
        import json as _json

        try:
            arguments = _json.loads(arguments)
        except Exception:
            return arguments
    if isinstance(arguments, dict):
        return ", ".join(f'{k}="{v}"' for k, v in arguments.items() if isinstance(v, (str, int)))
    return ""


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
