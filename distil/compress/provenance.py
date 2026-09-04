"""Which tool produced this result — and must the agent be able to quote it back byte-exact?

1.49.0 established the rule: content the agent has to reproduce character-for-character in
an ``Edit(old_string=...)`` is exempt from the digest at any age, because recency is
positional and an edit is not. It keyed that exemption on the tool's *name*
(``Read``/``Grep``/``view``/``str_replace_editor`` and friends).

On real traffic that misses the majority of the reads. Measured over 2,489 local Claude
Code sessions (52,937 tool results, 19.4M tool-result tokens): **33.6% of tool-result mass
is a whole-file read issued through the generic shell tool** — ``cat``, ``head``, ``tail``,
``nl``, ``sed -n '10,40p'`` — which the name-keyed rule digests. 629 of the 726 residual
"quote has no surviving byte-exact copy" failures had a shell source.

So provenance is read from the *command*, not just the tool name. A shell call whose every
stage is a whole-file reader, with no pipe and no redirection, produced file bytes the agent
may have to quote back — same guarantee, same exemption.

The classifier is deliberately narrow. It answers one question: *are the bytes the agent saw
in this tool result a verbatim slice of a file on disk?*

* pipe (``cat f | grep x``) — no. The result is the pipeline's output, not the file's.
* redirection (``cat f > out``, ``… | tee``) — no. The bytes went to a file, not the agent.
* ``sed`` without ``-n`` + a pure print range — no. That is a transform, not a read.
* ``cat a b``, ``head -n 50 f``, ``cat -n f``, ``sed -n '1,20p' f`` — yes. The agent quotes
  from what it saw, and what it saw is file content.

Sequences (``a && b``, ``a; b``) concatenate their stages onto the same stream, so *every*
stage must be a whole-file read for the result to qualify.

Cost control: keeping every whole-file shell read verbatim forever forfeits ~27pp of
tool-result mass. Keeping only the **latest** read per path recovers most of that (~7.7pp)
and still cuts the residual quote hazard from 39.3% to 16.2%, because a superseded read is
one the agent has a fresher copy of. Supersession is gated on the client's own cache
breakpoint (see :func:`exact_quote_ids`) — demoting a block the provider has already cached
rewrites the prefix, which costs more than the digest saves.

Stateless by design, like the rule it replaces: the exemption is recomputed from the message
history on every request, and latest-per-path is resolved inside that same walk. The
conversation *is* the session memory.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any, Iterable, NamedTuple, Sequence

__all__ = [
    "EXACT_QUOTE_TOOLS",
    "ToolCall",
    "command_text",
    "edit_quotes",
    "exact_quote_ids",
    "observed_view",
    "quote_hazard",
    "required_quotes",
    "whole_file_read_paths",
]

# Tools whose result the agent must be able to quote back BYTE-EXACT, by name.
# Lowercased for comparison; covers Claude Code, Codex/OpenAI and common MCP
# filesystem servers.
EXACT_QUOTE_TOOLS = frozenset(
    {
        "read",
        "read_file",
        "readfile",
        "view",
        "open",
        "cat",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "grep",
        "glob",
        "search_files",
        "notebookread",
    }
)

# Tools that edit by literal match. Their `old_string` is the quote that has to survive.
_EDIT_TOOLS = frozenset(
    {
        "edit",
        "multiedit",
        "edit_file",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "apply_patch",
    }
)

# Commands whose stdout is a verbatim slice of the files named on their command line.
_READERS = frozenset({"cat", "head", "tail", "nl", "less", "more", "bat"})

# Flags that consume the following token, so it is a value and not a path.
_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "head": frozenset({"-n", "-c", "--lines", "--bytes"}),
    "tail": frozenset({"-n", "-c", "--lines", "--bytes"}),
    "nl": frozenset({"-b", "-d", "-f", "-h", "-i", "-l", "-n", "-s", "-v", "-w"}),
    "bat": frozenset({"-r", "--line-range", "--language", "-l", "--style"}),
    "cat": frozenset(),
    "less": frozenset(),
    "more": frozenset(),
}

# A `sed` script that only prints line ranges: `5p`, `1,40p`, `10,$p`, `3,+5p`.
_SED_PRINT = re.compile(r"^[0-9$,+~]+p$")
_SED_VALUE_FLAGS = frozenset({"-e", "-f", "--expression", "--file"})
_SED_QUIET = frozenset({"-n", "--quiet", "--silent"})

# Where a shell tool puts its command line, across the tool schemas we see.
_COMMAND_KEYS = ("command", "cmd", "shell_command", "script")


class ToolCall(NamedTuple):
    """One tool invocation, normalised across the three request shapes.

    ``pos`` is the position of the *call* in the message/item list (named ``pos`` and not
    ``index``, which a NamedTuple inherits from tuple). The result it answers always lands
    after it, so comparing the call's position to the client's cache breakpoint is the
    conservative test for "is this block already committed?".
    """

    id: str
    name: str
    command: str
    pos: int


def command_text(inp: Any) -> str:
    """The shell command line inside a tool_use ``input``, or ``""`` if there is none."""
    if not isinstance(inp, dict):
        return ""
    for key in _COMMAND_KEYS:
        val = inp.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            return " ".join(val)
    return ""


def _stage_paths(stage: str) -> list[str]:
    """File paths a single command stage reads out verbatim; empty if it is not a read."""
    try:
        tokens = shlex.split(stage)
    except ValueError:
        return []  # unbalanced quotes — we cannot say what this reads, so we do not claim it
    if not tokens:
        return []
    cmd = tokens[0].rsplit("/", 1)[-1]
    if cmd == "sed":
        return _sed_paths(tokens[1:])
    if cmd not in _READERS:
        return []
    value_flags = _VALUE_FLAGS.get(cmd, frozenset())
    paths: list[str] = []
    skip = False
    end_of_flags = False
    for tok in tokens[1:]:
        if skip:
            skip = False
            continue
        if tok == "--" and not end_of_flags:
            end_of_flags = True  # everything after this is a path, dashes and all
            continue
        if not end_of_flags and tok.startswith("-") and tok != "-":
            skip = tok in value_flags
            continue
        paths.append(tok)
    return paths


def _sed_paths(args: list[str]) -> list[str]:
    """Paths for a ``sed`` call, but only when it is a pure ``-n '<range>p'`` printer."""
    quiet = False
    script: str | None = None
    paths: list[str] = []
    skip = False
    for tok in args:
        if skip:
            skip = False
            continue
        if tok.startswith("-") and tok != "-":
            if tok in _SED_QUIET or (re.fullmatch(r"-[a-zA-Z]+", tok) and "n" in tok):
                quiet = True
            if tok in _SED_VALUE_FLAGS:
                skip = True
            continue
        if script is None:
            script = tok
        else:
            paths.append(tok)
    if not quiet or script is None or not _SED_PRINT.match(script):
        return []
    return paths


def whole_file_read_paths(command: str) -> tuple[str, ...]:
    """Paths whose verbatim bytes *command* puts on stdout; empty when it is not such a read.

    Empty is the safe answer: it only means the result stays eligible for the digest.
    """
    if not command:
        return ()
    # `||` and `&&` are sequences, not pipes — normalise them away first so the pipe
    # test below cannot be fooled by them.
    norm = command.replace("||", ";").replace("&&", ";").replace("\n", ";")
    if "|" in norm:
        return ()  # a real pipe: the agent saw the pipeline's output, not the file's
    if ">" in norm:
        return ()  # redirection (`>`, `>>`): the bytes went to a file, not to the agent
    paths: list[str] = []
    stages = [s for s in norm.split(";") if s.strip()]
    if not stages:
        return ()
    for stage in stages:
        stage_paths = _stage_paths(stage)
        if not stage_paths:
            return ()  # one non-read stage and the combined output is no longer file bytes
        paths.extend(stage_paths)
    return tuple(dict.fromkeys(paths))


def exact_quote_ids(
    calls: Iterable[ToolCall],
    *,
    cached_through: int | None = None,
    widen: bool = False,
) -> set[str]:
    """Ids of the tool calls whose results must stay byte-exact, however old.

    Name-keyed reads (``Read``, ``Grep``, …) are always exempt — that is the 1.49.0 rule,
    unchanged. Shell whole-file reads are exempt too, except where a *later* read of every
    path they touched has superseded them: the agent has a fresher byte-exact copy, so the
    stale one may digest.

    ``cached_through`` is the client's cache breakpoint (``compress.recency.cached_prefix_end``,
    ``None`` when the client marks nothing). A superseded read at or before it is **kept
    anyway**: flipping a block the provider has already cached from verbatim to digest
    rewrites the whole prefix at the 1.25x write rate, which is measurably worse than the
    tokens the digest would save. Same anchor, same reason as the recency carve-out.

    ``widen`` disables supersession entirely — the reaction to an observed quote miss.
    """
    keep: set[str] = set()
    reads: list[tuple[ToolCall, tuple[str, ...]]] = []
    last_reader: dict[str, int] = {}
    for call in calls:
        if call.name.lower() in EXACT_QUOTE_TOOLS:
            keep.add(call.id)
            continue
        paths = whole_file_read_paths(call.command)
        if not paths:
            continue
        for path in paths:
            last_reader[path] = len(reads)
        reads.append((call, paths))
    for pos, (call, paths) in enumerate(reads):
        superseded = all(last_reader[p] != pos for p in paths)
        committed = cached_through is not None and call.pos <= cached_through
        if superseded and not committed and not widen:
            continue
        keep.add(call.id)
    return keep


def required_quotes(inputs: Iterable[tuple[str, Any]]) -> list[str]:
    """Every ``old_string`` a literal-match edit will have to find in the forwarded view.

    Takes ``(tool_name, tool_input)`` pairs. MultiEdit nests its quotes under ``edits``.
    """
    quotes: list[str] = []
    for name, inp in inputs:
        if name.lower() not in _EDIT_TOOLS or not isinstance(inp, dict):
            continue
        for key in ("old_string", "old_str"):
            val = inp.get(key)
            if isinstance(val, str) and val:
                quotes.append(val)
        edits = inp.get("edits")
        if isinstance(edits, list):
            for edit in edits:
                if not isinstance(edit, dict):
                    continue
                for key in ("old_string", "old_str"):
                    val = edit.get(key)
                    if isinstance(val, str) and val:
                        quotes.append(val)
    return quotes


def observed_view(messages: Iterable[Any]) -> str:
    """The forwarded payload as JSON, with the model's own turns removed.

    This is the population a quote has to survive in. Assistant turns are excluded for a
    specific reason: the ``Edit`` block *itself* carries ``old_string`` verbatim, so a view
    that included it would report every quote as surviving no matter how thoroughly the
    source read had been digested — a check that can only pass is not a check.
    """
    return json.dumps(
        [m for m in messages if not (isinstance(m, dict) and m.get("role") == "assistant")],
        default=str,
    )


def edit_quotes(messages: Iterable[Any]) -> list[str]:
    """:func:`required_quotes` over every ``tool_use`` block of a Messages-shaped history."""
    pairs: list[tuple[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                pairs.append((str(blk.get("name", "")), blk.get("input")))
    return required_quotes(pairs)


def quote_hazard(quotes: Sequence[str], view: str) -> tuple[int, int]:
    """``(survived, lost)`` — how many required quotes still occur verbatim in *view*.

    *view* is the serialised forwarded payload (see :func:`observed_view`), so the quotes
    are compared in their JSON-escaped form. Content-free: only the two counts leave here.
    """
    survived = lost = 0
    for quote in quotes:
        needle = json.dumps(quote)[1:-1]
        if needle and needle in view:
            survived += 1
        else:
            lost += 1
    return survived, lost
