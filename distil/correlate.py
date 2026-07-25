"""Correlate a wrap session with the agent's own conversation transcript.

Everything here is adapter-independent: it consumes the normalized
:class:`~distil.transcripts.base.Transcript` model, so a new agent gets all
of it by writing one adapter. The join is content-derived, not guessed:
digested blocks are matched into transcript tool results via the restore
store's original bytes.

Opt-in and read-only: transcripts are the agent's files, read at render time;
nothing is copied into distil's state. This is the one dissect feature that
crosses the content-free line — reports name tools, files and prompts — which
is why it never runs unless asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .transcripts import Transcript

if TYPE_CHECKING:  # pragma: no cover — import cycle guard (dissect imports us)
    from .dissect import Dissection

_MIN_MATCH = 40  # chars; below this substring matches are coincidence


@dataclass
class FoldSource:
    """A digested block, named by where it came from in the conversation."""

    handle: str
    sig: str
    tokens: int
    folds: int
    tool: str  # producing tool ("" when unattributed)
    turn: int  # human turn it happened under (0 = before the first)
    turn_text: str
    refetches: int  # distinct transcript results carrying this content


@dataclass
class TurnCost:
    index: int
    text: str
    requests: int
    baseline_tokens: int
    saved_tokens: int


@dataclass
class Correlation:
    agent: str
    label: str
    path: str
    fold_sources: list[FoldSource] = field(default_factory=list)
    unnamed_blocks: int = 0
    tools_defined: int = 0
    tools_invoked: int = 0
    unused_tools: list[tuple[str, int]] = field(default_factory=list)  # (name, tok/req)
    refetched: list[FoldSource] = field(default_factory=list)
    turns: list[TurnCost] = field(default_factory=list)  # costliest first

    @property
    def unused_tokens_per_request(self) -> int:
        return sum(t for _n, t in self.unused_tools)


def _contains(haystack: str, needle: str) -> bool:
    if len(needle) < _MIN_MATCH:
        return haystack == needle
    return needle in haystack or haystack in needle


def correlate(d: Dissection, tr: Transcript) -> Correlation:
    """Join one Dissection with one Transcript. Best-effort throughout:
    expired restore blobs and unmatched blocks degrade to counts, never errors."""
    from .dissect import _state_dir

    corr = Correlation(agent=tr.agent, label=tr.label, path=str(tr.path))
    turn_text = {t.index: t.text for t in tr.turns}

    # --- fold -> source, by content (restore blob matched into tool results)
    restore = _state_dir() / "restore"
    for handle, info in d.blocks.items():
        try:
            original = (restore / handle).read_text(encoding="utf-8").strip()
        except OSError:
            corr.unnamed_blocks += 1
            continue
        matches = [r for r in tr.tool_results if _contains(r.text, original)]
        if not matches:
            corr.unnamed_blocks += 1
            continue
        first = matches[0]
        src = FoldSource(
            handle=handle,
            sig=str(info.get("sig") or "?"),
            tokens=int(info.get("tokens") or 0),
            folds=int(info.get("folds") or 0),
            tool=first.tool,
            turn=first.turn,
            turn_text=turn_text.get(first.turn, ""),
            refetches=len(matches),
        )
        corr.fold_sources.append(src)
        if src.refetches >= 2:
            corr.refetched.append(src)
    corr.fold_sources.sort(key=lambda s: -s.tokens)
    corr.refetched.sort(key=lambda s: -s.tokens * s.refetches)

    # --- unused tools: defined on requests (paid for) but never invoked
    defined: dict[str, int] = {}
    for r in d.requests:
        for t in r.get("tools") or []:
            name = str(t.get("name") or "?")
            defined[name] = max(defined.get(name, 0), int(t.get("tokens") or 0))
    invoked = {c.name for c in tr.tool_calls}
    corr.tools_defined = len(defined)
    corr.tools_invoked = len(defined.keys() & invoked)
    corr.unused_tools = sorted(
        ((n, tok) for n, tok in defined.items() if n not in invoked),
        key=lambda t: -t[1],
    )

    # --- cost per human turn: each request lands on the last turn before it
    if tr.turns and d.requests:
        buckets: dict[int, TurnCost] = {}
        for r in d.requests:
            ts = float(r.get("ts") or 0.0)
            turn = 0
            for t in tr.turns:
                if t.ts <= ts:
                    turn = t.index
                else:
                    break
            b = buckets.setdefault(
                turn,
                TurnCost(
                    index=turn,
                    text=turn_text.get(turn, ""),
                    requests=0,
                    baseline_tokens=0,
                    saved_tokens=0,
                ),
            )
            b.requests += 1
            b.baseline_tokens += int(r.get("compressible_tokens") or 0) + int(
                r.get("overhead_tokens") or 0
            )
            b.saved_tokens += int(r.get("tokens_saved") or 0)
        corr.turns = sorted(buckets.values(), key=lambda b: -b.baseline_tokens)
    return corr
