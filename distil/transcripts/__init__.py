"""Agent transcript adapters — see ``base.py`` for the contract.

The registry key is the executable basename the wrap manifest records as
``tool``. Adding an agent is: write one adapter module, add one line here.
"""

from __future__ import annotations

from pathlib import Path

from .base import ToolCall, ToolResult, Transcript, TranscriptAdapter, UserTurn
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter
from .gemini_cli import GeminiCliAdapter

ADAPTERS: dict[str, TranscriptAdapter] = {
    "claude": ClaudeCodeAdapter(),
    "codex": CodexAdapter(),
    "gemini": GeminiCliAdapter(),
}

__all__ = [
    "ADAPTERS",
    "ToolCall",
    "ToolResult",
    "Transcript",
    "TranscriptAdapter",
    "UserTurn",
    "find_transcript",
]


def find_transcript(
    tool: str,
    window: tuple[float, float],
    cwd: str | None = None,
    path: str | Path | None = None,
) -> Transcript | None:
    """Locate and load the agent transcript matching a wrap session.

    ``path`` short-circuits discovery (the user pointed at a file); otherwise
    the adapter registered for *tool* searches, falling back to every adapter
    when the tool is unknown (old sessions without a manifest).
    """
    if path is not None:
        p = Path(path).expanduser()
        # A file the user pointed at may be any agent's: the named adapter first, then the rest.
        candidates = sorted(ADAPTERS.values(), key=lambda a: a.name != tool)
        for a in candidates:
            tr = a.load(p)
            if tr.turns or tr.tool_results:
                return tr
        return None
    # Registered agents used to be Claude-only: keep the old fallback to every adapter, the
    # named one first, so a manifest naming "codex"/"gemini" still correlates a Claude log.
    adapters = sorted(ADAPTERS.values(), key=lambda a: a.name != tool)
    for a in adapters:
        for candidate in a.discover(window, cwd)[:3]:
            tr = a.load(candidate)
            if tr.turns or tr.tool_results:
                return tr
    return None
