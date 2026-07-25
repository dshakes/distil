"""Agent transcript adapters — see ``base.py`` for the contract.

The registry key is the executable basename the wrap manifest records as
``tool``. Adding an agent is: write one adapter module, add one line here.
"""

from __future__ import annotations

from pathlib import Path

from .base import ToolCall, ToolResult, Transcript, TranscriptAdapter, UserTurn
from .claude_code import ClaudeCodeAdapter

ADAPTERS: dict[str, TranscriptAdapter] = {
    "claude": ClaudeCodeAdapter(),
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
        adapter = ADAPTERS.get(tool)
        candidates = [adapter] if adapter is not None else list(ADAPTERS.values())
        for a in candidates:
            tr = a.load(p)
            if tr.turns or tr.tool_results:
                return tr
        return None
    adapters = [ADAPTERS[tool]] if tool in ADAPTERS else list(ADAPTERS.values())
    for a in adapters:
        for candidate in a.discover(window, cwd)[:3]:
            tr = a.load(candidate)
            if tr.turns or tr.tool_results:
                return tr
    return None
