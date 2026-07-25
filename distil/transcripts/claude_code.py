"""Claude Code transcript adapter.

Claude Code writes one JSONL file per session under
``~/.claude/projects/<cwd-slug>/<session-uuid>.jsonl``. Lines carry a ``type``
("user", "assistant", "ai-title", ...); user/assistant lines hold an
Anthropic-shaped ``message`` plus ISO ``timestamp``, ``cwd`` and
``isSidechain``. Human prompts are user lines whose ``origin.kind`` is
"human" (tool results also arrive as user lines — they are not turns).
Subagent (sidechain) traffic lives in the same file and is kept: the wrap
proxied those requests too.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import ToolCall, ToolResult, Transcript, UserTurn

_EXCERPT = 80  # turn-label length


def _epoch(iso: str | None) -> float:
    if not iso:
        return 0.0
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _blocks(content: Any) -> list[dict[str, Any]]:
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _block_text(content: Any) -> str:
    """Flatten a message/tool_result content field to text."""
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text", "")) for b in _blocks(content) if b.get("type") == "text")


class ClaudeCodeAdapter:
    name = "claude"

    def _root(self) -> Path:
        return Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))) / "projects"

    def discover(self, window: tuple[float, float], cwd: str | None) -> list[Path]:
        root = self._root()
        dirs: list[Path] = []
        if cwd:
            slug = cwd.replace("/", "-").replace("\\", "-").replace("_", "-").replace(".", "-")
            cand = root / slug
            if cand.is_dir():
                dirs.append(cand)
        if not dirs:
            try:
                dirs = [p for p in root.iterdir() if p.is_dir()]
            except OSError:
                return []
        lo, hi = window
        scored: list[tuple[float, Path]] = []
        for d in dirs:
            try:
                files = list(d.glob("*.jsonl"))
            except OSError:
                continue
            for f in files:
                try:
                    mtime = f.stat().st_mtime
                except OSError:
                    continue
                # Cheap overlap filter: the file must have been written into
                # after the session started and begun before it ended.
                if mtime < lo - 60:
                    continue
                first_ts = self._first_ts(f)
                if first_ts and first_ts > hi + 60:
                    continue
                overlap = min(hi, mtime) - max(lo, first_ts or lo)
                scored.append((overlap, f))
        return [f for _o, f in sorted(scored, key=lambda t: -t[0])]

    @staticmethod
    def _first_ts(path: Path) -> float:
        try:
            with path.open(encoding="utf-8") as fh:
                for _ in range(50):
                    line = fh.readline()
                    if not line:
                        break
                    try:
                        ts = _epoch(json.loads(line).get("timestamp"))
                    except json.JSONDecodeError:
                        continue
                    if ts:
                        return ts
        except OSError:
            pass
        return 0.0

    def load(self, path: Path) -> Transcript:
        tr = Transcript(agent=self.name, path=path)
        call_names: dict[str, str] = {}  # tool_use_id -> tool name
        turn = 0
        try:
            fh = path.open(encoding="utf-8")
        except OSError:
            return tr
        with fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                kind = rec.get("type")
                if kind == "ai-title":
                    tr.label = str(rec.get("aiTitle") or tr.label)
                    continue
                if kind not in ("user", "assistant"):
                    continue
                ts = _epoch(rec.get("timestamp"))
                if ts:
                    tr.started = min(tr.started or ts, ts)
                    tr.ended = max(tr.ended, ts)
                if not tr.cwd and rec.get("cwd"):
                    tr.cwd = str(rec["cwd"])
                msg = rec.get("message") or {}
                content = msg.get("content")
                if kind == "assistant":
                    for b in _blocks(content):
                        if b.get("type") == "tool_use":
                            name = str(b.get("name") or "")
                            call_id = str(b.get("id") or "")
                            call_names[call_id] = name
                            tr.tool_calls.append(
                                ToolCall(ts=ts, name=name, call_id=call_id, turn=turn)
                            )
                    continue
                # user line: a human turn, tool results, or subagent input.
                origin = rec.get("origin")
                is_human = isinstance(origin, dict) and origin.get("kind") == "human"
                if is_human and not rec.get("isSidechain"):
                    text = _block_text(content).strip()
                    if text:
                        turn += 1
                        tr.turns.append(UserTurn(index=turn, ts=ts, text=text[:_EXCERPT]))
                for b in _blocks(content):
                    if b.get("type") == "tool_result":
                        text = _block_text(b.get("content")).strip()
                        if text:
                            tr.tool_results.append(
                                ToolResult(
                                    ts=ts,
                                    text=text,
                                    tool=call_names.get(str(b.get("tool_use_id") or ""), ""),
                                    turn=turn,
                                )
                            )
        return tr
