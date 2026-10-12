"""OpenAI Codex CLI transcript adapter.

Codex writes one rollout per session under
``$CODEX_HOME|~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl``. Each line is
``{"timestamp", "type", "payload"}``: ``session_meta`` (cwd), ``turn_context`` (model),
``response_item`` (an OpenAI Responses item: messages, ``function_call`` /
``custom_tool_call`` / ``local_shell_call`` and their ``*_output``) and ``event_msg``
``token_count`` (usage; ``info`` is null until the first response). Everything else is
ignored. Codex injects its own context (AGENTS.md, environment) as user messages that start
with ``<`` or ``# AGENTS.md`` — those are not human turns.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import ToolCall, ToolResult, Transcript, UserTurn
from .claude_code import _epoch

if TYPE_CHECKING:
    from ..whatif import Session

_EXCERPT = 80
_CALLS = {"function_call", "custom_tool_call", "local_shell_call"}
_OUTPUTS = {"function_call_output", "custom_tool_call_output"}


def sessions_root() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "sessions"


def _records(path: Path) -> list[tuple[float, str, dict[str, Any]]]:
    """``(epoch, tag, payload)`` per well-formed line; malformed lines are skipped."""
    out: list[tuple[float, str, dict[str, Any]]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("payload"), dict):
                out.append((_epoch(rec.get("timestamp")), str(rec.get("type")), rec["payload"]))
    return out


def _text(content: Any) -> str:
    """Flatten message content / a tool output (string or content-item array) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _text(content.get("content") or content.get("text"))
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("text")
        )
    return ""


def _is_human(p: dict[str, Any]) -> str:
    if p.get("type") != "message" or p.get("role") != "user":
        return ""
    text = _text(p.get("content")).strip()
    return "" if text.startswith(("<", "# AGENTS.md")) else text


def session(path: Path) -> Session:
    """The rollout as a Responses ``input`` list for :mod:`distil.whatif` (raises OSError).

    A request ends at the last input item before each model response, and the following
    ``token_count`` is that request's usage. OpenAI's ``input_tokens`` includes the cached
    tokens, so billed = input (+ cache writes, only when the usage reports them)."""
    from ..whatif import Session

    ends: list[int] = []
    sess = Session(shape="responses", ends=ends)
    model = ""
    pending: int | None = None
    prev_out = False
    for ts, tag, p in _records(path):
        kind = p.get("type")
        if tag == "turn_context" and isinstance(p.get("model"), str):
            model = p["model"]
        elif tag == "response_item":
            is_out = kind in _CALLS or kind == "reasoning" or p.get("role") == "assistant"
            if is_out and not prev_out and sess.messages:
                pending = len(sess.messages) - 1
            prev_out = is_out
            if kind != "reasoning":  # opaque encrypted blob: not measurable, not compressible
                sess.messages.append(p)
                sess.ts.append(ts)
        elif tag == "event_msg" and kind == "token_count" and pending is not None:
            info = p.get("info")
            u = info.get("last_token_usage") if isinstance(info, dict) else None
            if not isinstance(u, dict):
                continue
            try:
                write = int(u.get("cache_write_input_tokens") or 0)
                billed = int(u.get("input_tokens") or 0) + write
            except (TypeError, ValueError):
                continue
            sess.cache_write += write
            if model:
                sess.models[model] += 1
            if billed:
                sess.billed[pending + 1] = billed
            ends.append(pending)
            pending = None
    return sess


class CodexAdapter:
    name = "codex"

    def discover(self, window: tuple[float, float], cwd: str | None) -> list[Path]:
        lo, hi = window
        scored: list[tuple[float, bool, Path]] = []
        try:
            files = list(sessions_root().rglob("rollout-*.jsonl"))
        except OSError:
            return []
        for f in files:
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            if mtime < lo - 60:
                continue
            first_ts, f_cwd = self._head(f)
            if first_ts and first_ts > hi + 60:
                continue
            overlap = min(hi, mtime) - max(lo, first_ts or lo)
            scored.append((overlap, bool(cwd) and f_cwd == cwd, f))
        if any(m for _o, m, _f in scored):  # a cwd match narrows the search, never empties it
            scored = [t for t in scored if t[1]]
        return [f for _o, _m, f in sorted(scored, key=lambda t: -t[0])]

    @staticmethod
    def _head(path: Path) -> tuple[float, str]:
        """``(start epoch, cwd)`` from the ``session_meta`` line."""
        try:
            with path.open(encoding="utf-8") as fh:
                for _ in range(5):
                    try:
                        rec = json.loads(fh.readline())
                    except ValueError:
                        continue
                    p = rec.get("payload") if isinstance(rec, dict) else None
                    if isinstance(p, dict) and rec.get("type") == "session_meta":
                        return _epoch(str(p.get("timestamp") or rec.get("timestamp"))), str(
                            p.get("cwd") or ""
                        )
        except OSError:
            pass
        return 0.0, ""

    def load(self, path: Path) -> Transcript:
        tr = Transcript(agent=self.name, path=path)
        try:
            records = _records(path)
        except OSError:
            return tr
        names: dict[str, str] = {}  # call_id -> tool name
        turn = 0
        for ts, tag, p in records:
            if ts:
                tr.started = min(tr.started or ts, ts)
                tr.ended = max(tr.ended, ts)
            if tag == "session_meta":
                tr.cwd = tr.cwd or str(p.get("cwd") or "")
            elif tag == "response_item":
                kind = p.get("type")
                if kind in _CALLS:
                    name = str(
                        p.get("name") or ("local_shell" if kind == "local_shell_call" else "")
                    )
                    call_id = str(p.get("call_id") or "")
                    names[call_id] = name
                    tr.tool_calls.append(ToolCall(ts=ts, name=name, call_id=call_id, turn=turn))
                elif kind in _OUTPUTS:
                    text = _text(p.get("output")).strip()
                    if text:
                        tool = names.get(str(p.get("call_id") or ""), "")
                        tr.tool_results.append(ToolResult(ts=ts, text=text, tool=tool, turn=turn))
                elif text := _is_human(p):
                    turn += 1
                    tr.turns.append(UserTurn(index=turn, ts=ts, text=text[:_EXCERPT]))
        return tr
