"""Gemini CLI transcript adapter.

Gemini CLI records chats under ``~/.gemini/tmp/<project>/chats/session-*.jsonl`` (one
metadata line, then one message record per line, plus ``$set`` / ``$patch`` / ``$rewindTo``
mutation lines) or the legacy ``session-*.json`` (one object with a ``messages`` list). A
message is ``{id, timestamp, type: user|gemini|info|error|warning, content}``; ``gemini``
messages may carry ``toolCalls`` (each with its ``result`` Part list), ``tokens`` and
``model``. A record repeated under the same id replaces the earlier one; ``$rewindTo``
drops the named message and every one after it; ``$set`` / ``$patch`` are ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import ToolCall, ToolResult, Transcript, UserTurn
from .claude_code import _epoch

if TYPE_CHECKING:
    from ..whatif import Session

_EXCERPT = 80


def tmp_root() -> Path:
    return Path(os.environ.get("GEMINI_CLI_HOME") or Path.home()) / ".gemini" / "tmp"


def _read(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(metadata, messages)``; malformed lines are skipped (raises OSError)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        try:
            obj = json.loads(text)
        except ValueError:
            return {}, []
        msgs = obj.get("messages") if isinstance(obj, dict) else None
        return (obj if isinstance(obj, dict) else {}), [
            m for m in msgs or [] if isinstance(m, dict)
        ]
    meta: dict[str, Any] = {}
    msgs_by_id: dict[str, dict[str, Any]] = {}  # insertion-ordered; a repeated id replaces
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        if "$rewindTo" in rec:
            # Upstream (chatRecordingService.ts, the loader's `messageIds.splice(idx)`):
            # the named message and everything after it go; an unknown id clears all.
            ids = list(msgs_by_id)
            cut = ids.index(rec["$rewindTo"]) if rec["$rewindTo"] in ids else 0
            msgs_by_id = {k: msgs_by_id[k] for k in ids[:cut]}
        elif any(k.startswith("$") for k in rec):
            continue
        elif "sessionId" in rec and "type" not in rec:
            meta = rec
        elif isinstance(rec.get("id"), str):
            msgs_by_id[rec["id"]] = rec
    return meta, list(msgs_by_id.values())


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
    return [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []


def _text(content: Any) -> str:
    return "\n".join(str(p["text"]) for p in _parts(content) if p.get("text"))


def _result_text(result: Any) -> str:
    """Flatten a tool result Part list (``functionResponse.response`` or text) to text."""
    out: list[str] = []
    for p in _parts(result):
        fr = p.get("functionResponse")
        if isinstance(fr, dict):
            r = fr.get("response")
            out.append(
                str(r.get("output") or r.get("error") or "")
                if isinstance(r, dict) and ("output" in r or "error" in r)
                else json.dumps(r, ensure_ascii=False)
            )
        elif p.get("text"):
            out.append(str(p["text"]))
    return "\n".join(out)


def session(path: Path) -> Session:
    """The chat as a Gemini ``contents`` list for :mod:`distil.whatif` (raises OSError).

    A request ends at each user-role content (a prompt or a batch of function responses);
    the following ``gemini`` message's ``tokens.input`` (which includes the cached tokens)
    is that request's billed input."""
    from ..whatif import Session

    _meta, msgs = _read(path)
    ends: list[int] = []
    sess = Session(shape="gemini", ends=ends)
    for m in msgs:
        ts = _epoch(m.get("timestamp") if isinstance(m.get("timestamp"), str) else None)
        kind = m.get("type")
        if kind == "user":
            parts = _parts(m.get("content"))
            if parts:
                ends.append(len(sess.messages))
                sess.messages.append({"role": "user", "parts": parts})
                sess.ts.append(ts)
        elif kind == "gemini":
            calls = [c for c in m.get("toolCalls") or [] if isinstance(c, dict)]
            parts = _parts(m.get("content")) + [
                {"functionCall": {"name": c.get("name"), "args": c.get("args") or {}}}
                for c in calls
            ]
            if not parts:
                continue
            t = m.get("tokens")
            try:
                billed = int(t.get("input") or 0) if isinstance(t, dict) else 0
            except (TypeError, ValueError):
                billed = 0
            if billed:
                sess.billed[len(sess.messages)] = billed
            if isinstance(m.get("model"), str):
                sess.models[m["model"]] += 1
            sess.messages.append({"role": "model", "parts": parts})
            sess.ts.append(ts)
            resp = [
                p
                for c in calls
                for p in _parts(c.get("result"))
                if isinstance(p.get("functionResponse"), dict)
            ]
            if resp:
                ends.append(len(sess.messages))
                sess.messages.append({"role": "user", "parts": resp})
                sess.ts.append(ts)
    return sess


class GeminiCliAdapter:
    name = "gemini"

    def discover(self, window: tuple[float, float], cwd: str | None) -> list[Path]:
        lo, hi = window
        want = hashlib.sha256(cwd.encode()).hexdigest() if cwd else None
        scored: list[tuple[float, bool, Path]] = []
        try:
            files = [*tmp_root().glob("*/chats/session-*.json*")]
        except OSError:
            return []
        for f in files:
            if f.suffix not in (".json", ".jsonl"):
                continue
            try:
                mtime = f.stat().st_mtime
                meta = self._meta(f)
            except OSError:
                continue
            if mtime < lo - 60:
                continue
            first_ts = _epoch(str(meta.get("startTime") or ""))
            if first_ts and first_ts > hi + 60:
                continue
            overlap = min(hi, mtime) - max(lo, first_ts or lo)
            scored.append((overlap, want is not None and meta.get("projectHash") == want, f))
        if any(m for _o, m, _f in scored):  # a project match narrows the search, never empties it
            scored = [t for t in scored if t[1]]
        return [f for _o, _m, f in sorted(scored, key=lambda t: -t[0])]

    @staticmethod
    def _meta(path: Path) -> dict[str, Any]:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                try:
                    rec = json.loads(fh.readline())
                except ValueError:
                    return {}
            return rec if isinstance(rec, dict) else {}
        return _read(path)[0]

    def load(self, path: Path) -> Transcript:
        tr = Transcript(agent=self.name, path=path)
        try:
            meta, msgs = _read(path)
        except OSError:
            return tr
        # ponytail: Gemini CLI does not record the cwd, only its hash — tr.cwd stays "".
        tr.started = _epoch(str(meta.get("startTime") or ""))
        tr.ended = _epoch(str(meta.get("lastUpdated") or ""))
        turn = 0
        for m in msgs:
            ts = _epoch(m.get("timestamp") if isinstance(m.get("timestamp"), str) else None)
            if m.get("type") == "user":
                text = _text(m.get("content")).strip()
                if text:
                    turn += 1
                    tr.turns.append(UserTurn(index=turn, ts=ts, text=text[:_EXCERPT]))
            elif m.get("type") == "gemini":
                for c in m.get("toolCalls") or []:
                    if not isinstance(c, dict):
                        continue
                    name = str(c.get("name") or "")
                    cts = _epoch(
                        c.get("timestamp") if isinstance(c.get("timestamp"), str) else None
                    )
                    tr.tool_calls.append(
                        ToolCall(ts=cts or ts, name=name, call_id=str(c.get("id") or ""), turn=turn)
                    )
                    text = _result_text(c.get("result")).strip()
                    if text:
                        tr.tool_results.append(
                            ToolResult(ts=cts or ts, text=text, tool=name, turn=turn)
                        )
        return tr
