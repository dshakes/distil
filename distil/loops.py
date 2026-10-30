"""Deterministic loop / re-fetch detector — extra agent steps, and which ones compression caused.

The real cost risk of compression is not the bytes it removes but the steps the agent then
takes to get them back: the file it reads again, the command it re-runs because the output
came back folded. ``compress.refetch`` already *prevents* the worst case by content (a result
that re-fetches folded lines is forwarded verbatim). This module *measures* it by call: it
walks the tool calls of one request, in any of the three wire shapes, and counts the newest
round's calls that repeat an earlier one:

* ``identical`` - same tool, same arguments, seen before;
* ``reread``    - the same file read again (any range);
* ``rerun``     - the same shell command run again;
* ``after_digest`` - the repeated call's earlier result reached the model as a distil stub.
  This is the actionable one: compression likely caused the extra step, so the content
  type of what was folded (``learn.signature``) is fed to the learned keep policy.

Only the newest round (the calls whose results end the request) is counted. Every request
re-sends the whole history, so counting it all would book one repeat once per later turn.

Content-free: the report holds counts and tool names. Hashes of arguments are used inside
a walk and never leave it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .compress import provenance as _prov

__all__ = ["LoopReport", "detect", "feed_stats"]

_HANDLE_RE = re.compile(r"handle=([0-9a-f]{8}) >>")


@dataclass
class LoopReport:
    """Counts for the newest round of tool calls in one request. Content-free."""

    calls: int = 0
    repeats: int = 0
    identical: int = 0
    reread: int = 0
    rerun: int = 0
    after_digest: int = 0
    by_tool: dict[str, int] = field(default_factory=dict)
    # Handles of the digested results the repeats re-fetched; resolved to a content
    # signature by the caller (the store lives there). Never serialised.
    handles: list[str] = field(default_factory=list, repr=False)

    def as_record(self) -> dict[str, Any] | None:
        """The per-request log entry, or None when nothing repeated (keeps rows small)."""
        if not self.repeats:
            return None
        return {
            "calls": self.calls,
            "repeats": self.repeats,
            "identical": self.identical,
            "reread": self.reread,
            "rerun": self.rerun,
            "after_digest": self.after_digest,
            "by_tool": dict(self.by_tool),
        }


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)


def _args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _events(items: list[Any]) -> list[tuple[str, str, str, Any]]:
    """Flatten any request shape to ``(kind, id, name, payload)`` in conversation order.

    kind is ``call`` (payload = args), ``result`` (payload = the result as text) or
    ``other``. Gemini has no ids, so calls and responses are paired by name and order.
    """
    out: list[tuple[str, str, str, Any]] = []
    called: dict[str, int] = {}
    answered: dict[str, int] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        kind = it.get("type")
        if kind == "function_call":  # OpenAI Responses
            cid = it.get("call_id") or it.get("id")
            out.append(("call", str(cid), str(it.get("name", "")), _args(it.get("arguments"))))
            continue
        if kind == "function_call_output":
            out.append(("result", str(it.get("call_id")), "", _text(it.get("output"))))
            continue
        if it.get("role") == "tool":  # OpenAI Chat
            out.append(("result", str(it.get("tool_call_id")), "", _text(it.get("content"))))
            continue
        for tc in it.get("tool_calls") or ():  # OpenAI Chat
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict):
                out.append(
                    ("call", str(tc.get("id")), str(fn.get("name", "")), _args(fn.get("arguments")))
                )
        blocks = it.get("content")
        for blk in blocks if isinstance(blocks, list) else ():  # Anthropic
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "tool_use":
                out.append(
                    ("call", str(blk.get("id")), str(blk.get("name", "")), _args(blk.get("input")))
                )
            elif blk.get("type") == "tool_result":
                out.append(("result", str(blk.get("tool_use_id")), "", _text(blk.get("content"))))
        for part in it.get("parts") or ():  # Gemini
            if not isinstance(part, dict):
                continue
            fc, fr = part.get("functionCall"), part.get("functionResponse")
            if isinstance(fc, dict):
                n = str(fc.get("name", ""))
                called[n] = called.get(n, 0) + 1
                out.append(("call", f"{n}#{called[n]}", n, _args(fc.get("args"))))
            elif isinstance(fr, dict):
                n = str(fr.get("name", ""))
                answered[n] = answered.get(n, 0) + 1
                out.append(("result", f"{n}#{answered[n]}", n, _text(fr.get("response"))))
    return out


def _key(name: str, args: Any) -> tuple[str, str]:
    path = _prov.read_target(name, args)
    if path:
        return "read", path
    cmd = _prov.command_text(args)
    if cmd:
        return "run", cmd
    return "call", name


def _digest(name: str, args: Any) -> str:
    blob = json.dumps([name, args], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _closed_by_text(items: list[Any]) -> bool:
    """True when an item without tool calls or results comes after the last tool result."""
    last = max(
        (n for n, it in enumerate(items) if any(e[0] == "result" for e in _events([it]))),
        default=-1,
    )
    return any(isinstance(it, dict) and not _events([it]) for it in items[last + 1 :])


def detect(items: list[Any]) -> LoopReport:
    """Count the newest round's repeated calls in *items* (messages, ``input`` or ``contents``).

    Pass the list as *forwarded*: a result that reached the model as a distil stub carries
    ``handle=... >>``, which is how ``after_digest`` is told from an ordinary repeat.
    """
    ev = _events(items)
    results = {i: p for k, i, _n, p in ev if k == "result"}
    # Newest round: strip what trails the last call, then take the calls back to the
    # previous result. Only calls answered in that trailing stretch count as new.
    end = len(ev)
    while end and ev[end - 1][0] != "call":
        end -= 1
    start = end
    while start and ev[start - 1][0] != "result":
        start -= 1
    trailing = {i for k, i, _n, _p in ev[end:] if k == "result"}
    new = {i for k, i, _n, _p in ev[start:end] if k == "call" and i in trailing}
    if new and _closed_by_text(items):
        new = set()  # a text exchange followed the round: an earlier request already counted it
    rep = LoopReport()
    seen_key: dict[tuple[str, str], str] = {}
    seen_full: set[str] = set()
    for k, cid, name, args in ev:
        if k != "call" or _prov._is_expand_call(name):
            continue
        key, full = _key(name, args), _digest(name, args)
        if cid in new:
            rep.calls += 1
            earlier = seen_key.get(key)
            if earlier is not None:
                rep.repeats += 1
                rep.by_tool[name] = rep.by_tool.get(name, 0) + 1
                rep.identical += full in seen_full
                rep.reread += key[0] == "read"
                rep.rerun += key[0] == "run"
                handles = _HANDLE_RE.findall(results.get(earlier, ""))
                if handles:
                    rep.after_digest += 1
                    rep.handles.extend(handles)
        seen_key[key] = cid
        seen_full.add(full)
    return rep


def feed_stats(report: LoopReport, stats: Any, sig_of_handle: Any) -> int:
    """Book each digest-caused repeat as an expansion of its content type.

    A re-fetch is the same evidence as a ``distil_expand``: the agent needed the folded
    block back. It goes into the same ``ExpandStats`` so the existing learned keep policy
    (``learn.keep_predicate``) backs off that content type - no second mechanism. Returns
    the number booked. *sig_of_handle* maps a handle to its ``learn.signature`` or None.
    """
    booked = 0
    for h in report.handles:
        sig = sig_of_handle(h)
        if sig:
            stats.record_expand(sig)
            booked += 1
    return booked
