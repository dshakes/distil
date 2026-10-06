"""Compression policies: ``step(history, t) -> the messages that request would carry``.

A policy sees the untransformed history up to the new request and the request's start
time, and keeps whatever state it needs (what it sent before, when). It never sees the
cost model: the simulator prices whatever bytes it returns. To add a policy, subclass
:class:`Policy` (or wrap a function with :class:`EntryPolicy`) and register it in
:func:`build`.

Families:

* ``plain`` — identity.
* ``distil-lossless`` / ``distil-digest`` — the served adapter
  (``distil.adapters.anthropic.compress_messages``, ``persist=False``) over the WHOLE
  history every request, exactly as the harness's distil arm called it.
* entry policies — shape each tool result ONCE, when it first enters the context, and
  re-send that form byte-identically ever after (cache-stable by construction):
  ``entry-lossless`` (distil Tier-0), ``entry-digest`` (distil's per-result transform
  without the recency exemption), ``rtk-like`` (shell-command filters, see below),
  ``trunc-k`` (cap at k tokens: head/tail + every must-keep line, rest behind a handle).
* ``window`` — entry form at first, then distil's digest once a result is older than W
  requests, applied in batches every B requests (each batch is one deliberate cache bust).
* ``cold-only`` — the served adapter, but only on requests that start after the prefix
  cache has expired (the write is paid anyway); otherwise the last sent prefix + raw tail.
* ``headroom`` — the real headroom-ai package in a subprocess (``headroom_worker.py``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from distil.adapters import anthropic as _ad
from distil.compress.keep_policy import ContentKind, must_keep
from distil.compress.tier0 import collapse_runs

from .trajectory import Message, block_text, blocks

Shaper = Callable[[str, dict[str, Any]], str]  # (tool output, tool_use block) -> shaped


def _cc_last(messages: list[Message]) -> list[Message]:
    """The harness's breakpoint: ``cache_control`` on the newest block (agent.py)."""
    if not messages:
        return messages
    last = dict(messages[-1])
    c = last["content"]
    bl: list[Any] = [{"type": "text", "text": c}] if isinstance(c, str) else list(c)
    bl[-1] = {**bl[-1], "cache_control": {"type": "ephemeral"}}
    last["content"] = bl
    return [*messages[:-1], last]


def _strip_cc(messages: list[Message]) -> list[Message]:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [
                {k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b
                for b in c
            ]
        out.append({**m, "content": c})
    return out


class Policy:
    name = "plain"
    params: dict[str, Any] = {}
    #: Leaves recovery handles, so every request also carries the distil_expand tool.
    expand_tool = False

    def reset(self) -> None:
        pass

    def step(self, history: list[Message], t: float, ttl_s: float) -> list[Message]:
        return history

    def close(self) -> None:
        pass


class DistilServed(Policy):
    def __init__(self, verbatim: bool, expand_tool: bool | None = None) -> None:
        self.verbatim = verbatim
        self.expand_tool = (not verbatim) if expand_tool is None else expand_tool
        self.name = "distil-lossless" if verbatim else "distil-digest"
        if not verbatim and not self.expand_tool:
            self.name += "-sh"
        self.params = {"verbatim": verbatim}

    def step(self, history: list[Message], t: float, ttl_s: float) -> list[Message]:
        out, _ = _ad.compress_messages(_cc_last(history), verbatim=self.verbatim, persist=False)
        return _strip_cc(out)


# ------------------------------------------------------------------------ entry policies


def _tool_uses(history: list[Message]) -> dict[str, dict[str, Any]]:
    return {
        str(b.get("id")): b
        for m in history
        if m.get("role") == "assistant"
        for b in blocks(m)
        if isinstance(b, dict) and b.get("type") == "tool_use"
    }


def _with_text(block: dict[str, Any], text: str) -> dict[str, Any]:
    return {**block, "content": text}


class EntryPolicy(Policy):
    """Shape each tool result once, at entry; never touch an already-sent byte."""

    def __init__(self, name: str, shaper: Shaper, expand_tool: bool = True, **params: Any) -> None:
        self.name, self.shaper, self.params = name, shaper, params
        self.expand_tool = expand_tool
        self._shaped: dict[str, str] = {}

    def reset(self) -> None:
        self._shaped = {}

    def shaped(self, block: dict[str, Any], uses: dict[str, dict[str, Any]]) -> str:
        tid = str(block.get("tool_use_id"))
        if tid not in self._shaped:
            text = block_text(block)
            self._shaped[tid] = self.shaper(text, uses.get(tid, {}))
        return self._shaped[tid]

    def step(self, history: list[Message], t: float, ttl_s: float) -> list[Message]:
        uses = _tool_uses(history)
        out: list[Message] = []
        for m in history:
            c = m.get("content")
            if m.get("role") != "user" or not isinstance(c, list):
                out.append(m)
                continue
            nb = [
                _with_text(b, self.shaped(b, uses))
                if isinstance(b, dict) and b.get("type") == "tool_result"
                else b
                for b in c
            ]
            out.append({**m, "content": nb})
        return out


def tier0(text: str, use: dict[str, Any]) -> str:
    return _ad._apply_tier0(text)


def entry_digest(text: str, use: dict[str, Any]) -> str:
    """distil's own per-result transform, as served, minus the recency exemption."""
    return _ad._compress_tool_result_text(text, _ad.RestoreStore(persist=False), False, False)


def _handle(text: str) -> str:
    return _ad._handle(text)


# rtk-like: the `distil sh` filter set sh-v1 (branch feat/shell-at-source, distil/shell.py,
# ADR 0026), restated here because that branch is unmerged. RTK's own binary is linux-only
# and wraps the *command*, so it cannot be applied to recorded output offline; these are
# the same classes of filter (ANSI/CR, repeated lines, git advice, passing-test lines).
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_GIT_ADVICE = re.compile(r'^\s*\((?:use "git |commit or discard |fix conflicts and run ")[^\n]*\)$')
_PASS = {
    "pytest": re.compile(r"^\S.*::\S.* PASSED(?: +\[ *\d+%\])?$|^\S+\.py \.+ *(?:\[ *\d+%\])?$"),
    "unittest": re.compile(r"^\S.* \.\.\. ok$|^\.+$"),
}
_COUNTS = re.compile(r"\b\d+ +(?:passed|passing|failed|failing|skipped|errors?|errored)\b", re.I)
_PY = re.compile(r"(?:^|[\s;&|(])(?:\S*/)?python\S*\s+(?:-\S+\s+)*")


def _kind(cmd: str) -> str | None:
    if re.search(r"(?:^|[\s;&|])git\s+status\b", cmd):
        return "git-status"
    if re.search(r"(?:^|[\s;&|/])(?:py\.test|pytest)\b|-m\s+pytest\b", cmd):
        return "pytest"
    if re.search(r"-m\s+unittest\b|runtests\.py", cmd):
        return "unittest"
    return None


def protected(line: str) -> bool:
    return (
        must_keep(line, ContentKind.GENERIC)
        or "FAIL" in line
        or "ERROR" in line
        or _COUNTS.search(line) is not None
    )


def rtk_like(text: str, use: dict[str, Any]) -> str:
    if use.get("name") != "bash":
        return text
    kind = _kind(str((use.get("input") or {}).get("command", "")))
    out = _ANSI.sub("", text).replace("\r\n", "\n")
    if "\r" in out:
        out = "\n".join(ln.rsplit("\r", 1)[-1] for ln in out.split("\n"))
    if kind == "git-status":
        out = "\n".join(ln for ln in out.split("\n") if not _GIT_ADVICE.match(ln))
    out = collapse_runs(out)
    pat = _PASS.get(kind or "")
    if pat is not None:
        lines = out.split("\n")
        kept = [ln for ln in lines if not (pat.match(ln) and not protected(ln))]
        if len(lines) - len(kept) >= 5:
            out = "\n".join(kept) + (
                f"\n[{len(lines) - len(kept)} passing-test lines elided; "
                f"full output: handle {_handle(text)}]"
            )
    return out if len(out) < len(text) else text


def truncate(k: int, head_frac: float = 0.5) -> Shaper:
    """Cap at ~k tokens: head + tail lines, plus every must-keep line from the middle."""

    def shape(text: str, use: dict[str, Any]) -> str:
        from distil.tokenizer import DEFAULT as tok

        if tok.count(text) <= k:
            return text
        lines = text.split("\n")
        budget_head, budget_tail = k * head_frac, k * (1 - head_frac)
        head: list[int] = []
        used = 0.0
        for i, ln in enumerate(lines):
            c = tok.count(ln) + 1
            if used + c > budget_head:
                break
            head.append(i)
            used += c
        tail: list[int] = []
        used = 0.0
        for i in range(len(lines) - 1, (head[-1] if head else -1), -1):
            c = tok.count(lines[i]) + 1
            if used + c > budget_tail:
                break
            tail.append(i)
            used += c
        keep = set(head) | set(tail)
        keep |= {i for i, ln in enumerate(lines) if protected(ln)}
        out: list[str] = []
        gap = 0
        for i, ln in enumerate(lines):
            if i in keep:
                if gap:
                    out.append(f"[... {gap} lines elided; full output: handle {_handle(text)}]")
                    gap = 0
                out.append(ln)
            else:
                gap += 1
        if gap:
            out.append(f"[... {gap} lines elided; full output: handle {_handle(text)}]")
        res = "\n".join(out)
        return res if len(res) < len(text) else text

    return shape


# ------------------------------------------------------------------ history-rewriting ones


class Window(Policy):
    """Entry form first; distil's digest once a result is older than ``window`` requests,
    re-rendered only every ``batch`` requests (one cache bust per batch)."""

    def __init__(self, entry: Shaper, window: int, batch: int, name: str = "") -> None:
        self.entry = EntryPolicy("entry", entry)
        self.window, self.batch = window, batch
        self.expand_tool = True
        self.name = name or f"window-w{window}-b{batch}"
        self.params = {"window": window, "batch": batch}
        self._aged: dict[str, str] = {}
        self._first_seen: dict[str, int] = {}
        self._k = 0

    def reset(self) -> None:
        self.entry.reset()
        self._aged, self._first_seen, self._k = {}, {}, 0

    def step(self, history: list[Message], t: float, ttl_s: float) -> list[Message]:
        out = self.entry.step(history, t, ttl_s)
        k = self._k
        self._k += 1
        rerender = k % self.batch == 0
        res: list[Message] = []
        for m in out:
            c = m.get("content")
            if m.get("role") != "user" or not isinstance(c, list):
                res.append(m)
                continue
            nb = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = str(b.get("tool_use_id"))
                    self._first_seen.setdefault(tid, k)
                    if (
                        tid not in self._aged
                        and rerender
                        and k - self._first_seen[tid] >= (self.window)
                    ):
                        self._aged[tid] = entry_digest(block_text(b), {})
                    if tid in self._aged:
                        b = _with_text(b, self._aged[tid])
                nb.append(b)
            res.append({**m, "content": nb})
        return res


class ColdOnly(Policy):
    """Rewrite history with the served adapter only when the prefix cache is already cold."""

    def __init__(self, verbatim: bool = False) -> None:
        self.inner = DistilServed(verbatim)
        self.expand_tool = not verbatim
        self.name = "cold-only-" + ("lossless" if verbatim else "digest")
        self.params = {"verbatim": verbatim}
        self._sent: list[Message] = []
        self._t = -1e18

    def reset(self) -> None:
        self._sent, self._t = [], -1e18

    def step(self, history: list[Message], t: float, ttl_s: float) -> list[Message]:
        cold = t - self._t > ttl_s
        self._t = t
        if cold:
            self._sent = self.inner.step(history, t, ttl_s)
        else:
            self._sent = self._sent + history[len(self._sent) :]
        return self._sent


class Headroom(Policy):
    """headroom-ai in a subprocess; its proxy's default cache mode (append-only delta)."""

    name = "headroom"

    def __init__(self, python: str, model: str = "claude-sonnet-5-5") -> None:
        self.python, self.model = python, model
        self.params = {"package": "headroom-ai", "mode": "proxy cache mode (default)"}
        self._p: subprocess.Popen[str] | None = None
        self._conv = 0
        self.meta: dict[str, int] = {}

    def _proc(self) -> subprocess.Popen[str]:
        if self._p is None:
            worker = os.path.join(os.path.dirname(__file__), "headroom_worker.py")
            self._p = subprocess.Popen(
                [self.python, worker],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                env={**os.environ, "HEADROOM_TELEMETRY": "off", "HF_HUB_OFFLINE": "1"},
            )
        return self._p

    def reset(self) -> None:
        self._conv += 1

    def step(self, history: list[Message], t: float, ttl_s: float) -> list[Message]:
        p = self._proc()
        assert p.stdin is not None and p.stdout is not None
        req = {"conv": str(self._conv), "model": self.model, "messages": history}
        p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        resp = json.loads(p.stdout.readline() or '{"error": "worker exited"}')
        if "error" in resp:
            raise RuntimeError(f"headroom worker: {resp['error']}")
        mode = resp.get("meta", {}).get("mode", "?")
        self.meta[mode] = self.meta.get(mode, 0) + 1
        return list(resp["messages"])

    def close(self) -> None:
        if self._p is not None:
            if self._p.stdin:
                self._p.stdin.close()
            self._p.wait(timeout=30)
            self._p = None


@dataclass
class Spec:
    name: str
    make: Callable[[], Policy]
    params: dict[str, Any] = field(default_factory=dict)


def build(headroom_python: str | None = None) -> list[Spec]:
    """The named policies, plus the parameter grid the search explores."""
    specs = [
        Spec("plain", Policy),
        Spec("distil-lossless", lambda: DistilServed(True)),
        Spec("distil-digest", lambda: DistilServed(False)),
        Spec("distil-digest-sh", lambda: DistilServed(False, expand_tool=False)),
        Spec("rtk-like", lambda: EntryPolicy("rtk-like", rtk_like, expand_tool=False)),
        Spec("entry-lossless", lambda: EntryPolicy("entry-lossless", tier0, expand_tool=False)),
        Spec("entry-digest", lambda: EntryPolicy("entry-digest", entry_digest)),
        # Recovery through the agent's existing shell (`distil expand <h>`, as `distil sh`
        # does) instead of an injected tool: no per-request tool-definition overhead.
        Spec(
            "entry-digest-sh",
            lambda: EntryPolicy("entry-digest-sh", entry_digest, expand_tool=False),
        ),
        Spec("cold-only-digest", lambda: ColdOnly(False)),
    ]
    for k in (500, 1000, 2000, 4000, 8000):
        for rec in ("tool", "sh"):
            name = f"trunc-{k}" + ("-sh" if rec == "sh" else "")
            specs.append(
                Spec(
                    name,
                    partial(EntryPolicy, name, truncate(k), expand_tool=rec == "tool"),
                    {"k": k, "recovery": rec},
                )
            )
    for w in (4, 8, 16):
        for b in (4, 8, 16):
            specs.append(
                Spec(
                    f"window-w{w}-b{b}",
                    partial(Window, tier0, w, b),
                    {"window": w, "batch": b},
                )
            )
    if headroom_python:
        specs.append(Spec("headroom", lambda: Headroom(headroom_python)))
    return specs
