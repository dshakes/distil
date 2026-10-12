"""What would distil have changed on *your* recent agent sessions? — offline, no install.

Replays local Claude Code transcripts (``~/.claude/projects/**/*.jsonl``) through the
SHIPPED served adapter, :func:`distil.adapters.anthropic.compress_messages` with
``persist=False``, in both modes a user can get:

* **lossless-only** (``verbatim=True``) — Tier-0, the subscription default;
* **digest** (``verbatim=False``) — reversible Tier-1 digests, the metered-key default.

Transcript parsing, request reconstruction and the cache-breakpoint rule are shared with
``benchmarks/session_replay.py`` (which imports them from here): consecutive user lines
merge into one message, streamed assistant blocks sharing a ``message.id`` merge, every
user message ends one API request, and ``cache_control`` sits where the client put it
(Claude Code does not record it, so on the newest message, as Claude Code does).

Cache economics per replayed request: the prefix shared with the previous request is a
cache read, the rest a cache write (5-minute or 1-hour, in the proportion the session's own
``usage.cache_creation`` split recorded); the compressed read prefix ends at the first
message whose compressed bytes changed (a cache bust, measured not assumed). Priced with
:func:`distil.pricing.resolve` for the model the session ran on.

Content-free: only counts and dollar sums leave this module — never text, paths, session
ids or project names. Read-only: nothing is written anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tokenizer import DEFAULT as TOKENIZER

Message = dict[str, Any]

MIN_REQUESTS = 10  # below this a budget-trimmed session is skipped, not sampled thinner
_CC = {"type": "ephemeral"}
_CC_JSON = ',"cache_control":{"type":"ephemeral"}'

#: Replay caps. ``compress_messages`` costs O(prefix) per request, twice (two modes), so a
#: heavy machine is sampled: the most recent sessions first, evenly spaced requests within
#: each, and a wall-clock deadline that stops the replay early (and says so).
MAX_REQUESTS = 120
PER_SESSION = 24
SESSION_BUDGET_BYTES = 40_000_000
DEADLINE_S = 15.0
#: Ledger users only see the "digest would add" line above this share of billed input.
MEANINGFUL_SHARE = 0.01


# --------------------------------------------------------------------------- parsing


@dataclass
class Session:
    messages: list[Message] = field(default_factory=list)
    bad_lines: int = 0
    cache_recorded: bool = False
    #: Epoch seconds of each message's first line (0.0 when the line had none).
    ts: list[float] = field(default_factory=list)
    #: Model id of the session's assistant responses (most common).
    models: Counter[str] = field(default_factory=Counter)
    cache_write: int = 0
    cache_write_1h: int = 0
    #: Assistant message index -> billed input tokens (uncached + cache read + write) of the
    #: response, i.e. what the request ending just before it actually cost in tokens.
    billed: dict[int, int] = field(default_factory=dict)
    #: Wire shape of ``messages``: ``anthropic`` (Claude Code), ``responses`` (Codex CLI: an
    #: OpenAI Responses ``input`` list) or ``gemini`` (Gemini CLI: ``contents``).
    shape: str = "anthropic"
    #: Request-ending message indices when the shape has no user-message rule (non-Anthropic).
    ends: list[int] | None = None

    @property
    def model(self) -> str | None:
        return self.models.most_common(1)[0][0] if self.models else None


def _blocks(content: Any) -> list[Any] | None:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content) if isinstance(content, list) else None


def _epoch(iso: Any) -> float:
    from .transcripts.claude_code import _epoch as ep

    return ep(iso if isinstance(iso, str) else None)


def parse_transcript(lines: Iterable[str]) -> Session:
    """Reconstruct the main-thread message list from Claude Code JSONL lines.

    Tolerant of schema variants: non-message record types are ignored, malformed JSON or a
    message with a missing/ill-typed role or content is counted in ``bad_lines``.
    """
    sess = Session()
    last_id: str | None = None
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            sess.bad_lines += 1
            continue
        if not isinstance(rec, dict):
            sess.bad_lines += 1
            continue
        if rec.get("type") not in ("user", "assistant") or rec.get("isSidechain") is True:
            continue
        if rec.get("isApiErrorMessage") is True:
            continue
        msg = rec.get("message")
        role = msg.get("role") if isinstance(msg, dict) else None
        blocks = _blocks(msg.get("content")) if isinstance(msg, dict) else None
        if role not in ("user", "assistant") or blocks is None or not blocks:
            sess.bad_lines += 1
            continue
        if any(isinstance(b, dict) and b.get("cache_control") for b in blocks):
            sess.cache_recorded = True
        assert isinstance(msg, dict)  # role/blocks above were read from a dict
        mid = msg.get("id") if role == "assistant" and isinstance(msg.get("id"), str) else None
        prev = sess.messages[-1] if sess.messages else None
        # Streamed assistant blocks share a message id; parallel tool_results are
        # consecutive user lines. Both are ONE message in the real request.
        if prev is not None and prev["role"] == role == "user":
            prev["content"].extend(blocks)
        elif prev is not None and role == "assistant" and mid is not None and mid == last_id:
            prev["content"].extend(blocks)
        else:
            sess.messages.append({"role": role, "content": blocks})
            sess.ts.append(_epoch(rec.get("timestamp")))
            if role == "assistant":  # one response, one usage (repeated on every block line)
                _note_usage(sess, msg, len(sess.messages) - 1)
        last_id = mid
    return sess


def _note_usage(sess: Session, msg: dict[str, Any], index: int) -> None:
    if isinstance(msg.get("model"), str):
        sess.models[msg["model"]] += 1
    u = msg.get("usage")
    if not isinstance(u, dict):
        return
    try:
        write = int(u.get("cache_creation_input_tokens") or 0)
        billed = (
            int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0) + write
        )
        split = u.get("cache_creation")
        w1h = int(split.get("ephemeral_1h_input_tokens") or 0) if isinstance(split, dict) else 0
    except (TypeError, ValueError):
        return
    sess.cache_write += write
    sess.cache_write_1h += min(w1h, write)
    if billed:
        sess.billed[index] = billed


def _read(path: Path, shape: str) -> Session:
    """Parse one transcript of *shape* (raises OSError; the non-Claude readers are fail-open)."""
    if shape == "responses":
        from .transcripts.codex import session
    elif shape == "gemini":
        from .transcripts.gemini_cli import session
    else:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return parse_transcript(fh)
    return session(path)


def discover(root: Path) -> list[Path]:
    """Top-level transcripts, sorted so sampling is reproducible."""
    return sorted(p for p in root.rglob("*.jsonl") if "subagents" not in p.parts)


def request_indices(messages: list[Message]) -> list[int]:
    """Index of the last message of every request: each user message ends one."""
    return [i for i, m in enumerate(messages) if m["role"] == "user"]


def with_breakpoint(messages: list[Message], recorded: bool) -> list[Message]:
    """The request as sent: shallow copies, plus a breakpoint on the newest message if the
    transcript did not record any."""
    if recorded or not messages:
        return list(messages)
    *head, last = messages
    blocks = list(last["content"])
    for k in range(len(blocks) - 1, -1, -1):
        if isinstance(blocks[k], dict):
            blocks[k] = {**blocks[k], "cache_control": dict(_CC)}
            break
    return [*head, {**last, "content": blocks}]


# --------------------------------------------------------------------------- measuring


def _dump(msg: Message) -> str:
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":")).replace(_CC_JSON, "")


class _Sizer:
    """Bytes / tokens / identity of messages, memoised per process on content digest."""

    def __init__(self) -> None:
        self._tok: dict[bytes, int] = {}

    def of(self, msg: Message) -> tuple[bytes, int, int]:
        text = _dump(msg)
        key = hashlib.blake2b(text.encode(), digest_size=12).digest()
        tok = self._tok.get(key)
        if tok is None:
            tok = self._tok[key] = TOKENIZER.count(text)
        return key, len(text.encode()), tok


def plan_requests(
    ends: list[int], msg_bytes: list[int], max_requests: int, budget_bytes: int
) -> list[tuple[int, bool]] | None:
    """``(end, recorded)`` in replay order, or None if even the smallest plan is over budget.

    Replaying request i costs O(prefix bytes), so a thousand-request session is quadratic.
    Past ``max_requests`` it keeps evenly spaced requests (always including the last), and
    halves that until the bytes to compress fit ``budget_bytes``. Each kept request is preceded
    by its real predecessor, replayed but not recorded, so the cache-prefix comparison is
    against what the provider actually held.
    """
    cum = [0]
    for b in msg_bytes:
        cum.append(cum[-1] + b)
    k = max_requests
    while True:
        if len(ends) <= k:
            plan = [(e, True) for e in ends]
        else:
            keep = {round(j * (len(ends) - 1) / (k - 1)) for j in range(k)} if k > 1 else {0}
            plan = []
            for i in sorted(keep):
                if i > 0 and (not plan or plan[-1][0] != ends[i - 1]):
                    plan.append((ends[i - 1], False))
                plan.append((ends[i], True))
        if sum(cum[e + 1] for e, _ in plan) <= budget_bytes:
            return plan
        if k <= MIN_REQUESTS:
            return None
        k = max(MIN_REQUESTS, min(k, len(ends)) // 2)


# --------------------------------------------------------------------------- the what-if


@dataclass
class Arm:
    """One mode's replay totals over the sampled requests (distil's offline token estimate
    over the conversation messages; dollars cache-aware at the session's model price)."""

    tokens_before: int = 0
    tokens_after: int = 0
    usd_before: float = 0.0
    usd_after: float = 0.0
    cache_busts: int = 0

    def add(self, other: Arm) -> None:
        self.tokens_before += other.tokens_before
        self.tokens_after += other.tokens_after
        self.usd_before += other.usd_before
        self.usd_after += other.usd_after
        self.cache_busts += other.cache_busts

    @property
    def removed(self) -> int:
        return self.tokens_before - self.tokens_after


@dataclass
class WhatIf:
    sessions: int = 0
    requests_replayed: int = 0
    #: Billed input tokens (from the transcript's own usage) of the requests replayed —
    #: the denominator that turns the sample into a share of YOUR bill, system prompt and
    #: tool definitions included.
    billed_input_tokens: int = 0
    unpriced_requests: int = 0
    unparseable_lines: int = 0
    failed_sessions: int = 0
    stopped_early: bool = False
    elapsed_s: float = 0.0
    lossless: Arm = field(default_factory=Arm)
    digest: Arm = field(default_factory=Arm)

    def to_dict(self, window_input_tokens: int, requests_in_window: int) -> dict[str, Any]:
        """The sample, extrapolated to the window by billed input tokens (never scaled down).

        ``share_of_billed_input`` divides by the replayed requests' billed input, which
        includes the system prompt and tool definitions distil cannot see here — so it is
        the share of what you actually sent, not of the conversation alone."""
        b = self.billed_input_tokens
        k = max(1.0, window_input_tokens / b) if b else 0.0
        d: dict[str, Any] = {
            "sessions": self.sessions,
            "requests_replayed": self.requests_replayed,
            "requests_in_window": max(requests_in_window, self.requests_replayed),
            "billed_input_tokens_replayed": b,
            "unpriced_requests": self.unpriced_requests,
            "unparseable_lines": self.unparseable_lines,
            "failed_sessions": self.failed_sessions,
            "stopped_early": self.stopped_early,
            "elapsed_s": round(self.elapsed_s, 2),
            "scale": round(k, 3),
        }
        for name, a in (("lossless_only", self.lossless), ("digest", self.digest)):
            # ponytail: the offline estimate can overshoot the billed count; clamp, never >100%
            gone = min(a.removed, b)
            d[name] = {
                "input_tokens_removed": round(gone * k),
                "share_of_billed_input": round(gone / b, 4) if b else 0.0,
                "usd_saved": round((a.usd_before - a.usd_after) * k, 4),
                "cache_busts": a.cache_busts,
            }
        d["method"] = (
            "offline replay through distil.adapters.anthropic.compress_messages(persist=False); "
            "conversation messages only (the system prompt and tool definitions are not in "
            "transcripts); distil's offline token estimate; cache reads/writes priced per the "
            "session's own usage; scaled to the window by billed input tokens"
        )
        return d


def _cost(read: int, write: int, price: Any, w1h_share: float) -> float:
    w = price.cache_write * (1.0 - w1h_share) + price.cache_write_1h * w1h_share
    return read * price.cache_read + write * w


def _compress_other(shape: str, req: list[Message], verbatim: bool) -> list[Message]:
    """Compress a Codex (Responses ``input``) or Gemini (``contents``) request through its
    shipped adapter, memory-only (``persist=False``) so this module stays read-only."""
    from .adapters import gemini, openai

    if shape == "responses":
        return openai.compress_responses_input(req, verbatim=verbatim, persist=False)[0]
    body = gemini.compress_generate_request({"contents": req}, verbatim=verbatim, persist=False)
    return body[0]["contents"]


def _replay(
    sess: Session,
    plan: list[tuple[int, bool]],
    arms: dict[bool, Arm],
    *,
    sizer: _Sizer,
    price: Any,
    deadline: float,
) -> tuple[int, int, bool]:
    """Replay *plan* in every mode in lockstep (``arms`` keyed by ``verbatim``), so a deadline
    stop leaves the modes comparable. Returns (requests recorded, their billed input tokens,
    stopped at the deadline)."""
    from .adapters import anthropic as adapter

    shape = sess.shape
    orig = [sizer.of(m) for m in sess.messages]
    w1h = sess.cache_write_1h / sess.cache_write if sess.cache_write else 0.0
    prev_end = -1
    prev_keys: dict[bool, list[bytes]] = {v: [] for v in arms}
    n_rec = billed = 0
    for end, recorded in plan:
        if time.monotonic() >= deadline:
            return n_rec, billed, True
        # No billed response (an interrupted last request): replayed for the cache state,
        # not counted — there is nothing to measure it against.
        recorded = recorded and end + 1 in sess.billed
        # Only Anthropic needs an explicit breakpoint; OpenAI and Gemini cache prefixes themselves.
        req = (
            with_breakpoint(sess.messages[: end + 1], sess.cache_recorded)
            if shape == "anthropic"
            else sess.messages[: end + 1]
        )
        tb = sum(s[2] for s in orig[: end + 1])
        read_b = sum(s[2] for s in orig[: prev_end + 1])
        for verbatim, arm in arms.items():
            if shape == "anthropic":
                comp, _store = adapter.compress_messages(req, verbatim=verbatim, persist=False)
                adapter.take_quote_hazard()  # per-thread state the served path would consume
            else:
                comp = _compress_other(shape, req, verbatim)
            sized = [sizer.of(m) for m in comp]
            keys = [s[0] for s in sized]
            if recorded:
                ta = sum(s[2] for s in sized)
                # The compressed cache prefix ends at the first message whose bytes changed
                # since the previous request: everything after it is written again.
                old = prev_keys[verbatim]
                stable = 0
                while stable <= prev_end and keys[stable] == old[stable]:
                    stable += 1
                read_a = sum(s[2] for s in sized[:stable])
                arm.tokens_before += tb
                arm.tokens_after += ta
                arm.cache_busts += prev_end >= 0 and stable <= prev_end
                if price is not None:
                    arm.usd_before += _cost(read_b, tb - read_b, price, w1h)
                    arm.usd_after += _cost(read_a, ta - read_a, price, w1h)
            prev_keys[verbatim] = keys
        if recorded:
            n_rec += 1
            billed += sess.billed[end + 1]
        prev_end = end
    return n_rec, billed, False


def claude_projects_root() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))) / "projects"


def other_roots() -> list[tuple[Path, str]]:
    """Codex CLI and Gemini CLI transcript roots, each with its session shape."""
    from .transcripts.codex import sessions_root
    from .transcripts.gemini_cli import tmp_root

    return [(sessions_root(), "responses"), (tmp_root(), "gemini")]


def _discover_any(root: Path, shape: str) -> list[Path]:
    if shape == "responses":
        return sorted(root.rglob("rollout-*.jsonl"))
    if shape == "gemini":
        return sorted(p for p in root.glob("*/chats/session-*") if p.suffix in (".json", ".jsonl"))
    return discover(root)


def run(
    since: float | None,
    *,
    root: Path | None = None,
    max_requests: int = MAX_REQUESTS,
    per_session: int = PER_SESSION,
    deadline_s: float = DEADLINE_S,
    progress: Callable[[int, int], None] | None = None,
) -> WhatIf:
    """Replay the most recent requests in the window, both modes. Never raises on bad input:
    an unreadable or unparseable transcript is skipped."""
    from . import pricing

    t0 = time.monotonic()
    deadline = t0 + deadline_s
    # An explicit root is Claude Code's; with none, every agent's transcripts are replayed.
    roots = (
        [(root, "anthropic")] if root else [(claude_projects_root(), "anthropic"), *other_roots()]
    )
    out = WhatIf()
    found: list[tuple[float, Path, str]] = []
    for r, shape in roots:
        try:
            found += [(p.stat().st_mtime, p, shape) for p in _discover_any(r, shape)]
        except OSError:
            continue
    found = sorted(((m, p, sh) for m, p, sh in found if since is None or m >= since), reverse=True)
    for _m, path, shape in found:
        left = max_requests - out.requests_replayed
        if left <= 0:
            break
        if time.monotonic() >= deadline:
            out.stopped_early = True
            break
        try:
            sess = _read(path, shape)
        except OSError:
            continue
        out.unparseable_lines += sess.bad_lines
        ends = [
            e
            for e in (request_indices(sess.messages) if sess.ends is None else sess.ends)
            if since is None or sess.ts[e] >= since
        ]
        if not ends:
            continue
        sizer = _Sizer()
        plan = plan_requests(
            ends,
            [len(_dump(m).encode()) for m in sess.messages],
            max(1, min(per_session, left)),
            SESSION_BUDGET_BYTES,
        )
        if plan is None:
            continue
        price = pricing.resolve(sess.model)
        arms = {True: Arm(), False: Arm()}
        try:
            n, billed, late = _replay(sess, plan, arms, sizer=sizer, price=price, deadline=deadline)
        except (TypeError, KeyError, ValueError, AttributeError, IndexError, RecursionError):
            # A transcript shape the adapter rejects costs that session, never the screen;
            # counted (fixed-vocabulary), never echoed.
            out.failed_sessions += 1
            continue
        out.lossless.add(arms[True])
        out.digest.add(arms[False])
        out.sessions += 1
        out.requests_replayed += n
        out.billed_input_tokens += billed
        out.unpriced_requests += n if price is None else 0
        if progress is not None:
            progress(out.requests_replayed, max_requests)
        if late:
            out.stopped_early = True
            break
    out.elapsed_s = time.monotonic() - t0
    return out
