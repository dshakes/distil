"""Trajectories: a real agent run as the ordered list of requests it sent.

A trajectory is the *untransformed* message history plus, for each model call, the index
where that call's request ended (``request_ends[k]`` -> ``messages[:request_ends[k]]``).
Policies decide what bytes each request would have carried; the cost model prices them.

Sources:

* the SWE-bench outcome harness (``benchmarks/swebench_outcome``): one JSON list of
  messages per task under ``transcripts/<arm>/`` and the billed usage totals per task in
  ``results.jsonl``. The harness stores the history *before* its arm transform, so every
  arm's transcript is a raw trajectory; only arms whose transform is the identity
  (plain, rtk) also give the exact bytes that were billed.
* local Claude Code sessions, through :func:`distil.whatif.parse_transcript` (the same
  reader ``distil whatif`` uses). Content never leaves this process; only aggregates are
  written anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Message = dict[str, Any]

#: Claude Code asks for the 1-hour TTL; the harness uses the default (5 minutes).
TTL_5M = "5m"
TTL_1H = "1h"


@dataclass
class Trajectory:
    id: str
    source: str  # "swebench-lite" | "swebench-long" | "claude-code" | "synthetic"
    arm: str  # the arm that produced the trajectory (its behaviour, not its bytes)
    model: str
    messages: list[Message]
    request_ends: list[int]
    #: Start time (s) of each request; spacing drives TTL expiry.
    times: list[float]
    #: Billed output tokens for the whole trajectory (policy-invariant in replay).
    output_tokens: int = 0
    #: Billed usage totals {input, cache_write, cache_read, output} when known.
    billed: dict[str, int] | None = None
    ttl: str = TTL_5M
    steps: int = 0
    solved: bool | None = None
    #: Tool definitions + system prompt, as tokens (fitted for the harness, see calibrate).
    overhead_tokens: int = 0
    #: thinking signature -> hidden thinking tokens (allocated by ``allocate_hidden``).
    hidden: dict[str, int] = field(default_factory=dict)

    def request(self, k: int) -> list[Message]:
        return self.messages[: self.request_ends[k]]


def block_text(block: Any) -> str:
    """The billable text of a content block (what the model reads, not the JSON wrapper)."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    t = block.get("type")
    if t == "text":
        return str(block.get("text", ""))
    if t == "tool_use":
        return str(block.get("name", "")) + json.dumps(block.get("input", {}))
    if t == "tool_result":
        c = block.get("content", "")
        if isinstance(c, list):
            return "\n".join(block_text(x) for x in c)
        return str(c)
    if t in ("thinking", "redacted_thinking"):
        return ""  # hidden: priced from the allocation, never from the bytes
    return json.dumps({k: v for k, v in block.items() if k != "cache_control"})


def blocks(msg: Message) -> list[Any]:
    """Content blocks; string content is the single text block the API treats it as."""
    c = msg.get("content")
    return [{"type": "text", "text": c}] if isinstance(c, str) else list(c or [])


def thinking_key(block: Any) -> str | None:
    if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"):
        return str(block.get("signature") or block.get("data") or "")
    return None


def _request_ends(messages: list[Message]) -> list[int]:
    """Every user message followed by an assistant reply ended one model call."""
    return [
        i + 1
        for i, m in enumerate(messages)
        if m.get("role") == "user" and i + 1 < len(messages)
        if messages[i + 1].get("role") == "assistant"
    ]


def load_harness(
    run_dir: Path, arms: tuple[str, ...], source: str, solved: dict[tuple[str, str], bool]
) -> list[Trajectory]:
    """Harness transcripts with their billed rows. Rows with no transcript are skipped."""
    rows = [json.loads(x) for x in (run_dir / "results.jsonl").read_text().splitlines() if x]
    out: list[Trajectory] = []
    for r in rows:
        if r["arm"] not in arms or r.get("failure_class") in (
            "api_error",
            "env_error",
            "internal_error",
        ):
            continue
        f = run_dir / "transcripts" / r["arm"] / f"{r['instance_id']}.json"
        if not f.exists():
            continue
        msgs = json.loads(f.read_text())
        ends = _request_ends(msgs)
        n = len(ends)
        if n == 0:
            continue
        gap = float(r.get("wall_s") or 0.0) / n
        out.append(
            Trajectory(
                id=r["instance_id"],
                source=source,
                arm=r["arm"],
                model=r["model"],
                messages=msgs,
                request_ends=ends,
                times=[k * gap for k in range(n)],
                output_tokens=int(r["usage"]["output"]),
                billed=dict(r["usage"]),
                steps=int(r["steps"]),
                solved=solved.get((r["arm"], r["instance_id"])),
            )
        )
    return out


def load_grades(run_dir: Path) -> dict[tuple[str, str], bool]:
    p = run_dir / "grades.jsonl"
    if not p.exists():
        return {}
    out: dict[tuple[str, str], bool] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            out[(g.get("arm", ""), g.get("instance_id", ""))] = g.get("status") == "resolved"
    return out


def load_claude_code(
    root: Path, limit: int, min_requests: int = 10, max_requests: int = 150
) -> list[Trajectory]:
    """Recent Claude Code sessions via the ``distil whatif`` reader (read-only).

    Output tokens are not reconstructed (set to 0): these trajectories compare policies on
    the input side only, which is the only side a context policy changes in replay.
    """
    from distil import pricing
    from distil.whatif import discover, parse_transcript

    out: list[Trajectory] = []
    for path in discover(root):
        if len(out) >= limit:
            break
        try:
            sess = parse_transcript(path.read_text(errors="replace").splitlines())
        except OSError:
            continue
        ends = _request_ends(sess.messages)[:max_requests]  # ponytail: replay is O(n^2)
        if len(ends) < min_requests or pricing.resolve(sess.model) is None:
            continue
        t0 = next((t for t in sess.ts if t), 0.0)
        times = [max(0.0, (sess.ts[e - 1] or t0) - t0) for e in ends]
        for k in range(1, len(times)):  # missing timestamps: keep monotone
            times[k] = max(times[k], times[k - 1])
        share_1h = sess.cache_write_1h / sess.cache_write if sess.cache_write else 1.0
        out.append(
            Trajectory(
                id=f"cc{len(out)}",  # never the session id or path
                source="claude-code",
                arm="claude-code",
                model=sess.model or "",
                messages=sess.messages[: ends[-1]],
                request_ends=ends,
                times=times,
                ttl=TTL_1H if share_1h >= 0.5 else TTL_5M,
                steps=len(ends),
            )
        )
    return out
