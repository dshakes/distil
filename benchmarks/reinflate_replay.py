"""Offline replay for ADR 0025: does forwarding re-fetches verbatim make re-reads avoidable?

Replays recorded SWE-bench outcome transcripts (``benchmarks/swebench_outcome``, distil arm)
request by request, exactly as the harness sent them: every prefix that ends in a user
turn, with an ephemeral cache breakpoint on the newest block
(``agent.with_cache_breakpoint``), through ``compress_messages(persist=False)``. No API
calls; the agent's actions are fixed, so this measures what the agent would have *seen*,
not what it would have done.

Variants:

* ``baseline``  — the adapter with the re-fetch rule off (``refetch=False``).
* ``refetch``   — the adapter as shipped (``refetch=True``, ADR 0025).
* ``sticky``    — rejected: (a) that, once one re-fetch is seen, digests nothing new.
* ``sequence``  — rejected "likely-needed" retention: a sequence whose every stage is a
  plain read, a search or a ``cd`` (``sed -n 1,80p a; grep x b``) kept verbatim forever.
* ``compound``  — rejected, looser: any command with a reader stage anywhere
  (``cat f | sed -n 1,80p; python t.py``) kept verbatim forever.

The rejected variants are prototypes patched in here, never shipped.

The key metric is per tool call the agent actually made. A call is **redundant** when at
least 80% of its result's lines (``refetch.line_keys``: line-number prefixes stripped) were
already in an earlier tool result as the client sent it, and **visible** when they were
already in what distil forwarded in the request the agent was answering. A redundant call
that was not visible is **avoidable**: the content was in the conversation, but only as a
digest, and the agent fetched it again instead of calling ``distil_expand``.

Also reported: request-size savings (``proxy._count_messages`` over every request), final-
request digests, cache-aware input cost (a message forwarded byte-identical to the previous
request's is read at 0.1x, anything else written at 1.25x; system and tools excluded,
identical across variants), and prefix rewrites (a message already forwarded that changed
bytes on a later request — the thing ADR 0008 forbids).

Usage::

    uv run python benchmarks/reinflate_replay.py DIR [DIR ...] --out results.json
    uv run python benchmarks/reinflate_replay.py --claude-code ~/.claude/projects \\
        --sessions 200 --out claude-code.json

Each DIR holds ``transcripts/distil/*.json`` (a JSON list of Messages-API messages).
``--claude-code`` reads local Claude Code transcripts instead (read-only; baseline and
shipped variants only; aggregate counts only, see ``run_claude_code``).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.swebench_outcome.agent import with_cache_breakpoint  # noqa: E402
from distil.adapters import anthropic as A  # noqa: E402
from distil.compress import provenance as P  # noqa: E402
from distil.compress.refetch import Tracker, line_keys  # noqa: E402
from distil.proxy import _count_messages  # noqa: E402

VARIANTS = ("baseline", "refetch", "sticky", "sequence", "compound")

# A call is redundant / visible at this share of its lines. Fixed here, not imported from
# the rule, so tuning the rule cannot move the yardstick it is graded by.
REDUNDANT = 0.8


@contextlib.contextmanager
def _compound_reads(strict: bool = False) -> Iterator[None]:
    """Prototypes of the rejected alternative. Loose (``compound``): a reader stage anywhere
    exempts the result. Strict (``sequence``): only a sequence whose every stage is itself
    exempt — a plain read, a search, or a ``cd``."""
    original = P.exact_quote_ids

    def patched(calls: Any, **kw: Any) -> dict[str, str]:
        calls = list(calls)
        keep = original(calls, **kw)
        for call in calls:
            if call.id in keep or not call.command:
                continue
            norm = call.command.replace("||", ";").replace("&&", ";").replace("\n", ";")
            stages = [s for s in norm.split(";") if s.strip()]
            if strict:
                hit = all(
                    s.split()[0] == "cd"
                    or P.is_shell_search(s)
                    or ("|" not in s and P._stage_paths(s))
                    for s in stages
                )
            else:
                hit = any(P._stage_paths(s.split("|", 1)[0]) for s in stages)
            if hit:
                keep[call.id] = "tool_result_compound_read"
        return keep

    P.exact_quote_ids = patched
    try:
        yield
    finally:
        P.exact_quote_ids = original


@contextlib.contextmanager
def _sticky() -> Iterator[None]:
    """Prototype of (a') — once a re-fetch has fired in a pass, every later would-be digest
    goes out verbatim too. Still prefix-deterministic: the trip depends only on the prefix."""
    original = Tracker.is_refetch

    def patched(self: Tracker, text: str) -> bool:
        if getattr(self, "_tripped", False):
            return True
        hit = original(self, text)
        self._tripped = hit  # type: ignore[attr-defined]
        return hit

    Tracker.is_refetch = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        Tracker.is_refetch = original  # type: ignore[method-assign]


def _strip_cc(msg: Any) -> str:
    """A message's bytes with the moving cache marker removed — what must stay stable.

    String content is canonicalised to one text block: the harness turns the newest
    message's string into a block to carry the marker, and the proxy forwards a
    canonically-equal prefix as previously sent (ADR 0011), so that is not a rewrite.
    """
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        msg = {**msg, "content": [{"type": "text", "text": msg["content"]}]}
    if isinstance(msg, dict) and isinstance(msg.get("content"), list):
        msg = {
            **msg,
            "content": [
                {k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b
                for b in msg["content"]
            ],
        }
    return json.dumps(msg, sort_keys=True)


def _result_lines(messages: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                out |= line_keys(A._tool_result_text(b.get("content")))
    return out


def _results_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    for m in messages:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out[str(b.get("tool_use_id"))] = A._tool_result_text(b.get("content"))
    return out


def _first_word(call: dict[str, Any]) -> str:
    cmd = P.command_text(call.get("input")) or str(call.get("name", ""))
    for stage in cmd.replace("&&", ";").replace("\n", ";").split(";"):
        words = stage.split()
        if words and words[0] != "cd":
            return words[0] + (" |" if "|" in cmd else "")
    return "?"


def replay(path: str, variant: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return replay_messages(json.load(fh), variant)


def replay_messages(msgs: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    results = _results_by_id(msgs)
    c: Counter[str] = Counter()
    by_cmd: Counter[str] = Counter()
    census: Counter[str] = Counter()
    prev: list[str] = []
    final: list[dict[str, Any]] = []
    refetch = variant in ("refetch", "sticky")
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or i == 0:
            continue
        prefix = msgs[:i]
        send = with_cache_breakpoint(prefix)
        out, _ = A.compress_messages(send, persist=False, refetch=refetch)
        census.update(A.take_census() or {})
        final = out
        c["requests"] += 1
        c["raw_tokens"] += _count_messages(send)
        c["sent_tokens"] += _count_messages(out)
        # Cache-aware input cost, in token-units: unchanged leading messages read at 0.1x.
        cur = [_strip_cc(x) for x in out]
        raw_cur = [_strip_cc(x) for x in send]
        same = 0
        while same < min(len(prev), len(cur)) and prev[same] == cur[same]:
            same += 1
        c["prefix_rewrites"] += int(same < len(prev))
        c["cost_units"] += 0.1 * _count_messages(out[:same]) + 1.25 * _count_messages(out[same:])
        # The plain arm's cost on the same requests: history is never rewritten.
        c["raw_cost_units"] += 0.1 * _count_messages(send[: len(raw_cur) - 2]) + 1.25 * (
            _count_messages(send[len(raw_cur) - 2 :])
        )
        prev = cur
        sent_lines, seen_lines = _result_lines(prefix), _result_lines(out)
        for b in m.get("content") or []:
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            keys = line_keys(results.get(str(b.get("id")), ""))
            if len(keys) < 3:
                continue
            if P._is_expand_call(str(b.get("name", ""))):
                c["expand_calls"] += 1
                continue
            c["calls"] += 1
            redundant = len(keys & sent_lines) >= REDUNDANT * len(keys)
            visible = len(keys & seen_lines) >= REDUNDANT * len(keys)
            c["redundant"] += redundant
            c["visible"] += redundant and visible
            if redundant and not visible:
                c["avoidable"] += 1
                by_cmd[_first_word(b)] += 1
    raw_ids = {k for k, v in results.items() if "handle=" not in v}
    for m in final:
        content = m.get("content") if isinstance(m, dict) else None
        for b in content if isinstance(content, list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                c["final_results"] += 1
                tid = str(b.get("tool_use_id"))
                text = A._tool_result_text(b.get("content"))
                c["final_digested"] += tid in raw_ids and "handle=" in text
    return {"counts": c, "by_cmd": by_cmd, "census": census}


def run(dirs: list[str], limit: int | None = None, arm: str = "distil") -> dict[str, Any]:
    pattern = f"transcripts/{arm}/*.json"
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, pattern)))
    if limit:
        files = files[:limit]
    out: dict[str, Any] = {"transcripts": len(files), "arm": arm, "variants": {}}
    for variant in VARIANTS:
        totals: Counter[str] = Counter()
        by_cmd: Counter[str] = Counter()
        census: Counter[str] = Counter()
        ctx: Any = contextlib.nullcontext()
        if variant in ("compound", "sequence"):
            ctx = _compound_reads(strict=variant == "sequence")
        elif variant == "sticky":
            ctx = _sticky()
        with ctx:
            for f in files:
                r = replay(f, variant)
                totals.update(r["counts"])
                by_cmd.update(r["by_cmd"])
                census.update(r["census"])
        t = dict(totals)
        t["savings_pct"] = round(100 * (1 - t["sent_tokens"] / t["raw_tokens"]), 2)
        t["cache_aware_savings_pct"] = round(100 * (1 - t["cost_units"] / t["raw_cost_units"]), 2)
        t["cost_units"] = round(t["cost_units"])
        t["raw_cost_units"] = round(t["raw_cost_units"])
        out["variants"][variant] = {
            "totals": t,
            "avoidable_by_command": dict(by_cmd.most_common()),
            "census_tokens": {
                k: v
                for k, v in sorted(census.items())
                if k.startswith("tool_result_") and k != "tool_result_short"
            },
        }
    return out


def run_claude_code(
    root: Path, sessions: int, seed: int = 0, max_messages: int = 160
) -> dict[str, Any]:
    """The same metric over local Claude Code transcripts (read-only), baseline vs shipped.

    Private input, so the output is counts and percentages only: no command words (the
    SWE-bench ``avoidable_by_command`` table is dropped), no paths, no text. Each session is
    cut to its first *max_messages* messages — the replay is quadratic in session length.
    """
    import random

    from distil.whatif import discover, parse_transcript

    files = discover(root.expanduser())
    chosen = sorted(random.Random(seed).sample(files, min(sessions, len(files))))
    sessions_msgs = []
    for f in chosen:
        with f.open(encoding="utf-8", errors="replace") as fh:
            msgs = parse_transcript(fh).messages[:max_messages]
        if any(m["role"] == "assistant" for m in msgs[1:]):
            sessions_msgs.append(msgs)
    out: dict[str, Any] = {
        "source": "local Claude Code transcripts (read-only), aggregates only",
        "sessions_found": len(files),
        "sessions_replayed": len(sessions_msgs),
        "seed": seed,
        "max_messages_per_session": max_messages,
        "variants": {},
    }
    for variant in ("baseline", "refetch"):
        totals: Counter[str] = Counter()
        census: Counter[str] = Counter()
        for msgs in sessions_msgs:
            try:
                r = replay_messages(msgs, variant)
            except Exception:  # noqa: BLE001 — a malformed session is skipped, counted
                totals["sessions_failed"] += 1
                continue
            totals.update(r["counts"])
            census.update(r["census"])
        t = dict(totals)
        t["savings_pct"] = round(100 * (1 - t["sent_tokens"] / t["raw_tokens"]), 2)
        t["cache_aware_savings_pct"] = round(100 * (1 - t["cost_units"] / t["raw_cost_units"]), 2)
        t["cost_units"] = round(t["cost_units"])
        t["raw_cost_units"] = round(t["raw_cost_units"])
        out["variants"][variant] = {
            "totals": t,
            "census_tokens": {
                k: v
                for k, v in sorted(census.items())
                if k.startswith("tool_result_") and k != "tool_result_short"
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arm", default="distil", help="transcripts/<arm>/ to replay")
    ap.add_argument("--claude-code", type=Path, default=None, metavar="ROOT")
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.claude_code is not None:
        result = run_claude_code(args.claude_code, args.sessions, args.seed)
    elif args.dirs:
        result = run(args.dirs, args.limit, args.arm)
    else:
        ap.error("give transcript DIRs or --claude-code ROOT")
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    for name, v in result["variants"].items():
        t = v["totals"]
        print(
            f"{name:9} avoidable={t.get('avoidable', 0)}/{t.get('redundant', 0)} "
            f"savings={t['savings_pct']}% cache-aware={t['cache_aware_savings_pct']}% "
            f"digested={t.get('final_digested', 0)}/{t.get('final_results', 0)} "
            f"rewrites={t.get('prefix_rewrites', 0)}"
        )


if __name__ == "__main__":
    main()
