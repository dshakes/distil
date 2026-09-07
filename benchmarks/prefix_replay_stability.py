#!/usr/bin/env python3
"""prefix_replay_stability.py — how much forwarded prefix survives a client rewrite.

ADR 0011. Offline, deterministic, no network and no API key: it replays a recorded
multi-turn session shape through the same public adapter functions the proxy calls,
under the three ways agentic clients rewrite their own history without changing a token
the model reads, and reports the one number that bills — how many LEADING messages went
out byte-identical to the previous turn.

Why that number and not "how many messages changed": a provider cache entry covers a
byte PREFIX. Everything after the first changed byte is already uncached, so changing it
again is free. Counting total drift would score the volatile tail distil rewrites on
purpose (ADR 0008 clause (d)) as a regression, and would hide the only thing that costs
money — the prefix getting shorter.

This measures distil's own forwarding, not the provider's cache. It cannot say what the
bill did; ``distil dissect`` reports the provider's cache-read share from real traffic,
and that is where live confirmation comes from.

    python benchmarks/prefix_replay_stability.py [--turns 8] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil import prefixreplay  # noqa: E402
from distil.adapters.anthropic import compress_messages  # noqa: E402
from distil.adapters.gemini import compress_generate_request  # noqa: E402
from distil.adapters.openai import (  # noqa: E402
    compress_chat_completions,
    compress_responses_input,
)


def _wire(obj: Any) -> str:
    """The proxy's encoder (``proxy._serialize_if_changed``). Not ``sort_keys``: key
    order is part of the prefix the provider hashes."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _bare(node: Any) -> Any:
    """Same payload with every ``cache_control`` removed.

    The marker tells the provider where to cut the cached span; it is not part of the
    span, and Anthropic does not hash it into the entry. Leaving it in would score the
    one movement that is supposed to happen — the breakpoint advancing with the
    conversation — as drift, and every healthy session would read 85% instead of 100%.
    """
    if isinstance(node, dict):
        return {k: _bare(v) for k, v in node.items() if k != "cache_control"}
    if isinstance(node, list):
        return [_bare(x) for x in node]
    return node


def _log(n: int, tag: str) -> str:
    return "\n".join(
        f"2026-09-04 10:00:{i:02d} INFO {tag} worker-{i} handled request id={i} ok"
        for i in range(n)
    )


# --------------------------------------------------------------------------- shapes


def _anthropic(turns: int) -> list[dict[str, Any]]:
    """The Claude Code shape: the newest block is pinned, so the whole history caches."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "kick off the run"}]}
    ]
    for t in range(turns):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"t{t}",
                        "name": "bash",
                        "input": {"cmd": f"run {t}"},
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{t}", "content": _log(40, f"step{t}")}
                ],
            }
        )
    msgs[-1]["content"][0]["cache_control"] = {"type": "ephemeral"}
    return msgs


def _openai_chat(turns: int) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "you are a build agent"}]
    for t in range(turns):
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{t}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{t}", "content": _log(40, f"step{t}")})
    return msgs


def _openai_responses(turns: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}
    ]
    for t in range(turns):
        items.append(
            {"type": "function_call", "call_id": f"c{t}", "name": "bash", "arguments": "{}"}
        )
        items.append(
            {"type": "function_call_output", "call_id": f"c{t}", "output": _log(40, f"step{t}")}
        )
    return items


def _gemini(turns: int) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": "go"}]}]
    for t in range(turns):
        contents.append(
            {"role": "model", "parts": [{"functionCall": {"name": "bash", "args": {"cmd": t}}}]}
        )
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "bash",
                            "response": {"stdout": _log(40, f"step{t}")},
                        }
                    }
                ],
            }
        )
    return contents


SHAPES: list[tuple[str, Callable[[int], list[dict[str, Any]]], Callable[[list], list]]] = [
    ("anthropic/messages", _anthropic, lambda m: compress_messages(m)[0]),
    ("openai/chat-completions", _openai_chat, lambda m: compress_chat_completions(m)[0]),
    ("openai/responses", _openai_responses, lambda m: compress_responses_input(m)[0]),
    (
        "gemini/generateContent",
        _gemini,
        lambda c: compress_generate_request({"contents": c})[0]["contents"],
    ),
]


# ------------------------------------------------------------------- client rewrites


def _clone(x: Any) -> Any:
    return json.loads(_wire(x))


def _no_churn(items: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    return items


def _churn_marker(items: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    """The cache_control breakpoint advances to the newest block every turn."""
    out = [_bare(_clone(m)) for m in items]
    if out:
        blocks = out[-1].get("content")
        if isinstance(blocks, list) and blocks and isinstance(blocks[-1], dict):
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
        else:
            out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def _churn_index(items: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    """An SDK shim stamps positional `index` fields, renumbered every turn."""
    out = [_clone(m) for m in items]
    for m in out:
        for key in ("content", "parts", "tool_calls"):
            for j, b in enumerate(m.get(key) or []):
                if isinstance(b, dict):
                    b["index"] = j + turn
    return out


def _churn_sugar(items: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    """String content and a single text block, flipped on alternating turns."""
    to_list = turn % 2 == 0

    def flip(node: Any) -> Any:
        if isinstance(node, list):
            return [flip(x) for x in node]
        if not isinstance(node, dict):
            return node
        out = {k: flip(v) for k, v in node.items()}
        c = out.get("content")
        if to_list and isinstance(c, str):
            out["content"] = [{"type": "text", "text": c}]
        elif (
            not to_list
            and isinstance(c, list)
            and len(c) == 1
            and isinstance(c[0], dict)
            and set(c[0]) == {"type", "text"}
        ):
            out["content"] = c[0]["text"]
        return out

    return [flip(_clone(m)) for m in items]


CHURNS = [
    ("none (control)", _no_churn),
    ("marker advances", _churn_marker),
    ("index stamped", _churn_index),
    ("string/block sugar", _churn_sugar),
]


# ------------------------------------------------------------------------ measurement


def _run(build, forward, churn, turns: int, *, replay: bool) -> tuple[float, float]:
    """Mean stable-prefix share across turns, and the added milliseconds per request.

    The share is normalised by the number of messages the client re-sent, so a longer
    session does not flatter it.
    """
    prefixreplay.reset()
    key = f"{id(build)}/{id(churn)}/{replay}"
    prev: list[str] | None = None
    shares: list[float] = []
    overhead = 0.0
    for t in range(1, turns + 1):
        items = churn(build(t), t)
        sent = forward(items)
        if replay:
            t0 = time.perf_counter()
            sent, _stats = prefixreplay.replay(key, items, sent)
            overhead += time.perf_counter() - t0
        cur = [_wire(_bare(m)) for m in sent]
        if prev is not None:
            resent = min(len(prev), len(cur))
            n = 0
            while n < resent and prev[n] == cur[n]:
                n += 1
            shares.append(n / resent if resent else 1.0)
        prev = cur
    prefixreplay.reset()
    return (sum(shares) / len(shares) if shares else 0.0), 1000.0 * overhead / turns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turns", type=int, default=8, help="tool round-trips to replay")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for shape, build, forward in SHAPES:
        for cname, churn in CHURNS:
            off, _ = _run(build, forward, churn, args.turns, replay=False)
            on, ms = _run(build, forward, churn, args.turns, replay=True)
            rows.append(
                {
                    "shape": shape,
                    "rewrite": cname,
                    "stable_prefix_before": round(off, 4),
                    "stable_prefix_after": round(on, 4),
                    "replay_ms_per_request": round(ms, 3),
                }
            )

    if args.json:
        print(json.dumps({"turns": args.turns, "rows": rows}, indent=2))
        return 0

    print(f"forwarded-prefix stability — {args.turns} turns, offline, no API calls\n")
    print(f"{'shape':<24} {'client rewrite':<20} {'before':>8} {'after':>8}  {'+ms/req':>8}")
    print("-" * 74)
    for r in rows:
        print(
            f"{r['shape']:<24} {r['rewrite']:<20} "
            f"{r['stable_prefix_before']:>7.1%} {r['stable_prefix_after']:>8.1%} "
            f"{r['replay_ms_per_request']:>8.2f}"
        )
    print(
        "\nShare of re-sent messages forwarded byte-identical to the previous turn — the"
        "\nspan a provider can serve from its prompt cache. 'before' is distil without"
        "\nreplay; 'after' is with it. The control row must read the same in both columns:"
        "\na client that does not rewrite its history had nothing to repair. Compared with"
        "\nthe cache_control marker excluded — it is the breakpoint, not the content, and"
        "\nit is meant to advance."
        "\n\nThis measures what distil forwards, not what the provider billed. Live"
        "\nconfirmation is the cache-read share in `distil dissect`, after release."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
