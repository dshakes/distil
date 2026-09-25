"""What do request ``tools`` definitions actually cost, on real traffic?

Reads the per-request records ``distil wrap`` already writes
(``~/.distil/sessions/*.requests.jsonl``, read-only) and answers, before anyone
builds a tool-schema compressor:

* what share of billed input tokens are tool definitions, and
* what share of billed DOLLARS they are once prompt-cache pricing is applied.

Tools are the first element of the cache prefix (tools -> system -> messages), so
on a cache hit they bill at the cache-read rate (0.1x input). This script
allocates each request's tool tokens to the provider's own ``cache_read`` /
``cache_creation`` / uncached counts in prefix order, prices them with
:mod:`distil.pricing`, and reports the expected value of compressing them by X%.

Token counts in the records are distil's heuristic estimates; each request's tool
share is applied to the provider-billed input total, so the absolute counts are
on the billed scale. Content-free: only counts and tool names are read.

With ``--tools FILE`` (a JSON file holding a real request's ``tools`` array, or a
JSONL whose first line has a ``tools`` key) it also measures the LOSSLESS headroom:
how much of that array a schema compactor could remove without changing what the
model can call. Only two transforms qualify: dropping the ``$schema`` dialect URI,
and dropping a property ``title`` that merely restates the property's name.
``additionalProperties: false`` does NOT qualify — strict tool use requires it —
and descriptions, enums, defaults and bounds all change model behaviour.

Usage::

    uv run python benchmarks/tool_schema_share.py --tools captured.jsonl \
        --out benchmarks/results/tool_schema_share.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from pathlib import Path
from typing import Any

from distil import pricing
from distil.tokenizer import HeuristicTokenizer


def _records(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(root / "sessions" / "*.requests.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != 200 or r.get("usage_input_tokens") is None:
                    continue
                out.append(r)
    return out


def _price(model: str | None) -> pricing.Pricing:
    # ponytail: unknown/None model priced as opus; only the RATIO matters here and
    # every Anthropic model shares the same 1.25x/0.1x cache multipliers.
    # (Pricing properties are per-token USD.)
    return pricing.resolve(model) or pricing.get("claude-opus-4-8")


def _compact(o: Any) -> Any:
    """The provably-lossless compaction: ``$schema`` and name-restating ``title``s."""
    if isinstance(o, list):
        return [_compact(v) for v in o]
    if not isinstance(o, dict):
        return o
    out = {k: _compact(v) for k, v in o.items() if k != "$schema"}
    props = out.get("properties")
    if isinstance(props, dict):
        for name, sub in props.items():
            title = sub.get("title") if isinstance(sub, dict) else None
            if isinstance(title, str) and _norm(title) == _norm(name):
                del sub["title"]
    return out


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def lossless_headroom(tools: list[dict[str, Any]]) -> dict[str, Any]:
    tk = HeuristicTokenizer()

    def count(o: Any) -> int:
        return tk.count(json.dumps(o, separators=(",", ":"), ensure_ascii=False))

    after = [{**t, "input_schema": _compact(t.get("input_schema"))} for t in tools]
    before_n, after_n = count(tools), count(after)
    return {
        "tools": len(tools),
        "tool_tokens": before_n,
        "tool_tokens_after_lossless_compaction": after_n,
        "lossless_reduction": round((before_n - after_n) / before_n, 4) if before_n else 0.0,
    }


def _load_tools(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        d = json.loads(text.splitlines()[0])  # JSONL: first captured request
    tools = d.get("tools") if isinstance(d, dict) else d
    if not isinstance(tools, list):
        raise ValueError(f"{path}: no tools array found")
    return tools


def measure(root: Path) -> dict[str, Any]:
    recs = _records(root)
    tot_in = tot_tools = 0.0
    tools_read = tools_write = tools_uncached = 0.0
    usd_total = usd_input = usd_tools = 0.0
    with_tools = 0
    shares: list[float] = []
    for r in recs:
        u = int(r.get("usage_input_tokens") or 0)
        cr = int(r.get("usage_cache_read") or 0)
        cw = int(r.get("usage_cache_create") or 0)
        out = int(r.get("usage_output_tokens") or 0)
        billed = u + cr + cw
        est = int(r.get("overhead_tokens") or 0) + max(
            0, int(r.get("compressible_tokens") or 0) - int(r.get("tokens_saved") or 0)
        )
        p = _price(r.get("model"))
        in_usd = u * p.input + cw * p.cache_write + cr * p.cache_read
        usd_input += in_usd
        usd_total += in_usd + out * p.output
        tot_in += billed
        t_est = int(r.get("tools_tokens") or 0)
        if not t_est or not est or not billed:
            continue
        with_tools += 1
        share = min(1.0, t_est / est)
        shares.append(share)
        t = share * billed
        tot_tools += t
        # Tools lead the prefix, so they are the first tokens a cache read covers.
        rd = min(t, cr)
        wr = min(t - rd, cw)
        un = t - rd - wr
        tools_read += rd
        tools_write += wr
        tools_uncached += un
        usd_tools += rd * p.cache_read + wr * p.cache_write + un * p.input
    tool_cost_share = usd_tools / usd_total if usd_total else 0.0
    return {
        "source": "~/.distil/sessions/*.requests.jsonl (status 200, provider usage present)",
        "requests": len(recs),
        "requests_with_tools": with_tools,
        "billed_input_tokens": round(tot_in),
        "tool_tokens_billed_scale": round(tot_tools),
        "tool_share_of_input_tokens": round(tot_tools / tot_in, 4) if tot_in else 0.0,
        "tool_share_per_request_median": round(statistics.median(shares), 4) if shares else 0.0,
        "tool_tokens_cache_read_fraction": round(tools_read / tot_tools, 4) if tot_tools else 0.0,
        "tool_tokens_cache_write_fraction": round(tools_write / tot_tools, 4) if tot_tools else 0.0,
        "tool_tokens_uncached_fraction": round(tools_uncached / tot_tools, 4) if tot_tools else 0.0,
        "usd_total": round(usd_total, 2),
        "usd_input": round(usd_input, 2),
        "usd_tools": round(usd_tools, 2),
        "tool_share_of_billed_usd": round(tool_cost_share, 4),
        # EV of compressing tool definitions by X%, assuming the rewrite is byte-stable
        # (a non-deterministic one busts the prefix cache and is net NEGATIVE).
        "ev_share_of_billed_usd_at_x": {
            f"{x}%": round(tool_cost_share * x / 100, 4) for x in (5, 10, 20, 30)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.expanduser("~/.distil"))
    ap.add_argument("--tools", default=None, help="captured request tools array (JSON/JSONL)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = measure(Path(args.root))
    if args.tools:
        head = lossless_headroom(_load_tools(Path(args.tools)))
        res["lossless_headroom"] = head
        # The number that decides whether to build it: share of the whole bill a
        # perfect, byte-stable lossless compactor would save.
        res["ev_share_of_billed_usd_lossless"] = round(
            res["tool_share_of_billed_usd"] * head["lossless_reduction"], 4
        )
    text = json.dumps(res, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
