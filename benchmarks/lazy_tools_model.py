"""What would deferring unused MCP tool definitions save on real traffic?

Reads the proxy's own per-request records (``~/.distil/sessions/*.requests.jsonl``,
names + token counts + provider usage only) and prices two designs against them:

* **native** — Claude Code's own MCP tool search (``ENABLE_TOOL_SEARCH``), which
  Claude Code switches OFF whenever ``ANTHROPIC_BASE_URL`` is a non-first-party host,
  i.e. whenever distil is in the path. Deferred definitions leave the prompt; the
  API keeps the prefix stable (``tool_reference`` expands inline), so a discovery
  costs one extra round-trip, never a prefix rewrite.
* **distil_lazy** — a distil-side rewrite: hide never-called tools behind a
  ``distil_tools`` meta-tool. Same token removal, but every unlock changes the tools
  array, which sits at the TOP of the prefix, so each unlock re-writes the whole
  cached context at 1.25x instead of reading it at 0.10x.

Positional pricing: tool definitions precede system and messages, so a request's
MCP-definition tokens are charged against its cache read first, then its cache
write, then uncached input. Heuristic token counts are scaled to the provider's
billed input per request before pricing.

Discovery cost comes from the agent's own transcripts (``~/.claude/projects``),
tool NAMES only: each distinct MCP tool a transcript used is charged one discovery
(native) or one unlock (distil_lazy); the per-call rate is reported as a sensitivity.

    uv run --python 3.12 python benchmarks/lazy_tools_model.py \
        --window 2026-09-16T03:40:40+00:00 2026-09-24T20:34:27+00:00 \
        > benchmarks/results/2026-09-24/lazy_tools_model.json

MCP server names are replaced by ``server_01``, ``server_02``, ... (ranked by
definition tokens, then by calls) because the artifact is published and the names
say which accounts a machine has connected. ``--names`` prints the real names, for
local use only; never commit its output.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distil import pricing  # noqa: E402

SEARCH_OUTPUT_TOKENS = 300  # one ToolSearch call + its tool_reference reply, output side
NAME_INDEX_TOKENS_PER_TOOL = 15  # deferred tools are still listed by name


def _records(lo: float, hi: float) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for f in sorted(glob.glob(os.path.expanduser("~/.distil/sessions/*.requests.jsonl"))):
        sid = Path(f).name.split(".")[0]
        for line in open(f, encoding="utf-8", errors="ignore"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not lo <= float(r.get("ts") or 0) < hi:
                continue
            if r.get("status") == 200 and r.get("usage_input_tokens") is not None:
                out.append((sid, r))
    return out


def _ts(line: dict[str, Any]) -> float | None:
    raw = line.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _mcp_calls(since: float, until: float) -> tuple[int, int, int, Counter[str]]:
    """(assistant turns, MCP calls, distinct MCP tools per transcript, calls per server),
    counting only assistant lines timestamped inside the window."""
    turns = calls = distinct = 0
    per: Counter[str] = Counter()
    for f in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")):
        if os.path.getmtime(f) < since:
            continue
        seen: set[str] = set()
        for line in open(f, encoding="utf-8", errors="ignore"):
            if '"assistant"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "assistant":
                continue
            t = _ts(r)
            if t is None or not since <= t <= until:
                continue
            turns += 1
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    name = str(b.get("name", ""))
                    if name.startswith("mcp__"):
                        calls += 1
                        seen.add(name)
                        per["__".join(name.split("__")[:2])] += 1
        distinct += len(seen)
    return turns, calls, distinct, per


def _labels(by_tokens: Counter[str], by_calls: Counter[str]) -> dict[str, str]:
    """Stable anonymous label per server: token rank first, then call rank."""
    order = [s for s, _n in by_tokens.most_common()]
    order += [s for s, _n in by_calls.most_common() if s not in by_tokens]
    return {s: f"server_{i:02d}" for i, s in enumerate(order, 1)}


def _iso_ts(v: str) -> float:
    return datetime.fromisoformat(v).timestamp()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--window",
        nargs=2,
        metavar=("START", "END"),
        help="ISO-8601 bounds, END inclusive to the second; default = every record on disk",
    )
    ap.add_argument("--names", action="store_true", help="real MCP server names (local only)")
    args = ap.parse_args(argv)
    # END names a whole second (window_utc is printed at seconds precision), so the
    # bound is the start of the NEXT second — re-running on a printed window then
    # selects exactly the records it was printed from.
    lo, hi = (_iso_ts(args.window[0]), _iso_ts(args.window[1]) + 1) if args.window else (0.0, 1e12)
    recs = _records(lo, hi)
    if not recs:
        raise SystemExit("no priced request records in the window")
    since = min(r["ts"] for _s, r in recs)
    until = max(r["ts"] for _s, r in recs)
    turns, calls, distinct, per_server_calls = _mcp_calls(since, until)
    # A tool is discovered (native) or unlocked (distil_lazy) once per session and then
    # stays loaded, so the event rate is distinct tools per transcript, not calls.
    # Per-call is reported alongside as the pessimistic sensitivity.
    call_rate = distinct / turns if turns else 0.0
    per_call_rate = calls / turns if turns else 0.0

    bill = mcp_lo_usd = mcp_hi_usd = index_usd = 0.0
    search_usd = lazy_rewrite_usd = 0.0
    mcp_lo_tok = mcp_hi_tok = 0
    requests = priced = 0
    per_server_tok: Counter[str] = Counter()
    sessions_with_mcp: set[str] = set()
    for sid, r in recs:
        p = pricing.resolve(r.get("model"))
        requests += 1
        if p is None:
            continue
        priced += 1
        unc = int(r.get("usage_input_tokens") or 0)
        rd = int(r.get("usage_cache_read") or 0)
        wr = int(r.get("usage_cache_create") or 0)
        out = int(r.get("usage_output_tokens") or 0)
        billed_in = unc + rd + wr
        bill += unc * p.input + rd * p.cache_read + wr * p.cache_write + out * p.output

        tools = r.get("tools") or []
        top = sum(int(t.get("tokens") or 0) for t in tools)
        lo = sum(int(t.get("tokens") or 0) for t in tools if str(t.get("name")).startswith("mcp__"))
        # The record keeps the 24 largest definitions; anything past that is unnamed.
        rest = max(0, int(r.get("tools_tokens") or 0) - top) if len(tools) >= 24 else 0
        hi = lo + rest
        if not lo:
            continue
        sessions_with_mcp.add(sid)
        for t in tools:
            n = str(t.get("name"))
            if n.startswith("mcp__"):
                per_server_tok["__".join(n.split("__")[:2])] += int(t.get("tokens") or 0)
        est = int(r.get("overhead_tokens") or 0) + max(
            0, int(r.get("compressible_tokens") or 0) - int(r.get("tokens_saved") or 0)
        )
        scale = min(1.0, billed_in / est) if est else 1.0

        def positional(tok: float, rd: int = rd, wr: int = wr, unc: int = unc) -> float:
            a = min(tok, rd)
            b = min(tok - a, wr)
            c = min(tok - a - b, unc)
            return a * p.cache_read + b * p.cache_write + c * p.input

        lo_s, hi_s = lo * scale, hi * scale
        mcp_lo_tok += int(lo_s)
        mcp_hi_tok += int(hi_s)
        mcp_lo_usd += positional(lo_s)
        mcp_hi_usd += positional(hi_s)
        avg_tool = lo / max(1, sum(1 for t in tools if str(t.get("name")).startswith("mcp__")))
        index_usd += positional(hi_s / max(avg_tool, 1.0) * NAME_INDEX_TOKENS_PER_TOOL)
        # Expected extra round-trips on this request = call_rate. Native: one more
        # request reading the (now smaller) context from cache. distil_lazy: same, plus
        # the unlock rewrites the whole cached context (write instead of read).
        ctx_after = max(0.0, billed_in - hi_s)
        search_usd += call_rate * (ctx_after * p.cache_read + SEARCH_OUTPUT_TOKENS * p.output)
        lazy_rewrite_usd += call_rate * ctx_after * (p.cache_write - p.cache_read)

    def pct(x: float) -> float:
        return round(100.0 * x / bill, 3) if bill else 0.0

    native_lo = mcp_lo_usd - index_usd - search_usd
    native_hi = mcp_hi_usd - index_usd - search_usd
    lazy_lo = native_lo - lazy_rewrite_usd
    lazy_hi = native_hi - lazy_rewrite_usd
    labels = _labels(per_server_tok, per_server_calls)

    def name(s: str) -> str:
        return s if args.names else labels[s]

    result = {
        "source": "~/.distil/sessions/*.requests.jsonl (names+token counts+usage) and "
        "~/.claude/projects/*/*.jsonl (tool_use NAMES only, assistant lines in the window)",
        "window_utc": [
            datetime.fromtimestamp(since, tz=timezone.utc).isoformat(timespec="seconds"),
            datetime.fromtimestamp(until, tz=timezone.utc).isoformat(timespec="seconds"),
        ],
        "dollars": "list-price (distil.pricing); notional on a flat-rate plan",
        "requests": requests,
        "requests_priced": priced,
        "sessions_with_mcp_definitions": len(sessions_with_mcp),
        "billed_usd": round(bill, 2),
        "mcp_definition_tokens_billed": {"low": mcp_lo_tok, "high": mcp_hi_tok},
        "mcp_definition_usd": {"low": round(mcp_lo_usd, 2), "high": round(mcp_hi_usd, 2)},
        "mcp_definition_share_of_bill_pct": {"low": pct(mcp_lo_usd), "high": pct(mcp_hi_usd)},
        "server_names": "real" if args.names else "anonymized (server_NN, ranked by tokens)",
        "mcp_tokens_by_server_named": {name(s): n for s, n in per_server_tok.most_common(12)},
        "transcripts": {
            "assistant_turns": turns,
            "mcp_calls": calls,
            "distinct_mcp_tools_per_transcript_summed": distinct,
            "discovery_events_per_turn": round(call_rate, 5),
            "mcp_calls_per_turn": round(per_call_rate, 5),
            "calls_by_server": {name(s): n for s, n in per_server_calls.most_common(12)},
        },
        "costs_usd": {
            "name_index": round(index_usd, 2),
            "discovery_round_trips": round(search_usd, 2),
            "distil_lazy_prefix_rewrites": round(lazy_rewrite_usd, 2),
            "per_call_sensitivity_x": round(per_call_rate / call_rate, 2) if call_rate else None,
        },
        "net_saving_pct_of_bill": {
            "native_tool_search": {"low": pct(native_lo), "high": pct(native_hi)},
            "distil_lazy_tools": {"low": pct(lazy_lo), "high": pct(lazy_hi)},
        },
        "net_saving_usd": {
            "native_tool_search": {"low": round(native_lo, 2), "high": round(native_hi, 2)},
            "distil_lazy_tools": {"low": round(lazy_lo, 2), "high": round(lazy_hi, 2)},
        },
        "assumptions": {
            "search_output_tokens": SEARCH_OUTPUT_TOKENS,
            "name_index_tokens_per_tool": NAME_INDEX_TOKENS_PER_TOOL,
            "discovery": "one extra round-trip per distinct MCP tool per transcript, spread "
            "over every MCP-carrying request; multiply discovery and rewrite costs by "
            "per_call_sensitivity_x for the one-event-per-call worst case",
            "low_vs_high": "low counts only named (top-24) mcp__ definitions per request; "
            "high also counts the unnamed remainder past the 24 largest",
        },
    }
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
