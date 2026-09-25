"""In-window prefix breaks: who re-wrote a prefix the provider still had cached, and what it cost.

Two content-free measurements, both offline, no API calls:

1. **Ledger** (``~/.distil/sessions/*.requests.jsonl``, read-only). Pairs each request with
   the latest earlier request of the same lineage (session, model, system/tools size)
   inside the 5-minute cache TTL whose history it extends. The provider already held that
   predecessor's whole input, so any of it billed as a write instead of a read is an
   in-window break; its cost is priced at ``cache_write - cache_read`` per token. Breaks
   are split by where prefix replay (ADR 0011) stopped. Records written before 1.55 carry
   no ``replay_stop``, so on them "diverged" cannot say whose rewrite it was — reported as
   unattributed rather than guessed.

2. **Transcript replay** (``~/.claude/projects/*/*.jsonl``, read-only). Rebuilds each
   Claude Code request from the transcript as an append-only history — the client never
   rewrites anything — marks it the way Claude Code does, and drives it through
   ``compress_messages`` + ``prefixreplay``. Any forwarded-prefix change is therefore
   distil's by construction. Also counts how often the quote guard's widened pass ran and
   how often it rescued a quote.

    python benchmarks/in_window_prefix_breaks.py [--transcripts 12] [--out PATH]

The pre-fix artifact beside it is this same script run against ``git archive origin/main
distil`` at the commit before the quote-guard change, to show the transcript half moving.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distil import pricing  # noqa: E402

TTL_S = 300
SLACK = 2000  # tokens: breakpoint placement jitter, not a break


def _ledger_rows() -> list[dict[str, Any]]:
    rows = []
    for f in glob.glob(os.path.expanduser("~/.distil/sessions/*.requests.jsonl")):
        sid = os.path.basename(f).split(".")[0]
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("status") != 200 or d.get("usage_cache_create") is None:
                    continue
                if "replay_hits" not in d:
                    continue
                d["_sid"] = sid
                d["_tot"] = sum(
                    int(d.get(k) or 0)
                    for k in ("usage_input_tokens", "usage_cache_create", "usage_cache_read")
                )
                d["_n"] = d["replay_hits"] + d["replay_misses"]
                rows.append(d)
    return rows


def ledger() -> dict[str, Any]:
    rows = _ledger_rows()
    lin: dict[tuple, list] = defaultdict(list)
    total_usd = 0.0
    for d in rows:
        lin[(d["_sid"], d.get("model"), d.get("system_tokens"), d.get("tools_tokens"))].append(d)
        p = pricing.resolve(d.get("model"))
        if p is not None:
            total_usd += (
                int(d.get("usage_input_tokens") or 0) * p.input
                + int(d.get("usage_cache_create") or 0) * p.cache_write
                + int(d.get("usage_cache_read") or 0) * p.cache_read
                + int(d.get("usage_output_tokens") or 0) * p.output
            )
    pairs = clean = 0
    n: Counter = Counter()
    usd: Counter = Counter()
    for seq in lin.values():
        seq.sort(key=lambda d: d["ts"])
        for i, b in enumerate(seq):
            p = pricing.resolve(b.get("model"))
            if p is None:
                continue
            a = None
            for j in range(i - 1, -1, -1):
                c = seq[j]
                if b["ts"] - c["ts"] > TTL_S:
                    break
                if b["replay_hits"] <= c["_n"] <= b["_n"] and c["_tot"] <= b["_tot"] + SLACK:
                    a = c
                    break
            if a is None:
                continue
            pairs += 1
            excess = max(0, a["_tot"] - int(b.get("usage_cache_read") or 0) - SLACK)
            if not excess:
                clean += 1
                continue
            if b.get("replay_stop"):
                cls = f"stop:{b['replay_stop']}"
            elif b["replay_hits"] >= a["_n"]:
                cls = "replay_held"  # distil re-sent every prior item as before
            elif a.get("prefix_hash") != b.get("prefix_hash"):
                cls = "system_or_tools_changed"
            else:
                cls = "diverged_unattributed"
            n[cls] += 1
            usd[cls] += excess * (p.cache_write - p.cache_read)
    broke = sum(usd.values())
    return {
        "requests": len(rows),
        "billed_usd": round(total_usd, 2),
        "warm_pairs": pairs,
        "warm_pairs_clean": clean,
        "break_usd": round(broke, 2),
        "break_share_of_billed": round(broke / total_usd, 4) if total_usd else None,
        "by_cause": {
            k: {"pairs": n[k], "usd": round(v, 2)}
            for k, v in sorted(usd.items(), key=lambda kv: -kv[1])
        },
    }


# ------------------------------------------------------------------ transcript replay


def _load(path: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") not in ("user", "assistant") or d.get("isSidechain"):
                continue
            m = d.get("message") or {}
            role, c = m.get("role"), m.get("content")
            if role not in ("user", "assistant") or c is None:
                continue
            if isinstance(c, str):
                c = [{"type": "text", "text": c}]
            c = [
                b
                for b in c
                if isinstance(b, dict) and b.get("type") in ("text", "tool_use", "tool_result")
            ]
            if not c:
                continue
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"].extend(copy.deepcopy(c))
            else:
                msgs.append({"role": role, "content": copy.deepcopy(c)})
    return msgs


def _bare(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _bare(v) for k, v in node.items() if k != "cache_control"}
    if isinstance(node, list):
        return [_bare(x) for x in node]
    return node


def transcripts(limit: int, maxreq: int) -> dict[str, Any]:
    from distil import prefixreplay
    from distil.adapters import anthropic as A
    from distil.compress import provenance as P

    calls: list[tuple[int, int]] = []
    real = P.quote_hazard

    def counted(quotes, view):  # type: ignore[no-untyped-def]
        out = real(quotes, view)
        calls.append(out)
        return out

    files = sorted(
        glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")),
        key=os.path.getsize,
        reverse=True,
    )
    sel = [f for f in files if 200_000 < os.path.getsize(f) < 6_000_000][:limit]
    keys = ("requests", "widen_ran", "widen_rescued", "distil_breaks", "distil_break_bytes")
    c: Counter = Counter({k: 0 for k in keys})
    c["forwarded_bytes"] = 0  # summed over every request: what the provider is sent
    P.quote_hazard = counted
    try:
        for f in sel:
            msgs = _load(f)
            prefixreplay.reset()
            prev: list[str] | None = None
            ends = [i for i, m in enumerate(msgs) if m["role"] == "user"][1:][:maxreq]
            for e in ends:
                req = copy.deepcopy(msgs[: e + 1])
                req[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                if e >= 2 and req[e - 2]["role"] == "user":
                    req[e - 2]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                calls.clear()
                out, _ = A.compress_messages(req)
                if len(calls) == 2:
                    c["widen_ran"] += 1
                    c["widen_rescued"] += calls[1][1] < calls[0][1]
                body = prefixreplay.apply({"model": "m", "messages": out}, "messages", req)
                fwd = [
                    json.dumps(_bare(x), separators=(",", ":"), ensure_ascii=False)
                    for x in body["messages"]
                ]
                c["requests"] += 1
                c["forwarded_bytes"] += sum(len(x) for x in fwd)
                if prev is not None:
                    k = min(len(prev), len(fwd))
                    div = next((i for i in range(k) if prev[i] != fwd[i]), None)
                    if div is not None:
                        c["distil_breaks"] += 1
                        c["distil_break_bytes"] += sum(len(x) for x in prev[div:])
                prev = fwd
            c["transcripts"] += 1
    finally:
        P.quote_hazard = real
        prefixreplay.reset()
    return dict(c)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--transcripts", type=int, default=12)
    ap.add_argument("--maxreq", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # The restore store is written by every digest; keep it out of the real ~/.distil.
    os.environ["DISTIL_HOME"] = tempfile.mkdtemp(prefix="distil-ipb-")
    result = {
        "ttl_s": TTL_S,
        "slack_tokens": SLACK,
        "ledger": ledger(),
        "transcript_replay": transcripts(args.transcripts, args.maxreq),
    }
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
