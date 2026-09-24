"""Decompose LIVE, cache-priced billed dollars and bound what compression can recover.

Reads the content-free per-request accounting distil already writes
(``~/.distil/sessions/*.requests.jsonl``) read-only, prices every request with the
provider's own ``usage.*`` counts at ``distil.pricing`` list prices (cache write 1.25x,
cache read 0.10x), and splits the input side into system / tools / message buckets
using distil's own census (heuristic token counts, rescaled per request so the buckets
sum to the provider-billed input). Output: one JSON summary, no content.

    python benchmarks/live_savings_decomposition.py [--since 2026-09-01] [--out PATH]

Every figure is labelled in the JSON: ``measured`` = straight from provider usage;
``estimated`` = heuristic-token share rescaled to billed totals, or a lineage model.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distil import pricing  # noqa: E402

TTL_S = 300  # Anthropic default 5-minute cache TTL
TOOL_RESULT_PREFIX = "tool_result_"


def _rows(pattern: str, since_ts: float, until_ts: float = float("inf")):
    for f in sorted(glob.glob(pattern)):
        sid = os.path.basename(f).split(".")[0]
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("status") != 200 or not since_ts <= float(d.get("ts") or 0) < until_ts:
                    continue
                if not (
                    d.get("usage_cache_read")
                    or d.get("usage_cache_create")
                    or d.get("usage_input_tokens")
                ):
                    continue
                d["_sid"] = sid
                yield d


def analyse(rows) -> dict:
    tier = Counter()  # $ by billing tier (measured)
    tok = Counter()  # billed tokens by tier (measured)
    comp_usd = Counter()  # $ by content bucket (estimated)
    comp_tok = Counter()
    census_tok = Counter()  # heuristic, original (pre-compression) tokens
    mode = Counter()
    unpriced = 0
    realized_saved_usd = 0.0
    lineages: dict[tuple, list] = defaultdict(list)
    n = 0
    expand_turns = 0
    expand_usd = 0.0

    for d in rows:
        p = pricing.resolve(d.get("model"))
        if p is None:
            unpriced += 1
            continue
        n += 1
        mode[d.get("mode") or "?"] += 1
        u, w, r, o = (
            int(d.get(k) or 0)
            for k in (
                "usage_input_tokens",
                "usage_cache_create",
                "usage_cache_read",
                "usage_output_tokens",
            )
        )
        usd = {
            "uncached_input": u * p.input,
            "cache_write": w * p.cache_write,
            "cache_read": r * p.cache_read,
            "output": o * p.output,
        }
        for k, v in usd.items():
            tier[k] += v
        tok.update({"uncached_input": u, "cache_write": w, "cache_read": r, "output": o})
        in_usd = usd["uncached_input"] + usd["cache_write"] + usd["cache_read"]
        billed_in = u + w + r

        census = d.get("census") or {}
        census_tok.update(census)
        sys_t, tools_t = int(d.get("system_tokens") or 0), int(d.get("tools_tokens") or 0)
        msg_t = int(d.get("compressible_tokens") or 0) or sum(census.values())
        saved = int(d.get("tokens_saved") or 0)
        post_heur = sys_t + tools_t + max(0, msg_t - saved)
        scale = billed_in / post_heur if post_heur else 0.0
        blended = in_usd / billed_in if billed_in else 0.0  # $/billed input token, this request

        # Post-compression share of each bucket. Savings came out of tool_result buckets
        # (digest/fold) -> subtract them there proportionally.
        tr_total = sum(v for k, v in census.items() if k.startswith(TOOL_RESULT_PREFIX))
        buckets = {"system": sys_t, "tools": tools_t}
        other_msg = msg_t - sum(census.values())
        for k, v in census.items():
            if k.startswith(TOOL_RESULT_PREFIX) and tr_total:
                v = max(0.0, v - saved * v / tr_total)
            buckets[k] = v
        if other_msg > 0:
            buckets["messages_uncensused"] = other_msg
        for k, v in buckets.items():
            comp_tok[k] += v * scale
            comp_usd[k] += v * scale * blended
        comp_usd["output"] += usd["output"]

        realized_saved_usd += (
            saved * scale * blended
        )  # counterfactual: saved tokens at this request's blend
        if d.get("expanded"):
            expand_turns += 1
            expand_usd += in_usd + usd["output"]

        key = (d["_sid"], d.get("model"), sys_t, tools_t)
        lineages[key].append(
            (
                float(d["ts"]),
                msg_t,
                tr_total,
                w,
                u,
                r,
                p,
                scale,
                o,
                census,
                max(0, tr_total - saved),
            )
        )

    total = sum(tier.values())

    # Lineage model. A "clean" pair is two requests of one lineage within the cache TTL
    # whose message growth fits inside what the provider billed as new (write+uncached):
    # there the growth IS the fresh content, so its tool_result fraction is measurable.
    # Pairs that shrink or overshoot (parallel subagents sharing a lineage key, compaction)
    # are excluded from the ratio rather than guessed at.
    clean_tr = clean_msg = clean_fresh_billed = 0.0
    clean_pairs = dirty_pairs = 0
    fresh_by_bucket: Counter = Counter()
    warm_fresh_usd = 0.0  # write+uncached $ on warm requests (all first appearances)
    cold_write_tok = cold_write_usd = 0.0
    cold_by_gap: dict[str, list[float]] = {
        "5-15min": [0, 0.0],
        "15-60min": [0, 0.0],
        ">60min": [0, 0.0],
    }
    excess_write_tok = excess_write_usd = 0.0
    warm_reqs = cold_reqs = 0
    for seq in lineages.values():
        seq.sort()
        prev = None
        for ts, msg_t, tr, w, u, r, p, scale, o, cen, _ in seq:
            if prev is None:
                prev = (ts, msg_t, tr, cen)
                warm_fresh_usd += w * p.cache_write + u * p.input
                continue
            gap = ts - prev[0]
            if gap > TTL_S:
                cold_reqs += 1
                cold_write_tok += w
                cold_write_usd += w * p.cache_write
                bucket = "5-15min" if gap <= 900 else "15-60min" if gap <= 3600 else ">60min"
                cold_by_gap[bucket][0] += 1
                cold_by_gap[bucket][1] += w * p.cache_write
            else:
                warm_reqs += 1
                warm_fresh_usd += w * p.cache_write + u * p.input
                d_msg = (msg_t - prev[1]) * scale
                if 0 <= d_msg <= w + u + 2000:
                    clean_pairs += 1
                    clean_msg += d_msg
                    clean_tr += max(0, tr - prev[2]) * scale
                    clean_fresh_billed += w + u
                    for b, v in cen.items():
                        fresh_by_bucket[b] += max(0, v - prev[3].get(b, 0)) * scale
                    ex = max(0.0, w - d_msg - 2000)  # 2k slack: breakpoint/system jitter
                    excess_write_tok += ex
                    excess_write_usd += ex * (p.cache_write - p.cache_read)
                else:
                    dirty_pairs += 1
            prev = (ts, msg_t, tr, cen)
    # Cold-cache pool: at a request past the TTL the provider re-writes the whole prefix
    # anyway, so history can be re-compressed there with no cache penalty. The pool is the
    # $ the carried tool_results cost from that write until the next cold point/lineage end.
    cold_pool_usd = 0.0
    for seq in lineages.values():
        for i in range(1, len(seq)):
            if seq[i][0] - seq[i - 1][0] <= TTL_S:
                continue
            _, _, _, _, _, _, p, scale, _, _, tr = seq[i]  # post-compression tool_result tokens
            n_after = 0
            for j in range(i + 1, len(seq)):
                # stop at the next cold point, or once the history shrinks below the
                # cold point's (compaction, or an interleaved thread): conservative
                if seq[j][0] - seq[j - 1][0] > TTL_S or seq[j][1] < seq[i][1]:
                    break
                n_after += 1
            cold_pool_usd += (
                tr * scale * p.input * (p.cache_write_mult + p.cache_read_mult * n_after)
            )
    f_tr = clean_tr / clean_fresh_billed if clean_fresh_billed else 0.0
    fresh_tr_usd = f_tr * warm_fresh_usd

    def share(x: float) -> float:
        return round(x / total, 4) if total else 0.0

    tr_usd = sum(v for k, v in comp_usd.items() if k.startswith(TOOL_RESULT_PREFIX))
    return {
        "requests_priced": n,
        "requests_unpriced_model": unpriced,
        "modes": dict(mode),
        "billed_usd_total": round(total, 2),
        "by_tier_measured": {
            k: {"usd": round(v, 2), "share": share(v), "tokens": tok[k]} for k, v in tier.items()
        },
        "by_content_estimated": {
            k: {
                "usd": round(v, 2),
                "share": share(v),
                "tokens_billed_scale": int(comp_tok.get(k, 0)),
            }
            for k, v in sorted(comp_usd.items(), key=lambda x: -x[1])
        },
        "census_original_tokens_heuristic": dict(census_tok.most_common()),
        "realized": {
            "saved_usd_estimated": round(realized_saved_usd, 2),
            "saved_share_of_counterfactual_bill": round(
                realized_saved_usd / (total + realized_saved_usd), 4
            )
            if total
            else 0,
            "note": "tokens_saved rescaled to billed scale, priced at each request's own blended input rate",
        },
        "tool_results": {
            "share_of_billed_usd_post_compression": share(tr_usd),
            "fresh_tr_fraction_of_new_billed_tokens_clean_pairs": round(f_tr, 4),
            "fresh_growth_fraction_of_new_billed_tokens_clean_pairs": round(
                clean_msg / clean_fresh_billed, 4
            )
            if clean_fresh_billed
            else 0,
            "clean_pairs": clean_pairs,
            "fresh_growth_by_bucket_share": {
                b: round(v / clean_fresh_billed, 4) for b, v in fresh_by_bucket.most_common() if v
            },
            "excluded_pairs": dirty_pairs,
            "fresh_first_appearance_usd_estimated": round(fresh_tr_usd, 2),
            "fresh_first_appearance_share": share(fresh_tr_usd),
            "lifetime_over_first_appearance_multiplier": round(tr_usd / fresh_tr_usd, 2)
            if fresh_tr_usd
            else 0,
            "note": "first appearance = f_tr x (write+uncached $ on warm requests); lifetime = the tool_result buckets' post-compression $ share; the multiplier is what a first-sight byte saved is worth relative to its write",
        },
        "cache_breaks": {
            "warm_requests": warm_reqs,
            "cold_requests_gap_gt_ttl": cold_reqs,
            "cold_write_tokens": int(cold_write_tok),
            "cold_write_usd": round(cold_write_usd, 2),
            "cold_write_share": share(cold_write_usd),
            "cold_by_gap": {
                k: {"requests": v[0], "write_usd": round(v[1], 2), "share": share(v[1])}
                for k, v in cold_by_gap.items()
            },
            "cold_point_tool_result_pool_usd": round(cold_pool_usd, 2),
            "cold_point_tool_result_pool_share": share(cold_pool_usd),
            "warm_excess_write_tokens_clean_pairs": int(excess_write_tok),
            "warm_excess_write_usd_vs_read": round(excess_write_usd, 2),
            "warm_excess_write_share": share(excess_write_usd),
            "note": "excess = write beyond the lineage's message growth (+2k slack) on clean warm pairs; $ lost to prefix breaks of ANY cause (client, distil) relative to reading those tokens. Lower bound: dirty pairs excluded",
        },
        "expand_round_trips": {
            "requests": expand_turns,
            "usd": round(expand_usd, 2),
            "share": share(expand_usd),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=os.path.expanduser("~/.distil/sessions/*.requests.jsonl"))
    ap.add_argument("--since", default="2026-09-01")
    ap.add_argument(
        "--until", default=None, help="ISO date/time; freeze the window (live logs grow)"
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc).timestamp()
    until = (
        datetime.fromisoformat(a.until).replace(tzinfo=timezone.utc).timestamp()
        if a.until
        else float("inf")
    )
    res = {
        "source": a.sessions,
        "since": a.since,
        "pricing": "distil.pricing list prices; write 1.25x, read 0.10x",
        "until": a.until,
        **analyse(_rows(a.sessions, since, until)),
    }
    text = json.dumps(res, indent=2)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
