"""Size the cost levers that are NOT tool-output compression, offline and at $0.

    python -m benchmarks.policysim.levers --lite DIR --long DIR --claude-code 20 \\
        --out benchmarks/results/lever-sizing

Levers: prompt-cache placement and TTL on the stable prefix, output share, the Batch API,
model right-sizing (pure price arithmetic), and compression (copied from policysim.json).
Every number replays the SAME trajectory: behaviour is assumed unchanged. Output is
counts and dollars only, never transcript content.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from distil import pricing

from .calibrate import calibrate
from .costmodel import Anthropic, Price, TokenModel, Usage, segments
from .sim import allocate_hidden
from .trajectory import (
    Message,
    Trajectory,
    block_text,
    blocks,
    load_claude_code,
    load_harness,
    thinking_key,
)

#: Idle seconds between consecutive tasks when a corpus is replayed as one user's stream.
GAPS = (0.0, 120.0, 600.0, 1800.0)
#: A step "looks cheap" if its reply is one tool call, no prose, <= this many output tokens.
CHEAP_OUT = 300
#: Keep-alive: refresh a 1h entry every 55 minutes, for at most this long after idling.
PING_S = 3300.0
HORIZONS_H = (2, 4, 8, 24)
MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5")
SMALL = "claude-haiku-4-5"


def is_cheap(reply: Message | None, out_tokens: float) -> bool:
    """Observable proxy for a step a smaller model might take: a single tool_use, no
    non-empty text block, and a short reply (thinking included)."""
    if reply is None or reply.get("role") != "assistant" or out_tokens > CHEAP_OUT:
        return False
    bs = [b for b in blocks(reply) if thinking_key(b) is None]
    uses = [b for b in bs if isinstance(b, dict) and b.get("type") == "tool_use"]
    prose = [b for b in bs if not (isinstance(b, dict) and b.get("type") == "tool_use")]
    return len(uses) == 1 and not any(block_text(b).strip() for b in prose)


def step_outputs(t: Trajectory, tm: TokenModel) -> list[tuple[float, bool]]:
    """(output tokens, looks cheap) of the reply to each request; thinking from the
    allocation, so the sum is the billed output."""
    out = []
    for end in t.request_ends:
        reply = t.messages[end] if end < len(t.messages) else None
        n = 0.0
        for b in blocks(reply) if reply else []:
            k = thinking_key(b)
            n += t.hidden.get(k, 0) if k is not None else tm.count(block_text(b))
        out.append((n, is_cheap(reply, n)))
    return out


def usd_parts(u: Usage, p: Price) -> dict[str, float]:
    m = p.input / 1e6
    return {
        "input": u.input * m,
        "cache_write": (u.write_5m * p.write_5m_mult + u.write_1h * p.write_1h_mult) * m,
        "cache_read": u.read * p.read_mult * m,
        "output": u.output * p.output / 1e6,
    }


def _dur(t: Trajectory) -> float:
    return t.times[-1] + (t.times[1] - t.times[0] if len(t.times) > 1 else 0.0)


def stream(
    trajs: list[Trajectory],
    reqs: list[list[list[Any]]],
    model: str,
    framing: float,
    gap: float | None,
    ttl: str = "5m",
    prefix_bp: bool = False,
    ttls: list[str] | None = None,
) -> dict[str, Any]:
    """Price a corpus as one user running its tasks back to back, ``gap`` idle seconds
    apart, sharing one cache (``gap=None``: every task cold, policysim's convention).

    ``prefix_bp`` adds a breakpoint on tools+system (segment 0) so later tasks can read
    it; the harness and the default policysim replay have only the newest-block one."""
    prov = Anthropic(model=model, framing=framing)
    u, off = Usage(), 0.0
    prefix_writes = rewrites = 0.0
    for t, rs in zip(trajs, reqs):
        if gap is None:
            prov.reset()
        for k, segs in enumerate(rs):
            bps = [0, len(segs) - 1] if prefix_bp else None
            r = prov.request(segs, off + t.times[k], ttl, bps, ttls if prefix_bp else None)
            u += r
            w = r.write_5m + r.write_1h
            if r.read < segs[0].tokens and w > 0:
                prefix_writes += segs[0].tokens
            if k > 0 and r.read == 0 and w > 0:
                rewrites += w  # the whole prefix re-written after a TTL expiry
        u.output += t.output_tokens
        off += _dur(t) + (gap or 0.0)
    parts = usd_parts(u, prov.price)
    n = len(trajs)
    return {
        "usd_per_task": round(sum(parts.values()) / n, 5),
        "parts_per_task": {k: round(v / n, 5) for k, v in parts.items()},
        "prefix_tokens_written_per_task": round(prefix_writes / n, 1),
        "expiry_rewrite_tokens_per_task": round(rewrites / n, 1),
    }


def router(
    trajs: list[Trajectory],
    reqs: list[list[list[Any]]],
    tm: TokenModel,
    framing: float,
    main: str,
) -> dict[str, Any]:
    """Oracle per-step router: cheap-looking steps go to the small model, each model with
    its own cache (caches are per model). Also the cache-blind estimate of the same."""
    base = cache_aware = naive = 0.0
    steps = cheap = 0
    cheap_usd = 0.0
    for t, rs in zip(trajs, reqs):
        a = Anthropic(model=main, framing=framing)
        b = Anthropic(model=main, framing=framing)
        s = Anthropic(model=SMALL, framing=framing)
        for k, (segs, (out, c)) in enumerate(zip(rs, step_outputs(t, tm))):
            r = a.request(segs, t.times[k], t.ttl)
            r.output = out
            step_usd = a.price.usd(r)
            base += step_usd
            steps += 1
            if c:
                cheap += 1
                cheap_usd += step_usd
                naive += step_usd - s.price.usd(r)  # same token split, small-model rates
                q = s.request(segs, t.times[k], t.ttl)
                q.output = out
                cache_aware += s.price.usd(q)
            else:
                q = b.request(segs, t.times[k], t.ttl)
                q.output = out
                cache_aware += b.price.usd(q)
    return {
        "steps": steps,
        "cheap_steps_share": round(cheap / steps, 4) if steps else None,
        "cheap_steps_usd_share": round(cheap_usd / base, 4) if base else None,
        "oracle_router_vs_plain_cache_blind": round(-naive / base, 4) if base else None,
        "oracle_router_vs_plain_cache_aware": round(cache_aware / base - 1, 4) if base else None,
    }


def harness_gaps(run_dir: Path) -> dict[str, Any]:
    """Seconds between one arm's consecutive tasks in the real (serial, arm-rotating)
    run: the wall time of the other arms' rows in between. Reused rows are skipped."""
    rows = [json.loads(x) for x in (run_dir / "results.jsonl").read_text().splitlines() if x]
    rows = [r for r in rows if "reused_from" not in r]
    last: dict[str, int] = {}
    gaps = []
    for i, r in enumerate(rows):
        j = last.get(r["arm"])
        if j is not None:
            gaps.append(sum(float(x.get("wall_s") or 0) for x in rows[j + 1 : i]))
        last[r["arm"]] = i
    walls = [float(r.get("wall_s") or 0) for r in rows]
    return {
        "rows_run_here": len(rows),
        "median_task_wall_s": round(statistics.median(walls), 1) if walls else None,
        "median_same_arm_gap_s": round(statistics.median(gaps), 1) if gaps else None,
        "share_gaps_over_5m": round(sum(g > 300 for g in gaps) / len(gaps), 3) if gaps else None,
    }


def harness_corpus(trajs: list[Trajectory], tm: TokenModel, framing: float) -> dict[str, Any]:
    """Every lever that the harness trajectories can size, for one corpus."""
    model = trajs[0].model.split("@")[0]
    reqs = [
        [
            segments(t.request(k), tm, t.hidden, t.overhead_tokens)
            for k in range(len(t.request_ends))
        ]
        for t in trajs
    ]
    by_arm: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(trajs):
        by_arm[t.arm].append(i)

    def per_arm(**kw: Any) -> dict[str, Any]:
        """Stream each arm separately (arms' first requests can be byte-identical)."""
        parts: Counter[str] = Counter()
        pw = rw = 0.0
        for idx in by_arm.values():
            r = stream([trajs[i] for i in idx], [reqs[i] for i in idx], model, framing, **kw)
            m = len(idx)
            parts.update({k: v * m for k, v in r["parts_per_task"].items()})
            pw += r["prefix_tokens_written_per_task"] * m
            rw += r["expiry_rewrite_tokens_per_task"] * m
        n = len(trajs)
        return {
            "usd_per_task": round(sum(parts.values()) / n, 5),
            "parts_per_task": {k: round(v / n, 5) for k, v in parts.items()},
            "prefix_tokens_written_per_task": round(pw / n, 1),
            "expiry_rewrite_tokens_per_task": round(rw / n, 1),
        }

    base = per_arm(gap=None)
    b = base["usd_per_task"]
    scen: dict[str, Any] = {
        "cold_5m (policysim baseline)": base,
        "cold_1h": per_arm(gap=None, ttl="1h"),
    }
    for g in GAPS:
        tag = f"gap{int(g)}s"
        scen[f"{tag}_tail5m"] = per_arm(gap=g)
        scen[f"{tag}_prefixbp_5m"] = per_arm(gap=g, prefix_bp=True)
        scen[f"{tag}_prefixbp_1h"] = per_arm(gap=g, ttl="1h", prefix_bp=True)
        scen[f"{tag}_prefixbp1h_tail5m"] = per_arm(gap=g, prefix_bp=True, ttls=["1h", "5m"])
    for s in scen.values():
        s["vs_baseline"] = round(s["usd_per_task"] / b - 1, 4)

    parts = base["parts_per_task"]
    out_share = parts["output"] / b
    prefix_share = None
    if trajs[0].overhead_tokens:
        steps = sum(len(t.request_ends) for t in trajs) / len(trajs)
        p = Anthropic(model=model).price
        o = trajs[0].overhead_tokens
        # cold: written once per task (1.25x), read on every later request
        prefix_usd = o * (p.write_5m_mult + p.read_mult * (steps - 1)) * p.input / 1e6
        prefix_share = round(prefix_usd / b, 4)
    models = {}
    for m in MODELS:
        models[m] = stream(trajs, reqs, m, framing, gap=None)["usd_per_task"]
    return {
        "model": model,
        "trajectories": len(trajs),
        "requests": sum(len(t.request_ends) for t in trajs),
        "plain_usd_per_task": b,
        "bill_share": {k: round(v / b, 4) for k, v in parts.items()},
        "tools_system_prefix_share_est": prefix_share,
        "cache": scen,
        "output": {
            "share": round(out_share, 4),
            "reduction_sensitivity_vs_plain": {
                f"{x}%": round(-out_share * x / 100, 4) for x in (10, 20, 30)
            },
        },
        "batch": {"batchable_share_of_requests": 0.0},
        "model_reprice_usd_per_task": {m: v for m, v in models.items()},
        "model_reprice_vs_plain": {m: round(v / b - 1, 4) for m, v in models.items()},
        "router": router(trajs, reqs, tm, framing, model),
    }


def cc_sim(trajs: list[Trajectory], tm: TokenModel) -> dict[str, Any]:
    """Claude Code sessions in the simulator: input side only, real timestamps."""
    out = {}
    for ttl in ("5m", "1h"):
        u = Usage()
        usd = 0.0
        for t in trajs:
            prov = Anthropic(model=t.model.split("@")[0], framing=0.0)
            for k in range(len(t.request_ends)):
                r = prov.request(segments(t.request(k), tm, t.hidden, 0.0), t.times[k], ttl)
                u += r
                usd += prov.price.usd(r)
        out[f"input_side_usd_per_session_{ttl}"] = round(usd / len(trajs), 4)
    out["ttl_1h_vs_5m"] = round(
        out["input_side_usd_per_session_1h"] / out["input_side_usd_per_session_5m"] - 1, 4
    )
    out["sessions"] = len(trajs)
    return out


# ------------------------------------------------------------------ Claude Code billed usage


def _responses(path: Path) -> list[dict[str, Any]]:
    """Main-thread assistant responses (deduped by message id): ts, model, usage, blocks."""
    from distil.whatif import _epoch

    by: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or rec.get("type") != "assistant":
                continue
            if rec.get("isSidechain") is True or rec.get("isApiErrorMessage") is True:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict) or not isinstance(msg.get("usage"), dict):
                continue
            mid = str(msg.get("id") or len(order))
            model = msg.get("model")
            if not isinstance(model, str) or pricing.resolve(model) is None:
                continue
            r = by.get(mid)
            if r is None:
                r = by[mid] = {"ts": _epoch(rec.get("timestamp")), "model": model, "blocks": []}
                order.append(mid)
            r["usage"] = msg["usage"]
            c = msg.get("content")
            r["blocks"].extend(c if isinstance(c, list) else [])
    return [by[m] for m in order]


def _u(r: dict[str, Any]) -> tuple[int, int, int, int, int]:
    u = r["usage"]
    w = int(u.get("cache_creation_input_tokens") or 0)
    sp = u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
    w1 = min(w, int(sp.get("ephemeral_1h_input_tokens") or 0))
    return (
        int(u.get("input_tokens") or 0),
        w - w1,
        w1,
        int(u.get("cache_read_input_tokens") or 0),
        int(u.get("output_tokens") or 0),
    )


def keepalive_saving(
    gap: float, ctx: float, rewritten: float, write_usd: float, read_per_tok: float, horizon: float
) -> float:
    """$ saved on one resume after ``gap`` s if the cache had been pinged every ``PING_S``
    for up to ``horizon`` s: the resume reads the ``rewritten`` tokens instead of paying
    ``write_usd``; every ping reads the whole ``ctx``. Past the horizon the pings were waste."""
    pings = int(min(gap, horizon) // PING_S) * ctx * read_per_tok
    return -pings if gap > horizon else write_usd - rewritten * read_per_tok - pings


def _cost(p: pricing.Pricing, i: int, w5: float, w1: float, rd: float, o: int) -> float:
    return (
        i * p.input + w5 * p.cache_write + w1 * p.cache_write_1h + rd * p.cache_read + o * p.output
    )


def cc_billed(root: Path) -> dict[str, Any]:
    """Every local Claude Code session's billed usage, priced; levers as arithmetic on it.

    TTL counterfactuals use the real gap between consecutive responses: under a 5-minute
    TTL a request more than 300 s after the previous one would have re-written what it
    read. ``first`` responses keep their reads (another session wrote that prefix)."""
    from distil.whatif import discover

    comp: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    wclass: Counter[str] = Counter()
    ttl5 = ttl1 = actual = 0.0
    reprice: Counter[str] = Counter()
    steps = cheap = 0
    cheap_usd = cheap_small = 0.0
    first_n = first_read = 0
    sessions = 0
    t_min, t_max = float("inf"), 0.0
    keep: Counter[int] = Counter()
    small = pricing.resolve(SMALL)
    assert small is not None
    for path in discover(root):
        rs = _responses(path)
        if not rs:
            continue
        sessions += 1
        prev = None
        for r in rs:
            p = pricing.resolve(r["model"])
            assert p is not None
            i, w5, w1, rd, o = _u(r)
            usd = _cost(p, i, w5, w1, rd, o)
            actual += usd
            by_model[p.name] += usd
            comp["input"] += i * p.input
            comp["cache_write"] += w5 * p.cache_write + w1 * p.cache_write_1h
            comp["cache_read"] += rd * p.cache_read
            comp["output"] += o * p.output
            gap = r["ts"] - prev if (prev and r["ts"]) else None
            if r["ts"]:
                t_min, t_max = min(t_min, r["ts"]), max(t_max, r["ts"])
                prev = r["ts"]
            wusd = w5 * p.cache_write + w1 * p.cache_write_1h
            if gap is None:
                wclass["first_request"] += wusd
                first_n += 1
                first_read += rd > 0
            elif gap > 3600:
                wclass["after_gap_over_1h"] += wusd
                for h in HORIZONS_H:
                    keep[h] += keepalive_saving(
                        gap, w5 + w1 + rd, w5 + w1, wusd, p.cache_read, h * 3600.0
                    )
            elif gap > 300:
                wclass["after_gap_5m_to_1h"] += wusd
            else:
                wclass["within_5m"] += wusd
            w = w5 + w1
            if gap is not None and gap > 300:  # a 5m cache would have expired
                ttl5 += _cost(p, i, w + rd, 0, 0, o)
            else:
                ttl5 += _cost(p, i, w, 0, rd, o)
            ttl1 += _cost(p, i, 0, w, rd, o)  # same hits as billed (gaps > 1h missed anyway)
            for m in MODELS:
                q = pricing.resolve(m)
                assert q is not None
                reprice[m] += _cost(q, i, w5, w1, rd, o)
            steps += 1
            uses = [b for b in r["blocks"] if isinstance(b, dict) and b.get("type") == "tool_use"]
            prose = [
                b
                for b in r["blocks"]
                if isinstance(b, dict)
                and b.get("type") == "text"
                and str(b.get("text", "")).strip()
            ]
            if len(uses) == 1 and not prose and o <= CHEAP_OUT:
                cheap += 1
                cheap_usd += usd
                cheap_small += _cost(small, i, w5, w1, rd, o)
        # the session's last idle is never resumed: pings until the horizon are pure cost
        p = pricing.resolve(rs[-1]["model"])
        assert p is not None
        i, w5, w1, rd, o = _u(rs[-1])
        for h in HORIZONS_H:
            keep[h] += keepalive_saving(
                float("inf"), i + w5 + w1 + rd + o, 0, 0, p.cache_read, h * 3600.0
            )
    tot = actual or 1.0
    return {
        "sessions": sessions,
        "responses": steps,
        "span_days": round((t_max - t_min) / 86400, 1) if t_max else None,
        "usd_total": round(actual, 2),
        "usd_by_model_share": {k: round(v / tot, 4) for k, v in by_model.most_common()},
        "bill_share": {k: round(v / tot, 4) for k, v in comp.items()},
        "cache_write_usd_share_by_when": {k: round(v / tot, 4) for k, v in wclass.items()},
        "first_requests": first_n,
        "first_requests_with_cache_read": first_read,
        "ttl_all_5m_vs_actual": round(ttl5 / tot - 1, 4),
        "ttl_all_1h_vs_actual": round(ttl1 / tot - 1, 4),
        "output_reduction_sensitivity": {
            f"{x}%": round(-comp["output"] / tot * x / 100, 4) for x in (10, 20, 30)
        },
        "model_reprice_vs_actual": {m: round(v / tot - 1, 4) for m, v in reprice.items()},
        "keepalive_ping_55min_vs_actual": {f"{h}h": round(-keep[h] / tot, 4) for h in HORIZONS_H},
        "cheap_steps_share": round(cheap / steps, 4) if steps else None,
        "cheap_steps_usd_share": round(cheap_usd / tot, 4),
        "cheap_steps_to_haiku_cache_blind_vs_actual": round((cheap_small - cheap_usd) / tot, 4),
    }


def compression_reference(policysim: Path) -> dict[str, Any]:
    d = json.loads(policysim.read_text())
    keep = (
        "distil-lossless",
        "entry-lossless",
        "rtk-like",
        "headroom",
        "trunc-2000-sh",
        "distil-digest-sh",
        "entry-digest-sh",
    )
    return {
        c: {
            n: {
                "vs_plain_pure": r["vs_plain_pure"],
                "vs_plain_mid_penalty": r["vs_plain_penalty"]["mid"],
                "breakeven_lambda": r["breakeven_lambda"],
            }
            for n, r in d["results"][c]["anthropic"].items()
            if n in keep
        }
        for c in d["results"]
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.policysim.levers")
    ap.add_argument("--lite", type=Path, required=True)
    ap.add_argument("--long", type=Path, required=True)
    ap.add_argument("--claude-code", type=int, default=0, help="sessions for the simulator")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/lever-sizing"))
    ap.add_argument(
        "--policysim", type=Path, default=Path("benchmarks/results/policysim/policysim.json")
    )
    a = ap.parse_args(argv)

    lite_cal = load_harness(a.lite, ("rtk",), "swebench-lite", {})
    long_cal = load_harness(a.long, ("plain", "rtk"), "swebench-long", {})
    long_distil = load_harness(a.long, ("distil",), "swebench-long", {})
    tm, overhead, framing, _ = calibrate(lite_cal, long_cal, long_distil)
    lite = load_harness(a.lite, ("provider-cm", "selective"), "swebench-lite", {})
    long = [t for t in long_cal + long_distil if t.arm in ("plain", "distil")]
    for t in lite + long:
        t.overhead_tokens = round(overhead)
        allocate_hidden(t, tm)
    out: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d"),
        "cost": "$0: offline replay and arithmetic on recorded usage; no model or API calls",
        "token_model": {"overhead_tokens": round(overhead, 1), "framing": round(framing, 3)},
        "harness_timing": {
            "swebench-lite": harness_gaps(a.lite),
            "swebench-long": harness_gaps(a.long),
        },
        "corpora": {},
    }
    for name, trajs in (("swebench-lite", lite), ("swebench-long", long)):
        t0 = time.time()
        out["corpora"][name] = harness_corpus(trajs, tm, framing)
        print(f"{name}: {time.time() - t0:.0f}s", file=sys.stderr)
    from distil.whatif import claude_projects_root

    root = claude_projects_root()
    if a.claude_code:
        cc = load_claude_code(root, a.claude_code)
        out["corpora"]["claude-code-sim"] = cc_sim(cc, tm)
    out["corpora"]["claude-code-billed"] = cc_billed(root)
    if a.policysim.exists():
        out["compression_reference_anthropic"] = compression_reference(a.policysim)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "levers.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out / 'levers.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
