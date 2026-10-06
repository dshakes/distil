"""Evaluate every policy on every corpus under every provider; write aggregates only.

    python -m benchmarks.policysim --lite DIR --long DIR --out benchmarks/results/policysim \\
        [--headroom-python PATH] [--claude-code N]

$0: no network, no model calls. Output is counts and dollars, never transcript content.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .calibrate import calibrate, expand_tool_tokens
from .costmodel import PROVIDERS, TokenModel, provider
from .policies import DistilServed, Spec, build
from .sim import allocate_hidden, price, replay, simulate
from .trajectory import Trajectory, block_text, blocks, load_claude_code, load_grades, load_harness

# -------------------------------------------------------------------- re-run penalty fit


def _rows(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["arm"], r["instance_id"])] = r
    return out


def _ok(r: dict[str, Any] | None) -> bool:
    return r is not None and r.get("failure_class") not in (
        "api_error",
        "env_error",
        "internal_error",
    )


def _boot(pairs: list[tuple[float, float]], n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Ratio estimator sum(dsteps) / sum(removed ktok), bootstrap over tasks."""
    if not pairs or sum(p[1] for p in pairs) <= 0:
        return {"n": len(pairs), "lambda": None}
    rng = random.Random(seed)

    def est(ps: list[tuple[float, float]]) -> float:
        den = sum(p[1] for p in ps)
        return sum(p[0] for p in ps) / den if den else 0.0

    bs = sorted(est([rng.choice(pairs) for _ in pairs]) for _ in range(n))
    return {
        "n": len(pairs),
        "lambda": round(est(pairs), 5),
        "ci95": [round(bs[int(0.025 * n)], 5), round(bs[int(0.975 * n) - 1], 5)],
        "mean_dsteps": round(statistics.mean(p[0] for p in pairs), 3),
        "mean_removed_ktok": round(statistics.mean(p[1] for p in pairs), 3),
    }


def fit_penalty(
    lite_dir: Path,
    lite_raw: dict[str, Trajectory],
    long_distil: list[Trajectory],
    long_dir: Path,
    tm: TokenModel,
    tok_per_char: float,
) -> dict[str, Any]:
    """Extra agent steps per 1k tokens of information removed, from the paired live runs.

    lambda = sum(steps_arm - steps_plain) / sum(removed ktok), per arm, with a task
    bootstrap. It is an association across tasks, not a causal estimate. Removed tokens:

    * distil (the band): its trajectory replayed through the served adapter (long run);
      on Lite, where distil's transcripts were not kept, the same replay on another arm's
      raw trajectory of the same task (a proxy for what distil saw);
    * selective: chars it pruned (its own telemetry); provider-cm: cleared input tokens
      averaged per applied call (every request re-applies the clear, so the sum overcounts).
      Both delete WITHOUT a recovery handle, so they are reported as a stress case only.
    """
    lite = _rows(lite_dir)
    ids = sorted({i for a, i in lite if a == "plain"})
    expand = expand_tool_tokens(tm)
    out: dict[str, Any] = {}
    sel, cm, dlite = [], [], []
    for i in ids:
        p = lite.get(("plain", i))
        if not _ok(p):
            continue
        assert p is not None
        s = lite.get(("selective", i))
        if _ok(s) and s and s.get("arm_stats"):
            st = s["arm_stats"]
            rem = (st.get("chars_before", 0) - st.get("chars_after", 0)) * tok_per_char / 1000
            sel.append((float(s["steps"] - p["steps"]), rem))
        c = lite.get(("provider-cm", i))
        if _ok(c) and c and c.get("arm_stats"):
            st = c["arm_stats"]
            k = st.get("applied_calls", 0)
            rem = (st.get("cleared_input_tokens", 0) / k / 1000) if k else 0.0
            cm.append((float(c["steps"] - p["steps"]), rem))
        d = lite.get(("distil", i))
        raw = lite_raw.get(i)
        if _ok(d) and d and raw is not None:
            r = simulate(raw, DistilServed(False), provider("anthropic"), tm, expand_tokens=expand)
            dlite.append((float(d["steps"] - p["steps"]), r.removed / 1000))
    longr = _rows(long_dir)
    dlong = []
    for t in long_distil:
        p = longr.get(("plain", t.id))
        d = longr.get(("distil", t.id))
        if not (_ok(p) and _ok(d)) or (d and d.get("stop") == "max_tokens"):
            continue
        assert p is not None and d is not None
        r = simulate(t, DistilServed(False), provider("anthropic"), tm, expand_tokens=expand)
        dlong.append((float(d["steps"] - p["steps"]), r.removed / 1000))
    out["lite_distil_proxy"] = _boot(dlite)
    out["long_distil"] = _boot(dlong)
    out["distil_pooled"] = _boot(dlite + dlong)
    out["lite_selective"] = _boot(sel)
    out["lite_provider_cm"] = _boot(cm)
    dp = out["distil_pooled"]
    out["band"] = {
        "low": 0.0,
        "mid": max(0.0, dp["lambda"] or 0.0),
        "high": max(0.0, dp["ci95"][1] if dp.get("ci95") else 0.0),
        "stress": max(0.0, out["lite_selective"]["lambda"] or 0.0),
        "unit": "extra agent steps per 1k tokens of information removed",
        "source": "low = no behaviour change; mid/high = distil's paired estimate and its "
        "upper 95% bound; stress = selective-context (irrecoverable deletion)",
    }
    return out


# ----------------------------------------------------------------------------- evaluation


def _tok_per_char(trajs: list[Trajectory], tm: TokenModel) -> float:
    toks = chars = 0.0
    for t in trajs:
        for m in t.messages:
            if m.get("role") == "user":
                for b in blocks(m):
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        txt = block_text(b)
                        toks += tm.count(txt)
                        chars += len(txt)
    return toks / chars if chars else 0.25


def evaluate(
    corpora: dict[str, list[Trajectory]],
    specs: list[Spec],
    providers: tuple[str, ...],
    tm: TokenModel,
    framing: float,
    log: Any = sys.stderr,
) -> dict[str, Any]:
    expand = expand_tool_tokens(tm)
    agg: dict[str, Any] = {}
    for cname, trajs in corpora.items():
        res: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for spec in specs:
            pol = spec.make()
            t0 = time.time()
            for t in trajs:
                reqs, info = replay(t, pol, tm, expand)
                for pname in providers:
                    kw = {"framing": framing} if pname == "anthropic" else {}
                    prov = provider(pname, t.model.split("@")[0], **kw)
                    r = price(t, reqs, info, prov, tm)
                    a = res[pname][spec.name]
                    a["n"] += 1
                    a["usd"] += r.usd
                    a["removed"] += r.removed
                    a["violations"] += r.violations
                    a["requests"] += r.requests
                    a["read"] += r.usage.read
                    a["write"] += r.usage.write_5m + r.usage.write_1h
                    a["input"] += r.usage.input
                    out_usd = r.usage.output * prov.price.output / 1e6
                    a["step_usd"] += (r.usd - out_usd) / max(1, r.requests) + (
                        r.per_step_output * prov.price.output / 1e6
                    )
            pol.close()
            print(f"  {cname:>14} {spec.name:<22} {time.time() - t0:6.1f}s", file=log)
        agg[cname] = {p: {k: dict(v) for k, v in d.items()} for p, d in res.items()}
    return agg


def table(agg: dict[str, Any], band: dict[str, float]) -> dict[str, Any]:
    """Per corpus x provider: per-task $ (pure, and under the penalty band) vs plain."""
    out: dict[str, Any] = {}
    for cname, provs in agg.items():
        out[cname] = {}
        for pname, pols in provs.items():
            base = pols["plain"]["usd"] / pols["plain"]["n"]
            rows = {}
            for name, a in pols.items():
                n = a["n"]
                usd = a["usd"] / n
                rem_k = a["removed"] / n / 1000
                step = a["step_usd"] / n
                rows[name] = {
                    "usd_per_task": round(usd, 5),
                    "vs_plain_pure": round(usd / base - 1, 4),
                    "vs_plain_penalty": {
                        k: round((usd + band[k] * rem_k * step) / base - 1, 4)
                        for k in ("low", "mid", "high", "stress")
                    },
                    # the penalty at which the policy stops beating plain
                    "breakeven_lambda": round((base - usd) / (rem_k * step), 4)
                    if rem_k > 0 and step > 0 and usd < base
                    else None,
                    "removed_tokens_per_task": round(a["removed"] / n, 1),
                    "violations_per_task": round(a["violations"] / n, 4),
                    "prompt_tokens_per_request": round(
                        (a["read"] + a["write"] + a["input"]) / a["requests"], 1
                    ),
                    "cache_read_share": round(
                        a["read"] / max(1.0, a["read"] + a["write"] + a["input"]), 4
                    ),
                    "n": int(n),
                }
            out[cname][pname] = rows
    return out


def pareto(rows: dict[str, Any], key: str = "vs_plain_pure") -> list[str]:
    """Non-dominated policies: less information removed AND a lower $ vs plain."""
    pts = [(r["removed_tokens_per_task"], r[key], n) for n, r in rows.items()]
    front = [
        n
        for (x, y, n) in pts
        if not any((x2 <= x and y2 <= y) and (x2 < x or y2 < y) for (x2, y2, _) in pts)
    ]
    return sorted(front, key=lambda n: rows[n]["removed_tokens_per_task"])


def recommend(tab: dict[str, Any]) -> dict[str, Any]:
    """Per corpus x provider: the cheapest policy at the HIGH penalty with no must-keep
    violation (the fidelity constraint), and its effect vs plain and vs rtk-like."""
    out: dict[str, Any] = {}
    for cname, provs in tab.items():
        out[cname] = {}
        for pname, rows in provs.items():
            ok = {n: r for n, r in rows.items() if r["violations_per_task"] == 0}
            best = min(ok, key=lambda n: ok[n]["vs_plain_penalty"]["high"])
            b = rows[best]
            rtk = rows.get("rtk-like")
            out[cname][pname] = {
                "policy": best,
                "vs_plain_pure": b["vs_plain_pure"],
                "vs_plain_high_penalty": b["vs_plain_penalty"]["high"],
                "vs_rtk_like_pure": round(b["usd_per_task"] / rtk["usd_per_task"] - 1, 4)
                if rtk
                else None,
                "pareto_pure": pareto(rows),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.policysim")
    ap.add_argument("--lite", type=Path, required=True, help="swebench-outcome-300-h2h dir")
    ap.add_argument("--long", type=Path, required=True, help="swebench-verified-hard-max dir")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/policysim"))
    ap.add_argument("--headroom-python", default=None, help="python with headroom-ai installed")
    ap.add_argument("--claude-code", type=int, default=0, help="also replay N local sessions")
    ap.add_argument("--lite-limit", type=int, default=0, help="sample N Lite trajectories")
    ap.add_argument("--providers", default=",".join(PROVIDERS))
    a = ap.parse_args(argv)
    providers = tuple(p for p in a.providers.split(",") if p)

    lite_cal = load_harness(a.lite, ("rtk",), "swebench-lite", load_grades(a.lite))
    long_cal = load_harness(a.long, ("plain", "rtk"), "swebench-long", load_grades(a.long))
    long_distil = load_harness(a.long, ("distil",), "swebench-long", load_grades(a.long))
    tm, overhead, framing, cal = calibrate(lite_cal, long_cal, long_distil)

    # Raw trajectories: the harness stores history BEFORE an arm's transform, so these
    # carry unfiltered tool output (rtk's transcripts are already rtk-filtered).
    lite_raw = load_harness(a.lite, ("provider-cm", "selective"), "swebench-lite", {})
    if a.lite_limit:
        lite_raw = random.Random(0).sample(lite_raw, min(a.lite_limit, len(lite_raw)))
    long_raw = [t for t in long_cal + long_distil if t.arm in ("plain", "distil")]
    corpora: dict[str, list[Trajectory]] = {"swebench-lite": lite_raw, "swebench-long": long_raw}
    if a.claude_code:
        from distil.whatif import claude_projects_root

        corpora["claude-code"] = load_claude_code(claude_projects_root(), a.claude_code)
    for name, trajs in corpora.items():
        for t in trajs:
            t.overhead_tokens = round(overhead) if name != "claude-code" else 0
            allocate_hidden(t, tm) if t.output_tokens else None

    tpc = _tok_per_char(lite_raw, tm)
    by_id: dict[str, Trajectory] = {}
    for t in load_harness(a.lite, ("provider-cm",), "swebench-lite", {}):
        t.overhead_tokens = round(overhead)
        allocate_hidden(t, tm)
        by_id[t.id] = t
    pen = fit_penalty(a.lite, by_id, long_distil, a.long, tm, tpc)
    specs = build(a.headroom_python)
    agg = evaluate(corpora, specs, providers, tm, framing)
    tab = table(agg, pen["band"])
    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "cost": "$0: offline replay, no model or API calls",
        "corpora": {
            k: {"trajectories": len(v), "requests": sum(len(t.request_ends) for t in v)}
            for k, v in corpora.items()
        },
        "calibration": cal,
        "penalty": pen,
        "providers": list(providers),
        "results": tab,
        "recommendation": recommend(tab),
        "assumptions": [
            "token counts: a regex-piece model fitted to Anthropic billed totals; the same "
            "counts are priced under OpenAI/Gemini rules (their tokenizers differ)",
            "request times: uniform spacing over the task's wall time (harness has no "
            "per-step timestamps)",
            "Gemini implicit cache: 300 s life and hit probability 1.0 (undocumented: an "
            "upper bound on its caching); explicit-cache creation at the standard input rate",
            "pure-cost columns replay the SAME trajectory; behaviour (extra steps) enters "
            "only through the fitted re-run penalty band",
        ],
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "policysim.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out / 'policysim.json'}", file=sys.stderr)
    return 0
