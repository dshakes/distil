#!/usr/bin/env python3
"""Aggregate E7 predictions + harness scores into the canonical results JSON + paper macros.

Inputs (per condition in {full, distil_trunc500, llmlingua2}):
  * ``predictions/<cond>.jsonl`` — one row/instance with the agent's patch, the proxy's
    compression stats, and the upstream token ``usage`` (written by ``run_agent``);
  * ``scores/<cond>.json`` — the official-harness pass/fail aggregation (written by
    ``score`` from the harness's own report).

Outputs:
  * ``docs/paper/results/swe_bench_verified_e2e.json`` — pass@1 + Wilson 95% CI, cost
    (distil's own pricing catalog), realised compression, and a full per-instance
    breakdown for every condition. Every number traces to a file on disk.
  * ``docs/paper/generated/swe_e2e_macros.tex`` — LaTeX macros consumed by the paper.

Cost uses :mod:`distil.pricing` (claude-sonnet-4-6 = $3/$15 per Mtok, cache write 1.25x,
cache read 0.10x) with the Anthropic usage breakdown (fresh input / cache write / cache
read / output), so it reflects the prompt-caching discount the agent actually got.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.outpath import add_out_args, resolve_out  # noqa: E402
from benchmarks.swe_bench_e2e.stats import noninferiority_paired, wilson_ci  # noqa: E402
from distil.pricing import get as get_pricing  # noqa: E402

CONDITIONS = [
    ("full", "A. full context"),
    ("distil_trunc500", "B. distil (trunc@500)"),
    ("llmlingua2", "C. LLMLingua-2"),
    ("headroom", "F. Headroom"),
    ("distil_expand", "D. distil (reversible + distil_expand)"),
    ("distil_gated", "E. distil (reversible, relevance-gated)"),
]
MODEL = "claude-sonnet-4-6"
# Pre-specified non-inferiority margin for pass@1 vs full context (5 pp absolute). Declared
# before scoring; we report the paired difference + 95% CI so any margin can be applied.
NI_MARGIN = 0.05


def _cost_usd(usage: dict[str, int]) -> float:
    p = get_pricing(MODEL)
    fresh_in = usage.get("usage_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    out = usage.get("usage_output_tokens", 0)
    return (
        fresh_in * p.input
        + cache_write * p.cache_write
        + cache_read * p.cache_read
        + out * p.output
    )


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["instance_id"]] = r
    return rows


def aggregate_condition(cond: str, results_dir: Path, all_ids: list[str]) -> dict[str, Any]:
    preds = load_predictions(results_dir / "predictions" / f"{cond}.jsonl")
    score_path = results_dir / "scores" / f"{cond}.json"
    score = (
        json.loads(score_path.read_text())
        if score_path.exists()
        else {"per_instance": {}, "summary": {}}
    )
    per_score = score.get("per_instance", {})

    resolved = 0
    total_cost = 0.0
    blocks_seen = blocks_comp = chars_before = chars_after = 0
    in_tok = out_tok = 0
    per_instance = {}
    for iid in all_ids:
        pr = preds.get(iid, {})
        comp = pr.get("_compress", {})
        sc = per_score.get(iid, {})
        is_resolved = bool(sc.get("resolved", False))
        resolved += int(is_resolved)
        cost = _cost_usd(comp)
        total_cost += cost
        blocks_seen += comp.get("blocks_seen", 0)
        blocks_comp += comp.get("blocks_compressed", 0)
        chars_before += comp.get("chars_before", 0)
        chars_after += comp.get("chars_after", 0)
        in_tok += comp.get("usage_input_tokens", 0)
        out_tok += comp.get("usage_output_tokens", 0)
        per_instance[iid] = {
            "resolved": is_resolved,
            "empty_patch": pr.get("_empty_patch", True),
            "agent_status": pr.get("_run", {}).get("status", pr.get("_error", "missing")),
            "agent_seconds": pr.get("_run", {}).get("seconds"),
            "blocks_compressed": comp.get("blocks_compressed", 0),
            "blocks_seen": comp.get("blocks_seen", 0),
            "usage_input_tokens": comp.get("usage_input_tokens", 0),
            "usage_output_tokens": comp.get("usage_output_tokens", 0),
            "cost_usd": round(cost, 4),
            "harness_completed": sc.get("completed"),
            "harness_errored": sc.get("errored"),
        }

    n = len(all_ids)
    lo, hi = wilson_ci(resolved, n)
    ctx_reduction = (1 - chars_after / chars_before) if chars_before else 0.0
    return {
        "condition": cond,
        "n": n,
        "resolved": resolved,
        "pass_at_1": round(resolved / n, 4) if n else 0.0,
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "cost_usd": round(total_cost, 2),
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "blocks_seen": blocks_seen,
        "blocks_compressed": blocks_comp,
        "context_char_reduction": round(ctx_reduction, 4),
        "per_instance": per_instance,
    }


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (b, c)."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, j) for j in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_analysis(conditions: list[dict[str, Any]], all_ids: list[str]) -> list[dict[str, Any]]:
    """McNemar exact paired tests across conditions (same instances scored 3 ways).

    More powerful than comparing independent Wilson intervals because the 50 instances
    are common to every condition. Reports, for each ordered pair (a, b): how many a
    resolved that b did not (``a_only``) and vice versa, and the exact two-sided p-value.
    """
    res = {
        c["condition"]: {i: c["per_instance"][i]["resolved"] for i in all_ids} for c in conditions
    }
    pairs = [
        ("full", "distil_trunc500"),
        ("full", "llmlingua2"),
        ("llmlingua2", "distil_trunc500"),
        ("full", "distil_expand"),
        ("distil_trunc500", "distil_expand"),
        ("full", "distil_gated"),
        ("distil_expand", "distil_gated"),
        ("llmlingua2", "distil_gated"),
        ("full", "headroom"),
        ("headroom", "distil_gated"),
        ("distil_expand", "headroom"),
    ]
    # Pre-specified non-inferiority margin (proportion). A McNemar p only tells us whether
    # a difference exists; for pairs against full context we also ask the deployment-relevant
    # question — is b no worse than a by more than NI_MARGIN? — via a paired NI test.
    ni_against_full = {"full"}
    out = []
    for a, b in pairs:
        a_only = sum(1 for i in all_ids if res[a][i] and not res[b][i])
        b_only = sum(1 for i in all_ids if not res[a][i] and res[b][i])
        entry = {
            "a": a,
            "b": b,
            "a_only": a_only,
            "b_only": b_only,
            "discordant": a_only + b_only,
            "mcnemar_p": round(_mcnemar_exact(a_only, b_only), 4),
        }
        # b is the candidate, a the reference: a_only = candidate losses, b_only = gains.
        if a in ni_against_full:
            entry["noninferiority"] = noninferiority_paired(
                b=a_only, c=b_only, n=len(all_ids), margin=NI_MARGIN
            )
        out.append(entry)
    return out


def write_macros(agg: dict[str, Any], path: Path, prefix: str = "sweEseven") -> None:
    """Emit LaTeX macros for the paper (one set per condition + deltas).

    ``prefix`` namespaces every command so distinct experiments (E7 short-agent vs E8
    long-horizon) can both be ``\\input`` into the same document without redefinitions.
    """
    p = prefix
    by = {c["condition"]: c for c in agg["conditions"]}
    lines = ["% auto-generated by benchmarks/swe_bench_e2e/aggregate.py — do not edit by hand"]

    def pct(x: float) -> str:
        return f"{100 * x:.1f}"

    short = {
        "full": "Full",
        "distil_trunc500": "Distil",
        "llmlingua2": "Lingua",
        "headroom": "Headroom",
        "distil_expand": "Expand",
        "distil_gated": "Gated",
    }
    full_pass = by.get("full", {}).get("pass_at_1", 0.0)
    for cond, name in short.items():
        c = by.get(cond)
        if not c:
            continue
        lines += [
            f"\\newcommand{{\\{p}{name}Pass}}{{{pct(c['pass_at_1'])}\\%}}",
            f"\\newcommand{{\\{p}{name}Resolved}}{{{c['resolved']}}}",
            f"\\newcommand{{\\{p}{name}CIlo}}{{{pct(c['ci95_low'])}\\%}}",
            f"\\newcommand{{\\{p}{name}CIhi}}{{{pct(c['ci95_high'])}\\%}}",
            f"\\newcommand{{\\{p}{name}Cost}}{{\\${c['cost_usd']:.2f}}}",
            f"\\newcommand{{\\{p}{name}CtxRed}}{{{pct(c['context_char_reduction'])}\\%}}",
        ]
        # signed pass@1 delta vs full, in percentage points (B/C only)
        if cond != "full":
            delta = 100 * (c["pass_at_1"] - full_pass)
            lines.append(f"\\newcommand{{\\{p}{name}Delta}}{{{delta:+.1f}\\,pp}}")
    n = by.get("full", {}).get("n", 0)
    lines += [
        f"\\newcommand{{\\{p}N}}{{{n}}}",
        f"\\newcommand{{\\{p}Seed}}{{{agg.get('seed', 1729)}}}",
        f"\\newcommand{{\\{p}TotalCost}}{{\\${agg.get('total_cost_usd', 0):.2f}}}",
    ]

    def fmtp(p: float) -> str:
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    pair_macro = {
        ("full", "distil_trunc500"): "DistilVsFullP",
        ("full", "llmlingua2"): "LinguaVsFullP",
        ("llmlingua2", "distil_trunc500"): "DistilVsLinguaP",
        ("full", "distil_expand"): "ExpandVsFullP",
        ("distil_trunc500", "distil_expand"): "ExpandVsTruncP",
        ("full", "distil_gated"): "GatedVsFullP",
        ("distil_expand", "distil_gated"): "GatedVsExpandP",
        ("llmlingua2", "distil_gated"): "GatedVsLinguaP",
        ("full", "headroom"): "HeadroomVsFullP",
        ("headroom", "distil_gated"): "GatedVsHeadroomP",
        ("distil_expand", "headroom"): "HeadroomVsExpandP",
    }
    ni_macro = {
        ("full", "distil_expand"): "Expand",
        ("full", "distil_gated"): "Gated",
        ("full", "headroom"): "Headroom",
    }
    for pr in agg.get("paired_mcnemar", []):
        key = pair_macro.get((pr["a"], pr["b"]))
        if key:
            lines.append(f"\\newcommand{{\\{p}{key}}}{{{fmtp(pr['mcnemar_p'])}}}")
        # Non-inferiority (paired difference + 95% CI + margin) for candidates vs full.
        ni = pr.get("noninferiority")
        nk = ni_macro.get((pr["a"], pr["b"]))
        if ni and nk:
            lines += [
                f"\\newcommand{{\\{p}{nk}NIDelta}}{{{100 * ni['delta']:+.1f}\\,pp}}",
                f"\\newcommand{{\\{p}{nk}NICIlo}}{{{100 * ni['ci95_low']:+.1f}\\,pp}}",
                f"\\newcommand{{\\{p}{nk}NICIhi}}{{{100 * ni['ci95_high']:+.1f}\\,pp}}",
                f"\\newcommand{{\\{p}{nk}NIMargin}}{{{100 * ni['margin']:.0f}\\,pp}}",
                f"\\newcommand{{\\{p}{nk}NIVerdict}}{{{'non-inferior' if ni['noninferior'] else 'not established'}}}",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "docs/paper/results/swe_e2e")
    ap.add_argument("--sample", type=Path, default=ROOT / "docs/paper/results/swe_e2e/sample.json")
    add_out_args(ap, ROOT / "docs/paper/results/swe_bench_verified_e2e.json")
    ap.add_argument("--macros", type=Path, default=ROOT / "docs/paper/generated/swe_e2e_macros.tex")
    ap.add_argument("--wall-clock-seconds", type=float, default=None)
    # Overrides for non-E7 runs (e.g. the E8 long-horizon ReAct sweep on Haiku). Defaults
    # preserve the original E7/aider/Sonnet behaviour so existing invocations are unchanged.
    ap.add_argument("--model", default=MODEL, help="model for cost pricing + metadata")
    ap.add_argument("--agent", default="aider 0.86.2", help="agent label in output metadata")
    ap.add_argument(
        "--experiment",
        default="E7: SWE-bench Verified end-to-end task-success",
        help="experiment label in output metadata",
    )
    ap.add_argument(
        "--macro-prefix",
        default="sweEseven",
        help="LaTeX command namespace (use a distinct value per experiment to avoid redefinitions)",
    )
    args = ap.parse_args()

    MODEL = args.model  # _cost_usd() prices against this

    meta = json.loads(args.sample.read_text())
    all_ids = meta["instance_ids"]
    conditions = [aggregate_condition(c, args.results_dir, all_ids) for c, _ in CONDITIONS]
    total_cost = round(sum(c["cost_usd"] for c in conditions), 2)
    agg = {
        "experiment": args.experiment,
        "dataset": meta["dataset"],
        "split": meta["split"],
        "seed": meta["seed"],
        "n": len(all_ids),
        "agent": args.agent,
        "model": args.model,
        "temperature": 0.0,
        "harness": "swebench 4.1.0 (official run_evaluation, namespace=swebench, x86_64 under emulation)",
        "conditions": conditions,
        "paired_mcnemar": paired_analysis(conditions, all_ids),
        "total_cost_usd": total_cost,
        "wall_clock_seconds": args.wall_clock_seconds,
    }
    out = resolve_out(args, ROOT / "docs/paper/results/swe_bench_verified_e2e.json")
    out.write_text(json.dumps(agg, indent=2) + "\n")
    write_macros(agg, args.macros, prefix=args.macro_prefix)
    print(f"results -> {out}")
    print(f"macros  -> {args.macros}")
    for c in conditions:
        print(
            f"  {c['condition']:16s} pass@1={c['pass_at_1']:.3f} "
            f"[{c['ci95_low']:.3f},{c['ci95_high']:.3f}] "
            f"resolved={c['resolved']}/{c['n']} cost=${c['cost_usd']:.2f} "
            f"ctx_red={c['context_char_reduction']:.2f}"
        )
    print(f"  TOTAL cost=${total_cost:.2f}")


if __name__ == "__main__":
    main()
