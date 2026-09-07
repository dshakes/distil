#!/usr/bin/env python3
"""E9: per-turn certificate -> trajectory task-success — the composition analysis.

The decision-equivalence certificate (E2) bounds the *per-turn* decision-change rate at
``alpha``. Task success, however, is a *trajectory-level* property: an agent runs T turns
and succeeds or fails as a whole. A reviewer's sharpest question is therefore: what does a
per-turn guarantee imply for the trajectory outcome? This module answers it three ways and
validates against the committed E8 long-horizon data.

1. **Conservative composition (provable).** If each turn flips with probability <= alpha,
   the probability a T-turn trajectory *ever* flips is

       P(diverge) <= 1 - (1 - alpha)^T   <=   T * alpha            (union bound)

   This is the honest, guaranteed link from the certificate to the trajectory.

2. **It is vacuous at agentic horizons.** With the certified alpha = 0.08 and E8's mean
   T ~ 27, the bound is ~89% (union: 214%). So a per-turn certificate, composed naively,
   says essentially *nothing* about a long trajectory — which formally explains why E7/E8
   find the proxy does not transfer once compression is aggressive (large alpha).

3. **Why reality is far better, and what it implies.** The *observed* outcome-divergence
   between the relevance-gated condition and full context in E8 is only ~14%. Inverting the
   composition gives an **effective number of consequential turns**

       k = d / alpha            (linear)   or   k = ln(1-d)/ln(1-alpha)   (exact)

   ~1.8 out of ~27. Of a long coding trajectory, fewer than two turns are outcome-
   determining; the rest are exploration the agent can get "wrong" (or have compressed)
   without changing the result, because (a) the reversible tier lets it recover, and
   (b) the gate never compresses the active working set where the consequential decisions
   live. The trajectory guarantee becomes tight exactly when per-turn equivalence is
   certified on the *consequential* turns — which is what the relevance-gate targets.

HONEST SCOPE: ``alpha`` here is the per-turn decision-change rate certified on the E2
SWE-bench_Lite localization corpus, not re-measured on the E8 ReAct workload (the two runs
are unpaired, so a per-turn paired flip-rate is not recoverable from E8 alone). The
composition is therefore parametric in ``alpha``; we plug in the certified operating point
as the reference and report ``k`` as a descriptive quantity, not a new guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from benchmarks.outpath import add_out_args, resolve_out

ROOT = Path(__file__).resolve().parents[1]
LH = ROOT / "docs/paper/results/swe_e2e_longhorizon"

# Per-turn decision-change rate at distil's certified operating point (E2, alpha=0.15
# budget, realized 8.0% out-of-sample on 500 SWE-bench_Lite splits).
DEFAULT_ALPHA = 0.08


def _resolved_ids(condition: str) -> set[str]:
    rep = LH / "reports" / f"distil-lh-{condition}.lh_{condition}.json"
    return set(json.loads(rep.read_text())["resolved_ids"])


def _instance_turns(condition: str) -> dict[str, int]:
    """Per-instance turn counts. Prefers the raw prediction dump; falls back to
    the committed compact summary (`trajectory_bound_inputs.json`) — the raw
    per-instance dumps are distributed via release artifacts, not git (see
    docs/paper/results/README.md)."""
    path = LH / "predictions" / f"{condition}.jsonl"
    if path.exists():
        out: dict[str, int] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["instance_id"]] = d.get("_run", {}).get("turns", 0)
        return out
    summary = json.loads((LH / "trajectory_bound_inputs.json").read_text())
    return dict(summary["conditions"][condition])


def analyze(
    alpha: float = DEFAULT_ALPHA,
    candidate: str = "distil_gated",
    reference: str = "full",
) -> dict[str, float]:
    """Compose the per-turn certificate to a trajectory statement and fit ``k`` from E8."""
    cand_turns = _instance_turns(candidate)
    mean_turns = st.mean(cand_turns.values())
    rc, rr = _resolved_ids(candidate), _resolved_ids(reference)
    ids = set(cand_turns) & set(_instance_turns(reference))
    diverged = sum(1 for i in ids if (i in rr) != (i in rc))
    d = diverged / len(ids)

    naive = 1 - (1 - alpha) ** mean_turns
    union = mean_turns * alpha
    k_exact = math.log(1 - d) / math.log(1 - alpha) if 0 < d < 1 else float("nan")
    k_linear = d / alpha
    return {
        "alpha": alpha,
        "mean_turns": mean_turns,
        "n": len(ids),
        "diverged": diverged,
        "observed_divergence": d,
        "naive_bound": naive,
        "union_bound": union,
        "k_consequential_exact": k_exact,
        "k_consequential_linear": k_linear,
        "consequential_fraction": k_linear / mean_turns,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--candidate", default="distil_gated")
    ap.add_argument("--reference", default="full")
    add_out_args(ap, LH / "trajectory_bound.json")
    args = ap.parse_args()
    out = resolve_out(args, LH / "trajectory_bound.json")
    r = analyze(args.alpha, args.candidate, args.reference)
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"alpha={r['alpha']:.3f}  mean_turns={r['mean_turns']:.1f}  n={r['n']}")
    print(
        f"observed trajectory divergence ({args.candidate} vs {args.reference}): "
        f"{r['observed_divergence'] * 100:.1f}%  ({r['diverged']}/{r['n']})"
    )
    print(
        f"naive composition 1-(1-a)^T = {r['naive_bound'] * 100:.1f}%  "
        f"(union Ta = {r['union_bound'] * 100:.0f}%)  -> vacuous"
    )
    print(
        f"effective consequential turns k = {r['k_consequential_linear']:.2f} "
        f"(exact {r['k_consequential_exact']:.2f}) of {r['mean_turns']:.0f} "
        f"= {r['consequential_fraction'] * 100:.1f}% of the trajectory"
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
