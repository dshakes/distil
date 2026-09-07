#!/usr/bin/env python3
"""E10: the Trajectory Decision-Equivalence Certificate.

E2 certifies the *per-turn* decision-change rate; E9 shows that bound does not naively
compose to a trajectory. This closes the gap with a guarantee at the unit users actually
care about — the **whole run** — using the same distribution-free Learn-Then-Test /
Hoeffding–Bentkus engine, lifted from per-turn to per-trajectory.

For a compressed condition vs full context, scored on the same N instances, define per
trajectory i:
  * **divergence** Lᵈᵢ = 1[outcome differs from full]  (the decision-equivalence contract)
  * **harm**       Lʰᵢ = 1[full resolved ∧ compressed did not]  (compression *cost* a task)

The certificate is the (1−δ) upper confidence bound on each rate
(:func:`distil.conformal.certified_risk_bound`):

    with confidence 1−δ,  P(trajectory diverges from full)  ≤  β        (exchangeable tasks)

We then *prove it out-of-sample* exactly as E2 does: over many random calibration/test
splits, certify β on the calibration half and check the realized rate on the disjoint test
half stays ≤ β. Coverage must hold ≥ 1−δ. Honest scope: the guarantee is conditional on
exchangeability with this distribution (SWE-bench Verified, this agent + model).

Runs offline on the committed E8 outcomes — no API, no Docker.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

from benchmarks.outpath import add_out_args, resolve_out
from distil.conformal import certified_risk_bound

ROOT = Path(__file__).resolve().parents[1]
LH = ROOT / "docs/paper/results/swe_e2e_longhorizon"


def _resolved_ids(condition: str) -> set[str]:
    rep = glob.glob(str(ROOT / f"distil-lh-{condition}.*.json"))
    path = rep[0] if rep else str(LH / "reports" / f"distil-lh-{condition}.lh_{condition}.json")
    return set(json.loads(Path(path).read_text())["resolved_ids"])


def _all_ids() -> list[str]:
    return json.loads((ROOT / "docs/paper/results/swe_e2e/sample_500.json").read_text())[
        "instance_ids"
    ]


def _losses(reference: str, candidate: str) -> tuple[list[int], list[int]]:
    """Per-trajectory divergence and harm losses for ``candidate`` vs ``reference``."""
    ref, cand = _resolved_ids(reference), _resolved_ids(candidate)
    ids = _all_ids()
    divergence = [int((i in ref) != (i in cand)) for i in ids]
    harm = [int((i in ref) and (i not in cand)) for i in ids]
    return divergence, harm


def oos_coverage(
    losses: list[int], *, delta: float, frac_cal: float = 0.5, reps: int = 1000, seed: int = 1729
) -> dict[str, float]:
    """Out-of-sample coverage proof: certify β on a calibration split, check the disjoint
    test split's realized rate ≤ β. Returns coverage (fraction of splits where it holds —
    must be ≥ 1−δ), and mean certified β vs mean realized test rate."""
    rng = random.Random(seed)
    n = len(losses)
    n_cal = int(n * frac_cal)
    held = covered = 0
    sum_beta = sum_test = 0.0
    for _ in range(reps):
        idx = list(range(n))
        rng.shuffle(idx)
        cal = [losses[i] for i in idx[:n_cal]]
        test = [losses[i] for i in idx[n_cal:]]
        if not test:
            continue
        beta = certified_risk_bound(sum(cal) / len(cal), len(cal), delta)
        test_rate = sum(test) / len(test)
        held += 1
        covered += int(test_rate <= beta)
        sum_beta += beta
        sum_test += test_rate
    return {
        "coverage": round(covered / held, 4) if held else 0.0,
        "target": round(1 - delta, 4),
        "mean_certified_beta": round(sum_beta / held, 4) if held else 1.0,
        "mean_test_rate": round(sum_test / held, 4) if held else 0.0,
        "reps": held,
    }


def certificate(reference: str, candidate: str, *, delta: float = 0.05) -> dict:
    div, harm = _losses(reference, candidate)
    n = len(div)
    out = {"candidate": candidate, "reference": reference, "n": n, "delta": delta}
    for name, losses in (("divergence", div), ("harm", harm)):
        rhat = sum(losses) / n
        out[name] = {
            "empirical_rate": round(rhat, 4),
            "certified_bound": round(certified_risk_bound(rhat, n, delta), 4),
            "oos": oos_coverage(losses, delta=delta),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default="full")
    ap.add_argument("--candidate", default="distil_gated")
    ap.add_argument("--delta", type=float, default=0.05)
    add_out_args(ap, LH / "trajectory_certificate.json")
    args = ap.parse_args()
    out = resolve_out(args, LH / "trajectory_certificate.json")
    cert = certificate(args.reference, args.candidate, delta=args.delta)
    out.write_text(json.dumps(cert, indent=2) + "\n")
    print(
        f"=== Trajectory Decision-Equivalence Certificate: {args.candidate} vs {args.reference} ==="
    )
    print(f"n={cert['n']}  confidence 1−δ={1 - args.delta:.2f}")
    for name in ("divergence", "harm"):
        c = cert[name]
        o = c["oos"]
        print(
            f"  {name:10}: rate {c['empirical_rate'] * 100:.1f}%  →  certified ≤ "
            f"{c['certified_bound'] * 100:.1f}% (conf {1 - args.delta:.0%});  "
            f"OOS coverage {o['coverage'] * 100:.1f}% (target {o['target'] * 100:.0f}%, "
            f"β̄={o['mean_certified_beta'] * 100:.1f}% vs test {o['mean_test_rate'] * 100:.1f}%)"
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
