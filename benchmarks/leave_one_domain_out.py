#!/usr/bin/env python3
"""E3: leave-one-domain-out — does the certificate survive distribution shift?

E2 and E10 certify a risk bound under **exchangeability**: calibration and test items
are drawn from one pool and split at random. Deployments violate that. The agent that
was calibrated on one codebase gets pointed at another, and the honest question is
whether the certificate still holds when the held-out data is a *different domain*
rather than a random half.

This is that stress test, run offline on the committed E8 artifacts — no API calls, no
Docker, no model. The unit is the **trajectory**, exactly as in E10
(:mod:`benchmarks.trajectory_certificate`): for each SWE-bench Verified instance and a
compressed condition vs. full context,

  * divergence  L^d_i = 1[outcome differs from full]
  * harm        L^h_i = 1[full resolved and compressed did not]

Both labels come from the **official SWE-bench harness**, which is deterministic and
identical across conditions — so unlike the per-turn corpus (τ-bench graded by gpt-4o,
SWE graded by Claude), there is no cross-grader confound to explain away a failure.

**Domain = source repository.** SWE-bench instance ids are ``<org>__<repo>-<n>``, so the
500 instances partition into 12 repositories with genuinely different codebases, file
layouts, test conventions, and issue styles. Repository is the strongest domain label
the committed artifacts actually carry; we do not invent one.

For each repository r we calibrate on the other 11 (n_cal = 500 - n_r), certify the
(1-delta) Hoeffding--Bentkus upper bound beta on the calibration risk, and check the
realized risk on the held-out repository against it. Coverage is the fraction of
repositories where the bound held.

To separate *domain shift* from *small-sample noise* we also run a matched control: for
each repository, many random held-out sets of the **same size** drawn from the whole
pool. If a repository fails where a same-sized random set almost never does, the failure
is the shift, not the n.

Usage::

    python benchmarks/leave_one_domain_out.py
    python benchmarks/report_to_latex.py \\
        docs/paper/results/leave_one_domain_out.json --only loo loomacros
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil.conformal import certified_risk_bound  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LH = ROOT / "docs/paper/results/swe_e2e_longhorizon"
DEFAULT_OUT = ROOT / "docs/paper/results/leave_one_domain_out.json"


def _resolved_ids(condition: str) -> set[str]:
    path = LH / "reports" / f"distil-lh-{condition}.lh_{condition}.json"
    return set(json.loads(path.read_text())["resolved_ids"])


def _all_ids() -> list[str]:
    return json.loads((ROOT / "docs/paper/results/swe_e2e/sample_500.json").read_text())[
        "instance_ids"
    ]


def domain_of(instance_id: str) -> str:
    """SWE-bench ids are ``<org>__<repo>-<number>``; the repo is the domain label."""
    return instance_id.split("__", 1)[1].rsplit("-", 1)[0]


def losses(reference: str, candidate: str) -> dict[str, dict[str, int]]:
    """Per-instance divergence and harm losses, keyed by instance id.

    Mirrors ``trajectory_certificate._losses`` so E3 rests on the identical labels E10
    reports; the only difference is that we keep the ids instead of collapsing to lists.
    """
    ref, cand = _resolved_ids(reference), _resolved_ids(candidate)
    return {
        i: {
            "divergence": int((i in ref) != (i in cand)),
            "harm": int((i in ref) and (i not in cand)),
        }
        for i in _all_ids()
    }


def _turns(condition: str) -> dict[str, int]:
    inputs = json.loads((LH / "trajectory_bound_inputs.json").read_text())
    return inputs["conditions"].get(condition, {})


def _dispersion(risks: list[float], sizes: list[int], pooled: float) -> float:
    """Size-weighted between-domain spread of the realized risk — the chi-square-like
    statistic a permutation test uses to ask whether domains differ at all."""
    return sum(n * (r - pooled) ** 2 for r, n in zip(risks, sizes))


def _permutation_reference(
    pool: list[int], sizes: list[int], *, delta: float, reps: int, rng: random.Random
) -> tuple[list[list[float]], list[float], list[float]]:
    """Re-partition the same losses into same-sized blocks at random, ``reps`` times.

    This is the exchangeability null: identical block sizes, identical marginal risk,
    domain labels destroyed. Returns per-block null risk samples, per-block coverage
    (was the bound certified on the complement respected?), and the null dispersion
    samples used for the global test.
    """
    n = len(pool)
    pooled = sum(pool) / n
    idx = list(range(n))
    null_risks: list[list[float]] = [[] for _ in sizes]
    null_cov = [0] * len(sizes)
    null_disp: list[float] = []
    total = sum(pool)
    for _ in range(reps):
        rng.shuffle(idx)
        start = 0
        risks = []
        for b, size in enumerate(sizes):
            block = [pool[i] for i in idx[start : start + size]]
            start += size
            hits = sum(block)
            risk = hits / size
            risks.append(risk)
            null_risks[b].append(risk)
            # bound certified on everything outside this block, exactly as LOO does
            beta = certified_risk_bound((total - hits) / (n - size), n - size, delta)
            null_cov[b] += int(risk <= beta)
        null_disp.append(_dispersion(risks, sizes, pooled))
    return null_risks, [c / reps for c in null_cov], null_disp


def _pvalue(observed: float, null: list[float]) -> float:
    """One-sided permutation p-value with the standard +1 correction."""
    return (1 + sum(1 for x in null if x >= observed)) / (1 + len(null))


def leave_one_domain_out(
    reference: str = "full",
    candidate: str = "distil_gated",
    *,
    delta: float = 0.05,
    control_reps: int = 400,
    seed: int = 1729,
) -> dict:
    per_item = losses(reference, candidate)
    turns = _turns(candidate)
    ids = list(per_item)
    domains: dict[str, list[str]] = {}
    for i in ids:
        domains.setdefault(domain_of(i), []).append(i)

    rng = random.Random(seed)
    order = sorted(domains, key=lambda d: -len(domains[d]))
    sizes = [len(domains[d]) for d in order]
    rows: list[dict] = [
        {
            "domain": d,
            "n_calib": len(ids) - len(domains[d]),
            "n_test": len(domains[d]),
            "mean_turns": (
                round(statistics.mean([turns[i] for i in domains[d] if i in turns]), 1)
                if any(i in turns for i in domains[d])
                else None
            ),
        }
        for d in order
    ]

    summary = {"reference": reference, "candidate": candidate, "delta": delta, "n": len(ids)}
    for loss in ("divergence", "harm"):
        pool = [per_item[i][loss] for i in ids]
        pooled = sum(pool) / len(pool)
        null_risks, null_cov, null_disp = _permutation_reference(
            pool, sizes, delta=delta, reps=control_reps, rng=rng
        )
        realized_all = []
        for b, d in enumerate(order):
            test = [per_item[i][loss] for i in domains[d]]
            cal = [per_item[i][loss] for i in ids if i not in set(domains[d])]
            beta = certified_risk_bound(sum(cal) / len(cal), len(cal), delta)
            realized = sum(test) / len(test)
            realized_all.append(realized)
            rows[b][loss] = {
                "calib_risk": round(sum(cal) / len(cal), 4),
                "certified_bound": round(beta, 4),
                "realized_risk": round(realized, 4),
                "held": bool(realized <= beta),
                # exchangeable control: a same-sized RANDOM block, same certify-then-check
                "exchangeable_coverage": round(null_cov[b], 4),
                # is this domain worse than a same-sized random block would be?
                "shift_pvalue": round(_pvalue(realized, null_risks[b]), 4),
                # the smallest realized risk this domain could have shown and still been
                # distinguishable from chance at p<0.05 — i.e. the experiment's power
                "detectable_at_05": round(
                    sorted(null_risks[b])[min(len(null_risks[b]) - 1, int(0.95 * control_reps))],
                    4,
                ),
            }
        held = [r for r in rows if r[loss]["held"]]
        big = [r for r in rows if r["n_test"] >= 10]
        observed_disp = _dispersion(realized_all, sizes, pooled)
        summary[loss] = {
            "pooled_risk": round(pooled, 4),
            "domains": len(rows),
            "domains_held": len(held),
            "coverage": round(len(held) / len(rows), 4),
            "coverage_n10": (
                round(sum(r[loss]["held"] for r in big) / len(big), 4) if big else None
            ),
            "domains_n10": len(big),
            "worst_domain": max(rows, key=lambda r: r[loss]["realized_risk"])["domain"],
            "worst_realized": max(r[loss]["realized_risk"] for r in rows),
            "spread": round(max(realized_all) - min(realized_all), 4),
            # what coverage the SAME procedure gets on exchangeable blocks of the same
            # sizes — the honest yardstick for the leave-one-domain-out coverage above
            "mean_exchangeable_coverage": round(statistics.mean(null_cov), 4),
            # global test: do the domains differ at all, beyond same-size sampling noise?
            "dispersion": round(observed_disp, 4),
            "dispersion_pvalue": round(_pvalue(observed_disp, null_disp), 4),
            "permutation_reps": control_reps,
        }
    return {"summary": summary, "domains": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default="full")
    ap.add_argument("--candidate", default="distil_gated")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--control-reps", type=int, default=2000, help="permutation reps")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    res = leave_one_domain_out(
        args.reference,
        args.candidate,
        delta=args.delta,
        control_reps=args.control_reps,
    )
    args.out.write_text(json.dumps(res, indent=2) + "\n")

    s = res["summary"]
    print(f"=== E3 leave-one-domain-out: {args.candidate} vs {args.reference} ===")
    print(f"n={s['n']}  domains={s['divergence']['domains']}  confidence 1-d={1 - args.delta:.2f}")
    for loss in ("divergence", "harm"):
        d = s[loss]
        print(
            f"  {loss:10}: pooled {d['pooled_risk'] * 100:.1f}%  "
            f"LOO coverage {d['domains_held']}/{d['domains']} ({d['coverage'] * 100:.0f}%)  "
            f"exchangeable control {d['mean_exchangeable_coverage'] * 100:.1f}%  "
            f"worst {d['worst_domain']} @ {d['worst_realized'] * 100:.1f}%  "
            f"dispersion p={d['dispersion_pvalue']:.4f}"
        )
    for r in res["domains"]:
        dv, hm = r["divergence"], r["harm"]
        print(
            f"    {r['domain']:14} n={r['n_test']:3}  "
            f"div {dv['realized_risk'] * 100:5.1f}% vs b={dv['certified_bound'] * 100:.1f}% "
            f"{'FAIL' if not dv['held'] else 'ok  '} p={dv['shift_pvalue']:.3f}   "
            f"harm {hm['realized_risk'] * 100:5.1f}% vs b={hm['certified_bound'] * 100:.1f}% "
            f"{'FAIL' if not hm['held'] else 'ok  '} p={hm['shift_pvalue']:.3f}"
        )
    print(f"-> {args.out}")


def _selftest() -> None:
    """Guard the two things that would silently invalidate E3: the domain parser and
    agreement with the labels E10 already publishes."""
    assert domain_of("astropy__astropy-12907") == "astropy"
    assert domain_of("scikit-learn__scikit-learn-10297") == "scikit-learn"
    assert domain_of("pydata__xarray-4094") == "xarray"
    per_item = losses("full", "distil_gated")
    cert = json.loads((LH / "trajectory_certificate.json").read_text())
    n = len(per_item)
    for loss in ("divergence", "harm"):
        rate = sum(v[loss] for v in per_item.values()) / n
        assert abs(rate - cert[loss]["empirical_rate"]) < 5e-4, (loss, rate)
    res = leave_one_domain_out(control_reps=20)
    assert sum(r["n_test"] for r in res["domains"]) == n
    print("selftest ok")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
