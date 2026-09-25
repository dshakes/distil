"""Known-answer Monte Carlo for ``distil ab`` (distil/abtest) — offline, no API calls.

Every number docs/ab.html prints about the estimator comes from this script's
tracked output, ``benchmarks/results/abtest-montecarlo.json``:

* type-I error under 20 repeated looks and final-look coverage, session log-sd
  {0.6, 1.0, 1.5, 2.0} x holdout {5%, 50%}, N=4,000 — for the SHIPPED headline (capped
  empirical-Bernstein ratio) and, for comparison, the normal-mixture mSPRT it replaced;
* power of the headline at the shipped 5% holdout;
* bias when distil changes the DENOMINATOR (users re-ask more): the headline (cost per
  session) against the misleading cost per task;
* bias under a model change (explicit id change; silent) that adds 40% turns in both arms;
* the change detector at a 5% holdout: arm-blind vs holdout-only vs none;
* CUPED on the secondary (efficient) estimate; CUSUM false-alarm rates;
* sessions a 5% holdout needs to resolve a 10/20/30% effect with the headline method.

    python -m benchmarks.abtest_montecarlo                 # scratch output
    python -m benchmarks.abtest_montecarlo --write-tracked # the published artifact

The simulator :func:`sim` is also what ``tests/test_abtest.py`` uses.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from dataclasses import replace

from benchmarks.outpath import ROOT, add_out_args, resolve_out
from distil.abtest import DISTIL, HOLDOUT, stats
from distil.abtest.outcomes import SessionOutcome
from distil.abtest.report import HOLDOUT_H, summarize

TRACKED = ROOT / "benchmarks" / "results" / "abtest-montecarlo.json"


def sim(
    n: int,
    *,
    effect: float = -0.2,
    p: float = 0.5,
    sigma: float = 0.6,
    turns_mult: float = 1.0,
    model: str = "claude-opus-4-8",
    client: str = "claude-cli/2.1",
    t0: float = 1_000_000.0,
    seed: int = 0,
    ws_sd: float = 0.0,
    n_ws: int = 25,
    rate: float | None = None,
    prefix: str = "s",
    scale: float = 7.5,
    tasks: float = 1.0,
    distil_tasks_mult: float = 1.0,
    distil_cost_per_session: float | None = None,
) -> list[SessionOutcome]:
    """Sessions whose true relative effect on mean cost per task is exactly ``effect``.

    distil multiplies each session's cost by ``1+effect``; turns (~20, log-sd 0.3) are
    untouched by distil and scale with ``turns_mult`` — the model's behaviour — in BOTH
    arms. Per-session cost noise is lognormal (log-sd ``sigma``, mean-preserving), plus a
    per-workspace log offset with sd ``ws_sd`` (what CUPED can remove). ``scale`` is $ per
    turn: the default puts a session near $150, the scale of the maintainer's own.

    Treatment-affected denominators: ``tasks`` user turns per session in the holdout,
    times ``distil_tasks_mult`` under distil (compression making users re-ask), with
    ``distil_cost_per_session`` (if given) overriding ``effect`` as distil's true
    relative effect on cost PER SESSION — the headline's estimand.
    """
    rng = random.Random(seed)
    ws_eff = [rng.gauss(0, ws_sd) for _ in range(n_ws)]
    out = []
    for i in range(n):
        hold = rng.random() < p
        w = rng.randrange(n_ws)
        turns = max(1, round(20 * turns_mult * math.exp(rng.gauss(0, 0.3))))
        cost = scale * turns * math.exp(ws_eff[w] + rng.gauss(0, sigma) - sigma**2 / 2)
        n_tasks = max(1, round(tasks * math.exp(rng.gauss(0, 0.2))))
        if not hold:
            cost *= 1 + (effect if distil_cost_per_session is None else distil_cost_per_session)
            n_tasks = max(1, round(tasks * distil_tasks_mult * math.exp(rng.gauss(0, 0.2))))
        start = t0 + i * 1000.0
        out.append(
            SessionOutcome(
                sid=f"{prefix}{seed}-{i}",
                arm=HOLDOUT if hold else DISTIL,
                rate=p if rate is None else rate,
                start=start,
                end=start + 500.0,
                model=model,
                client=client,
                ws=f"w{w}",
                billing="metered",
                cost=cost,
                turns=turns,
                tasks=n_tasks if tasks != 1.0 or distil_tasks_mult != 1.0 else 1,
                in_tokens=1000 * turns,
                out_tokens=100 * turns,
                errors=0,
                expanded=0,
                wall_s=500.0,
                ended="ok",
                unpriced=0,
                src=start,
            )
        )
    return out


def _contrast(d: list[SessionOutcome]) -> stats.Contrast | None:
    return stats.ratio_contrast(
        [stats.Unit(o.cost or 0.0, 1, o.arm == DISTIL) for o in d], cuped=False
    )


def _mean_se(xs: list[float]) -> dict[str, float]:
    return {
        "mean": st.mean(xs),
        "se": st.stdev(xs) / math.sqrt(len(xs)),
        "sd": st.stdev(xs),
        "n": len(xs),
    }


def _robust(d: list[SessionOutcome], cap: float = stats.COST_CAP) -> stats.RobustRatio | None:
    return stats.robust_ratio(
        [o.cost or 0.0 for o in d if o.arm == DISTIL],
        [o.cost or 0.0 for o in d if o.arm == HOLDOUT],
        cap=cap,
        alpha=0.05,
    )


def type_one_table(reps: int, n: int = 4000, looks: int = 20) -> list[dict[str, float]]:
    """Null effect. 'ever' = rejected at any of the looks; 'coverage' = the final-look
    interval contains the null."""
    out = []
    for p in (0.05, 0.5):
        for sd in (0.6, 1.0, 1.5, 2.0):
            ever_r = ever_m = cov_r = cov_m = 0
            for rep in range(reps):
                data = sim(n, effect=0.0, p=p, sigma=sd, seed=10_000 + rep)
                hr = hm = False
                r = c = None
                for k in range(1, looks + 1):
                    part = data[: n * k // looks]
                    r, c = _robust(part), _contrast(part)
                    hr = hr or bool(r and r.significant)
                    hm = hm or bool(c and c.significant(0.05))
                ever_r += hr
                ever_m += hm
                cov_r += bool(r and not r.significant)
                cov_m += bool(c and not c.significant(0.05))
            out.append(
                {
                    "p": p,
                    "log_sd": sd,
                    "n": n,
                    "looks": looks,
                    "reps": reps,
                    "headline_type_one": ever_r / reps,
                    "headline_coverage": cov_r / reps,
                    "msprt_type_one": ever_m / reps,
                    "msprt_coverage": cov_m / reps,
                }
            )
    return out


def power(reps: int) -> list[dict[str, float]]:
    out = []
    for sd in (1.0, 1.5):
        for n in (4000, 20000):
            hits = cov = 0
            for rep in range(reps):
                r = _robust(sim(n, effect=-0.2, p=0.05, sigma=sd, seed=20_000 + rep))
                hits += bool(r and r.significant)
                cov += bool(r and r.lo < -0.2 < r.hi)
            out.append(
                {
                    "p": 0.05,
                    "log_sd": sd,
                    "n": n,
                    "effect": -0.2,
                    "power": hits / reps,
                    "coverage": cov / reps,
                }
            )
    return out


def denominator(reps: int) -> dict[str, float]:
    """distil: +10% cost per session, +25% tasks per session (users re-ask)."""
    head, cpt = [], []
    for rep in range(reps):
        d = sim(
            3000,
            seed=60_000 + rep,
            sigma=0.6,
            tasks=4,
            distil_tasks_mult=1.25,
            distil_cost_per_session=0.10,
        )
        s = summarize(d, rate=0.5, now=3e9)
        if s.cost is not None and s.mediators.get("cost_per_task") is not None:
            head.append(s.cost.rel)
            cpt.append(s.mediators["cost_per_task"].rel)  # type: ignore[union-attr]
    return {
        "true_cost_per_session": 0.10,
        "true_tasks_per_session": 0.25,
        "headline_mean": st.mean(head),
        "headline_bias": st.mean(head) - 0.10,
        "headline_bias_se": st.stdev(head) / math.sqrt(len(head)),
        "cost_per_task_mean": st.mean(cpt),
        "reps": len(head),
    }


def model_change(reps: int) -> dict[str, dict[str, float]]:
    res = {}
    for kind in ("explicit", "silent"):
        errs, cover, eras, naive = [], 0, 0, []
        for rep in range(reps):
            a = sim(600, seed=130_000 + rep)
            b = sim(
                600,
                seed=140_000 + rep,
                turns_mult=1.4,
                model="claude-opus-5" if kind == "explicit" else "claude-opus-4-8",
                t0=a[-1].start + 1000,
            )
            naive.append(
                st.mean(o.cost or 0 for o in b if o.arm == DISTIL)
                / st.mean(o.cost or 0 for o in a if o.arm == DISTIL)
                - 1
            )
            s = summarize(a + b, rate=0.5, now=3e9)
            if s.cost is None:
                continue
            errs.append(s.cost.rel + 0.2)
            cover += s.cost.lo < -0.2 < s.cost.hi
            eras += any(e.kind == "shift" for e in s.events)
        m = _mean_se(errs)
        res[kind] = {
            "bias": m["mean"],
            "bias_se": m["se"],
            "sd": m["sd"],
            "coverage": cover / len(errs),
            "estimated": len(errs),
            "reps": reps,
            "cusum_era_opened": eras,
            "before_after_distil_arm": st.mean(naive),
            "true_effect": -0.2,
            "turns_shift": 0.4,
            "sessions_per_model": 600,
        }
    return res


def detector(reps: int) -> dict[str, dict[str, float]]:
    """At the shipped 5% holdout (N=4,000, log-sd 1.0): which series should the change
    detector read? Headline = capped robust ratio on the era the detector leaves current."""

    def eras(d: list[SessionOutcome], mode: str) -> list[SessionOutcome]:
        if mode == "none":
            return d
        src = [o for o in d if mode == "arm_blind" or o.arm == HOLDOUT]
        h = 12.0 if mode == "arm_blind" else HOLDOUT_H  # the shipped setting
        cuts = stats.cusum_changepoints([math.log(o.cost or 1e-9) for o in src], h=h)
        if not cuts:
            return d
        last_alarm = src[cuts[-1][1]].start
        return [o for o in d if o.start > last_alarm]

    res: dict[str, dict[str, float]] = {}
    scenarios = {
        "stationary": ({}, -0.2),
        "distil_upgrade_-20_to_-40": ({"step": -0.4}, None),
        "silent_model_+40pct": ({"shift": 0.4}, -0.2),
    }
    for name, (kw, truth) in scenarios.items():
        for mode in ("arm_blind", "holdout_only", "none"):
            rels, n_eras = [], []
            for rep in range(reps):
                a = sim(2000, p=0.05, sigma=1.0, seed=80_000 + rep, effect=-0.2)
                b = sim(
                    2000,
                    p=0.05,
                    sigma=1.0,
                    seed=90_000 + rep,
                    effect=kw.get("step", -0.2),
                    turns_mult=1.0 + kw.get("shift", 0.0),
                    t0=a[-1].start + 1000,
                    prefix="b",
                )
                cur = eras(a + b, mode)
                n_eras.append(1 if len(cur) == 4000 else 2)
                r = _robust(cur, cap=1e12)
                if r is not None:
                    rels.append(r.rel)
            row = {"mean_eras": st.mean(n_eras), "headline_mean": st.mean(rels), "n": len(rels)}
            if truth is not None:
                row["bias"] = st.mean(rels) - truth
                row["bias_se"] = st.stdev(rels) / math.sqrt(len(rels))
            res[f"{name}/{mode}"] = row
    return res


def cuped(reps: int) -> dict[str, float]:
    vr, wr = [], []
    for rep in range(reps):
        d = sim(1000, seed=50_000 + rep, ws_sd=1.0, sigma=0.4, n_ws=40)
        w = summarize(d, rate=0.5, now=3e9).efficient
        nw = summarize([replace(o, ws="") for o in d], rate=0.5, now=3e9).efficient
        assert w is not None and nw is not None
        vr.append(w.variance_reduction)
        wr.append((w.hi - w.lo) / (nw.hi - nw.lo))
    return {"reps": reps, "variance_reduction": st.mean(vr), "width_ratio": st.mean(wr)}


def cusum_rates(reps: int) -> dict[str, float]:
    rng = random.Random(1)
    out: dict[str, float] = {"reps": reps}
    for h in (8.0, 12.0):
        for n in (150, 1000):
            out[f"false_alarm_h{h:g}_{n}"] = (
                sum(
                    bool(stats.cusum_changepoints([rng.gauss(0, 1) for _ in range(n)], h=h))
                    for _ in range(reps)
                )
                / reps
            )
    return out


def need() -> list[dict[str, float]]:
    return [
        {
            "effect": e,
            "cv": v,
            "holdout": stats.robust_needed(v, 0.05, target=e),
            "sessions": stats.robust_needed(v, 0.05, target=e) / 0.05,
        }
        for e in (0.1, 0.2, 0.3)
        for v in (1.0, 1.5)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--reps", type=int, default=200, help="replications per scenario (default 200)")
    add_out_args(ap, TRACKED)
    args = ap.parse_args(argv)
    r = args.reps
    result = {
        "note": "distil/abtest known-answer Monte Carlo; seeds fixed, reproducible offline",
        "cost_cap_usd": stats.COST_CAP,
        "alpha": 0.05,
        "type_one": type_one_table(r),
        "power": power(max(50, r // 3)),
        "denominator": denominator(max(40, r // 5)),
        "model_change": model_change(max(100, r // 2)),
        "detector": detector(max(100, r // 2)),
        "cuped": cuped(max(20, r // 10)),
        "cusum": cusum_rates(max(100, r)),
        "robust_needed": need(),
    }
    out = resolve_out(args, TRACKED)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"-> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
