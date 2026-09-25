"""Known-answer Monte Carlo for ``distil ab`` (distil/abtest) — offline, no API calls.

Every number docs/ab.html prints about the estimator comes from this script's
tracked output, ``benchmarks/results/abtest-montecarlo.json``:

* type-I error of the anytime-valid interval under 19 repeated looks, next to the
  naive fixed-n interval checked at the same looks;
* power and coverage at one look for a true −20% effect;
* bias and coverage when a model change moves turns per task +40% in BOTH arms
  (explicit model-id change, and a silent one behind an unchanged id), next to what a
  before/after comparison of the distil arm would have booked;
* era-detection selection bias, arm-blind vs holdout-only (why the detector is blind);
* CUPED variance reduction on workspace-correlated costs;
* the holdout sessions a 5% holdout needs to resolve a 10/20/30% effect.

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
from distil.abtest.report import _units, summarize

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
) -> list[SessionOutcome]:
    """Sessions whose true relative effect on mean cost per task is exactly ``effect``.

    distil multiplies each session's cost by ``1+effect``; turns (~20, log-sd 0.3) are
    untouched by distil and scale with ``turns_mult`` — the model's behaviour — in BOTH
    arms. Per-session cost noise is lognormal (log-sd ``sigma``, mean-preserving), plus a
    per-workspace log offset with sd ``ws_sd`` (what CUPED can remove).
    """
    rng = random.Random(seed)
    ws_eff = [rng.gauss(0, ws_sd) for _ in range(n_ws)]
    out = []
    for i in range(n):
        hold = rng.random() < p
        w = rng.randrange(n_ws)
        turns = max(1, round(20 * turns_mult * math.exp(rng.gauss(0, 0.3))))
        cost = 0.05 * turns * math.exp(ws_eff[w] + rng.gauss(0, sigma) - sigma**2 / 2)
        if not hold:
            cost *= 1 + effect
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
                tasks=1,
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


def type_one(reps: int) -> dict[str, float]:
    seq = naive = 0
    for rep in range(reps):
        data = sim(400, effect=0.0, seed=10_000 + rep, sigma=0.8)
        hs = hn = False
        for n in range(40, 401, 20):
            c = _contrast(data[:n])
            if c is None:
                continue
            hs = hs or c.significant(0.05)
            hn = hn or abs(c.delta) > 1.959964 * math.sqrt(c.var)
        seq += hs
        naive += hn
    return {
        "reps": reps,
        "looks": 19,
        "alpha": 0.05,
        "sequential": seq / reps,
        "naive": naive / reps,
    }


def power(reps: int) -> list[dict[str, float]]:
    out = []
    for n in (400, 800, 1600):
        hits = cov = 0
        for rep in range(reps):
            c = _contrast(sim(n, effect=-0.2, seed=20_000 + rep))
            assert c is not None
            lo, hi = c.interval(0.05)
            hits += c.significant(0.05)
            cov += lo < -0.2 < hi
        out.append({"n": n, "effect": -0.2, "power": hits / reps, "coverage": cov / reps})
    return out


def model_change(reps: int) -> dict[str, dict[str, float]]:
    res = {}
    for kind in ("explicit", "silent"):
        errs, cover, eras, naive = [], 0, 0, []
        for rep in range(reps):
            a = sim(300, seed=130_000 + rep)
            b = sim(
                300,
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
            "sessions_per_model": 300,
        }
    return res


def era_selection(reps: int, h: float = 8.0) -> dict[str, dict[str, float | None]]:
    """Stationary data (no change, effect −20%): bias of eras a detector CLOSES,
    reading the holdout arm only vs both arms.

    Run at h=8, not the shipped h=12: a selection bias can only be measured on eras
    that were cut, and at h=12 a stationary 1,500-session series is almost never cut.
    This is the measurement that decided the detector reads both arms."""
    from distil.abtest import stats as S

    res = {}
    for blind in (False, True):
        closed, current, n_eras = [], [], []
        for rep in range(reps):
            d = sim(1500, effect=-0.2, seed=70_000 + rep, sigma=0.4)
            src = [o for o in d if blind or o.arm == HOLDOUT]
            ys = [math.log(o.cost or 1e-9) for o in src]
            spans = [(src[c].start, src[a].start) for c, a in S.cusum_changepoints(ys, h=h)]
            eras: list[list[SessionOutcome]] = []
            cur: list[SessionOutcome] = []
            b = 0
            for o in d:
                while b < len(spans) and o.start > spans[b][1]:
                    if cur:
                        eras.append(cur)
                    cur, b = [], b + 1
                if b < len(spans) and spans[b][0] <= o.start <= spans[b][1]:
                    continue
                cur.append(o)
            if cur:
                eras.append(cur)
            n_eras.append(len(eras))
            cs = [S.ratio_contrast(_units(e, "cost", {}), cuped=False) for e in eras]
            closed += [c.rel + 0.2 for c in cs[:-1] if c and c.n0 >= 5]
            if cs[-1] and cs[-1].n0 >= 5:
                current.append(cs[-1].rel + 0.2)
        res["arm_blind" if blind else "holdout_only"] = {
            "mean_eras": st.mean(n_eras),
            "closed_era_bias": st.mean(closed) if closed else None,
            "h": h,
            "closed_eras": len(closed),
            "current_era_bias": st.mean(current),
        }
    return res


def cuped(reps: int) -> dict[str, float]:
    vr, wr = [], []
    for rep in range(reps):
        d = sim(1000, seed=50_000 + rep, ws_sd=1.0, sigma=0.4, n_ws=40)
        w = summarize(d, rate=0.5, now=3e9).cost
        nw = summarize([replace(o, ws="") for o in d], rate=0.5, now=3e9).cost
        assert w is not None and nw is not None
        vr.append(w.variance_reduction)
        wr.append((w.hi - w.lo) / (nw.hi - nw.lo))
    return {"reps": reps, "variance_reduction": st.mean(vr), "width_ratio": st.mean(wr)}


def cusum_rates(reps: int) -> dict[str, float]:
    rng = random.Random(1)
    fa = {
        n: sum(
            bool(stats.cusum_changepoints([rng.gauss(0, 1) for _ in range(n)])) for _ in range(reps)
        )
        / reps
        for n in (150, 1000)
    }
    return {"false_alarm_150": fa[150], "false_alarm_1000": fa[1000], "reps": reps}


def need() -> list[dict[str, float]]:
    return [
        {
            "effect": e,
            "cv2": v,
            "holdout": stats.holdout_needed(v, 0.05, target=e),
            "sessions": stats.holdout_needed(v, 0.05, target=e) / 0.05,
        }
        for e in (0.1, 0.2, 0.3)
        for v in (0.5, 1.0, 2.0)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--reps", type=int, default=300, help="replications per scenario (default 300)")
    add_out_args(ap, TRACKED)
    args = ap.parse_args(argv)
    r = args.reps
    result = {
        "note": "distil/abtest known-answer Monte Carlo; seeds fixed, reproducible offline",
        "type_one_error": type_one(max(r, 1000) if r >= 300 else r),
        "power": power(r),
        "model_change": model_change(r),
        "era_selection": era_selection(max(50, r // 2)),
        "cuped": cuped(max(20, r // 6)),
        "cusum": cusum_rates(max(100, r)),
        "holdout_needed": need(),
    }
    out = resolve_out(args, TRACKED)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"-> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
