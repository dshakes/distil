"""The pre-registered analysis (protocol §8), the power calculation (§7) and the cost estimate.

Primary estimand, per arm A vs control C:

    R_A = ($ billed across ALL of A's attempts / A's solved attempts)
          / ($ billed across ALL of C's attempts / C's solved attempts)

Inference is a paired cluster bootstrap: resample *tasks* with replacement, carrying every
seed and every arm of a task together, so the pairing and the within-task correlation of
seeds are both respected. Three comparisons against one control -> Bonferroni alpha/3.

Co-primary: success-rate difference A - C, non-inferior iff the one-sided (alpha/3) lower
bootstrap bound exceeds -margin. TOST equivalence and the attempt-level McNemar test
(``distil.certify.stats``) are reported as secondaries.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import NormalDist
from typing import Any

from distil import pricing
from distil.certify.stats import mcnemar_noninferiority
from distil.conformal import BUDGET_DELTA, CERT_MARGIN

from benchmarks.cost_truth.runner import COUNTED

ALPHA = 0.05
#: Co-primary non-inferiority margin. CERT_MARGIN (2 pp) needs ~2,300 paired attempts per
#: comparison at realistic discordance (see ``ni_margin``); the feasible, pre-registered
#: margin is BUDGET_DELTA (5 pp). CERT_MARGIN is still tested and reported as a secondary.
NI_MARGIN = BUDGET_DELTA
SECONDARY_MARGIN = CERT_MARGIN
BOOTSTRAP_B = 10_000
_Z = NormalDist().inv_cdf


def quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list (numpy's default method)."""
    if not sorted_vals:
        return float("nan")
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    a, b = sorted_vals[lo], sorted_vals[hi]
    if lo == hi or a == b:
        return a
    return a + (b - a) * (pos - lo)


def complete_blocks(
    runs: list[dict[str, Any]], arms: list[str]
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    """task -> seed -> arm -> run, keeping only blocks where every arm has a COUNTED status."""
    by: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        by[r["task"]][r["seed"]][r["arm"]] = r
    out: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for task, seeds in by.items():
        keep = {
            s: b
            for s, b in seeds.items()
            if all(a in b and b[a]["status"] in COUNTED for a in arms)
        }
        if keep:
            out[task] = keep
    return out


def _per_task(
    blocks: dict[str, dict[int, dict[str, dict[str, Any]]]], arms: list[str]
) -> dict[str, dict[str, tuple[float, int, int]]]:
    """task -> arm -> (cost, solved, attempts)."""
    return {
        t: {
            a: (
                sum(b[a]["cost_usd"] for b in seeds.values()),
                sum(1 for b in seeds.values() if b[a]["solved"]),
                len(seeds),
            )
            for a in arms
        }
        for t, seeds in blocks.items()
    }


def _ratio(
    pt: dict[str, dict[str, tuple[float, int, int]]], tasks: list[str], arm: str, control: str
) -> float:
    ca = sum(pt[t][arm][0] for t in tasks)
    sa = sum(pt[t][arm][1] for t in tasks)
    cc = sum(pt[t][control][0] for t in tasks)
    sc = sum(pt[t][control][1] for t in tasks)
    if sa == 0 or sc == 0 or cc == 0:
        return math.inf if sa == 0 else 0.0
    return (ca / sa) / (cc / sc)


def _succ_diff(
    pt: dict[str, dict[str, tuple[float, int, int]]], tasks: list[str], arm: str, control: str
) -> float:
    n = sum(pt[t][arm][2] for t in tasks)
    return (sum(pt[t][arm][1] for t in tasks) - sum(pt[t][control][1] for t in tasks)) / n


def _task_log_ratio(
    pt: dict[str, dict[str, tuple[float, int, int]]], tasks: list[str], arm: str, control: str
) -> float:
    vals = [
        math.log(pt[t][arm][0] / pt[t][control][0])
        for t in tasks
        if pt[t][arm][0] > 0 and pt[t][control][0] > 0
    ]
    return sum(vals) / len(vals) if vals else float("nan")


def analyze(
    runs: list[dict[str, Any]],
    arms: list[str],
    control: str = "control",
    alpha: float = ALPHA,
    margin: float = NI_MARGIN,
    b: int = BOOTSTRAP_B,
    seed: int = 0,
) -> dict[str, Any]:
    blocks = complete_blocks(runs, arms)
    pt = _per_task(blocks, arms)
    tasks = sorted(pt)
    k = max(len(arms) - 1, 1)
    a_k = alpha / k
    rng = random.Random(seed)
    samples = [[rng.choice(tasks) for _ in tasks] for _ in range(b)] if tasks else []
    per_arm: dict[str, Any] = {}
    for arm in arms:
        cost = sum(pt[t][arm][0] for t in tasks)
        solved = sum(pt[t][arm][1] for t in tasks)
        n = sum(pt[t][arm][2] for t in tasks)
        per_arm[arm] = {
            "attempts": n,
            "solved": solved,
            "success_rate": solved / n if n else float("nan"),
            "cost_usd": round(cost, 6),
            "usd_per_solved": cost / solved if solved else math.inf,
        }
    comparisons: dict[str, Any] = {}
    for arm in arms:
        if arm == control:
            continue
        rs = sorted(_ratio(pt, s, arm, control) for s in samples)
        ds = sorted(_succ_diff(pt, s, arm, control) for s in samples)
        ls = sorted(_task_log_ratio(pt, s, arm, control) for s in samples)
        r_lo, r_hi = quantile(rs, a_k / 2), quantile(rs, 1 - a_k / 2)
        d_lo, d_hi = quantile(ds, a_k), quantile(ds, 1 - a_k)
        cost_verdict = "cheaper" if r_hi < 1 else "dearer" if r_lo > 1 else "inconclusive"
        noninf = d_lo > -margin
        pairs = [bl for t in tasks for bl in blocks[t].values()]
        bb = sum(1 for bl in pairs if bl[control]["solved"] and not bl[arm]["solved"])
        cc = sum(1 for bl in pairs if bl[arm]["solved"] and not bl[control]["solved"])
        comparisons[arm] = {
            "usd_per_solved_ratio": _ratio(pt, tasks, arm, control),
            "ratio_ci": [r_lo, r_hi],
            "ratio_ci_level": 1 - a_k,
            "cost_verdict": cost_verdict,
            "success_diff": _succ_diff(pt, tasks, arm, control) if tasks else float("nan"),
            "success_diff_ci": [d_lo, d_hi],
            "ni_margin": margin,
            "noninferior": noninf,
            "tost_equivalent": noninf and d_hi < margin,
            "verdict": "saves" if cost_verdict == "cheaper" and noninf else "not shown",
            "secondary": {
                "task_mean_log_ratio": _task_log_ratio(pt, tasks, arm, control),
                "total_usd_ratio": (
                    sum(pt[t][arm][0] for t in tasks) / sum(pt[t][control][0] for t in tasks)
                    if tasks
                    else float("nan")
                ),
                "leave_one_task_out_ratio": (
                    [
                        min(_ratio(pt, [u for u in tasks if u != t], arm, control) for t in tasks),
                        max(_ratio(pt, [u for u in tasks if u != t], arm, control) for t in tasks),
                    ]
                    if len(tasks) > 1
                    else None
                ),
                "task_mean_log_ratio_ci": [quantile(ls, a_k / 2), quantile(ls, 1 - a_k / 2)],
                "mcnemar_cert_margin": vars(
                    mcnemar_noninferiority(
                        bb, cc, len(pairs), margin=SECONDARY_MARGIN, z=_Z(1 - a_k)
                    )
                ),
                "warm_start_ratio": _ratio(
                    _per_task(_reprice_warm(blocks), arms), tasks, arm, control
                ),
                "claims": _claims(pairs, arm, control),
                "mean_turns": {a: _mean(bl[a]["turns"] for bl in pairs) for a in (arm, control)},
                "mean_wall_s": {
                    a: _mean(bl[a].get("wall_s", 0.0) for bl in pairs) for a in (arm, control)
                },
            },
        }
    return {
        "tasks": len(tasks),
        "blocks": sum(len(v) for v in blocks.values()),
        "excluded_runs": sum(1 for r in runs if r["status"] not in COUNTED),
        "alpha_per_comparison": a_k,
        "bootstrap_b": b,
        "per_arm": per_arm,
        "comparisons": comparisons,
    }


def _mean(xs: Any) -> float:
    v = list(xs)
    return sum(v) / len(v) if v else float("nan")


def _reprice_warm(
    blocks: dict[str, dict[int, dict[str, dict[str, Any]]]],
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    """Sensitivity: bill each run's first-request cache reads (a warm start possibly inherited
    from an earlier run) as if they had been cold writes. Removes cross-run cache luck."""
    out: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for t, seeds in blocks.items():
        out[t] = {}
        for s, bl in seeds.items():
            out[t][s] = {}
            for a, r in bl.items():
                p = pricing.resolve(r["model"])
                extra = r["first_request_cache_read"] * (p.cache_write - p.cache_read) if p else 0.0
                out[t][s][a] = {**r, "cost_usd": r["cost_usd"] + extra}
    return out


def _claims(pairs: list[dict[str, dict[str, Any]]], arm: str, control: str) -> dict[str, Any]:
    """The tool's own savings claim next to the billed truth on the same (task, seed) pairs."""
    claimed = [bl[arm].get("claim", {}).get("tokens_saved") for bl in pairs]
    known = [c for c in claimed if isinstance(c, (int, float))]

    def input_side(r: dict[str, Any]) -> int:
        u = r["usage"]
        return u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"]

    billed_tok = sum(input_side(bl[control]) - input_side(bl[arm]) for bl in pairs)
    billed_usd = sum(bl[control]["cost_usd"] - bl[arm]["cost_usd"] for bl in pairs)
    return {
        "claimed_tokens_saved": sum(known) if known else None,
        "runs_with_claim": len(known),
        "billed_input_tokens_saved": billed_tok,
        "billed_usd_saved": round(billed_usd, 6),
        "claim_over_billed_tokens": (sum(known) / billed_tok) if known and billed_tok > 0 else None,
    }


# --------------------------------------------------------------------------- power (§7)


def n_tasks_for_mde(
    mde_ratio: float,
    sigma_w: float,
    sigma_b: float,
    k: int,
    alpha: float = ALPHA,
    power: float = 0.8,
    comparisons: int = 3,
) -> int:
    """Tasks needed to detect a true cost ratio of ``mde_ratio`` (e.g. 0.9 = 10% cheaper).

    Normal approximation on the per-task paired difference of mean log cost over ``k``
    seeds: Var = 2*sigma_w^2/k + sigma_b^2 (within-task attempt noise, task x arm
    heterogeneity). Two-sided, Bonferroni over ``comparisons``.
    """
    d = abs(math.log(mde_ratio))
    var = 2 * sigma_w**2 / k + sigma_b**2
    z = _Z(1 - alpha / (2 * comparisons)) + _Z(power)
    return math.ceil(z * z * var / (d * d))


def mde_ratio(
    n_tasks: int,
    sigma_w: float,
    sigma_b: float,
    k: int,
    alpha: float = ALPHA,
    power: float = 0.8,
    comparisons: int = 3,
) -> float:
    var = 2 * sigma_w**2 / k + sigma_b**2
    z = _Z(1 - alpha / (2 * comparisons)) + _Z(power)
    return math.exp(-z * math.sqrt(var / n_tasks))


def ni_margin(
    n_pairs: int,
    discordance: float,
    deff: float = 1.0,
    alpha: float = ALPHA,
    power: float = 0.8,
    comparisons: int = 3,
) -> float:
    """Smallest NI margin provable with ``power`` when the true difference is 0.

    ``discordance`` = P(exactly one arm solves a pair); ``deff`` = design effect of seeds
    clustered in tasks (1 + (k-1)*ICC).
    """
    z = _Z(1 - alpha / comparisons) + _Z(power)
    return z * math.sqrt(discordance * deff / n_pairs)


# --------------------------------------------------------------------------- cost estimate


def attempt_tokens(turns: int, prefix: int, new_per_turn: int, out_per_turn: int) -> dict[str, int]:
    """Token profile of one agent attempt with Claude Code's rolling cache breakpoint.

    Turn t re-reads everything before it (cache read) and writes its own new tokens; the
    shared prefix (system + tools + task) is written once.
    """
    return {
        "cache_read_input_tokens": turns * prefix + new_per_turn * turns * (turns - 1) // 2,
        "cache_creation_input_tokens": prefix + turns * new_per_turn,
        "input_tokens": turns * 3,
        "output_tokens": turns * out_per_turn,
    }


def price_tokens(model: str, t: dict[str, int], cached: bool = True) -> float:
    p = pricing.resolve(model)
    if p is None:
        raise KeyError(model)
    inp = t["input_tokens"] + t["cache_read_input_tokens"] + t["cache_creation_input_tokens"]
    if not cached:
        return inp * p.input + t["output_tokens"] * p.output
    return (
        t["input_tokens"] * p.input
        + t["cache_read_input_tokens"] * p.cache_read
        + t["cache_creation_input_tokens"] * p.cache_write
        + t["output_tokens"] * p.output
    )
