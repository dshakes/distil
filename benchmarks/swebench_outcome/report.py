"""Paired analysis + markdown report. Every arm is compared with `plain` on the instances both
solved-or-failed (graded outcomes only); the headline summary lists all arms side by side."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .names import ARM_NAMES, BASELINE
from .stats import mcnemar_exact, paired_diff, sample_size_noninferiority, wilson

MARGIN = 0.05  # pre-registered non-inferiority margin (5 pts); see specs/swebench-outcome-eval.md


PILOT_N = 100  # below this many pairs the report gives no non-inferiority verdict


def outcomes(
    results: list[dict[str, Any]], grades: list[dict[str, str]]
) -> dict[str, dict[str, bool]]:
    """{arm: {instance: resolved}} over graded outcomes only. Empty-patch gave_up counts as
    unresolved (a real agent failure); api/env/internal/budget classes are excluded, not scored."""
    g = {(r["instance_id"], r["arm"]): r["status"] for r in grades}
    out: dict[str, dict[str, bool]] = {}
    for r in results:
        k, cls = (r["instance_id"], r["arm"]), r.get("failure_class")
        if cls == "gave_up" and not (r.get("patch") or "").strip():
            res = False
        elif g.get(k) in ("resolved", "unresolved"):
            res = g[k] == "resolved"
        else:
            continue
        out.setdefault(r["arm"], {})[r["instance_id"]] = res
    return out


def arm_order(present: set[str]) -> list[str]:
    """Canonical arms first, then any unknown names alphabetically."""
    return [a for a in ARM_NAMES if a in present] + sorted(present - set(ARM_NAMES))


def compare(
    base: dict[str, bool], arm: dict[str, bool], margin: float, ids: list[str] | None = None
) -> dict[str, Any]:
    """Paired comparison of *arm* against *base* (b = base-only wins, c = arm-only wins)."""
    ids = sorted(set(base) & set(arm)) if ids is None else ids
    b = sum(base[i] and not arm[i] for i in ids)
    c = sum(arm[i] and not base[i] for i in ids)
    n = len(ids)
    d, lo, hi = paired_diff(b, c, n)
    verdict = "NON-INFERIOR" if lo >= -margin else "INFERIOR" if hi < -margin else "INCONCLUSIVE"
    if n < PILOT_N:
        # The Wald interval collapses with few discordant pairs (0 -> [0, 0]); no verdict.
        verdict = "PILOT (no verdict)"
    pd = (b + c) / n if n else 0.0
    return {
        "n_pairs": n,
        "both": sum(base[i] and arm[i] for i in ids),
        "base_only": b,
        "arm_only": c,
        "neither": sum(not base[i] and not arm[i] for i in ids),
        "base_rate": sum(base[i] for i in ids) / n if n else 0.0,
        "arm_rate": sum(arm[i] for i in ids) / n if n else 0.0,
        "diff": d,
        "ci": (lo, hi),
        "mcnemar_p": mcnemar_exact(b, c),
        "verdict": verdict,
        "p_discordant": pd,
        "n_needed": sample_size_noninferiority(margin, pd) if pd else None,
    }


def _arm_stats(rs: list[dict[str, Any]], oc: dict[str, bool], ids: list[str]) -> dict[str, Any]:
    k = sum(oc[i] for i in ids)
    n = len(ids)
    meta: dict[str, Any] = next((r["arm_meta"] for r in rs if r.get("arm_meta")), {})
    return {
        "n": n,
        "resolved": k,
        "rate": k / n if n else 0.0,
        "wilson": wilson(k, n),
        "tasks": len(rs),
        "classes": dict(Counter(r.get("failure_class") or "ok" for r in rs)),
        "cost_usd": sum(r.get("cost_usd", 0) for r in rs),
        "steps": sum(r.get("steps", 0) for r in rs),
        "tokens_in": sum(
            sum((r.get("usage") or {}).get(k2, 0) for k2 in ("input", "cache_write", "cache_read"))
            for r in rs
        ),
        "tokens_out": sum((r.get("usage") or {}).get("output", 0) for r in rs),
        "expand_calls": sum(r.get("expand_calls", 0) for r in rs),
        "reused": sum(1 for r in rs if r.get("reused_from")),
        "meta": meta,
    }


def analyse(
    results: list[dict[str, Any]],
    grades: list[dict[str, str]],
    margin: float = MARGIN,
    arms: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """`arms` limits and orders the arms analysed (default: every arm in *results*).

    Per-arm rows are scored on the instances graded in every arm (one shared denominator);
    each `comparisons[arm]` uses the pairs graded in both that arm and plain. The top-level
    n_pairs/plain_only/distil_only/... keys are the headline comparison (distil if present,
    else the first challenger) and exist for backward compatibility."""
    oc = outcomes(results, grades)
    names = list(arms) if arms else arm_order({r["arm"] for r in results})
    challengers = [a for a in names if a != BASELINE]
    if challengers and BASELINE not in names:
        raise SystemExit(f"report needs the {BASELINE!r} arm to compare against")
    graded = [a for a in names if oc.get(a)]
    common = sorted(set.intersection(*(set(oc[a]) for a in graded))) if graded else []
    stats = {
        a: _arm_stats([r for r in results if r["arm"] == a], oc.get(a, {}), common) for a in names
    }
    comps = {a: compare(oc.get(BASELINE, {}), oc.get(a, {}), margin) for a in challengers}
    head_name = "distil" if "distil" in comps else next(iter(comps), None)
    head = comps[head_name] if head_name else compare({}, {}, margin)
    return {
        "n_pairs": head["n_pairs"],
        "both": head["both"],
        "plain_only": head["base_only"],
        "distil_only": head["arm_only"],
        "neither": head["neither"],
        "diff": head["diff"],
        "ci": head["ci"],
        "mcnemar_p": head["mcnemar_p"],
        "margin": margin,
        "verdict": head["verdict"],
        "p_discordant": head["p_discordant"],
        "n_needed": head["n_needed"],
        "arms": stats,
        "comparisons": comps,
        "n_common": len(common),
    }


def _comparison_lines(arm: str, c: dict[str, Any], margin: float) -> list[str]:
    lo, hi = c["ci"]
    L = [
        f"Discordant: plain-only {c['base_only']}, {arm}-only {c['arm_only']} "
        f"(both {c['both']}, neither {c['neither']}).",
        f"Paired difference ({arm} - plain): {c['diff'] * 100:+.1f} pts, 95% CI "
        f"[{lo * 100:+.1f}, {hi * 100:+.1f}] (Wald); exact McNemar p = {c['mcnemar_p']:.4f}.",
        f"Decision (non-inferiority margin {margin * 100:.0f} pts): **{c['verdict']}**.",
        "",
    ]
    if c["n_needed"]:
        L.append(
            f"At the observed discordance ({c['p_discordant']:.1%}) ~{c['n_needed']} pairs "
            f"would be needed for 80% power at this margin (true diff 0)."
        )
    if c["n_pairs"] < PILOT_N:
        L.append("n < 100: the Wald interval is unreliable; treat this as a pilot, not a verdict.")
    return L


_EXCLUDED = (
    "Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, "
    "never counted as unresolved."
)


def markdown(a: dict[str, Any], reuse: dict[str, dict[str, Any]] | None = None) -> str:
    arms = a["arms"]
    if set(arms) == {"plain", "distil"}:
        return _markdown_pair(a)
    return _markdown_multi(a, reuse or {})


def _markdown_pair(a: dict[str, Any]) -> str:
    L = ["# SWE-bench Lite outcome eval", ""]
    n = a["n_pairs"]
    L += [
        f"Paired instances with a graded outcome in both arms: **{n}**",
        "",
        "| arm | resolved | rate (95% Wilson) | cost $ | steps | tok in | tok out | expand calls | classes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, s in a["arms"].items():
        lo, hi = s["wilson"]
        L.append(
            f"| {arm} | {s['resolved']}/{s['n']} | {s['rate']:.1%} ({lo:.1%}-{hi:.1%}) | "
            f"{s['cost_usd']:.2f} | {s['steps']} | {s['tokens_in']} | {s['tokens_out']} | "
            f"{s['expand_calls']} | {s['classes']} |"
        )
    L += ["", *_comparison_lines("distil", a["comparisons"]["distil"], a["margin"])]
    L.append(_EXCLUDED)
    return "\n".join(L) + "\n"


def _markdown_multi(a: dict[str, Any], reuse: dict[str, dict[str, Any]]) -> str:
    arms, comps = a["arms"], a["comparisons"]
    L = ["# SWE-bench Lite outcome eval (head-to-head)", ""]
    L.append(
        f"Arms: {', '.join(arms)}. Summary rows are scored on the **{a['n_common']}** instances "
        "with a graded outcome in every arm; each comparison below uses the pairs graded in "
        "both that arm and plain."
    )
    for arm, info in sorted(reuse.items()):
        L.append(
            f"Arm `{arm}` was **reused** from `{info['from']}` ({info['rows']} rows copied, "
            "grades carried over): not re-run in this directory."
        )
    L += [
        "",
        "| arm | library | version | resolved | rate (95% Wilson) | vs plain (pts) | 95% CI | "
        "McNemar p | verdict | cost $ | steps | tok in | tok out | classes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, s in arms.items():
        lo, hi = s["wilson"]
        m = s["meta"]
        head = (
            f"| {arm} | {m.get('library') or '-'} | {m.get('version') or '-'} | "
            f"{s['resolved']}/{s['n']} | {s['rate']:.1%} ({lo:.1%}-{hi:.1%}) | "
        )
        c = comps.get(arm)
        cmp_ = (
            f"{c['diff'] * 100:+.1f} | [{c['ci'][0] * 100:+.1f}, {c['ci'][1] * 100:+.1f}] | "
            f"{c['mcnemar_p']:.4f} | {c['verdict']} | "
            if c
            else "- | - | - | baseline | "
        )
        L.append(
            head
            + cmp_
            + f"{s['cost_usd']:.2f} | {s['steps']} | {s['tokens_in']} | {s['tokens_out']} | "
            f"{s['classes']} |"
        )
    for arm, c in comps.items():
        L += ["", f"## {arm} vs plain ({c['n_pairs']} pairs)", ""]
        L += _comparison_lines(arm, c, a["margin"])
    if len(comps) > 1:
        L += [
            "",
            f"{len(comps)} comparisons against one baseline: the McNemar p-values and CIs are "
            f"per-comparison and uncorrected (Bonferroni alpha = {0.05 / len(comps):.4f}).",
        ]
    L += ["", _EXCLUDED]
    return "\n".join(L) + "\n"
