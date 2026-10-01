"""Paired analysis + markdown report."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .stats import mcnemar_exact, paired_diff, sample_size_noninferiority, wilson

MARGIN = 0.05  # pre-registered non-inferiority margin (5 pts); see specs/swebench-outcome-eval.md


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


def analyse(
    results: list[dict[str, Any]], grades: list[dict[str, str]], margin: float = MARGIN
) -> dict[str, Any]:
    oc = outcomes(results, grades)
    plain, distil = oc.get("plain", {}), oc.get("distil", {})
    ids = sorted(set(plain) & set(distil))
    b = sum(plain[i] and not distil[i] for i in ids)  # plain-only
    c = sum(distil[i] and not plain[i] for i in ids)  # distil-only
    n = len(ids)
    d, lo, hi = paired_diff(b, c, n)
    arms: dict[str, Any] = {}
    for arm in ("plain", "distil"):
        rs = [r for r in results if r["arm"] == arm]
        k = sum(oc.get(arm, {})[i] for i in ids)
        arms[arm] = {
            "n": n,
            "resolved": k,
            "rate": k / n if n else 0.0,
            "wilson": wilson(k, n),
            "tasks": len(rs),
            "classes": dict(Counter(r.get("failure_class") or "ok" for r in rs)),
            "cost_usd": sum(r.get("cost_usd", 0) for r in rs),
            "steps": sum(r.get("steps", 0) for r in rs),
            "tokens_in": sum(
                sum(
                    (r.get("usage") or {}).get(k2, 0)
                    for k2 in ("input", "cache_write", "cache_read")
                )
                for r in rs
            ),
            "tokens_out": sum((r.get("usage") or {}).get("output", 0) for r in rs),
            "expand_calls": sum(r.get("expand_calls", 0) for r in rs),
        }
    verdict = "NON-INFERIOR" if lo >= -margin else "INFERIOR" if hi < -margin else "INCONCLUSIVE"
    pd = (b + c) / n if n else 0.0
    return {
        "n_pairs": n,
        "both": sum(plain[i] and distil[i] for i in ids),
        "plain_only": b,
        "distil_only": c,
        "neither": sum(not plain[i] and not distil[i] for i in ids),
        "diff": d,
        "ci": (lo, hi),
        "mcnemar_p": mcnemar_exact(b, c),
        "margin": margin,
        "verdict": verdict,
        "p_discordant": pd,
        "n_needed": sample_size_noninferiority(margin, pd) if pd else None,
        "arms": arms,
    }


def markdown(a: dict[str, Any]) -> str:
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
    lo, hi = a["ci"]
    L += [
        "",
        f"Discordant: plain-only {a['plain_only']}, distil-only {a['distil_only']} "
        f"(both {a['both']}, neither {a['neither']}).",
        f"Paired difference (distil - plain): {a['diff'] * 100:+.1f} pts, 95% CI "
        f"[{lo * 100:+.1f}, {hi * 100:+.1f}] (Wald); exact McNemar p = {a['mcnemar_p']:.4f}.",
        f"Decision (non-inferiority margin {a['margin'] * 100:.0f} pts): **{a['verdict']}**.",
        "",
    ]
    if a["n_needed"]:
        L.append(
            f"At the observed discordance ({a['p_discordant']:.1%}) ~{a['n_needed']} pairs "
            f"would be needed for 80% power at this margin (true diff 0)."
        )
    if n < 100:
        L.append("n < 100: the Wald interval is unreliable; treat this as a pilot, not a verdict.")
    L.append(
        "Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, "
        "never counted as unresolved."
    )
    return "\n".join(L) + "\n"
