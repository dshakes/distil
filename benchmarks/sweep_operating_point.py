#!/usr/bin/env python3
"""sweep_operating_point.py — pick distil's compression operating point HONESTLY.

Phase-1's E5 used the ``quick`` ladder (byte-exact, lossless, head-truncation),
whose aggressive rungs flip too many decisions to certify on the edit-localization
task. distil also ships *salience-protected* aggressive levels
(:func:`distil.compress.salience.protect`) that keep the decision-bearing lines while
crushing the rest — but ``protect`` has hyperparameters (``min_entropy``, ``min_len``)
and a truncation budget that nobody has tuned for this corpus.

This script sweeps those operating points the textbook way, with **no test-set
tuning**:

  1. Split the trajectories into disjoint CALIBRATION and TEST halves (seeded).
  2. Grade every candidate operating point on BOTH halves (grading only *measures*
     decision-change; it tunes nothing).
  3. SELECT on CALIBRATION only: among the points whose calibration decision-change
     certifies (Hoeffding–Bentkus p ≤ δ at risk α), keep the highest-SAVINGS one.
  4. EVALUATE that single winner ONCE on TEST: report its realized decision-change,
     savings, and whether the certificate holds out-of-sample.

The honest question: is there a salience-protected operating point that certifies
positive savings on TEST after being chosen on CAL? If yes, distil's ladder simply
needed the protected rungs; if no, the localization task genuinely resists certified
compression and we say so.

Decisions are cached/namespaced exactly like ``prove.py`` (same grader, same
``--expand`` recovery loop), so re-runs are free and comparable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil.compress.adaptive import byte_exact  # noqa: E402
from distil.compress.salience import protect  # noqa: E402
from distil.compress.strategies import distil  # noqa: E402
from distil.conformal import _truncate_level, hb_pvalue  # noqa: E402
from distil.replay import realtrace  # noqa: E402

import benchmarks.prove as prove  # noqa: E402


# --------------------------------------------------------------------------- #
# Candidate operating points (the sweep grid)
# --------------------------------------------------------------------------- #


def candidate_ladder(limits, entropies, lengths):
    """Build the operating-point ladder: two anchors (byte-exact, lossless) plus the
    full grid of salience-protected truncations. Ordered least→most aggressive so the
    list is also a valid LTT ladder, though selection here is per-point."""
    ladder = [("byte-exact", byte_exact), ("lossless", distil)]
    # protected points, ordered by decreasing truncation budget (less→more aggressive)
    for limit in sorted(limits, reverse=True):
        for ent in entropies:
            for ln in lengths:
                name = f"protect+trunc@{limit},e{ent},l{ln}"
                strat = protect(_truncate_level(limit), min_entropy=ent, min_len=ln)
                ladder.append((name, strat))
    # plain (unprotected) truncations as a reference baseline within the sweep
    for limit in sorted(limits, reverse=True):
        ladder.append((f"trunc@{limit}", _truncate_level(limit)))
    return ladder


# --------------------------------------------------------------------------- #
# Per-point decision-change / savings over a set of trajectories
# --------------------------------------------------------------------------- #


def point_stats(matrix, names, tids):
    """For each operating-point name, the decision-change rate, n, and token savings
    over the trajectories ``tids``. Reads the (already graded) loss matrix."""
    stats = {}
    for name in names:
        losses, base_t, comp_t = [], 0, 0
        for tid in tids:
            for tr in matrix[tid]["turns"]:
                cell = tr["levels"].get(name)
                if cell is None:
                    continue
                losses.append(cell["loss"])
                base_t += tr["base_tok"]
                comp_t += cell["comp_tok"]
        n = len(losses)
        rhat = (sum(losses) / n) if n else 1.0
        stats[name] = {
            "n": n,
            "decision_change": rhat,
            "savings": (1.0 - comp_t / base_t) if base_t else 0.0,
        }
    return stats


def select_on_calibration(cal_stats, *, alpha, delta):
    """Among points whose CAL decision-change certifies (HB p ≤ δ at α), return the
    highest-SAVINGS one's name; returns ``None`` if none certify (byte-exact, with
    rhat≈0, certifies in any real run, so ``None`` is pathological in practice). This
    selection is exploratory over the candidate grid — only the subsequent single
    evaluation on the disjoint TEST half carries the finite-sample δ guarantee."""
    certifying = [
        (name, s)
        for name, s in cal_stats.items()
        if hb_pvalue(s["decision_change"], s["n"], alpha) <= delta
    ]
    if not certifying:
        return None
    return max(certifying, key=lambda kv: kv[1]["savings"])[0]


# --------------------------------------------------------------------------- #
# LaTeX fragment for the paper (representative rows, default-hyperparameter protect)
# --------------------------------------------------------------------------- #


def _latex_rows(report: dict) -> list[str]:
    """Representative operating points, one per family, default protect hyperparameters
    (e3.2,l10) so the table is not a cherry-pick of the best protect variant."""
    cal, test = report["calibration"], report["test"]
    selected = (report.get("selected") or {}).get("operating_point")
    limits = sorted(report["grid"]["limits"], reverse=True)
    order = ["byte-exact", "lossless"]
    for lim in limits:
        order.append(f"trunc@{lim}")
        order.append(f"protect+trunc@{lim},e3.2,l10")
    rows = []
    for name in order:
        if name not in cal:
            continue
        c, t = cal[name], test[name]
        cc = (
            "\\checkmark"
            if hb_pvalue(c["decision_change"], c["n"], report["alpha"]) <= report["delta"]
            else "$\\times$"
        )
        tc = (
            "\\checkmark"
            if hb_pvalue(t["decision_change"], t["n"], report["alpha"]) <= report["delta"]
            else "$\\times$"
        )
        label = name.replace("protect+trunc", "protect+t").replace("trunc", "t").replace("_", "\\_")
        if name == selected:
            label = "\\textbf{" + label + "}"
        rows.append(
            f"{label} & {c['savings'] * 100:.1f}\\% & {c['decision_change'] * 100:.1f}\\% & {cc} "
            f"& {t['savings'] * 100:.1f}\\% & {t['decision_change'] * 100:.1f}\\% & {tc} \\\\"
        )
    return rows


def sweep_latex(report: dict) -> str:
    """A booktabs table of the calibration-selection sweep for the paper."""
    body = "\n".join(_latex_rows(report))
    header = "operating point & cal sav & cal dc & cal? & test sav & test dc & test? \\\\"
    return (
        "% auto-generated operating-point sweep (Phase 2; cal/test split)\n"
        "\\begin{tabular}{@{}lrrcrrc@{}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def sweep_macros(report: dict) -> str:
    """\\renewcommand macros for the prose: the selected point's test savings/dc."""
    sel = report.get("selected") or {}
    out = ["% auto-generated Phase-2 sweep macros — do not edit"]
    if sel:

        def pct(x):
            return f"{x * 100:.1f}\\%"

        name = sel["operating_point"].replace("trunc", "t").replace("_", "")
        out.append(f"\\renewcommand{{\\SweepPoint}}{{\\texttt{{{name}}}}}")
        out.append(f"\\renewcommand{{\\SweepCalSav}}{{{pct(sel['cal_savings'])}}}")
        out.append(f"\\renewcommand{{\\SweepTestSav}}{{{pct(sel['test_savings'])}}}")
        out.append(f"\\renewcommand{{\\SweepTestDC}}{{{pct(sel['test_decision_change'])}}}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--path", required=True, help="corpus json (e.g. the shuffled E5 corpus)")
    ap.add_argument("--runner", default="anthropic", choices=["smoke", "anthropic", "openai"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=100, help="cap #trajectories (matches E5)")
    ap.add_argument("--seed", type=int, default=0, help="subsample seed (matches E5)")
    ap.add_argument("--split-seed", type=int, default=1729, help="cal/test split seed")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--limits", type=int, nargs="+", default=[120, 250, 500])
    ap.add_argument("--entropies", type=float, nargs="+", default=[2.6, 3.2, 3.8])
    ap.add_argument("--lengths", type=int, nargs="+", default=[6, 10])
    ap.add_argument("--report", help="write the full sweep + selection + test eval here")
    ap.add_argument("--dry-run", action="store_true", help="plan only: count uncached decisions")
    ap.add_argument(
        "--latex-dir",
        help="also write LaTeX fragments (sweep table + macros) here, e.g. docs/paper/generated",
    )
    args = ap.parse_args()

    entries = realtrace.load_swe_bench(args.path, model=args.model)
    # same stratified subsample as prove.main (deterministic given seed)
    if args.limit and args.limit < len(entries):
        ok = [e for e in entries if realtrace.success_label(e) is True]
        no = [e for e in entries if realtrace.success_label(e) is False]
        other = [e for e in entries if realtrace.success_label(e) is None]
        rng0 = random.Random(args.seed)
        for grp in (ok, no, other):
            rng0.shuffle(grp)
        picked, i, pools = [], 0, [ok, no, other]
        while len(picked) < args.limit and any(pools):
            pool = pools[i % len(pools)]
            if pool:
                picked.append(pool.pop())
            i += 1
        entries = picked

    gold = realtrace.gold_actions(entries)
    ladder = candidate_ladder(args.limits, args.entropies, args.lengths)
    names = [n for n, _ in ladder]
    print(f"loaded {len(entries)} trajectories · {len(ladder)} operating points")

    # --- runner (same machinery / namespace as prove.py) -------------------- #
    if args.runner == "anthropic":
        from distil.replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner(model=args.model, samples=args.samples)
        ns = f"anthropic_{args.model}_s{args.samples}"
    elif args.runner == "openai":
        from distil.replay.openai_runner import OpenAIRunner

        runner = OpenAIRunner(args.model, samples=args.samples)
        ns = f"openai_{args.model.replace('/', '_')}_s{args.samples}"
    else:
        from distil.replay.smoke_runner import SmokeRunner

        runner = SmokeRunner()
        ns = "smoke"
    if args.expand:
        from distil.replay.expand_runner import ExpandAwareRunner

        runner = ExpandAwareRunner(runner, samples=args.samples)
        ns += "+expand"

    cache = prove.DecisionCache(runner, ns)

    if args.dry_run:
        from distil.replay.expand_runner import build_restore

        seen = set()
        for e in entries:
            for turn in e.trajectory.turns:
                restore = build_restore(turn.blocks) if args.expand else None
                comps = [turn.blocks] + [s(turn.blocks, turn.index) for _, s in ladder]
                for c in comps:
                    seen.add(cache._compose_key(c, restore))
        miss = sum(1 for k in seen if k not in cache.store)
        print(f"DRY-RUN: {len(seen)} unique contexts, {miss} uncached (×{args.samples} samples)")
        return 0

    matrix = prove.build_matrix(
        entries,
        cache,
        ladder,
        gold,
        expand=args.expand,
        baselines=[],
        workers=args.workers,
    )
    cache.flush()
    print(f"decisions: {cache.hits} cached / {cache.misses} computed")

    # --- cal/test split (disjoint, trajectory-level) ------------------------ #
    tids = list(matrix.keys())
    rng = random.Random(args.split_seed)
    rng.shuffle(tids)
    half = len(tids) // 2
    cal, test = tids[:half], tids[half:]

    cal_stats = point_stats(matrix, names, cal)
    test_stats = point_stats(matrix, names, test)
    winner = select_on_calibration(cal_stats, alpha=args.alpha, delta=args.delta)

    # full-data stats too (for the report table)
    full_stats = point_stats(matrix, names, tids)

    test_eval = None
    if winner is not None:
        s = test_stats[winner]
        test_eval = {
            "operating_point": winner,
            "cal_savings": cal_stats[winner]["savings"],
            "cal_decision_change": cal_stats[winner]["decision_change"],
            "test_savings": s["savings"],
            "test_decision_change": s["decision_change"],
            "test_n": s["n"],
            "test_certifies": hb_pvalue(s["decision_change"], s["n"], args.alpha) <= args.delta,
        }

    print(f"\nCAL={len(cal)} traj   TEST={len(test)} traj   α={args.alpha} δ={args.delta}")
    print(f"selected on CAL: {winner}")
    if test_eval:
        print(
            f"  CAL : savings {test_eval['cal_savings'] * 100:.1f}%  "
            f"dec-change {test_eval['cal_decision_change'] * 100:.1f}%"
        )
        print(
            f"  TEST: savings {test_eval['test_savings'] * 100:.1f}%  "
            f"dec-change {test_eval['test_decision_change'] * 100:.1f}%  "
            f"certifies={'YES' if test_eval['test_certifies'] else 'NO'}"
        )
    else:
        print("  no operating point certified on CAL (only byte-exact is trivially safe)")

    report = {
        "args": vars(args),
        "n_trajectories": len(entries),
        "n_cal": len(cal),
        "n_test": len(test),
        "alpha": args.alpha,
        "delta": args.delta,
        "grid": {
            "limits": args.limits,
            "entropies": args.entropies,
            "lengths": args.lengths,
        },
        "calibration": cal_stats,
        "test": test_stats,
        "full": full_stats,
        "selected": test_eval,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\nreport → {args.report}")
    if args.latex_dir:
        out = Path(args.latex_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep.tex").write_text(sweep_latex(report))
        (out / "sweepmacros.tex").write_text(sweep_macros(report))
        print(f"latex → {out}/sweep.tex, sweepmacros.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
