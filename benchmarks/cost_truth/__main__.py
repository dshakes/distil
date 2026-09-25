"""CLI: ``python -m benchmarks.cost_truth {plan,dry-run,analyze,estimate,power}``.

Nothing here spends money. There is deliberately no ``live`` subcommand yet: the live
driver lands only after the protocol is frozen and the arm specs are verified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.cost_truth import analysis as an
from benchmarks.cost_truth import arms as arms_mod
from benchmarks.cost_truth.runner import ARMS, build_schedule, dry_run, load_runs
from benchmarks.outpath import ROOT, SCRATCH

ESTIMATE_PATH = ROOT / "benchmarks" / "results" / "cost_truth" / "cost_estimate.json"

#: Pre-registered design (protocol §7). tasks x seeds x arms per model.
DESIGN: dict[str, dict[str, Any]] = {
    "pilot": {"model": "claude-sonnet-5", "tasks": 10, "seeds": 2},
    "primary": {"model": "claude-sonnet-5", "tasks": 89, "seeds": 5},
    "replication": {"model": "claude-haiku-4-5", "tasks": 89, "seeds": 3},
}

#: One Terminal-Bench 2.1 attempt under Claude Code (assumption, protocol §9). Chosen to be
#: ~1.8x Quesma's observed per-attempt spend so the budget errs high.
PROFILE = {"turns": 30, "prefix": 22_000, "new_per_turn": 1_800, "out_per_turn": 450}
SAFETY = 1.25


def _fmt(x: float) -> str:
    return "inf" if math.isinf(x) else f"{x:.3f}"


def report(res: dict[str, Any]) -> str:
    lines = [
        f"tasks={res['tasks']} blocks={res['blocks']} excluded_runs={res['excluded_runs']} "
        f"alpha/comparison={res['alpha_per_comparison']:.4f} B={res['bootstrap_b']}",
        "",
        f"{'arm':<10}{'attempts':>9}{'solved':>8}{'success':>9}{'$ total':>11}{'$/solved':>10}",
    ]
    for a, v in res["per_arm"].items():
        lines.append(
            f"{a:<10}{v['attempts']:>9}{v['solved']:>8}{v['success_rate']:>9.3f}"
            f"{v['cost_usd']:>11.4f}{_fmt(v['usd_per_solved']):>10}"
        )
    lines += [
        "",
        f"{'vs control':<10}{'$/solved ratio [CI]':>30}{'success diff [CI]':>28}  verdict",
    ]
    for a, c in res["comparisons"].items():
        rc = f"{_fmt(c['usd_per_solved_ratio'])} [{_fmt(c['ratio_ci'][0])}, {_fmt(c['ratio_ci'][1])}]"
        dc = f"{c['success_diff']:+.3f} [{c['success_diff_ci'][0]:+.3f}, {c['success_diff_ci'][1]:+.3f}]"
        lines.append(
            f"{a:<10}{rc:>30}{dc:>28}  {c['cost_verdict']}, NI={c['noninferior']} -> {c['verdict']}"
        )
        cl = c["secondary"]["claims"]
        lines.append(
            f"{'':<10}claim {cl['claimed_tokens_saved']} tok vs billed {cl['billed_input_tokens_saved']} tok "
            f"(${cl['billed_usd_saved']:.4f}); warm-start-repriced ratio {_fmt(c['secondary']['warm_start_ratio'])}"
        )
    return "\n".join(lines)


def estimate() -> dict[str, Any]:
    tok = an.attempt_tokens(**PROFILE)
    out: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d"),
        "basis": "list price from distil.pricing (cache read 0.1x, 5m write 1.25x); token profile is an "
        "ASSUMPTION, not a measurement - replace with pilot medians before the confirmatory run",
        "profile_per_attempt": {**PROFILE, "tokens": tok},
        "cross_check": "Quesma (2026-09-11): Claude Code + Fable 5.0 on Terminal-Bench 2.1 baseline "
        "$731 / 425 attempts = $1.72/attempt at $10/$50 per Mtok; the same token mix at Sonnet 5 "
        "prices is roughly $0.52/attempt, so this profile (~$0.92) is ~1.8x conservative",
        "arm_assumption": "every arm priced at the control profile; a cache-busting arm can cost up to "
        "~2x its control (distil measured this on itself, 2026-08); that would exhaust the cap, which "
        "stops the run and the study is reported as truncated, not extended",
        "safety_multiplier": SAFETY,
        "phases": {},
    }
    total = 0.0
    for phase, d in DESIGN.items():
        per_cached = an.price_tokens(d["model"], tok, cached=True)
        per_uncached = an.price_tokens(d["model"], tok, cached=False)
        n_arm = d["tasks"] * d["seeds"]
        cap = per_cached * n_arm * len(ARMS) * SAFETY
        total += cap
        out["phases"][phase] = {
            **d,
            "arms": list(ARMS),
            "runs_per_arm": n_arm,
            "runs_total": n_arm * len(ARMS),
            "usd_per_attempt_cached": round(per_cached, 4),
            "usd_per_attempt_if_no_cache": round(per_uncached, 4),
            "per_arm_usd_cached": round(per_cached * n_arm, 2),
            "per_arm_usd_if_no_cache": round(per_uncached * n_arm, 2),
            "phase_usd_cached": round(per_cached * n_arm * len(ARMS), 2),
            "phase_usd_if_no_cache": round(per_uncached * n_arm * len(ARMS), 2),
            "hard_cap_usd": round(cap, 2),
        }
    out["total_hard_cap_usd"] = round(total, 2)
    return out


def power_table() -> dict[str, Any]:
    rows = []
    for sw in (0.4, 0.6, 0.8):
        for sb in (0.1, 0.2):
            rows.append(
                {
                    "sigma_within": sw,
                    "sigma_between": sb,
                    "mde_ratio_89x5": round(an.mde_ratio(89, sw, sb, 5), 4),
                    "n_tasks_for_10pct_k5": an.n_tasks_for_mde(0.9, sw, sb, 5),
                }
            )
    ni = {
        f"discordance={p},deff={de}": {
            "margin_445_pairs": round(an.ni_margin(445, p, de), 4),
            "margin_267_pairs": round(an.ni_margin(267, p, de), 4),
        }
        for p in (0.1, 0.15)
        for de in (1.0, 1.5)
    }
    return {"cost": rows, "success_ni": ni}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.cost_truth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "plan", help="print the design, the schedule size and every arm's exact commands"
    )
    d = sub.add_parser("dry-run", help="full pipeline offline: mock upstream + scripted agent, $0")
    d.add_argument("--tasks", type=int, default=4)
    d.add_argument("--seeds", type=int, default=2)
    d.add_argument("--model", default="claude-sonnet-5")
    d.add_argument("--cap-usd", type=float, default=5.0)
    d.add_argument("--bootstrap", type=int, default=2000)
    d.add_argument("--out", type=Path, default=None, help="default: gitignored scratch dir")
    a = sub.add_parser("analyze", help="pre-registered analysis of a results dir")
    a.add_argument("results", type=Path)
    a.add_argument("--bootstrap", type=int, default=an.BOOTSTRAP_B)
    e = sub.add_parser("estimate", help="write the live-run cost estimate")
    e.add_argument("--out", type=Path, default=ESTIMATE_PATH)
    sub.add_parser("power", help="print the power table")
    args = ap.parse_args(argv)

    if args.cmd == "plan":
        for phase, dd in DESIGN.items():
            sched = build_schedule(
                [f"t{i}" for i in range(dd["tasks"])],
                list(range(dd["seeds"])),
                ARMS,
                dd["model"],
                0,
            )
            print(
                f"{phase}: {dd['model']} {dd['tasks']} tasks x {dd['seeds']} seeds x {len(ARMS)} arms = {len(sched)} runs"
            )
        claude = ["claude", "-p", "<instruction>", "--model", "<model>", "--output-format", "json"]
        for arm in arms_mod.ARMS.values():
            print(f"\n[{arm.name}] version={arm.version} verified={arm.verified}")
            for step in arm.install:
                print("  install:", " ".join(arms_mod.render(step, tools="$TOOLS")))
            print(
                "  launch: ",
                " ".join(
                    arms_mod.render(
                        arm.launch, claude, tools="$TOOLS", meter="$METER", state="$STATE"
                    )
                ),
            )
            print(
                "  env:    ",
                {k: v.format(meter="$METER", state="$STATE") for k, v in arm.upstream_env.items()},
            )
            if arm.claim:
                print(
                    "  claim:  ",
                    " ".join(arms_mod.render(arm.claim, tools="$TOOLS")),
                    f"({arm.claim_unit})",
                )
        pending = arms_mod.unverified()
        if pending:
            print(
                f"\nUNVERIFIED arm specs (the live driver must refuse to start): {', '.join(pending)}"
            )
        return 0
    if args.cmd == "dry-run":
        out = args.out or SCRATCH / "cost_truth" / time.strftime("dryrun-%Y%m%d-%H%M%S")
        runs = dry_run(
            out,
            [f"task{i:02d}" for i in range(args.tasks)],
            list(range(args.seeds)),
            args.model,
            args.cap_usd,
        )
        res = an.analyze(runs, list(ARMS), b=args.bootstrap)
        (out / "analysis.json").write_text(
            json.dumps(res, indent=2, sort_keys=True, default=str) + "\n"
        )
        print("DRY RUN - SYNTHETIC EFFECTS, NOT A RESULT")
        print(report(res))
        print(f"\nwrote {out}")
        return 0
    if args.cmd == "analyze":
        res = an.analyze(load_runs(args.results), list(ARMS), b=args.bootstrap)
        (args.results / "analysis.json").write_text(
            json.dumps(res, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(report(res))
        return 0
    if args.cmd == "estimate":
        est = estimate()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(est, indent=2) + "\n")
        print(
            json.dumps({k: v["hard_cap_usd"] for k, v in est["phases"].items()}),
            f"total ${est['total_hard_cap_usd']}",
        )
        print(f"wrote {args.out}")
        return 0
    print(json.dumps(power_table(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
