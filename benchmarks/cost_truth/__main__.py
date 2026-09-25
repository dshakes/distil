"""CLI: ``python -m benchmarks.cost_truth {plan,dry-run,analyze,estimate,power,live}``.

Only ``live`` without ``--mock`` can spend money, and it refuses unless
``--i-approve-spend`` equals the phase's pre-registered hard cap to the cent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.cost_truth import analysis as an
from benchmarks.cost_truth import arms as arms_mod
from benchmarks.cost_truth import live
from benchmarks.cost_truth.live import DESIGN, PROFILE, SAFETY
from benchmarks.cost_truth.runner import ARMS, build_schedule, dry_run, load_runs
from benchmarks.outpath import ROOT, SCRATCH

ESTIMATE_PATH = ROOT / "benchmarks" / "results" / "cost_truth" / "cost_estimate.json"

TOOLS_DIR = SCRATCH / "cost_truth" / "tools"
TASKS_ROOT = SCRATCH / "cost_truth" / "tasks"
HARBOR = ["uvx", "--python", "3.12", "--from", f"harbor=={arms_mod.HARBOR_VERSION}", "harbor"]


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
    lv = sub.add_parser("live", help="run a phase: canary preflight, then the paired schedule")
    lv.add_argument("--phase", required=True, choices=sorted(DESIGN))
    lv.add_argument(
        "--i-approve-spend",
        dest="approve",
        default=None,
        metavar="USD",
        help="must equal the phase's hard cap to the cent (pilot: see `plan`)",
    )
    lv.add_argument(
        "--mock",
        action="store_true",
        help="$0: mock upstream + host stand-ins (real distil wrap); no approval needed",
    )
    lv.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help=f"default: harbor download {arms_mod.TASK_DATASET}",
    )
    lv.add_argument("--tools-dir", type=Path, default=TOOLS_DIR)
    lv.add_argument("--out", type=Path, default=None)
    lv.add_argument("--concurrency", type=int, default=4)
    lv.add_argument("--bind", default="127.0.0.1", help="meter bind address")
    lv.add_argument(
        "--preflight-only", action="store_true", help="run only the four canaries, then stop"
    )
    lv.add_argument(
        "--already-spent",
        type=float,
        default=0.0,
        metavar="USD",
        help="billed by earlier attempts of this phase; subtracted from the hard cap",
    )
    lv.add_argument(
        "--url-host", default="host.docker.internal", help="meter host as seen from containers"
    )
    args = ap.parse_args(argv)

    if args.cmd == "plan":
        for phase, dd in DESIGN.items():
            n = len(
                build_schedule(
                    [f"t{i}" for i in range(dd["tasks"])],
                    list(range(dd["seeds"])),
                    ARMS,
                    dd["model"],
                    0,
                )
            )
            print(
                f"{phase}: {dd['model']} {dd['tasks']} tasks x {dd['seeds']} seeds x {len(ARMS)} arms = {n} runs, cap ${live.phase_cap(phase):.2f}"
            )
        for arm in arms_mod.ARMS.values():
            print(f"\n[{arm.name}] version={arm.version} verified={arm.verified}")
            print(
                "  install (root):\n    "
                + arms_mod.install_script(arm.name).strip().replace("\n", "\n    ")
            )
            print("  setup (agent):  " + arms_mod.agent_setup_script(arm.name).strip())
            print("  env:            " + json.dumps(arms_mod.arm_env(arm.name, "$METER")))
            print(
                "  launch:\n    "
                + arms_mod.launch_script(arm.name, "<model>", "/logs/agent")
                .strip()
                .replace("\n", "\n    ")
            )
            for ev in arm.evidence:
                print(f"  evidence: {ev}")
        pending = arms_mod.unverified()
        print(
            f"\nUNVERIFIED arm specs: {', '.join(pending)}"
            if pending
            else "\nall arm specs verified"
        )
        return 0
    if args.cmd == "live":
        return _live(args)
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


def _live(args: argparse.Namespace) -> int:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.mock:
        from benchmarks.cost_truth import mock

        out = args.out or SCRATCH / "cost_truth" / f"live-mock-{args.phase}-{stamp}"
        d = DESIGN[args.phase]
        tasks = [f"task{i:02d}" for i in range(d["tasks"])]
        distil_bin = Path(sys.executable).parent / "distil"
        with mock.MockUpstream() as up:
            cfg = live.LiveConfig(
                args.phase,
                live.phase_cap(args.phase),
                out,
                upstream=up.url,
                url_host="127.0.0.1",
                concurrency=args.concurrency,
                min_gap_s=0.0,
                tasks=tasks,
            )
            runs = live.run_phase(cfg, live.LocalExecutor(out / "work", distil_bin))
        shutil.rmtree(out / "work", ignore_errors=True)
        print("LIVE PIPELINE AGAINST THE MOCK UPSTREAM - STAND-INS, SYNTHETIC, NOT A RESULT")
    else:
        try:
            cap = live.check_approval(args.phase, args.approve)
        except live.ApprovalError as e:
            print(f"cost-truth: {e}", file=sys.stderr)
            return 2
        if not 0 <= args.already_spent < cap:
            print(f"cost-truth: --already-spent must be in [0, {cap})", file=sys.stderr)
            return 2
        cap = round(cap - args.already_spent, 6)  # one approval covers every attempt of the phase
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("cost-truth: ANTHROPIC_API_KEY is not set", file=sys.stderr)
            return 2
        if (
            subprocess.call(
                ["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            != 0
        ):
            print("cost-truth: docker is not reachable", file=sys.stderr)
            return 2
        live.prepare_tools(args.tools_dir)
        tasks_dir = args.tasks_dir
        if tasks_dir is None:
            # harbor stores a hub dataset "org/name[@ver]" under TASKS_ROOT/name
            tasks_dir = TASKS_ROOT / arms_mod.TASK_DATASET.split("@")[0].split("/")[-1]
            if not tasks_dir.is_dir():
                subprocess.run(
                    [*HARBOR, "download", arms_mod.TASK_DATASET, "-o", str(TASKS_ROOT)], check=True
                )
        manifest = sorted(p.name for p in tasks_dir.iterdir() if (p / "task.toml").exists())
        if not manifest:
            print(f"cost-truth: no tasks under {tasks_dir}", file=sys.stderr)
            return 2
        out = args.out or ROOT / "benchmarks" / "results" / "cost_truth" / f"{args.phase}-{stamp}"
        trials = SCRATCH / "cost_truth" / "trials" / f"{args.phase}-{stamp}"  # content: gitignored
        executor = live.HarborExecutor(
            tasks_dir, trials, args.tools_dir.resolve(), args.url_host, HARBOR
        )
        cfg = live.LiveConfig(
            args.phase,
            cap,
            out,
            bind=args.bind,
            url_host=args.url_host,
            concurrency=args.concurrency,
            tasks=manifest,
            logs_root=trials / "agent-logs",
        )
        try:
            runs = live.run_phase(cfg, executor, preflight_only=args.preflight_only)
        except live.PreflightError as e:
            print(f"cost-truth: {e}\nreport: {out / 'preflight.json'}", file=sys.stderr)
            return 1
        if args.preflight_only:
            print((out / "preflight.json").read_text())
            print(f"all four canaries passed; wrote {out}")
            return 0
    summary = live.pilot_summary(runs) if args.phase == "pilot" else None
    if summary is not None:
        (out / "pilot_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        res = an.analyze(runs, list(ARMS))
        (out / "analysis.json").write_text(
            json.dumps(res, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(report(res))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
