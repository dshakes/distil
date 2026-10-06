"""python -m benchmarks.swebench_outcome {plan,run,grade,report}"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agent import Budget, Cfg, direct_client
from .arms import (
    ARM_NAMES,
    CM_CLEAR_AT_LEAST,
    CM_KEEP,
    CM_TRIGGER,
    Arm,
    ArmUnavailable,
    distil_arm,
    parse_arms,
    plain_arm,
    provider_cm_arm,
)
from .run import (
    ARMS,
    DATASET,
    UNMEASURED_FACTOR,
    calibrate,
    estimate,
    estimate_arms,
    load_ids,
    load_records,
    pilot_cost,
    read_manifest,
    read_results,
    reuse_arm,
    run_all,
    select_ids,
    spent_in,
)

DEFAULT_OUT = "benchmarks/results/swebench_outcome"


def _reuse_pairs(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for it in items:
        name, sep, path = it.partition("=")
        if not sep or name not in ARM_NAMES or not path:
            raise SystemExit(f"--reuse-arm expects ARM=DIR with ARM in {ARM_NAMES}; got {it!r}")
        out[name] = Path(path)
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swebench_outcome", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "grade", "report"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=DEFAULT_OUT)
        s.add_argument(
            "--dataset",
            default=DATASET,
            help=f"HuggingFace dataset, test split (default: {DATASET})",
        )
        s.add_argument(
            "--arms",
            default=",".join(ARMS),
            help=f"comma list from {','.join(ARM_NAMES)} (default: {','.join(ARMS)})",
        )
        if name in ("plan", "run"):
            s.add_argument(
                "--instances", help="file of instance ids (default: SWE-bench Lite test)"
            )
            s.add_argument(
                "--difficulty",
                help="comma list of `difficulty` annotations to keep (SWE-bench Verified only), "
                "e.g. '1-4 hours,>4 hours'",
            )
            s.add_argument("--limit", type=int)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--model", default="claude-sonnet-5-5")
            s.add_argument(
                "--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium"
            )
            s.add_argument("--max-steps", type=int, default=40)
            s.add_argument("--task-timeout", type=float, default=1800.0)
            s.add_argument(
                "--max-tokens",
                type=int,
                default=16_000,
                help="per-response cap; raise it at effort xhigh/max, where adaptive thinking "
                "can spend 16k before any tool call",
            )
            s.add_argument("--price-in", type=float)
            s.add_argument("--price-out", type=float)
            s.add_argument(
                "--reuse-arm",
                action="append",
                default=[],
                metavar="ARM=DIR",
                help="copy ARM's rows from a finished run (same model/effort) instead of "
                "re-running it; its grades are carried over and the report says so. Repeatable.",
            )
        if name == "plan":
            s.add_argument("--pilot", help="results.jsonl from a pilot to calibrate the estimate")
            s.add_argument(
                "--calibrate",
                metavar="DIR",
                help="a finished run dir: per-arm cost from its results.jsonl (unmeasured arms "
                f"are priced at plain x{UNMEASURED_FACTOR})",
            )
        if name == "run":
            s.add_argument("--budget-usd", type=float)
            s.add_argument("--i-understand-this-costs-money", action="store_true")
            s.add_argument("--rtk-bin", help="linux x86_64 rtk binary (default: pinned download)")
            s.add_argument(
                "--selective-python",
                help="python with selective-context installed (default: this interpreter)",
            )
            s.add_argument("--cm-trigger", type=int, default=CM_TRIGGER, help="provider-cm trigger")
            s.add_argument("--cm-keep", type=int, default=CM_KEEP, help="provider-cm keep")
            s.add_argument("--cm-clear-at-least", type=int, default=CM_CLEAR_AT_LEAST)
        if name == "grade":
            s.add_argument("--max-workers", type=int, default=4)
        if name == "report":
            s.add_argument("--margin", type=float, default=0.05)
            s.add_argument(
                "--cost-per-solved",
                action="store_true",
                help="append $ per solved task vs plain (paired cluster bootstrap)",
            )
    return p


def build_arms(names: tuple[str, ...], a: argparse.Namespace) -> tuple[list[Arm], list[Any]]:
    """Real arms for `run`. Returns (arms, things to close). Raises ArmUnavailable before spend."""
    arms: list[Arm] = []
    closers: list[Any] = []
    for n in names:
        if n == "plain":
            arms.append(plain_arm())
        elif n == "distil":
            arms.append(distil_arm())
        elif n == "provider-cm":
            arms.append(provider_cm_arm(a.cm_trigger, a.cm_keep, a.cm_clear_at_least))
        elif n == "rtk":
            from .arm_rtk import fetch_rtk, rtk_arm

            binary = Path(a.rtk_bin) if a.rtk_bin else fetch_rtk()
            if not binary.is_file():
                raise ArmUnavailable(f"rtk: --rtk-bin {binary} is not a file")
            arms.append(rtk_arm(binary))
        elif n == "selective":
            from .arm_selective import make_selective

            arm, worker = make_selective(a.selective_python or sys.executable)
            arms.append(arm)
            closers.append(worker)
    return arms, closers


def _plan(a: argparse.Namespace, cfg: Cfg, ids: list[str], names: tuple[str, ...]) -> int:
    print(f"model={cfg.model} effort={cfg.effort} arms={names} tasks={len(ids)} seed={a.seed}")
    print("\n".join(ids))
    reuse = _reuse_pairs(a.reuse_arm)
    if a.calibrate:
        cal = calibrate(Path(a.calibrate) / "results.jsonl", cfg)
        done = {
            arm: len(
                {r["instance_id"] for r in read_results(src / "results.jsonl") if r["arm"] == arm}
                & set(ids)
            )
            for arm, src in reuse.items()
        }
        est = estimate_arms(len(ids), names, cal, done)
        if cal["mismatch"]:
            print(
                f"WARNING: calibration ran {', '.join(cal['mismatch'])}, not {cfg.model}@{cfg.effort}"
            )
        for arm, r in est["arms"].items():
            print(
                f"  {arm}: ${r['per_task_usd']:.4f}/task x {r['tasks_to_run']} tasks = "
                f"${r['total_usd']:.2f} ({r['basis']})"
            )
        print(
            f"ESTIMATE ${est['total_usd']:.2f} total; calibrated from {cal['source']}; "
            "reused arms cost nothing"
        )
        return 0
    if reuse:
        print("note: --reuse-arm only changes the estimate when --calibrate is given")
    est1 = estimate(len(ids), len(names), cfg, pilot_cost(Path(a.pilot)) if a.pilot else None)
    print(
        f"ESTIMATE ${est1['total_usd']:.2f} total (${est1['per_task_arm_usd']:.2f}/task-arm); "
        f"{est1['basis']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    out = Path(a.out)
    names = parse_arms(a.arms)
    if a.cmd in ("plan", "run"):
        cfg = Cfg(
            a.model,
            a.effort,
            a.max_steps,
            a.task_timeout,
            max_tokens=a.max_tokens,
            pin=a.price_in,
            pout=a.price_out,
        )
        diff = [d.strip() for d in a.difficulty.split(",") if d.strip()] if a.difficulty else None
        ids = select_ids(load_ids(a.instances, a.dataset, diff), a.seed, a.limit)
    if a.cmd == "plan":  # never touches the Anthropic API, the network, or any arm dependency
        return _plan(a, cfg, ids, names)
    if a.cmd == "run":
        if not a.i_understand_this_costs_money or a.budget_usd is None:
            print("refusing: run needs --i-understand-this-costs-money and --budget-usd")
            return 2
        from .env import DockerEnv

        reuse = _reuse_pairs(a.reuse_arm)
        try:
            to_run = tuple(n for n in names if n not in reuse)
            arms, closers = build_arms(to_run, a)
        except ArmUnavailable as e:
            print(f"refusing: {e}")
            return 2
        try:
            for arm, src in reuse.items():
                if arm in names:
                    n = reuse_arm(out, arm, src, ids, cfg)
                    print(f"reused {n} {arm} rows from {src}")
            if not arms:
                print("nothing to run: every requested arm was reused")
                return 0
            tasks = {r["instance_id"]: r for r in load_records(ids, a.dataset)}
            ordered = [tasks[i] for i in ids if i in tasks]
            budget = Budget(a.budget_usd)
            budget.spent = spent_in(out / "results.jsonl")
            stats = run_all(ordered, out, direct_client(), DockerEnv, cfg, budget, tuple(arms))
            print(json.dumps(stats), f"spent=${budget.spent:.2f}")
        finally:
            for c in closers:
                c.close()
        return 0
    if a.cmd == "grade":
        from .grade import grade

        rows = grade(out, a.max_workers, dataset=a.dataset, arms=names)
        print(f"graded {len(rows)} (instance, arm) rows -> {out / 'grades.jsonl'}")
        return 0
    from .report import analyse, cost_section, dataset_title, markdown

    gp = out / "grades.jsonl"
    grades = [json.loads(ln) for ln in gp.read_text().splitlines()] if gp.exists() else []
    results = read_results(out / "results.jsonl")
    md = markdown(
        analyse(results, grades, a.margin, names),
        {k: v for k, v in read_manifest(out).items() if k in names},
        title=dataset_title(a.dataset),
    )
    if a.cost_per_solved:
        md += "\n" + cost_section(results, grades, names)
    (out / "report.md").write_text(md)
    print(md)
    return 0
