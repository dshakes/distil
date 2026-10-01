"""python -m benchmarks.swebench_outcome {plan,run,grade,report}"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import Budget, Cfg, direct_client
from .run import (
    ARMS,
    estimate,
    load_ids,
    load_records,
    pilot_cost,
    read_results,
    run_all,
    select_ids,
)

DEFAULT_OUT = "benchmarks/results/swebench_outcome"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swebench_outcome", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "grade", "report"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=DEFAULT_OUT)
        if name in ("plan", "run"):
            s.add_argument(
                "--instances", help="file of instance ids (default: SWE-bench Lite test)"
            )
            s.add_argument("--limit", type=int)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--model", default="claude-sonnet-5-5")
            s.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
            s.add_argument("--max-steps", type=int, default=40)
            s.add_argument("--task-timeout", type=float, default=1800.0)
            s.add_argument("--price-in", type=float)
            s.add_argument("--price-out", type=float)
        if name == "plan":
            s.add_argument("--pilot", help="results.jsonl from a pilot to calibrate the estimate")
        if name == "run":
            s.add_argument("--budget-usd", type=float)
            s.add_argument("--i-understand-this-costs-money", action="store_true")
        if name == "grade":
            s.add_argument("--max-workers", type=int, default=4)
        if name == "report":
            s.add_argument("--margin", type=float, default=0.05)
    return p


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    out = Path(a.out)
    if a.cmd in ("plan", "run"):
        cfg = Cfg(a.model, a.effort, a.max_steps, a.task_timeout, pin=a.price_in, pout=a.price_out)
        ids = select_ids(load_ids(a.instances), a.seed, a.limit)
    if a.cmd == "plan":  # never touches the Anthropic API
        est = estimate(len(ids), len(ARMS), cfg, pilot_cost(Path(a.pilot)) if a.pilot else None)
        print(f"model={cfg.model} effort={cfg.effort} arms={ARMS} tasks={len(ids)} seed={a.seed}")
        print("\n".join(ids))
        print(
            f"ESTIMATE ${est['total_usd']:.2f} total (${est['per_task_arm_usd']:.2f}/task-arm); "
            f"{est['basis']}"
        )
        return 0
    if a.cmd == "run":
        if not a.i_understand_this_costs_money or a.budget_usd is None:
            print("refusing: run needs --i-understand-this-costs-money and --budget-usd")
            return 2
        from .env import DockerEnv

        tasks = {r["instance_id"]: r for r in load_records(ids)}
        ordered = [tasks[i] for i in ids if i in tasks]
        budget = Budget(a.budget_usd)
        budget.spent = sum(r.get("cost_usd", 0) for r in read_results(out / "results.jsonl"))
        stats = run_all(ordered, out, direct_client(), DockerEnv, cfg, budget)
        print(json.dumps(stats), f"spent=${budget.spent:.2f}")
        return 0
    if a.cmd == "grade":
        from .grade import grade

        rows = grade(out, a.max_workers)
        print(f"graded {len(rows)} (instance, arm) rows -> {out / 'grades.jsonl'}")
        return 0
    from .report import analyse, markdown

    gp = out / "grades.jsonl"
    grades = [json.loads(ln) for ln in gp.read_text().splitlines()] if gp.exists() else []
    md = markdown(analyse(read_results(out / "results.jsonl"), grades, a.margin))
    (out / "report.md").write_text(md)
    print(md)
    return 0
