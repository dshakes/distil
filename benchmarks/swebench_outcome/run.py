"""Instance selection, plan/cost estimate, crash-safe resumable paired run."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .agent import Budget, Cfg, price, run_agent
from .env import Env

ARMS = ("plain", "distil")
DATASET = "princeton-nlp/SWE-bench_Lite"
# Placeholders, used ONLY until a pilot calibrates them (`plan --pilot results.jsonl`).
DEFAULT_STEPS, DEFAULT_IN_PER_STEP, DEFAULT_OUT_PER_STEP = 30, 35_000, 1_500


def select_ids(ids: list[str], seed: int, limit: int | None) -> list[str]:
    ids = sorted(set(ids))
    random.Random(seed).shuffle(ids)
    return ids[:limit] if limit else ids


def load_ids(path: str | None) -> list[str]:
    if path:
        return [
            ln.strip()
            for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    return [r["instance_id"] for r in load_records(None)]


def load_records(ids: Iterable[str] | None) -> list[dict[str, Any]]:
    """Full instance records (need problem_statement). Network: HuggingFace only, not Anthropic."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise SystemExit("the `datasets` package is required (pip install datasets)") from e
    rows = list(load_dataset(DATASET, split="test"))
    want = set(ids) if ids is not None else None
    return [dict(r) for r in rows if want is None or r["instance_id"] in want]


def estimate(
    n_tasks: int, arms: int, cfg: Cfg, per_task_usd: float | None = None
) -> dict[str, Any]:
    pin, pout = price(cfg.model, cfg.pin, cfg.pout)
    if per_task_usd is None:
        s = DEFAULT_STEPS
        per_task_usd = s * (DEFAULT_IN_PER_STEP * pin + DEFAULT_OUT_PER_STEP * pout) / 1e6
        basis = f"PLACEHOLDER: {s} steps x {DEFAULT_IN_PER_STEP} in + {DEFAULT_OUT_PER_STEP} out tok/step"
    else:
        basis = "calibrated from pilot (mean cost per task-arm)"
    return {
        "per_task_arm_usd": per_task_usd,
        "total_usd": per_task_usd * n_tasks * arms,
        "basis": basis,
    }


def pilot_cost(results_path: Path) -> float:
    rows = read_results(results_path)
    if not rows:
        raise SystemExit(f"no pilot results in {results_path}")
    return sum(r["cost_usd"] for r in rows) / len(rows)


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue  # torn last line from a crash; the pair is simply re-run
    return out


def _append(path: Path, rec: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_all(
    tasks: list[dict[str, Any]],
    out: Path,
    client: Any,
    env_factory: Callable[[str], Env],
    cfg: Cfg,
    budget: Budget,
    arms: tuple[str, ...] = ARMS,
    compress: Callable[..., Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Tasks are already in seeded order. Arm order alternates per task to cancel drift."""
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {(r["instance_id"], r["arm"]) for r in read_results(results)}
    stats = {"ran": 0, "skipped": 0, "stopped_on_budget": 0}
    for i, task in enumerate(tasks):
        order = arms if i % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            key = (task["instance_id"], arm)
            if key in done:
                stats["skipped"] += 1
                continue
            if budget.exhausted:
                stats["stopped_on_budget"] = 1
                log(f"budget ${budget.limit} reached (${budget.spent:.2f}); stopping")
                return stats
            try:
                env = env_factory(task["instance_id"])
            except Exception as e:  # noqa: BLE001 - env build failure is its own class
                rec = {
                    "instance_id": key[0],
                    "arm": arm,
                    "failure_class": "env_error",
                    "error": repr(e)[:500],
                    "patch": "",
                    "steps": 0,
                    "cost_usd": 0.0,
                    "usage": {},
                    "expand_calls": 0,
                    "wall_s": 0.0,
                    "stop": "env_error",
                }
            else:
                try:
                    rec = run_agent(client, env, task, arm, cfg, budget, compress=compress)
                finally:
                    env.close()
            if rec["failure_class"] == "budget_stopped":
                stats["stopped_on_budget"] = 1
                log("budget reached mid-task; partial task not recorded (re-run on resume)")
                return stats
            tdir = out / "transcripts" / arm
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / f"{key[0]}.json").write_text(json.dumps(rec.pop("transcript", []), default=str))
            _append(results, rec)
            stats["ran"] += 1
            log(
                f"{key[0]} [{arm}] class={rec['failure_class']} steps={rec['steps']} "
                f"${rec['cost_usd']:.3f} (total ${budget.spent:.2f})"
            )
    return stats
