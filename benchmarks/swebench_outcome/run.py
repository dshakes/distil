"""Instance selection, plan/cost estimate, crash-safe resumable paired run."""

from __future__ import annotations

import json
import os
import random
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .agent import Budget, Cfg, price, run_agent
from .arms import BASELINE, Arm
from .env import Env

ARMS = ("plain", "distil")  # default pair; any subset of arms.ARM_NAMES is allowed
UNMEASURED_FACTOR = 1.5  # an arm with no cached results is priced at plain's mean x this
REUSE_MANIFEST = "reuse.json"
DATASET = "princeton-nlp/SWE-bench_Lite"
# Placeholders, used ONLY until a pilot calibrates them (`plan --pilot results.jsonl`).
DEFAULT_STEPS, DEFAULT_IN_PER_STEP, DEFAULT_OUT_PER_STEP = 30, 35_000, 1_500


def select_ids(ids: list[str], seed: int, limit: int | None) -> list[str]:
    ids = sorted(set(ids))
    random.Random(seed).shuffle(ids)
    return ids[:limit] if limit else ids


def load_ids(
    path: str | None, dataset: str = DATASET, difficulty: Iterable[str] | None = None
) -> list[str]:
    """Ids from *path*, else every test instance of *dataset* (optionally only the given
    `difficulty` annotations, e.g. SWE-bench Verified's "1-4 hours")."""
    if path:
        if difficulty:
            raise SystemExit(
                "--difficulty filters the dataset; it cannot be combined with --instances"
            )
        return [
            ln.strip()
            for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    rows = load_records(None, dataset)
    if difficulty:
        want = set(difficulty)
        if rows and "difficulty" not in rows[0]:
            raise SystemExit(
                f"{dataset} has no `difficulty` field; --difficulty needs SWE-bench Verified"
            )
        rows = [r for r in rows if r["difficulty"] in want]
    return [r["instance_id"] for r in rows]


def load_records(ids: Iterable[str] | None, dataset: str = DATASET) -> list[dict[str, Any]]:
    """Full instance records (need problem_statement). Network: HuggingFace only, not Anthropic."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise SystemExit("the `datasets` package is required (pip install datasets)") from e
    rows = list(load_dataset(dataset, split="test"))
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


def calibrate(results_path: Path, cfg: Cfg) -> dict[str, Any]:
    """Per-arm mean cost per task from cached results. Rows that never reached the API
    (env_error before any step) are not priced. `mismatch` lists model/effort differences."""
    rows = [r for r in read_results(results_path) if r.get("steps")]
    if not rows:
        raise SystemExit(f"no calibration results in {results_path}")
    per_arm: dict[str, list[float]] = {}
    for r in rows:
        per_arm.setdefault(r["arm"], []).append(r["cost_usd"])
    seen = {(r.get("model"), r.get("effort")) for r in rows}
    return {
        "mean": {a: sum(v) / len(v) for a, v in per_arm.items()},
        "n": {a: len(v) for a, v in per_arm.items()},
        "mismatch": sorted(f"{m}@{e}" for m, e in seen if (m, e) != (cfg.model, cfg.effort)),
        "source": str(results_path),
    }


def estimate_arms(
    n_tasks: int,
    arms: tuple[str, ...],
    cal: dict[str, Any],
    done: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Cost per arm. Measured mean when the calibration has that arm, else plain's mean x
    UNMEASURED_FACTOR (a safety margin, not a prediction). `done[arm]` tasks are already covered
    by reused rows and cost nothing."""
    mean = cal["mean"]
    if BASELINE not in mean:
        raise SystemExit("calibration has no plain arm to price unmeasured arms from")
    rows: dict[str, dict[str, Any]] = {}
    for a in arms:
        measured = a in mean
        per = mean[a] if measured else mean[BASELINE] * UNMEASURED_FACTOR
        todo = max(n_tasks - (done or {}).get(a, 0), 0)
        rows[a] = {
            "per_task_usd": per,
            "tasks_to_run": todo,
            "total_usd": per * todo,
            "basis": f"measured, n={cal['n'][a]}"
            if measured
            else f"proxy: plain mean x{UNMEASURED_FACTOR} (no cached results for this arm)",
        }
    return {"arms": rows, "total_usd": sum(r["total_usd"] for r in rows.values())}


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
    arms: tuple[str | Arm, ...] = ARMS,
    compress: Callable[..., Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Tasks are already in seeded order. The arm order rotates per task (two arms: swaps) so no
    arm always runs first; this cancels time-of-day drift."""
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {(r["instance_id"], r["arm"]) for r in read_results(results)}
    stats = {"ran": 0, "skipped": 0, "stopped_on_budget": 0}
    for i, task in enumerate(tasks):
        k = i % len(arms)
        for arm_ in (*arms[k:], *arms[:k]):
            arm = arm_ if isinstance(arm_, str) else arm_.name
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
                    rec = run_agent(client, env, task, arm_, cfg, budget, compress=compress)
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


def spent_in(path: Path) -> float:
    """Live spend already in *path*. Reused rows were paid for in another run: not counted."""
    return sum(r.get("cost_usd", 0) for r in read_results(path) if "reused_from" not in r)


def read_manifest(out: Path) -> dict[str, dict[str, Any]]:
    mp = out / REUSE_MANIFEST
    manifest: dict[str, dict[str, Any]] = json.loads(mp.read_text()) if mp.exists() else {}
    return manifest


def reuse_arm(out: Path, arm: str, src: Path, ids: Iterable[str], cfg: Cfg) -> int:
    """Copy *arm*'s rows (and transcripts) for *ids* from a finished run into *out*, tagged
    `reused_from`; returns how many were copied. The source must have used the same model and
    effort, otherwise the paired comparison would silently mix configurations."""
    src_rows = [r for r in read_results(src / "results.jsonl") if r["arm"] == arm]
    if not src_rows:
        raise SystemExit(f"--reuse-arm: no {arm!r} rows in {src / 'results.jsonl'}")
    out.mkdir(parents=True, exist_ok=True)
    want, dest = set(ids), out / "results.jsonl"
    have = {(r["instance_id"], r["arm"]) for r in read_results(dest)}
    n = 0
    for r in src_rows:
        if r["instance_id"] not in want or (r["instance_id"], arm) in have:
            continue
        if (r.get("model"), r.get("effort")) != (cfg.model, cfg.effort):
            raise SystemExit(
                f"--reuse-arm {arm}: {src} ran {r.get('model')}@{r.get('effort')}, this run is "
                f"{cfg.model}@{cfg.effort}; refusing to mix configurations"
            )
        t = src / "transcripts" / arm / f"{r['instance_id']}.json"
        if t.exists():
            (out / "transcripts" / arm).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(t, out / "transcripts" / arm / t.name)
        _append(dest, {**r, "reused_from": str(src)})
        n += 1
    manifest = read_manifest(out)
    manifest[arm] = {"from": str(src), "rows": manifest.get(arm, {}).get("rows", 0) + n}
    (out / REUSE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return n
