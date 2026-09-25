"""Schedule, isolate, run, and record — the part that is identical for every arm.

* ``build_schedule`` — paired blocks (task x seed) in random order, arm order randomised
  within each block (protocol §5).
* ``next_eligible`` — cache-TTL spacing: a task is not re-run until ``min_gap_s`` after its
  previous run ended, so no arm inherits another arm's warm prompt cache for that task.
* ``isolated_env`` — fresh HOME/XDG/CLAUDE_CONFIG_DIR/DISTIL_HOME per run, built from an
  allowlist so nothing (a base URL, a tool's config) leaks in from the operator's shell.
* ``summarise_run`` — one content-free record per run, derived ONLY from the meter log.
* ``dry_run`` — the full pipeline against the mock upstream + scripted agent.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.cost_truth import meter as m
from benchmarks.cost_truth import mock

ARMS = ("control", "rtk", "headroom", "distil")

#: Outcome statuses. COUNTED ones enter the analysis (intention-to-treat: an arm's own
#: crash is that arm's failure, and its spend still counts). EXCLUDED ones remove the
#: whole (task, seed) block from every arm, because they are not attributable to an arm.
COUNTED = ("solved", "failed", "timeout", "arm_crash")
EXCLUDED = ("infra_error", "budget_stop")

#: 5-minute prompt-cache TTL plus a minute of margin. Use 3900 s if the pilot shows any
#: 1-hour cache writes (``cache_creation_1h_input_tokens`` > 0).
MIN_GAP_S = 360.0

#: Environment variables that survive into a run. Everything else is dropped.
ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "TERM", "TZ")


@dataclass(frozen=True)
class RunSpec:
    task: str
    seed: int
    arm: str
    model: str
    priority: int

    @property
    def run_id(self) -> str:
        return f"{self.model}.{self.task}.s{self.seed}.{self.arm}"


def build_schedule(
    tasks: list[str], seeds: list[int], arms: tuple[str, ...], model: str, rng_seed: int
) -> list[RunSpec]:
    rng = random.Random(rng_seed)
    blocks = [(t, s) for t in tasks for s in seeds]
    rng.shuffle(blocks)
    out: list[RunSpec] = []
    for t, s in blocks:
        order = list(arms)
        rng.shuffle(order)
        out.extend(RunSpec(t, s, a, model, len(out) + i) for i, a in enumerate(order))
    return out


def next_eligible(
    pending: list[RunSpec], last_end: dict[str, float], now: float, min_gap_s: float
) -> int | None:
    """Index of the highest-priority pending run whose task is outside the cache TTL."""
    for i, r in enumerate(pending):
        if now - last_end.get(r.task, float("-inf")) >= min_gap_s:
            return i
    return None


def isolated_env(run_dir: Path, base: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    """A fresh, empty per-run home. Refuses a run dir that already has anything in it."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run dir is not fresh: {run_dir}")
    dirs = {
        "HOME": run_dir / "home",
        "XDG_CONFIG_HOME": run_dir / "home" / ".config",
        "XDG_CACHE_HOME": run_dir / "home" / ".cache",
        "XDG_DATA_HOME": run_dir / "home" / ".local" / "share",
        "XDG_STATE_HOME": run_dir / "home" / ".local" / "state",
        "CLAUDE_CONFIG_DIR": run_dir / "home" / ".claude",
        "DISTIL_HOME": run_dir / "state" / "distil",
        "TMPDIR": run_dir / "tmp",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    env = {k: base[k] for k in ENV_ALLOW if k in base}
    env.update({k: str(v) for k, v in dirs.items()})
    env.update(
        {
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",  # no autoupdate/telemetry calls
            "DISABLE_AUTOUPDATER": "1",
            "DO_NOT_TRACK": "1",  # neutral opt-out every arm honours; no per-tool knobs
        }
    )
    env.update(extra)
    return env


def summarise_run(
    spec: RunSpec, status: str, log: list[dict[str, Any]], **extra: Any
) -> dict[str, Any]:
    """Collapse a run's meter log into one record. Billing comes from the log and nothing else."""
    ok = [r for r in log if r.get("usage")]
    tot = {
        k: sum(r["usage"][k] for r in ok) for k in (*m.USAGE_KEYS, "cache_creation_1h_input_tokens")
    }
    main = [r for r in ok if r.get("model") and r["model"].startswith(spec.model)]
    if any(r.get("status") == 402 for r in log):
        status = "budget_stop"
    elif not ok and status in ("solved", "failed", "timeout"):
        status = "arm_crash"  # the meter saw no billed traffic: the chain is broken
    return {
        "run_id": spec.run_id,
        "task": spec.task,
        "seed": spec.seed,
        "arm": spec.arm,
        "model": spec.model,
        "priority": spec.priority,
        "status": status,
        "solved": status == "solved",
        "requests": len(ok),
        "turns": len(main),
        "usage": tot,
        "cost_usd": round(sum(r["cost_usd"] for r in ok), 8),
        "first_request_cache_read": main[0]["usage"]["cache_read_input_tokens"] if main else 0,
        **extra,
    }


def dry_run(
    out_dir: Path,
    tasks: list[str],
    seeds: list[int],
    model: str,
    cap_usd: float,
    rng_seed: int = 20260925,
    min_gap_s: float = MIN_GAP_S,
    run_seconds: float = 90.0,
) -> list[dict[str, Any]]:
    """Every stage of a live run except the real agent and provider. Virtual clock for spacing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = build_schedule(tasks, seeds, ARMS, model, rng_seed)
    spend = m.SpendMeter(cap_usd)
    runs: list[dict[str, Any]] = []
    last_end: dict[str, float] = {}
    now = 0.0
    with mock.MockUpstream() as up:
        while pending:
            i = next_eligible(pending, last_end, now, min_gap_s)
            if i is None:
                now = min(last_end[r.task] for r in pending) + min_gap_s
                continue
            spec = pending.pop(i)
            run_dir = out_dir / "runs" / spec.run_id
            isolated_env(run_dir, {}, {})
            log_path = out_dir / "meter" / f"{spec.run_id}.jsonl"
            cfg = m.MeterConfig(upstream=up.url, log_path=log_path, spend=spend, run_id=spec.run_id)
            t0 = time.monotonic()
            with m.UsageMeter(cfg) as meter:
                try:
                    res = mock.scripted_agent(meter.base_url, spec.task, spec.seed, spec.arm, model)
                    status, claim = ("solved" if res.solved else "failed"), res.claimed_tokens_saved
                except OSError:  # urllib raises HTTPError (an OSError) on the meter's 402
                    status, claim = "arm_crash", None
            rec = summarise_run(
                spec,
                status,
                m.read_log(log_path),
                virtual_start_s=now,
                wall_s=round(time.monotonic() - t0, 3),
                claim={"tokens_saved": claim, "unit": "synthetic"},
            )
            runs.append(rec)
            shutil.rmtree(run_dir)  # isolation: nothing survives into the next run
            last_end[spec.task] = now + run_seconds
            now += run_seconds
            if rec["status"] == "budget_stop":
                break  # hard stop: nothing further is launched once the cap refuses a call
    write_results(
        out_dir,
        runs,
        {
            "mode": "dry-run",
            "synthetic": True,
            "model": model,
            "cap_usd": cap_usd,
            "spent_usd": round(spend.spent, 6),
            "rng_seed": rng_seed,
            "min_gap_s": min_gap_s,
            "tasks": tasks,
            "seeds": seeds,
            "arms": list(ARMS),
            "effects": {k: asdict(v) for k, v in mock.SYNTHETIC_EFFECTS.items()},
        },
    )
    return runs


def write_results(out_dir: Path, runs: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    with (out_dir / "runs.jsonl").open("w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_runs(out_dir: Path) -> list[dict[str, Any]]:
    lines = (out_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]
