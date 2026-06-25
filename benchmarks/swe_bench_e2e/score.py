#!/usr/bin/env python3
"""Score E7 prediction patches with the OFFICIAL SWE-bench harness and aggregate.

This is a thin, auditable wrapper around ``swebench.harness.run_evaluation`` — it does
not re-implement any scoring. For each condition it:

1. shells out to the official harness on that condition's predictions JSONL
   (``-p predictions/<cond>.jsonl``), which builds/pulls the per-instance Docker image,
   applies the model patch + the hidden test patch, runs the test command, and writes
   the canonical ``<run_id>.<model>.json`` report;
2. reads that report's ``resolved_ids`` to get the per-instance pass/fail;
3. computes pass@1 with a Wilson 95% CI (:mod:`benchmarks.swe_bench_e2e.stats`).

Every number it emits traces back to a harness-written report on disk — nothing here
decides resolution itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.swe_bench_e2e.stats import pass_at_1  # noqa: E402

DATASET = "princeton-nlp/SWE-bench_Verified"
SWEBENCH_PYTHON = ROOT / ".venv-swebench/bin/python"


def run_harness(
    predictions: Path,
    run_id: str,
    *,
    max_workers: int = 4,
    namespace: str = "swebench",
    timeout: int = 1800,
    cache_level: str = "env",
) -> dict[str, Any]:
    """Invoke the official harness on a predictions file. Returns the parsed report."""
    cmd = [
        str(SWEBENCH_PYTHON),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        DATASET,
        "--predictions_path",
        str(predictions),
        "--run_id",
        run_id,
        "--max_workers",
        str(max_workers),
        "--namespace",
        namespace,
        "--timeout",
        str(timeout),
        "--cache_level",
        cache_level,
    ]
    env = {"DOCKER_DEFAULT_PLATFORM": "linux/amd64"}
    import os

    full_env = {**os.environ, **env}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=full_env, text=True)
    elapsed = round(time.time() - t0, 1)
    # The harness writes <run_id>.<model_name>.json into the cwd.
    model_name = _model_name(predictions)
    report_path = ROOT / f"{model_name}.{run_id}.json"
    if not report_path.exists():
        # swebench names it <model>.<run_id>.json; fall back to a glob.
        candidates = sorted(ROOT.glob(f"*{run_id}.json"))
        report_path = candidates[-1] if candidates else report_path
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["_harness_seconds"] = elapsed
    report["_harness_returncode"] = proc.returncode
    report["_report_path"] = str(report_path)
    return report


def _model_name(predictions: Path) -> str:
    for line in predictions.read_text().splitlines():
        if line.strip():
            return json.loads(line)["model_name_or_path"]
    return "unknown"


def aggregate(report: dict[str, Any], all_ids: list[str]) -> dict[str, Any]:
    """Turn a harness report into a pass@1 + per-instance breakdown over ``all_ids``."""
    resolved = set(report.get("resolved_ids", []))
    completed = set(report.get("completed_ids", []))
    errored = set(report.get("error_ids", []))
    unresolved = set(report.get("unresolved_ids", []))
    n = len(all_ids)
    n_resolved = sum(1 for i in all_ids if i in resolved)
    per_instance = {
        i: {
            "resolved": i in resolved,
            "completed": i in completed,
            "errored": i in errored,
            "unresolved": i in unresolved,
        }
        for i in all_ids
    }
    summary = pass_at_1(n_resolved, n)
    summary.update(
        {
            "n_completed": len(completed & set(all_ids)),
            "n_errored": len(errored & set(all_ids)),
            "harness_seconds": report.get("_harness_seconds"),
            "report_path": report.get("_report_path"),
        }
    )
    return {"summary": summary, "per_instance": per_instance}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pred", type=Path, required=True, help="predictions JSONL for one condition"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--sample", type=Path, default=ROOT / "docs/paper/results/swe_e2e/sample.json"
    )
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument(
        "--harness-timeout",
        type=int,
        default=1800,
        help="per-instance test timeout (s)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    all_ids = json.loads(args.sample.read_text())["instance_ids"]
    report = run_harness(
        args.pred,
        args.run_id,
        max_workers=args.max_workers,
        timeout=args.harness_timeout,
    )
    agg = aggregate(report, all_ids)
    args.out.write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps(agg["summary"], indent=2))


if __name__ == "__main__":
    main()
