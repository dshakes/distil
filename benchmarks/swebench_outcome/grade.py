"""Predictions JSONL -> official swebench.harness.run_evaluation -> per-instance table."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .run import DATASET, read_manifest, read_results

# Classes whose patch is still graded (the agent ran and produced what it produced).
GRADED_CLASSES = (None, "gave_up", "timeout")


def write_predictions(results: list[dict[str, Any]], out: Path) -> dict[str, Path]:
    paths = {}
    for arm in sorted({r["arm"] for r in results}):
        p = out / f"predictions_{arm}.jsonl"
        rows = [r for r in results if r["arm"] == arm and r.get("failure_class") in GRADED_CLASSES]
        p.write_text(
            "".join(
                json.dumps(
                    {
                        "instance_id": r["instance_id"],
                        "model_name_or_path": f"swo-{arm}",
                        "model_patch": r.get("patch") or "",
                    }
                )
                + "\n"
                for r in rows
            )
        )
        paths[arm] = p
    return paths


def parse_report(rep: dict[str, Any]) -> dict[str, str]:
    """swebench report -> {instance_id: resolved|unresolved|grader_error}. Keys verified against
    swebench/harness/reporting.py: resolved_ids, unresolved_ids, error_ids, empty_patch_ids."""
    out = {i: "resolved" for i in rep.get("resolved_ids", [])}
    out.update({i: "unresolved" for i in rep.get("unresolved_ids", [])})
    out.update({i: "unresolved" for i in rep.get("empty_patch_ids", [])})
    out.update({i: "grader_error" for i in rep.get("error_ids", [])})
    return out


def find_report(out: Path, run_id: str, model_name: str) -> Path:
    """Current swebench writes logs/run_evaluation/<run_id>/results.json; older versions wrote
    <model>.<run_id>.json in cwd. Accept both."""
    for p in (
        out / "logs" / "run_evaluation" / run_id / "results.json",
        out / f"{model_name}.{run_id}.json",
    ):
        if p.exists():
            return p
    raise FileNotFoundError(f"no swebench report for run_id={run_id} under {out}")


def carry_over(
    out: Path, results: list[dict[str, Any]], arms: tuple[str, ...] | None
) -> list[dict[str, str]]:
    """Grades for reused arms come from the run they were copied from (same patches, same
    official grader); re-grading hundreds of identical patches would only add flake."""
    rows: list[dict[str, str]] = []
    for arm, info in read_manifest(out).items():
        gp = Path(info["from"]) / "grades.jsonl"
        if (arms and arm not in arms) or not gp.exists():
            continue
        mine = {r["instance_id"] for r in results if r["arm"] == arm}
        for ln in gp.read_text().splitlines():
            g = json.loads(ln)
            if g["arm"] == arm and g["instance_id"] in mine:
                rows.append(g)
    return rows


def grade(
    out: Path,
    max_workers: int = 4,
    dataset: str = DATASET,
    timeout: int = 1800,
    run: Any = subprocess.run,
    arms: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    try:
        import swebench  # type: ignore  # noqa: F401
    except ImportError as e:
        raise SystemExit("swebench is not installed: pip install swebench (and Docker)") from e
    if not shutil.which("docker"):
        raise SystemExit("docker CLI not found; grading needs Docker")
    rows: list[dict[str, str]] = []
    results = read_results(out / "results.jsonl")
    carried = carry_over(out, results, arms)
    rows += carried
    skip = {r["arm"] for r in carried}
    for arm, pred in write_predictions(results, out).items():
        if arm in skip or (arms and arm not in arms) or not pred.read_text().strip():
            continue
        run_id = f"swo-{arm}"
        run(
            [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                dataset,
                "--predictions_path",
                str(pred),
                "--max_workers",
                str(max_workers),
                "--timeout",
                str(timeout),
                "--run_id",
                run_id,
            ],
            cwd=out,
            check=True,
        )
        rep = json.loads(find_report(out, run_id, f"swo-{arm}").read_text())
        rows += [{"instance_id": i, "arm": arm, "status": s} for i, s in parse_report(rep).items()]
    with open(out / "grades.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return rows
