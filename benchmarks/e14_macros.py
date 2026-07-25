#!/usr/bin/env python3
"""Emit generated/e14_macros.tex from the E14 harness report + committed E8 data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.swe_bench_e2e.stats import pass_at_1, noninferiority_paired  # noqa: E402
from distil.certify.trajectory_risk import (  # noqa: E402
    TrajectoryOutcome,
    certify_trajectory_risk,
)

ROOT = Path(__file__).resolve().parents[1]
LH = ROOT / "docs/paper/results/swe_e2e_longhorizon"


def main(report_path: str) -> None:
    rep = json.loads(Path(report_path).read_text())
    sur = set(rep["resolved_ids"])
    full = set(json.loads((LH / "reports/distil-lh-full.lh_full.json").read_text())["resolved_ids"])
    ids = sorted(
        json.loads((LH / "trajectory_bound_inputs.json").read_text())["conditions"]["full"]
    )
    n = len(ids)
    k = sum(1 for i in ids if i in sur)
    p = pass_at_1(k, n)
    preds = [
        json.loads(line)
        for line in (LH / "predictions/distil_gated_surprise.jsonl").read_text().splitlines()
        if line.strip()
    ]
    nonempty = sum(1 for r in preds if r.get("model_patch", "").strip())
    b = sum(1 for i in ids if i in full and i not in sur)
    c = sum(1 for i in ids if i not in full and i in sur)
    ni = noninferiority_paired(b=b, c=c, n=n, margin=0.05)
    cert = certify_trajectory_risk(
        [TrajectoryOutcome(i, i in full, i in sur) for i in ids], alpha=0.10, delta=0.05
    )
    gated_rep = json.loads((LH / "reports/distil-lh-distil_gated.lh_distil_gated.json").read_text())
    gated_patch = gated_rep["completed_instances"] / gated_rep["total_instances"]

    out = "\n".join(
        [
            f"\\newcommand{{\\sweEfourteenPass}}{{{k / n * 100:.1f}\\%}}",
            f"\\newcommand{{\\sweEfourteenResolved}}{{{k}}}",
            f"\\newcommand{{\\sweEfourteenCIlo}}{{{p['ci95_low'] * 100:.1f}\\%}}",
            f"\\newcommand{{\\sweEfourteenCIhi}}{{{p['ci95_high'] * 100:.1f}\\%}}",
            f"\\newcommand{{\\sweEfourteenPatchRate}}{{{nonempty / n * 100:.1f}\\%}}",
            f"\\newcommand{{\\sweEfourteenB}}{{{b}}}",
            f"\\newcommand{{\\sweEfourteenC}}{{{c}}}",
            f"\\newcommand{{\\sweEfourteenDelta}}{{{ni['delta'] * 100:+.1f}pp}}",
            f"\\newcommand{{\\sweEfourteenNI}}{{{'yes' if ni['noninferior'] else 'no'}}}",
            f"\\newcommand{{\\sweEfourteenTrajBound}}{{{cert.risk_bound * 100:.1f}\\%}}",
            f"\\newcommand{{\\sweEeightGatedPatchRate}}{{{gated_patch * 100:.1f}\\%}}",
        ]
    )
    dest = ROOT / "docs/paper/generated/e14_macros.tex"
    dest.write_text(out + "\n")
    print(out)
    print(f"-> {dest}")


if __name__ == "__main__":
    main(sys.argv[1])
