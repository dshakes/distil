# SWE-bench Lite (300): plain vs distil with the 2000-token lossy floor

Same model, effort, steps, seed and tasks as `../swebench-lite-300-headroom-042/`
(`claude-sonnet-5-5`, medium, 60 steps, seed 0). The distil arm runs PR #249's floor
(`_DIGEST_MIN_TOKENS = 2000`) on top of 1.58.0; its rows report version 1.58.0 because the change
was not yet released. The prose condenser in the same PR was added after this run started and is
not in it (SWE-bench reads are exact-quote exempt, so it would almost never fire). The `plain`
rows are reused from that run, not re-run. Agent spend $7.58, run 2026-10-31. Graded with swebench
4.1.0. Numbers below are copied from `report.md`, or computed from `results.jsonl` (steps, cost).

| arm | solved | vs plain | steps | cost |
|---|---|---|---|---|
| plain | 209/299 | - | 1287 | $7.45 |
| distil 1.58.0 | 208/299 | non-inferior | 1436 (+11.6%, p = 0.002) | $7.95 |
| **distil + floor** | **206/299 (68.9%, Wilson 63.4% to 73.9%)** | **-1.0 pts [-4.4, +2.4], non-inferior (p = 0.70)** | **1321 (+2.6%, p = 0.59)** | **$7.58 (+1.9%)** |

A second plain run (2026-10-31, $7.46) solved 208 with 1305 steps, so step noise between identical
runs is about ±1.4%. Against 1.58.0 the floor takes fewer steps on 84 tasks and more on 48
(sign test p = 0.002). The cost interval still crosses zero: no cost difference from plain is shown.

Committed: `results.jsonl`, `grades.jsonl`, `predictions_*.jsonl`, `report.md`.
