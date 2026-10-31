# SWE-bench Lite (300): plain vs distil with the visible-recall change

Same model, effort, steps, seed and tasks as `../swebench-lite-300-headroom-042/`
(`claude-sonnet-5-5`, medium, 60 steps, seed 0, 2026-10-30). The distil arm runs the
visible-recall change (identifier lines kept in view; XML/JSON envelopes no longer stripped as
HTML chrome) on top of 1.58.0; its rows report version 1.58.0 because the change was not yet
released. The `plain` rows are reused from that run, not re-run. Agent spend $7.58. Graded with
swebench 4.1.0. Numbers below are copied from `report.md`.

| arm | solved | vs plain | $ per solved | ratio vs plain (95% CI) |
|---|---|---|---|---|
| plain | 209/299 | - | 0.0356 | 1 |
| distil + visible recall | 207/299 | -0.7 pts, non-inferior (p = 0.85) | 0.0366 | 1.028 [0.961, 1.098] |

The change cut distil's cost overhead on these tasks from +7.4% (1.58.0) to +2.8%, but the
interval still crosses 1: no cost difference is shown in either direction. The arm took 1389
steps against plain's 1287 and sent 6.53M input tokens against 5.67M.

Committed: `results.jsonl`, `grades.jsonl`, `predictions_*.jsonl`, `report.md`.
