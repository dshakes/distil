# SWE-bench Lite (300): plain vs distil 1.58.0 vs Headroom 0.42.0

`claude-sonnet-5-5`, effort medium, 60 steps, seed 0, run 2026-10-30. Graded locally with
swebench 4.1.0 (5.x fails on Lite with `KeyError: 'image'`). 299 of 300 tasks have a graded
outcome in every arm. Agent spend $23.11 for the three arms. `report.md` is the generated
analysis; every number below is copied from it.

| arm | solved | vs plain | $ per solved | ratio vs plain (97.5% CI) |
|---|---|---|---|---|
| plain | 209/299 | - | 0.0356 | 1 |
| distil 1.58.0 | 208/299 | -0.3 pts, non-inferior | 0.0382 | 1.074 [0.990, 1.163] |
| Headroom 0.42.0 | 211/299 | +0.7 pts, non-inferior | 0.0365 | 1.026 [0.954, 1.101] |

What it shows:

- Neither compressor changes task success: both are non-inferior at a 5-point margin and
  neither differs from plain (McNemar p = 1.00 and 0.83).
- Neither saves money here. Both cost intervals cross 1; both point estimates are above it.
- The median task takes 4 steps. On tasks this short there is little history to compress,
  and the compressed arms took more steps (distil 1436, Headroom 1350, plain 1287) and sent
  more input tokens (7.10M, 6.50M, 5.67M). What compression saved per request, the extra
  steps spent.

The distil arm here is the 1.58.0 release. The visible-recall change that followed it ran on
the same tasks in `../swebench-lite-300-distil-recall/`.

Committed: `results.jsonl` (per-task usage, cost, patch), `grades.jsonl`, `predictions_*.jsonl`,
`report.md`. Harness logs and transcripts (about 940 MB) are not committed.
