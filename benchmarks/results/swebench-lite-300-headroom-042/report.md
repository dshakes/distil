# SWE-bench Lite outcome eval (head-to-head)

Arms: plain, distil, headroom. Summary rows are scored on the **299** instances with a graded outcome in every arm; each comparison below uses the pairs graded in both that arm and plain.

| arm | library | version | resolved | rate (95% Wilson) | vs plain (pts) | 95% CI | McNemar p | verdict | cost $ | steps | tok in | tok out | classes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | - | - | 209/299 | 69.9% (64.5%-74.8%) | - | - | - | baseline | 7.45 | 1287 | 5667126 | 337456 | {'ok': 300} |
| distil | distil | 1.58.0 | 208/299 | 69.6% (64.1%-74.5%) | -0.3 | [-3.5, +2.8] | 1.0000 | NON-INFERIOR | 7.95 | 1436 | 7098825 | 358900 | {'ok': 300} |
| headroom | headroom-ai | 0.42.0 | 211/299 | 70.6% (65.2%-75.4%) | +0.7 | [-2.4, +3.7] | 0.8318 | NON-INFERIOR | 7.71 | 1350 | 6501980 | 343059 | {'ok': 300} |

## distil vs plain (299 pairs)

Discordant: plain-only 12, distil-only 11 (both 197, neither 79).
Paired difference (distil - plain): -0.3 pts, 95% CI [-3.5, +2.8] (Wald); exact McNemar p = 1.0000.
Decision (non-inferiority margin 5 pts): **NON-INFERIOR**.

At the observed discordance (7.7%) ~242 pairs would be needed for 80% power at this margin (true diff 0).

## headroom vs plain (299 pairs)

Discordant: plain-only 10, headroom-only 12 (both 199, neither 78).
Paired difference (headroom - plain): +0.7 pts, 95% CI [-2.4, +3.7] (Wald); exact McNemar p = 0.8318.
Decision (non-inferiority margin 5 pts): **NON-INFERIOR**.

At the observed discordance (7.4%) ~232 pairs would be needed for 80% power at this margin (true diff 0).

2 comparisons against one baseline: the McNemar p-values and CIs are per-comparison and uncorrected (Bonferroni alpha = 0.0250).

Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.

## $ per solved task

299 tasks graded in every arm. $ per solved = the arm's whole spend on those tasks (failed attempts included) / tasks it solved. Ratio vs plain with a paired cluster bootstrap over tasks (10000 resamples, 97.50% level: Bonferroni over the comparisons).

| arm | solved | total $ | $ per solved | ratio vs plain | CI | cost verdict | median steps |
|---|---|---|---|---|---|---|---|
| plain | 209/299 | 7.43 | 0.0356 | 1 | - | baseline | 4 |
| distil | 208/299 | 7.94 | 0.0382 | 1.074 | [0.990, 1.163] | inconclusive | 4 |
| headroom | 211/299 | 7.69 | 0.0365 | 1.026 | [0.954, 1.101] | inconclusive | 4 |
