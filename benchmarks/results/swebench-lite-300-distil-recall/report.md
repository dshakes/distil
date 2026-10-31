# SWE-bench Lite outcome eval

Paired instances with a graded outcome in both arms: **299**

| arm | resolved | rate (95% Wilson) | cost $ | steps | tok in | tok out | expand calls | classes |
|---|---|---|---|---|---|---|---|---|
| plain | 209/299 | 69.9% (64.5%-74.8%) | 7.45 | 1287 | 5667126 | 337456 | 0 | {'ok': 300} |
| distil | 207/299 | 69.2% (63.8%-74.2%) | 7.58 | 1389 | 6529208 | 337718 | 3 | {'ok': 299, 'gave_up': 1} |

Discordant: plain-only 14, distil-only 12 (both 195, neither 78).
Paired difference (distil - plain): -0.7 pts, 95% CI [-4.0, +2.7] (Wald); exact McNemar p = 0.8450.
Decision (non-inferiority margin 5 pts): **NON-INFERIOR**.

At the observed discordance (8.7%) ~274 pairs would be needed for 80% power at this margin (true diff 0).
Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.

## $ per solved task

299 tasks graded in every arm. $ per solved = the arm's whole spend on those tasks (failed attempts included) / tasks it solved. Ratio vs plain with a paired cluster bootstrap over tasks (10000 resamples, 95.00% level: Bonferroni over the comparisons).

| arm | solved | total $ | $ per solved | ratio vs plain | CI | cost verdict | median steps |
|---|---|---|---|---|---|---|---|
| plain | 209/299 | 7.43 | 0.0356 | 1 | - | baseline | 4 |
| distil | 207/299 | 7.57 | 0.0366 | 1.028 | [0.961, 1.098] | inconclusive | 4 |
