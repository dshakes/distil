# SWE-bench Lite outcome eval (head-to-head)

Arms: plain, distil, rtk, selective, provider-cm. Summary rows are scored on the **298** instances with a graded outcome in every arm; each comparison below uses the pairs graded in both that arm and plain.
Arm `distil` was **reused** from `benchmarks/results/swebench-outcome-300-medium` (300 rows copied, grades carried over): not re-run in this directory.
Arm `plain` was **reused** from `benchmarks/results/swebench-outcome-300-medium` (300 rows copied, grades carried over): not re-run in this directory.

| arm | library | version | resolved | rate (95% Wilson) | vs plain (pts) | 95% CI | McNemar p | verdict | cost $ | steps | tok in | tok out | classes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | - | - | 218/298 | 73.2% (67.9%-77.9%) | - | - | - | baseline | 8.52 | 1374 | 6891062 | 361161 | {'ok': 300} |
| distil | - | - | 213/298 | 71.5% (66.1%-76.3%) | -1.7 | [-5.1, +1.7] | 0.4421 | INCONCLUSIVE | 8.44 | 1444 | 7174289 | 354539 | {'ok': 299, 'internal_error': 1} |
| rtk | rtk | 0.51.0 | 219/298 | 73.5% (68.2%-78.2%) | +0.3 | [-2.7, +3.3] | 1.0000 | NON-INFERIOR | 6.79 | 1326 | 5933865 | 344314 | {'ok': 300} |
| selective | selective-context | 0.1.4 | 210/298 | 70.5% (65.1%-75.4%) | -2.7 | [-6.7, +1.4] | 0.2559 | INCONCLUSIVE | 10.03 | 2081 | 11034533 | 502657 | {'ok': 300} |
| provider-cm | anthropic | 1.11.0 | 210/298 | 70.5% (65.1%-75.4%) | -2.7 | [-5.6, +0.2] | 0.1153 | INCONCLUSIVE | 8.26 | 1438 | 5745703 | 345668 | {'ok': 299, 'gave_up': 1} |

## distil vs plain (298 pairs)

Discordant: plain-only 16, distil-only 11 (both 202, neither 69).
Paired difference (distil - plain): -1.7 pts, 95% CI [-5.1, +1.7] (Wald); exact McNemar p = 0.4421.
Decision (non-inferiority margin 5 pts): **INCONCLUSIVE**.

At the observed discordance (9.1%) ~285 pairs would be needed for 80% power at this margin (true diff 0).

## rtk vs plain (299 pairs)

Discordant: plain-only 10, rtk-only 11 (both 208, neither 70).
Paired difference (rtk - plain): +0.3 pts, 95% CI [-2.7, +3.3] (Wald); exact McNemar p = 1.0000.
Decision (non-inferiority margin 5 pts): **NON-INFERIOR**.

At the observed discordance (7.0%) ~221 pairs would be needed for 80% power at this margin (true diff 0).

## selective vs plain (299 pairs)

Discordant: plain-only 23, selective-only 15 (both 195, neither 66).
Paired difference (selective - plain): -2.7 pts, 95% CI [-6.7, +1.4] (Wald); exact McNemar p = 0.2559.
Decision (non-inferiority margin 5 pts): **INCONCLUSIVE**.

At the observed discordance (12.7%) ~400 pairs would be needed for 80% power at this margin (true diff 0).

## provider-cm vs plain (299 pairs)

Discordant: plain-only 14, provider-cm-only 6 (both 204, neither 75).
Paired difference (provider-cm - plain): -2.7 pts, 95% CI [-5.6, +0.2] (Wald); exact McNemar p = 0.1153.
Decision (non-inferiority margin 5 pts): **INCONCLUSIVE**.

At the observed discordance (6.7%) ~211 pairs would be needed for 80% power at this margin (true diff 0).

4 comparisons against one baseline: the McNemar p-values and CIs are per-comparison and uncorrected (Bonferroni alpha = 0.0125).

Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.

## Cost confound

These arms read a prompt cache they did not write (cache_write + input < cache_read / (steps - 1), impossible for a cold run): provider-cm 139/300, rtk 190/300, selective 167/300. Their cost columns are not their own cold cost; success rates are unaffected. New runs give each arm its own cache namespace (on by default, `--no-cache-namespace` turns it off).
