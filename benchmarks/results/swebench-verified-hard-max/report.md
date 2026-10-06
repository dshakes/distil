# SWE-bench Verified outcome eval (head-to-head)

Arms: plain, distil, rtk. Summary rows are scored on the **7** instances with a graded outcome in every arm; each comparison below uses the pairs graded in both that arm and plain.

| arm | library | version | resolved | rate (95% Wilson) | vs plain (pts) | 95% CI | McNemar p | verdict | cost $ | steps | tok in | tok out | classes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | - | - | 6/7 | 85.7% (48.7%-97.4%) | - | - | - | baseline | 24.75 | 526 | 59662910 | 954515 | {'ok': 8} |
| distil | distil | 1.57.0 | 6/7 | 85.7% (48.7%-97.4%) | +0.0 | [+0.0, +0.0] | 1.0000 | PILOT (no verdict) | 18.16 | 475 | 46957534 | 658560 | {'ok': 6, 'gave_up': 1} |
| rtk | rtk | 0.51.0 | 7/7 | 100.0% (64.6%-100.0%) | +14.3 | [-11.6, +40.2] | 1.0000 | PILOT (no verdict) | 27.73 | 626 | 71671788 | 994735 | {'ok': 8, 'gave_up': 1} |

## distil vs plain (7 pairs)

Discordant: plain-only 0, distil-only 0 (both 6, neither 1).
Paired difference (distil - plain): +0.0 pts, 95% CI [+0.0, +0.0] (Wald); exact McNemar p = 1.0000.
Decision (non-inferiority margin 5 pts): **PILOT (no verdict)**.

n < 100: the Wald interval is unreliable; treat this as a pilot, not a verdict.

## rtk vs plain (7 pairs)

Discordant: plain-only 0, rtk-only 1 (both 6, neither 0).
Paired difference (rtk - plain): +14.3 pts, 95% CI [-11.6, +40.2] (Wald); exact McNemar p = 1.0000.
Decision (non-inferiority margin 5 pts): **PILOT (no verdict)**.

At the observed discordance (14.3%) ~449 pairs would be needed for 80% power at this margin (true diff 0).
n < 100: the Wald interval is unreliable; treat this as a pilot, not a verdict.

2 comparisons against one baseline: the McNemar p-values and CIs are per-comparison and uncorrected (Bonferroni alpha = 0.0250).

Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.

## $ per solved task

7 tasks graded in every arm. $ per solved = the arm's whole spend on those tasks (failed attempts included) / tasks it solved. Ratio vs plain with a paired cluster bootstrap over tasks (10000 resamples, 97.50% level: Bonferroni over the comparisons).

| arm | solved | total $ | $ per solved | ratio vs plain | CI | cost verdict | median steps |
|---|---|---|---|---|---|---|---|
| plain | 6/7 | 21.63 | 3.6050 | 1 | - | baseline | 63 |
| distil | 6/7 | 18.16 | 3.0267 | 0.840 | [0.389, 1.240] | inconclusive | 67 |
| rtk | 7/7 | 21.54 | 3.0773 | 0.854 | [0.499, 1.148] | inconclusive | 69 |
