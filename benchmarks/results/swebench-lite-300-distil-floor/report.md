# SWE-bench Lite outcome eval

Paired instances with a graded outcome in both arms: **299**

| arm | resolved | rate (95% Wilson) | cost $ | steps | tok in | tok out | expand calls | classes |
|---|---|---|---|---|---|---|---|---|
| plain | 209/299 | 69.9% (64.5%-74.8%) | 7.45 | 1287 | 5667126 | 337456 | 0 | {'ok': 300} |
| distil | 206/299 | 68.9% (63.4%-73.9%) | 7.58 | 1321 | 6636650 | 332122 | 0 | {'ok': 299, 'gave_up': 1} |

Discordant: plain-only 15, distil-only 12 (both 194, neither 78).
Paired difference (distil - plain): -1.0 pts, 95% CI [-4.4, +2.4] (Wald); exact McNemar p = 0.7011.
Decision (non-inferiority margin 5 pts): **NON-INFERIOR**.

At the observed discordance (9.0%) ~284 pairs would be needed for 80% power at this margin (true diff 0).
Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.
