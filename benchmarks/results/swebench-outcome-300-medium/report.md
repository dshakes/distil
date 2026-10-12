# SWE-bench Lite outcome eval

Paired instances with a graded outcome in both arms: **298**

| arm | resolved | rate (95% Wilson) | cost $ | steps | tok in | tok out | expand calls | classes |
|---|---|---|---|---|---|---|---|---|
| plain | 218/298 | 73.2% (67.9%-77.9%) | 8.52 | 1374 | 6891062 | 361161 | 0 | {'ok': 300} |
| distil | 213/298 | 71.5% (66.1%-76.3%) | 8.44 | 1444 | 7174289 | 354539 | 4 | {'ok': 299, 'internal_error': 1} |

Discordant: plain-only 16, distil-only 11 (both 202, neither 69).
Paired difference (distil - plain): -1.7 pts, 95% CI [-5.1, +1.7] (Wald); exact McNemar p = 0.4421.
Decision (non-inferiority margin 5 pts): **INCONCLUSIVE**.

At the observed discordance (9.1%) ~285 pairs would be needed for 80% power at this margin (true diff 0).
Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.


## Run notes (2026-10-03)

- All 300 SWE-bench Lite instances x 2 arms with the shell-search fix (1.56.3), `claude-sonnet-5-5` at effort **medium** and `--max-steps 60` (vs low / 30 in `../swebench-outcome-300-grepfix/`), prompt caching on both arms, three batches of 100 with images pre-pulled; spent $16.97 (sum of `cost_usd`).
- Intended as a long-horizon check, it was not one: the median task took 4 steps (90th percentile 7–8) in both arms, so contexts stayed short and the savings question for long sessions is still unmeasured.
- Outcome: −1.7 pts (95% CI −5.1 to +1.7, McNemar p = 0.44), non-inferiority at the 5-point margin not shown — the second independent run at about −2 pts, consistent with the low-effort fix run (−2.0, CI −5.5 to +1.5). Cost: distil $8.44 vs plain $8.52 (parity); steps 1,444 vs 1,374.
- One distil run (`sympy__sympy-23191`) ended as `internal_error`: the sandbox decoded non-UTF-8 test output strictly. Excluded, never counted as a loss; the harness now decodes with `errors="replace"`.
- Transcripts not committed (size).
