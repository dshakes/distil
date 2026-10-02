# SWE-bench Lite outcome eval

Paired instances with a graded outcome in both arms: **299**

| arm | resolved | rate (95% Wilson) | cost $ | steps | tok in | tok out | expand calls | classes |
|---|---|---|---|---|---|---|---|---|
| plain | 210/299 | 70.2% (64.8%-75.1%) | 7.18 | 1254 | 5121601 | 303787 | 0 | {'ok': 300} |
| distil | 204/299 | 68.2% (62.7%-73.2%) | 7.18 | 1293 | 5535181 | 296705 | 5 | {'ok': 299, 'gave_up': 1} |

Discordant: plain-only 17, distil-only 11 (both 193, neither 78).
Paired difference (distil - plain): -2.0 pts, 95% CI [-5.5, +1.5] (Wald); exact McNemar p = 0.3449.
Decision (non-inferiority margin 5 pts): **INCONCLUSIVE**.

At the observed discordance (9.4%) ~295 pairs would be needed for 80% power at this margin (true diff 0).
Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.


## Run notes (2026-10-02)

- Same 300 SWE-bench Lite instances, harness and settings as `../swebench-outcome-300/` (`claude-sonnet-5-5` @ `low`, `--max-steps 30`, prompt caching on both arms), with one change on the distil side: shell search output (`grep`/`rg`/`ag`/`ack`, `git grep`) is kept verbatim like the `Grep` tool's (`distil.compress.provenance.is_shell_search`).
- **Only the distil arm was re-run.** The plain arm's rows (patches, usage, grades) are reused from `../swebench-outcome-300/`; the plain agent never calls distil code, so it is the same control. The re-run distil arm's recorded cost is $7.18 (sum of `cost_usd` in `results.jsonl`).
- Versus the unfixed arm on the same tasks: steps +29.9% → +3.1% over plain, cost +12.4% → ±0% ($7.18 vs $7.18), give-ups 4 → 1, `distil_expand` calls 13 → 5. Resolved 201 → 204 of 299 (−3.0 → −2.0 pts vs plain); that change is within noise.
- Non-inferiority at the pre-registered 5-point margin is still **not shown** (lower bound −5.5). The fix removes the re-search penalty; it does not show that serving keeps coding tasks solved, and on these tasks serving is cost-neutral, not cheaper.
- Offline, on the unfixed run's transcripts, the fix cuts final-request digests from 704 to 275 and request-size savings from 24.1% to 9.1%: the price of keeping search output verbatim.
