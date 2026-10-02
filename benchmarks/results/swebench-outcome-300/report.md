# SWE-bench Lite outcome eval

Paired instances with a graded outcome in both arms: **299**

| arm | resolved | rate (95% Wilson) | cost $ | steps | tok in | tok out | expand calls | classes |
|---|---|---|---|---|---|---|---|---|
| plain | 210/299 | 70.2% (64.8%-75.1%) | 7.18 | 1254 | 5121601 | 303787 | 0 | {'ok': 300} |
| distil | 201/299 | 67.2% (61.7%-72.3%) | 8.07 | 1629 | 7234387 | 343280 | 13 | {'ok': 296, 'gave_up': 4} |

Discordant: plain-only 18, distil-only 9 (both 192, neither 80).
Paired difference (distil - plain): -3.0 pts, 95% CI [-6.4, +0.4] (Wald); exact McNemar p = 0.1221.
Decision (non-inferiority margin 5 pts): **INCONCLUSIVE**.

At the observed discordance (9.0%) ~284 pairs would be needed for 80% power at this margin (true diff 0).
Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.


## Run notes (2026-10-02)

- All 300 SWE-bench Lite instances x 2 arms, `claude-sonnet-5-5` @ `low`, `--max-steps 30`, run in three batches of 100 (`--instances`, dataset order), $15 cap per batch; spent $15.25. 299 pairs have a graded outcome in both arms. 4 distil runs gave up (no patch); no api/env/internal errors in the final rows.
- **This run supersedes the 100-task run** (`../swebench-outcome-100/`), whose narrow non-inferiority does not hold at full size: here distil-served is −3.0 pts (95% CI −6.4 to +0.4, McNemar p = 0.12), so non-inferiority at the 5-point margin is **not shown**. The discordant pairs lean against distil (18 plain-only vs 9 distil-only).
- Both arms use prompt caching (a breakpoint on the newest turn, before compression). Even so, distil cost 12.4% more ($8.07 vs $7.18) and took 29.9% more steps (1,629 vs 1,254).
- Mechanism (`tool_calls.json`): distil made 1,583 tool calls vs 1,184. The extra are mostly the agent re-reading files in slices — `sed` 624 vs 406, `grep` 539 vs 410 — while it called `distil_expand` only 13 times. Reading: when the served path digests an earlier file view, the agent re-reads instead of expanding. distil's exact-quote exemption keys on tool names (e.g. `goto`, `scroll_up`), so file views made through generic `bash` are not exempt. A fix is being measured; until then, this run is the current result.
- Docker on arm64 runs the images as `linux/amd64`; graded with `swebench>=4,<5`. Transcripts not committed (size).
