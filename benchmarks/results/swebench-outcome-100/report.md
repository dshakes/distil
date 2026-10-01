# SWE-bench Lite outcome eval

Paired instances with a graded outcome in both arms: **100**

| arm | resolved | rate (95% Wilson) | cost $ | steps | tok in | tok out | expand calls | classes |
|---|---|---|---|---|---|---|---|---|
| plain | 69/100 | 69.0% (59.4%-77.2%) | 4.19 | 397 | 1609344 | 97306 | 0 | {'ok': 100} |
| distil | 70/100 | 70.0% (60.4%-78.1%) | 4.54 | 408 | 1779229 | 98082 | 0 | {'ok': 100} |

Discordant: plain-only 4, distil-only 5 (both 65, neither 26).
Paired difference (distil - plain): +1.0 pts, 95% CI [-4.9, +6.9] (Wald); exact McNemar p = 1.0000.
Decision (non-inferiority margin 5 pts): **NON-INFERIOR**.

At the observed discordance (9.0%) ~283 pairs would be needed for 80% power at this margin (true diff 0).
Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.

## Run notes (2026-10-01)

- 100 SWE-bench Lite instances (`--limit 100 --seed 1`) x 2 arms, `claude-sonnet-5-5` @ `low`, `--max-steps 30`, `--budget-usd 25`; spent $8.73. All 200 runs completed (no api/env/internal errors).
- Docker on arm64 (Apple Silicon): images run as `linux/amd64` under emulation. Graded with `swebench>=4,<5`.
- Pre-registered margin: 5 pts (specs/swebench-outcome-eval.md). The lower bound (-4.9) clears it narrowly; ~283 pairs would be needed for 80% power.
- Cost: on these short, low-effort tasks distil sent 10.6% more input tokens (1,779,229 vs 1,609,344) and cost 8% more ($4.54 vs $4.19), with 0 expand calls. Per step the distil arm sent 4,361 input tokens against 4,054 (+307), and it took 408 steps against 397. The run used no prompt caching (0 cache reads), so the injected expand tool's definition (about 150 tokens by a characters/4 estimate) was paid at full price on every step; an agent that caches pays it at the cached-read rate after its first turn. The rest of the per-step gap was not broken down. Later harness versions cache like a real agent. Outcome safety, not savings, is what this run measures.
- Transcripts not committed (size); results.jsonl carries every patch, grades.jsonl the official verdicts, so `report` reproduces this file.
