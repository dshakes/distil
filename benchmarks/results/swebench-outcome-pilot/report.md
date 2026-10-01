# SWE-bench Lite outcome eval

Paired instances with a graded outcome in both arms: **10**

| arm | resolved | rate (95% Wilson) | cost $ | steps | tok in | tok out | expand calls | classes |
|---|---|---|---|---|---|---|---|---|
| plain | 8/10 | 80.0% (49.0%-94.3%) | 0.32 | 37 | 114195 | 9623 | 0 | {'ok': 10} |
| distil | 8/10 | 80.0% (49.0%-94.3%) | 0.36 | 38 | 132934 | 9130 | 0 | {'ok': 10} |

Discordant: plain-only 0, distil-only 0 (both 8, neither 2).
Paired difference (distil - plain): +0.0 pts, 95% CI [+0.0, +0.0] (Wald); exact McNemar p = 1.0000.
Decision (non-inferiority margin 5 pts): **PILOT (no verdict)**.

n < 100: the Wald interval is unreliable; treat this as a pilot, not a verdict.
Excluded (api_error/env_error/internal_error/budget) instances are listed in `classes`, never counted as unresolved.

## Run notes (2026-09-30)

- 10 SWE-bench Lite instances (`--limit 10 --seed 1`) x 2 arms, `claude-sonnet-5-5` @ `low`, `--max-steps 30`, `--budget-usd 20`; spent $0.68.
- Docker on arm64 (Apple Silicon): images run as `linux/amd64` under emulation.
- Graded with `swebench>=4,<5` — swebench 5.x expects an `image` field the `princeton-nlp/SWE-bench_Lite` dataset does not carry (`KeyError: 'image'`).
- Short tasks: few tool outputs were long enough to digest, so distil sent *more* input tokens here (132,934 vs 114,195, +16%; the run used no prompt caching, so the injected expand-tool definition was paid in full each step, plus one extra step; see the 100-task report for the per-step breakdown). Savings appear on long-horizon traces (see docs/model-migration.html); this pilot speaks to outcome safety, not cost.
- Transcripts not committed (size); per-run records in results.jsonl.
