# SWE-bench Verified, hard tasks, effort max: long-horizon head-to-head (7 tasks)

plain, distil and rtk on the hardest SWE-bench Verified tasks (human estimate "1-4 hours" or
">4 hours"), with claude-sonnet-5-5 at effort `max`, where the agent runs ~50-100 steps per task.
Pre-registered in `docs/research/long-horizon-protocol.md` (amendment A1 fixed effort, arms and
budget before the main run). `report.md` is the harness output; `results.jsonl` and
`grades.jsonl` are the raw rows.

## Result, in one paragraph

On the 7 tasks graded in all three arms, every arm solved 6 or 7. distil's $ per solved task is
0.84x plain's and rtk's 0.85x, but both intervals cross 1 ([0.39, 1.24] and [0.50, 1.15]): **no
cost difference is shown for either**. distil's lower headline comes from one task
(pytest-10356) where its attempt **crashed after 4 steps** (`stop=max_tokens`, $0.37) while
plain spent $5.42 failing the same task. On the other 6 tasks distil cost **more** than plain
($17.79 vs $16.21, +9.7%; dearer on 4 of 6), took more steps (median 75 vs 59.5) and called
`distil_expand` 37 times; its prompt per step was only ~4.5% smaller. **On this run distil does
not save money on long-horizon tasks.** rtk solved one task plain did not; with 7 pairs that
is noise.

## Read this before the numbers

- **7 tasks.** The $100 cap bought 7 complete tasks at about $8.80 per task for the three arms.
  Success comparisons are `PILOT (no verdict)` (the harness gives no verdict below 100 pairs);
  the $-per-solved intervals are wide. This is a measurement at the size the cap buys, not a
  verdict. 45 tasks were eligible; which 7 completed was decided by the seeded order and the
  per-shard budget, not by outcomes.
- **Long horizon, achieved.** Median steps: plain 63, distil 67, rtk 69 (7 tasks); 4 of the 21
  paired attempts hit the 100-step limit. At effort `high` and `xhigh` the same
  agent took a median of 7 and 10 steps on these tasks (set-aside pilots), so only `max` reaches
  the regime this run is about.
- **Two arms dropped by the pre-registered rule.** At `max`, one task cost ~$2 per arm and
  provider-cm $5.03 (its aggressive clearing setting rewrote the cached prefix on 70 of 72 steps).
  Five arms were unaffordable; selective, then provider-cm, were dropped (protocol §5.3, A1).
  Their scoreboard rows say "pending run".
- **The max_tokens ceiling still bites.** Even at 32k max-tokens, adaptive thinking at effort
  `max` sometimes spends the whole response before a tool call: distil on pytest-10356 (4 steps)
  and rtk on pytest-6197 (14 steps, a task that never completed in all arms). The harness records
  that as `gave_up` and the grader as unresolved, as pre-registered; it is a property of
  effort `max`, not of either compressor, and it moves the headline (above).
- **Unpaired rows.** Three attempts ran before a shard's budget stopped (pytest-6197 rtk,
  pylint-8898 rtk, django-14631 plain). They are graded and kept in the raw files but are not in
  any comparison. The `cost $` column of the first table in `report.md` sums every row of an
  arm, including plain's unpaired django-14631 ($3.12); the `$ per solved` table uses only the 7
  common tasks.
- **No infrastructure failures.** No api/env/internal errors, so nothing was re-run.
- Run under emulated x86_64 on an arm64 Mac (7-37 min per attempt); task timeout 3600 s was
  never reached. Three shards ran concurrently on 2026-10-06.
- Prices: claude-sonnet-5-5 at $2 / $10 per million tokens, cache write 1.25x, read 0.1x.
- distil is this branch: 1.57.0 with auto-reinflate (ADR 0025). rtk 0.51.0 pinned binary.

## Spend (harness accounting)

| part | $ |
|---|---|
| pilot P1, effort high, 3 tasks x 5 arms | 0.91 |
| pilot P2, effort max, max-tokens 16k (budget-stopped) | 10.01 |
| pilot P3, effort xhigh, plain only | 0.44 |
| main run, 3 shards x $24 | 72.07 |
| **total** | **83.43** of the $100 cap |

The pilots' rows are not in this directory (set aside, as pre-registered). The $16.57 held for
infrastructure re-runs was not needed and was not spent.

## Reproduce

```
python -m benchmarks.swebench_outcome run --dataset princeton-nlp/SWE-bench_Verified \
  --instances shard_K.txt --seed 0 --arms plain,distil,rtk --model claude-sonnet-5-5 \
  --effort max --max-tokens 32000 --max-steps 100 --task-timeout 3600 --budget-usd 24 \
  --i-understand-this-costs-money --out shard_K
# shard_K.txt: the 45 ids of `plan --dataset princeton-nlp/SWE-bench_Verified
# --difficulty '1-4 hours,>4 hours' --seed 0`, line i -> shard i mod 3; results.jsonl is the
# three shards' results concatenated in shard order.
python -m benchmarks.swebench_outcome grade --dataset princeton-nlp/SWE-bench_Verified --arms plain,distil,rtk --out DIR
python -m benchmarks.swebench_outcome report --dataset princeton-nlp/SWE-bench_Verified --arms plain,distil,rtk --cost-per-solved --out DIR
```

## What is committed

`results.jsonl`, `grades.jsonl`, `predictions_*.jsonl`, `report.md`, this README. Not
committed: `transcripts/`, `logs/` and the `swo-*.json` grader working files.
