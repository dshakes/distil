# Pre-registered protocol: long-horizon SWE-bench head-to-head

*Registered 2026-10-05, before any paid call of this run. Amendments are appended at the end,
dated, and made only before the main run starts (except the infra re-run rule, §7). No outcome
number appears above the amendments section.*

Harness: `benchmarks/swebench_outcome/` (`python -m benchmarks.swebench_outcome`). Spend
authority: the maintainer approved "a long-horizon run: about 100 tasks, at least 40 steps,
all arms, about $60–100". **Hard cap: $100 total** for everything in this run (pilot, main run,
re-runs, every arm), enforced by the harness's own `--budget-usd` accounting.

## 0. Conflict of interest

Run by the authors of distil, one of the five arms. Every choice below was fixed before any
result of this run existed. The task set is chosen from a human annotation, not from any arm's
outcome. Raw rows are committed.

## 1. Question

On SWE-bench Lite the median task takes ~4 agent steps, so little context accumulates for any
compressor to remove (`benchmarks/results/swebench-outcome-300-h2h`). distil is designed for
long sessions with large tool outputs. Here: on the hardest SWE-bench tasks we can grade, what
is the **$ per solved task** and the **success rate** of plain, distil, rtk, provider-cm and
selective?

## 2. Task set

- Dataset: `princeton-nlp/SWE-bench_Verified`, split `test` (500 tasks; graded by the official
  `swebench>=4,<5` harness, prebuilt `swebench/sweb.eval.x86_64.*` images).
- Filter: the dataset's `difficulty` field (verified by loading the dataset 2026-10-05; values
  `<15 min fix` 194, `15 min - 1 hour` 261, `1-4 hours` 42, `>4 hours` 3) in
  **{`1-4 hours`, `>4 hours`}: 45 tasks**. This is the human annotators' time estimate; it does
  not look at any arm's outcome.
  `--dataset princeton-nlp/SWE-bench_Verified --difficulty '1-4 hours,>4 hours'`.
- Order: the harness's seeded shuffle, `--seed 0`. The order decides which tasks run first if the
  budget stops the run early.
- 45 is below the harness's 100-pair threshold for a non-inferiority verdict, so the success
  comparison will be reported as `PILOT (no verdict)` by the harness whatever happens. That is
  accepted: there are only 45 tasks in the hard buckets. No top-up from easier buckets: it would
  dilute the "long-horizon" question this run exists to answer.

## 3. Arms and configuration

| setting | value |
|---|---|
| arms | plain (control), distil, rtk, provider-cm, selective — exactly as in the Lite head-to-head (`specs/swebench-outcome-eval.md`): rtk v0.51.0 pinned binary, selective-context 0.1.4 defaults, provider-cm at the harness setting (trigger 3000, keep 2, clear at least 1000), distil = this branch (1.57.0 + auto-reinflate) |
| model | `claude-sonnet-5-5`, $2 / $10 per MTok, cache write 1.25x, read 0.1x |
| effort | `high` |
| max steps | 100 |
| task timeout | 3600 s |
| seeds | one attempt per task per arm (seed 0) |
| arm order | the harness rotates arm order per task |

## 4. Metrics

- **Primary: $ per solved task per arm**, = the arm's whole spend on the tasks graded in every
  arm (failed attempts counted) / tasks it solved; reported as the ratio vs plain with a paired
  cluster bootstrap over tasks (10,000 resamples, seed 0, Bonferroni over the comparisons), the
  estimand and code of `benchmarks/cost_truth/analysis.py`, via
  `report --cost-per-solved`. Cost verdict: "cheaper" iff the whole interval is < 1, "dearer"
  iff > 1, else "inconclusive".
- **Secondary: success rate**, paired vs plain: the harness's exact McNemar, Wald CI, NI margin
  5 pts (verdicts as the harness prints them, including `PILOT (no verdict)`).
- **Descriptive:** median and p90 steps per arm, tokens in/out, wall time.

## 5. Calibration and pilot (before the main run)

1. `plan` ($0) to fix the task list.
2. **Paid pilot:** the first 3 tasks of the seeded order × all 5 arms, same configuration,
   `--budget-usd 12`, written into the main run directory. If the configuration is unchanged
   after the pilot, its rows **are** the main run's first 3 tasks (they would be run identically
   anyway); if any setting changes, they are set aside (backup outside the repo) and not counted.
3. From the pilot's per-arm mean $/task (`plan --calibrate`), choose arms and N so that the
   projected total ≤ $85 (headroom under $100):
   - all 5 arms × 45 tasks if it fits;
   - else drop **selective** first, then **provider-cm**, keeping plain, distil, rtk;
   - else reduce N to a prefix of the seeded order.
   The choice and the numbers are recorded as a dated amendment before the main run.
4. **Horizon check:** median steps across the pilot's runs should be ≥ ~20. If it is not at
   effort `high` / max-steps 100, there is no harder graded SWE-bench bucket to move to; the run
   proceeds and is labelled with its measured median steps rather than called "long-horizon".

## 6. Running

The main run may be split into shards (disjoint slices of the seeded task list, separate output
directories, merged by concatenating `results.jsonl` before grading) to cut wall time. The sum of
the shards' `--budget-usd` never exceeds the remaining cap.

## 7. Exclusions, re-runs and stopping

- `api_error`, `env_error`, `internal_error`, `budget_stopped` rows are infrastructure, never
  scored as unresolved. They are set aside (copied outside the repo, removed from
  `results.jsonl`) and **re-run once** within the same cap. A row still failing after its re-run
  is excluded and listed.
- `gave_up`, `timeout` and step-limit rows are graded (the agent's own outcome).
- A task enters the analysis only if it is graded in every arm analysed.
- Stopping rule: the harness stops each shard when its budget is spent; a partially run task is
  not recorded. The analysis is on whatever tasks completed in every arm. No task is ever
  re-run because of its outcome, and no paid run happens outside this protocol.

## Amendments

(none yet)
