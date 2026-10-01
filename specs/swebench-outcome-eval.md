# SWE-bench outcome eval: does distil change whether the agent solves the task?

Status: harness implemented and offline-tested; **never run against the API** (as of authoring).
Code: `benchmarks/swebench_outcome/`; tests: `tests/test_swebench_outcome.py`.

## Question
Decision-level evals show distil's digests preserve the model's next decision. They cannot show
the end-to-end effect: does a coding agent behind distil's served transform resolve as many
real tasks as the same agent without it? Metric: SWE-bench Lite resolve rate (official grader).

## Arms (identical except one thing)
- `plain`: messages sent as-is.
- `distil`: before every model call, `distil.adapters.anthropic.compress_messages(messages)` (the
  served transform, default non-verbatim mode), plus a `distil_expand` tool backed by the returned
  `RestoreStore`s (the agent can recover digested content by handle). Nothing else differs:
  same model/effort, system prompt, tools, step budget, task order (seeded), adaptive thinking.
  Arm order alternates per task to cancel time-of-day drift.
Agent: minimal loop on the Anthropic SDK, `bash_20250124` + `text_editor_20250728`, `tool_choice` auto,
client pinned to `https://api.anthropic.com` (ignores a distil-wrapped shell's `ANTHROPIC_BASE_URL`).
Patch = `git add -A; git diff --cached HEAD` in `/testbed` at the end.

## Pre-registered decision rule
Unit of analysis: instances with a graded outcome in **both** arms. `d` = resolve(distil) -
resolve(plain). Non-inferiority margin **X = 5 points**.
- **NON-INFERIOR** if the 95% lower bound of `d` >= -5 pts.
- **INFERIOR** if the 95% upper bound < -5 pts. Otherwise **INCONCLUSIVE** (report, do not claim).
Reported alongside: per-arm resolve rate with Wilson CI, exact McNemar p on discordant pairs,
cost/tokens/steps per arm, distil expand-call count. The CI is a paired Wald interval
(`stats.paired_diff`); it is unreliable below ~100 pairs, and the report says so. The decision
rule, margin, model, effort and step budget are fixed before the first graded run.
Failure classes (`api_error`, `env_error`, `internal_error`, `budget_stopped`) are **excluded** and
listed, never scored as unresolved. `gave_up` (no patch), `timeout` and step-limit-with-patch
are the agent's own outcomes and are graded (empty patch = unresolved).

## Sample size / power
Paired binary data, true `d = 0`: `n = (z_.975 + z_.80)^2 * p_disc / X^2`, `p_disc` = fraction of
pairs where exactly one arm solves (`stats.sample_size_noninferiority`).

| p_disc | X = 5 pts | X = 10 pts |
|---|---|---|
| 0.05 | 157 | 40 |
| 0.10 | 314 | 79 |
| 0.15 | 471 | 118 |
| 0.20 | 628 | 157 |

SWE-bench Lite has only **300** instances, so a 5-pt margin at 80% power needs `p_disc` <= ~0.095:
plausible only if both arms are very consistent. Expect run-to-run agent noise to push `p_disc` to
0.15-0.25. Options: accept the 10-pt margin on one pass, repeat each arm 2-3x (needs a code change:
not implemented), or use SWE-bench Verified (500). Measure `p_disc` in the pilot and re-plan.

## Cost model
Per task-arm cost = sum over steps of `(input_tok * p_in + output_tok * p_out)`; input grows with
context, so cost is super-linear in steps. `plan` uses PLACEHOLDER defaults (30 steps, 35k in,
1.5k out per step) and prints an estimate only. Calibrate from a pilot:
1. Run 5 tasks x 2 arms (`--limit 5`), then `plan --pilot <out>/results.jsonl --limit N`, which
   replaces the placeholder with the pilot's mean cost per task-arm (5 tasks is a rough guide:
   expect +-50%; add a 1.5x safety factor to `--budget-usd`).
2. Price table in `agent.PRICES` (sonnet-5-5 = $2/$10 per MTok, copied from
   `model_migration_eval.py`); override with `--price-in/--price-out`. Cache tokens use the usual
   1.25x/0.1x multipliers; the agent does not itself set `cache_control`, so cost is an upper bound.
`run` hard-stops when live spend (including prior results in the out dir) reaches `--budget-usd`;
overshoot is at most one API call.

## Infra requirements
- Docker; SWE-bench instance images are ~1-3 GB each; budget **100 GB+ disk** for a full Lite pass
  (ASSUMED from swebench docs, not measured here). x86_64 images; Apple Silicon needs emulation
  (slow, occasionally flaky): prefer a Linux x86_64 host.
- `pip install anthropic datasets 'swebench>=4,<5'` (5.x needs an `image` field the princeton-nlp Lite dataset lacks) (optional extra; deliberately NOT in pyproject).
- Network: api.anthropic.com for `run`; HuggingFace for the dataset (`plan` with `--instances
  file.txt` touches no network at all).

## Verified vs assumed
Verified in the authoring environment: anthropic SDK 1.9.0 types (`bash_20250124` name `bash`,
`text_editor_20250728` name `str_replace_based_edit_tool`, adaptive thinking, `output_config.effort`
in low/medium/high/xhigh/max); `compress_messages`/`RestoreStore.expand` signatures and behavior
(tests call the real `compress_messages`); swebench `run_evaluation.py` and `reporting.py`
source on GitHub main (CLI flags `--dataset_name --predictions_path --max_workers --timeout
--run_id`; report keys `resolved_ids/unresolved_ids/error_ids/empty_patch_ids`; report at
`logs/run_evaluation/<run_id>/results.json`; the parser also accepts the older
`<model>.<run_id>.json`); all stats vs hand-computed values; everything in
`tests/test_swebench_outcome.py`.
**Assumed, unverified** (swebench/docker were not installed; no images pulled): image name
`swebench/sweb.eval.x86_64.<id with __ -> _1776_>:latest` (tries `make_test_spec(...).instance_image_key`
first, API path unconfirmed), repo at `/testbed`, conda env `testbed` at `/opt/miniconda3`, images
already pulled (DockerEnv does not pull), the `datasets` Lite schema (`instance_id`,
`problem_statement`), the `distil_expand` tool schema (no canonical definition exists in
`distil/`; the harness defines `{handle: str}`), and the real end-to-end behavior of the agent.

## Pilot and full run
```
pip install anthropic datasets swebench      # in a venv; Docker running
python -m benchmarks.swebench_outcome plan --limit 5 --seed 0
# DockerEnv does not pull: pre-pull each planned instance image (swebench/sweb.eval.x86_64.<id>:latest)
python -m benchmarks.swebench_outcome run --limit 5 --seed 0 --budget-usd 25 \
    --i-understand-this-costs-money --out benchmarks/results/swebench_outcome/pilot
python -m benchmarks.swebench_outcome grade  --out benchmarks/results/swebench_outcome/pilot
python -m benchmarks.swebench_outcome report --out benchmarks/results/swebench_outcome/pilot
python -m benchmarks.swebench_outcome plan --pilot benchmarks/results/swebench_outcome/pilot/results.jsonl
```
Run from a shell that is not distil-wrapped (the harness ignores `ANTHROPIC_BASE_URL` regardless).
`run` is resumable by (instance, arm): re-running the same command skips completed pairs;
`results.jsonl` is append-only and fsynced; a torn last line is dropped and re-run.
Pilot sanity: confirm in `transcripts/distil/*.json` that digests appear and `expand_calls` is
sensible before spending on the full run.

## Threats to validity
- **Contamination**: SWE-bench Lite is public; absolute rates overstate ability. The paired
  difference is less affected, but memorized fixes may make compression look harmless.
- **Scaffold != your agent**: this loop has no caching, no planning tools, no subagents, 20k-char
  tool-output cap (identical in both arms). Real agents differ in how much tool output they carry.
- **Noise**: single attempt per arm; agent stochasticity dominates. See power table.
- **Single model/effort**: a result for sonnet-5-5 at medium says nothing about other models.
- **Compression is applied to the full history each call**: matches the served transform, but the
  harness does no prompt-cache interplay, so cost savings are not a caching-aware estimate.
- **Expand tool is model-discoverable only via the digest markers**: the system prompt is identical
  across arms by design; if distil injects extra instructions in deployment, this under-represents it.
- **Env**: each bash call is a fresh shell (no persistent cwd/env); the patch includes untracked
  files; grading uses the official harness, so eval-time flakiness affects both arms equally.
- **Class exclusion** can bias the paired set if api/env failures correlate with arm (check the
  `classes` column).
- Not implemented: parallel workers, repeated attempts, per-step transcripts in the report.
