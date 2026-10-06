# SWE-bench outcome eval: does distil change whether the agent solves the task?

Status: harness implemented and offline-tested; powered runs of `plain` vs `distil` are committed
under `benchmarks/results/swebench-outcome-*`. The head-to-head arms below (`rtk`, `selective`,
`provider-cm`) are implemented and offline-tested; **no paid run of them exists yet**.
Code: `benchmarks/swebench_outcome/`; tests: `tests/test_swebench_outcome.py`,
`tests/test_swebench_arms.py`.

## Question
Decision-level evals show distil's digests preserve the model's next decision. They cannot show
the end-to-end effect: does a coding agent behind distil's served transform resolve as many
real tasks as the same agent without it? Metric: SWE-bench Lite resolve rate (official grader).

## Arms (identical except one thing)
An arm (`arms.Arm`) is a name plus independent hooks: a request `transform` (messages ->
messages[, store]), extra `tools` + their `handlers`, extra request `params` / `betas`, a
`wrap_env` at the tool-output boundary, an `on_response` telemetry hook, and `meta` (library +
version, copied into every result row as `arm_meta`; per-run telemetry goes in `arm_stats`).
The agent loop never branches on an arm's name. Select arms with `--arms a,b,c` on
`plan`/`run`/`grade`/`report` (default `plain,distil`).

- `plain`: no hooks; the baseline request, only a cache breakpoint on the newest block.
- `distil`: before every model call, `distil.adapters.anthropic.compress_messages(messages)` (the
  served transform, default non-verbatim mode), plus a `distil_expand` tool backed by the returned
  `RestoreStore`s (the agent can recover digested content by handle).
- `rtk`, `selective`, `provider-cm`: competitor arms, each calling the real thing at its own
  layer; pinned versions and sources in the table below.

Nothing else differs between arms: same model/effort, system prompt, bash/editor tools, step
budget, task order (seeded), adaptive thinking. The arm order rotates per task (two arms: swaps)
to cancel time-of-day drift.

### Head-to-head arms (pinned)

| arm | what it is (upstream) | where it acts | pin | license |
|---|---|---|---|---|
| `rtk` | [rtk-ai/rtk](https://github.com/rtk-ai/rtk), Rust CLI proxy that rewrites dev commands to filtered `rtk <cmd>` equivalents | bash tool boundary: each command goes through `rtk rewrite "<cmd>"` exactly as the documented Claude Code hook does, inside the task container, and the rewritten command's output is what the model sees | **v0.51.0**, `rtk-x86_64-unknown-linux-musl.tar.gz` sha256 `5028d3b1...eb5` (release `checksums.txt`), copied into each container as `/usr/local/bin/rtk` and version-checked | Apache-2.0 |
| `selective` | [Selective Context](https://github.com/liyucheng09/Selective_Context) (Li et al., EMNLP 2023), prunes low-self-information lexical units using GPT-2 | request transform on tool-result text only, memoised per distinct result | `selective-context==0.1.4` (PyPI), `SelectiveContext("gpt2", "en")`, upstream defaults `reduce_ratio=0.35`, `reduce_level="phrase"` | MIT |
| `provider-cm` | Anthropic API context editing, strategy `clear_tool_uses_20250919` ([docs](https://platform.claude.com/docs/en/build-with-claude/context-editing)) | server side: `client.beta.messages.create(betas=["context-management-2025-06-27"], context_management={"edits": [...]})`; response `context_management.applied_edits` is recorded | beta header `context-management-2025-06-27`; SDK version recorded per row (`anthropic` 0.111.0 verified to expose `betas` and `context_management`) | n/a (API feature) |

**rtk** is a command wrapper, not a text compressor, so it is modelled at the tool boundary and
nowhere else. Protocol (from `hooks/claude/rtk-rewrite.sh` v4, verified against the v0.51.0
binary): `rtk rewrite` exits 0 (allow rule) or 3 (ask / no rule, the default config) with the
rewritten command on stdout, which is then run; exit 1 (no RTK equivalent) and 2 (deny rule) run
the command unchanged; any other exit (binary missing or crashed) is an `env_error`, never a
silent pass-through. `git status` -> `rtk git status`, `cat f` -> `rtk read f`,
`python -m pytest ..` -> `rtk pytest ..`, `grep -rn x .` -> `rtk grep -rn x .`; `echo`, `sed` stay
as they are. Like the real hook, only bash is rewritten (the editor tool, like Claude Code's
Read/Grep/Glob, is not) and the system prompt is unchanged (RTK.md is silent by default).
`--rtk-bin` supplies a binary instead of the pinned download. `arm_stats` records
`commands` / `rewritten` per task.

**selective** needs a local LM and a Python the harness cannot share: `selective-context==0.1.4`
pins `spacy==3.2.0` (wheels only up to CPython 3.10) and `click==8.0.4`. It therefore runs
out-of-process (`selective_worker.py`, JSON lines) under `--selective-python`, a venv with
`pip install selective-context==0.1.4 'numpy<2' && python -m spacy download en_core_web_sm` (`numpy<2`: spaCy 3.2's compiled extensions fail against numpy 2 with "numpy.dtype size changed"). Downloads:
GPT-2 124M from the Hugging Face hub on first use (`openai-community/gpt2`, MIT,
`model.safetensors` 548 MB; the repo's tokenizer files add ~1.5 MB), spaCy `en_core_web_sm`, and
torch (hundreds of MB, not measured here). CPU is enough. If any of that is missing or the
pinned version differs, `run` prints `refusing: selective: ...` with the install line and exits 2
**before any spend**; the arm is simply left out of the `--arms` list. Forced adaptations, all
recorded in `arm_meta`: results under 200 chars are not pruned; text is cut at line boundaries
into <=800-char pieces (upstream scores whole "sentences" through GPT-2's 1024-token window and
logs/code have no sentence breaks); upstream collapses whitespace inside a piece, so line
structure survives only at piece boundaries; a result that comes back empty becomes
`[output removed by selective-context]` and one that is not smaller is kept verbatim.

**provider-cm** `trigger`/`keep`/`clear_at_least` default to 3000 input tokens / 2 tool uses /
1000 tokens (`--cm-trigger/--cm-keep/--cm-clear-at-least`), recorded per row. These are a
harness choice, **not Anthropic's defaults** (100k tokens, keep 3): the committed 300-task runs
average ~3.9k prompt tokens per step (p90 ~5.9k), so the documented defaults would never fire
and the arm would equal `plain`. Anthropic notes that clearing invalidates the cached prefix;
that cost is part of what is measured.

Reuse: `run --reuse-arm plain=<dir>` copies that arm's rows (and transcripts) for the selected
instances from a finished run into `--out`, tagged `reused_from`, and does not run the arm. The
source must have the same model and effort (anything else is refused); reused rows cost nothing
against `--budget-usd`; `grade` carries their grades over from the source's `grades.jsonl`; the
report states which arms were reused and from where. Not checked: `--max-steps` (rows do not
record it) and the distil version that produced reused `distil` rows. Reuse is the operator's
statement that the configurations match.
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
With more than two arms the same rule (margin, Wald CI, exact McNemar) is applied to each
challenger against `plain` on the pairs graded in both; the summary table scores every arm on the
instances graded in all arms. The p-values/CIs are per comparison and uncorrected (the report
prints the Bonferroni alpha); the head-to-head is exploratory about *which* competitor differs
and is not a pre-registered claim beyond the distil-vs-plain rule above.
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
`run` hard-stops when live spend (including prior results in the out dir, excluding rows copied by
`--reuse-arm`) reaches `--budget-usd`; overshoot is at most one API call.

Per-arm estimate from cached results: `plan --arms ... --calibrate <run dir>` uses each arm's
mean cost per task in that run's `results.jsonl` (rows that never reached the API excluded);
an arm with no cached rows is priced at plain's mean x1.5 (a safety margin, labelled `proxy`, not
a prediction); `--reuse-arm` arms cost nothing; a model/effort mismatch with the calibration run
prints a warning. From `swebench-outcome-300-medium` (sonnet-5-5, medium, cached): plain $0.0284,
distil $0.0281 per task; the three competitor arms priced at $0.0426 per task.

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
pip install anthropic datasets 'swebench>=4,<5'  # in a venv; Docker running
python -m benchmarks.swebench_outcome plan --limit 5 --seed 0
# DockerEnv does not pull: pre-pull each planned instance image (swebench/sweb.eval.x86_64.<id>:latest)
python -m benchmarks.swebench_outcome run --limit 5 --seed 0 --budget-usd 25 \
    --i-understand-this-costs-money --out benchmarks/results/swebench_outcome/pilot
python -m benchmarks.swebench_outcome grade  --out benchmarks/results/swebench_outcome/pilot
python -m benchmarks.swebench_outcome report --out benchmarks/results/swebench_outcome/pilot
python -m benchmarks.swebench_outcome plan --pilot benchmarks/results/swebench_outcome/pilot/results.jsonl
```
Head-to-head on all 300 tasks, reusing the committed plain and distil rows (same model, effort
medium, `--max-steps 60`, caching on) and running only the competitors:
```
D=benchmarks/results/swebench-outcome-300-medium
python -m benchmarks.swebench_outcome plan --max-steps 60 --arms plain,distil,rtk,selective,provider-cm \
    --calibrate $D --reuse-arm plain=$D --reuse-arm distil=$D          # no Anthropic call, no spend
python -m benchmarks.swebench_outcome run --max-steps 60 --arms plain,distil,rtk,selective,provider-cm \
    --reuse-arm plain=$D --reuse-arm distil=$D --selective-python /path/to/py310/bin/python \
    --budget-usd 50 --i-understand-this-costs-money --out benchmarks/results/swebench-outcome-h2h
python -m benchmarks.swebench_outcome grade  --arms plain,distil,rtk,selective,provider-cm --out ...
python -m benchmarks.swebench_outcome report --arms plain,distil,rtk,selective,provider-cm --out ...
```
Reused distil rows come from the distil build that produced them (1.56.3 shell-search fix); re-run
`distil` fresh (drop its `--reuse-arm`) if the comparison must use today's build.
Run from a shell that is not distil-wrapped (the harness ignores `ANTHROPIC_BASE_URL` regardless).
`run` is resumable by (instance, arm): re-running the same command skips completed pairs;
`results.jsonl` is append-only and fsynced; a torn last line is dropped and re-run.
Pilot sanity: confirm in `transcripts/distil/*.json` that digests appear and `expand_calls` is
sensible before spending on the full run.

## Verified vs assumed: competitor arms
Verified (2026-10-04): RTK v0.51.0 release `checksums.txt`; the linux musl tarball downloads,
matches the pinned sha256 and contains an ELF `rtk` (`arm_rtk.fetch_rtk`); the darwin v0.51.0
binary's `rtk rewrite` exit codes and outputs (`git status` -> exit 0 `rtk git status`, `cat f` ->
exit 3 `rtk read f`, `echo` -> exit 1); in a real `alpine:3.20` linux/amd64 container run through
`DockerEnv` with `--cap-drop ALL`: binary install, version check, real `rtk rewrite`, `rtk read`.
`selective-context` 0.1.4 source (API, defaults, the 1024-token window, stdout printing, MIT in
PyPI classifiers) and its spaCy 3.2.0 pin; GPT-2 file sizes via the Hugging Face API; the
context-editing beta header, parameter names, response shape and `anthropic` 0.111.0's
`betas`/`context_management` parameters, against the official docs.
**Not verified**: a real SWE-bench container with rtk (conda activation, `rtk pytest` on the
repos' test runners); the worker against the real `selective-context`/GPT-2 (tested with a stub
library only; install it before spending); `provider-cm` against the live API (beta behaviour, and
whether `bash_20250124`/`text_editor_20250728` + adaptive thinking are accepted on the beta
endpoint); the effect sizes of any arm.

## Cost accounting: each arm pays for its own cache

The 2026-10-05 Lite head-to-head arms (rtk, selective, provider-cm) ran the same day with
byte-identical first requests and read each other's prompt cache, so their billed cost was not
their own (`benchmarks/results/swebench-outcome-300-h2h/README.md`, "Cost confound";
`docs/research/why-rtk-wins.md`). Two guards follow.

- **Cache namespace (default on).** `run` sets a per-run nonce (`Cfg.cache_ns`) and each arm's
  system prompt starts with `[cache namespace ARM-NONCE]`, so no two arms (and no two runs) share
  a cached prefix. Within one arm the prompt is identical from task to task, as a real agent's is.
  The row records `cache_ns`. `--no-cache-namespace` sends the bare prompt (what runs before
  2026-10-06 sent); use it only to reproduce one of them.
- **Cold-accounting check.** An append-only run that starts cold writes at least its largest
  prompt, so `cache_write + input >= cache_read / (steps - 1)` for every row with 2+ steps.
  `report.cache_check` lists the arms with rows below that and `report` appends it to
  `report.md` as `## Cost confound`; `scripts/build_scoreboard.py` applies the same check
  (`report.cache_contamination`) and shows such an arm's dollars only as a full cold re-pricing
  (named per run in `benchmarks/results/scoreboard-runs.json`, `cold_repriced`), otherwise
  "confounded". Success rates are not affected by either.

## Threats to validity
- **Competitor arms are not like-for-like layers.** RTK shrinks bash output only (the editor
  tool is untouched); Selective Context is lossy and question-agnostic and flattens whitespace;
  `provider-cm` clears whole old tool results server-side and invalidates the cache; distil
  digests with a restore tool. The comparison is end-to-end outcome and cost, not compression
  ratio. The `provider-cm` thresholds are scaled to these short tasks (see above) and a different
  choice would give a different result.
- **Contamination**: SWE-bench Lite is public; absolute rates overstate ability. The paired
  difference is less affected, but memorized fixes may make compression look harmless.
- **Scaffold != your agent**: this loop has no caching, no planning tools, no subagents, 20k-char
  tool-output cap (identical in both arms). Real agents differ in how much tool output they carry.
- **Noise**: single attempt per arm; agent stochasticity dominates. See power table.
- **Single model/effort**: a result for sonnet-5-5 at medium says nothing about other models.
- **Compression is applied to the full history each call**: matches the served transform, but the
  harness does no prompt-cache interplay, so cost savings are not a caching-aware estimate.
- **Expand tool is model-discoverable only via the digest markers**: the system prompt is identical
  across arms by design apart from the one-line cache namespace; if distil injects extra instructions in deployment, this under-represents it.
- **Env**: each bash call is a fresh shell (no persistent cwd/env); the patch includes untracked
  files; grading uses the official harness, so eval-time flakiness affects both arms equally.
- **Class exclusion** can bias the paired set if api/env failures correlate with arm (check the
  `classes` column).
- Not implemented: parallel workers, repeated attempts, per-step transcripts in the report.
