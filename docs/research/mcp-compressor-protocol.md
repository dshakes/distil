# Pre-registered protocol — `distil mcp` tool-definition and result compression

- **Protocol version:** 1
- **Registered:** 2026-09-25, committed before any live run. A live result file records
  the git SHA of this document; any change to it after that commit is listed under
  *Deviations* below, dated, with its reason.
- **Executable form:** `distil/mcpproxy/bench.py` (`distil mcp bench`). Where this text
  and that code disagree, the disagreement is a defect to be logged, not resolved
  silently in either direction.
- **Status:** live run 1 made 2026-09-25 on `claude-haiku-4-5` ($21.68 of an $85
  ceiling): L0, L1, L2, L3 and R all **certified** for that model. See *Results* at the
  end. Amended once before data (Amendment 1, under *Deviations*).

## 1. Question

`distil mcp` rewrites what an agent is told it can call (tool definitions, levels
L0–L3) and what a call returned (results, R). Token savings are measurable offline and
exactly; the question that is not is **whether the model still picks the right tool
with the right arguments**. Other MCP compressors publish the first number only. This
protocol decides the second, per level, before anyone looks at an outcome.

## 2. Hypotheses

For each definition level `ℓ ∈ {L0, L1, L2, L3}`, against the uncompressed reference
arm `raw` (the backends' own `tools/list`, no proxy rewriting), on the same tasks:

- **H1ℓ (selection):** tool-selection accuracy under `ℓ` is non-inferior to `raw`
  within margin `CERT_MARGIN`.
- **H2ℓ (arguments):** argument exact-match under `ℓ` is non-inferior to `raw` within
  `CERT_MARGIN`.

A level is certified only if **both** H1ℓ and H2ℓ are accepted (intersection–union: no
multiplicity correction is needed within a level).

For results:

- **H5 (R):** answer accuracy from an R-digested tool result (with the
  `<server>_expand` recovery tool available) is non-inferior to answer accuracy from the
  raw result within `CERT_MARGIN`.

Secondary, reported for every arm, never gating: task success (below), extra round
trips per task, model-facing definition tokens, input tokens per task, expand calls per
result task.

## 3. Metrics (exact definitions — `bench.score`, `bench._schema_ok`)

Every task has a gold tool set (usually one tool; two where the server ships an alias,
e.g. `read_text_file` / deprecated `read_file`) and gold arguments (only the keys the
prompt fully determines).

- **Selection** = 1 iff the backend tool finally called is in the gold set. A call made
  through `<server>_invoke_tool` counts as a call of its `tool_name`. A schema fetch is
  not a final call. A text answer, an unknown name, or running out of turns
  (`MAX_TURNS = 4`) scores 0.
- **Arguments** = 1 iff selection = 1 **and** every gold key is present with an equal
  value. Equality: numbers compare numerically (`5 == 5.0 == "5"`), booleans accept
  their string spelling, lists compare as multisets, strings exactly. Extra keys are not
  penalised here.
- **Task success** (secondary) = arguments = 1 **and** the call is valid against the
  real schema: every required key present, and no undeclared key where the schema sets
  `additionalProperties: false`.
- **Extra round trips** = model tool calls before the final one (schema fetches).
- **Answer accuracy** (R) = 1 iff the gold answer string occurs, case-insensitively, in
  the model's final text.

## 4. Materials

- **Tool catalog.** The verbatim `tools/list` of eight MCP reference servers, captured
  over stdio on 2026-09-24 and vendored in `distil/mcpproxy/fixtures/` (filesystem, git,
  github, memory, fetch, time, everything, sequential-thinking — 78 tools; licences in
  `fixtures/NOTICE.md`). **All eight servers are exposed at once** in every arm: the
  realistic multi-server setting, and the one where selection is hardest.
- **Tool tasks.** `bench.generate_tasks(1000, seed=0)`: 47 templates (single-call
  requests whose arguments the prompt fully specifies), slots filled from fixed pools
  with `random.Random(0)`, duplicates rejected. Deterministic.
- **Result tasks.** `bench.generate_result_tasks(630, seed=0)`: three kinds in rotation —
  a 150–400-row directory listing (*which file is largest?*, the answer planted mid-list
  where a digest folds it), a 200–500-line build log (*what error code?*), a 60–160-record
  JSON array (*what is order N's status?*).
- **L3's learned usage** is the gold-tool counts of the **first half** of the tool tasks
  (`bench.usage_profile`). It is fixed before the run and every task is scored, so the
  pins cannot be tuned on outcomes. (The pins do see the task distribution; that is what
  "learned from use" means, and it is disclosed as a threat below.)
- **Sessions.** Each task runs in a fresh proxy session, so no unlock carries across
  tasks; every call, final ones included, goes through the real `proxy.Proxy`.

## 5. Analysis

For each level and metric, the paired per-task differences `d_i = arm_i − raw_i`:

1. **TOST lower test** (`certify.stats.tost`): reject `H0: mean(d) ≤ −CERT_MARGIN` at
   one-sided `α = BUDGET_DELTA`.
2. **Bootstrap:** 2,000 paired percentile resamples (`random.Random(0)`); the
   one-sided `(1−α)` lower bound of `mean(d)` must exceed `−CERT_MARGIN`.
3. Both must hold (`bench.compare`). The McNemar Wald interval
   (`certify.stats.mcnemar_noninferiority`) is reported alongside, not gating.

Constants (single risk budget — no second copy of either number anywhere):

| constant | value | source |
|---|---|---|
| `CERT_MARGIN` | 0.02 | `conformal.CERT_MARGIN` where it exists; else `certify.stats.tost`'s default margin, read by `inspect` |
| `α` | 0.05 | `drift.BUDGET_DELTA` |

**Multiplicity across levels:** fixed-sequence testing in the order **L0 → L1 → L2 →
L3**, each at full α, stopping at the first level that is not certified; later levels
are reported `not-tested`. This controls the family-wise error rate at α without
splitting it, because the order is fixed here, before data. R is a separate family
(different outcome, different tasks) tested at its own α.

**Underpowering guard:** a level whose point estimates pass but whose observed
discordant-pair rate exceeds the design assumption (§6) is `inconclusive`, never
certified, and stops the sequence.

## 6. Sample size

Paired binary non-inferiority at a true difference of zero (the McNemar variance of
correlated proportions — Connor 1987, Nam 1997):

`n = (z₁₋α + z_power)² · p_d / CERT_MARGIN²`, with `α = 0.05` one-sided, power 0.80.

| suite | assumed discordant rate `p_d` | required `n` | planned `n` |
|---|---|---|---|
| tool tasks (per level) | 0.06 | 928 | **1000** |
| result tasks (R) | 0.04 | 619 | **630** |

`bench.required_n` computes these; `tests/test_mcp_bench.py` fails if a planned `n`
ever drops below its requirement.

## 7. Models and settings

- **Primary:** `claude-sonnet-5`. **Replication:** `claude-haiku-4-5` (smaller models
  are expected to be the more sensitive to compressed descriptions).
- `temperature 0`, `max_tokens 1024`, `tool_choice` default (auto), the fixed system
  prompt in `bench.SYSTEM`, the tool list marked `cache_control: ephemeral` (caching
  changes cost, not outputs).
- Live runs go through `bench.AnthropicModel`, which refuses to start without
  `--live`, `--budget-usd` and `ANTHROPIC_API_KEY`, refuses a run whose estimated upper
  bound exceeds the cap, and stops the moment actual billed spend reaches the cap.

## 8. Stopping rules

- **One look.** Each level is analysed once, at the planned `n`. No interim peeks, no
  extension, no re-run with a new seed to rescue a result.
- **Budget cap reached** before the run completes → the run has **no verdict**; nothing
  is certified from a partial run.
- **Unrecoverable API error** (a non-retryable HTTP error, or four failed attempts on a
  retryable one) aborts the run → no verdict.

## 9. What counts as a failure

- A level **fails** if either primary metric's TOST does not reject or its bootstrap
  lower bound is `≤ −CERT_MARGIN`.
- `inconclusive` (§5) and `not-tested` are not certified.
- R **fails** on the same rule for answer accuracy.
- A failure is published, with its numbers, exactly as a pass would be.

## 10. Decision rule for the default

The shipped default is the most aggressive definition level certified on the primary
model **and not failed** on the replication model, plus R if H5 is certified on both.
Until a live run exists the default is **L0 only**: L0 is validation-equivalent by
construction (`tests/test_mcp_levels.py` proves every change is an annotation keyword
or whitespace, on all 78 real schemas). R and L1–L3 are opt-in and say so on every start.
`distil/certificates/mcp.json` is edited only by a human, from a committed live run.

## 11. Dry-run validation (plumbing, not a result)

`distil mcp bench --behavior <b>` (no network, no spend) drives the real proxy with a
scripted mock model. Committed outputs: `benchmarks/results/mcp_toolbench/dryrun-*.json`.
They show the harness and the statistics behave as designed:

- `oracle` / `invoke` — every arm scores 1.0; lazy arms take the schema-fetch round trip;
  the invoke fallback reaches the same tools; R needs `expand` on the listing tasks.
- `noisy` — independent errors at the same rate in every arm (no true difference):
  every level certified.
- `degraded` — a true loss injected on L2 only: L0 and L1 certified, **L2 failed**, L3
  `not-tested` (the fixed sequence stopped).
- `no-expand` — a model that never calls `expand`: **R failed**.

These verdicts describe the mock model. They are not, and are never presented as,
evidence about any real model.

## 12. Cost of the live run (estimate — nothing has been spent)

`benchmarks/results/mcp_toolbench/cost_estimate.json`, from the oracle dry run's exact
turn structure × `distil.pricing` list prices, including Anthropic's documented
tool-use system-prompt overhead and a ×1.25 tokenizer safety factor. "Upper bound"
bills every input token at the base rate; "cached" assumes the arm's tool list is read
from the prompt cache after its first write. Full protocol (5 arms × 1000 tool tasks,
2 arms × 630 result tasks) per model:

| model | cached (expected) | upper bound |
|---|---|---|
| claude-haiku-4-5 | $26.36 | $110.69 |
| claude-sonnet-5 | $79.08 | $332.06 |
| claude-opus-5 | $131.79 | $553.43 |

The pre-registered run is primary + replication: **$105.44 expected, $442.75 upper
bound**. The cap passed to `--budget-usd` must be at least the model's upper bound, or
the harness refuses to start. **Approval of this spend is the maintainer's; it has not
been given.**

## 13. Threats to validity

- **Synthetic, single-call, fully-specified tasks.** Real requests are vaguer and
  multi-step; accuracy here is an upper bound on what compression can preserve, and the
  paired design only licenses the *difference*.
- **Reference servers only.** Enterprise servers with hundreds of long-described tools
  are where L2/L3 save most and where selection is hardest; they are not in the catalog.
- **One client shape.** The harness refreshes its tool list on `list_changed`; clients
  that do not are covered by the `invoke` fallback, measured only in the dry run.
- **L3's pins see the task distribution** (first half). A deployment's usage is its own.
- **Heuristic tokenizer** for token counts and the cost estimate (±, hence the ×1.25).
- **Templated prompts** share phrasing within a template; tasks are not independent
  draws from real usage. The bootstrap resamples tasks, not templates.

## Results — live run 1 (2026-09-25, `claude-haiku-4-5`)

Artifacts: `benchmarks/results/mcp_toolbench/live_claude-haiku-4-5.json` (verdicts,
paired statistics, per-arm tokens and dollars) and `live_calls_claude-haiku-4-5.jsonl`
(one content-free line per API attempt: arm, task id, attempt, status, input / cache
write / cache read / output tokens, dollars). Harness at `fe40c2b`, after Amendment 1
(`54ea03d`, committed 2026-09-25T08:26:11-04:00, before the first model call).

**Spend:** $21.6812 of the $85 ceiling (expected before the run: $26.36). 7,564 API
attempts, 0 failed, 0 retries. The run completed; nothing was stopped by the meter.

**Tool family** (n = 1000 per arm, same tasks, fixed sequence L0 → L1 → L2 → L3). Every
level was certified, so the sequence never stopped.

| arm | selection | args | Δ selection (boot. 95% lower) | Δ args (boot. 95% lower) | losses / gains (args) | extra round trips / task | verdict |
|---|---|---|---|---|---|---|---|
| raw | 0.944 | 0.912 | — | — | — | 0.0 | reference |
| L0 | 0.952 | 0.921 | +0.008 (+0.004) | +0.009 (+0.004) | 0 / 9 | 0.0 | **certified** |
| L1 | 0.954 | 0.954 | +0.010 (+0.005) | +0.042 (+0.032) | 0 / 42 | 0.0 | **certified** |
| L2 | 0.998 | 0.963 | +0.054 (+0.042) | +0.051 (+0.038) | 4 / 55 | 1.006 | **certified** |
| L3 | 0.988 | 0.954 | +0.044 (+0.033) | +0.042 (+0.031) | 3 / 45 | 0.088 | **certified** |

The non-inferiority bound is −0.02. Every TOST p-value is below 1e-18. The largest
observed discordance was 0.059 (L2 args), under the design's 0.06, so no level is
`inconclusive`.

**R** (n = 630): answer accuracy 0.9968 raw vs 0.9952 R, Δ −0.0016, 2 losses / 1 gain,
bootstrap 95% lower bound −0.0063 > −0.02, TOST p 2.4e-11: **certified**. 0.33 expand
calls per result task.

**Read these carefully.**

- Every compressed level scored *higher* than raw. The protocol tests non-inferiority
  only, so this is reported as observed, **not** as a superiority claim. Why the
  uncompressed 78-tool list did worse was not investigated.
- The certificates cover `claude-haiku-4-5` on the eight reference servers with
  synthetic, single-call tasks (§13). There was no replication model this round
  (Amendment 1).
- **Cost is not accuracy.** Billed dollars on the tool suite, from `spend_by_arm`: raw
  $2.17, L0 $2.00, L1 $1.90, **L2 $8.79**, L3 $1.82. On the result suite: raw $3.08,
  R $1.92. L2 was billed 7,748,814 uncached input tokens against raw's 378,154, because
  its short session-start prompt was never written to the prompt cache on this model
  (cache writes on 0 of its 1,000 first turns, per the call log), and every L2 task took
  a second round trip. Every task here is a fresh session. In a long session the unlocked
  set persists and the cache can warm up, but that was not measured. L3 was cheaper
  than raw only because its pins were learned in advance (§4). A fresh L3 install has
  no usage yet and starts as L2.

**Default decision (§10).** The rule designates the most aggressive certified level
that has not failed on the replication model, plus R if it is certified on *both*
models. There is no replication model this round, so the R clause cannot be met and the
level clause is met only vacuously. **The shipped default stays L0.** L1–L3 and R stay
opt-in, now shown as `certified` for `claude-haiku-4-5` in `distil mcp watch`, on the
dashboard, and in `distil/certificates/mcp.json`, so the pending-certificate warning no
longer prints for them. Making L3 (and/or R) the default is a maintainer decision. It
should wait for the replication model and a measurement of L2's cached cost in long
sessions.

- **Deviation (logged after the run):** the header says a live result file records the git SHA
  of this document. The harness did not write it. The SHAs above were added to the
  artifact's `provenance` block after the run, from `git log`, along with
  `spend_by_arm` (computed by `bench.spend_by_arm` from the committed call log). No
  measured value changed.

## Deviations

- **2026-09-25, before any live run** (security review of the first implementation):
  1. §10's interim default changed from L0 + R to **L0 only**; R is opt-in until H5 is
     certified. No hypothesis, metric, sample size or analysis changed.
  2. The harness now runs **one proxy per server** — the deployment `distil mcp install`
     produces — instead of one proxy fronting all eight, because a multi-server proxy now
     namespaces tool names (`<server>__<tool>`), which would confound the comparison with
     the raw arm. The dry-run outputs are unchanged by this.

### Amendment 1 (pre-data) — 2026-09-25

Made after the maintainer approved spend and **before any live model call** (no model
output has been seen by anyone). This commit's timestamp precedes every live artifact.

1. **Budget:** a **hard ceiling of $85** for this round (`--budget-usd 85`).
2. **Models (§7):** the primary model for this round is **`claude-haiku-4-5`**. The
   `claude-sonnet-5` run is **deferred** for budget; there is **no replication run this
   round**. §10's "not failed on the replication model" clause therefore cannot be
   satisfied yet: any default decided from this round is decided on the primary model
   alone and says so.
3. **Unchanged:** tasks and seeds, planned `n` (1000 tool tasks per level, 630 result
   tasks), metrics, TOST at `CERT_MARGIN` with `α = BUDGET_DELTA`, the bootstrap rule, the
   underpowering guard, fixed-sequence testing L0 → L1 → L2 → L3 stopping at the first
   level not certified, R as its own family, one look, and §8's rule for unrecoverable API
   errors.
4. **Stopping rule for spend** (replaces §7's "refuses a run whose estimated upper bound
   exceeds the cap"): the harness refuses to start unless the *expected* (cached) cost
   is at most the budget **and** a live spend meter enforces the ceiling. The meter bills
   every response from the provider's returned `usage` (input, cache write, cache read,
   output × `distil.pricing` list price) and, before every call — retries included —
   refuses to send a call whose worst case could take total spend past the ceiling.
   Failed retryable attempts (429/5xx) are booked conservatively at their estimated
   input cost. If the meter stops the run before the pre-registered `n` completes for a
   family (the tool family L0–L3, or R), **that family reports NO VERDICT** — never a
   partial or underpowered claim — and every family not yet run reports NO VERDICT too.
5. **Execution details that do not touch outcomes:** levels run in the fixed sequence,
   and a level after the first non-certified one is not run (it would be `not-tested`
   regardless; this only saves spend). Tasks within an arm may be sent concurrently (each
   task is an independent, fresh session at temperature 0); retry backoff honours the
   provider's `retry-after`. Before the first model call, each arm's tool list is
   validated against the provider's free token-counting endpoint (no model output is
   produced, nothing is billed). Per-call usage (tokens and dollars, no content) is
   recorded so spend is auditable.
