# Pre-registered protocol — `distil mcp` tool-definition and result compression

- **Protocol version:** 1
- **Registered:** 2026-09-25, committed before any live run. A live result file records
  the git SHA of this document; any change to it after that commit is listed under
  *Deviations* below, dated, with its reason.
- **Executable form:** `distil/mcpproxy/bench.py` (`distil mcp bench`). Where this text
  and that code disagree, the disagreement is a defect to be logged, not resolved
  silently in either direction.
- **Status:** registered; **no live run has been made**. Every level's certificate in
  `distil/certificates/mcp.json` is `pending`.

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
Until a live run exists the default is **L0 + R**: L0 is validation-equivalent by
construction (`tests/test_mcp_levels.py` proves every change is an annotation keyword
or whitespace, on all 78 real schemas); R reuses the digest distil's LLM proxy already
ships, with byte-exact recovery. L1–L3 are opt-in and say so on every start.
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

## Deviations

None.
