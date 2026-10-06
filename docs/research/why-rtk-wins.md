# Why RTK looks cheaper than distil on coding agents (2026-10-06)

An offline, $0 root-cause analysis of the three benchmarks where distil did not save money
and RTK appeared to. No API calls. Reproduce with:

```
python benchmarks/why_rtk_wins.py \
  --lite  <scoreboard>/benchmarks/results/swebench-outcome-300-h2h \
  --longh <longh>/benchmarks/results/swebench-verified-hard-max \
  --tbench benchmarks/results/cost_truth/pilot-20260925-145427 \
  --out benchmarks/results/why-rtk-wins/results.json --replay
```

`results.json` is aggregate-only. Prices: claude-sonnet-5(-5) $2 / $10 per MTok, cache write
1.25x, cache read 0.1x (the Terminal-Bench pilot is re-priced from its billed $3/$15 to $2/$10).

## The answer

1. **Most of RTK's SWE-bench Lite win is a measurement artifact.** RTK, selective and
   provider-cm ran on the same day and send a byte-identical first request (system prompt + tools
   + task statement). Whichever arm reached a task first paid to write that prefix, and the other
   arms read it at 0.1x. **189 of 298 RTK tasks** report cache accounting that a cold run cannot
   produce (`cache_write + input < cache_read / (steps - 1)`). For selective it is 167 and for
   provider-cm 138. plain and distil have none: they ran alone two days earlier, and distil's extra
   tool changes its prefix. The prompt served from another arm's cache is **0.99x the first
   request** (median, RTK). Re-priced as a cold run (total prompt tokens kept exact, only the
   write/read split moved), RTK goes from **-20.7% to -6.5% per task vs plain, 95% CI
   [-13.1%, +0.1%]**. That is not significant. **69% of the measured gap ($1.20 of $1.75 on
   298 tasks) is the artifact.**
2. **On short tasks there is almost nothing for a tool-output compressor to remove.** In a cold
   reconstruction of plain-equivalent Lite trajectories, tool output is **11.5% of the bill**.
   Output tokens are 42.2%, the system prompt and tool definitions (written fresh for each task)
   29.7%, and the task statement 10.4%. distil removes **7.7%** of tool-output characters at
   entry, which is ~0.9% of the bill. Its injected `distil_expand` tool adds **~200 prompt
   tokens to every request** (estimated from 2- and 3-step tasks; UNVERIFIED tokenization). At
   Lite's 4.6 steps that costs ~2.3% of the bill, more than the compression saves. Paired
   observed result: -0.9%, CI [-9%, +7%].
3. **distil no longer rewrites cached history in this harness.** Replaying distil 1.57.0's served
   transform step by step reproduces the provider's own counts on the long-horizon run (simulated
   write 0.964M vs billed 0.949M tokens; read 45.5M vs 46.0M), and finds **0 prefix rewrites in
   1,134 replayed steps** (475 long-horizon, 659 Lite). The cache-bust bug from the earlier live
   A/B is fixed here. Fixing it again will not win anything.
4. **Long horizon: the money is in the agent's own history and output.** At effort max (66
   steps per task), plain's bill is output 38.3%, re-sent assistant history (encrypted thinking +
   tool calls, re-read every step) 39.4%, tool output 20.9%, system+tools+task 1.4%. distil is
   **the cheapest arm per step** ($0.0382 vs plain $0.0467 and rtk $0.0421), but it took more
   steps and made 37 `distil_expand` calls. Its compression accounts for only 6.4% of prompt
   tokens. n = 7 tasks: no cost difference is shown.
5. **Terminal-Bench (Claude Code): distil loses on the number of requests, not on their price.**
   Per request distil is cheapest ($0.0161 vs control $0.0217 and rtk $0.0182). But it made
   **60% more requests** (573 vs 358), more than the control in 14 of 18 paired runs. Per solved
   task: distil $0.657, control $0.555, rtk $0.400. The cause of the extra requests is
   UNVERIFIED: the meter is content-free.

## 1. Cost decomposition

### SWE-bench Lite (298 tasks graded in every arm)

| arm | $ total | uncached | cache write | cache read | output | steps/task | $/solved |
|---|---|---|---|---|---|---|---|
| plain | 8.44 | 0.01 | 3.80 (45%) | 1.06 (13%) | 3.58 (42%) | 4.56 | 0.0387 |
| distil | 8.37 | 0.01 | 3.73 (45%) | 1.12 (13%) | 3.51 (42%) | 4.79 | 0.0393 |
| rtk (as billed) | 6.69 | 0.01 | 2.31 (34%) | 0.98 (15%) | 3.40 (51%) | 4.39 | 0.0306 |
| **rtk (cold re-priced)** | **7.89** | 0.01 | 3.61 (46%) | 0.87 (11%) | 3.40 (43%) | 4.39 | **0.0360** |
| selective | 9.86 | 0.01 | 2.98 | 1.92 | 4.95 | 6.89 | 0.0469 |
| provider-cm (as billed) | 8.15 | 0.02 | 3.91 | 0.81 | 3.41 | 4.76 | 0.0388 |

Tokens per step: plain writes 1,118, reads 3,887 and outputs 263. rtk as billed writes 706,
reads 3,737 and outputs 260. Cold, rtk writes 1,106 and reads 3,337. The billed difference is
almost entirely in cache writes, and the cache writes are where the shared prefix shows up.
After re-pricing, rtk is cheaper than plain by $0.19 in writes (fewer and smaller tool outputs),
$0.19 in reads and $0.18 in output. The read and output savings follow from fewer steps
(-0.18 per task, CI [-0.43, +0.06]).

Paired per task vs plain (mean $ difference, 95% bootstrap CI; plain = $0.0283/task):

| arm | as billed | cold re-priced | step diff |
|---|---|---|---|
| distil | -0.00026 [-0.0026, +0.0020] | n/a (no shared prefix) | +0.23 [-0.11, +0.57] |
| rtk | -0.00587 [-0.0078, -0.0040] | -0.00184 [-0.0037, +0.00003] | -0.18 [-0.43, +0.06] |
| selective | +0.00476 [+0.0023, +0.0075] | n/a | +2.33 [+1.93, +2.76] |

### Long horizon (SWE-bench Verified hard, effort max, 7 tasks common to all arms)

| arm | $ total | cache write | cache read | output | steps/task | $/step | $/solved |
|---|---|---|---|---|---|---|---|
| plain | 21.63 | 3.10 (14%) | 10.27 (47%) | 8.25 (38%) | 66.1 | 0.0467 | 3.61 |
| distil | 18.16 | 2.37 (13%) | 9.20 (51%) | 6.59 (36%) | 67.9 | 0.0382 | 3.03 |
| rtk | 21.54 | 2.99 (14%) | 10.82 (50%) | 7.73 (36%) | 73.1 | 0.0421 | 3.08 |

Cache hit ratio is 97.6-98.0% in every arm. No arm has impossible cache accounting.

### Terminal-Bench 2.1 pilot (Claude Code, 18 attempts per arm, infra blocks excluded)

| arm | requests | $ at $2/$10 | $/solved | uncached | write | read | output | prefix breaks (extra $) |
|---|---|---|---|---|---|---|---|---|
| control | 358 | 7.77 | 0.555 | 9% | 20% | 35% | 36% | 9 (0.30) |
| distil | 573 | 9.20 | 0.657 | 4% | 21% | 47% | 28% | 18 (0.58) |
| headroom | 315 | 6.80 | 0.523 | 1% | 37% | 35% | 28% | 15 (0.83) |
| rtk | 329 | 6.00 | 0.400 | 0% | 21% | 46% | 33% | 1 (0.12) |

A prefix break is a main-model request whose cache read is below 90% of the previous request's
prompt. This detector is a heuristic: it also fires on Claude Code's own subagent and compaction
traffic. distil's breaks cost about 4% of the control's bill more than the control's own breaks.
That is not the 18% gap.

## 2. Steps and behaviour

- Lite: every arm makes about 0.97 tool calls per step. Identical re-runs of a command are
  0.0-0.15% of calls. distil made 4 `distil_expand` calls in 1,427 steps. There were no
  `max_tokens` stops. selective's pruning added 2.3 steps per task, which is why it is the most
  expensive arm.
- Long horizon: 1.17-1.20 calls per step, identical re-runs 0.2-0.4%. distil made 37 expand
  calls (24 of them on one task). One `max_tokens` stop for distil and one for rtk: adaptive
  thinking at effort max spent the whole budget.
- **No arm wins on cheaper steps on Lite.** RTK's residual edge is fewer steps and smaller
  prompts (-3,354 prompt tokens per task, CI [-6,868, -209]). On the long horizon distil has the
  cheapest steps but not the fewest.
- Provider drift: output per step is 263 for plain (10-03), 246 for distil (10-03), 260 for rtk
  (10-05) and 240 for provider-cm (10-05). No day effect is visible. The only day effect found is
  the shared-cache one above.

## 3. Cache behaviour, and the rewrite hypothesis quantified

A token that enters context at step j is billed once as a write (1.25x) and then 0.1x for each
of the r steps that remain. On average, from the reconstruction, one tool-output token costs
**1.35x the input price on Lite and 5.9x on the long horizon**. Removing a token at entry saves
that whole amount. Removing x tokens from history that is already cached saves only 0.1·x·r,
and it costs 1.15·(P - p) once to rewrite everything after the edit point (P - p ≥ x tokens). It
breaks even only when **r > 11.5·(P - p)/x ≥ 11.5 remaining steps**, and only for edits near
the tail. The median Lite task has 4 steps, so the condition is never met there.

The data confirms both sides:

- distil, which compresses at entry, has 0 rewrites and a simulation that matches the provider's
  counts (section 1).
- provider-cm, which clears cached history server-side, rewrote the cached prefix on 70 of 72
  steps at effort max. It cost $5.03 on a task where the other arms spent about $2 (longh
  README). On Lite it cut reads 26% per step but its writes did not fall, and it ended no cheaper
  than plain.

## 4. Why RTK "wins" on Lite

- **69% of it is the shared prefix** (section 1). This is a benchmark bug, not an RTK property.
- RTK rewrote **537 of 1,265 bash commands (42%)**. Tool output per call is 968 characters vs
  1,070 for the same-day raw arm (-9.5%). By command class:

  | command | rtk chars | raw chars |
  |---|---|---|
  | grep (408 calls) | 1,388 | 1,492 |
  | sed (343 calls) | 1,036 | 1,157 |
  | git (45 calls) | 294 | 970 |
  | python -c | 577 | 643 |
  | cat | 1,807 | 2,017 |

  That is about as much as distil removes at entry (7.7%). Because tool output is 11.5% of the
  bill, either one is worth about 1%.
- The rest (-6.5% cold, not significant) is fewer steps and less output. The harness leaves
  RTK's system prompt unchanged (RTK.md is silent), so there is no instruction-level cause.
  It is either a behavioural effect of terser output or noise; this data cannot tell them apart.

## 5. Where the money is

Shares of the reconstructed cold bill (plain-equivalent requests):

| source | Lite | long horizon |
|---|---|---|
| (a) new tool output entering context + its re-reads | 11.5% | 20.9% |
| (b) re-reading cached history (all sources, 0.1x) | ~7% | ~47% |
| assistant history re-sent (thinking signatures, tool calls) | 6.2% | 39.4% |
| (c) output tokens incl. thinking | 42.2% | 38.3% |
| (d) system prompt + tool definitions (+ task statement) | 29.7% (+10.4%) | 0.9% (+0.5%) |

The ceiling for any tool-output compressor is 11.5% on Lite and 20.9% on the long horizon.
Output and encrypted thinking (together 48% on Lite, 78% on the long horizon) cannot be touched
by input compression. Headroom's handler says the same about thinking: "an encrypted handle we
can't shrink".

## 6. Cross-provider cache economics

| provider | write | read | TTL | what pays |
|---|---|---|---|---|
| Anthropic (Sonnet 5.x; `distil/pricing.py`) | 1.25x (5m), 2x (1h) | 0.1x (Opus 5.5: 0.05x) | 5m / 1h | Compress at entry only. A rewrite costs 1.15x per token rewritten. Cheaper reads (Opus 5.5) make history even less worth compressing. |
| OpenAI GPT-5.6+ | 1.25x | 0.1x (GPT-6.1: 0.05x) | 30m | Same as Anthropic. |
| OpenAI before 5.6 | no surcharge | model-dependent, automatic | 5-10m in memory / 24h | A rewrite only loses the discount, (1-d) per token. With a 50% discount, compressing history pays after about 1-2 remaining steps. |
| Gemini 2.5+ implicit | no surcharge | 10% of input | best effort | Entry compression still dominates: a rewrite costs 0.9x per token and hits are not guaranteed. Explicit caches add hourly storage. |

Sources: developers.openai.com/api/docs/guides/prompt-caching, ai.google.dev/gemini-api/docs/caching
and /pricing (fetched 2026-10-06). `distil/pricing.py` models Anthropic multipliers only.
`distil/compress/recency.py` already treats implicitly cached providers as "everything is
committed", which means entry-only compression.

## 7. Headroom (v0.40.0 source, read-only)

- **It compresses only the live zone.** `crates/headroom-core/src/transforms/live_zone.rs`
  compresses blocks in the latest user message only. It never goes below
  `frozen_message_count` (derived from `cache_control` markers) and never touches the latest
  assistant turn. That is the same rule distil enforces through
  `distil/compress/recency.py::exempt_indices`, so **neither tool rewrites warm cached
  history.** This design does not explain a cost gap between them.
- **Cold recompaction.** In `headroom/proxy/handlers/anthropic.py`, `HEADROOM_COLD_RECOMPACT`
  recompacts the whole prefix only when the cache has already lapsed (idle > TTL), so there is
  nothing to bust. The recompaction is deterministic, so the result re-caches byte-stable.
  `headroom/cache/ttl_estimator.py` learns per-model TTLs, conservatively (it overestimates
  rather than risk busting a warm cache). `headroom/pricing/cache_ttl.py` models the 5m/1h write
  trade. distil has no cold-only recompaction. On these benchmarks it is worth $0, because the
  agents never sit idle longer than the TTL.
- **Breakpoint placement.**
  `crates/headroom-proxy/src/cache_stabilization/anthropic_cache_control.rs` (PR-E3) inserts
  a breakpoint for PAYG clients that place none. `tool_def_normalize.rs` sorts tools so their
  bytes stay stable. That would have cached the system+tools head across sessions, the 29.7% item
  above. It does not apply to this harness, which places its own marker.
- On Terminal-Bench, Headroom's per-request cost ($0.0216) equals the control's. Its edge over
  distil is fewer requests (315 vs 573), not cache design. `claude_analysis_ttl.py` analyses
  idle-gap re-creation cost for interactive Claude Code sessions, which none of these benchmarks
  have.

## Ranked fixes

Estimates are shares of plain's bill per task, taken from the decomposition above:
effect = (share of bill) × (fraction removed) − (overhead).

| # | fix | expected effect | effort | risk |
|---|---|---|---|---|
| 1 | **Fix the benchmark.** Give each arm a unique cache namespace (an arm nonce at the start of the system prompt), or run arms in separate windows more than 1h apart, or report the cold re-pricing. Re-run plain in the same session as the competitors. | Removes a 14-percentage-point bias in RTK's favour (-20.7% → -6.5%). Changes the conclusion more than any product change. | low | none |
| 2 | **Make `distil_expand` nearly free.** Cut its definition from ~200 tokens to ~50, and put usage guidance in the digest marker instead. Do not add it mid-session: changing `tools` invalidates the whole cached prefix. | Lite +1.7% (≈$0.0005/task). Long horizon ~0.1%. | low | low (check expand calls still resolve) |
| 3 | **Stronger entry compression for the commands SWE agents actually run** (grep 40%, sed -n 21%, python -c 12%): group grep hits by file, strip the shared `/testbed/` prefix, fold duplicate lines, and compact numbered-line runs. Raise entry reduction from 7.7% (Lite) and 20% (long horizon) to about 35%. | Lite +3.1%. Long horizon +3.1% (≈$0.10/task). | medium | accuracy: gate with the outcome harness |
| 4 | **Find and fix the extra requests** (Terminal-Bench +60%, 14 of 18 runs; long horizon +1.7 steps per task and 37 expand calls). Instrument expand calls and re-runs after a digest in the proxy. | If distil matched the control's request count at its own per-request price: Terminal-Bench about -26% vs control (UNVERIFIED cause). | medium | unknown until the cause is known |
| 5 | **Cache the static head across sessions** (a breakpoint after system/tools, stable tool order) for clients that place none. | Up to ~23% of Lite's cold bill when sessions arrive within the TTL. It only beats a plain client that does not already do this (Claude Code does). It is exactly what the competitor arms got by accident. | low | none for accuracy; the 1h TTL write costs 2x |
| 6 | **Cold-only history recompaction** (Headroom-style). | $0 on these benchmarks (no idle gaps); positive for interactive sessions | medium | low if deterministic |

**Levers this data says cannot win:**

- Further fixes to prefix rewrites (there are 0 left).
- Compressing cached history on short tasks (break-even needs more than 11.5 remaining steps).
- Capping identical re-runs (0.1-0.4% of calls).
- Compressing thinking (encrypted).
- Anything that targets output tokens through input compression.

On Lite, even fixes 2 and 3 together (about -5%) only bring distil level with RTK's cold
-6.5%. With n=300 and a tool-output pool of 11.5%, no compressor can show a significant cost
win on short tasks. **The long horizon is where entry compression is worth 5.9x per token. That
is where to prove it, with fix 4 done first.**

## Method and validation

- **Token model.** Block size is estimated from characters by kind (system+tools constant,
  task statement, tool result, tool call, text, thinking signature). It is fitted by relative
  least squares on two sets: the long-horizon plain and rtk tasks (clean caches; final-prompt
  and read equations), and half of the Lite raw transcripts, using only the cache-invariant total
  prompt tokens. Held-out Lite total prompt: median 0.999, p10-p90 [0.93, 1.05]. Long-horizon
  final prompt: median 1.02 [0.93, 1.09]. The fitted system+tools head is 1,805 tokens.
- **Cold re-pricing** keeps each task's exact total prompt tokens and output. Only the
  write/read split moves to the append-only prediction (`correct_fresh`).
- **Replay** runs `compress_messages` exactly as `benchmarks/swebench_outcome/agent.py` calls it,
  with `persist=False` and DISTIL_HOME/HOME pointed at a temp dir. The cache simulation counts a
  read as the longest earlier request that is a block-exact prefix of this one.
- **Where the data is missing.** The Lite plain and distil transcripts do not exist anywhere on
  disk (`swebench-outcome-300-medium` has no `transcripts/` in any worktree). For those arms,
  everything comes from the `usage` totals, and the Lite distil replay runs on the same-day raw
  provider-cm trajectories with no edits applied (193 tasks). Terminal-Bench has no transcripts,
  so its per-request causes are UNVERIFIED.
- **Inferred, not measured (UNVERIFIED):**
  - the `distil_expand` tool cost (~200 tokens, from 2- and 3-step usage);
  - the residual RTK step effect;
  - the cause of the extra Terminal-Bench requests;
  - the effect sizes of fixes 2-6.
