# Policy simulator: compression policies priced against real trajectories, offline (2026-10-06)

`benchmarks/policysim/` replays real agent trajectories request by request through a
compression *policy* (which bytes each request would have carried) and an exact provider
prompt-cache cost model (what those bytes would have been billed). It costs $0: no model
or API call is made. Aggregates are in `benchmarks/results/policysim/policysim.json`
(counts and dollars only, no transcript content).

```
python -m benchmarks.policysim \
  --lite  <scoreboard>/benchmarks/results/swebench-outcome-300-h2h \
  --long  <longh>/benchmarks/results/swebench-verified-hard-max \
  --out benchmarks/results/policysim --claude-code 20 \
  --headroom-python <venv with headroom-ai==0.40.0>/bin/python
```

## The answer

1. **The simulator reproduces Anthropic's bill.** Fitted on half of the SWE-bench Lite
   trajectories, it predicts the long-horizon run it never saw within **-2.5% in dollars**
   (cache write -1.4%, cache read -4.7%), and the *distil* arm, replayed through the served
   adapter, within **-2.8%** (write -2.3%, read -4.9%). Section 2.
2. **Rewriting cached history is the expensive mistake, and distil 1.57 no longer makes
   it.** Policies that re-render old tool results cost +3% to +303% more than plain, worst
   on Anthropic with a 1-hour TTL (writes at 2x). distil 1.57's served adapter, replayed
   step by step, never changes an already-sent byte: `distil-lossless` is -0.1% vs plain.
3. **Nothing lossless saves money on these trajectories.** Every cache-stable lossless
   policy (distil Tier-0, `distil sh`'s filters, our rtk-like) lands within 0.2% of plain.
   On 236 bash commands run byte-identically in the rtk and raw arms, RTK's output was
   **2.0% smaller** (chars): RTK's live Lite result is not output filtering.
4. **The injected `distil_expand` tool is the largest behaviour-free cost on short
   tasks.** The same digest with recovery through the agent's own shell
   (`distil expand <h>`, as `distil sh` does) instead of a tool definition on every
   request: Lite **-1.9% -> -4.9%** vs plain.
5. **Lossy entry compression is the only lever with real money, and its sign depends on
   behaviour we cannot see offline.** Same-trajectory savings on Anthropic: 2-15% (Lite),
   2-16% (long horizon), 10-45% (Claude Code sessions). Each stops beating plain once the agent spends
   more than **0.25-0.78 extra steps per 1,000 tokens removed** (breakeven lambda). The
   paired live runs put distil's lambda at **1.03 [0.23, 1.80]**. At that value every lossy
   policy loses on every corpus and provider. Live confirmation is required.
6. **Recommended: cache-aware entry compression** (ADR 0027). Never rewrite a sent byte,
   shape each tool output once at entry, recover through the shell, not a tool. Ship the
   lossless form now: about 0.1% better than plain, plus removing the per-request tool
   overhead. Gate any lossy level on a live A/B that measures lambda directly.

## 1. Method

**Trajectory.** A trajectory is a run's untransformed message history plus the index each
model call ended at. The SWE-bench harness stores history *before* its arm transform, so
the provider-cm and selective arms' transcripts are raw tool output (600 Lite
trajectories, 3,519 requests). The plain and distil long-horizon transcripts give 15
trajectories and 1,001 requests at effort `max`. 20 local Claude Code sessions (2,088
requests, capped at 150 each) are read through `distil.whatif.parse_transcript`.

**Policy.** `step(history, t) -> messages`. A policy keeps its own state and never sees the
cost model. Adding one means writing a class or an `EntryPolicy(shaper)` and registering it
in `policies.build()`.

**Cost model** (`costmodel.py`, rules quoted from the provider docs in its docstring):

- **Anthropic.** Entries are written only at breakpoints, as a hash of the prefix up to
  that block. Reads look back at most 20 positions; a run of `tool_use` or `tool_result`
  blocks counts as one position. Per-model minimum (512 tokens for Sonnet 5.5). Write 1.25x
  for 5m, 2x for 1h; read 0.1x (0.05x on Opus 5.5). A hit refreshes the TTL from the start
  of the request. `input_tokens` = tokens after the last breakpoint.
- **OpenAI.**
  - GPT-5.6+ (gpt-5.6-terra, $2.00 / $0.20 / $12.00): exact-boundary prefix caching,
    minimum 1,024, writes 1.25x, reads 0.1x, 30-minute TTL.
  - Pre-5.6 (gpt-5.4, $2.50 / $0.25 / $15.00): cached length floored to 128s, no write
    charge.
- **Gemini 3.1 Pro.**
  - Implicit caching: minimum 4,096, cached input $0.20 vs $2.00, no write charge.
  - Explicit cache objects: storage $4.50 / Mtok / hour, a new object every 4 requests.
  - The implicit cache's lifetime (assumed 300 s) and hit rate (assumed 1.0, an upper
    bound) are not documented. Explicit creation is assumed to bill at the input rate.

The same token counts are priced under every provider. Their tokenizers differ, so the
cross-provider *ratios* are more reliable than the absolute dollars.

**Token model.** Claude's tokenizer is not public, so the token count is
`scale x regex-pieces + per_block`, plus a constant for tools and system. It is fitted by
least squares to billed **total** input per trajectory: uncached + cache write + cache
read. The docs define that sum as every input token the request carried, so it does not
depend on what happened to be cached. Thinking text is not in the transcripts (only its
signature), but it is re-sent and billed as input. Billed output minus the visible
assistant tokens fixes each trajectory's thinking total exactly. It is split across blocks
in proportion to signature length.

**Information removed** is measured per tool result at its most-compressed rendering: the
tokens a policy dropped beyond distil's lossless Tier-0. A **violation** is any line the
keep policy marks must-keep (an error, failure, warning or count summary) whose text is
absent from the rendering.

## 2. Calibration (error vs billed usage)

Fitted parameters: 1.57 tokens per regex piece, 28.8 per block, 1,644 tokens of
tools+system, 2.47 uncached framing tokens per request. The fit used 151 Lite rtk-arm
trajectories; rtk sends its transcript bytes unchanged.

Error = (simulated - billed) / billed, summed over the set (median per-trajectory |error|):

| set | n | cache write | cache read | uncached | $ |
|---|---|---|---|---|---|
| Lite train, priced cold | 151 | +62.3% (90.0%) | -12.7% (20.6%) | -0.0% (7.5%) | +19.5% (27.8%) |
| Lite test, priced cold | 149 | +53.8% (80.6%) | -11.0% (17.1%) | +1.1% (7.5%) | +17.0% (27.2%) |
| Lite test, first request pre-cached | 149 | -17.8% (21.0%) | +1.8% (4.3%) | +1.1% (7.5%) | -5.9% (7.5%) |
| **long horizon plain+rtk, never fitted** | 17 | **-1.4%** (1.6%) | **-4.7%** (4.7%) | +21.5% | **-2.5%** (2.6%) |
| **long horizon distil arm, served adapter replayed** | 7 | **-2.3%** (3.6%) | **-4.9%** (4.2%) | +21.5% | **-2.8%** (1.6%) |

The uncached component is about 2 tokens per request, so its +21.5% error is under $0.0001
per task.

**The Lite cache split is not a cold run.** Priced cold, Lite writes are 54% too high. If
the first request (system + tools + task) was already in the cache, the error falls to
-18% and reads match to +1.8%. A mixture of about 75% warm tasks fits both. This matches
the driver's finding: rtk, selective and provider-cm ran the same day with byte-identical
first requests and read each other's cache. 190/300 rtk rows are impossible for a cold
run; it is also in `docs/research/why-rtk-wins.md`. The fit is unaffected, because it uses
the cache-independent total. **All policy results below are priced cold:** each
trajectory pays its own first write.

## 3. Results

Each cell is **$ vs plain, same trajectory / with the fitted re-run penalty (mid)**. Pure
cost replays the identical trajectory. The penalty adds
`lambda x (removed ktok) x mean $ per step`. Plain's $ per task is shown under each
corpus. Full tables, including the high and stress columns, breakeven lambda, violations,
read share and every grid point, are in the JSON.

### SWE-bench Lite (600 trajectories; plain: Anthropic $0.0342, Gemini explicit $0.0676, Gemini implicit $0.0427, OpenAI 5.4 $0.0433, OpenAI 5.6 $0.0370)

| policy | Anthropic | OpenAI 5.6 | OpenAI 5.4 | Gemini implicit | Gemini explicit |
|---|---|---|---|---|---|
| distil-lossless (served, 1.57) | -0.1 / -0.0 | -0.0 / -0.0 | -0.0 / -0.0 | -0.0 / +0.0 | -0.1 / -0.1 |
| distil-digest (served) | -1.9 / +5.7 | -1.7 / +5.9 | -1.6 / +5.9 | -1.5 / +6.6 | -2.1 / +5.2 |
| distil-digest, shell recovery | -4.9 / +2.4 | -4.5 / +2.8 | -4.1 / +3.1 | -0.9 / +7.1 | -5.1 / +1.8 |
| rtk-like (`distil sh` sh-v1 filters) | -0.2 / +0.1 | -0.1 / +0.1 | -0.1 / +0.1 | -0.1 / +0.2 | -0.1 / +0.1 |
| headroom 0.40.0 (real package) | -0.7 / +0.4 | -0.7 / +0.5 | -0.6 / +0.5 | -0.1 / +1.2 | -0.8 / +0.3 |
| entry-lossless | -0.0 / -0.0 | -0.0 / -0.0 | -0.0 / -0.0 | +0.0 / +0.0 | -0.1 / -0.1 |
| entry-digest, shell recovery | -14.8 / +5.8 | -13.7 / +7.1 | -12.6 / +8.3 | +3.7 / +29.6 | -15.6 / +3.7 |
| trunc-500, shell recovery | -5.3 / +2.5 | -4.9 / +2.9 | -4.5 / +3.2 | -0.3 / +8.4 | -6.5 / +0.8 |
| trunc-2000, shell recovery | -0.2 / +0.1 | -0.1 / +0.1 | -0.1 / +0.1 | -0.1 / +0.1 | -0.2 / -0.0 |
| cold-only digest | +3.0 / +3.0 | +2.8 / +2.8 | +2.5 / +2.5 | +0.0 / +0.0 | +3.5 / +3.5 |
| window w16 b16 (rewrites history) | +3.5 / +4.0 | +3.1 / +3.6 | +2.8 / +3.3 | +0.2 / +0.8 | +4.2 / +4.7 |

### Long horizon (15 trajectories, effort max; plain: Anthropic $2.84, OpenAI 5.6 $3.05, OpenAI 5.4 $3.72, Gemini implicit $2.97, Gemini explicit $7.01)

| policy | Anthropic | OpenAI 5.6 | OpenAI 5.4 | Gemini implicit | Gemini explicit |
|---|---|---|---|---|---|
| distil-lossless | -0.1 / -0.0 | -0.1 / +0.0 | -0.1 / +0.0 | -0.1 / +0.0 | -0.1 / -0.0 |
| distil-digest | -3.7 / +12.5 | -3.5 / +13.0 | -3.4 / +13.1 | -3.4 / +13.1 | -4.8 / +10.1 |
| distil-digest, shell recovery | -3.9 / +12.3 | -3.6 / +12.8 | -3.5 / +13.0 | -3.5 / +13.0 | -5.0 / +9.8 |
| rtk-like | -0.1 / +0.1 | -0.1 / +0.1 | -0.1 / +0.1 | -0.1 / +0.1 | -0.2 / +0.0 |
| headroom | -0.8 / +1.7 | -0.8 / +1.8 | -0.8 / +1.8 | -0.7 / +1.8 | -1.1 / +1.3 |
| entry-lossless | -0.1 / -0.1 | -0.1 / -0.1 | -0.1 / -0.1 | -0.1 / -0.1 | -0.1 / -0.1 |
| entry-digest, shell recovery | -16.2 / +33.9 | -15.1 / +36.5 | -14.8 / +37.0 | -14.7 / +37.2 | -21.5 / +21.9 |
| trunc-500, shell recovery | -9.8 / +22.3 | -9.2 / +23.7 | -9.0 / +23.9 | -9.0 / +24.0 | -13.0 / +15.6 |
| trunc-2000, shell recovery | -1.6 / +4.2 | -1.5 / +4.5 | -1.5 / +4.5 | -1.5 / +4.5 | -2.2 / +3.3 |
| cold-only digest | +0.2 / +0.2 | +0.1 / +0.1 | +0.2 / +0.2 | +0.1 / +0.1 | +0.2 / +0.2 |
| window w16 b16 | +16.6 / +72.2 | +6.0 / +58.3 | +3.3 / +54.5 | +3.3 / +54.5 | +19.2 / +72.3 |

### Claude Code (20 local sessions, input side only, real timestamps, 1h TTL; plain: Anthropic $12.37)

| policy | Anthropic | OpenAI 5.6 | OpenAI 5.4 | Gemini implicit | Gemini explicit |
|---|---|---|---|---|---|
| distil-lossless | -0.1 / -0.1 | -0.1 / -0.1 | -0.1 / -0.1 | -0.1 / -0.1 | -0.1 / -0.0 |
| distil-digest | -9.5 / +6.4 | -9.2 / +6.8 | -9.3 / +6.7 | -9.1 / +7.1 | -8.3 / +8.0 |
| headroom | -0.8 / +0.3 | -0.9 / +0.3 | -0.9 / +0.3 | -0.8 / +0.3 | -0.7 / +0.4 |
| entry-digest, shell recovery | -24.0 / +11.9 | -23.2 / +13.2 | -23.3 / +13.1 | -23.2 / +13.7 | -20.1 / +18.0 |
| trunc-500, shell recovery | -45.1 / +23.8 | -47.8 / +17.2 | -47.7 / +17.4 | -47.5 / +18.4 | -52.2 / +7.4 |
| cold-only digest | -1.6 / +3.5 | -1.4 / +3.8 | -1.4 / +3.8 | -1.3 / +3.9 | +0.7 / +6.0 |
| window w16 b16 | **+82.0** / +155.1 | -0.7 / +39.9 | -3.7 / +35.6 | -5.5 / +33.2 | +13.7 / +59.9 |

Provider-specific findings:

- **History rewriting is uniquely expensive on Anthropic.** The window policy re-renders
  old results every 16 requests. On Claude Code sessions it costs **+82%** on Anthropic,
  where Claude Code writes at the 1h rate of 2x, and roughly break-even on OpenAI and
  Gemini, where a miss costs only the uncached rate.
- **Gemini's 4,096-token minimum punishes compression on short tasks.** On Lite,
  entry-digest is -14.8% on Anthropic but **+3.7%** on Gemini implicit. Compressed Lite
  prompts (~5k tokens) fall under the minimum and stop caching at all.
- **Gemini explicit caching costs 1.6-4.1x what implicit does** on agent loops, because of
  the per-object creation and storage charges. Its relative savings from compression are
  the largest, because compression also shrinks the storage charge.
- **cold-only rewrites earn money only where caches actually expire.** On Claude Code
  sessions, with real idle gaps, it is -1.6%. On the harness, where requests are seconds
  apart, it is +0.2% to +3%: the expand tool's overhead with no rewrite to pay for it.

## 4. Behaviour: the re-run penalty

Offline replay holds the trajectory fixed, but a policy that removes information can cost
extra steps (re-runs, `distil_expand` calls). The model is linear: `lambda` extra steps
per 1,000 tokens removed. It is fitted from the paired live runs as
`sum(steps_arm - steps_plain) / sum(removed ktok)`, with a task bootstrap. This is an
association across tasks, not a causal estimate.

| source | n | lambda | 95% CI |
|---|---|---|---|
| distil, long horizon (served adapter replayed on its own trajectories) | 6 | 1.24 | [0.40, 2.21] |
| distil, Lite (replay on another arm's raw trajectory of the same task: a proxy) | 299 | 0.85 | [-0.36, 2.10] |
| **distil pooled: the band (mid, high)** | 305 | **1.03** | **[0.23, 1.80]** |
| selective-context, Lite (irrecoverable deletion): stress | 300 | 2.56 | [2.20, 2.95] |
| provider context editing, Lite (clears old results; per-call mean) | 300 | 0.27 | [-0.14, 0.62] |

The **breakeven lambda** is the penalty at which a policy stops beating plain:

- Lite: 0.57-0.78 for every lossy policy, except served distil-digest at 0.25, which also
  pays the tool overhead.
- Long horizon: 0.24-0.34. rtk-like is 0.70, but on only 130 tokens removed.
- Claude Code: 0.33-0.73.

All of these lie inside distil's 95% CI but below its point estimate. So the sign of every
lossy policy's effect is *not determined* by the data we have. Two things are worth
noting:

- provider-cm removes *old* results and has a much smaller lambda than selective, which
  deletes phrases from the newest output.
- Entry compression shows the agent a compressed *newest* output. distil's adapter instead
  keeps recent turns verbatim. Entry compression's true lambda may therefore be higher than
  distil's.

Only a live A/B can settle it.

## 5. The search and the recommended policy

The grid:

- distil served in both modes, with tool or shell recovery;
- entry-lossless, entry-digest (tool or shell), rtk-like;
- trunc-k for k in {500, 1k, 2k, 4k, 8k} x {tool, shell} recovery;
- window W in {4, 8, 16} x batch B in {4, 8, 16};
- cold-only;
- headroom.

That is 34 policies x 5 providers x 3 corpora. Exhaustive search over a grid this small
beats a Bayesian optimiser: it is exact and costs about 25 minutes on a laptop.

The objective was $ per task under the **high** penalty, subject to the fidelity
constraint: no must-keep line dropped, and everything dropped recoverable by a handle. The
winner is **entry-lossless** on every corpus and every provider. The exception is Lite on
OpenAI and Gemini, where distil-lossless or plain wins; all three are within 0.1% of each
other. Its effect is -0.0% to -0.1% vs plain, and -0.1% to +0.15% vs rtk-like.

The Pareto frontier ($ saved vs information removed), on the long horizon with Anthropic,
runs in order:

1. entry-lossless
2. rtk-like
3. trunc-4000-sh
4. headroom
5. trunc-2000-sh
6. distil-digest-sh
7. trunc-1000-sh
8. trunc-500-sh
9. entry-digest-sh

Lite and Claude Code have the same shape; the JSON lists every corpus and provider. Each
point further along is worth more pure $ and needs a smaller lambda to stay ahead.

So the recommendation is a *policy shape* plus a *gate*:

- **Ship (behaviour-free):**
  - Cache-stable, entry-only shaping, with the distil 1.57 invariant that no sent byte is
    ever rewritten.
  - Recovery through the agent's shell instead of the injected `distil_expand` tool. On
    Lite that alone moves distil-digest from -1.9% to -4.9%.
- **Gate (needs live lambda):** the lossy level.
  - The candidate is **trunc-k with shell recovery**. It has zero must-keep violations by
    construction (entry-digest drops some repeated error shapes) and the highest
    breakeven per token removed.
  - k = 2,000: -0.2% on Lite, -1.6% long horizon, -38% on Claude Code, pure.
  - It wins only if the live lambda is below ~0.3 on long-horizon SWE-bench and below ~0.6
    on Claude Code.
  - A live A/B of trunc-2000-sh vs plain at about 100 long tasks measures lambda directly.

## 6. Headroom (headroom-ai 0.40.0, the real package)

The worker reproduces the Headroom proxy's default *cache mode*. For details see
`benchmarks/policysim/headroom_worker.py`; paths below are in the headroom source.

- **It is append-only.** The previously forwarded prefix is replayed byte-for-byte
  (`cache/prefix_tracker.py` `extract_cache_stable_delta`). Only the new delta is
  compressed, through `headroom.compress(..., frozen_message_count=N)`. Its read share
  matches plain's on every corpus, so it never busts the cache, the same property as the
  entry policies here.
- **Its cost-level extras over distil** are all in the proxy, not the library:
  - an opt-in recompaction of the whole prefix only when the cache is already cold
    (`HEADROOM_COLD_RECOMPACT`, `transforms/cold_prefix.py`): the `cold-only` policy here;
  - an opt-in net-cost gate that decays the probability the cache is alive with idle time
    (`HEADROOM_NET_COST_POLICY`);
  - provider-confirmed freezing from `cache_read_input_tokens`;
  - an offline TTL learner (`cache/ttl_estimator.py`);
  - a 5m-vs-1h breakeven analysis (`pricing/cache_ttl.py`): 1h pays only if more than
    39.5% of writes recur after a 5-60 minute gap. It does not choose the TTL; it adopts
    the client's.
- **Measured here:** -0.7% to -0.9% pure, which turns positive under the fitted penalty.
  It shows 0.07 (Lite) to 2.13 (long horizon) must-keep-line violations per task. Some of
  these may be lossless reformatting that our substring check counts as a drop. Its token
  counts used its character estimator (no tiktoken vocabulary offline), and ONNX/ML
  compression was not exercised.

## 7. Limitations

- **Token counts are a fitted model**, not Claude's tokenizer. The long-horizon error is
  under 5% per component. Lite per-trajectory error is about 17% (median) on reads. The
  same counts are priced under OpenAI and Gemini rules, whose tokenizers differ.
- **Request timing is uniform** over each harness task's wall time; there are no per-step
  timestamps. TTL expiry is therefore modelled exactly only on Claude Code sessions.
- **Claude Code sessions are input-side only.** Output is not reconstructed, and the
  system prompt and tools are not in the transcript (overhead 0). That makes absolute $
  low and relative savings high for the session corpus.
- **Gemini's implicit-cache lifetime and hit rate are undocumented.** They are assumed to
  be 300 s and 1.0, an upper bound on its caching.
- **rtk-like is distil's sh-v1 filter set, not RTK.** RTK's binary wraps the command and
  cannot be applied to recorded output offline. Its measured output reduction on identical
  commands (2.0%) is consistent with the near-zero rtk-like result.
- **lambda is an across-task association.** It comes from small or proxy samples: the
  long-horizon n is 6, and Lite uses replay on another arm's trajectory. It is linear, and
  it is borrowed from distil's recency-exempt digest for policies that compress the newest
  output.
- **$ per solved task needs solve rates** that offline replay cannot produce. Every
  behavioural claim here is UNVERIFIED until run live.

## Reproduce / extend

- `tests/test_policysim.py` covers:
  - hand-computed Anthropic costs (lookback, minimum, TTL, 1h);
  - OpenAI 128-token rounding and Gemini's minimum;
  - the plain-policy identity on a fixture;
  - a property: entry policies never change a sent byte;
  - fit recovery of known parameters.
- To add a policy, implement `Policy.step` and add a `Spec` in `policies.build()`.
- To add a provider, implement `request(segs, t, ttl) -> Usage` and add it to
  `costmodel.provider()`.
