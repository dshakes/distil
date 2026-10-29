# Lever sizing: what else could save coding-agent users money (2026-10-11)

$0, offline. Same machinery as [policy-simulator.md](policy-simulator.md): the calibrated
Anthropic prompt-cache cost model replays recorded trajectories, plus plain arithmetic on
the billed `usage` of every local Claude Code session. Aggregates are in
`benchmarks/results/lever-sizing/levers.json`; the code is `benchmarks/policysim/levers.py`.

```
python -m benchmarks.policysim.levers \
  --lite  <scoreboard>/benchmarks/results/swebench-outcome-300-h2h \
  --long  <longh>/benchmarks/results/swebench-verified-hard-max --claude-code 20
```

**Every number replays the same trajectory with behaviour held fixed.** None of them says
what an agent would do differently; anything that changes what the model sees or produces
needs a live run before it is a saving.

## Corpora

| corpus | what | plain bill | output share |
|---|---|---|---|
| Lite | 600 SWE-bench Lite trajectories, Sonnet 5.5, simulated cold, 5m TTL | $0.0342 / task | 41.4% |
| Long horizon | 15 SWE-bench Verified-hard trajectories, effort max | $2.84 / task | 37.9% |
| Claude Code (billed) | 1,697 local sessions, 37,997 main-thread responses, 109 days, priced at list | $9,334 total ($5.50 / session) | 6.6% |
| Claude Code (simulated) | the policysim 20 sessions, input side only | $13.86 / session at 5m | n/a |

The Claude Code bill is 63.1% cache reads and 30.1% cache writes. Its mix is 55% Opus 5,
21% Fable 5.1, 21% Opus 5.5. The dollars are list-price equivalents, not what a
subscription user pays.

## Ranked levers

The percentages are of the plain bill for the corpus. "Measured" means computed from the
data with behaviour held fixed. "Upper bound" means price arithmetic that assumes the
outcome does not change.

| # | lever | Lite | Long | Claude Code | kind | what would prove it live | rough cost |
|---|---|---|---|---|---|---|---|
| 1 | **Cache breakpoint on tools+system (1h), conversation tail at 5m**, so later tasks read the prefix | **-11.0%** ($0.0038/task) | -0.1% | ~0: Claude Code already does this (1,619 of 1,697 first requests read the cache; first-request writes are 1.6% of the bill) | measured | ~50 back-to-back Lite tasks with the extra breakpoint; check `cache_read` on first requests | ~$2 |
| 2 | **Keep-alive ping** every 55 min while a session is idle, up to a horizon, so a resume reads instead of re-writing | n/a (no idle) | n/a | rewrites after >1h idle are **17.8%** of the bill. Net of ping cost: **-2.8% (2h), -5.1% (4h), -6.8% (8h, ~$630), -1.7% (24h)** | measured, arithmetic | write a large context, ping at 55 min, resume at 2h; confirm `cache_read_input_tokens` | <$5 |
| 3 | **TTL choice** | 1h everywhere: **+25.4%**; mixed (row 1): -11.0% | 1h everywhere: +8.4% | 1h is already worth it: all-5m would be **+46%**; all-1h +1.4% vs actual | measured (Claude Code side: arithmetic on real gaps) | none needed (cost-only, deterministic) | $0 |
| 4 | Output reduction by X% (10/20/30) | -4.1 / -8.3 / -12.4% | -3.8 / -7.6 / -11.4% | -0.7 / -1.3 / -2.0% | **hypothetical**: harm to solve rate is unmeasurable offline | paired A/B on solve rate and steps (lower effort, terser prompts): Lite 2x150 tasks, Long 2x50 | ~$15 Lite; ~$300 Long |
| 5 | Tool-output compression (policysim, Anthropic): lossless (distil / entry-lossless) | -0.0% | -0.1% | -0.1% (input side only) | measured | already done | — |
| 5b | lossy compression: entry-digest-sh / trunc-2000-sh | -14.8% / -0.2% | -16.2% / -1.6% | -24.0% / -37.9% (input side only) | measured pure. Sign depends on lambda: breakeven 0.25–0.78 vs measured 1.03 [0.23, 1.80] | live A/B of trunc-2000-sh at ~100 long tasks (policysim §5) | ~$600 |
| 6 | Whole-run model swap (same tokens) | Haiku 4.5: -32.5%; Opus 5.5: +83.9% | Haiku: -49.9%; Opus 5.5: +52.0% | Opus 5.5: -38.3%; Sonnet 5.5: -51.2%; Haiku: -75.6% vs actual mix | **upper bound**: solve rate unknown | SWE-bench solve-rate A/B per model | Lite on Haiku ~$7; Long on Opus 5.5 ~$65 |
| 7 | Per-step router: "cheap-looking" steps to Haiku (oracle) | 44% of steps / 41% of $. Cache-blind -20.4%; **cache-aware +10.0%** | 12% / 8% of $. Cache-blind -4.0%; **cache-aware +15.0%** | 32% of responses / 27.5% of $. Cache-blind -20.5% (cache-aware not computable) | upper bound; cache-aware is measured and **loses money** | only worth testing if the router keeps one model per cache window | — |
| 8 | Batch API (50% off) | 0% batchable | 0% | 0% | not applicable | deployment question: no data here | — |

## Method, per lever

**1–3, cache strategy.** `costmodel.Anthropic.request` now takes a TTL per breakpoint. The
API allows mixed TTLs, provided the longer TTL comes first. Writes are split by the
breakpoint whose span they fall in. `stream()` replays a corpus as one user running its
tasks back to back, with one shared cache and idle gaps of 0, 2, 10 and 30 minutes. Arms
are streamed separately, because their first requests can be byte-identical.

- **Row 1** applies only to harnesses that, like the SWE-bench one, mark only the newest
  block (`agent.with_cache_breakpoint`). Their tools+system prefix (1,644 tokens) is
  re-written cold by every task.
  - A 5m prefix breakpoint keeps the full -11.0% only while tasks start within 5 minutes
    of each other. At a 10-minute gap the saving falls to zero.
  - A 1h prefix keeps the -11.0% at every gap tested.
  - 1h on the tail as well costs +7.2%.
- **Transferability of row 1.** The harness's cache-namespace nonce (`agent.system_prompt`,
  one per run per arm) blocks only *cross-arm* sharing; tasks within an arm share the
  prefix. In the real serial run, one arm's consecutive tasks started a median **20.6 s**
  apart on Lite and **1,416 s** apart on Long, so only the 1h form would have hit on Long.
  - For real users, the saving applies to custom agent loops and SDK apps with a stable
    system+tools prefix of at least 512 tokens and short tasks.
  - It does not apply to Claude Code, which already caches the prefix across sessions.
- **Row 2** is arithmetic on billed usage.
  - **Rewrites.** Responses more than 1h after the previous one in the same session are
    94% cache writes, so these are real expiries. Responses 5m–1h apart are 98% reads,
    which shows the 1h TTL is working.
  - **The keep-alive case.** Each ping reads the whole context at the read rate (0.05x on
    Opus 5.5). A resume inside the horizon then reads instead of re-writing.
  - **Waste.** The pings of a resume beyond the horizon, and of every session's final idle
    period up to the horizon, are counted as pure waste.
  - **What is ignored.** A ping's own output tokens, and any context change from Claude
    Code's own compaction.
  - **Open question.** Whether automated pings are acceptable for subscription users is a
    product and ToS question, not answered here.
- **Row 3 (Claude Code).** The all-5m counterfactual turns every read made more than 300 s
  after the previous response into a 1.25x write.
- **Rewrites on the harness.** Within-task expiry rewrites are 0 by construction there:
  requests are seconds apart and spaced uniformly.

**4, output.** The share comes from the simulated bill: Lite and Long cold, Claude Code
billed. The sensitivity is linear in the share. It is a lower bound on the $, because
shorter output is also re-sent as shorter input later, and that is not counted. Fewer
output tokens could cost extra steps; that cannot be seen offline.

**6–7, models.** Row 6 re-prices the same trajectory with each model's prices and cache
rules from `distil/pricing.py`. Haiku 4.5's 4,096-token cache minimum leaves most short
Lite prompts uncached, which is why it saves 32% there, not 50%.

Row 7's proxy for a "cheap-looking" step:
- the reply is exactly one `tool_use`;
- it has no prose;
- it is at most 300 output tokens, thinking included.

The proxy is an oracle: it reads the reply it would have to predict. The cache-blind
figure prices those same steps' token split at Haiku rates. The cache-aware figure gives
Haiku its own cache, since caches are per model, so every switch re-writes the context
on Haiku. That turns the saving into a **loss** on both harness corpora.

**8, batch.** Every request in these workloads waits for the previous response. None can
go through a batch API with a turnaround of up to 24 hours. Batch fits only independent
calls, such as offline evals, LLM judges and bulk summarisation, and none of those are in
this data. **UNVERIFIED:** whether batch and cache discounts stack for such a workload.

## Caveats

- Token counts are the policysim fitted model. Long-horizon error is within 5% per
  component; the per-trajectory Lite error is about 17%. The Claude Code billed figures
  use real usage and need no token model.
- **Lite's share for system+tools is 16.7% here, not the 29.7% quoted in the brief.** Here
  it is computed as one 1.25x write plus a 0.1x read per later request, as a share of
  total $. 16.7% is 28.5% of the non-output bill, so the 29.7% figure is probably a share
  of input. UNVERIFIED.
- The two Claude Code TTL estimates disagree:
  - simulated, 20 sessions: 1h is 7.6% cheaper than 5m, input side only;
  - billed arithmetic, 1,697 sessions: 5m would be 46% dearer.

  The simulator caps each session at 150 requests and uses the oldest-sorted 20, so it
  misses the long, idle-heavy tails. The billed figure is the better one, but it treats
  each response's gap as the cache age.
- The Claude Code dollars are this one user's sessions at list price. That is one
  workload, not a population.
- Whether the cheap-step proxy picks out steps a smaller model would get right is not
  validated. Rows 6–7 are ceilings until a solve-rate run.
