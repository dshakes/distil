# 0027 — Cache-aware entry compression

- **Status:** proposed (draft; offline evidence only, the lossy half needs a live run)
- **Date:** 2026-10-06
- **Relates to:** `benchmarks/policysim/`, `benchmarks/results/policysim/policysim.json`,
  `docs/research/policy-simulator.md`, `distil/adapters/anthropic.py` (`compress_messages`),
  `distil/expand.py` (`EXPAND_TOOL`), ADR 0008, ADR 0011, ADR 0021, ADR 0025, ADR 0026
  (`distil sh`, branch `feat/shell-at-source`)

## Context

Across three benchmarks distil did not lower what coding agents pay. The leading
hypothesis was cache economics: compressing already-cached history saves reads billed at
0.1x and forces writes billed at 1.25x or 2x. On that view distil was optimising tokens,
not cache-aware dollars.

The policy simulator replays real trajectories offline through exact provider cache
rules. Its Anthropic model reproduces billed usage within 5% per component on runs it was
not fitted to. It found four things:

1. **The hypothesis holds for history rewriting, and distil 1.57 no longer does it.**
   Policies that re-render old tool results cost +3% to +303% more than plain, worst on
   Anthropic's 1h TTL. distil 1.57's served adapter never changes a sent byte. Its
   lossless mode is -0.1% vs plain.
2. **Lossless shaping is worth about 0.1%.** That holds for distil Tier-0, `distil sh`
   filters, an rtk-like filter set, and Headroom's append-only mode.
3. **The injected `distil_expand` tool definition costs about 3 points on short tasks.**
   It rides on every request. Recovering through the agent's own shell instead moves
   served digest from -1.9% to -4.9% vs plain on SWE-bench Lite.
4. **Lossy entry compression is where the money is.** It saves 2-16% on SWE-bench and
   10-45% on Claude Code sessions, holding the trajectory fixed. It loses once the agent
   spends more than 0.25-0.78 extra steps per 1,000 tokens removed. The paired live runs
   put that penalty at 1.03 [0.23, 1.80], so the sign of the effect is undetermined.

## Decision

Compression in distil takes one shape, **entry-only and cache-stable**:

1. **A sent byte is never rewritten.** A tool result is rendered once, on the first
   request that carries it, and re-sent byte-identically after that. This is already true
   of 1.57; this ADR makes it the contract rather than a property of the current code,
   and `benchmarks/policysim` is the regression check: the read share must equal plain's.
2. **Recovery goes through the agent's existing shell** (`distil expand <handle>`, as
   `distil sh` prints it), not through a tool definition injected into every request.
   Adding a tool partway through a conversation would bust the cache, since tools come
   first in the prefix, so it is not an option either.
3. **History is re-rendered only when the cache is already cold.** That means the gap
   since the previous request exceeds the TTL in use. On real Claude Code sessions this is
   worth 1-2% before behaviour.
4. **The lossy level ships off by default, behind a live gate.** The candidate is
   must-keep-preserving truncation at entry with shell recovery: head and tail plus every
   error, failure, warning and count line, the rest behind a handle, at k = 2,000 tokens.
   It is promoted only if a live A/B against plain, of about 100 long-horizon tasks,
   measures:
   - lambda below that policy's breakeven: about 0.3 on long-horizon SWE-bench, about 0.6
     on Claude Code;
   - a non-inferior solve rate.

## Consequences

- distil stops paying for a tool definition on every request. That is the one change here
  with a behaviour-free expected saving: about 3% of the bill on short tasks, under 0.2%
  on long ones.
- The default product promise becomes "never costs more than plain", which the simulator
  can check offline on every release. It no longer claims a saving that depends on agent
  behaviour.
- Lossy savings stay unclaimed until measured live. The 1.03 point estimate says they lose
  today, and the lower bound (0.23) says a better-targeted transform might not.
- Rejected:
  - **Bounded-window re-rendering**: rewrites the cached prefix and costs more on every
    corpus.
  - **Explicit Gemini cache objects for agent loops**: the storage and creation charges
    made them 1.6-4.1x implicit caching.
  - **Compressing below a provider's cache minimum**: Gemini's minimum is 4,096 tokens.
    Compressing short Lite prompts under it makes entry-digest +3.7% on Gemini, against
    -14.8% on Anthropic. The policy must not shrink a request across that boundary.
