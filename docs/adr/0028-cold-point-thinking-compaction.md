# 0028 — Drop prior-turn thinking at the cold point (opt-in until certified)

- **Status:** accepted (shipped OFF by default)
- **Date:** 2026-10-29
- **Relates to:** ADR 0014 (cold-point recompression), ADR 0022 (certification hold),
  `distil/adapters/anthropic.py` (`cold_thinking_candidates`, `drop_cold_thinking`),
  `tests/test_cold_thinking.py`

## Context

Claude Opus 4.5 and every model numbered 4.6 or higher keep earlier turns' thinking blocks
in context and bill them as input; older models strip them server-side ("Thinking block
preservation by model", platform.claude.com/docs/en/build-with-claude/thinking). With the
default `display: "omitted"` on 4.7+, the blocks arrive with empty text, so the tokens
are billed but invisible to distil's local counter.

Removing them while the prompt cache is warm rewrites the cached prefix (ADR 0014's
measured 2x cost). On a cold turn the prefix is re-written anyway, so it is the one free
moment.

## Decision

1. Thinking blocks ride ADR 0014's evicted set as `th:<sha256(signature)[:16]>` keys,
   so they inherit everything that set already guarantees: decided only on a cold turn
   distil observed itself, re-applied byte-identically on every later turn, persisted
   across restarts, no fresh decision while the drift guard holds, kill switch
   `DISTIL_COLD_POINT=0`.
2. API rules, enforced at render time on every turn (any doubt drops nothing):
   - only a **leading run** of thinking blocks is removed, oldest first. Preserved
     thinking (Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5) chains each block to the
     previous one, and removing from the middle invalidates every later block;
   - never the turn in progress: everything from the last user message with no
     `tool_result` onward. Its thinking must go back "complete and unmodified" during tool
     use, interleaved blocks included. Never the last assistant message either;
   - whole blocks only. A signature is never altered, an assistant message is never
     left empty, and nothing happens if a `compaction` block is present;
   - only on models that bill prior thinking, which is where there is anything to save.
3. **Off by default** (`DISTIL_COLD_THINKING=1` opts in). This is lossy for the model's
   own reasoning, and no shadow or certify evidence covers it yet. The certification
   rule (ADR 0022) needs at least `VERDICT_MIN_AB` paired rows of evidence before a mode
   counts as inside the budget. Once opted in, cold turns sampled by shadow are booked
   under the serving mode, so the per-mode hold watches them.

## Consequences

- `x-distil-cold-evicted` counts thinking keys together with tool ids.
- On `display: "omitted"` traffic, `tokens-saved` under-reports this transform, because
  the dropped text is empty locally while the billed reasoning is not. The provider's
  `usage` is the only true measure.
- To turn it on by default, collect ≥50 paired shadow rows on opted-in cold turns and
  require the paired-harm upper bound to sit inside `BUDGET_ALPHA`.
