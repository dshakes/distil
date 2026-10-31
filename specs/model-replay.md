# model-replay: "does the cheaper model make the same decisions on YOUR traffic?"

Status: step 1 implementation — `distil replay-model`, dry-run + live path.
Code: `distil/replay_model.py`; tests: `tests/test_replay_model.py`.
Related: `distil/referee.py` (compressor auditor, same A/B design); `distil/shadow.py`
(live equivalence, same decision signature and Wilson CI).

## Question

Haiku 5.5 solves SWE-bench tasks at ~0.17x Sonnet 5.5's cost per solved task. But that
result is on benchmark traffic. The question this feature answers is: **on your own recorded
traffic, does claude-haiku-5-5 make the same next decision as the model you are already
paying for?** If it does, at some confidence level, you have empirical grounds to switch —
and a per-$/request ratio that shows what you would save.

## Source of requests

**Primary (installed distil)**: `~/.distil/sessions/*.requests.jsonl` — one row per
proxied request, written by the distil proxy. Each row is content-free: model name, billed
token counts (input/cache_read/cache_write/output), compression mode, status code,
timestamp. No prompt text, no response text. These records are used for the **dry-run**:
token counts are sufficient to estimate the cost of replaying each request against a
cheaper model.

**Live replay**: Claude Code session transcripts at `~/.claude/projects/**/*.jsonl`.
Each `assistant` row carries the model's actual response content (tool_use blocks, text)
and the billed usage. The prior conversation context can be reconstructed from the
parent-uuid chain. Content stays local: the live path calls the target model with the
reconstructed context, compares the decision signature, and reports only the equivalence
rate and token/dollar counts — no content is persisted or reported.

**Why not the restore store?** `~/.distil/restore/<handle>` holds compressed content
blocks in the DSTL1 binary format. Reconstructing full request bodies from them requires
decompressing every handle for a receipt, reassembling the message list, and mapping back
to the original structure — expensive and fragile. CC transcripts carry the same
information in the standard API message format.

## Equivalence definition

A pair (original, replay) is **equivalent** if `decision_signature(original_response) ==
decision_signature(replay_response)`. The signature is `tool:<name>:<args_hash>` for a
tool call, `text` for a plain text reply, `none` when the response structure cannot be
read. This is the same signature used by shadow mode (`distil/shadow.py`) and the referee
(`distil/referee.py`).

**What this measures**: whether the cheaper model calls the same tool (or abstains) in the
same turn, not whether it produces identical output text. For agentic workflows the next
tool call is the decision that matters: a wrong tool choice derails the task; different
words in a text reply usually do not.

**Noise**: at temperature > 0 the same model gives different answers on repeat. This step
does not run a control A/A' arm (that is referee's design); the raw equivalence rate
therefore conflates model-switch effects with sampling noise. If the rate looks low, run
`distil shadow-stats` to see the model's own self-agreement before attributing the
difference to the model switch.

## Statistics

Wilson score interval at 95% confidence (`distil.shadow.wilson_ci`) on the fraction of
turns where the cheaper model's decision signature matched the original. The same
estimator shadow-stats reports, so numbers are directly comparable.

## Cost model

`claude-haiku-5-5` uses a two-tier price:
- Prompts **≤ 100 000 input tokens**: $0.10 input / $0.50 output per Mtok.
- Prompts **> 100 000 input tokens**: every token billed at **5x** ($0.50 / $2.50 per Mtok).

The threshold applies to the TOTAL prompt (input + cache_read + cache_write). Cache reads
and writes use the standard Anthropic multipliers (1.25x write, 0.10x read) at the tier
rate.

The dry-run cost estimate uses each request's recorded billed token breakdown from
`sessions/*.requests.jsonl` to compute what that exact request would have cost against the
target model. This is an upper bound on real haiku-5-5 cost because the prompt passed
through distil compression before the original call — a shorter prompt that the cache
would still see as novel for haiku-5-5.

## Privacy

No prompt content, tool arguments, or response text is written to any file or reported in
output at any time. The sessions store is already content-free. The live path reads CC
transcripts to reconstruct context in memory only; the report contains only: equivalence
rate, CI bounds, n, model names, token counts, and dollar estimates.

## Non-goals (step 1)

- ~~No A/A' noise-control arm~~ — superseded: the shipped command runs an A' subsample
  (the original model replayed on the same context) so sampling noise is separated from
  model-switch signal, sized for a Wilson half-width within the equivalence margin rather
  than a full third arm.
- No automatic switching recommendation: the feature reports evidence, the human decides.
- No persistent ledger: results are printed and the command exits. Add persistence if
  users want rolling accumulation.
- No Bedrock / Vertex routing for the live path: the replay goes to `api.anthropic.com`
  directly (same as the SWE-bench harness). Add routing when there is demand.
