# distil vs Headroom 0.42.0 — offline head-to-head (2026-10-29)

$0, offline, no LLM API calls. Code: `benchmarks/headroom_h2h.py`. Aggregates:
`results.json` (this directory). No tool-output content is written here, only counts,
hashes and timings.

## Pre-registered metric definitions

Written and committed **before** any number was computed. If a definition below changes
after the first run, the change is listed under "Deviations" with the reason.

### Inputs (real data, not synthetic)

* **Block samples**: real tool results from local Claude Code transcripts
  (`~/.claude/projects/**/*.jsonl`, read through `distil.whatif.parse_transcript`), plus
  every VOLATILE block of `corpus/*.json` that passes the size filter. Filters: 1,500 to
  40,000 characters; any result already carrying a distil or Headroom marker
  (`handle=`, `<<distil`, `hash=`, `<<ccr:`, `lines omitted`) is excluded (it was
  compressed at capture time and would bias toward whichever tool did that); exact
  duplicates dropped. Up to 60 per content type, `random.Random(0)`.
* **Content type** (assigned before compression, by this fixed rule, first match wins):
  `json` (whole text parses as a JSON object/array) → `html` (≥ 20 HTML tags and an
  `<html|body|div|head|!doctype|table|p>` tag) → `search` (tool Grep/Glob/WebSearch, or a
  shell command starting `grep|rg|ag|find|fd|ls|git grep`) → `code` (Read of a source-file
  extension, or a shell `cat|sed|head|tail|nl|git diff|git show` of one) → `prose` (Read of
  `.md/.txt/.rst`, WebFetch, page-text tools) → `logs` (every other shell/Monitor output).
* **Sessions** (cache metric): 30 real Claude Code main-thread sessions with ≥ 12 model
  requests, first 12 requests each, same marker filter, no tool result > 40,000 chars,
  `random.Random(0)`. Thinking blocks are dropped (identical across arms, opaque to both
  tools). Each request carries `cache_control` on its newest block, as Claude Code sends.
* **Placement of a block sample**: `[user query, assistant tool_use, user tool_result
  (the sample), then 3 × (tool_use, short "ok" tool_result)]`. The sample is therefore
  older than both tools' recency carve-outs (distil: last 2 tool turns; Headroom
  `protect_recent=4` messages), so both defaults are allowed to compress it. The query is
  the real preceding user text (≤ 400 chars). Tool name and input are the real ones.

### Arms

Each tool runs in its DEFAULT configuration and its most aggressive DOCUMENTED one, plus a
lossless tier so like is compared with like.

| tier | distil 1.58.0 (`compress_messages`, `persist=False`) | Headroom 0.42.0 |
|---|---|---|
| lossless | `verbatim=True` (subscription default, Tier-0 only) | `densify()` (`mode="agent"`, verified lossless) |
| default | digest (PAYG default) | library `compress()` default config; **and** the proxy's out-of-box `coding` profile (env seeded with `profile.proxy_env()`) |
| aggressive | digest + cold-point eviction of the sample (`evict=`), the most aggressive documented path | `savings_profile="agent-90"` with its proxy env (target 90%, `force_kompress`) |

Headroom's Kompress model is loaded synchronously (`warm_kompress_model`) before timing,
so no request is measured while the model is still loading in the background.

### Metrics

1. **Token reduction** = `1 − tokens(compressed sample) / tokens(original sample)`, one
   tokenizer for every arm: tiktoken `o200k_base` (Claude's tokenizer is not public; the
   ratio, not the absolute count, is the claim). Reported as the mean of per-sample ratios.
2. **Fact recall** — distil's retention harness (`distil/retention.py`) definitions,
   unchanged: facts are `extract_targets()` (key=number numerics; URLs/paths/hashes/UUIDs;
   error lines) from the ORIGINAL; `retained` = `_survives()` in the compressed text;
   `recoverable` = absent but `_in_recovery()` in the bytes returned by the tool's own
   retrieval path for a handle that the compressed block itself carries (distil:
   `RestoreStore.expand(handle)`; Headroom: `CompressionStore.retrieve(hash)` for `hash=` /
   `<<ccr:…>>` markers); `lost` = neither. **Visible recall** = retained/total; **true
   recall** = (retained+recoverable)/total, per sample, then averaged per content type
   (macro over samples with ≥ 1 fact). A lossy tool with no retrieval store has
   recoverable = 0 by construction.
3. **Prompt-cache friendliness** over the 30 sessions: per request, is the previously
   forwarded request a byte-identical (canonical JSON, `cache_control` stripped) prefix of
   this one? Reported as the fraction of requests 2..12 whose prefix held. **Effective $**:
   every request is priced by `benchmarks/policysim/costmodel.Anthropic` (Anthropic rules:
   writes at breakpoints, 20-block lookback, hits refresh TTL) at Sonnet 5.5 list price
   ($2 in / $10 out per MTok, cache read 0.05×, 5-minute write 1.25×), token counts from
   `o200k_base`, requests 10 s apart (no TTL expiry), messages only (tools+system prefix
   excluded: identical for every arm). Output tokens are not priced (offline: behaviour is
   held fixed). Headline = per-session $ relative to the uncompressed (`plain`) arm.
   Headroom is run both through its proxy's cache-mode delta engine
   (`extract_cache_stable_delta`: replay the previously forwarded prefix, compress only
   the appended delta) and statelessly (whole history each request); distil through its
   served adapter (whole history each request) followed by its proxy's prefix replay
   (`distil.prefixreplay.replay`), which is what `distil proxy` forwards.
4. **Latency** = wall time of one compression call (whole request), p50/p95 per arm, on the
   session requests (realistic request sizes) and on the block samples. **Failure rate** =
   calls that raised, plus Headroom calls that returned its swallowed-exception signature
   (`tokens_before == 0` on a non-empty input).
5. **Reversibility** per block sample: `identical` (output == input), `exact` (some
   retrieval blob for a handle in the block == the original bytes), `decoder` (the tool's
   own shipped decoder proves a round trip; Headroom `_string_pair_is_lossless`; distil ships
   no decoder for its Tier-0 forms), else `facts-intact` (true recall = 1 but no
   byte-exact path) or `lossy`. "Exactly recoverable" = identical + exact + decoder.

### Statistics

95% CIs: percentile bootstrap (10,000 resamples, seed 0) of the mean **paired**
per-sample (or per-session) difference between a distil arm and a Headroom arm of the same
tier. n is reported next to every number. A difference whose CI excludes 0 is called a
win; otherwise "no detectable difference".
