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

### Deviations from the pre-registration (made before the final run, reasons given)

1. **Headroom profile arms go through the real proxy pipeline.** Pre-registered as
   `compress(savings_profile=…)` with the profile's env seeded. A probe showed the library
   call does not deliver a profile's router knobs (`force_kompress`, `max_items_after_crush`,
   `min_chars_for_block`, …: Headroom's own `agent_savings.py` says so), and the two paths
   gave different outputs on the same input. `headroom-coding` (the proxy's out-of-box
   profile) and `headroom-agent90` therefore call `HeadroomProxy(_proxy_config_from_env())
   .anthropic_pipeline.apply(...)` with the exact kwargs the Anthropic handler passes
   (`proxy_pipeline_kwargs(config)`, `prefix_replay_guaranteed=True`, `frozen_message_count`,
   `context=extract_user_query`). `headroom-default` stays the library `compress()` default.
   Made AFTER the first run's library-path profile numbers had been seen (for the record:
   `coding` via the library = 6.3% mean reduction, 99.3% true recall; `agent-90` via the
   library = 16.8% reduction, 80 facts lost); motivated by the
   fidelity gap, not by those numbers.
2. A session's 12th request needs the assistant reply after it to be recognised as a
   request; the first collection kept 11. Fixed before any Headroom number existed.

## Setup actually run

* distil-llm 1.58.0 (this branch, base `d751c73`); headroom-ai **0.42.0** built from source
  (`pyproject.toml` sha256 `b7fc457b…`) into an isolated uv venv, Python 3.12.13, extras
  **`[proxy,code,ml,html,relevance]`** (proxy pipeline, tree-sitter code compressor,
  Kompress via torch/transformers + onnxruntime 1.31.1, trafilatura HTML, fastembed
  relevance). Kompress weights `chopratejas/kompress-v2-base` (snapshot `b1563631`), loaded
  synchronously on CPU (`warm_kompress_model()` → `True` for every Headroom arm).
* tiktoken 0.14.0 `o200k_base`. Inputs: 360 block samples (60 per type; 358 from local
  Claude Code transcripts after the marker filter, 2 from `corpus/`) and 30 sessions × 12
  requests. Pool sizes before sampling: logs 1448, json 67, code 1053, html 199,
  search 457, prose 448. Wall time: distil arms 1–5 s each, Headroom arms 19–245 s.
* Reproduce: `PY=<venv>/bin/python; $PY -m benchmarks.headroom_h2h all --scratch <dir>`.

## Results

Tables are rendered from `results.json`; CIs are 95% percentile bootstrap on paired
per-sample (per-session) differences, distil minus Headroom, in percentage points.

### 1. Token reduction (% of the sample's tokens removed, mean per sample, n=60 per type)

| arm | logs | json | code | html | search | prose | all (n=360) |
|---|---|---|---|---|---|---|---|
| distil-lossless | 0.6 | 8.7 | 0.0 | 0.0 | 0.2 | 0.0 | 1.6 |
| headroom-densify | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| distil-default | 27.4 | 24.6 | 10.8 | 11.7 | 6.9 | 0.8 | 13.7 |
| headroom-default (library) | 3.1 | 33.2 | 1.4 | 2.2 | 3.3 | 0.0 | 7.2 |
| headroom-coding (proxy default) | 1.5 | 33.2 | 0.0 | 0.6 | 2.8 | 0.0 | 6.3 |
| distil-aggressive | 78.6 | 87.8 | 28.7 | 29.3 | 25.8 | 4.8 | 42.5 |
| headroom-agent90 | 22.2 | 0.0 | 31.7 | 31.1 | 22.6 | 35.0 | 23.8 |

### 2. Fact recall, true / visible (%), 6,007 facts

| arm | logs | json | code | html | search | prose | all | facts lost |
|---|---|---|---|---|---|---|---|---|
| distil-lossless | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 0 |
| headroom-densify | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 0 |
| distil-default | 100 / 77.4 | 99.9 / 68.0 | 100 / 86.8 | 100 / 91.6 | 100 / 95.4 | 100 / 98.4 | 100 / 86.1 | 2* |
| headroom-default | 98.2 / 96.7 | 98.0 / 69.9 | 100 / 96.6 | 99.7 / 99.0 | 98.8 / 98.7 | 100 / 100 | 99.1 / 93.2 | 80 |
| headroom-coding | 98.2 / 98.2 | 98.0 / 69.9 | 100 / 100 | 99.8 / 99.8 | 99.9 / 99.8 | 100 / 100 | 99.3 / 94.3 | 73 |
| distil-aggressive | 100 / 17.5 | 99.4 / 10.0 | 100 / 69.8 | 99.9 / 69.6 | 99.5 / 73.7 | 99.9 / 94.6 | 99.8 / 55.2 | 20* |
| headroom-agent90 | 98.2 / 82.0 | 100 / 100 | 100 / 68.2 | 99.7 / 74.7 | 98.8 / 80.7 | 100 / 66.5 | 99.4 / 79.0 | 19 |

\* **Matcher artifact, not compression loss.** 23 of the 6,007 facts cannot be matched by
`_in_recovery()` even against the byte-identical original (error lines cut at 160 chars
mid-token fail the right-boundary test). Every block distil changed has a byte-exact restore
blob, so by construction all 2 + 20 distil "lost" facts are among those 23. Headroom's
73–80 can include at most 23 of them; the rest are real drops (concentrated in JSON:
12 lossy JSON samples for the library default).

### Paired differences, distil − Headroom (pp, mean [95% CI])

| pair | ctype | reduction | true recall | visible recall |
|---|---|---|---|---|
| lossless vs densify | all (n=360/339) | +1.6 [0.9, 2.4] | 0.0 [0, 0] | 0.0 [0, 0] |
| lossless vs densify | json | +8.7 [4.9, 13.0] | 0.0 | 0.0 |
| default vs coding (proxy) | all | **+7.4 [3.8, 10.8]** | +0.7 [0.2, 1.4] | **−8.3 [−12.4, −4.3]** |
| default vs coding | logs | +25.9 [18.2, 33.9] | +1.8 [0.0, 5.3] | −20.8 [−29.7, −12.1] |
| default vs coding | json | −8.6 [−25.0, 7.5] | +1.9 [0.6, 3.6] | −1.9 [−19.9, 16.7] |
| default vs coding | code | +10.8 [5.8, 16.5] | 0.0 | −13.2 [−21.2, −6.4] |
| default vs coding | html | +11.2 [6.3, 16.4] | +0.2 [−0.1, 0.5] | −8.2 [−14.2, −3.0] |
| default vs coding | search | +4.1 [−0.9, 10.0] | +0.1 [0.0, 0.2] | −4.4 [−9.3, −0.8] |
| default vs coding | prose | +0.8 [0.0, 2.0] | 0.0 | −1.6 [−3.9, 0.0] |
| default vs library default | all | +6.5 [3.0, 10.0] | +0.9 [0.3, 1.7] | −7.1 [−11.3, −3.2] |
| aggressive vs agent-90 | all | **+18.7 [12.3, 25.2]** | +0.3 [−0.2, 1.2] | **−23.9 [−30.8, −16.9]** |
| aggressive vs agent-90 | logs | +56.4 [44.5, 67.7] | +1.8 [0.0, 5.3] | −64.4 [−76.1, −52.1] |
| aggressive vs agent-90 | json | +87.8 [79.8, 94.5] | −0.6 [−1.1, −0.2]* | −90.0 [−96.7, −81.7] |
| aggressive vs agent-90 | code | −3.0 [−17.3, 12.0] | 0.0 | +1.6 [−15.5, 18.9] |
| aggressive vs agent-90 | html | −1.9 [−15.3, 12.3] | +0.2 | −5.1 [−22.0, 11.0] |
| aggressive vs agent-90 | search | +3.2 [−9.5, 16.1] | +0.7 [−0.9, 3.4] | −7.0 [−21.3, 7.0] |
| aggressive vs agent-90 | prose | **−30.2 [−38.6, −20.8]** | −0.1 [−0.4, 0.0] | +28.1 [16.0, 39.8] |

### 3. Prompt cache: 30 real sessions × 12 requests, Sonnet 5.5, messages only

| arm / serving mode | $/session | vs plain [95% CI] | prefix held | cache-read share | request p50 / p95 ms |
|---|---|---|---|---|---|
| plain | 0.0426 | — | 100% | 85.5% | — |
| distil-default / served (+ prefix replay) | 0.0286 | **−29.7% [−35.8, −24.0]** | 100% | 85.3% | 3.1 / 17.7 |
| distil-lossless / served | 0.0425 | −0.3% [−0.4, −0.1] | 100% | 85.5% | 1.8 / 8.3 |
| headroom-coding / delta (proxy default, cache mode) | 0.0385 | −7.7% [−11.8, −4.0] | 100% | 84.9% | 3.4 / 526 |
| headroom-coding / stateless | 0.0458 | +3.7% [−5.4, 14.4] | 87.9% | 80.6% | 10.0 / 43.8 |
| headroom-agent90 / delta | 0.0358 | −13.6% [−18.4, −9.5] | 100% | 85.4% | 3.2 / 1,394 |
| headroom-agent90 / stateless (its proxy runs token mode) | 0.0381 | −9.3% [−15.4, −3.3] | 94.5% | 83.4% | 5.0 / 167 |
| headroom-default (library) / delta | 0.0425 | −0.2% [−0.4, 0.0] | 100% | 85.5% | 2.0 / 28.1 |
| headroom-default (library) / stateless | 0.0582 | **+31.3% [20.1, 44.1]** | 78.8% | 75.0% | 5.2 / 1,226 |
| headroom-densify / either | 0.0426 | 0.0% | 100% | 85.5% | ≤ 2.6 / 16.4 |

Paired, per session, as a share of the plain bill: distil-default − headroom-coding/delta
= **−22.0 pp [−29.8, −14.2]**; − headroom-agent90/delta = −16.1 pp [−21.1, −10.4]; −
headroom-agent90/stateless = −20.4 pp [−26.2, −14.1]. distil-lossless − headroom-coding/delta
= +7.4 pp [3.7, 11.6] (Headroom's proxy default beats distil's lossless mode).

The token-ratio trap, measured: the Headroom library default removes 7.2% of block tokens
yet, called statelessly on a growing history, **costs 31% more than sending nothing
compressed**, because it rewrites already-cached messages (prefix held on only 78.8% of
requests). Its proxy's delta engine fixes that (100% held), which is why it must be judged
in that mode.

### 4. Latency and failures

Block samples, one compression call per request (p50 / p95 ms): distil-lossless 0.2 / 1.2,
distil-default 0.6 / 12.9, distil-aggressive 0.3 / 1.0; headroom-densify 1.5 / 14.3,
headroom-default 2.3 / 1,252, headroom-coding 4.2 / 403, headroom-agent90 210 / 1,334
(Kompress on CPU). Session requests: table above. **Failures: 0 in every arm** (360 sample
calls and 360 session requests per arm and mode; Headroom's swallowed-exception signature
never seen).

### 5. Reversibility (n=360 block samples)

| arm | identical | exact blob | decoder | facts intact, no byte path | lossy | exactly recoverable |
|---|---|---|---|---|---|---|
| distil-lossless | 333 | 0 | 0 | 27 | 0 | 92.5% |
| headroom-densify | 360 | 0 | 0 | 0 | 0 | 100% (it changed nothing) |
| distil-default | 237 | 116 | 0 | 7 | 0 | 98.1% |
| headroom-default | 255 | 46 | 0 | 43 | 16 | 83.6% |
| headroom-coding | 300 | 14 | 0 | 31 | 15 | 87.2% |
| distil-aggressive | 202 | 158 | 0 | 0 | 0 | 100% |
| headroom-agent90 | 149 | 184 | 0 | 22 | 5 | 92.5% |

distil's Tier-0 reshapes (`«rows=… cols=…«`) carry every fact but ship no decoder, so they
count as "facts intact" rather than exact. Headroom's densify does ship one, but it fired on
nothing here (see caveats).

## Verdict

* **Where Headroom genuinely wins.** (a) *Visible* recall in every non-lossless tier:
  +8.3 pp over distil's default (+20.8 on logs, +13.2 on code), +23.9 pp at the aggressive
  tier. distil gets its savings by moving detail behind `distil_expand` handles; Headroom
  keeps more of it in front of the model. (b) Prose at the aggressive tier: agent-90's
  Kompress takes 35.0% off Markdown reads vs distil's 4.8% (−30.2 pp [−38.6, −20.8]).
  (c) JSON objects at the default tier: 33.2% vs 24.6% (CI crosses 0, so not a detectable
  win). (d) Headroom's proxy default beats distil's *lossless* mode on cost (−7.7% vs −0.3%).
* **Where distil wins.** Token reduction at the default (+7.4 pp) and aggressive (+18.7 pp)
  tiers, driven by logs, code and HTML; true recall (no real fact lost vs 73–80 for
  Headroom's defaults); exact recoverability (98–100% vs 84–93%); per-session cost with the
  cache priced (−29.7% vs −7.7% for Headroom's proxy default, −13.6% for agent-90); and
  latency (p95 ≤ 18 ms vs 0.4–1.4 s with Kompress on CPU).
* **Tie.** Lossless tier: both ≈ 0 on this data.

## Caveats

1. **Offline, behaviour held fixed.** Output tokens, extra turns and `distil_expand`
   round-trips are not priced. distil's cost lead rests on hiding detail (visible recall 86%
   vs Headroom's 93–94%); every expand an agent needs is an extra request this replay does
   not charge. Only a live A/B can settle whether the visible-recall gap costs more than the
   token gap saves. The `distil_expand` tool definition (~1 cached block) is also not counted.
2. **One machine's traffic.** All 358 transcript samples come from a single user's Claude
   Code sessions; `corpus/` contributed only 2 (its blocks are mostly < 1,500 chars).
   "prose" is 57/60 `Read`s of Markdown files and "json" is MCP/`TaskStop` *objects*; there
   is not one top-level JSON *array* in the sample, which is exactly what Headroom's
   SmartCrusher and `densify()` target. Headroom's 0% lossless result reflects that mix, not
   a defect. Both tools exclude or protect file reads by design.
3. **Serving emulation.** Headroom's Anthropic proxy clamps its frozen prefix to the
   provider-confirmed cache count; offline we use its cache-mode delta engine
   (`extract_cache_stable_delta`, as `benchmarks/policysim/headroom_worker.py` does) and also
   report stateless; agent-90's real proxy mode is "token", which those two bracket.
   distil is served whole-history + `prefixreplay`, which is what `distil proxy` forwards.
4. **Pricing scope.** Messages only (tools + system prefix excluded, identical for all arms),
   requests 10 s apart (no TTL expiry, so distil's cold-point eviction never fires and
   distil-aggressive is not run on sessions), `o200k_base` counts (Claude's tokenizer is not
   public), Sonnet 5.5 list prices.
5. **Latency hardware.** Kompress on CPU (Apple silicon). A GPU/CoreML backend would cut
   Headroom's tail; distil has no model in the path.
6. **Matcher artifact** (above): 23 facts the retention harness cannot match even against the
   identical original; reported, not adjusted, per the pre-registration.
