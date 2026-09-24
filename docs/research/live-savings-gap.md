# The live savings gap: where cache-priced dollars actually go

*Research note, 2026-09-24. Data: `benchmarks/results/2026-09-24/live_savings_decomposition.json`,
produced by `benchmarks/live_savings_decomposition.py --since 2026-09-01 --until 2026-09-24T20:20:00`
over 13,191 live Claude Code requests (one machine, all `mode: digest`, 13 days, $2,510 at list price).*

Labels: **M** = measured from provider `usage.*` counts. **E** = estimated; assumption stated.

## 1. Where the bill goes

| slice | $ | share | label |
|---|---|---|---|
| cache reads (0.10x) | 1,813.53 | **72.2%** | M |
| cache writes (1.25x) | 541.65 | 21.6% | M |
| uncached input | 7.67 | 0.3% | M |
| output | 147.60 | 5.9% | M |
| — tool schemas | 960.90 | 38.3% | E ¹ |
| — injected user text (CLAUDE.md, memory, skill/agent listings, reminders) | 603.32 | 24.0% | E ¹ |
| — tool_results, all buckets, post-compression | 619.74 | 24.7% | E ¹ |
|   · digested | 255.54 | 10.2% | E |
|   · kept verbatim for Edit safety (`exact_quote` + `shell_read`) | 286.80 | 11.4% | E |
|   · too short to digest (`short`) | 64.84 | 2.6% | E |
| — system prompt | 102.83 | 4.1% | E |
| — assistant text, images | 76.08 | 3.0% | E |
| **realized distil savings** | 286.26 | **10.2%** of the counterfactual bill | E ² |

¹ distil's heuristic per-bucket token counts, rescaled per request so they sum to the billed input,
priced at that request's blended input rate. The sibling branch measured tools at 35.6% by a
different method (`feat/tool-schema-compaction:…/tool_schema_share.json`); treat ±3pp as the band.
² `tokens_saved` rescaled the same way; assumes the counterfactual had the same cache structure.

Three facts this changes:

* **The August numbers are stale.** "0.4% live" was a token ratio on lossless-only traffic. On
  September digest traffic distil saves an estimated **10.2% of dollars** with the cache intact:
  cache hit ratio is 97.6% (3.15B read vs 76.5M written, M).
* **The recency flaw (lever 2 in the brief) is already fixed.** #123 (1.45.0) anchored the
  carve-out to the client's `cache_control` breakpoint (`distil/compress/recency.py:exempt_indices`).
  Claude Code marks its last message, so nothing is exempt and every tool_result is digested at
  first sight. Evidence: `tool_result_recent` is $0.11 of $2,510. Digest now compounds with caching.
* **72% of dollars are cache reads of a mostly static prefix.** Tools + injected text + system =
  66% of the bill, is identical on every turn, and is already billed at 0.1x.

## 2. Levers, ranked by estimated share of the current bill

A tool_result token saved at first sight is worth **4.8x its own write cost** over its lifetime
(it is re-read on every later turn; E, lineage model). That multiplier is why first-sight and
cache-safe beat everything else.

Every "realistic" figure is **E and judgment**: an assumed reduction rate applied to the ceiling.
The rate is not measured.

| # | lever | ceiling | realistic (E, judgment) | conf. |
|---|---|---|---|---|
| 1 | **Cold-point recompaction**: when a turn arrives past the 5-min TTL the provider re-writes the whole prefix anyway, so evict old tool_results to stubs there, at zero cache penalty | 18.4% (E: carried post-compression tool_result $ from each cold point to the next break) | **5–9%**, assuming 30–50% eviction | low-med |
| 2 | Tighter first-sight digest of fresh tool_results (short-block fold, harder digest) | 13.3% (E: digested 10.2 + short 2.6 + html 0.3 + declined 0.2) | 3–5%, assuming 25–35% further reduction | med |
| 3 | Warm prefix breaks, any cause (incl. the provenance re-expand P0, where a digested block turns verbatim when an Edit arrives later) | 2.0% (E: writes past lineage growth, clean pairs only, lower bound) | 1–2%, assuming most breaks are fixable | med |
| 4 | Output shaping | 5.9% (M) | 1–2%, assuming a 20–30% cut, PAYG only | med |
| 5 | Lossless tool-schema compaction (sibling branch) | 1.1% (E: 3.2% lossless schema reduction, measured on the schemas, × the tool $ share) | 1.1% | high |
| 6 | 1-hour TTL for idle gaps | 5.2% (write $ is M; bucketing by gap is E, via the lineage model) | **negative** ³ | high |
| — | Expand round-trips (the cost of lossy digest) | −0.4% (M) | — | — |

³ This assumes a 1-hour write costs 2.0x input, Anthropic's list price for the 1-hour TTL. That
multiplier is **not** in `distil/pricing.py`, which models only the 5-minute 1.25x write.
- Added cost: every write moves from 1.25x to 2.0x, so $541.65 × (2.0/1.25 − 1) = **+$324.99**.
- Recovered: the 5–60 min cold writes become reads, so ($100.11 + $29.26 = $129.37) × (1 − 0.10/1.25) =
  **≤$119.02**. This is an upper bound, because gaps over 60 min stay cold under a 1-hour TTL too.
- Net: at least −$206.

Fresh tool_results at first appearance are only **5.2%** of the bill (E: 41% of newly written
tokens on clean lineage pairs × warm write $). Their value is in the reads that follow, not the
write. Whatever is done has to land before, or exactly at, a cache write.

**Honest total:** 10% today, plus roughly 11–19pp from levers 1–5, gives **~21–29% of the
counterfactual bill** on this workload. Beyond that the remaining money is static context.

## 3. Design for lever 1: cold-point recompaction

**Invariant:** a byte may change only on a turn where the provider cache is already lost. After
that, the new form must be byte-stable for the rest of the lineage.

1. **Detect cold.** `distil/prefixreplay.py:_Lineage` gets a `last_ts`. In `replay()`, a gap
   longer than the TTL (300 s, or the `ttl` on the client's `cache_control`) marks the turn cold.
   The fallback is `usage_cache_read == 0` on the previous response, recorded in `proxy.py` beside
   `usage_cache_read`.
2. **Decide once.** On a cold turn, choose the eviction set. Candidates are tool_result handles
   older than K tool turns that have not been referenced since. Exclude `provenance` keep-sets
   (`compress/provenance.py`), learned-keep and expanded handles. Store the set as
   `_Lineage.evicted: frozenset[str]`.
3. **Apply every turn.** Add an `evict: frozenset[str]` kwarg to
   `adapters/anthropic.py:compress_messages`, threaded down to the tool_result path at
   `anthropic.py:517`. A handle in the set renders a fixed stub with no digest body, and the
   stub stays recoverable through `distil_expand` / `RestoreStore`. The stub must be a pure
   function of the handle, so replay's canonical-equality guard (`prefixreplay.replay`) holds
   and the evicted form survives replay.
4. **Never un-evict.** If lineage state is lost (hot-swap, LRU), the next turn un-evicts, and
   that costs one rewrite. This is the same cost replay already accepts. Persist only the handle
   set, which is content-free, if that cost shows up in `distil cache`.
5. **Gate.** Add census bucket `tool_result_evicted`. Certify decision-equivalence with the
   shadow harness before this is on by default. Evicted blocks are expand candidates, so count
   the round trips: at the measured 0.4% it stays cheap, but the same cold-turn rule must price
   it.

Open risk: lineage keys merge parallel subagents (38% of pairs excluded as dirty), so the 18.4%
pool is imprecise. A live A/B on `distil cache` read/write totals is the real gate.

## 4. What is not achievable, honestly

* **"Up to 95%" does not transfer.** That figure is per block, on compressible tool output, in
  uncached token counts. Here every tool_result, compressed or not, is **32.4%** of the
  counterfactual bill (E). Deleting all of them caps dollar savings near 32%, and 32% of that is
  Edit-safety verbatim that no honest compressor may touch.
* **The static 66% is out of reach for a cache-safe compressor.** Tool schemas, injected
  CLAUDE.md/memory and system prompt are resent verbatim and read at 0.1x. Lossless tool
  compaction yields 1.1%. Anything lossy rewrites instructions, and the user's config choices
  (fewer MCP servers, smaller memory files; see `distil discover`) are worth more than any
  transform.
* **Compressing later than first sight is a loss.** A byte rewritten after it enters the cached
  prefix trades a 0.1x read for a 1.25x write of everything after it. The August 2x regression
  was this.
* **Output is 5.9% of dollars.** Shaping cannot move the total by more than about 2%.
