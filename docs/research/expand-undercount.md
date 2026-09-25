# How much the published live savings figure could be overstated by unrecorded spend

- **Date:** 2026-09-25
- **Window:** 2026-09-01 to 2026-09-24T20:20:00Z. This is the window behind the published
  **10.2%** (`benchmarks/live_savings_decomposition.py`, reproduced here as 0.1024 on
  13,191 priced requests, $2,510.45 billed, $286.26 saved).
- **Method:** read-only and content-free over `~/.distil/sessions/*.requests.jsonl`
  and `~/.distil/shadow.jsonl`, using the decomposition script's own row filter and
  pricing. The published number is not changed here.

## The defect

Until this fix, the per-request ledger kept one upstream call's usage per client
request. That was the first call on the streaming `distil_expand` splice, and the last
call on the buffered expand loops. Chat Completions and Gemini recorded no usage at all.
Every `distil_expand` re-query re-sends the whole prompt plus the recovered block, and
none of it reached the ledger. Shadow replays, which are on by default at 2%, make two or
three extra full-context calls per sampled request on the user's key. They were never
netted out of any savings figure either.

The fix sums usage over every upstream call, records `upstream_calls` and
`expand_requery_usage`, and nets both kinds of spend into the savings ledger. Shadow
spend also goes to `sessions/<sid>.overhead.jsonl`. None of the historical rows in this
window carry the new fields (0 of 13,191), so the size of the gap is estimated rather
than measured.

## Expand re-queries: 0.07–0.34 pp

- 169 requests in the window resolved an expand (225 handles). All 169 went through the
  streaming splice, where the re-query's input and cache usage was dropped and its output
  was already counted.
- **Low ($1.93):** one re-query per request, its prefix served entirely from cache (0.1×
  the recorded input), plus the recovered block written once at 1.25×.
- **High ($9.54):** one re-query per handle, up to the loop's cap of 4, each priced like
  the recorded call, plus the block write and 50 output tokens per round.
- Netted against the counterfactual bill ($2,796.71), the published **10.24% would be
  10.17% (low) to 9.89% (high)**, an overstatement of **0.07–0.34 percentage points**.

## Shadow replays: at least 0.92 pp, likely about 1.6 pp

- 1,226 shadow rows fall in the window: 890 paired (three replays), 254 A/B (two) and 82
  A/A (two). Only 701 carry the per-arm token counts (`in_a`, `in_b`, `out_a`, `out_b`),
  and those counts are uncached input plus output only, because cache fields were never
  stored.
- The spend the counted rows prove is **$25.80**, which is **0.92 pp** of the
  counterfactual bill. That is a floor. It excludes the 525 rows without counts and every
  cache read and write the replays paid for. Scaling to all 1,226 rows at the same rate
  gives about $45, or about **1.6 pp**, still before cache fields.
- Shadow is a measurement cost, not a compression cost, but it is spend distil causes on
  the user's key. From this fix on it is netted in the savings ledger, and it is included
  in the distil arm's cost in `distil ab`.

## Bound

Taking the two together, the published **10.2% is overstated by at least 1.0 pp** (0.07 +
0.92) and **plausibly by about 2 pp** (0.34 + 1.6). A 10.2% gross figure is roughly an
8–9% net one. A re-run of the decomposition on a window recorded after this fix will
measure it directly: `upstream_calls`, `expand_requery_usage` and `overhead.jsonl` make
the netting exact.

## Reproduce

```
git show docs/first-impression:benchmarks/live_savings_decomposition.py > lsd.py
python lsd.py --since 2026-09-01 --until 2026-09-24T20:20:00   # 0.1024
```

The expand estimate joins `expanded_handles` (or the legacy `expanded` flag) with the
same session's `blocks[].tokens`. The shadow floor sums `in_*`/`out_*` per replay arm
at list price. Neither reads any content.
