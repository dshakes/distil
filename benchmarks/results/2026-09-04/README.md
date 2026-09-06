# Fresh head-to-head, 2026-09-04

Raw, unedited stdout from the two benchmark runners cited on `docs/compare.html`
(`#headroom-fresh`) and in `docs/claims.json` (`v-compare-fresh-headtohead`).
Nothing in these four files has been reformatted; the tables quoted on the
site are copy-pasted from the "technique"/"method" table each run prints.

## Versions

- distil: 1.51.1 (this repo, `main`)
- headroom-ai: 0.37.0, extras `[ml,code]`
- llmlingua: 0.2.2
- Python: 3.12
- model used for cost estimates: `claude-opus-4-8`

## Files

| File | Runner | Notes |
|---|---|---|
| `distil-benchmark-corpus-2026-09-04.out` | `distil benchmark` (corpus gate, 9 domains) | cold process: Headroom's Kompress model had not finished loading (`Kompress model not ready; requests will not be compressed`), so Headroom scores 0.0%/0.0% here — a loading artifact, not a compression result |
| `distil-benchmark-corpus-warm-2026-09-04.out` | `distil benchmark` (corpus gate, 9 domains) | same corpus and process, run again after Kompress finished loading; Headroom 1.7% tok / 2.0% $ / 81% decision-equiv / fails the gate |
| `codebench-2026-09-04.out` | `benchmarks/codebench.py` (read→edit→reread coding-agent workload, 20 sessions / 320 turns) | first run in the process; Headroom's ModernBERT weights load during this run |
| `codebench-warm-headroom-2026-09-04.out` | `benchmarks/codebench.py`, same workload | re-run after Headroom's weights were already resident; timings differ slightly (ms/turn), token/dollar percentages do not |

## Warm vs. cold

Both runners load Headroom's local models (ONNX/ModernBERT via `headroom-ai[ml]`)
lazily on first use. The first invocation in a fresh process pays that load
cost and, in the corpus-gate case, can see Headroom skip compression entirely
because the model isn't ready yet by the time the corpus gate's fixed-timeout
requests run. The site quotes the **warm** numbers for Headroom (post-load) as
the fair comparison, and calls out the cold artifact in a footnote rather than
using the misleadingly-favorable 0.0%/0.0% cold row. distil has no equivalent
lazy-load step (its causal/lossless/digest paths are pure Python, no model
weights), so its numbers are identical cold or warm.

## Commands (approximate, as actually invoked)

```bash
# corpus gate (cold, then warm re-run in the same venv)
python -m benchmarks.run_benchmark --external headroom_adapter:compress
python -m benchmarks.run_benchmark --external headroom_adapter:compress   # warm

# codebench (coding-agent read/edit/reread workload)
python benchmarks/codebench.py --external headroom_adapter:compress
python benchmarks/codebench.py --external headroom_adapter:compress       # warm
```

See `docs/compare.html#headroom-fresh` for the tables rendered from these
files, and `docs/claims.json` (`v-compare-fresh-headtohead`) for the ledger
entry backing that section.
