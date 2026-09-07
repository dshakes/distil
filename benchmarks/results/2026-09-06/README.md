# Re-read delta, before and after — 2026-09-06

Raw, unedited stdout backing the one number the re-read delta puts on the site
(`docs/claims.json`, `v-reread-delta-codebench`): on the read → edit → **re-read**
workload, distil's PAYG digest row moves from **0.0% to 10.3% token savings and
0.0% to 12.2% cache-aware dollar savings**, with the exact-quote invariant intact.

This is a **distil-vs-distil** comparison, not a head-to-head. `headroom-ai` and
`llmlingua` were not installed in this environment, so their rows are absent from both
files — the runner prints that on stderr and continues. The competitor tables on
`docs/compare.html` still come from `benchmarks/results/2026-09-04/`, which was run with
both packages present; nothing here changes them.

## Versions

- distil before: `030c159` (`origin/main`, 1.51.1)
- distil after: this branch (`feat/reread-delta`)
- Python: 3.12.13
- model used for cost estimates: `claude-opus-4-8`

## Files

| File | Runner | Notes |
|---|---|---|
| `codebench-before-030c159.out` | `benchmarks/codebench.py` on `origin/main` | the digest row is 0.0% — every read is exact-quote-exempt and nothing else is long enough to digest |
| `codebench-after-reread-delta.out` | `benchmarks/codebench.py` on this branch | same corpus, same seeds; only the digest rows move |
| `distil-bench-after-reread-delta.out` | `distil bench` (corpus gate, 9 domains) | unchanged from `main`, byte for byte: that corpus has no re-read of one path through a name-keyed read tool, so the delta never fires there |
| `distil-validate-after-reread-delta.out` | `distil validate` | 150/150 checks over 25 cases, including four new re-read shapes |

## What moved, and what did not

| row | before | after |
|---|---|---|
| `distil (PAYG digest)` tokens | 0.0% | 10.3% |
| `distil (PAYG digest)` cache-aware $ | 0.0% | 12.2% |
| `distil+cache-delta` tokens | 34.9% | 34.9% |
| `distil-verbatim` tokens | 18.5% | 18.5% |

The cache-delta and verbatim rows are unchanged by construction. `--session-delta` runs
before compression and already references whole blocks; verbatim mode does not emit
cross-block references at all (see `compress/rereaddelta.py`).

`ms/turn` on the digest row rises from ~1.0 to ~21. Profiling attributes that to
`mcp_server.record_restore` — the on-disk restore store and its LRU scan, which every
distil stub pays and which this row simply never paid before, because before this change
it emitted no stubs on this corpus. The planner itself measures **0.053 ms/turn** over the
same 320 turns.

## Commands (exact, reproduces the committed outputs)

```bash
# before
git worktree add ../distil-baseline --detach origin/main
cd ../distil-baseline && PYTHONPATH=. python benchmarks/codebench.py

# after
cd ../distil-hotpath2 && PYTHONPATH=. python benchmarks/codebench.py
PYTHONPATH=. distil bench
PYTHONPATH=. distil validate
```

## Read this with the corpus caveat

`benchmarks/codebench.py` is deliberately an **upper bound** on what the exact-quote
guarantee costs and therefore a **floor** on distil's digest savings: 55% of its
tool-result tokens are file reads that must stay byte-exact and the rest is too short to
digest. It is the right corpus for measuring the re-read mechanism on its hot path and the
wrong one for ranking compressors. On the 2,489 measured Claude Code sessions the
addressable share is smaller — name-keyed reads are 10.7% of tool-result mass, about half
of those are re-reads, and about half of a re-read's tokens were already delivered — so the
live ceiling is closer to **~2.8% of tool-result mass**. See `docs/adr/0010-the-re-read-delta.md`.
