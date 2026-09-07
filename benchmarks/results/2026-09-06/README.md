# Re-read delta, before and after — 2026-09-06

Raw, unedited stdout backing the numbers the re-read delta puts on the site
(`docs/claims.json`, `v-reread-delta-codebench`). Headline, on the read → edit → **re-read**
workload under the client shape that actually bills — Claude Code pins its newest turn, so
the whole history is cached:

| `distil (PAYG digest)` | before (`4cf076b`) | after |
|---|---|---|
| token savings | 31.4% | **41.7%** |
| cache-aware dollar savings | 35.8% | **49.1%** |

This is a **distil-vs-distil** comparison, not a head-to-head. `headroom-ai` and
`llmlingua` were not installed in this environment, so their rows are absent from both
`codebench` files — the runner prints that on stderr and continues. The competitor tables
on `docs/compare.html` still come from `benchmarks/results/2026-09-04/`, which was run with
both packages present; nothing here changes them.

## Read the unmarked row with its caveat

`benchmarks/codebench.py` builds its sessions with **no** `cache_control` marker and then
prices them with a cache: `_session_dollars` bills the longest prefix identical to the
previous turn at the cache-read rate. No real Anthropic client looks like that. Anthropic
caches **only what the client marks**, so an unmarked request has no cached prefix at all —
and any transform whose rendering legitimately changes one turn later (every
recency-anchored carve-out distil has, including this one) is charged there for busting a
cache that was never created.

| client shape | before | after (tok) | after ($) |
|---|---|---|---|
| no marker (codebench as shipped) | 0.0% / 0.0% | 7.1% | **−15.4%** |
| newest turn pinned (Claude Code) | 31.4% / 35.8% | **41.7%** | **49.1%** |

The −15.4% is left visible rather than hidden. It is the artefact above, and it was
invisible before only because the digest row was 0.0% and nothing moved. `distil-verbatim`
already shows the same shape on the same corpus (18.5% tokens for 3.7% dollars).

`benchmarks/codebench_marked.py` replays the corpus under both shapes so the artefact is
reproducible rather than asserted.

## Versions

- distil before: `4cf076b` (`origin/main`, 1.52.0)
- distil after: this branch (`feat/reread-delta`)
- Python: 3.12.13
- model used for cost estimates: `claude-opus-4-8`

## Files

| File | Runner | Notes |
|---|---|---|
| `codebench-marked-before-4cf076b.out` | `benchmarks/codebench_marked.py` on `origin/main` | both client shapes, baseline |
| `codebench-marked-after-reread-delta.out` | same, on this branch | the headline pair |
| `codebench-before-4cf076b.out` | `benchmarks/codebench.py` on `origin/main` | full method table, unmarked shape |
| `codebench-after-reread-delta.out` | same, on this branch | only the digest row moves |
| `distil-bench-after-reread-delta.out` | `distil bench` (corpus gate, 9 domains) | unchanged from `main`, byte for byte: that corpus has no re-read of one path through a name-keyed read tool, so the delta never fires there |
| `distil-validate-after-reread-delta.out` | `distil validate` | 150/150 checks over 25 cases, including four new re-read shapes |

## What did not move

The cache-delta and verbatim rows are unchanged by construction. `--session-delta` runs
before compression and already references whole blocks; verbatim mode does not emit
cross-block references at all (see `distil/compress/rereaddelta.py`).

`ms/turn` on the digest row rises from ~1.0 to ~6. Profiling attributes that to
`mcp_server.record_restore` — the on-disk restore store and its LRU scan, which every
distil stub pays and which this row simply never paid before, because before this change
it emitted no stubs on this corpus. The planner itself measures **0.053 ms/turn** over the
same 320 turns.

## Commands (exact, reproduces the committed outputs)

```bash
# before
git worktree add ../distil-baseline --detach origin/main
cp benchmarks/codebench_marked.py ../distil-baseline/benchmarks/
cd ../distil-baseline
PYTHONPATH=. python benchmarks/codebench_marked.py
PYTHONPATH=. python benchmarks/codebench.py

# after
cd ../distil-hotpath2
PYTHONPATH=. python benchmarks/codebench_marked.py
PYTHONPATH=. python benchmarks/codebench.py
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
live ceiling is closer to **~2.8% of tool-result mass**. See
`docs/adr/0010-the-re-read-delta.md`.
