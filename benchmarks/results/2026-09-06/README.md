# Re-read delta, before and after — 2026-09-06 (re-run 2026-09-15)

> **Provenance note.** The `.out` files named below were produced on 2026-09-06 but never
> committed — `benchmarks/.gitignore` ignores `*.out`, and unlike the 2026-09-04 set they
> were not force-added, so this directory held only this README. They were regenerated on
> **2026-09-15** with the same commands and are force-added now. The headline pair
> reproduced exactly (31.4% → 41.7% tokens, 35.8% → 49.1% dollars). Two secondary figures
> did not, because "after" is now `main` at 1.53.0rc1 rather than the `feat/reread-delta`
> branch; both are corrected below and the superseded values are named.

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
| no marker (codebench as shipped) | 0.0% / 0.0% | 10.3% | **+12.2%** |
| newest turn pinned (Claude Code) | 31.4% / 35.8% | **41.7%** | **49.1%** |

On the 2026-09-06 run of `feat/reread-delta` the unmarked dollar row read **−15.4%** — the
artefact above, left visible rather than hidden. On the 2026-09-15 re-run against 1.53.0rc1
it reads **+12.2%**: forwarded-bytes prefix replay (1.53.0rc1) stops charging the unmarked
shape for rewriting a prefix it never cached, so the sign flips. The artefact is still real
in principle — `distil-verbatim` shows the same shape on the same corpus, 18.5% tokens for
3.7% dollars — it is simply no longer negative for the digest row on this corpus.

`benchmarks/codebench_marked.py` replays the corpus under both shapes so the artefact is
reproducible rather than asserted.

## Versions

- distil before: `4cf076b` (`origin/main`, 1.52.0)
- distil after: `main` at `f7091ac` (1.53.0rc1) for the 2026-09-15 re-run; originally
  `feat/reread-delta` on 2026-09-06
- Python: 3.12 (`uv run --python 3.12 --no-project`; neither `headroom-ai` nor `llmlingua`
  installed, so their rows are absent from both `codebench` files, as before)
- model used for cost estimates: `claude-opus-4-8`

## Files

| File | Runner | Notes |
|---|---|---|
| `codebench-marked-before-4cf076b.out` | `benchmarks/codebench_marked.py` on `origin/main` | both client shapes, baseline |
| `codebench-marked-after-reread-delta.out` | same, on this branch | the headline pair |
| `codebench-before-4cf076b.out` | `benchmarks/codebench.py` on `origin/main` | full method table, unmarked shape |
| `codebench-after-reread-delta.out` | same, on this branch | only the digest row moves |
| `distil-bench-after-reread-delta.out` | `distil bench` (corpus gate, 9 domains) | unchanged from `main`, byte for byte: that corpus has no re-read of one path through a name-keyed read tool, so the delta never fires there |
| `distil-validate-after-reread-delta.out` | `distil validate` | 175/175 checks over 25 cases, including four new re-read shapes (was 150/150 on 2026-09-06; the prefix-replay-semantics invariant added in 1.53.0rc1 raises the check count, not the case count) |

## What did not move

The cache-delta and verbatim rows are unchanged by construction. `--session-delta` runs
before compression and already references whole blocks; verbatim mode does not emit
cross-block references at all (see `distil/compress/rereaddelta.py`).

`ms/turn` on the digest row rises from ~1.0 to ~36 on the 2026-09-15 re-run (it was ~6 on
2026-09-06). Profiling attributes that to
`mcp_server.record_restore` — the on-disk restore store and its LRU scan, which every
distil stub pays and which this row simply never paid before, because before this change
it emitted no stubs on this corpus. The planner itself measured **0.053 ms/turn** over the
same 320 turns when profiled on 2026-09-06; the figure was not re-profiled on 2026-09-15
and is not quoted on the site.

## Commands (exact, reproduces the committed outputs)

```bash
# before (4cf076b = origin/main at 1.52.0)
git worktree add ../distil-baseline --detach 4cf076b
cp benchmarks/codebench_marked.py ../distil-baseline/benchmarks/
cd ../distil-baseline
PYTHONPATH=. python benchmarks/codebench_marked.py
PYTHONPATH=. python benchmarks/codebench.py

# after
cd ../distil
PYTHONPATH=. python benchmarks/codebench_marked.py
PYTHONPATH=. python benchmarks/codebench.py
PYTHONPATH=. distil bench
PYTHONPATH=. distil validate
```

`distil bench` was run on both trees on 2026-09-15 and its output is byte-identical, which
is why only the "after" copy is committed: this corpus has no re-read of one path through a
name-keyed read tool, so the delta never fires there.

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
