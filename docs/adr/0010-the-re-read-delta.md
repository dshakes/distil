# 0010 — The re-read delta

- **Status:** accepted
- **Date:** 2026-09-06
- **Relates to:** `distil/compress/rereaddelta.py`, `distil/compress/provenance.py`,
  `distil/adapters/anthropic.py`, `distil/cachedelta.py`, ADR 0008, `tests/test_reread_delta.py`

## Context

1.49.0 and 1.51.0 made file content the agent may have to quote back byte-exact exempt
from the digest at any age. That is the right rule and it is expensive: on 2,489 measured
Claude Code sessions the exemption forfeits ~27pp of tool-result mass, because under a
client that marks most of its history cacheable — which is what Claude Code does — a
superseded read cannot be demoted either (ADR 0008 forbids rewriting a cached prefix).
The bytes are kept forever, and the coding hot path is where they pile up.

The same investigation measured what those kept bytes actually are.

| | |
|---|---|
| `Read`/`cat` results that re-read a path already read this session | 51.4% |
| of those, byte-identical to the earlier read | 12.8% |
| of a re-read's tokens, lines already delivered verbatim earlier in the conversation | 50.6% |

Half the mass of a re-read is a second copy of lines the model is already looking at. The
two mechanisms distil had could not see it:

- **Exact dedup** needs identical bytes, and 87% of re-reads are not identical.
- **`cachedelta`'s near-duplicate gate** compares whole blocks with a 0.5 `difflib` ratio.
  608 of 726 changed re-reads fail it, because two disjoint windows onto one file are not
  50% similar *as blocks* even when every line they share is byte-identical. It fires on
  5.2% of re-reads and removes 0.23% of tool-result mass.

The unit is wrong. Block similarity asks "is this nearly the same document?"; the useful
question is "which of these lines have I already sent?".

## Decision

When a read of path *P* arrives and an earlier read of *P* is still being forwarded
verbatim, replace the longest contiguous run of lines the two share with a reversible
reference stub, and keep every other line verbatim.

```
«distil-reread handle=a1b2c3d4» lines 1-20 of this result are byte-identical to
lines 41-60 of the earlier read of /app/handlers.py, which is still in this
conversation verbatim. Call distil_expand with this handle to recover them here.
```

### The quote-safety argument

The exact-quote guarantee is a statement about the **conversation**, not about any one
block: an `Edit(old_string=…)` applies if its quote occurs byte-exact somewhere in the
payload distil forwarded. An agent quoting lines it saw can be served by either copy. Four
rules keep that true.

1. **The base must be exempt unconditionally.** Only reads matched by the tool-NAME table
   (`Read`, `view`, `read_file`, …) may be referenced. Those are never superseded and never
   digested at any age, so the referenced lines cannot leave the conversation later. Shell
   reads (`cat`, `sed -n`) are **not** eligible as bases: their exemption is conditional on
   not being superseded, and a whole-file shell re-read supersedes precisely the block it
   would want to reference. Pinning the base to prevent that would forfeit the base's whole
   digest in order to save the same bytes on the copy — a wash at best. Shell reads remain
   eligible as *targets*.
2. **References never chain.** A block that has been elided is not itself a base, so every
   stub points at literal bytes.
3. **Cuts internal to the block are pulled back by a margin** (`EDGE_MARGIN`, 20 lines). A
   quote lying wholly inside the elided run survives in the base; one lying wholly outside
   survives here. Only a quote *straddling* a cut is in neither block contiguously, and the
   margin means it would have to overhang by more than 20 lines to break. A run reaching
   the block's own first or last line takes no margin there — the agent never saw anything
   beyond it in this block.
4. **A minimum run length** (`MIN_RUN`, 8 lines) keeps a coincidental match — a run of
   blank lines, a repeated `    return None` — from producing a stub.

The stub is reversible through the same `RestoreStore` every other distil stub uses;
`distil_expand` returns the elided lines byte-exact.

### The cache argument

`rereaddelta.plan` is a pure function of the message **prefix**: block *i*'s elision
depends only on reads before it. So a given block encodes to the same bytes on every turn
of a growing conversation, which is exactly what ADR 0008 clause (a) requires and the same
construction `cachedelta` relies on.

There is deliberately **no volatile-suffix gate**. That is a departure worth stating
plainly, because clause (b) is phrased as "compression touches only the volatile suffix".
Under the client shape that actually bills — Claude Code pins its newest turn, so the whole
history is cached and there is no uncached tail (ADR 0008 says so in as many words) — such
a gate would leave nothing to compress and the feature would be dead on arrival on the only
traffic it targets. Worse, it would *introduce* the failure clause (a) exists to prevent: a
rendering that flips from stub to verbatim as the boundary advances past a block. Prefix
determinism gives the stronger property (a block's bytes never change at all) without the
gate, and `tests/test_reread_delta.py` asserts it against the moving-marker shape.

### Measurement

- `distil validate` gains four re-read shapes under the existing quote-survival invariant:
  a re-read at a different offset, a quote straddling a cut, read → edit → re-read, and a
  whole-file read after a partial one. A separate test fails if none of them still produces
  a stub, so a silently-dead transform cannot pass as a silently-safe one.
- The proxy's quote-hazard counter already reports, per request and content-free, whether
  every literal-match edit's quote survived. On a miss the reaction now also turns the
  re-read delta off for the rest of the session, alongside dropping supersession.
- That counter is now wired for the **OpenAI Responses shape** as well, which 1.51 left as
  a known follow-up. Codex edits through `apply_patch`, a freeform custom tool whose
  argument is the patch body rather than JSON; its per-hunk pre-image (the context and
  removed lines, running through the added ones) is what has to be found byte-exact, and
  `provenance.patch_quotes` extracts it from the tool's own Lark grammar.

## Consequences

- On `benchmarks/codebench.py` (read → edit → re-read, 20 sessions / 320 turns) the PAYG
  digest row moves from **0.0% to 10.3% token savings and 0.0% to 12.2% cache-aware dollar
  savings**. Raw before/after output: `benchmarks/results/2026-09-06/`.
- `distil bench` is **unchanged byte for byte**. Its corpus contains no re-read of one path
  through a name-keyed read tool, so the delta never fires there. That is the honest
  reading: this transform is narrow.
- **The live ceiling is small and should be quoted as such.** Name-keyed reads are 10.7% of
  tool-result mass on real traffic, about half of those are re-reads, and about half of a
  re-read's tokens were already delivered — roughly **2.8% of tool-result mass**, in line
  with the ~2.3% the investigation projected. codebench overstates it because that corpus
  is nothing but reads.
- Cost is `0.053 ms/turn` for the planner over the same 320 turns. The digest row's
  `ms/turn` rise (≈1 → ≈21) is the on-disk restore store, which every stub pays and which
  this row had never paid before because it emitted no stubs on this corpus.
- A future transform that wants cross-block state now has a second precedent for how to do
  it under ADR 0008: not by dodging the cached prefix, but by making the encoding a
  function of the prefix so there is nothing to invalidate.
