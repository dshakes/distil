# ADR 0003 — Content-type coverage, and where the certificate can follow

- Status: Accepted — decision 2 partially implemented (see Implementation status)
- Date: 2026-07-27
- Deciders: distil maintainers

## Context

A capability audit against the current state of the art in context compression
(a mature competing implementation, its published documentation set, and its
open pull requests) found that distil's **compression quality is competitive or
ahead on the content types it handles, and that the set of types it handles is
materially narrower**.

That is the honest summary. distil's differentiator — a per-request certificate
that compression did not change the agent's next decision — is unmatched: the
surveyed alternative documents savings percentages and accuracy claims, but has
no equivalent of `distil certify` / decision-equivalence gating. Nothing in this
ADR trades that away.

### What the audit found, by content type

| Content type | distil today | Assessed gap |
|---|---|---|
| Tool-result text / logs | Tier-0 + Tier-1 digest, reversible | **Parity or ahead** — ours is byte-reversible; theirs is not |
| JSON arrays / records | `compress/structured.py` columnar fold | **Near parity**; they score items statistically (errors, anomalies, change points) where we fold structurally |
| Source code | Python `ast` + zero-dep brace skeleton | **Partial** — they parse 8 languages via a grammar toolkit; we cover Python well and brace-languages shallowly |
| Images / vision blocks | **Nothing.** `adapters/anthropic.py` explicitly passes `image` blocks through | **Total gap** |
| Prefix / cache alignment | `compress/cache_aware.py`, cache-aware costing | **Ahead** — we model cache economics; they report drift only |
| Retrieval of originals | `distil_expand` + `RestoreStore` | **Parity** — same compress/cache/retrieve shape |
| Cross-session memory | Nothing | Gap (see Decision 3) |
| Inter-agent context handoff | Nothing | Gap (see Decision 3) |
| Cost-aware model routing | Nothing | Out of scope (see Decision 4) |

The single largest *quantifiable* gap is images. A 1024×1024 image costs roughly
750–1600 input tokens depending on provider; an agent that screenshots, reads
diagrams, or inspects UI pays that on **every turn it stays in context**. distil
currently compresses the text around such a block and leaves the block itself
at full cost.

### Why we did not simply adopt their techniques

Two of their headline techniques are **lossy by construction**:

- statistical array pruning drops items it scores as unimportant;
- image compression downscales or reduces provider detail level.

distil's contract is that Tier-0/Tier-1 output is byte-reversible and that any
lossy tier is gated on a decision-equivalence certificate. Adopting a technique
that silently drops content would violate the property the project exists to
defend — and it is the property nobody else offers.

## Decision

**1. Extend content-type coverage, and make every new type enter through the
existing gate.** A new compressor ships disabled until `distil certify
--strategy <name>` reports non-inferior on the corpus, exactly like every
existing strategy. Coverage is a gap worth closing; the gate is not negotiable.

**2. Images become a first-class, certifiable content type.** Vision blocks get
a compressor whose transforms are declared, reversible where the original bytes
are retained locally (the `RestoreStore` already does this for text), and graded
by the same decision-equivalence machinery. The novel part — and the reason this
is worth doing rather than copying — is that **no published implementation
certifies that an image transform preserved the model's answer**. Savings claims
in this space are currently asserted, not proven. We can prove ours.

**3. Memory and inter-agent context are deferred, not rejected.** Both are real
gaps, but both are *stateful across sessions*, which puts them outside the
current certificate's unit of analysis (a single request's next decision).
Shipping them before we can certify them would invert the project's own
argument. They need their own ADR and their own gate definition first.

**4. Cost-aware model routing is out of scope.** It reduces spend by changing
*which model answers*, not by compressing context. distil's claim is that the
answer does not change; routing changes the answerer. Mixing the two would make
the certificate unfalsifiable.

## Consequences

- Every new content type costs a corpus domain and a certification run. That is
  deliberate friction: it is what stops coverage growth from eroding the claim.
- distil will still handle fewer *integration surfaces* (framework-specific
  wrappers) than the surveyed alternative. That is an accepted difference in
  strategy: distil is a proxy, so any `base_url`-honoring client is already
  supported without per-framework code. Framework wrappers are documentation and
  convenience, not capability, and are tracked separately.
- The audit's uncomfortable finding stands and is recorded here rather than
  buried: **on content-type breadth we are behind, and breadth is a legitimate
  user need.** Being right about rigor does not make a missing capability
  present.

## Implementation status

**Decision 2 — images, step 1 of 2: shipped, disabled.** `distil/compress/vision.py`
implements the one image transform that is *byte-reversible*: the second and
later appearances of a byte-identical image become a short reference stub
carrying a `RestoreStore` handle; the first appearance is untouched, so the model
still sees the image. It covers both top-level `image` blocks and the nested
`tool_result` case, which is where computer-use and browser screenshots actually
arrive. Dimensions are read from the file header with the stdlib (PNG/JPEG/GIF/
WEBP), so the zero-runtime-dependency property holds and savings accounting is
real rather than assumed; an unreadable header reports a deliberately low
estimate rather than a flattering one.

Per decision 1 it is **off until certified**: `vision.enabled()` is False unless
`~/.distil/certificates/vision.json` exists, and with no certificate the adapter
is byte-for-byte what it was before. It is additionally skipped in verbatim mode
and on recent turns, so it can never make an agent reason blind over its freshest
input.

**Decision 2 — step 2 of 2: CERTIFIED.** The blocker recorded in ADR 0004 rank 1
is closed. `Block` now carries an optional `media` list, the live Anthropic runner
renders it as real provider image blocks, `vision` is a registered strategy, and
`corpus/vision-ci-dashboard.json` is a decision-bearing vision trajectory whose
context accumulates so byte-identical screenshots actually pile up.

Result, live, against `claude-opus-4-8`:

```
A/A self-agreement floor: 100.0%
  turn 0: ok    turn 1: ok    turn 2: ok    turn 3: ok
decision-equivalence match rate: 100.0%
TOST non-inferiority (margin=0.02, alpha=0.05): mean diff=+0.000, p=<0.0001
VERDICT: PASS  (certified non-inferior)
```

The first live run **FAILED**, and that is the most useful thing in this record.
Turn 3 diverged: the baseline chose `promote_release`, the compressed arm chose
`open_failing_build`. The cause was not the compression — it was the runner
hoisting every image to the front of the turn, severing each screenshot from the
caption identifying it. An A/B whose arms differ in prompt *shape* measures the
shape, not the compression. Interleaving media with its own block's text fixed it.
A deterministic offline oracle could never have surfaced that.

The corpus is deliberately **excluded from `distil bench`**. That gate's
`DeterministicRunner` grades by reading `DECISION:` markers out of block text, and
`corpus.validate()` requires one on a decision-relevant volatile block. Satisfying
it here would mean writing the tile's colour into the text — at which point the
image is decorative and the certificate is a claim about text. It is listed under
`live_only` in the manifest with that reasoning attached.

*Remaining:* the corpus needs a
vision domain and `distil certify --strategy vision` needs to run and pass. Until
then no user gets this behavior. The step deliberately stops at the gate rather
than shipping enabled — this is the friction decision 1 asked for, applied to its
own first case.

**Decisions 3 and 4 remain as written** — memory and inter-agent context deferred
pending their own gate definition; routing out of scope.
