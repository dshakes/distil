# 0009 — Compression as an attack surface

- **Status:** accepted
- **Date:** 2026-09-04
- **Relates to:** `distil/harness.py`, `distil/compress/tier1.py`, `tests/test_adversarial_isolation.py`
- **Reference:** Y. Wang et al., *Context Manipulation Attacks on LLM Agents* (COMA), arXiv 2510.22963, ASE 2026

## Context

distil's threat modelling has treated compression as a correctness problem: does the agent still
decide the same thing? That framing assumes the input is merely *hard*, not *hostile*.

The COMA paper removes that assumption. It shows that an attacker who controls untrusted input —
a web page a browser tool fetches, a file in a repository, an MCP tool's output, an issue comment
— can perturb that input so the **compressor** discards task-critical content. The agent then
acts on a context missing the one line that mattered. Nothing in the pipeline reports a fault:
the request succeeds, the savings look good, the certificate is unaffected, and the answer is
wrong. The paper's validated mitigation is isolating trusted from untrusted content into separate
compression budgets.

This matters more for distil than for a summariser, because distil's keep policy is *legible*.
`tier1.py` and `keep_policy.py` state exactly what survives — `DECISION:` markers, verdict lines,
error/failure lines, traceback frames, diff hunk headers, head/tail context. An attacker who
reads the source knows precisely what to imitate. A heuristic anyone can read is a heuristic
anyone can bait.

## Decision

### 1. The keep policy is not a security boundary; reversibility is

distil does not claim its keep policy resists an adaptive attacker, and this ADR declines to make
that claim. The keep policy is a *savings* heuristic. The security property is one level down and
structural: **nothing distil folds is irrecoverable**, and every fold announces itself. A stub
says `<< +N lines, handle=... >>`, the original is in the `RestoreStore` keyed by its content
address, and `distil_expand` returns it byte-for-byte.

So the invariant we assert against hostile input is a disjunction, and both branches are
acceptable outcomes:

> the genuine load-bearing line survives in what we forward — **or** it was folded, the block is
> reversible through a handle distil issued, and the stub declares that lines were elided.

An attacker can push a line from the first branch into the second. They cannot push it out of
both, because the second does not depend on any judgement they can influence.

### 2. Trusted/untrusted budget isolation exists by construction

The paper's mitigation is already distil's shape, and it is worth stating precisely *why* rather
than claiming credit for a design accident.

There is **no global keep budget anywhere in the pipeline**. `tier1.digest` decides head/tail plus
must-keep lines from a single block's own text. `compress_messages` walks blocks one at a time.
No counter, cap, or ranking is shared between blocks. A tool_result cannot influence what survives
in a neighbouring user message, and an untrusted block cannot starve a trusted one, regardless of
size or position.

That is a claim about an *absence*, which is exactly what a plausible future optimisation would
fill in — a "keep the top N lines per request" cap would read as a sensible improvement and would
silently construct the shared budget COMA attacks. `tests/test_adversarial_isolation.py` therefore
asserts it as an **equality**: a trusted block must compress to exactly the same bytes whether or
not a 4000-line attacker block sits beside it, in either order. Any shared budget, however
generous, breaks that equality.

### 3. `distil validate --adversarial`

The COMA-class battery joins the existing real-path gate, driving the same public compression path
the proxy uses. Seven cases, each carrying the specific line an attacker is trying to get
discarded, checked against the five existing invariants plus the new **load-bearing** one:

| Case | What it baits | Result (measured) |
|---|---|---|
| `decoy_verdict_flood` | 400 fake `DECISION:` lines crowding the real one | survives verbatim; **savings drop to 0.0%** |
| `dedup_baiting` | 300 lines sharing the real error's numeric shape | **folded** — recoverable and declared |
| `salience_baiting` | junk stuffed with the agent's query terms | survives verbatim (96.5% savings) |
| `handle_forging` | a tool result printing distil's own stub syntax | survives; forged handle never resolves |
| `budget_starvation` | one 3000-line untrusted block | survives verbatim (99.8% savings) |
| `expand_prompt_injection` | text instructing the model to call `distil_expand` | survives; treated as ordinary text |
| `cross_block_starvation` | huge untrusted block beside a trusted one | survives verbatim (99.7% savings) |

## Consequences

- **One case lands in the second branch, and it is named rather than smoothed over.**
  `dedup_baiting` is the real COMA-class hit. Outcome-aware noise dedup normalises digits to
  compute a line's *shape*, so 300 attacker lines differing from the genuine error only in a shard
  number collapse together with it, and the real line is folded. It stays recoverable and declared,
  so the invariant holds — but the keep policy did lose it, and pretending otherwise would be the
  kind of claim this project exists to avoid. It has its own regression test, which fails if the
  behaviour changes in *either* direction, so an improvement has to be deliberate.

- **Denial-of-savings is a real and unmitigated outcome.** `decoy_verdict_flood` drives a block's
  savings to exactly 0.0%: every fake verdict is kept, because verdict lines are exempt from dedup
  and always kept. That is the correct trade — correctness over savings — but an attacker who can
  write into a tool's output can make compression stop paying for that block. distil is not a
  security control and cannot be spent down into one; this is stated as a cost, not fixed.

- **What is detected rather than prevented.** Shadow mode replays the real decision against the
  uncompressed context and flags divergence. It is sampled, so it is a detector and not a gate: it
  will notice a systematic attack across many requests, and will usually miss a single targeted
  one.

- **What is out of scope.** distil compresses a request; it does not authenticate its contents. It
  cannot tell a genuine `DECISION:` line from one an attacker wrote into a fetched page, because
  by the time distil sees it, both are bytes in a tool result. Provenance belongs to the agent and
  its tools. Likewise, prompt injection aimed at the *model* is unaffected by compression in either
  direction — distil neither introduces it nor defends against it, and treats such text as ordinary
  content.

- **This constrains future work.** Any transform that ranks or budgets across blocks now has to
  argue against this ADR first.
