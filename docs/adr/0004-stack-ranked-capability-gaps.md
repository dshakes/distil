# ADR 0004 — The remaining capability gaps, stack-ranked

- Status: Accepted
- Date: 2026-07-27
- Deciders: distil maintainers
- Supersedes nothing; extends [ADR 0003](0003-content-type-coverage-and-the-certificate-boundary.md)

## Context

ADR 0003 recorded a capability audit against the current state of the art in
context compression and closed its single largest finding (images). This ADR
re-checks the rest of that audit **against the code as it actually exists**,
stack-ranks what is genuinely left, and decides what to do about each item.

The re-check matters because the original audit's gap column was assembled from
the *surveyed alternative's* feature list, not from a line-by-line read of ours.
Doing the read changed three of the answers, and in the direction that should
make us suspicious of our own audits: **we were understating what we already
had.** A gap register that invents work is as expensive as one that hides it.

### Corrections to the ADR 0003 table

| Claim in ADR 0003 | What the code actually shows |
|---|---|
| Git diffs — not listed, implicitly a gap | **Already handled.** `compress/keep_policy.py` carries a `DIFF` content kind that detects `diff --git` / `@@` and pins hunk headers and file markers as must-keep lines. |
| Source code — "we cover Python well and brace-languages shallowly" | **Understated.** `skeleton.py` does Python via `ast` (imports, signatures, decorators, first docstring line, bodies elided) *and* a brace-depth structural skeleton that covers the whole C-family — Go, Rust, Java, Swift, Kotlin, JS/TS, C/C++, C#. It declines rather than guesses when braces do not balance. |
| Search-result ranking, log triage, plain-text redundancy | **Already handled** by `compress/query_relevance.py`, `compress/intent.py` + `compress/salience.py`, and the learned keep-model respectively. |

The honest post-correction position: **on content types we are at or near parity
everywhere we have looked, once the vision work lands.** The remaining gaps are
not compression capabilities. They are a certification debt and a distribution
question.

## Decision

Gaps are ranked by *user-visible cost of leaving them open*, not by effort.

### Rank 1 — ~~Vision is uncertified, therefore inert~~ **CLOSED: certified live, 100% decision-equivalence.**

ADR 0003 decision 2 shipped `compress/vision.py` behind `vision.enabled()`,
which is False until a certificate exists. That was deliberate — but it means the
capability currently does **nothing for any user**, and a capability that exists
only in the repository is not a capability. This is the highest-cost open item
precisely because it looks finished.

Closing it needs a corpus vision domain (a trajectory carrying repeated image
blocks, which is what a UI-automation agent actually produces) and a passing
`distil certify --strategy vision`. Until both exist, no release note may claim
image compression.

**It is blocked on a data-model change, not on writing a corpus file.** The
certification path is text-only end to end:

- `Block.text` is a `str` (`distil/trajectory.py`), so a trajectory cannot carry
  an image at all;
- `AgentRunner.decide(blocks: list[Block]) -> str` is the entire grading
  interface, and `DeterministicRunner` reaches the decision by scanning
  `b.text` for `DECISION:` markers.

There is a tempting shortcut that must be named so nobody takes it: serialize the
image as base64 into `Block.text` and certify that. It would run, go green, and
mean nothing — it would measure whether deleting a base64 blob from a *text*
prompt changes a *text* decision, which is not the claim. The claim is that the
model, having actually seen an image once, decides the same way when a later
identical copy is replaced by a reference. Grading that requires sending real
image content to a vision model.

So rank 1 decomposes into: (a) let a Block carry non-text content, (b) teach the
live runner to build real provider content blocks from it, (c) build the corpus
domain, (d) run the certification against a vision model. (a) touches the core
type every existing strategy depends on, which is why this is its own piece of
work and not a follow-up commit.

Until then the honest statement of what is proven about `compress/vision.py` is:
**reversibility yes, decision-equivalence not yet.** The round-trip is verified by
test (the elided `source` object is recovered byte-exact); the effect on the
model's next action is unmeasured. Those are different claims and the second one
is the one distil exists to make.

### Rank 2 — Integration breadth is a documentation gap, and is closed by documenting it.

The surveyed alternative names more framework integrations than we do. Read
closely, each of theirs is a thin client that redirects an endpoint. So is ours:
distil is a proxy, so a framework is supported the moment it lets you set a
base URL.

Verified against each framework's own documentation and added to the integration
matrix: **Agno** (`OpenAILike(base_url=…)`) and **Strands Agents**
(`OpenAIModel(client_args={"base_url": …})`). Both are two-line redirects.

**We will not ship per-framework packages.** A `distil-<framework>` package is a
version to maintain, a thing to break on the framework's next release, and it
buys nothing a base URL does not. The in-process hooks (LiteLLM, LangChain,
LangGraph) stay as they are — they exist for compression *without* a proxy, which
is a different capability, not a redirect.

### Rank 3 — Grammar-based multi-language code parsing. **Declined for now, revisit on evidence.**

The surveyed alternative parses ~8 languages with a grammar toolkit. Our brace
skeleton covers the same language families structurally without the dependency.
A grammar toolkit would buy semantic precision — knowing a `}` closes a *method*
rather than a block — at the cost of a native dependency, which breaks the
pure-Python install that is a stated non-negotiable.

Revisit **only** with evidence: a corpus domain where the brace skeleton
measurably underperforms on decision-equivalence, not on compression ratio.
Ratio is not the contract.

### Rank 4 — Cross-session memory and inter-agent handoff. **Still deferred, per ADR 0003 decision 3.**

Unchanged. Both are stateful *across* sessions, which puts them outside the
certificate's unit of analysis (a single request's next decision). They need
their own ADR and their own gate definition before any code. Shipping them under
the current certificate would let us claim a proof we do not have.

### Rank 5 — Cost-aware model routing. **Still out of scope, per ADR 0003 decision 4.**

Unchanged. It reduces spend by changing *which model answers*. Our claim is that
the answer does not change; routing changes the answerer. Mixing them makes the
certificate unfalsifiable.

## Consequences

- The gap list is now shorter than the one we started with, and one item on it
  (rank 1) is work we created ourselves by gating correctly. That is the right
  trade and it is also a standing reminder: **shipping disabled is not shipping.**
- Ranks 3–5 are decisions to *not* build, recorded so they are not silently
  re-litigated each time a competitor's feature list is read.
- The audit-correction table above should be the template for the next capability
  review: read our own code before believing a gap.
