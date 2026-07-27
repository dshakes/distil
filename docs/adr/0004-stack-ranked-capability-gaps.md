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

**Closed 2026-07-27.** Recorded in past tense below, because what blocked it is
more useful than the fact that it is done.

*The gap.* ADR 0003 decision 2 shipped `compress/vision.py` behind
`vision.enabled()`, which was False until a certificate existed — and none could,
so the capability did nothing for any user. A capability that exists only in the
repository is not a capability, and this was the highest-cost open item precisely
because it looked finished.

*Why it was not a corpus file.* The certification path was text-only end to end:
`Block.text` is a `str`, so a trajectory could not carry an image, and
`AgentRunner.decide(blocks) -> str` was the whole grading interface, with
`DeterministicRunner` reading `DECISION:` markers out of block text. The
available shortcut — serialize the base64 into `Block.text` and certify that —
would have run, gone green, and measured whether deleting a blob from a *text*
prompt changes a *text* decision. Not the claim.

*How it was closed.* `Block` gained an optional `media` list; the live Anthropic
runner renders it as real provider image content blocks; `vision` became a
registered strategy; and `corpus/vision-ci-dashboard.json` is a decision-bearing
vision trajectory whose context accumulates, so byte-identical screenshots
genuinely pile up. Against `claude-opus-4-8`: **100% decision-equivalence, A/A
floor 100%, TOST p<0.0001, PASS.**

*What it cost to find out.* The first live run FAILED — the compressed arm chose
`open_failing_build` where the baseline chose `promote_release`. The cause was
the harness, not the compression: it hoisted every image to the front of the
turn, severing each screenshot from its caption. An A/B whose arms differ in
prompt *shape* measures the shape. No offline oracle would have surfaced that,
which is the argument for live certification, produced by the thing itself.

*Standing constraint.* The vision corpus stays out of `distil bench`. That gate's
runner grades from text; this domain's decision is in the pixels. Making it pass
offline would mean writing the tile's colour into the text — image decorative,
certificate about text. `test_manifest_covers_every_corpus_file` enforces that the
`live_only` reason exists so the omission is not "fixed" later.

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
