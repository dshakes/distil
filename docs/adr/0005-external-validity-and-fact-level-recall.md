# ADR 0005 — External validity, and measuring recall instead of only equivalence

- Status: Accepted
- Date: 2026-07-29
- Deciders: distil maintainers
- Extends [ADR 0004](0004-stack-ranked-capability-gaps.md)

## Context

Every quality signal distil shipped before this ADR shares two properties:

1. It is graded **against our own corpus** (7 hand-built trajectories) using **our own
   oracle** (planted `DECISION:` markers), or against live traffic using our own
   decision signature.
2. It answers a **block-level** question — `verify` proves bytes round-trip,
   `certify`/`shadow` prove the next action is unchanged.

Both properties are defensible and the statistics are genuinely rigorous (TOST
non-inferiority, conformal bounds, bootstrap holdout CIs). But together they leave two
holes that no amount of added rigour inside the existing frame can close.

**Hole 1 — the numbers are unfalsifiable by a stranger.** A reader cannot check
"non-inferior on the distil corpus" against anything they already trust. Our corpus, our
markers, our verdict. That is not an accusation of dishonesty; it is a structural limit
on how much an outside reader can rationally update on our claims.

**Hole 2 — nobody asks whether the *next action* survived.** They ask whether *the
number they needed is still there*. "Decision-equivalent" is the right invariant for
certification and the wrong answer to "did compression eat my stack trace?" We had no
recall metric at all — not a low one, none — and `verify_reversible` was measuring
something subtly different: *can these bytes be reconstructed from the restore table*,
at block granularity, which is true by construction for Tier-0/1 and therefore always
100%. A metric that cannot fail is not a gate.

## Decision

### 1. Grade against public ground truth, over the network, with no new dependencies

`distil retention --dataset {hotpotqa,squad}` loads real public benchmarks whose answer
keys were written by someone else — HotpotQA's `supporting_facts` (the exact gold
sentences, amid 8 distractor paragraphs) and SQuAD v2's answer spans.

Rows come from the HuggingFace **datasets-server REST API** — plain JSON over HTTPS —
not the `datasets` package. This preserves `dependencies = []`: pulling in Arrow and a
model stack to read 100 rows would cost more than distil's entire core. Rows cache to
`$DISTIL_HOME/datasets` so a published number reproduces offline, and ordering is the
split's own order with no shuffle, so "n=100" means the same 100 cases for everyone.

**Rejected: vendoring the datasets.** Redistribution terms differ per dataset and a
committed copy silently ages. Fetch-and-cache keeps provenance honest.

### 2. Compare at matched savings, or do not compare

A recall number alone is meaningless — an identity transform scores 100%. Every dataset
run therefore also scores a **truncation baseline tuned to reproduce distil's own token
savings on that same case**, and prints both savings figures so a reader can confirm
they match. On HotpotQA (n=100, 14.3% savings): distil 100% answer recall / 100% support
recall, truncation at 14.1% savings 91.6% / 82.7%.

### 3. Recall is the headline; precision belongs to the savings number

The loss is asymmetric. A dropped fact is a wrong answer; retained-but-unneeded content
is merely fewer tokens saved. So we report recall and let savings carry the other side,
rather than blending both into an F1 that hides which direction the error ran.

### 4. Three states, not two — and recoverability must be *proven*

Facts are classified `retained` / `recoverable` / `lost`. `recoverable` means the fact is
absent from what the model sees but reachable via `distil_expand`. This distinction only
exists because distil is reversible, and it is the first time that property has had a
number: on the corpus, reversibility is worth **9.8% recall** (90.2% visible → 100%).

Crucially, recoverability is **verified, not inferred**: the handle must appear in *that
block's own* compressed text, and the fact must appear in *that handle's* restore bytes.
Approaches that infer recoverability from a retrieval marker appearing anywhere in the
prompt overcount, and can only claim comparative validity. Ours is absolute, and two
tests pin it (`test_handle_without_the_fact_is_lost_not_recoverable`,
`test_foreign_handle_is_not_credited`).

We also report **visible recall** alongside true recall, because "lossless via a tool
call" still costs a round trip. On SQuAD (84.8% savings) true recall is 100% but visible
recall is 0% — every answer sits behind a handle. That is honest and actionable, and the
report says so explicitly rather than printing only the reassuring number.

### 5. The live meter scores real traffic without ever writing content

Corpus and public benchmarks are still not *your* traffic. The obvious fix — record real
sessions and score them offline — would mean prompts and tool output in plaintext on
disk, breaking the content-free posture that the census, ledger, and shadow ledger all
maintain. **We will not add a plaintext session recorder.**

Instead the meter scores **in-process**, where original and compressed text are both
already in memory, and persists **counts only** (three integers per dimension). There is
deliberately no field in the record that could carry content, and a test plants a unique
marker in tool output and asserts it never reaches disk.

Request-path rules, matching shadow mode: **sampled** (`--retention RATE`), **bounded**
(64 KiB scanned per request), **fail-open** (any exception is swallowed). It defaults to
`0.05` under `wrap` and `0` under a bare `proxy` — unlike `--shadow` it costs no extra
tokens, because there is no second upstream call.

### 6. Gate placement follows cost, and silence is never success

| Gate | Where | Why |
|---|---|---|
| `distil retention --max-lost 0` | per-commit CI | zero cost, no network; Tier-0/1 are lossless by construction, so any lost fact is a real regression |
| `distil retention --dataset …` | nightly | needs a third-party host; a required check gated on someone else's uptime makes main's health hostage to it |

The dataset command exits non-zero on a fetch failure, on an upstream schema change, and
on **compression that did not engage** (<1% savings) — because a recall number produced
by a near-identity transform is arithmetic, not evidence. Nightly fails hard if *both*
benchmarks fail, since that means no external-validity signal was collected at all.

### 7. HTML gets a reversible transform, because the harness found it had none

Building the above exposed a capability gap the harness could then measure: distil
compressed **0.0%** of an HTML tool result. The cause is structural rather than a
missing heuristic — minified markup is one enormous line, so Tier-1's line-folding has
nothing to fold and the JSON/record folds do not recognise markup. Every agent with a
fetch or browser tool was paying full price for `<script>`, `<style>`, and chrome.

`compress/htmlx.py` extracts content with stdlib `html.parser` (no lxml, no bs4, no
readability port — the core stays dependency-free) and records the exact original under
the marker's handle. Real pages: Wikipedia 281,093 → 14,260 tokens (**94.9%**), Python
docs 32,322 → 4,229 (**86.9%**), **0 facts lost** in both. Both measured end-to-end
through the adapter (the envelope a user actually sends), not on the extractor alone.

The extraction is deliberately **recall-biased**, for the same asymmetry as §3: only
tags that cannot hold article content are dropped outright, plus four unambiguous
chrome landmarks (`nav`/`footer`/`aside`/`form`). `<header>` is *kept* — it usually
wraps the `<h1>` — and `img` alt text is kept because it is often a figure's only
description. `div`/`section` are never dropped: their meaning is site-specific.

Being reversible is what licenses that aggressiveness. A lossy extractor's mistake is
permanent; ours costs one `distil_expand`. Skipped in `verbatim` mode (no expand tool
exists to recover with, so an unrecoverable elision is forbidden) and on recency-exempt
blocks (the agent's most recent output must stay byte-exact).

**Rejected: keeping `href` URLs inline.** On raw HTML most probe-able artifacts are
link URLs; retaining them would cut savings hard on link-dense pages, and they remain
recoverable. The consequence is honest and now *measured* rather than assumed: visible
recall on the artifact dimension is low for HTML (4.8% overall on the Wikipedia page)
while true recall is 100%. If evidence from real expand signals says agents need those
links, the meter will show it and the default can change on data.

**Detection is conservative.** A doctype/`<html>`/`<body>` landmark accepts outright;
without one, tag density alone is insufficient — a log line reading `parse failed near
<div>` repeated 40 times clears any density bar. Markup closes its elements, so a
doctype-less fragment must also show closing tags substantially balancing its open
ones. This was a real false positive caught by its own test, not a hypothetical.

### 8. Findings from cross-audit, and what they changed

The PR's independent cross-audit raised three findings. All three reproduced, so all
three are fixed — and one of them exposed a defect in the metric itself that had been
flattering us.

**Unclosed chrome tags swallowed the article.** Chrome skipping keyed off a matching
close tag, and real HTML frequently never sends one. Content *before* an unclosed
`<aside>` was emitted while everything after it — the actual article — was dropped.
(The fully-swallowed case was benign: extraction returned empty, declined, and the
block passed through uncompressed. The partial case was not.) Fixed with two bounds: an
`<article>`/`<main>` landmark ends any active skip, since an unclosed sidebar cannot
legitimately contain the page's main content; and a skipped-data budget abandons the
skip on `<div>`-built chrome where no landmark follows. `script`/`style` are exempt from
the budget — their payload is legitimately huge.

**Synchronous disk I/O in the request path.** `RestoreStore.expand()` falls back to a
disk read for handles it does not hold in memory, and the live meter called it for every
`handle=` match in the compressed text. A long transcript — or tool output that merely
*contains* the string `handle=deadbeef`, which distil's own adversarial harness plants —
would turn one sampled request into a series of file reads. The meter now intersects
matched handles with the store's in-memory set, which removes the disk path entirely and
is also the only recoverability provable for the current turn.

**Short answers false-matched in the recovered bytes.** `case.answer in recovery` is a
bare substring test: the gold answer `"12"` matches inside `file-12.csv`, and `"5"`
matches almost anything. Recoverability now requires token boundaries, and values
shorter than four characters require whitespace boundaries — erring toward *not*
crediting recovery, which is the safe direction for a metric whose job is to surface
loss.

That third fix then failed the corpus gate, which is how it earned its keep: tightening
the match revealed that `_NUMERIC_RE` had been extracting values that end **mid-token** —
`invoices=88` clipped out of `invoices=88ms`, and junk like `T09:14` carved out of a
timestamp. Those are strictly worse probes (the first loses its unit and would
false-match `invoices=889`; the second is not a fact at all), and they could not satisfy
a right-boundary test. Extraction now captures the whole token including its unit and
refuses to start or end mid-token. The probe set is smaller and cleaner as a result —
417 facts rather than 503 — which moved the corpus figures to **90.2% visible / 100%
true / 0 lost**, and the measured value of reversibility from 12.3% to **9.8%**. The
earlier number was inflated by junk probes; this one is the honest one.

### 9. The HTML transform is certified (added after 1.38.0)

§7 shipped the transform default-on while noting it had no corpus coverage. That was
debt, and ADR 0003/0004 are explicit that a new content type is certified before it goes
default-on — so `corpus/web-research.json` now carries it: a 4-turn web-research agent
whose tool results are real HTML documents, chrome and all, with the graded DECISION
inside the `<article>`. It certifies at 89.8% savings, and an extractor patched to
swallow `<article>` drops `match_rate` to 0.0, so the fixture can fail.

Two things it taught immediately.

**The synthetic oracle compares markup, not meaning.** `DeterministicRunner` splits each
*line* on `DECISION:` and compares the remainder as an exact string. With a document
minified onto one line, the baseline remainder drags the following markup along
(`…fetching.</p></article><aside class=`) while the extracted text does not — a
divergence about tags, not about the decision. The fixture puts the marker alone on its
line so the gate measures what it means to. Worth remembering before reading any future
divergence on markup-bearing content as a real one.

**The retention headline was fact-weighted, so corpus composition moved it.** Adding this
trajectory took the corpus from 417 facts to 1083 and the reversibility figure from 9.8%
to 62.6% — not because anything improved, but because `web-research` alone contributes
666 facts at 4.4% visible, HTML being dense in `href` URLs that extraction drops as
navigation. The other seven still read 90.2% visible.

### 10. The headline is a per-domain macro average

A number one fixture can move by 53 points is measuring the corpus, not the compressor.
So the headline is now the **mean of the per-domain ratios, each domain counted once**:
**21.4%**, against 62.6% fact-weighted.

Macro is the right default here because the corpus is a *deliberately* unbalanced sample
— it exists to cover diverse agent shapes, not to model a traffic distribution, so
weighting a domain by how many probe-able facts its fixtures happen to contain has no
meaning to recover. Under macro, adding a ninth domain moves the figure by at most 1/9
of its distance, and adding fixtures *within* a domain cannot move it at all.

Micro is still printed beside it rather than dropped, because the two diverging is
information: it says one domain dominates the fact count. The report also gained
per-domain rows, which show what a single number cannot — the spread runs from 0.0% on
`coding` to 95.6% on `web-research`. The JSON payload leads with `macro` and tags `micro`
with the reason it moves, so a consumer cannot grab the swingable number by habit.

`--max-lost` is unaffected: a lost fact is a count, not a ratio, and no averaging choice
changes whether something was unrecoverable.

## Consequences

- distil can state a quality claim checkable against data the reader already trusts.
- The reversibility moat has a measured value (+9.8% recall on the corpus, +8.4% answer
  / +17.3% support vs matched-savings truncation on HotpotQA) instead of an argument.
- New maintenance surface: two third-party dataset shapes. Adapters return `None` on
  anything unexpected and the loader raises rather than reporting a short run, so an
  upstream schema change fails loudly.
- Two findings worth recording, both discovered by building this and both initially
  wrong in our favour:
  - HotpotQA's yes/no comparison answers are never spans in the passage. Grading them
    naively reported the dataset's answer *format* as 10% compression loss. Answers are
    now graded only when present in the **uncompressed** context.
  - distil's compressors correctly decline to touch short natural-language prose, so a
    bare-prose framing of these benchmarks yields 0% savings and a vacuous 100% recall.
    The default `--shape json` reflects how a retrieval tool actually returns documents
    (a JSON array, which is what distil compresses); `--shape prose` remains available
    and warns when compression did not engage.
