# Visible recall: keep what the agent acts on in view

The Headroom 0.42 head-to-head (`research/headroom-0-42-h2h`) found distil ahead on tokens
and true recall, but behind on **visible** recall (facts left in the text the model reads,
not behind a `distil_expand` handle): default 86.1% vs 94.3%, aggressive 55.2% vs 79.0%.
Every hidden fact the agent needs costs a round trip or a wrong step.

## Method

* Same 360 real tool outputs (60 per content type), same scorer (`headroom_h2h.score`,
  i.e. `distil.retention`'s fact extraction), same tokenizer (o200k).
* Split, fixed before tuning: per content type, sort sample ids by
  `sha256("visible-recall-split-v1" + id)`; first half = dev (180), rest = held-out (180).
  `ids_sha256` in `results.json` pins each set.
* All tuning (the 0.10 identifier budget, which levers to keep) on dev only. Held-out was
  run once on the tuned code. One change came after it: the held-out run showed 4 blocks
  whose digest cost *more* o200k tokens than the original (reject-if-bigger was checked in
  characters only), so a token check with a 5% margin was added and both splits re-run.
  That is an invariant fix, not a tuned knob; it is disclosed because it touched code
  after the held-out set was seen.
* Headroom rows are the h2h run's own per-sample results; distil baseline rows are that
  run's `distil-*` arms (origin/main code, no distil diff on that branch).
* Paired percentile bootstrap, 10k resamples, 95% CI, per-sample differences.

## Held-out results (n=180)

| arm | token reduction | visible recall | true recall |
|---|---:|---:|---:|
| distil default, main | 13.7% | 86.1% | 100.0% |
| **distil default, new** | **11.0%** | **98.4%** | 100.0% |
| Headroom default | 7.1% | 92.4% | 99.1% |
| Headroom `coding` profile | 6.1% | 93.7% | 99.1% |
| distil aggressive, main | 41.9% | 55.4% | 99.7% |
| **distil aggressive, new** | **29.2%** | **88.7%** | 100.0% |
| Headroom `agent-90` | 23.9% | 81.0% | 99.3% |

Paired differences (new minus other), percentage points:

| comparison | reduction | visible recall |
|---|---|---|
| default new − main | −2.8 [−4.5, −1.2] | +12.3 [+8.4, +16.5] |
| default new − Headroom default | +3.8 [−0.7, +8.3] | +6.0 [+2.2, +10.3] |
| default new − Headroom coding | +4.8 [+0.3, +9.3] | +4.7 [+1.1, +8.7] |
| aggressive new − main | −12.7 [−16.9, −8.9] | +33.2 [+26.7, +39.9] |
| aggressive new − Headroom agent-90 | +5.3 [−2.2, +13.1] | +7.6 [+1.3, +13.8] |

Both tiers are now at or past Headroom on both axes. The visible-recall lead is
significant at both tiers; the token lead is a point estimate whose CI crosses zero
against Headroom default and agent-90 (it clears zero against the coding profile).
True recall is 100% in every new arm (0 facts lost).

Not closed: **prose**. Headroom agent-90 cuts prose 37.5% at 70.2% visible; distil cuts
2.7%. No lossless/structural prose reduction was found worth shipping, and no lossy prose
model was added (it would have to pass the certification gate first).

## Overfitting guard: `distil retention` on corpus/*.json (offline Tier-1 path)

| | mean savings | macro visible recall | micro visible recall | recall |
|---|---:|---:|---:|---:|
| main | 9.65% | 76.9% | 37.7% | 100% |
| new | 8.37% | 82.5% | 39.8% | 100% |

Same direction, smaller size: the corpus is dominated by one web-research domain the
change does not touch.

Caveat: the identifier keep and the scorer share a notion of "fact" (paths, URLs, hashes,
key=value). The rule was written as "what an agent acts on", but the h2h scorer rewards it
directly; the retention corpus is the same matcher on different data, not an independent
judge. An agent-outcome A/B is what would settle it.

## What changed

1. `keep_policy.identifier_lines`: the digest keeps the densest identifier lines (distinct
   unseen identifiers per char) within 10% of the block; lines the noise dedup governs are
   never re-added, so true repetition still folds.
2. Cold-point eviction stub carries one sample per shape of the must-keep lines plus the
   same identifier lines (still a pure function of the content, still reject-if-bigger).
3. `htmlx`: link targets (`href`/`src`) survive extraction; a doctype-less block whose tags
   are mostly not HTML elements (a docs API's `<paragraph h="…">` XML) is not HTML, so its
   node handles are no longer stripped as chrome; a JSON envelope is never extracted whole:
   only string values that are pages are.
4. Adapter digest is declined unless it saves at least 5% tokens (local estimate).

Ruled out on dev: folding JSON/templates in place before the digest (+0.4 pp visible,
−0.1 pp tokens, broke expand-loop and census expectations).
