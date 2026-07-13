# Spec: Query-Aware Salience — Phase 2 (learned relevance scorer)

**Status:** implemented (1.16.0) — `distil/compress/query_relevance.py` (model + features + gated
plug), `distil/query_flywheel.py` (content-free dark collection), `distil/query_train.py` (train +
promotion gate), `distil query-relevance` (CLI). Additive-only over phase 1; inert until a
retrain promotes weights, so shipping is behavior-neutral. Validated end-to-end (a `claude -p`-
confirmed semantic answer that phase 1 folds is recovered by phase 2), full gate + bench + verify PASS.
**Author:** distil core (assisted)
**Date:** 2026-07-12
**Builds on:** `specs/query-aware-salience.md` (phase 1, shipped 1.15.0)
**Reuses:** `distil/codec/learned.py`, `distil/codec/features.py`, `distil/online.py`, the expand flywheel (`distil/expand.py:record_signal`, `distil/proxy.py` `_learn_stats.record_digest`/`record_expand`), `distil/certify/gate.py`

## Context — why phase 2

Phase 1 keeps a tool-output line when it **lexically** shares a salient token with the agent's
intent (its `tool_use` args + latest ask). That closes the common case — a `grep` hit, a config
key, a SHA — because the answer literally contains the term the agent used.

It misses **semantic** relevance. The agent asks "what's the retry limit?"; the answer line reads
`max_attempts = 5` — zero lexical overlap with "retry limit", so phase 1 folds it (recoverable via
`distil_expand`, but a round-trip). Synonyms, paraphrase, and indirect references are exactly where
a fixed lexical rule can't know which line matters.

**Goal:** learn line↔query relevance from distil's own traffic and pin semantically-relevant lines,
while keeping every phase-1 invariant (additive-only, reversibility structural, certified before
promotion) and distil's zero-runtime-dependency, pure-Python posture.

## The insight — the flywheel already labels this for free

distil already records, per block, whether the agent **expanded** it: `record_digest` /
`record_expand` (`proxy.py:552,612,711`) and `record_signal` (`expand.py:84`). An expand is a
*label*: "given the conversation at that point, this folded content was actually needed." Phase 2
adds the missing dimension — the **query** — to that label:

- **Positive:** a block (its lines) that was expanded, paired with the intent terms live at digest
  time (phase 1 already computes these in `compress_messages` via `extract_intent`).
- **Weak negative:** a block digested and never expanded within the session, under its query.

No new labeling effort, no human annotation — the moat traffic is the training set.

## Design

### 1. Query-conditioned features
`featurize(line, kind)` (`codec/features.py:40`) currently ignores the query. Add an optional
`query_terms` arg and a small block of query features appended to `FEATURE_NAMES`:

- lexical-overlap count (line tokens ∩ query terms),
- has-rare-query-term (a query term that is selective in this block — the phase-1 signal, as a
  feature),
- normalized distance to the nearest lexical query hit (relevant lines cluster near hits),
- query-term-in-line boolean, line position, digit density (already present).

Keep it cheap and embedding-free — the whole point is no transformer, no dependency.

### 2. Query-conditioned logistic model
Extend `LogisticKeepModel` (`codec/learned.py:59`) — same `sigmoid(dot(weights, featurize(...)))`,
just the longer feature vector — into a `QueryRelevanceModel` (or a `query=` param on the existing
class). Persisted to `codec/weights.json`'s sibling. It implements the same `KeepModel` protocol,
so it drops into the existing plug point.

### 3. Training + promotion — reuse `online.py` verbatim
`online.py` already is the pipeline: `collect_causal_labels` → `retrain` (logistic regression +
train/test split + metrics, `online.py:122`) → `certify_promotion` (TOST non-inferiority on every
corpus trajectory; promote **only if ALL pass**, `online.py:208`). Phase 2 changes only the label
source (expand-flywheel tuples with query terms) and the feature function. The retrain math and —
critically — the **certification gate** are reused unchanged.

### 4. Integration — additive, on top of phase 1
`intent.relevant_lines(lines, intent)` stays as the deterministic floor. Phase 2 adds: score each
line with the learned model; union the above-threshold indices into the keep set. The union with
phase 1 means the learned scorer can only **widen** keeps — it never removes a lexical or base keep.
If no trained weights are present, behavior is exactly phase 1.

## Invariants (unchanged from phase 1, re-verified)

1. **Additive-only.** Learned keeps ∪ phase-1 keeps ∪ base keeps. The model may only widen; a
   mis-score can waste a little compression or (safely) miss a line that stays recoverable via
   `distil_expand`. It can never make a line unrecoverable — the full block is always in
   RestoreStore.
2. **Certified before promotion.** New weights ship only if `certify_promotion` (`online.py:208`)
   passes TOST non-inferiority on the whole corpus — the same LTT/CRC gate the deterministic
   strategies pass. No weights promote on a decision-equivalence regression.
3. **Monotone / never-regress.** Like `guideline.py`, retraining only ever tightens toward keeping
   what was needed; a worse candidate fails the gate and is not promoted.
4. **Zero runtime deps.** Pure-Python logistic over cheap features, weights in JSON. No model
   server, no transformer, no new dependency.

## Verification

- **Offline:** precision/recall of "was this line expanded under this query" on a held-out split
  (`online.retrain` already reports train/test metrics); require the learned scorer to beat the
  phase-1 lexical baseline on recall at equal kept-fraction.
- **Certificate:** `certify_promotion` must pass on all corpus trajectories (hard gate, reused).
- **Live shadow:** the expand-rate should fall further than phase 1 — fewer round-trips because the
  semantically-needed line is already inline. That drop is the measurable win, tracked via the
  shadow counters shipped in 1.15.0.

## Rollout

1. **Collect (dark):** extend the flywheel record to carry intent terms at digest time; accumulate
   `(query_terms, line_features, was_expanded)` tuples. Ship this recording first — it is content-free
   and safe — and let labels accumulate from real traffic.
2. **Train + certify offline:** `retrain` → `certify_promotion`. Only promote weights that pass.
3. **Enable behind the plug:** union the learned scorer over phase 1; shadow-measure the expand-rate.
4. **Iterate:** periodic retrain as labels grow; each candidate re-certified before promotion.

## Cost / risk

- **Cost:** a training run (offline, cheap — logistic on cheap features) plus label storage. No
  runtime cost beyond a dot product per line.
- **Risk:** the model keeps the wrong lines (wasted compression) — bounded by the additive union and
  caught by `certify_promotion` before it ships. Missing the right line is safe (reversible). The
  gate makes a bad model un-shippable, not dangerous.

## Why phase 1 shipped without this

Phase 1 (lexical) already keeps the needle in the common case and is the floor phase 2 improves on.
Phase 2 needs accumulated flywheel labels that only exist after 1.15.0 is in real use — so it is
correctly sequenced *after* phase 1, not bundled with it.
