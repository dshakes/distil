# #2 — Ship a certified learned-salience model

**Status (2026-07-21): built and gated, data-starved. Not a coding task — a data task.**

The competitive framing (vs Headroom's shipped Kompress-v2 ModernBERT/ONNX model) is
"distil ships only hand-tuned heuristics." That's already false in *architecture* — the
learned path exists end to end — but true in *practice*: no certified weights have been
produced yet, so `query_relevance.get_model()` returns `None` and the runtime is exactly
phase-1 lexical. Closing the gap means producing weights that pass the certification gate,
which requires **labeled data volume we don't yet have**, not new code.

## What already exists (verified)

- **Model** — `distil/compress/query_relevance.py::QueryRelevanceModel`: pure-Python
  `sigmoid(dot(weights, x))` over cheap per-line features, weights in JSON. Zero runtime
  deps (no ONNX, no native ext — the moat holds). `get_model()` returns `None` until
  certified weights are present, so an absent/rejected model is *exactly* phase 1.
- **Data collection** — `distil/query_flywheel.py`: content-free dark-collection, **already
  enabled in the running proxy** (`proxy.py:323`, sample_rate 0.25). Per dropped line it
  records only numeric query-features keyed by the block's content-free handle; the expand
  side logs the handle. Training joins by handle: a dropped-line row under a later-expanded
  handle is a **positive** (the model should have kept it), under a never-expanded handle a
  **negative**. No content, ever.
- **Training + certification** — `distil/query_train.py::certify_and_promote`: trains the
  logistic on the joined labels, runs `online.certify_promotion` (the decision-equivalence
  gate — a candidate that could regress equivalence is rejected), and writes
  `query_weights.json` **only on pass**. Maker ≠ checker: the trainer cannot self-promote.

## The only blocker: label volume

`~/.distil/query_flywheel.jsonl` has **16 rows**. Positives require real `distil_expand`
calls, which are infrequent — organic accumulation is slow. A model trained on tens of rows
overfits; the cert gate correctly rejects it (which is the system working, not failing).
Rough target: **≥ ~2–5k labeled dropped-line rows spanning ≥ a few hundred expands** before
`certify_and_promote` has any chance of clearing the gate.

## Path to shipping (decision points for the maintainer)

1. **Accumulate.** Collection is already on for every proxy session. Left alone it grows at
   whatever the real expand rate is — could be weeks. **Decision:** accept organic pace, or
2. **Bootstrap the corpus (faster).** Replay a labeled trace set (e.g. SWE-bench agent runs
   with known-needed lines) through the digest+expand path to synthesize positives at volume.
   Content-free features only, so no privacy cost. **Decision:** worth the setup, or not?
3. **Train + certify** — one command when data is sufficient:
   `python -c "from distil import query_train as qt; print(qt.certify_and_promote(...))"`.
   Passes → `query_weights.json` written, `get_model()` starts returning the model, phase-2
   semantic pins activate additively (only ever widen the keep set). Fails → nothing changes,
   report says how far short (need more data / precision too low).
4. **Compute is trivial** — logistic over cheap features, CPU-seconds. No GPU, no budget ask.

## Honesty rule

Never ship weights that didn't clear `certify_promotion`. An uncertified model that "looks
good on train" is exactly the inert-artifact trap the competitor fell into. `get_model()`
== `None` is the correct, safe default until real certified weights exist.

## Readiness check (run anytime)

```
python -c "from distil import query_flywheel as q; p=q._default_path(); print('rows:', sum(1 for _ in open(p)) if p.exists() else 0)"
```
When rows are in the thousands with a healthy positive fraction, attempt `certify_and_promote`.
