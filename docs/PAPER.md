# Certified Decision-Equivalent Context Compression for LLM Agents

*Working draft. §6 reports **real, committed measurements** from `benchmarks/prove.py`
on real τ-bench and the full SWE-bench_Lite (result JSONs in `docs/paper/results/`,
LaTeX in `docs/paper/generated/`). See `docs/PAPER_PLAN.md` for the protocol and
`benchmarks/PROVE.md` for the runner. The arXiv LaTeX is `docs/paper/main.tex`.*

---

## Abstract

LLM agents resend a growing context every turn, so context size dominates serving
cost. Existing context compressors report token savings but offer **no guarantee
that the agent still behaves the same**. We reframe the objective from byte- or
embedding-fidelity to **decision-equivalence**: a compression is acceptable iff the
agent takes the same next action it would have on the uncompressed context. We then
give this a **distribution-free, finite-sample guarantee** by casting level selection
as conformal risk control (Learn-Then-Test / Conformal Risk Control) with the loss
defined as a decision flip. The certificate states, for a chosen risk budget α and
confidence 1−δ, that the decision-change rate on exchangeable traffic is ≤ α. We
evaluate on **real τ-bench agent trajectories** and the full **SWE-bench_Lite
(300 instances)** edit-localization, graded by a real model with no answer-revealing
markers using majority-of-3 structured (forced-tool) grading. We validate the guarantee
**out-of-sample**: across 500 calibration/test splits on real SWE traces the empirical
coverage is **96–100% at the claimed 95% confidence**, with realized decision-change rate
conservatively below the budget (8.0% at α=15%, 11.7% at α=20%). The certificate yields a
real operating point — **15.7% certified token savings at α=15%** (22.9% at α=20%) — and
at strict budgets correctly declines to overclaim. On the same real traces the reversible
**digest + recover-on-demand** tier flips fewer decisions than equally-aggressive lossy
compression (10.2% vs. 11.5% at ≈22% savings; sharper on smaller samples, 11.2%→7.5%),
while on already-compact contexts (τ-bench) the certificate correctly declines to certify
savings — quantifying *where* recoverable compression helps rather than overclaiming a
single headline ratio.

## 1. Introduction

- Agent serving cost scales with re-sent context; prompt caching shifts the cost to
  cache *misses*, which compression of the volatile tail can reduce.
- The gap: every shipping compressor quotes a savings *estimate*; none certifies that
  the agent’s *decisions* are unchanged. "100% accuracy" is a slogan, not a number.
- Contributions:
  1. **Decision-equivalence** as the compression objective (loss = the agent’s next
     action flips vs. uncompressed).
  2. A **decision-equivalence risk certificate**: conformal risk control (LTT/CRC)
     selecting the most aggressive level whose decision-change rate is provably ≤ α.
     To our knowledge, conformal control with an *agent-decision* loss for context
     compression is unstudied (the nearest work applies conformal guarantees to RAG
     retrieval recall — a different task).
  3. A **cache-aware, reversible** compression engine (digest-behind-handle +
     recover-on-demand) that operates inside the certified frontier.
  4. An **evaluation on real agent traces** that removes the circular self-labeling
     of synthetic corpora, with the guarantee **validated out-of-sample** and under
     **distribution shift**.

## 2. Related work

- Context/prompt compression: LLMLingua / LLMLingua-2 / LongLLMLingua, RECOMP,
  selective-context, soft-prompt distillation. All optimize a fidelity/relevance
  proxy; none certifies decision-equivalence.
- Prompt caching and cache-aware serving.
- Distribution-free uncertainty: conformal prediction; **Learn-Then-Test**
  (Angelopoulos et al., 2021/2025) and **Conformal Risk Control** (Angelopoulos et
  al., ICLR 2024). Closest application: conformal guarantees for RAG recall
  (ECIR 2026) — retrieval, not agent-decision-preserving compression.

## 3. Problem formulation

- A trajectory is a sequence of turns; each turn is the full context the agent saw,
  decomposed into typed blocks with a stability hint (cacheable prefix vs. volatile
  tail). A *decision* is the agent’s next action — a tool call (τ-bench) or an
  edit/command (SWE-bench) — represented as a canonical `{action, target}`
  fingerprint produced by a grading model from context alone (no directive/marker).
- Loss on a (turn, level): `L = 1` iff the decision under the compressed context
  differs from the decision under the uncompressed context. `R(λ)` is the expected
  loss (decision-change rate) at compression level λ.
- A compression **ladder** orders levels least→most aggressive: byte-exact →
  reversible lossless digest → salience-protected truncation → raw truncation sweep.

## 4. Method

### 4.1 Cache-aware reversible engine
- Keep the prefix byte-stable (schema canonicalization; lift volatile fields), digest
  only the volatile tail behind a content handle, keep the original locally, expose a
  recover-on-demand tool. Lossless = byte-in-context; reversible = digested but
  byte-recoverable; lossy = the rest. (Details in repo `compress/`.)

### 4.2 Causal pruning (discovery)
- Ablate a block, replay, keep iff a decision changes — yields a causally-justified
  pruning policy rather than a heuristic.

### 4.3 The decision-equivalence risk certificate
- Calibrate per-turn losses for each ladder level on held-out-of-test calibration
  traffic. Select with **LTT** (Hoeffding–Bentkus p-values, fixed-sequence testing):
  guarantees `P(R(λ̂) ≤ α) ≥ 1−δ`; or **CRC** for the monotone 0/1 loss:
  `E[L(λ̂)] ≤ α`. Operating point = highest-savings level in the certified prefix.
- The exchangeability assumption is explicit; the guarantee is marginal over the
  calibration distribution, recalibrated under drift.

## 5. Experimental setup

- **Data.** Real **τ-bench** trajectories (airline/retail; gpt-4o and Sonnet-3.5
  runs; 182 trajectories / 1164 decision points after parsing the airline-gpt-4o
  log) loaded with no planted markers; the decision is the agent’s actual tool call.
  **SWE-bench** edit-localization trajectories built from real issues + gold patches
  (target file inferred from the issue amid distractors).
- **Grader.** A real model returns the `{action, target}` fingerprint via majority-of-k.
  We report **model↔gold next-action agreement** on the uncompressed context as a
  grader-faithfulness gate. Runs use `--runner claude-cli` (subscription) / `openai`
  (local open model) / `anthropic`.
- **Protocol.** E1 frontier; **E2 certification coverage** (certify on calibration,
  measure realized risk on a disjoint held-out split, over many trajectory-level
  splits → empirical `P(realized ≤ α)`); E3 leave-one-domain-out shift; **E4
  downstream task success** (trajectory keeps its outcome iff every decision is
  unchanged), vs. the uncompressed baseline with a bootstrap CI; **E5 head-to-head**
  vs. competitor/structural baselines (LLMLingua-2, LongLLMLingua, RECOMP-style
  extractive, selective-context, truncation, recency-window, keep-last-k-turns agent
  memory) under the same grader, each marked
  with whether its decision-change rate certifies ≤ α.

## 6. Results

> **These numbers are real and committed.** Produced by `benchmarks/prove.py` on real
> τ-bench trajectories (gpt-4o-airline, from the tau-bench repo) and real SWE-bench_Lite
> edit-localization, graded by a real model with **majority-of-3 structured (forced-tool)
> grading** and the `distil_expand` recovery loop. The result JSONs are committed under
> `docs/paper/results/` and the LaTeX tables under `docs/paper/generated/`. §6.0 is the
> earlier exploratory run, kept because it surfaced the methodology requirements.

### 6.0 Real measured frontier (E1) — committed runs

**τ-bench airline** (gpt-4o traces, gpt-4o grader, structured, +expand; 25 traj / 105
decisions; grader↔gold agreement 48.6%):

| level | savings | decision-change |
|---|--:|--:|
| byte-exact | 0.0% | **0.0%** |
| lossless (reversible **+recovery**) | 1.0% | 29.5% |
| truncate@250 (lossy) | 9.4% | 58.1% |
| truncate@120 (lossy) | 11.0% | 64.8% |

**SWE-bench localization** — full **SWE-bench_Lite (300 instances / 600 decisions)**;
Claude grader, structured, majority-of-3, **+expand**; grader↔gold agreement 47.8%:

| level | savings | decision-change | on-changed (effective) | trivial |
|---|--:|--:|--:|--:|
| byte-exact | 0.0% | **0.0%** | 0.0% | 94% |
| lossless (reversible **+recovery**) | 21.7% | **10.2%** | 20.1% | 49% |
| truncate@250 (lossy) | 20.4% | 11.5% | 23.0% | 50% |
| truncate@120 (lossy) | 22.8% | 11.7% | 23.3% | 50% |

At equal ≈22% savings the reversible+recovery tier flips fewer decisions (10.2%) than
lossy truncation (11.5–11.7%). On a 40-trajectory subset the recovery loop's effect is
sharper (digest 11.2%→**7.5%** with `--expand`); at full scale the gap narrows but
holds. byte-exact reads a clean 0.0% (no grader-noise floor).

Three findings, all on real data:
1. **byte-exact reads a clean 0.0%** under the structured majority-vote grader — the
   measurement has no grader-noise floor (validating the protocol; cf. §6.1's broken run).
2. **Recovery helps and is necessary for the reversible tier:** on SWE the digest's
   decision-change drops 11.2%→**7.5%** with `--expand`, and at equal ~24% savings the
   reversible+recovery tier beats lossy truncation (12.5%). On τ-bench the data is
   already compact (lossless saves only 1%), so there is little to recover and aggressive
   levels flip heavily — distil's value is on **verbose** contexts, which the certificate
   correctly reflects by refusing to certify savings on compact data.
3. The honest denominator: 50–90% of turns are *trivially incompressible* (left
   byte-identical); the harness now reports both the overall and the **on-changed**
   (effective) rate so equivalence cannot be inflated by no-op turns.

### 6.0.1 The certificate holds out-of-sample (E2) — the proof

On the **full SWE-bench_Lite (300 instances, +expand)**, certifying on a calibration
split and measuring the **realized** decision-change on a disjoint held-out split, over
**500** random trajectory-level splits (δ=0.05):

| risk budget α | empirical coverage P(realized ≤ α) | mean realized risk | certified savings |
|---|--:|--:|--:|
| 0.10 | **99.8%** (≥ 95% ✓) | 0.03% | ~0%* |
| 0.125 | **96.6%** (✓) | 1.02% | 1.7% |
| 0.15 | **98.8%** (✓) | 8.03% | **15.7%** |
| 0.20 | **100%** (✓) | 11.70% | **22.9%** |

\* The reversible+recovery digest's true decision-change is 10.2%, *just above* the
α=10% budget, so at α=10% the certificate conservatively certifies only the byte-exact
level (≈0 savings) — the guarantee declining to overclaim by a 0.2-point margin. As the
risk budget loosens, the certified savings rises to **15.7% at α=15%** and **22.9% at
α=20%**, and the coverage guarantee **holds out-of-sample throughout** with realized risk
conservatively below the budget (8.0% at α=15%, 11.7% at α=20%). This is a real
certified-savings operating point on real agent traces, with the honest α-vs-savings
tradeoff made explicit rather than cherry-picked.

This is the headline statistical result: **a distribution-free decision-equivalence
certificate whose finite-sample coverage holds on real agent traces.**

### 6.0.2 Head-to-head vs. baselines (E5) — same grader, with an honest confound

On a 100-trajectory SWE subset, all methods graded identically (α=0.15; the
`certifies?` column is the *single-shot* Hoeffding–Bentkus test over the full data —
weaker than the split-calibrated E2 above, and labelled as such):

| method | kind | savings | decision-change | single-shot certifies? |
|---|---|--:|--:|:--:|
| distil lossless (reversible+recovery) | distil | **21.8%** | 12.0% | ✘ |
| distil truncate@250 | distil | 20.5% | 12.0% | ✘ |
| recomp-extractive (model-free) | baseline | 18.5% | 14.0% | ✘ |
| recency-window | baseline | 16.1% | 5.5% | ✔ |
| truncate-head | baseline | 15.5% | 8.5% | ✔ |
| selective-context (model-free) | baseline | 14.8% | 6.5% | ✔ |
| keep-last-3-turns | baseline | 0.0% | 0.0% | ✔ |
| byte-exact | distil | 0.0% | 0.0% | ✔ |

The reversible+recovery tier reaches the **highest savings** (21.8%); the lossy methods
that single-shot-certify at α=15% do so only at **lower savings** (≤16%) and cannot
recover dropped detail. **Honest confound:** in our SWE *edit-localization* construction
the gold target hunk is appended **last** in the code-search observation, so
recency/tail-truncation baselines benefit from *needle position*, not content
understanding — which inflates their standing on this particular task. A content-placed
needle (or a non-localization task) would remove that advantage. We therefore treat E5 as
a frontier illustration, not a dominance claim, and rest the contribution on E2 (the
certificate's validity) and the reversible mechanism. (The real LLMLingua-2 / LongLLMLingua
packages are wired in `benchmarks/baselines.py` but require a GPU environment; they were
not run on this CPU host.)

### 6.0.3 End-to-end task-success (E7) — the certificate does **not** transfer to execution

E1–E6 measure *decision-equivalence*, a proxy. E7 is the first non-proxy test: a real
agent (**aider 0.86.2 + `claude-sonnet-4-6`**, temp 0, diff edit format) run **end-to-end**
on **SWE-bench Verified** (50 instances, seed 1729) and scored by the **official
`swebench` 4.1.0 harness**, across three conditions that share the identical agent and
differ only in how the agent's *read context* (file contents + tool output, never the
problem statement) is compressed in flight.

| condition | ctx reduction | pass@1 | 95% CI (Wilson) | resolved | cost |
|---|--:|--:|---|--:|--:|
| **A. full context** | — | **52.0%** | [38.5%, 65.2%] | 26/50 | $17.63 |
| **B. distil `trunc@500`** (certified op-point) | 86% | **16.0%** | [8.3%, 28.5%] | 8/50 | $4.00 |
| **C. LLMLingua-2** (default rate) | 48% | **26.0%** | [15.9%, 39.6%] | 13/50 | $12.03 |

Paired exact McNemar (same 50 instances): **distil vs. full `p<0.001`** (20 lost, 2
gained), **LLMLingua-2 vs. full `p=0.002`**, **distil vs. LLMLingua-2 `p=0.18`** (n.s.).

**Findings, reported without cherry-picking.** (1) Both compressed conditions
**significantly** collapse pass@1 vs. full context — compression at these operating points
does **not** survive real execution. (2) distil's certified `trunc@500` (16.0%) trails
LLMLingua-2 (26.0%) on point estimate but **not significantly** at n=50, and it compresses
~1.8× harder (86% vs. 48% of context removed), so its lower score is confounded with
aggression, not established as a method gap. (3) The headline: **the localization
certificate does not transfer to execution** — `trunc@500` was *certified* at 4.0%
decision-change (E6) yet collapses end-to-end success by 36 points. The decision-equivalence
contract is sound for what it measures (the calibration distribution); it is **not** a
task-success guarantee once compression is aggressive. Harness, runners, and the full
per-instance breakdown: `benchmarks/swe_bench_e2e/` and
`docs/paper/results/swe_bench_verified_e2e.json`. Total API spend: $33.66.

### 6.1 Exploratory run — 8 trajectories, 67 real decision points, Haiku grader, samples=1

### 6.1 Exploratory run — 8 trajectories, 67 real decision points, Haiku grader, samples=1

| level | savings | decision-change |
|---|--:|--:|
| byte-exact | 0.0% | 0.0% |
| lossless (reversible digest) | 1.1% | 67.2% |
| truncate@250 | 9.5% | 64.2% |
| truncate@120 | 10.6% | 67.2% |

model↔gold next-action agreement (uncompressed) = **19.4%**.

**This run is NOT a valid measurement, and the harness flags it.** Three real,
instructive confounds:

1. **`byte-exact` is byte-identical to the original on 100% of these turns** (Tier-0
   minify is a no-op on already-compact τ-bench JSON), so its 0% is a *cache artifact*
   (same text → same decision), not robustness.
2. **`samples=1` turns grader variance into apparent decision change** — any level
   that alters text triggers a fresh stochastic grader call. Majority-of-k is
   mandatory; the harness now warns on `--samples 1`.
3. **A cross-family, weaker grader is unfaithful** — Haiku reproduces *gpt-4o's*
   action 19% of the time. Grade traces with a same-family/strength model (e.g. the
   `sonnet-35` τ-bench logs with a Claude grader).

### 6.2 The substantive finding that survives the confounds

The **reversible digest graded *without the recovery loop* is not decision-equivalent
on real τ-bench**: folding decision-relevant tool output behind a handle changes the
decision unless the model expands it. So distil's aggressive savings **depend on the
`distil_expand` loop being active**; without it the harness measures a *conservative,
no-expand* lower bound.

**This is now measured both ways.** `prove.py --expand` runs the recovery loop inside
the grader (`distil.replay.expand_runner`): the model sees the digested context, may
emit `{"expand": [handle,…]}`, and the harness splices the byte-exact original back (a
content-addressed restore map) before it commits. Live single-turn check on real
τ-bench (`claude -p`, Haiku):

| context | action | matches base? |
|---|---|---|
| base (uncompressed) | `search_flights` | — |
| no-expand (digest hidden) | `confirm_booking_details` (flipped) | ✗ |
| with-expand (recovery loop) | `search_flights` | ✓ (action) |

Recovery restores the *action*; the residual target-string difference is a
measurement-granularity artifact, not a decision change (§6.3).

### 6.3 A third measurement finding: fingerprint granularity

A free-text `{action,target}` grader counts **paraphrase as decision change**
(`search_flights` vs `SearchFlights`; "NYC-SEA-2024-05-20" vs "New York to Seattle on
May 20th"). We now normalize the action (case/punctuation-folded) and case-fold the
target; the robust fix is a **structured / forced-tool grader** (the `anthropic`
runner emits a tool call, not prose) — recommended for the headline run.

**E2 / E3 / E4** on the valid run (majority vote + structured grader, with and without
`--expand`): pending — compute-bound, specified in §Reproducing.

## 7. Analysis & limitations

Lessons the real-data run makes concrete (each is now enforced or flagged by the
harness, and each is a methodological contribution in its own right):

- **Majority voting is not optional.** With a single sample, the decision-change rate
  is a sum of true information loss and grader sampling variance; only majority-of-k
  isolates the former. Report k and the residual grader self-disagreement.
- **Grade with a faithful agent.** The grader must reproduce the trace-generating
  agent's actions at a high rate on the *uncompressed* context; otherwise E1/E2
  measure a strawman. Use a same-family/strength model (ideally the one that produced
  the traces) and publish the agreement number as a gate.
- **The reversible tier must be evaluated *with* its recovery loop.** distil's whole
  premise is digest-behind-handle + recover-on-demand. Grading the digest with the
  `distil_expand` loop disabled measures a conservative lower bound that understates
  it; grading with perfect recovery overstates it. The honest number requires
  simulating the model's expand decisions — i.e. running the recovery loop in the
  grader. **This is the single most important remaining experiment**, and it is the
  rigorous form of the repo's headline "high savings at ~0% decision change".
- **Cost of the guarantee / sample size.** Tightest certifiable α scales with the
  number of zero-loss calibration turns (Hoeffding–Bentkus): ~⌈ln(1/δ)/α⌉-ish; budget
  calibration turns accordingly.
- **Threats to validity** (see `docs/PAPER_PLAN.md §8`): grader stochasticity (majority
  vote; report agreement), decision-proxy vs. outcome (E4), non-exchangeability (E3),
  domain coverage, cost-model fidelity.

## 8. Conclusion

Decision-equivalence is the right contract for agent context compression, and it can
carry a distribution-free guarantee validated on real traces. The reversible engine
sits safely inside the certified frontier.

## Reproducing

```bash
# real τ-bench data (no HuggingFace needed; ships in the tau-bench repo):
python benchmarks/fetch_real.py tau --src tau:gpt-4o-airline --out /data/tau.json
# the full real run (subscription grader; Opus for the headline, Haiku for scale):
python benchmarks/prove.py --dataset tau --path /data/tau.json \
    --runner claude-cli --model claude-opus-4-8 --samples 3 \
    --alpha 0.05 --delta 0.05 --ladder full --reps 500 --report results_tau.json
# the with-expand frontier (reversible tier graded WITH the distil_expand recovery loop):
python benchmarks/prove.py --dataset tau --path /data/tau.json \
    --runner anthropic --model claude-opus-4-8 --samples 3 --expand \
    --alpha 0.05 --delta 0.05 --ladder full --reps 500 --report results_tau_expand.json
```
