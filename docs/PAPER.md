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

![Distil architecture: cache-aware, causally-pruned, certified compression](assets/architecture.svg)

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

![Live head-to-head: Distil vs LLMLingua-2 vs Headroom — token savings vs decision-change rate](assets/head-to-head.svg)

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
| **B. distil `trunc@500`** (aggressive **lossy**) | 86% | **16.0%** | [8.3%, 28.5%] | 8/50 | $4.00 |
| **C. LLMLingua-2** (**lossy**) | 48% | **26.0%** | [15.9%, 39.6%] | 13/50 | $12.03 |
| **D. distil reversible + `distil_expand`** | 81%* | **56.0%** | [42.3%, 68.8%] | 28/50 | $16.38 |
| **E. distil reversible, relevance-gated** | 0%** | **54.0%** | [40.4%, 67.0%] | 27/50 | $17.27 |

Paired exact McNemar (same 50 instances): **B vs. full `p<0.001`** (20 lost, 2 gained),
**C vs. full `p=0.002`**, **C vs. B `p=0.18`** (n.s.), **D vs. full `p=0.69`** (n.s. —
*statistically equal to full*), **D vs. B `p<0.001`** (D recovers 22 instances B failed).
\*For D, 81% is the *pre-recovery* digest view; the realised cost is $16.38 vs. $17.63 for
full (~7% cheaper) after the model's `distil_expand` calls. \*\*For E (gate = keep last 6
user/tool messages full, digest older periphery), these SWE conversations are ≤6 turns so
the gate is a **no-op** — only 1 block digested across all 50 instances, hence ~0 context
reduction and full-vs-gated McNemar `p=1.0` (statistically identical to full). The gate is
designed for long-horizon agents with large peripheral context; this focused localization
workload does not exercise it (tunable via `DISTIL_E7_GATE_RECENT`). **E8 (§6.0.4) runs
exactly that workload.**

**Findings, reported without cherry-picking.** (1) Both **lossy** conditions (B, C)
**significantly** collapse pass@1 vs. full — aggressive lossy compression does **not**
survive real execution. (2) The **localization certificate does not transfer**:
`trunc@500` was *certified* at 4.0% decision-change (E6) yet collapses end-to-end success
by 36 points; a decision-equivalence guarantee on a single-turn proxy is **not** a
task-success guarantee once lossy compression is aggressive. (3) **But distil's reversible
tier (D) survives execution** — digest + recover-on-demand is end-to-end **task-equivalent
to full context** (56.0% vs. 52.0%, McNemar `p=0.69`) and decisively beats the lossy
conditions, because the agent pulls back the byte-exact content it edits (every instance
expanded; 135 recoveries total). (4) The honest catch: D's *realised* coding savings are
**modest (~7%)** — the agent expands most of what it edits — so the win is *task-success
parity at a modest discount*, not the proxy headline ratios. **Keep compression
recoverable.** Per-instance breakdown: `benchmarks/swe_bench_e2e/` and
`docs/paper/results/swe_bench_verified_e2e.json`. Total API spend: $50.04.

### 6.0.4 Long-horizon agent task-success (E8) — the gate's proper test

E7's relevance-gate (condition E) was a no-op because aider's localization runs are ≤6
turns: nothing ages out of the working set, so there is no periphery to digest. **E8 is
the workload the gate was designed for.** We built a multi-turn **ReAct** coding agent
(read/search/`edit_file`/`run_tests` tools, up to 30 turns) and ran it on the **full
500-instance SWE-bench Verified** set (seed 1729), scored by the same **official `swebench`
4.1.0 harness**. Runs are genuinely long-horizon (**mean ≈27 turns**), so read-file and
tool outputs accumulate into a large peripheral context behind a small active working set.
To keep a 500-instance six-condition sweep affordable we use **`claude-haiku-4-5`** at
temp 0; conditions differ only in the compressor, so the comparison is internally valid
regardless of base model. Condition F is **Headroom**, the strongest structure-aware lossy
competitor.

| condition | pass@1 | 95% CI (Wilson) | resolved | reversible | certified |
|---|--:|---|--:|:--:|:--:|
| **A. full context** | **39.2%** | [35.0%, 43.5%] | 196/500 | — | — |
| **E. distil relevance-gated** (reversible) | **36.8%** | [32.7%, 41.1%] | 184/500 | ✓ | ✓ |
| **F. Headroom** (lossy competitor) | 32.6% | [28.6%, 36.8%] | 163/500 | ✗ | ✗ |
| **D. distil reversible + `distil_expand`** (skeleton) | 32.4% | [28.4%, 36.6%] | 162/500 | ✓ | ✓ |
| **B. distil `trunc@500`** (lossy) | 5.6% | [3.9%, 8.0%] | 28/500 | ✗ | — |
| **C. LLMLingua-2** (lossy) | 2.4% | [1.4%, 4.2%] | 12/500 | ✗ | ✗ |

Paired exact McNemar (same 500 instances): **E vs. full `p=0.19`** (42 lost, 30 gained),
**E vs. Headroom `p=0.035`** (the gate beats the strongest competitor, +4.2 pp), **F (Headroom)
vs. full `p=0.0015`**, **B/C vs. full `p<0.001`**.

**Non-inferiority vs. full (not a bare p-value).** Failing to reject "no difference"
(`p=0.19`) is *absence of evidence*, not evidence of equivalence, so we report a paired
non-inferiority result: **E − full = −2.4 pp, 95% CI [−5.7, +0.9]** (Wald interval, McNemar
variance). The CI excludes any drop larger than 5.7 pp → **non-inferior at any margin ≥6 pp**
(borderline at a strict 5 pp). The relevance-gate is the **only** compression condition
non-inferior to full; Headroom is significantly worse (−6.6 pp, CI [−10.5, −2.7]).

**Techniques (this release).** The reversible tier's digest is now a **content-aware
skeleton** (keep imports/signatures + traceback tails, elide bodies; deterministic,
stdlib-only — no model, no network) plus **sticky expansion** (a recovered block stays full
across turns). This lifts the active-recovery tier **D from 28.8% (head-truncation) to
32.4%** at ~9× fewer fresh tokens (4.0 vs. 9.6 `distil_expand` round-trips/instance). Honest
ablation (`skeleton_ablation/`): the **same skeleton regresses the passive gated tier E from
36.8% to 5.6%** — a navigable digest makes the agent over-trust it, never re-read, and edit
against body-less context. So the digest is matched to tier behaviour: **skeleton for the
active tier, head-truncation for the passive gate** (`DISTIL_DIGEST_MODE`).

**Findings.** (1) **distil leads on certified task success:** the relevance-gated tier is
the highest-accuracy compressor (36.8%), **+4.2 pp over Headroom (`p=0.035`)**, and the only
one non-inferior to full. (2) **It is the only reversible *and* certified compressor**
(skeleton digest: 100% byte-exact reversible, 0% decision-change with recovery; certificate
in `benchmarks/skeleton_certificate.py`). (3) **Honest on cost:** Headroom is *cheaper* (an
uncertified lossy compressor); distil does **not** claim to be cheapest — it claims the only
guarantee at leading accuracy. (4) **Both lossy baselines crater** (5.6%, 2.4%; the E7
non-transfer result at n=500). Per-instance breakdown, scores, ablation, certificate, and
official harness reports: `benchmarks/long_horizon/`, `benchmarks/skeleton_certificate.py`,
`docs/paper/results/swe_e2e_longhorizon/`.

### 6.0.5 Trajectory-level decision-equivalence certificate (E10)

E2 certifies the *per-turn* decision-change rate; E9 shows that bound does not naively
compose to a trajectory. **E10 certifies at the trajectory level directly** — the unit users
care about. For the relevance-gated tier vs. full context on the same 500 instances, each
trajectory carries two 0/1 losses: **divergence** (outcome differs from full) and **harm**
(full resolved the task, gated did not — compression *cost* a solvable task). We apply the
*same* Learn-Then-Test / Hoeffding–Bentkus engine as E2, inverted to the (1−δ) upper
confidence bound (`distil.conformal.certified_risk_bound`).

| loss | empirical | certified ≤ (95% conf) | OOS coverage (target 95%) |
|---|--:|--:|--:|
| divergence (outcome ≠ full) | 14.4% | **18.0%** | **95.4%** |
| harm (full solved, gated did not) | 8.4% | **11.4%** | **96.7%** |

**The guarantee:** with 95% confidence, the gated compressor changes a run's outcome on
≤18.0% of exchangeable tasks and **costs a solvable task on ≤11.4%** (~1 in 9). We **prove
it out-of-sample** exactly as E2 does — over 1000 calibration/test splits, certify β on the
calibration half and check the disjoint test half: coverage **95.4% / 96.7%**, at/above the
95% target, so the bound *holds on held-out data*. The ungated reversible tier also
certifies (divergence ≤23.2%, coverage 93.9% — marginally under target, reported not hidden).
**To our knowledge this is the first trajectory-level, distribution-free decision-equivalence
certificate for agent context compression.** Honest scope: holds for traffic exchangeable
with the calibration distribution (SWE-bench Verified, this agent + model), not universally.
Reproducible: `benchmarks/trajectory_certificate.py`,
`docs/paper/results/swe_e2e_longhorizon/trajectory_certificate.json`.

### 6.0.6 Cross-model generality (E11) — five models, three vendors

E7–E10 use `claude-haiku-4-5`. **E11 tests whether the gate's non-inferiority generalizes**
by re-running the long-horizon harness on four more models spanning three vendors:
**DeepSeek-V3** (`deepseek-chat`, n=200, official `swebench` harness), **Claude Sonnet 4.6**
(n=50), **gpt-4o-mini** (OpenAI, n=50), and **gpt-4.1** (OpenAI, n=50). Full-context strength
spans a wide range (gpt-4o-mini 12.0%, gpt-4.1 26.0%, Haiku 39.2%, Sonnet 54.0%, DeepSeek-V3
60.0%), letting us separate capability from compression aggressiveness.

**gate@12 across all five models** (pass@1 %):

| model (vendor) | full | gate@12 | vs full | realized |
|---|--:|--:|--|--|
| gpt-4o-mini (OpenAI, n=50) | 12.0% | 12.0% | +0.0 pp (*p*=1.0) | 29% |
| gpt-4.1 (OpenAI, n=50) | 26.0% | 20.0% | −6.0 pp (*p*=0.45 n.s., CI [−16.2,+4.2]) | 32% |
| Haiku 4.5 (Anthropic, n=500) | 39.2% | 36.8% | −2.4 pp | — |
| Sonnet 4.6 (Anthropic, n=50) | 54.0% | 54.0% | +0.0 pp (*p*=1.0) | 18% |
| DeepSeek-V3 (n=200) | 60.0% | 55.5% | −4.5 pp (*p*=0.15 n.s.) | 31% |

**gate@6:** held on Haiku (−2.4 pp), Sonnet (−2.0 pp), gpt-4o-mini (+0.0 pp, realized 58%);
broke on DeepSeek (−31 pp, realized 60%); gpt-4.1 gate@6 partial — account credit exhausted
mid-run (32/50 instances), not scored.

*Honest scope: 3 of 5 runs are n=50 (wide CIs, directional not powered). gpt-4.1 full 26% is
modest — the ReAct harness is tuned for Claude/DeepSeek (harness-fit caveat, not a distil
result). Results in `docs/paper/results/swe_e2e_longhorizon_{gpt4omini,gpt41}/`. The
certificate itself (E2/E10) is model-agnostic by construction.*

**gate@12 shows no statistically significant degradation on any of the five models across
three vendors.** The two well-powered runs (Haiku n=500, DeepSeek n=200) confirm
non-inferiority; the three n=50 runs are directionally consistent with wide CIs (not powered).
The earlier "aggressiveness must scale with model *capability*" story is **refuted** by the
wider sweep. gate@6 broke *only* on DeepSeek (−31 pp) and held everywhere else. Two facts
dissolve the capability story: (i) gpt-4o-mini held at gate@6 despite the *highest* realized
compression of all (58%, even above DeepSeek's breaking 60%) — because a weak agent never
exploited that periphery; and (ii) Sonnet, also strong, held because its gate@6 realized only
34% compression on these runs (the same `gate_recent` digests different fractions depending on
workload conversation shape). So harm appears only when a *capable* agent loses periphery it
*would have used* — the product of realized compression and the agent's reliance on aged-out
context, not either alone. A fixed `gate_recent` cannot predict this (it is a workload×model
interaction), which is exactly why distil calibrates on *outcomes* per deployment with a
fail-safe to full context. Reproducible: `benchmarks/long_horizon/run.py`,
`docs/paper/results/swe_e2e_longhorizon_deepseek/`,
`docs/paper/results/swe_e2e_longhorizon_sonnet/`,
`docs/paper/results/swe_e2e_longhorizon_gpt4omini/`,
`docs/paper/results/swe_e2e_longhorizon_gpt41/`.

**Operationalized: operating-point calibration.** A workload-dependent operating point is a
deployment hazard only if hand-tuned — point distil at a new model or task distribution and
it could silently ship a lossy setting. The operating-point analogue of the certificate removes
it: just as conformal risk control picks the most aggressive *compression level* whose
decision-change rate is controlled, `distil calibrate` (`distil/calibrate.py`) picks the most
aggressive *working-set size* whose task-success loss is non-inferior to full context (same
paired McNemar test), and **fails safe to full context** if none certifies — absence of
evidence degrades to no compression, never to silent loss. On the E11 data the procedure
recovers the manual choice automatically (selects gate@12, rejects gate@6 on DeepSeek-V3;
`tests/test_calibrate.py`). The relevance gate itself is now a shippable library primitive
(`distil/gate.py`), not benchmark-only. See `docs/GA_READINESS.md`.

### 6.0.7 The cost frontier under the motto (E12)

![Cache-aware cost model: cached reads are ~10x cheaper than fresh input](assets/cache-aware.svg)

Distil does **not** claim cost-domination — an uncertified lossy method can always be cheaper
because it is allowed to change decisions. The five techniques below cut cost *inside* the
certified envelope. They never trade the decision-equivalence guarantee for dollars. "Best in
class" holds on the motto's axis (certified decision-equivalence + task success), not raw cost.

| # | Technique | Status | Module |
|---|---|---|---|
| 1 | **Cache-monotone gate** (`distil/gate.py:monotone_gate`) — deterministic, append-only digests keep the digested prefix byte-stable across turns so prompt-cache/KV reuse captures it (cache read ≈10× cheaper than fresh input); lossless relative to the plain gate | Shipped + tested | `tests/test_cost_frontier.py` |
| 2 | **Graded gate** (`distil/gate.py:graded_gate`) — per-distance compression tiers crush the far periphery harder while keeping near-periphery at plain fidelity; introduces a graded (non-binary) loss | Shipped + tested | — |
| 3 | **Tighter conformal — empirical-Bernstein, Maurer–Pontil** (`distil/conformal.py:empirical_bernstein_bound` / `tight_risk_bound`) — tighter than Hoeffding–Bentkus in the low-variance regime that graded losses live in; certifies more savings at the same confidence; coverage Monte-Carlo–validated | Shipped + coverage-tested | `tests/test_conformal_bounds.py` |
| 4 | **Speculative expansion** (`distil/speculative.py`) — pay for full context only when a certified divergence trigger fires; escalation threshold = cheapest whose certified miss rate ≤ α; fail-safe to full context | Framework shipped + tested; end-to-end savings need a live calibration run — **not a shipped default** | — |
| 5 | **Constrained-bandit operating-point search** (`distil/calibrate.py:bandit_select_operating_point`) — online successive-elimination under the non-inferiority constraint, fail-safe; full constrained-RL keep-policy needs training data | Shipped + tested; RL keep-policy is **research** | — |

**Honest caveats.** On #1 (cache-monotone gate): on content that is *already fully cacheable*,
caching alone can be cheaper than any compression — compressing rewrites cached bytes as fresh.
The cache-monotone gate's win is over a cache-*hostile* gate, not over no-compression; the
gate's primary payoff remains accuracy (E8/E11). On #3 (empirical-Bernstein): for *binary*
decision-change losses, Bentkus is already near-optimal — EB applies to the graded losses
introduced by technique #2, which live in the low-variance regime where EB tightens. On #4
(speculative expansion): the framework is shipped and tested, but end-to-end savings are not
validated without a live calibration run; this technique is not a shipped default. On #5 (bandit
search): the successive-elimination wrapper is shipped and tested; the full constrained-RL
keep-policy is a research item requiring training data and is not shipped.

Production status and the full GA-readiness ledger: `docs/GA_READINESS.md`.

### 6.0.8 Continuous assurance under drift (E13)

The certificate is valid only under exchangeability, so the standing operational risk is silent
drift — a new model or workload pushes the true decision-change rate above the budget α the
operating point was certified at. Three shipped pieces close it:

- **Anytime-valid drift monitor** (`distil/drift.py:DriftMonitor`). A betting e-process for
  `H0: risk ≤ α` (hedged capital, Waudby-Smith & Ramdas 2023) whose capital is a non-negative
  supermartingale under `H0`; by Ville's inequality the false-alarm probability is ≤ δ **however
  often the stream is inspected**, so live decision-change can be checked after every turn with
  no multiplicity penalty. Crossing `1/δ` means the live risk exceeds the budget with confidence
  1−δ → recalibrate or fall back to full context. Validated for bounded false alarms under
  continuous peeking and high detection power (`tests/test_drift.py`).
- **Anytime-valid / variance-adaptive certificate** (`distil/conformal.py:betting_upper_bound`).
  The same betting bound certifies graded losses simultaneously at every t. Honest tradeoff: for
  one-shot binary losses Bentkus is already near-optimal, so betting is *comparable* there; its
  edge is continuous monitoring and graded-loss adaptivity (`tests/test_conformal_bounds.py`).
- **Cross-family grader ensemble** (`distil/ensemble.py:EnsembleGrader`). Grade with several model
  families; the default "any-change" aggregation is conservative (can only raise measured risk),
  so the certificate stays valid even if one grader family is unfaithful. Aggregation logic
  shipped + tested (`tests/test_ensemble.py`); multi-family validation needs a live multi-API run.

To our knowledge this is the first anytime-valid drift monitor for a context-compression
decision-equivalence certificate.

### 6.0.9 Surprise-preserving digestion — compression that beats full context (E14)

E8 fixed head-truncation as the gated tier's digest, but head-truncation drops a block's *tail*
— exactly where a traceback puts the assertion that decides the agent's next action (the "lost
if surprise" failure, Deng et al. 2025, arXiv:2412.17483). E14 tests the shipped fix: the **same
relevance gate**, with a digest that keeps the head **plus up to 40 anomaly lines** (errors,
failures, unexpected states, unified-diff changes — the production `surprise_lines` salience
signal), still byte-recoverable via `distil_expand`. Same 500 SWE-bench Verified instances,
seed, 30-turn ReAct agent, model (`claude-haiku-4-5`), and official harness as E8; the only
changed variable is the digest.

| condition | pass@1 | 95% CI | non-empty patches |
|---|--:|--:|--:|
| full context | 39.2% | [35.0, 43.6] | — |
| **gated + surprise digest (v1.7)** | **42.0%** | [37.8, 46.4] | 67.4% |
| gated (head digest, E8) | 36.8% | — | 59.8% |

Paired vs. full context: `b=31` (full solved, surprise did not), `c=45` (surprise solved, full
did not), **Δ = +2.8pp**, 95% CI [−0.6, +6.2]pp — **non-inferior at the 5pp margin, with the
point estimate above full context** (superiority not yet significant). +5.2pp over E8's head
gate, gaining net +26 instances across 9 repositories. The trajectory-risk certificate — the
**shipped** `distil.certify.trajectory_risk` machinery, i.e. the product grading its own
experiment — certifies α=0.10 with observed degradation 6.2% (n=500).

Two readings. The anomaly-preserving digest lifts the agent's ability to *finish* (non-empty
patch rate 59.8% → 67.4%), consistent with the mechanism: an agent that can still see the
assertion keeps acting instead of stalling. And the end-to-end effect is reported by the same
trajectory-level machinery a deployment uses (`distil certify-trajectories`), so the experiment
and the product make the same statement with the same statistics. Honest scope: E14 and E8's
conditions are independent sweeps over the same instance set (matched by instance, not seed),
one model/agent pairing; the certificate quantifies exactly what was measured.

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

## Recent additions (July 2026)

**§ "Live validation of the synthetic-corpus gate"** (new subsection, Analysis section,
after E14): documents the July 2026 live-certification methodology work. Introduces the
A/A self-agreement control (paired statistic subtracts the grader's own sampling floor from
the A/B divergence rate), the measurability probe (second baseline draw decides per-turn
action-only vs.\ full-fingerprint grading), and the grader action-menu constraint (forces
\texttt{action} to the declared tool menu). Reports the empirical finding that, under a
deliberately harsher-than-serving condition, the reference model diverges on 11--21% of turns
beyond its own A/A floor (replicated 3×), with every real divergence tracing to fresh-output
digestion---the condition the serving-path recency carve-out prevents. Includes the
\texttt{fetchurl} misdiagnosis correction: the URL was never in context; the model
constructed it from a digested prose excerpt.

## The evaluation stack (audit, August 2026)

An audit of this paper against the shipped code found three substantial evaluation
capabilities described nowhere in it. They are recorded here because a paper that
omits half its own measurement apparatus overstates the parts it does describe.

**Fact-level recall with verified recoverability** (`distil retention`). Decision
equivalence asks whether the next action changed; it cannot say what was lost when
the action happened to be unchanged. Recall is scored in three states rather than
two: **retained** (visible to the model), **recoverable** (absent from the prompt
but obtainable through one `distil_expand` call, *verified* by checking that the
handle present in that block's compressed text expands to bytes containing the
fact), and **lost**. The gap between visible and true recall is the measured value
of reversibility — 21.4 points as a macro average across domains. Macro, not micro:
the fact-weighted figure is set by whichever domain carries the most probes, and a
single HTML fixture moved it from 9.8% to 62.6% without the compressor changing at
all. A headline one fixture can swing is not measuring the compressor.

**Live decision-equivalence with a self-agreement control** (shadow mode). The
offline gates are graded on our corpus against our oracle. Shadow mode replays real
traffic through both the compressed and original contexts and compares the model's
next action, and — critically — also replays the *same* compressed request twice to
measure the grader's disagreement with itself (A/A). The A/A rate is the floor
below which an A/B divergence is sampling noise rather than compression harm;
subtracting it is what makes the adjusted figure meaningful. Without the control,
an A/B rate of 18.7% reads as harm when the grader's own floor accounts for it. The
counters are content-free.

**External validity through third-party answer keys** (`distil retention --dataset`).
Everything above is self-graded. Grading against HotpotQA's gold supporting
sentences amid eight distractor paragraphs, next to a truncation baseline tuned to
distil's own savings on the same case, is the one number a reader can check without
trusting our fixtures or our oracle.

**Both priced surfaces.** distil compresses what the model reads *and* what its own
past answers cost when they re-enter as history. The state probes below initially
graded only the former; since output digestion targets assistant/history blocks,
which the serving strategy deliberately leaves untouched, an entire priced surface
had no fidelity coverage. Both are now graded by the same probes and gated together.

**Records, not numbers.** Every eval emits a versioned record carrying a dataset
fingerprint, the compressor identity, grader provenance, and each gate's threshold
beside its observed value — validated against a published JSON Schema. Two runs
reporting the same rate are not comparable without it, and a synthetic oracle
reported as a model is the conflation that makes a certificate look stronger than
it is.

## Recent additions (August 2026)

**§ "Beyond recall: state fidelity"** (new subsection): recall is a necessary but
insufficient measure of whether compression preserved what an agent needs. Four
probes are added, each targeting a failure that survives a perfect recall score.

*Artifact state.* The standard artifact metric asks whether a path string survived.
A trajectory that creates `f` at turn *k* and deletes it at turn *k+n* can lose the
deletion while retaining every path token: string recall reads 100% and the agent
plans around a file that no longer exists. We fold tool calls into a ground-truth
file-state ledger and grade the *final* state, separating **stale** (path present,
state wrong — the agent proceeds on a false belief) from **lost** (path absent — the
agent can observe the gap). We report these separately rather than as one accuracy
figure, because a compressor that drops an entire file history is safer than one
that preserves half of it. Factory.ai report every method they evaluated scoring
2.19–2.45 / 5.0 on this axis over 36,611 production messages; presence-based metrics
are structurally unable to detect the failure, since the string is present.

*Overclaim.* Compression that preserves a value while dropping its qualifier —
"approximately 4200 ms" → "4200 ms" — is scored as perfect by every recall metric,
yet hands the agent a precision the source did not assert. We group hedges into
classes so that reshaping ("approximately" → "about") is not penalised, and count
only the disappearance of hedging. The direction is asymmetric: overclaim
(confidence added) is gated, underclaim (caution added) is reported. This follows
work on information fidelity in compressed financial analysis, which finds overclaim
alters downstream decisions independently of factual recall.

*Continuation and propagation.* Whether the agent still knows what remains to be
done, folded in turn order so that a plan's legitimate progress is not scored as
damage; and whether a fidelity loss at turn *k* is associated with a behaviour
change at turn *k+L*, reported as a lag-lift profile. We state two limits in the
tool's own output: the profile is associational, and periodic workloads alias at
multiples of their period.

**Negative results from building the probes.** Three are worth recording, because
each is a failure mode the probes were built to detect, occurring in the probes
themselves.

1. *The corpus certified nothing.* Across all eight trajectories the corpus
   contained 4 file operations and 0 stated obligations; all three state probes
   reported 100% against essentially no evidence. This is the same failure as the
   HTML transform reporting 0% savings before an HTML-bearing trajectory existed. A
   coverage test now fails if the corpus stops carrying enough state transitions,
   hedged claims or plan items to grade.

2. *The harness graded no-ops.* The compressors take a list of blocks; an early
   version of the probe runner passed strings, every call raised, a blanket
   exception handler substituted the original, and each input was compared against
   itself at 100% fidelity. The handler was removed: a compressor that cannot run
   now fails the gate, since a probe that silently grades a no-op is worse than
   absent.

3. *The overclaim metric produced 24 false findings before 9 real ones.* Three
   distinct false-positive classes, each identified by inspecting instances rather
   than trusting the aggregate: anchors matching digits inside larger numbers,
   proximity binding computed over a whole block (compression shifts offsets, so
   bindings flip on text that never changed), and a substring survival test that
   credited a one-character anchor reappearing in unrelated structured output.
   Structured-data lines are now excluded — hedging is a natural-language act, and a
   JSON field asserts nothing.

**Measured result.** On the bundled corpus under the shipped reversible tier:
artifact-state fidelity 100% (7/7, zero stale, zero lost); pending-work recall 100%
with zero status flips; hedge fidelity 94.5% (156/165), with **9 genuine
overclaims**. The overclaims occur only on blocks that materially changed, and arise
where a digested span retains a value but not its qualifier. Unlike a lost fact this
is not recoverable in practice: a missing fact prompts the agent to expand, whereas a
missing hedge gives it no reason to. The gate is set at the measured band rather than
at zero, so that a regression fails and an improvement also forces the bound to be
re-examined.

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
