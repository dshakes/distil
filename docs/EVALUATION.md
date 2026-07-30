# Evaluation methodology

How we measure whether distil is safe to ship, and why the obvious metric
(compression ratio) is not the one we report against.

## 1. Compression ratio alone is meaningless

Token reduction and task success are not linearly related, and the relationship
can cliff-edge: a setting that looks fine on average can collapse resolution
rate outright. "On Problems of Implicit Context Compression for SE Agents"
([arxiv.org/pdf/2605.11051](https://arxiv.org/pdf/2605.11051)) found that 4x
compression collapsed SWE-bench resolution from 86% to 7% — not a gradual
decline, a cliff. Factory.ai makes the same point from the practitioner side
([factory.ai/news/evaluating-compression](https://factory.ai/news/evaluating-compression)):
compression has to be graded on whether it changes the agent's *next action*,
not on how small the context got.

Distil's rule follows from this: **every compression claim we publish reports
token savings and task-success delta together, at the setting we actually ship**,
never a peak ratio measured in isolation. A number like "38% smaller" with no
paired success-rate number next to it is not a result, it's an invitation to
find the cliff the hard way, in production.

## 2. distil's own cliff-edge: E7

We hit this failure mode ourselves. E7, an internal SWE-bench Verified
end-to-end experiment (`benchmarks/swe_bench_e2e/`), ran the full agent loop —
not a single-step proxy — under distil's *aggressive* lossy compression tier
and measured actual task resolution: **52% → 16%**. The **reversible** tier,
by contrast, held (56% vs. 52% full-context baseline). Per-step
decision-equivalence had passed; end-to-end task success still cratered under
the aggressive tier. That's the exact failure the literature above describes,
reproduced on our own harness.

We publish E7 as evidence, not confession. It's the reason the trajectory-risk
certificate and the fail-safe gate exist, and why distil's shipped default is
the conservative/reversible tier rather than the tier that scores best on a
compression-ratio leaderboard. If you turn off the safety machinery, E7 is
what you get back.

## 2.1 Live grading of the synthetic-corpus gate (2026-07)

The per-commit CI gate certifies decision-equivalence on a bundled synthetic
corpus with a deterministic oracle. In July 2026 we pointed a **real** grader
(claude-opus-4-8, majority-of-3, with an A/A self-agreement control) at the
same corpus under the same certified strategy — and it failed the live margin,
twice, reproducibly: 5–6 of 28 turns diverged beyond a 93–100% A/A floor.

Characterizing every divergence taught us three things:

1. **Two were metric artifacts** — the grader free-typing paraphrased action
   names, and a one-token target paraphrase. Fixed at the root: the decision
   tool's `action` is now constrained to the tool menu actually declared in
   context, so the grader picks from the same menu the agent would.
2. **Three were real** — and all three were the model losing detail from the
   *freshest tool output being digested*: digesting a page excerpt cost the
   model the prose evidence from which it had constructed a direct `fetchurl`
   (the URL itself was never in context — tracing this precisely took a
   follow-up debug session; the first write-up wrongly called it "a dropped
   URL"), so it fell back to a `websearch` detour; dropped success-evidence
   made the model re-execute a completed runbook; a digested policy check let
   it skip a verification step before issuing a refund. This is precisely the condition
   the **serving path never exhibits** — production keeps the last
   `RECENCY_KEEP_TURNS` tool outputs byte-exact and offers `distil_expand` —
   and the certified strategy deliberately omits both protections to be
   harsher than serving.
3. **The deterministic oracle cannot see any of this.** It passes the same
   inputs at 100%.

So the live run measured, with a real model, exactly why the recency carve-out
is load-bearing: without it, even the reversible digest changes the reference
model's next action on ~1 turn in 9 on this corpus. Like E7, we publish this
as evidence, not confession — it moved the nightly live gate off the
synthetic-corpus/harsher-strategy configuration (which measures a condition
production never runs) and onto `benchmarks/prove.py`'s real-trace,
serving-semantics grading. Two fixes shipped from the characterization and one
was validated live the same day: constraining the grader's action menu removed
both paraphrase artifacts on the next opus run (third replication: mean diff
−0.179 — the divergence *rate* is stable across runs even as individual turns
churn). URLs also joined SHAs and paths as load-bearing artifacts at BOTH layers: the
salience keep-patterns (protecting the conformal ladder's `+salience` rungs)
and `keep_policy`'s generic net, which tier1's digest consults for every
content kind — so an in-context URL now survives digestion on the plain
strategy too (URL-spam logs still fold via the per-shape repeat cap). Note the
honest limit found by live A/B: the `fetchurl` divergence above is NOT fixed
by URL retention, because that URL was never in context — the model built it
from world knowledge off a prose excerpt that the harsher-than-serving
strategy digests and serving keeps verbatim (it is the turn's freshest
output). That residual is a property of the deliberately harsher certification
condition, not of production behavior.

This same control is now available in the offline harness too:
`benchmarks/prove.py --aa-control` adds an independent second draw of each base
context (cache-busted, so it's a genuine fresh grade) and reports each E1
level's decision-change as the excess above the measured A/A floor. It exists
because prove.py's raw per-turn rates carry the grader's full variance term —
and its byte-exact rung reads 0% only because it shares the baseline's decision
cache key (a structural artifact, not a measured floor). The flag is off by
default (it costs one extra draw per turn); the E2 conformal gate the nightly
enforces is unaffected either way.

## 3. Live decision-equivalence: shadow mode

The trajectory certificate (below) is offline, run against a fixed corpus.
Shadow mode (`distil/shadow.py`) is the live check: it runs continuously on
real traffic, replaying **2% of requests** by default (`distil wrap --shadow`,
`0` opts out).

Each shadow sample is one of two kinds, at a 2:1 ratio:

- **A/B (2/3 of samples)** — replay the same turn with the original context
  and with the compressed context, and compare the model's decision signature
  (tool call + args, or response digest) between the two. This is the number
  people want: did compression change what the model decided to do?
- **A/A (1/3 of samples)** — replay the *same already-compressed* request
  twice and compare its decision signature against itself. This is not a
  compression measurement at all — it's the model's own sampling
  nondeterminism, measured on live traffic with a live model.

The reason A/A exists: a raw A/B agreement number is uninterpretable on its
own. If the model only agrees with itself 90% of the time under identical
input (temperature, tool-choice noise, provider-side nondeterminism), an A/B
agreement rate of 88% is not a 12-point compression problem — it's within the
model's own noise floor. `distil shadow-stats` reports the decomposition
explicitly: **raw A/B rate, A/A baseline rate, and the adjusted rate** (A/B
agreement relative to the A/A baseline), so a regression in the adjusted rate
means compression, not the provider having a noisy day.

Every shadow row is content-free by construction: a digest and both decision
signatures, never the underlying prompt or completion.

This A/A self-agreement control is ahead of current published practice.
ACON ([arxiv.org/abs/2510.00615](https://arxiv.org/abs/2510.00615)), Context
Codec ([arxiv.org/abs/2605.17304](https://arxiv.org/abs/2605.17304)), and
Decision-Aware Memory Cards
([arxiv.org/abs/2606.08151](https://arxiv.org/abs/2606.08151)) all evaluate
decision-equivalence, but none of them measure or subtract out the model's own
sampling nondeterminism first — which means a fraction of the harm they
attribute to compression is actually the model disagreeing with itself.

## 4. Trajectory-risk certificates

`distil certify-trajectories` (`distil/certify/trajectory_risk.py`) is the
offline statistical gate: run your eval suite twice, full context and
compressed, on the same tasks; feed in the matched pass/fail pairs; it returns
a distribution-free bound of the form **P(degradation ≤ α) ≥ 1 − δ**, refusing
to certify below a minimum sample size, and states its exchangeability
assumption in the certificate itself.

What it proves: on *this measured workload*, compression is very unlikely to
have cost you more than α percentage points of task success, at confidence
1 − δ. What it does not prove: that the same bound holds on a workload you
haven't measured. Transfer across workloads is not guaranteed — E7 is the
concrete demonstration of why that caveat is load-bearing, not boilerplate:
per-step certification on one workload does not imply end-to-end safety on
another. An anytime-valid drift monitor
(`distil.certify.trajectory_risk.drift_monitor`) flags when live traffic has
moved far enough from the certified distribution that the certificate should
be considered stale and re-run.

## 5. Fact-level recall, and grading against someone else's answer key

Sections 1–4 all measure the same kind of thing: whether the agent's *next action*
survives compression, graded on distil's own corpus against planted `DECISION:`
markers. That is the right invariant to certify, but it is not the question users
ask, and it is not checkable by an outsider. Two additions close both gaps
([ADR 0005](adr/0005-external-validity-and-fact-level-recall.md)).

### 5.1 Three states, and why recall is the headline

A compressor's error is asymmetric. Dropping a fact produces a wrong answer;
keeping content you did not need merely saves fewer tokens. So we report **recall**
and let the savings number carry the other direction, rather than blending both into
an F1 that hides which way the error ran.

Every fact — keyed numerics, artifacts (paths/URLs/hashes/UUIDs), error lines — lands
in one of three buckets:

| Bucket | Meaning |
|---|---|
| `retained` | in the text the model sees, verbatim or after a format conversion |
| `recoverable` | absent, but provably reachable via `distil_expand` |
| `lost` | absent with no recovery path — the only bucket that can cause a wrong answer |

`recoverable` exists only because distil is reversible, and it is **verified, not
assumed**: the handle must appear in that block's own compressed text, *and* the fact
must appear in that handle's restore bytes. Two numbers fall out —

    visible recall = retained / total                    (free, no tool call)
    true recall    = (retained + recoverable) / total     (the information bound)

— and the gap between them is what reversibility buys. On the committed corpus:
**100% true recall, 90.2% visible, 0 lost** — reversibility is worth 9.8% recall.

```
distil retention                 # corpus (zero cost, offline) — a CI gate
distil retention --live          # YOUR traffic, from the sampled meter
```

The live meter scores **in-process** and stores **counts only** — three integers per
dimension, with deliberately no field that could carry content, so no plaintext
session recorder exists to secure. It is sampled (`--retention RATE`, default 0.05
under `wrap`, 0 under a bare `proxy`), bounded (64 KiB per request), and fail-open. It
costs no extra tokens: unlike shadow mode there is no second upstream call.

### 5.2 Public benchmarks — external validity

```
distil retention --dataset list
distil retention --dataset hotpotqa -n 100
```

Graded against ground truth written by someone else: HotpotQA's `supporting_facts`
(the exact gold sentences, amid 8 distractor paragraphs) and SQuAD v2's answer spans.
Rows come from the HuggingFace datasets-server REST API as plain JSON, so the loader
stays stdlib-only and `dependencies = []` holds; they cache under `$DISTIL_HOME/datasets`
so a published number reproduces offline.

A recall number alone proves nothing — an identity transform scores 100%. So each run
also scores a **truncation baseline tuned to reproduce distil's own token savings on
that same case**, and prints both savings figures so the match is auditable:

| HotpotQA, n=100 | savings | answer recall | support recall |
|---|---|---|---|
| distil (reversible) | 14.3% | **100.0%** | **100.0%** |
| truncation @ matched savings | 14.1% | 91.6% | 82.7% |

Three honesty rules the command enforces, each of which caught a real defect while
this was being built:

- **Only probeable answers are graded.** HotpotQA's comparison questions answer
  "yes"/"no", which is never a span in the passage. Grading them reported the
  dataset's answer *format* as 10% compression loss. Answers now count only when
  present in the **uncompressed** context; the rest are excluded and counted.
- **Unanswerable questions are excluded, not passed.** SQuAD v2 ships them
  deliberately; scoring them as retained would be a free win on 55% of the split.
- **A run where compression did not engage fails.** Below 1% savings the recall
  number is arithmetic, not evidence — so `--min-recall` refuses to pass on it. This
  matters because distil's compressors correctly decline to touch short prose:
  `--shape prose` yields 0% savings and a vacuous 100%. The default `--shape json`
  reflects how a retrieval tool actually returns documents.

Note also what SQuAD exposes: at 84.8% savings true recall is 100% but **visible
recall is 0%** — every answer sits behind a handle. Lossless, but one round trip per
answer. The report says so rather than printing only the reassuring number.

### 5.3 What the harness found: HTML was compressing 0%

A recall metric earns its keep by finding things, and the first thing this one found was
a capability gap rather than a regression. distil compressed **0.0%** of an HTML tool
result — structurally, because minified markup is a single enormous line, so Tier-1's
line-folding had nothing to fold and the JSON/record folds do not recognise markup. Any
agent with a fetch or browser tool paid full price for `<script>`, `<style>`, and chrome.

`distil/compress/htmlx.py` now extracts the content (stdlib `html.parser`) and keeps the
exact original behind an expand handle:

| real page | before | after | saved | facts lost |
|---|---|---|---|---|
| Wikipedia article | 281,093 tok | 14,260 tok | **94.9%** | **0** |
| Python docs page | 32,322 tok | 4,229 tok | **86.9%** | **0** |

Recall-biased on purpose: `<header>` is kept (it usually wraps the `<h1>`), `img` alt
text is kept, and `div`/`section` are never dropped. Being reversible is what licenses
the aggressiveness — a lossy extractor's mistake is permanent, this one costs one
`distil_expand`. Skipped under `--verbatim` (nothing to recover with) and on the
recency-exempt latest tool result.

One measured tradeoff, stated rather than buried: on raw HTML most probe-able artifacts
are `href` URLs, which extraction drops as navigation. True recall stays 100% because
they are recoverable, but **visible** recall on that dimension is low (4.8% overall on
the Wikipedia page). That is a tunable with a number attached now, not an unknown.

## 6. How to reproduce

- `distil shadow-stats` — live decision-equivalence, raw/baseline/adjusted
  decomposition, from real traffic collected by `distil wrap --shadow`.
- `distil certify-trajectories` — offline trajectory-risk certificate from a
  matched full-vs-compressed outcome file; see `distil certify-trajectories -h`
  for the input format.
- `benchmarks/swe_bench_e2e/` — the E7 harness (`run_all_agents.sh`,
  `run_all_scores.sh`, `aggregate.py`): runs the full SWE-bench Verified agent
  loop under each compression tier and scores real resolution, not a proxy.
- `benchmarks/PROVE.md` (`prove.py`) — decision-equivalence measured against
  real agent traces graded by a real model, avoiding the circularity of
  self-graded `DECISION:` markers.
- `BENCHMARKS.md` — the live-graded head-to-head numbers and how they're run.
