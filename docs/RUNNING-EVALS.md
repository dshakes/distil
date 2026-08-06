# Running distil's evals

Every gate here is **free** unless the section says otherwise: no API key, no model
in the loop, seconds to run. That is deliberate — a gate you skip because it costs
money is not a gate.

Most are fully offline. The exception is `distil suite`, which fetches public
benchmark rows over HTTP the first time and caches them; after that it runs offline
too, and `--offline` forces cached-only. It still spends nothing, because grading is
deterministic rather than judged by a model.

**This file is HOW to run the evals.** For **WHY** these are the metrics — why a
compression ratio without a task-success delta is meaningless, why `stale` is
reported apart from `lost`, what our own negative results showed — see
[EVALUATION.md](EVALUATION.md).

(The two filenames were close enough to confuse; this one was `EVALS.md`.)

---

## Setup

```bash
git clone https://github.com/dshakes/distil && cd distil
uv sync                      # or: pip install -e ".[dev]"
uv run distil --help
```

`uv` is the supported path; `make setup` wraps it. Everything below assumes
`uv run` — drop it if you have the venv active.

---

## The six gates

Run all of them the way CI does:

```bash
make gate
```

Or individually:

| Command | Asks | Cost |
|---|---|---|
| `distil bench` | Does compression change the agent's decision? | free, ~10s |
| `distil verify` | Is every compression exactly reversible? | free, ~5s |
| `distil retention` | Which facts stay visible, recoverable, or lost? | free, ~2s |
| `distil fidelity` | Do state, hedging and plan survive? | free, ~3s |
| `distil validate` | Do the invariants hold on hostile input? | free, ~15s |
| `distil suite` | Does it hold on PUBLIC benchmarks? | free, first run fetches |

### 1. `distil bench` — decision equivalence

```bash
uv run distil bench                        # 9-domain corpus, deterministic oracle
uv run distil conformal --runner anthropic # graded by a real model (needs a key, costs money)
```

Fails if compression changes a decision the oracle can detect. This is the
non-inferiority gate.

### 2. `distil verify` — byte fidelity

```bash
uv run distil verify
```

Every Tier-0/Tier-1 compression must reverse to the original bytes. No flags; it
either holds or it does not.

### 3. `distil retention` — fact recall

```bash
uv run distil retention                      # corpus, tri-state recall
uv run distil retention --json               # machine-readable
uv run distil retention --max-lost 0         # gate: fail on any unrecoverable fact
uv run distil retention --live               # YOUR traffic, content-free counts only
uv run distil retention --dataset hotpotqa   # graded against a PUBLIC answer key
uv run distil retention --dataset list       # what public datasets are wired up
```

Reports **visible** recall (in front of the model) separately from **recoverable**
(one `distil_expand` away, verified against the handle's restore bytes). The
headline is the macro average — per-domain mean, each domain counted once —
because the fact-weighted number is set by whichever domain happens to carry the
most probes.

### 4. `distil fidelity` — the state probes

```bash
uv run distil fidelity                  # all four probes over the corpus
uv run distil fidelity --json
uv run distil fidelity --max-silent 15  # gate on SILENT failures
uv run distil fidelity --no-propagation # gate on delayed damage
```

Four probes, described in [EVALUATION.md §6](EVALUATION.md):

- **artifact state** — is the file's final state right, or just its name present?
  Splits `stale` (present, wrong — silent) from `lost` (absent — loud).
- **overclaim** — did the value keep its uncertainty? `"approximately 4200 ms"`
  → `"4200 ms"` is a distortion every recall metric scores as perfect. Reversing a
  bound (`"at least 3"` → `"at most 3"`) is reported separately as `inverted`,
  because a dropped hedge leaves a visible gap and a reversed one leaves a
  confident wrong number.
- **continuation** — does the agent still know what is left to do?
- **propagation** — does a loss at turn *k* show up as a change at turn *k+n*?

**On `--max-silent`.** It counts `stale + overclaimed + inverted + dropped_work` —
the failures an agent cannot see. Loud loss is deliberately *not* gated here; `retention
--max-lost` owns it, and gating one regression in two places obscures which
property actually broke.

The shipped reversible tier currently reports **9 overclaims out of 171 hedged
claims** on the corpus, so `--max-silent 0` fails today. That is a real measured
property, not a bug in the gate — see EVALUATION.md §6.2. CI gates at the measured
band.

### The eval record — what `--json` emits

`--json` does not emit bare metrics. It emits a **record**: the metrics plus the
five things you need to reproduce or compare them.

```jsonc
{
  "schema": "distil.eval/1",          // so a consumer knows when the shape changed
  "run":     { "at": "...", "duration_ms": 218,
               "env": { "distil": "1.40.1", "python": "3.9.25", "platform": "darwin-arm64" } },
  "subject": { "compressor": "_ServingSurface", "module": "distil.compress.strategies" },
  "dataset": { "name": "corpus", "trajectories": 9,
               "domains": [...],
               "fingerprint": "sha256:051b836358932883" },
  "grader":  { "kind": "deterministic",
               "detail": "synthetic DECISION: oracle — NOT a model" },
  "metrics": { "artifact_state": {...}, "overclaim": {...}, ... },
  "gates":   [ { "name": "max_silent", "threshold": 15, "observed": 9,
                 "passed": true, "rationale": "..." } ],
  "passed":  true
}
```

Why each field earns its place:

- **`schema`** — a consumer can detect a shape change instead of silently
  mis-parsing it.
- **`dataset.fingerprint`** — a content hash of exactly what was graded, stable
  across load order. Adding a trajectory changes the numbers; without this that
  looks like a compressor regression. This distinction has already caught a test
  in this repo failing for a reason it was never written to detect.
- **`subject`** — "94.7% hedge fidelity" means nothing without knowing which
  compressor produced it.
- **`grader`** — follows the norm set by `conformal.render_grader`: a synthetic
  oracle is never reported as a model, because that conflation is what makes a
  result look stronger than it is.
- **`gates`** — threshold, observed value and outcome together. A gate whose bound
  is invisible cannot be audited, and `"passed": true` with no threshold beside it
  is not evidence. An **empty** gate list is explicitly *not* a pass: nothing was
  checked.

Under `--json`, failure diagnostics go to **stderr**, so stdout stays parseable at
exactly the moment CI most needs to read it.

```bash
uv run distil fidelity --json --max-silent 15 | jq '.gates'
uv run distil fidelity --json --max-silent 15 | jq -r '.dataset.fingerprint'
```

---

### 6. `distil suite` — public benchmarks, third-party ground truth

```bash
uv run distil suite --tier 1              # tool-calling + retrieval (the evidence)
uv run distil suite --tier 1 --tier 2     # add the harder payloads
uv run distil suite --tier 3              # the thin-payload controls
uv run distil suite --only bfcl -n 50     # one benchmark
uv run distil suite --tier 1 --offline    # cached rows only, no network
uv run distil suite --json --max-lost 0   # gate on unrecoverable golds
uv run distil suite --min-answer-recall 0.95 --min-support-recall 0.95
```

**A collapse always fails, flag or no flag.** A rich benchmark that graded cases and
recalled *nothing* is a broken run, not a low score, so the suite exits 1 on it even
with no threshold passed. `--max-lost` counts unrecoverable golds of **both** kinds —
answers and support facts — because benchmarks like SQuAD carry no support facts at
all, and counting only those made the gate structurally unable to see an answer
regression.

Twelve public benchmarks whose answer keys were written by someone else, so a
stranger can falsify our numbers against data they already trust.

**It costs nothing.** Grading is deterministic recall against the answer key — no
model in the loop, no API key, no spend. Suites that grade with an LLM judge cost
real money per tier, which makes them something you run before a launch rather than
before a merge. This one is wired into `make gate` and the CI gate job (tier 1, n=25).

**Payload class is reported with every row, and it is the most important column.**

| class | meaning | examples |
|---|---|---|
| `rich` | real context to compress — **evidence** | `bfcl`, `hotpotqa`, `msmarco`, `narrativeqa`, `squad`, `codesearchnet`, `humaneval` |
| `thin` | a one-line question with nothing to compress — **control** | `gsm8k`, `mmlu`, `arc`, `truthfulqa`, `triviaqa` |

A GSM8K case is a word problem; there is nothing in it to compress, so an unchanged
score proves the compressor left it alone. That is a control, not a demonstration.
Only `rich` rows can show compression quality, and a run that grades **only**
controls exits 1 rather than reporting a clean sheet.

`bfcl` leads tier 1 deliberately. Berkeley Function Calling compresses the **tool
schema** and checks that every name the gold call is built from — the function and
each argument it passes — survives compression. That is the failure an agent proxy
is most likely to cause and least likely to notice, because no QA benchmark ever
asks the model to *act*: a schema can keep `calculate_triangle_area` and lose the
`base` parameter, and the call cannot be formed at all.

Measured over 100 cases: **91.4% savings, 378 gold names, 4 lost (98.9% retained)**.
Names are de-duplicated — the function name arrives from both the schema and the
gold call, and counting it twice inflated the earlier 478/99.2% figure.
Most survive as *recoverable* rather than visible (2.3% visible) — they sit behind
expand handles, which is what the reversible tier is supposed to do.

The gold call itself is **not** graded as an answer. It never appears in the schema
text, so a text-recall grader cannot see it, and checking whether a model still
emits it would need a model in the loop — which would make this suite cost money
and stop it running in CI. What is measured is stated exactly: the names the call
depends on.

A benchmark that cannot be fetched is reported as a FAILED row and exits 1. It is
never dropped: a suite that silently skips what it could not load reports a clean
sheet for a run that measured less than it claimed.

*Not wired:* LongBench — upstream ships it only as a `data.zip`, with no rows API
and no per-task files, so there is no stdlib-only path to it that keeps distil's
zero-dependency promise.

### 5. `distil validate` — adversarial invariants

```bash
uv run distil validate
```

Drives the compressor against huge, unicode, nested, malformed, marker-injecting
and secret-looking inputs, asserting reversibility, reject-if-bigger,
recency-exactness, fail-open and content-free telemetry on every one. This gate
exists because a green unit suite kept coexisting with real-traffic bugs.

---

## Statistical certification (beyond the gates)

```bash
uv run distil certify           # conformal risk certificate at a chosen alpha
uv run distil shadow-stats      # live decision-equivalence from wrap --shadow
uv run distil certify-trajectories runs.jsonl  # offline trajectory-risk certificate
                                # runs.jsonl: {task_id, full_success, compressed_success} per line
```

`certify` runs Learn-Then-Test and Conformal Risk Control to pick the most
aggressive compression level whose decision-change rate is provably bounded at
α, distribution-free and finite-sample. It assumes **exchangeability**: the
calibration traffic must resemble live traffic. `distil.drift` implements an
anytime-valid sequential test for that assumption breaking — it runs inside the
proxy rather than as a subcommand, so recalibrate on a rolling window when your
workload shifts.

---

## Measuring on your own traffic

Nothing here uploads content. The live meters store counts only.

```bash
distil wrap --shadow 0.1 -- claude  # collect real decision-equivalence data (10% of turns)
distil retention --live        # fact recall on your traffic
distil shadow-stats            # what shadow mode measured
```

---

## Extending the corpus

The probes are only as good as what the corpus makes them look at. Twice now a
probe has reported a perfect score against nothing:

- the HTML transform compressed 0% because no trajectory carried HTML;
- the state probes saw 4 file operations and 0 obligations across all trajectories.

If you add a transform or a probe, add a trajectory that exercises it:

```bash
# generators live in benchmarks/ — edit these, not the JSON
uv run python benchmarks/gen_web_research.py     # HTML-bearing trajectory
uv run python benchmarks/gen_agent_worklog.py    # file ops + plan + hedges

# then register it
$EDITOR corpus/manifest.json
uv run pytest tests/test_fidelityprobes.py -k exercises   # coverage is pinned
```

`test_corpus_actually_exercises_every_probe` fails if the corpus stops carrying
enough state transitions, hedged claims or plan items to grade. That test is the
reason a green result means something.

---

## CI

All five gates run per-commit. The workflow is `.github/workflows/ci.yml`; the
same commands work locally, which is the point — `make gate` before pushing and
CI should tell you nothing new.

**Known limitation of the BFCL number.** Recall is scored by `retention`'s generic
matcher, which anchors on non-word boundaries — right for prose, loose for short
identifiers. A lost argument `id` can still be credited if `"id"` appears as a key
elsewhere in the schema, so the figure is an **upper bound** on name retention for
short identifiers. Tightening it by quoting the identifiers was tried and broke the
recoverable-match path (recall fell to 2.9%, which measures the matcher rather than
the compressor), so identifier-aware matching is deferred rather than shipped
half-working. The CI band is set with this in mind.
