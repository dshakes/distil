# Running distil's evals

Every gate here is **offline and free** unless the section says otherwise. No API
key, no network, seconds to run. That is deliberate: a gate you skip because it
costs money is not a gate.

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

## The five gates

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
