# `prove.py` — decision-equivalence, measured (not asserted)

This harness turns the central claim into a result by running it on **real agent
traces** graded by a **real model** — i.e. with the circular `DECISION:`-marker
oracle removed (see `docs/PAPER_PLAN.md` for why that matters).

## What it runs

| | | |
|---|---|---|
| **E1** | Frontier | token savings vs. decision-change rate, per ladder level |
| **E2** | Certification coverage — *the proof* | certify at α on a calibration split, then measure the **realized** decision-change rate on a **disjoint held-out** split, over many random trajectory-level splits. The certificate is sound iff empirical `P(realized ≤ α) ≥ 1−δ`. |
| **E3** | Distribution shift | leave-one-domain-out: calibrate on all domains but one, test on the held-out one (the exchangeability stress test) |
| **E4** | Downstream task success | converts per-turn equivalence into the **outcome**: a trajectory keeps its result iff *every* decision is unchanged, so `retained-success(level)` = originally-successful ∧ fully-equivalent, vs. the uncompressed baseline, with a bootstrap CI. Needs outcome labels (τ-bench reward / SWE-bench resolved). |

Decisions are cached on disk per rendered-context hash, so the live-model pass is
paid **once** and the statistics are reproducible.

## Grading backends (`--runner`)

| `--runner` | What it uses | When |
|---|---|---|
| `smoke` | offline heuristic, **non-evidential** | plumbing / CI, no key |
| `claude-cli` | the **`claude -p` CLI** — your Claude Code subscription, **no API key** | easiest real-model run if you already use Claude Code |
| `openai` | any **OpenAI-compatible endpoint** (vLLM/Ollama/LM Studio/OpenAI) via `--base-url` | **free at scale** with a local open model |
| `anthropic` | the Anthropic API (needs `ANTHROPIC_API_KEY`) | billing-grade reference |

```bash
# your Claude subscription, no API key (Haiku = cheap large sweeps; Opus = headline):
python benchmarks/prove.py --dataset tau --path runs.json \
    --runner claude-cli --model claude-haiku-4-5-20251001 --samples 3

# a local open model via vLLM (zero per-call cost):
#   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
python benchmarks/prove.py --dataset swe --path swe_trajs/ \
    --runner openai --base-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct
```

## Offline plumbing check (no key, no download)

```bash
python benchmarks/fixtures/make_fixtures.py        # regenerate the fixtures
python benchmarks/prove.py --dataset fixtures --runner smoke --alpha 0.2
```

The `smoke` runner is a **NON-EVIDENTIAL** stand-in (it models "did the load-bearing
record survive?", treating reversible folds as recoverable). It exists only to verify
the harness mechanics; it is not evidence about real agents. On the bundled fixtures
it reproduces the expected shape:

```
E1: lossless = 16% savings @ 0.0% decision-change ; aggressive truncation flips ~43%
E2 (α=0.2):  certified 100% of splits · empirical coverage 100% (≥95%) · 16% held-out savings
E3:          lossless certificate transfers across tau→swe and swe→tau
```

At a tight α (e.g. 0.05) on this tiny fixture it **correctly refuses to certify** —
too few calibration turns — which is the honest, conservative behavior (more turns
certify tighter α, exactly as the README's 320→α2%, 640→α1% result shows).

## Real τ-bench data — no HuggingFace required

The tau-bench repo ships **200 real trajectories per domain** (gpt-4o & Sonnet-3.5,
airline + retail) under `historical_trajectories/`, reachable from GitHub even where
HuggingFace is blocked. The adapter loads them **directly** (182 trajectories /
1164 real decision points for airline-gpt-4o; tool calls, reward labels, no markers).

```bash
python benchmarks/fetch_real.py tau --src tau:gpt-4o-airline --out /data/tau.json
#   choices: tau:gpt-4o-airline | tau:gpt-4o-retail | tau:sonnet-35-airline | tau:sonnet-35-retail
python benchmarks/prove.py --dataset tau --path /data/tau.json --runner claude-cli ...
```

SWE-bench: point `swe-traj` at a SWE-agent `.traj` dir, or build edit-localization
trajectories from instances with `swe-hf` (needs `datasets` + HF reachable).

### Shuffled-position E5 (removing the recency/tail confound)

The default edit-localization build appends the gold hunk **last** in the code-search
observation, so tail-truncation / recency baselines keep the needle for free. Pass
`--shuffle-gold-seed` to place the gold hunk at a deterministic random position instead
(seeded per-instance, so the corpus is reproducible and the placement is independent of
which subsample `prove.py --limit` later draws). Everything else — distractors, gold
hunk text — is byte-identical to the gold-last build; only the index changes.

```bash
# gold-last (original) and shuffled-position corpora, same instances:
python benchmarks/fetch_real.py swe-hf --split test --out /data/swe_loc_full.json
python benchmarks/fetch_real.py swe-hf --split test --shuffle-gold-seed 1729 \
    --out /data/swe_loc_full_shuffled.json
# grade the shuffled variant exactly like E5, then emit its additive table/macros:
python benchmarks/prove.py --dataset swe --path /data/swe_loc_full_shuffled.json \
    --runner anthropic --model claude-sonnet-4-6 --samples 3 --baselines --ladder quick \
    --alpha 0.15 --delta 0.05 --reps 300 --workers 6 --limit 100 --expand \
    --report docs/paper/results/swe_localization_e5_shuffled_headtohead.json
python benchmarks/report_to_latex.py docs/paper/results/swe_localization_e5_shuffled_headtohead.json \
    --only headtohead e5macros --suffix _shuffled --macro-prefix Shuf
```

The result (seed 1729, $n{=}100$): the confound is **real but not load-bearing** — the
recency baseline's decision-change rises 5.5%→8.5% once the needle is no longer pinned
to the tail, yet it still certifies, and distil's aggressive levels still do not (only
byte-exact certifies). The E2 certificate holds out-of-sample on the shuffled corpus too.

## The real run (the publishable result)

Use `--runner claude-cli` (subscription, no key) or `--runner anthropic` (key).
**Cost/latency note:** real τ-bench contexts are 3–11 KB, so each graded decision is
~5–20 s of model time; majority-of-3 triples it. **Use `--workers 8` (or 16) with the
API runner** — decisions are graded concurrently, turning a full-corpus run from hours
into minutes. (`claude -p` spawns a process per call; the API runner is the right tool
at scale.) Start with a few trajectories and `--ladder quick` to smoke-test, then scale.

**From results to the paper (no hand-copying).** After a run with `--report`:
```bash
python benchmarks/report_to_latex.py results.json   # writes docs/paper/generated/*.tex
```
The paper (`docs/paper/main.tex`) `\IfFileExists`-includes those fragments, so the
headline macros, the E1 frontier figure, and the E2/E3/E4/E5 tables fill in with your
real numbers on the next compile. Commit them with `git add -f docs/paper/generated/`.

**`--baselines` (head-to-head).** Add `--baselines` to grade competitor/structural
baselines under the **same grader** and print a comparison table (token savings,
decision-change rate, and whether each *certifies* ≤ α at 1−δ): LLMLingua-2 and
LongLLMLingua (if `pip install llmlingua`), RECOMP-style extractive,
selective-context, truncation, recency-window, and **keep-last-k-turns** (the classic
agent-memory sliding window). The honest contrast: aggressive baselines save more but flip decisions
(don't certify); distil's certified level is the most aggressive one that does.

**`--expand` (with-expand frontier).** The reversible digest is only decision-equivalent
*with* its recovery loop. By default the harness grades the conservative *no-expand*
lower bound; add `--expand` to let the grader recover digested content (`distil_expand`)
before deciding — `distil.replay.expand_runner` drives any runner through a text
protocol and splices byte-exact originals from a content-addressed restore map. Report
both: no-expand (floor) and with-expand (the deployed behavior).

```bash
# τ-bench (decisions = real tool calls; nothing tells the model what to pick)
python benchmarks/prove.py --dataset tau --path /data/taubench_runs.json \
    --runner anthropic --samples 3 --alpha 0.05 --report tau_proof.json

# SWE-bench (SWE-agent .traj dir or a single json list of trajectories)
python benchmarks/prove.py --dataset swe --path /data/swe_trajs/ \
    --runner anthropic --samples 3 --alpha 0.05 --report swe_proof.json
```

With `--runner anthropic` the harness also prints **model↔gold next-action
agreement** on the uncompressed context — a sanity gate: if the grader doesn't
reproduce the agents' real actions, fix that before trusting E1/E2.

## Expected trace formats

- **τ-bench**: JSON list of episodes, each `{"messages":[{role, content, tool_calls}], "tools":[...]}`.
  Assistant tool calls are the decision points.
- **SWE-bench**: SWE-agent `.traj` `{"problem_statement", "trajectory":[{action, observation}], "info":{resolved}}`
  — a directory of them, or one JSON file holding a list.

Both adapters live in `distil/replay/realtrace.py`; the parsers are defensive about
the common public shapes. The gold action recorded in each trace is kept for
downstream metrics (agreement, task success) but **never injected into context**.
