# Referee runs: what fills the scoreboard, and what it costs

Companion to ADR 0024. `docs/scoreboard.html` shows only committed artifacts. Today
that means `plain` and `distil` on SWE-bench Lite, at two effort settings, plus the Run 1
head-to-head (rtk, selective, provider-cm). Every other row reads *pending run*. **Run 1 has
been run (2026-10-05); Runs 2 onward have not.** Each
paid step is launched by hand after review, with the caps below. The total hard ceiling is
**$200.07**, under the $250 envelope.

## Per-task cost, from the committed artifacts

Each row is the sum of `cost_usd` over `results.jsonl`, divided by the number of tasks.

| run | effort | plain $/task | distil $/task |
|---|---|---|---|
| `swebench-outcome-300-medium` (2026-10-03) | medium, 60 steps | 0.0284 | 0.0281 |
| `swebench-outcome-300-grepfix` (2026-10-02) | low, 30 steps | 0.0239 | 0.0239 |

The "~$0.028/task/arm on Lite medium" figure is confirmed. `plan --calibrate` prices an
arm with no cached rows at plain × 1.5 (`run.UNMEASURED_FACTOR`), which gives $0.0426 a
task. That margin covers an arm that adds steps or busts the cache.

## Run 1: SWE-bench Lite 300 × {rtk, selective, provider-cm, headroom}, reusing plain and distil

The new arms run on the same 300 tasks, at the same model and effort as the medium run.
`plain` and `distil` are copied from it with `--reuse-arm`. The harness checks that model
and effort match, the copies cost nothing, and the report says they were reused.

```bash
OUT=benchmarks/results/swebench-outcome-300-h2h
SRC=benchmarks/results/swebench-outcome-300-medium
ARMS=plain,distil,rtk,selective,provider-cm,headroom
COMMON="--out $OUT --arms $ARMS --effort medium --max-steps 60 --reuse-arm plain=$SRC --reuse-arm distil=$SRC"

# $0: the estimate (verified offline: $51.13 = 4 arms x 300 x $0.0426; the first three
# arms were $38.35 on 2026-10-04, headroom adds $12.78)
python -m benchmarks.swebench_outcome plan $COMMON --calibrate $SRC

# PAID: hard stop at $80 (1.56x the estimate)
python -m benchmarks.swebench_outcome run $COMMON \
    --selective-python /path/to/python-with-selective-context \
    --headroom-python /path/to/python-with-headroom-ai-proxy \
    --budget-usd 80 --i-understand-this-costs-money

# $0 (local Docker grader), then the report
python -m benchmarks.swebench_outcome grade --out $OUT --arms $ARMS
python -m benchmarks.swebench_outcome report --out $OUT --arms $ARMS
```

- **Ran 2026-10-05 without the headroom arm (it landed later): estimated $38.35, capped at $60,
  cost $25.07.** Results: `benchmarks/results/swebench-outcome-300-h2h/` (README.md has the
  caveats, including the cost confound) and the head-to-head section of `docs/scoreboard.html`.
  rtk NON-INFERIOR (+0.3 pts [-2.7, +3.3]); distil, selective and provider-cm INCONCLUSIVE.
- **The headroom arm alone: estimated $12.78 of the $51.13 above, capped at $80 for the full
  command.** Not run yet. Run it with `--arms plain,distil,headroom` and the same `--reuse-arm`s.
- Without the `datasets` package, `plan` and `run` need `--instances FILE`: the 300 ids from
  `$SRC/results.jsonl`, one per line.
- Before any spend, the arms refuse to run if their pinned dependency is missing:
  - `rtk` downloads and verifies RTK v0.51.0. `--rtk-bin` overrides it.
  - `selective` needs `selective-context==0.1.4` in a numpy<2 Python. See #227.
  - `provider-cm` needs nothing installed.
  - `headroom` needs `headroom-ai[proxy]==0.40.0` in a CPython >= 3.10 (3.12 works):
    `uv venv --python 3.12 hr-venv && uv pip install --python hr-venv/bin/python
    'headroom-ai[proxy]==0.40.0'`, then `--headroom-python hr-venv/bin/python`. A missing
    package, a missing `[proxy]` extra, a version other than 0.40.0, or a proxy that dies or
    is not live on `/livez` within 300 s makes `run` print `refusing: ...` and exit 2, before
    any API call.
- `headroom` is the real proxy, not a re-implementation. The harness starts
  `python -m headroom.cli proxy --port N --workers 1` (the command `headroom wrap claude` runs,
  with `HEADROOM_AGENT_TYPE=claude`, `HEADROOM_STACK=wrap_claude` and Headroom's own default
  `cache` mode) and sends only this arm's requests to it, with the run's API key; the proxy
  forwards to api.anthropic.com. It is started per run and stopped at the end. Harness-only
  deviations, recorded in each row's `arm_meta`: `--no-subscription-tracking`, and a throwaway
  HOME/`HEADROOM_WORKSPACE_DIR` so the run never touches `~/.headroom` or `~/.claude`. Headroom's
  CCR retrieval is handled inside the proxy, so the agent loop has no extra tool for it. The proxy
  loads its ML models on first start (HF cache), so the first start can take minutes.
  The arm is priced as unmeasured (plain x1.5) until it has a run of its own.
  Reviewed against Headroom `main` (0.40.0 source in `temp/headroom-main`): the pinned wheel's
  proxy launch path (`cli/wrap.py _start_proxy`) is identical, but `main` has unreleased changes
  to `transforms/content_router.py` (~190 lines), `transforms/kompress_*.py` and
  `cli/wrap.py` (provider key checks). Compression behaviour of `main` can therefore differ from
  the pinned 0.40.0 release; the scoreboard row is labelled with the pinned version.
- `provider-cm` runs at the harness's aggressive setting (`--cm-trigger 3000 --cm-keep 2
  --cm-clear-at-least 1000`), so it fires on short SWE-bench contexts. The live
  `distil audit` uses Anthropic's documented defaults instead. These are two different
  questions, and the scoreboard labels the harness config.
- To publish, add `{"id": "swebench-lite-300-h2h", "kind": "swebench_outcome", "dir": "$OUT",
  "date": ...}` to `benchmarks/results/scoreboard-runs.json`. Then run
  `python3 scripts/build_scoreboard.py` and add the new rates to `docs/claims.json`.
  `tests/test_claims_coverage.py` names any that are missing.
- Caveat already on the page: median tasks are about 4 steps, so this is a short-context test.

## Run 2: Terminal-Bench pilot (cost_truth), Headroom through `wrap claude`

This is the other Headroom measurement (the SWE-bench arm above is Run 1). `benchmarks/cost_truth` already installs `headroom-ai==0.38.0`, RTK and distil
through `harbor`, behind a neutral meter (ADR 0018).

```bash
python -m benchmarks.cost_truth plan          # $0: arms, pins, exact commands
python -m benchmarks.cost_truth live --phase pilot --i-approve-spend 92.07
python -m benchmarks.cost_truth analyze benchmarks/results/cost_truth/<pilot dir>
```

- **Hard cap $92.07** (`benchmarks/results/cost_truth/cost_estimate.json`, pilot phase:
  10 tasks × 2 seeds × 4 arms = 80 attempts, $73.65 at the cached profile × 1.25).
- The committed `pilot-*` directories hold canary preflights only, with every canary
  trial `failed`. Fix that before spending. If a pilot was already run elsewhere, commit
  its runs and `analyze` them instead, for $0.
- At 20 attempts per arm the result is a pilot. The scoreboard prints whatever verdict
  `analysis.json` carries, and it will not be a pass.
- The confirmatory **primary** phase has a $2,048.49 cap and is out of scope for this
  budget.

## Run 3: live dogfood of `distil audit` (maintainer's own traffic)

```bash
distil audit --compressor anthropic-context-editing --cap-usd 1   # 14 days
distil audit --compressor distil --cap-usd 1                       # the next 14 days
distil audit report
```

- **At most $1/day per compressor: $28 for both.** One compressor is active at a time.
- Expect "can't tell yet" for weeks. The floor is 50 paired samples where the compressor
  fired. With the documented 100k-token trigger, context editing fires only on long
  sessions, and at the default cap a ~100k-token request is about one sample a day.
- This is not published on the scoreboard. It is per-machine evidence.

## Budget

| run | estimate | hard cap |
|---|---|---|
| 1. SWE-bench Lite × {rtk, selective, provider-cm, headroom} | $51.13 | $80.00 |
| 2. Terminal-Bench pilot (control, rtk, headroom, distil) | $73.65 | $92.07 |
| 3. `distil audit` dogfood, 2 × 14 days | ≤ $28.00 | $28.00 |
| **total** | **≈ $152.78** | **$200.07** |

## Not planned (and why)

- OpenAI compaction offline: the SWE-bench harness is Anthropic-only. It remains a live
  `distil audit` target, and the provider certificate already covers it
  (`benchmarks/results/provider-compaction/openai*`).
- An RTK live audit: not faithful per request (ADR 0024, decision 4).
