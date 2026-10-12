# Re-fetch verbatim — offline replay (ADR 0025)

`benchmarks/reinflate_replay.py` replays the distil-arm transcripts of two 300-task
SWE-bench Lite outcome runs through `compress_messages`, request by request, exactly as the
harness sent them (cache breakpoint on the newest block, `persist=False`). No API calls. The
agent's actions are fixed, so these numbers say what it would have **seen**, not what it
would have **done**.

- `grepfix-300.json` — transcripts of `../swebench-outcome-300-grepfix/` (the current
  adapter: shell search kept verbatim). The headline.
- `prefix-300.json` — transcripts of `../swebench-outcome-300/` (before that fix), replayed
  with today's adapter. A robustness check on a run with more re-reads.

Transcripts are not committed (size); the script takes their directories as arguments.

## Metric

Per tool call the agent made (`distil_expand` excluded, results of at least three
comparable lines): **redundant** if ≥ 80% of its result's lines were already in an earlier
tool result as the client sent it; **avoidable** if, in addition, fewer than 80% were in
what distil forwarded in the request the agent was answering — the content was in the
conversation, but only as a digest. Lines are compared after stripping reader line-number
prefixes (`cat -n`, `view`, `grep -n`).

Cache-aware savings price each request's messages: unchanged since the previous request's
forwarded bytes at 0.1x, everything else at 1.25x, against the same requests uncompressed
(system and tool definitions excluded — identical in every variant). A **prefix rewrite**
is a message already forwarded that changed bytes on a later request; ADR 0008 forbids it.

## Results

`grepfix-300.json` (300 tasks, 1,293 requests, 894 calls, 61 redundant):

| variant | avoidable | request-size savings | cache-aware savings | final digests | prefix rewrites |
|---|---|---|---|---|---|
| baseline (rule off) | 36 | 11.90% | 9.34% | 210 / 1,189 | 0 |
| **refetch (shipped, ADR 0025)** | **31** | **11.21%** | **8.87%** | **194 / 1,189** | **0** |
| sticky (rejected) | 29 | 9.37% | 7.73% | 170 / 1,189 | 0 |
| sequence (rejected) | 22 | 7.83% | 6.68% | 179 / 1,189 | 0 |
| compound (rejected) | 8 | 3.48% | 3.37% | 119 / 1,189 | 0 |

`prefix-300.json` (300 tasks, 245 redundant): baseline 58 avoidable, refetch 55, sticky 50,
sequence 27, compound 8; cache-aware savings 10.75%, 10.29%, 9.62%, 5.75%, 2.86%.

`h2h-pilot-9x3.json` — a smoke replay, not a result: the first 9 tasks of each competitor
arm of the head-to-head pilot (`../swebench-outcome-300-h2h/`, still running when taken,
2026-10-05), replayed with `--arm`. These are short trajectories (3 to 13 steps), so the
rule has little to act on:

| arm's transcripts | requests | calls | redundant | avoidable, rule off → on | request-size savings, off → on | cache-aware, off → on | `tool_result_refetch` tokens | prefix rewrites |
|---|---|---|---|---|---|---|---|---|
| provider-cm | 35 | 22 | 1 | 1 → 1 | 5.48% → 5.39% | 6.47% → 6.21% | 113 | 0 |
| rtk | 33 | 19 | 0 | 0 → 0 | 9.45% → 9.45% | 18.8% → 18.8% | 0 | 0 |
| selective | 53 | 43 | 6 | 1 → 1 | 12.74% → 12.71% | 14.47% → 14.45% | 1,841 | 0 |

The rule fired where there was anything to fire on and avoided no re-read in this sample:
each avoidable call was a *first* re-read, which the rule cannot reach (ADR 0025). The
300-task medium run (`../swebench-outcome-300-medium/`) cannot be replayed: its transcripts
were not kept. `grepfix-300.json` stays the measurement; the adapter code it replayed is
unchanged on main since (1.57.0 touched no file under `distil/adapters/` or
`distil/compress/`).

`--claude-code ROOT` replays local Claude Code transcripts the same way (read-only, baseline
and shipped only, counts only: no command words, paths or text). No result from it is
committed.

The rule's own cost is the `tool_result_refetch` census bucket in each file. Re-running it:

```bash
uv run python benchmarks/reinflate_replay.py <run>_0 <run>_1 <run>_2 --out grepfix-300.json
```

## Live validation (not run — costs money)

The rule is on by default, so the distil arm needs no flag. `--reuse-arm` copies the plain arm's rows from
`../swebench-outcome-300/` (free against the budget), so only the distil arm runs, at the
settings of the earlier runs:

```bash
OUT=benchmarks/results/swebench-outcome-300-refetch
SRC=benchmarks/results/swebench-outcome-300
A="--arms plain,distil --model claude-sonnet-5-5 --effort low --max-steps 30"
python -m benchmarks.swebench_outcome plan $A --calibrate $SRC --reuse-arm plain=$SRC   # no spend
python -m benchmarks.swebench_outcome run $A --out $OUT --reuse-arm plain=$SRC \
  --budget-usd 10 --i-understand-this-costs-money
python -m benchmarks.swebench_outcome grade  --arms plain,distil --out $OUT
python -m benchmarks.swebench_outcome report --arms plain,distil --out $OUT
```

Needs `datasets` (and `swebench` to grade), as in `specs/swebench-outcome-eval.md`. `plan`
above estimates $8.07; the grepfix distil arm cost $7.18; `--budget-usd 10` caps it. Compare
against `../swebench-outcome-300-grepfix/report.md`: steps, cost, `distil_expand` calls and
resolved. Expect a small effect — five avoidable re-reads in 300 tasks offline.
