# Terminal-Bench 2.1 cost-truth pilot, 2026-09-25

Pilot phase of `benchmarks/cost_truth`: arms control, rtk (0.50.0), headroom (0.38.0) and
distil (1.54.0), model `claude-sonnet-5`, Claude Code 2.1.282 under harbor 0.23.0, 10 tasks
x 2 seeds. 85 runs were recorded in `runs.jsonl`; billed spend was $47.41 (manifest.json).
Protocol section 7 applies: a pilot does not compare arms. `analysis.json` therefore carries no
passing verdict for any arm (`"verdict": "not shown"`), and nothing here names a winner.

Regenerate the analysis for $0: `python -m benchmarks.cost_truth analyze <this dir>`.

## Read these before using any dollar figure

- **Prices are overstated 1.5x.** The run was billed while distil priced `claude-sonnet-5` at
  $3 / $15 per million tokens. The official price is $2 / $10 (fixed in #228). Every absolute
  `cost_usd` in this directory, and every `$ total` and `$ per solved task` in
  `analysis.json`, is 1.5x too high. Ratios between arms use one price table for all arms, so
  they are unaffected. The files were not rewritten.
- **Emulated x86_64.** The run executed under emulated x86_64 on an arm64 Mac (protocol
  Amendment 2). Wall-clock times (`wall_s`, `mean_wall_s`) include emulation overhead and
  must not be read as native speed. Token counts and outcomes are not affected by the emulation.
- **10 runs ended `infra_error`** (4 headroom, 4 distil, 2 rtk), all `harbor exit 2, no
  result.json`, all on seed 0 of `fix-ocaml-gc` and `protein-assembly`, retries included.
  An `infra_error` is not attributable to an arm, so analysis (`runner.py`, `EXCLUDED`)
  removes the whole (task, seed) block from every arm, the control's run included. That leaves
  18 of 20 blocks, so 18 attempts per arm, and `analysis.json` reports `excluded_runs: 10`.
  Their spend is not in the per-arm totals. They are not counted as failures.
- **Small.** 18 attempts per arm; all intervals are wide and every cost and success
  comparison is `inconclusive`.

## Publishability

`meter/*.jsonl` (84 files, 1,741 lines) holds usage metadata only: model, path, status, byte
counts, timings, token usage and `cost_usd`. It has no request or response bodies, prompts,
transcripts or keys. Nothing was dropped from any file. `grep -rn "sk-ant"` and a search for
the maintainer's home path both come back empty.
