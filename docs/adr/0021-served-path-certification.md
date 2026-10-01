# 0021 — Certify what is served: the `served` strategy

- **Status:** accepted (the strategy is shipped and graded; the result it produces on coding traffic is an open risk, not a pass)
- **Date:** 2026-09-30
- **Relates to:** `distil/compress/strategies.py` (`served`), `distil/adapters/anthropic.py` (`compress_messages`), `distil/replay/expand_runner.py` (`ExpandAwareRunner`, `build_restore`), `distil/replay/realtrace.py` (`load_swe_bench`), `benchmarks/model_migration_eval.py`, `benchmarks/swebench_outcome/`, `specs/swebench-outcome-eval.md`, `benchmarks/results/model-migration/summary.json`, `docs/model-migration.html`, ADR 0003, ADR 0008

## Context

`distil certify` grades the `distil` strategy. That strategy digests only the volatile blocks of a turn, which was the right object when the certificate was designed. It is not what a tool-use client is served. A caching client such as Claude Code or an Agent SDK loop puts its cache breakpoint on the newest message, so everything before it is committed prefix, and the serving adapter (`distil.adapters.anthropic.compress_messages`) digests every earlier tool result.

Nobody had measured how far apart those two are. On real SWE-agent trajectories (4,616 turns) the certified `distil` strategy saves 2.6% of tokens, while the served path saves 52.6%. Most of what is served was never certified: the certificate was true of a slice of the traffic that carries almost none of the savings.

## Decision

1. **A new strategy, `served`** (`distil certify --strategy served`, and a `served (adapter)` point on the `frontier`). It does not reimplement compression. It builds the Messages request a caching tool-use client would send for the turn, runs the real `compress_messages` on it, and maps the result back onto the blocks:
   - stable system and tool blocks stay out of the request, because the system prompt is never rewritten;
   - each tool output becomes a `tool_result` answering a paired assistant `tool_use`; history becomes assistant text; everything else is user text;
   - the cache breakpoint sits on the newest message, so nothing is carved out as "recent": the freshest output digests too, and each block's served bytes depend only on its own text, which keeps the same block byte-identical on every turn that carries it (ADR 0008);
   - reject-if-bigger applies on top, and handles are the adapter's own (`sha256` of the original tool result, eight characters), so the recovery loop resolves them.
2. **The recovery loop commits through the runner's structured decision tool.** `ExpandAwareRunner` now finishes through `structured_decision`, the same strict tool the plain arm uses. The free-text format difference alone had flipped about one action in five, which read as compression harm and was a grading artefact.
3. **Live renders strip `DECISION:` annotation lines.** They are offline-oracle markers and they leaked the answer to live graders. The deterministic runner is unaffected.
4. **`load_swe_bench` reads the current SWE-agent `.traj` history format:** system prompt, tool menu and issue, actions and observations in the right order, gold = the next command. Before, it dropped the issue and the system prompt and mis-paired steps, so coding decision points were not real decision points. Exact-quote provenance now covers `goto`, `scroll_up` and `scroll_down` file views, like `open`.
5. **The claim is stated as open.** The served strategy is reported beside the certified one, never merged into it, and the docs say what it shows.

## What it certifies, and what it does not

Certifies: the digests of earlier tool outputs as the serving adapter produces them, for a client with a breakpoint on the newest message, graded by the same certifier and metrics as every other arm.

Does not certify:
- a client with no cache breakpoint, or one earlier in the history. Its newest turns stay verbatim (gentler), but its uncached blocks get query-aware salience and superseded shell reads may digest; neither is exercised here;
- the exact-quote exemption beyond the tool names the trajectory reveals;
- images (`vision` certifies media), verbatim mode, subscription mode;
- a sub-span handle from a `<distil:keep>` span or a re-read delta. `build_restore` does not map it, so the recovery arm fails to recover it, which errs harsher, the safe direction;
- **task success.** A next-action match is a decision-level result. Whether a digested history costs solved tasks is the question of `specs/swebench-outcome-eval.md`, which has not been run.

## Evidence

From `benchmarks/results/model-migration/summary.json`, 100 SWE-agent decision points (59 held out), `claude-sonnet-5-5` at low effort, $0.1797 per case:

| arm | next action kept |
|---|---|
| two identical uncompressed calls (noise floor) | 94.1% |
| certified `distil` alone | 51.7% |
| certified `distil` with recovery | 77.1% |
| `served` alone | 59.3% |
| `served` with recovery | 78.0% |

Served with recovery keeps the next action in 78.0% of held-out decisions against 94.1% for two identical uncompressed calls. That is a real gap of about sixteen points, on a small set, at a decision level that is stricter than task success (a different command can still reach the same outcome). It is not a verdict that serving hurts task success. It is the reason the outcome eval exists.

## Alternatives rejected

- **Leave certification on the `distil` strategy.** It certifies 2.6% of tokens and says nothing about the 52.6%. Keeping it as the only gate would keep a number on the page that is true of the wrong object.
- **Change the serving adapter until the gap closes, then certify.** The gap is what should be measured first. Tuning the adapter to a metric before the metric is validated against task outcomes optimises the instrument.
- **Replace `distil` with `served` as the default gate.** The served arm does not match the noise floor on coding traffic, so a default gate on it would fail today. It is added as a separate strategy so the two can be compared honestly.

## Consequences

- The frontier and `distil certify` can now grade the served path. On coding traffic the served arm is graded against the same noise floor as every other arm, so the gap above is now visible instead of hidden.
- The 78.0% and 94.1% figures are a decision-level risk on one model and one set. They are quoted with that scope wherever they appear.
- `benchmarks/swebench_outcome/` (a plain versus distil-served agent on SWE-bench Lite, the official grader, paired McNemar) is built and tested offline and **has not been run**. No page claims a task-success result from it.
