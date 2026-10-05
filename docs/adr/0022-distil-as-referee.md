# 0022 — distil as referee: score any compressor in dollars and decisions

- **Status:** proposed
- **Date:** 2026-10-04
- **Relates to:** `distil/referee.py` (`distil audit`), `distil/proxy.py` (`_spawn_audit`),
  `distil/shadow.py` (the paired estimator), `distil/abtest/` (session holdout),
  `distil/certify/provider_compaction.py`, `benchmarks/swebench_outcome/` (arms),
  `benchmarks/cost_truth/` (ADR 0018), `scripts/build_scoreboard.py`, `docs/scoreboard.html`,
  `docs/research/referee-runs.md`, `tests/test_referee.py`, `tests/test_scoreboard.py`

## Context

Every context compressor publishes a saving: the tokens it removed. None of them publishes
what the user pays: dollars per solved task with the failed attempts counted, and whether
the agent still makes the same decisions. A tool that removes 60–90% of a tool output's
tokens can leave the bill roughly where it was once caching, retries and extra turns are
counted. Quesma's public measurement of RTK showed this.

distil already has every instrument needed to answer the question for itself:

- the **paired shadow** (ADR 0001, `shadow.SIG_VERSION` 5): A, A′ and B replays of one
  request, the per-request statistic `1{A==B} − 1{A==A′}`, its bootstrap interval, and the
  one reporting floor (50 A/B, 30 A/A);
- the **session holdout** (ADR 0019, `distil ab`): cost per session, turns and tasks, with
  an anytime-valid interval;
- the **provider certificate** (`distil certify-provider`): Anthropic context editing,
  measured with an A/A arm on a pre-registered protocol
  (`benchmarks/results/provider-compaction/`);
- two **outcome harnesses** that run competitors through their real pinned packages:
  `benchmarks/swebench_outcome` (arms `plain`, `distil`, `rtk`, `selective`, `provider-cm`)
  and `benchmarks/cost_truth` (Terminal-Bench, neutral meter; arms `control`, `rtk`,
  `headroom`, `distil`).

Nobody else measures this. The decision is to point those instruments at other compressors
as well as distil, and publish the results under one set of rules.

## Decision

1. **What is measured.**
   - *Offline, per task* (the scoreboard): dollars per **solved** task, i.e. the arm's
     total spend on the commonly graded tasks divided by the tasks it solved, so failed
     attempts are paid for. Also the success rate with a Wilson interval, the paired
     difference against plain (Wald CI, exact McNemar, pre-registered 5-point
     non-inferiority margin, `swebench_outcome.report.MARGIN`), tokens, steps, n, run date
     and pinned versions. On Terminal-Bench the estimand is cost_truth's own
     (`R_A`, a paired task-cluster bootstrap; ADR 0018).
   - *Live, per request* (`distil audit`): the decision change relative to the model's own
     self-agreement, `1{A==B} − 1{A==A′}`, and the dollar difference between A and B at
     uncached list price, both with bootstrap intervals.
   - *Live, per session*: turns and cost per session stay with `distil ab`. A replay is one
     request, so `distil audit` does not claim to measure turns, and its report says so.

2. **Estimators are reused, not reinvented.** `distil audit` writes rows in the shadow's
   own `kind: "paired"` format and reads them with `ShadowLedger.equivalence()`, so its
   interval, floor and `below_floor` logic are the shadow's. Dollars use
   `shadow.bootstrap_ci`. The scoreboard calls `swebench_outcome.report.analyse`, the
   function that writes each run's `report.md`, and reads cost_truth's `analysis.json`.

3. **The verdict.** It is computed per compressor:
   - **can't tell yet** below the shadow's floor, on the requests where the compressor
     actually changed something, with the number still needed;
   - **hurt** when the decision interval lies entirely below 0 (more changed actions than
     the model's own noise), or the dollar interval lies entirely below 0;
   - **helped** only when the dollar interval lies above 0 *and* the decision interval's
     lower bound is within 5 points of self-agreement;
   - otherwise **can't tell yet**, with an approximate sample size from the observed spread.

4. **Other tools run as themselves, never as imitations.**
   - *Offline:* only through the harness arms, which install or call the real pinned
     package (RTK v0.51.0 binary, `selective-context==0.1.4`, `headroom-ai==0.38.0`,
     Anthropic's API parameter). The scoreboard reads those runs.
   - *Live:* `distil audit` builds B only where that is faithful:

   | compressor | B in `distil audit` | faithful? |
   |---|---|---|
   | `distil` | the body distil actually served | yes |
   | `anthropic-context-editing` | the same request plus `context_management: {"edits": [{"type": "clear_tool_uses_20250919"}]}` (documented defaults: trigger 100k input tokens, keep 3 tool uses) and beta `context-management-2025-06-27`; fired iff the response carries `context_management.applied_edits` | yes: the provider applies its own edit |
   | `openai-compaction` | Responses API only: `context_management: [{"type": "compaction", "compact_threshold": 200000}]` (the documented example; no default is documented); fired iff a `compaction` output item comes back | yes, with the same caveat |
   | `headroom` | the same request sent through the user's own running Headroom proxy (`--via`, loopback only); A and A′ go straight to the provider | yes: Headroom's own code shapes B. UNVERIFIED against a live Headroom in this change |
   | `rtk` | none | **no.** RTK rewrites commands inside the agent before they run, so the request distil sees already holds RTK's output and the uncompressed output never existed. RTK is scored offline only. |

   A request that already carries `context_management` is audited the other way round: A
   drops it. That is the "upstream tool already in the path" case. For Headroom placed in
   front of distil, distil only ever sees post-Headroom bodies, so it cannot be audited and
   the topology is refused by construction (`--via` must point at Headroom, not at distil).

5. **Opt-in, capped, content-free.**
   - It is off until `distil audit --compressor X` is confirmed (a prompt, or `--yes`). The
     disclosure names the rate, the cap and where B is sent.
   - The default is a 0.02 sample rate and a **$1/day hard cap**. A sample reserves its
     estimate before its first replay: input at the base rate, about 3.5 bytes per token,
     plus output capped at 4,096 tokens, times three arms. It settles to the
     provider-reported cost under a file lock, so concurrent sessions cannot jointly
     overshoot. A sample that would not fit never sends a byte.
   - An unpriceable model is never audited. A failed arm stops the sample.
   - Replays are booked as overhead (`_book_overhead("audit", …)`), so savings and
     `distil ab` net them out.
   - The ledger (`~/.distil/audit.jsonl`) holds signatures (hashes), token counts and
     dollars. It never holds content. Nothing leaves the machine except to where the
     request was already going, or to the loopback proxy the user named.
   - It is never active on a held-out (`distil ab`) session or a diagnostic handler.

6. **The scoreboard shows only artifacts.** `scripts/build_scoreboard.py` recomputes every
   number from committed run files listed in `benchmarks/results/scoreboard-runs.json`.
   An arm with no rows reads **pending run**; nothing is estimated. Each public number has
   a `docs/claims.json` entry, and `tests/test_scoreboard.py` fails when the page drifts
   from its artifacts.

## Consequences

- distil is one row on its own scoreboard. Its current rows are not a win: −1.7 and
  −2.0 points, non-inferiority not shown, at cost parity. They are published under the
  same rules as everyone else's.
- The live audit is slow to say anything. At the default cap a ~100k-token Sonnet request
  costs about one sample a day, and the floor is 50 paired samples where the compressor
  fired. The report says "can't tell yet (need N more)" instead of a number; users who
  want an answer sooner raise `--cap-usd`.
- Dollars per request are priced at uncached list price. Prompt-cache effects are real.
  Context editing invalidates the cached prefix when it fires, and that is the main way it
  can cost more. But a single replay's cache state is not the steady state, so cache-aware
  dollars come from the outcome harnesses and `distil ab`, not from `distil audit`.
- The reservation is not a worst case. A sample whose cache writes (1.25–2× input) push
  it past its estimate overshoots by that difference, once; every later sample sees the
  overspend. Reserving every byte as a 1-hour cache write would have refused every large
  request at the default cap.
- Decision change is a stricter test than task success: a different command can reach the
  same outcome. That is why the per-task scoreboard exists alongside it.
- Paid runs to fill the scoreboard are planned, with costs, in
  `docs/research/referee-runs.md`. None was run in this change.
