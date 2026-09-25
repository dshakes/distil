# 0018 — Cost-truth benchmark: billed dollars per solved task, metered at the edge

- **Status:** proposed (protocol draft; no live run; frozen by tag `cost-truth-protocol-v1` after competitor review)
- **Date:** 2026-09-25
- **Relates to:** `docs/research/cost-truth-protocol.md`, `benchmarks/cost_truth/`, `tests/test_cost_truth.py`, `benchmarks/results/cost_truth/cost_estimate.json`, `distil/pricing.py`, `distil/conformal.py` (`BUDGET_DELTA`, `CERT_MARGIN`), `distil/certify/stats.py`, `docs/research/live-savings-gap.md`, ADR 0012, ADR 0015
- **Numbering:** 0017 is `distil mcp` (branch `feat/mcp-compressor`); 0018 was free on `main` and every `origin/*` branch on 2026-09-25.

## Context

Every compressor in this space — distil included — reports savings in tokens it removed.
The bill is dominated by prompt-cache reads at 0.1x (72% of $ on distil's own live traffic),
so token savings and dollar savings diverge, sometimes in sign. Quesma's 1,740-run study of
RTK on Terminal-Bench 2.1 (2026-09-11) found an 89% `rtk gain` claim next to a −5%/+5%
bill change, most of it from one task. Distil itself measured a mode that halved tokens and
doubled cost (2026-08). A public comparison of distil against RTK and Headroom has to be on
the bill, has to include task success, and — because we build one of the tools — has to be
built so that we cannot tilt it.

## Decision

1. **Primary outcome is billed $ per solved task**, computed from the provider's `usage`
   object including cache reads and writes (1h writes at 2x), never from token estimates.
   Co-primary is success non-inferiority at `BUDGET_DELTA` (5 pp); `CERT_MARGIN` (2 pp) is a
   secondary because it needs ~5x the feasible sample.
2. **One neutral meter for every arm**: a content-free, usage-only reverse proxy is the last
   hop before the provider in all four arms. It also enforces a hard spend cap by reserving
   each request's worst case before forwarding.
3. **Terminal-Bench 2.1 via Harbor**, Claude Code headless, Sonnet 5 primary (89 tasks x 5
   seeds x 4 arms) and Haiku 4.5 replication (x 3 seeds); paired blocks, randomised arm order,
   cache-TTL spacing, fresh isolated home per run.
4. **Pre-registered, one look**, cluster bootstrap over tasks, Bonferroni over three
   comparisons; each tool's own savings claim is recorded next to the billed truth.
5. **Conflict of interest is handled structurally**: competitor versions pinned at latest GA
   with their documented default configs, distil tested from its published wheel, a 14-day
   competitor review before freeze, all raw per-run usage published regardless of outcome.
6. **This phase is $0**: protocol, harness, dry-run (mock upstream + scripted agent) and a
   cost estimate (hard caps: pilot $92, primary $2,048, replication $410; total $2,550).
   The live driver (Harbor custom agent + wrap shim) is built only after the arm specs are
   verified by preflight, and it must refuse to start while `arms.unverified()` is non-empty.
   There is deliberately no `live` subcommand yet.

## Consequences

* A negative or null result for distil will be published. That is the point.
* The study can detect roughly a 13% $/solved change (σ_w ≈ 0.6); smaller effects will be
  reported as not shown, not as absent.
* Claims on the site and README about cost savings vs competitors wait for this result; no
  number is added before the live run.
* Harness code lives under `benchmarks/`, not `distil/`: it ships nothing to users and adds
  no dependency (stdlib + `distil.pricing`/`distil.certify.stats`).

## Alternatives rejected

* **Tokens as the outcome** — the exact error this study exists to measure.
* **Each tool's own savings report** — the claims under test.
* **Claude Code transcript usage as the meter** — downstream of the arm's proxy; kept as a
  cross-check only.
* **SWE-bench Verified as primary** — Read/Edit-heavy, less terminal output (RTK's surface),
  heavier infra; kept as the replication suite.
* **distil's trajectory corpus** — no pass/fail, and we built it.
