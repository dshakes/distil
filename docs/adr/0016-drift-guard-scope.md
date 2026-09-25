# 0016 — Drift guard scope: global per machine, and not the gateway

- **Status:** accepted
- **Date:** 2026-09-24
- **Relates to:** `distil/drift.py` (`fold`, `DriftGuard`), `distil/proxy.py`, `distil/aproxy.py`, `distil/gateway.py`, `tests/test_drift_guard.py`

## Context

The drift alarm used to be a report: an anytime-valid e-process over shadow mode's paired
verdicts that printed `BREACHED` when the wrap exited, while the proxy went on compressing.
It now acts. When the e-process crosses the budget (`distil.conformal.BUDGET_ALPHA`), every
guarded proxy serves lossless-only (Tier-0, no digest, no output shaping) until the hold is
released with `distil reset --drift-guard`.

Once a hold changes traffic, its scope matters. Too narrow, and a restart silently resumes
the compression that was just proven harmful. Too wide, and one workload's evidence
switches off compression for workloads it says nothing about.

## Decision

1. **One e-process per `DISTIL_HOME`, and one hold.** The capital lives in `drift.json`.
   Proxies are its only writers. Each proxy folds the verdict it just produced under a
   file lock (`drift.fold`), so every row is bet exactly once, in one global order.
   Restarts, hot-swaps and concurrent wraps all continue the same capital, so none of
   them re-branches from stale capital or adds extra looks. Every reporting surface only
   reads the state: the wrap exit, `distil stats` and the status line.
2. **The hold is global, not per session.** The paired verdicts that feed it come from
   one machine's agents running against one operating point. A per-session hold would let
   the next `distil wrap` resume lossy compression the moment after a certified breach.
   A long-lived proxy notices a trip or a release within `DriftGuard.POLL_S` seconds. It
   does this by stat-ing the state file from a daemon thread, never on the request path.
3. **The multi-tenant gateway is exempt.** `distil gateway` runs no shadow mode, so it
   produces no per-tenant drift evidence. The global e-process is fed by whatever local
   wraps and proxies share the host's `DISTIL_HOME`, and none of that evidence is about
   any gateway tenant. Holding every tenant on it would let one workload switch off
   compression for unrelated tenants, which breaks the gateway's per-tenant isolation.
   So the gateway neither feeds the guard nor obeys it.

## Consequences

- On a single-user machine, the alarm holds everything it has evidence about.
- **The gateway has no enforcing drift alarm.** That is a known gap, not an oversight. To
  close it, the gateway needs per-tenant shadow sampling, with one e-process and one hold
  per tenant key. Adding shadow to the gateway must come with that scoping. It must never
  reuse the global file.
- Releasing the hold is deliberate and narrow. `distil reset --drift-guard` archives only
  `drift.json` and leaves a fresh state in its place, so the next proxy start does not
  re-fold the same rows into the same breach. It does not touch the savings ledger or the
  shadow ledger. What a missing state means depends on the release archive:
  - **No `drift.json.reset-*` beside it:** a first-ever start (a fresh install, or an
    upgrade). The existing `shadow.jsonl` evidence is folded once, so harm the machine
    already measured is not discarded.
  - **A release archive beside it:** the user released before. The state starts fresh
    and never re-folds.
  - **State file and every archive deleted:** this is indistinguishable from a first
    install and re-bootstraps, which is accepted.

  A quarantined `.corrupt-*` file is not a release. An unreadable state is held. It may have
  recorded a breach, so it is copied aside before a held state replaces it, and it stays
  in place if either step fails. Known limits: a lock that cannot be taken fails open, so
  two processes may write two trip receipts; archives are never pruned.
