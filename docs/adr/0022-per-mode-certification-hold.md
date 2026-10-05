# 0022 — The drift guard decides per mode, on the bound every verdict already uses

- **Status:** accepted
- **Date:** 2026-10-04
- **Relates to:** `distil/drift.py` (`uncertified`, `guarded`, `DriftGuard`), `distil/shadow.py` (`equivalence_by_mode`), `distil/cli.py` (`shadow-stats`), ADR 0016, `tests/test_drift_guard.py`

## Context

The public live number, "97.5% decision-equivalence, CI [95.5, 99.5]", comes from
`benchmarks/results/shadow-live-2026-09-15.json`. It pools two modes. The same file's
`by_mode` block, read per mode under the same paired estimator:

| mode | A/B agree | A/A agree | paired harm | 95% CI of harm |
|---|---|---|---|---|
| digest | 102/206 | 113/206 | **5.3 pp** | at best [2.4, 8.7] |
| lossless-only | 109/192 | 109/193 | ~0 | inside the budget |
| pooled | 211/398 | 222/399 | 2.5 pp | [0.5, 4.5] |

(Only marginals were published, so the digest interval is computed for the pairing most
favourable to digest those counts allow — 11 one-way flips, none back. Any other pairing
widens it.)

Digest alone is over the 5% budget (`conformal.BUDGET_ALPHA`) on its point estimate, and
its upper bound is well over it. The pooled number is inside the budget only because
lossless-only, which changes almost no bytes, is averaged in.

The 1.54.0 guard ("one risk budget and a drift alarm that acts") did not hold digest. Two
reasons, both reproduced in `test_the_published_ledger_pooled_passes_but_digest_alone_is_not_certified`:

1. **Pooled.** `drift.fold` and `_bootstrap` bet every paired row, whatever mode produced
   it. Lossless-only rows (harm ~0) pull the capital down; on these counts the pooled
   e-process ends at capital ~0.08.
2. **Wrong question for this failure.** The e-process tests `H0: harm <= alpha` and trips
   only when a breach is *proven* (capital >= 1/delta = 20). Harm sitting near the budget
   is undetectable by design: digest's rows alone end at capital ~0.9. The guard was not
   under-powered so much as never able to say "not certified".

Meanwhile the published verdicts (`shadow-stats`'s "inside the 5% certified budget", auto output shaping
in `output.resolve_shape_output`) calls a mode "inside the budget" only when the upper end
of its paired-harm interval is. The guard and the verdicts used opposite burdens of proof.

## Decision

1. **Report per mode, always.** `ShadowLedger.equivalence_by_mode()` applies the same
   paired estimator and bootstrap CI to each mode. `shadow-stats` prints each mode's
   paired difference with its interval directly under the headline, and labels the
   headline "POOLED across modes" when there is more than one. `--json` and `--record`
   carry the per-mode difference, CI and floor flag.
2. **Hold on the certification rule, per mode.** `drift.uncertified()` holds the proxy at
   lossless-only when, for a mode the hold can switch off (`digest`, or rows with no mode),
   the paired-harm upper bound (`-diff_ci[0]`, the bootstrap 95% CI already reported) is
   above `BUDGET_ALPHA`. It is the same rule and the same confidence convention as the
   published verdicts, not a new threshold.
3. **Not trigger-happy.** It acts only above the shared reporting floor
   (`VERDICT_MIN_AB` = 50 paired rows *for that mode*), only on the last
   `SHAPE_EVIDENCE_DAYS` (7) of rows, and never on lossless-only or verbatim rows, whose
   harm a lossless-only hold cannot reduce.
4. **Not sticky, but with hysteresis.** Unlike a proven breach, this hold is re-derived
   from evidence at proxy start and hourly by the existing watcher. It **engages on the
   first over-budget check** and **lifts only after two consecutive clear checks**
   (`drift.CERT_CLEAR_CHECKS`), at least 30 minutes apart (`CERT_CLEAR_GAP_S`, so a
   second proxy or a restart re-checking minutes later does not count twice). The state
   (reason, clear streak, time of the last counted clear) is persisted in
   `~/.distil/cert-hold.json` under the drift lock with an atomic replace, so a restart
   does not reset it; an unreadable file fails closed (held, streak zero) and is released
   by the same two clear checks. Without this, a bound sitting on the budget flipped the
   served mode every hour — each flip rewrites the request shape and costs a prompt-cache
   rebuild. Liveness is unchanged: rows served while held are booked as lossless-only, so
   once the window passes digest drops below the floor, the next two checks are clear,
   and digest resumes and has to re-earn its certificate on fresh traffic. `distil reset
   --drift-guard` releases the e-process only and says when the certification hold
   remains. `DISTIL_NO_DRIFT_GUARD=1` opts out of both, as before.
5. **The e-process stays, folding guarded modes only** (`drift.guarded`; schema 3, so a
   pooled schema-2 state is rebuilt once from the ledger unless it is already held).

## Consequences

- A machine whose recent digest traffic looks like the 2026-09-15 ledger is served
  lossless-only, with a startup line naming the mode, the harm, the bound and the window.
  This is per machine and evidence-driven, not a global default change: with no shadow
  evidence, or clean evidence, digest runs exactly as before.
- Expect the hold to cycle on traffic whose true harm is near 5%: held for a window,
  resumed, re-measured. That is the honest behaviour for a mode that cannot be certified.
  The hysteresis makes the cycle slow (at least two clear hourly checks to resume, one
  bad one to hold), not absent; `tests/test_drift_guard.py` pins the oscillating-bound
  case (never lifts) and the aged-out case (lifts after two clear checks, across a restart).
- Public copy states the per-mode result instead of the pooled 97.5%.

## Alternatives considered

- **Keep the e-process alone, per mode.** Correct about false alarms, but it can never
  act on "near the budget", which is exactly the case in the evidence.
- **One-sided 95% bound** (tighter than the two-sided CI's end). Rejected for
  consistency: the published verdicts use the two-sided interval, and two rules for one
  budget is the drift this ADR removes.
- **Turn digest off globally.** Not supported by the evidence: it is one machine's
  ledger, and the hold already does this wherever the evidence says so.
