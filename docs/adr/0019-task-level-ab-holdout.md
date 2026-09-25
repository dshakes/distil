# 0019 — Task-level A/B: a session-randomised holdout, model-aware

- **Status:** proposed
- **Date:** 2026-09-25
- **Relates to:** `distil/abtest/` (`assign`, `outcomes`, `stats`, `report`), `distil/proxy.py` (`build_handler(arm=)`, `wrap_run`), `distil/hotswap.py` (`WorkerConfig.arm`), `benchmarks/abtest_montecarlo.py`, `tests/test_abtest.py`, `docs/ab.html`

## Context

Every savings number distil prints today is per request: tokens removed from what was
sent. It cannot say what a *task* cost, and it cannot survive a model change. A new model
version changes turns per task, cost per turn and cost per task by itself. A before/after
comparison books all of that to distil. In the known-answer Monte Carlo, a model change
that adds 40% turns per task in both arms reads as +40% cost on the distil arm
before/after, while distil's true effect was −20% throughout.

The existing holdout (`distil/certify/holdout.py`) partitions offline trajectories and
prices a simulation. It measures what compression would do to a fixed trajectory, not what
it does to a live agent's behaviour. The shadow A/A′/B (`distil/shadow.py`) measures
decision change per request, not cost per task.

## Decision

1. **Randomise sessions, not requests.** A new wrap session (`DISTIL_SESSION`) is
   assigned `distil` or `holdout` by `HMAC-SHA256(per-install secret, session id) < rate`.
   The session is the unit because the outcome (a task's cost) is a property of the whole
   trajectory, and compressing half of one conversation would contaminate both arms. The
   assignment is written to the session manifest, and a nested or resumed wrap that
   inherits the id reads it back instead of re-drawing. The secret is not the census id:
   `distil census off` must not re-randomise anyone.
2. **The holdout forwards the original bytes.** Every lever is off: no Tier-0, no digest,
   no expand tool, no output shaping, no cold-point, no prefix replay, no shadow replays.
   The request goes upstream as `raw`. That is always safe, because it is exactly what the
   agent would have sent without distil. It is also safe on subscription and lossless-only
   sessions. Receipts and the per-request record still run, tagged `arm=holdout`. Nothing
   is booked to `savings.jsonl`, so the per-request savings view stays a statement about
   compressed traffic only.
3. **Default 5%, disclosed, opt-out.** `DISTIL_HOLDOUT_RATE` > `ab.json` > 5%. A value of
   0 disables the holdout, and so does an unparseable env value, because the only reading
   that cannot override an opt-out is "off". The rate is printed by `distil setup`,
   by `distil ab`, and by the wrap itself on every held-out session.
4. **Outcomes survive the sweep.** `sessions/*` is swept after 7 days, and a 5% holdout
   needs months. Each randomised session is folded, content-free, into `ab.jsonl` at wrap
   start (before the sweep) and at wrap exit. Dollars are the provider's own usage fields
   priced at list, including cache reads (0.1×) and writes (1.25×). Unknown models are not
   priced.
5. **Estimation.** The unit is the session. The metric is a ratio of per-arm sums (cost
   per task, turns per task, cost per turn), and the effect is the log ratio distil/holdout,
   with delta-method variance. CUPED uses the workspace's mean cost per task from
   sessions that ended before this one started (independent of this session's arm).
   Strata are model × client (major.minor). A newer version in the same family × client
   *supersedes* the older one, so the headline is within the new model only, and the pair
   is reported as a difference-in-differences. Within a stratum, eras are cut at a
   holdout-rate change and at a self-starting two-sided CUSUM alarm (k=0.5, h=12) on log
   cost per task. The headline is an inverse-variance pool of the current-era cells. The
   difference-in-differences is taken only across a rollout: the old version's last era
   against the new version's first.
6. **Anytime-valid inference.** The interval is the normal-mixture confidence sequence
   (mSPRT, τ=0.2 on the log scale) at α = `conformal.BUDGET_DELTA`, the single failure
   budget. Continuous monitoring from any surface does not inflate error. A percentile
   bootstrap is printed as a fixed-n cross-check. Below 20 holdout sessions in the current
   cells, the report says "not enough data yet (n=…, need ≈…)".

## Why these choices

- **Change detection is arm-blind.** The first draft ran CUSUM on the holdout arm only,
  as specified, because that series is the one distil cannot move. The Monte Carlo showed
  that any change detector selects what it cuts on. Reading one arm put all of that
  selection on one side of the contrast, and eras closed by a holdout-only alarm were
  biased by several points. Reading both arms spreads the selection evenly across the
  contrast, and at 5% it gives 20× the data. The cost of that: a change in distil's own
  effect can open an era, so an era is labelled "shift", never "model change".
- **Eras start after the alarm.** Starting at the estimated change time kept the very
  sessions that tripped the detector, and it biased the post-change contrast. A new era
  starts at the first session after the alarm, which is a stopping time: no data in it
  informed the decision. The sessions in between are dropped from both arms.
- **h=12, not the textbook 5.** At h=5 (ARL₀ ≈ 465) a stationary series false-alarms at
  about the rate the ARL predicts. At h=8, a simulated 7,500-session history still got
  two false eras, and the current era was left with 7 holdout sessions. Every false era
  throws away the data before it. The arm-blind series is dense, so the extra detection
  delay is cheap: at h=12 a 1σ shift is flagged in a median 22 sessions. A missed shift
  costs only a mixed-regime average, because randomisation keeps the contrast unbiased.
- **Randomisation, not change points, protects the estimate.** Because both arms run
  concurrently, a model change moves both and cancels in the contrast even inside one
  cell. Eras exist so the number describes the current regime.

## Consequences

- The number is causal and survives a model change. It costs what the holdout sessions
  would have saved (printed as "cost of the holdout").
- At 5%, resolving a 10% effect takes tens of thousands of sessions. A 20–30% effect
  takes thousands. CUPED helps, and a larger rate helps more. The report says "not
  enough data yet" rather than printing noise.
- Only `distil wrap` sessions are randomised. A managed install (`distil proxy` under a
  launch agent) has no session id and is not in the experiment.
- Streaming `distil_expand` re-queries report only the first request's input usage, so
  the distil arm's cost is a lower bound when expand fires. The report counts those
  requests and says so.
- No DiD is taken across a CUSUM era. It is not a rollout, and a closed era carries the
  detector's small selection bias.
- Real session costs drift with the workload (a different project on a different day).
  The detector will read some of that as a shift. That costs data, never validity.
