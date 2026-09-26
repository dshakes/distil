# 0019 — Session-level A/B: a randomised holdout, model-aware

- **Status:** proposed
- **Date:** 2026-09-25 (revised the same day after an independent statistical review)
- **Relates to:** `distil/abtest/` (`assign`, `ConversationArms`, `outcomes`, `stats`, `report`), `distil/proxy.py` (`build_handler(arm=)`, `wrap_run`, `serve`), `distil/hotswap.py` (`WorkerConfig.arm`), `benchmarks/abtest_montecarlo.py`, `tests/test_abtest.py`, `tests/test_expand_usage.py`, `docs/ab.html`, `docs/research/expand-undercount.md`

## Context

Every savings number distil prints today is per request: the tokens removed from what was
sent. It cannot say what a session cost, and it cannot survive a model change. A new model
version changes turns, cost per turn and cost per session by itself, and a before/after
comparison books all of that to distil. In the known-answer Monte Carlo, a model change
that adds 40% turns in both arms reads as roughly +40% cost on the distil arm before and
after, while distil's true effect was −20% throughout.

The existing holdout (`distil/certify/holdout.py`) prices a simulation of fixed
trajectories. The shadow A/A′/B (`distil/shadow.py`) measures per-request decision change,
not cost.

## Decision

1. **Randomise sessions, not requests.** Under `distil wrap`, each new session
   (`DISTIL_SESSION`) is assigned `distil` or `holdout` by
   `HMAC-SHA256(per-install secret, session id) < rate`. The arm is written to the
   manifest, and a nested or resumed wrap reads it back instead of re-drawing. Under a
   managed install (`distil proxy`, no wrap), the unit is the client's conversation:
   - Claude Code's own `X-Claude-Code-Session-Id` header, which was verified in the
     2.1.282 bundle and carries the same id as `metadata.user_id.session_id`;
   - otherwise the metadata field;
   - otherwise a lineage hash that survives proxy restarts.

   The first draw is persisted as a keyed hash in `ab-conversations.json`, so a resumed
   conversation keeps its arm across restarts and rate changes. The secret is not the
   census id.
2. **The holdout forwards the original bytes.** Every lever is off: no Tier-0, digest,
   expand tool, shaping, cold-point, prefix replay or shadow replay. That is always safe.
   Holdout requests are not booked as savings.
3. **Default 5%, disclosed, opt-out.** The rate comes from `DISTIL_HOLDOUT_RATE`, then
   `ab.json`, then the 5% default. A value of 0 disables it, and so does an unparseable
   env value. It is disclosed by `distil setup`, `distil proxy`, `distil ab`, and the wrap
   of every held-out session.
4. **Outcomes survive the sweep.** Randomised sessions and conversations are folded,
   content-free, into `ab.jsonl` before the 7-day session sweep. Cost is list price over
   the provider's own usage. It includes every upstream call distil made on the session's
   behalf: expand re-queries (summed per request since the accounting fix) and shadow
   replays (`overhead.jsonl`).
5. **The headline estimand is mean list-price cost per randomised session**, distil ÷
   holdout, over every randomised session in the current era that has ended:
   - failed sessions are counted at their actual cost (usually $0);
   - zero-cost sessions stay at 0;
   - unpriceable sessions (non-Claude models) are counted, per arm, and excluded.

   Turns per session, tasks per session, cost per task and cost per turn are
   **mediators**: outcomes distil can change, reported beside the headline and never as it.
6. **Headline inference is anytime-valid without asymptotics.** One empirical-Bernstein
   betting confidence sequence per arm (Waudby-Smith & Ramdas 2023, PrPl-EB) runs on
   per-session cost winsorised at a cap fixed in advance, **$500**, at α/2 each. That
   gives an interval on the capped-mean ratio. α = `conformal.BUDGET_DELTA`. This is the
   one primary claim.

   Everything else is secondary, asymptotic and not multiplicity-adjusted:
   - the efficient estimate (strata-pooled, CUPED on workspace pre-period cost per
     session, normal-mixture mSPRT);
   - the mediators and the per-cell table;
   - the rollout difference-in-differences.

   A sliding `--window` is a fixed-n look and is labelled as such.
7. **Strata and eras.** The stratum is the agent loop's model (the first request that
   carries tool definitions), which is fixed before treatment acts, together with the
   client at major.minor. A newer version supersedes the older one, and the rollout is
   reported as a DiD. Within a stratum, eras are cut at a holdout-rate change and at a
   self-starting CUSUM alarm (k=0.5, h=12) on the holdout arm's log cost per session. A new
   era starts after the alarm, and transition sessions are dropped.
8. **Scope.** Every number is about one machine's own sessions. Nothing is pooled across
   installs. Any future cross-install estimate must cluster by install. The census
   receives nothing new.

## Why these choices (and what the review changed)

- **Cost per session, not cost per task.** Tasks are user turns, and compression can
  change how often a user re-asks. A session that is 10% dearer with 25% more re-asks
  reads as 12% cheaper per task. Only the randomised unit has an unaffected denominator.
  In the simulation of exactly that case, the headline recovers +10%, and cost per task
  reads a saving.
- **A heavy-tail-robust interval.** The first version used the normal-mixture mSPRT for
  the headline. At a 5% holdout with session log-sd 1.5 it rejected a true null 8.3% of
  the time against a 5% budget, and 10.7% at log-sd 2.0. The maintainer's own sessions
  measure log-sd 1.94. Bounded data is all the EB sequence needs. The price is the
  estimand: the capped mean. Capping trims the dearer arm more, so a real saving reads
  smaller, never larger. The report counts the capped sessions per arm.
- **The detector reads the holdout arm, and it is chosen on principle, not on a measured
  gain.** An earlier revision read both arms, based on simulations at a 50/50 split. At
  the shipped 5% the treated arm is 95% of what an arm-blind detector sees, so a distil
  upgrade can look like a model change. At that rate the tracked Monte Carlo finds the
  choice barely matters for the headline: under a silent +40% model change, arm-blind,
  holdout-only and no detector all land within noise of each other. A quick 200-run probe
  had suggested +3.2 pp against +1.1 pp. The tracked run did not reproduce it, so that
  figure is not claimed. The threshold is h=12, not 8: h=8 false-alarmed on 12% of
  1,000-point stationary series, and each false era throws away scarce data.
- **The stratum is fixed before treatment.** "The model that carried most cost" is
  chosen after compression has acted. Claude Code's first requests are Haiku side calls,
  so "the first request's model" is not usable either. The agent loop's model is.
- **Failed sessions are kept.** Dropping sessions with no priced 2xx made distil-caused
  failures vanish from the distil arm. How each arm's sessions ended is printed as a
  balance check.
- **Randomisation, not change points, is what protects the estimate.** Both arms run
  concurrently, so a model change cancels in the contrast even inside one era. Eras only
  make the number describe the current regime.

## Consequences

- **Being honest costs sample size.** At 5% and a session cost CV of 1.5, resolving a
  10% effect takes tens of thousands of sessions. The report says "Individual results
  take months to become conclusive: need ≈N sessions" up front, and a larger holdout gets
  there faster.
- **The capped-mean estimand is conservative for savings.** Its size depends on how many
  sessions exceed the cap. `DISTIL_AB_COST_CAP` overrides the cap; changing it changes
  the estimand.
- **Capping and the ratio of means both bias a saving toward zero, never away from it.**
  Against the uncapped truth, a true −20% read about +2 pp smaller when sessions averaged
  $150–$210 against the $500 cap. The Jensen term of the ratio (about CV²/n₀) is smaller
  than that. It is documented, not corrected.
- **Power at the shipped rate is low, by design.** A true −20% at log-sd 1.0 is detected
  a few percent of the time after 4,000 sessions and about three times in four after
  20,000.
- **A managed conversation that spans a proxy restart is folded as two sessions** in the
  same arm. That is symmetric across arms.
- **No DiD is taken across a CUSUM era.** A closed era carries the detector's selection
  bias. The DiD across a version rollout does not.
