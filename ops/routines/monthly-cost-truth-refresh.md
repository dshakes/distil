# Routine: monthly cost-truth refresh

| | |
|---|---|
| **Schedule** | Monthly. |
| **Trigger** | Manual (or a calendar reminder — this routine has no workflow; see "Why no workflow"). |
| **Inputs** | `python -m benchmarks.cost_truth {plan,estimate,power}` (all $0, offline), the previous month's `benchmarks/results/cost_truth/cost_estimate.json` and `analysis.json` if present. |
| **Outputs** | An updated `cost_estimate.json` (free to regenerate), and a **spend plan**: the exact live-run command, its pre-registered hard cost cap per phase, and what changed since last month. Never the live run itself. |
| **Approval point** | Every live dollar. `benchmarks/cost_truth` deliberately ships with **no `live` subcommand** yet ("the live driver lands only after the protocol is frozen and the arm specs are verified" — its own docstring). Once it does, running it against a real API is a paid action and needs the maintainer's explicit go-ahead *each time*, not a standing approval — costs drift with pricing and design changes. |

> As of this writing, `benchmarks/cost_truth` lives on branch `research/cost-truth-benchmark`
> (not yet on `main`). Confirm it has merged (`git log --oneline -- benchmarks/cost_truth`
> on `main`) before running this routine; if it hasn't, the refresh is "read the protocol,
> note what changed, wait" for that month.

## Why no workflow

This is the one routine in the set that would burn real money if a scheduling mistake
ran the *live* half unattended. Cost estimation and planning are free and could in
principle run on a cron — but the entire point of `benchmarks/cost_truth`'s design (a
pre-registered, hard-capped protocol with a synthetic $0 dry run and a spend gate before
the live driver even exists) is that no automation ever crosses from "compute a number"
to "spend a dollar" without a human in the loop that specific month. Automating the free
half on a schedule doesn't remove that need, and running it as a subagent invocation on
demand costs nothing extra. So: manual trigger, always.

## The model's job

1. **Refresh the free numbers.**
   ```bash
   python -m benchmarks.cost_truth plan       # design + schedule size, every arm's commands
   python -m benchmarks.cost_truth estimate   # writes benchmarks/results/cost_truth/cost_estimate.json
   python -m benchmarks.cost_truth power      # power table, sanity-check MDE hasn't drifted
   ```
   These use `distil.pricing` list prices and an assumed per-attempt token profile
   (`PROFILE` in `benchmarks/cost_truth/__main__.py`) — re-check that assumption still
   holds (has `distil.pricing` changed? has a competitor's published per-attempt cost
   moved, changing the cross-check?) before trusting a new `estimate()` output blindly.
2. **Diff against last month.** If `total_hard_cap_usd` or any phase's `hard_cap_usd`
   moved by more than pricing drift alone explains, say why in the plan (a design change,
   a new arm, a token-profile correction).
3. **Write the spend plan**, not the spend: the exact command that would run the live
   phase (pilot → primary → replication, per `DESIGN`), its dollar cap, and one line on
   what would trigger stopping early (the study's own truncation rule: "a cache-busting
   arm can cost up to ~2x its control... that would exhaust the cap, which stops the run
   and the study is reported as truncated, not extended" — carry that rule into the plan
   verbatim, don't soften it).
4. **Stop.** Hand the plan to the maintainer. Never invoke `claude`, never call a paid
   API, never run anything beyond `plan`/`dry-run`/`analyze`/`estimate`/`power` — those
   five subcommands are the entire free surface; there is no sixth to reach for.
