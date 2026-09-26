"""Cost-truth benchmark: what context compressors save on a real, cache-inclusive bill.

Pre-registration: ``docs/research/cost-truth-protocol.md``. ADR 0018.

Modules:

* ``meter``    — the neutral usage-metering reverse proxy + hard spend cap (every arm's
                 traffic crosses it last, just before the provider).
* ``arms``     — the four arms: pinned installs, documented integration, claim command.
* ``runner``   — paired, randomised schedule; per-run isolation; content-free results.
* ``mock``     — offline upstream + scripted agent for ``dry-run`` (no network, $0).
* ``analysis`` — $/solved ratio (paired cluster bootstrap), success
                 non-inferiority, analytic power, live-run cost estimate.

CLI: ``python -m benchmarks.cost_truth {plan,dry-run,analyze,estimate,power}``.
"""
