# Adoption telemetry — spec & phases

Goal: know (a) installs by version, (b) active usage, (c) community tokens/$
saved — without breaking the "nothing leaves your machine" promise.
Decision record: `docs/adr/0002-adoption-telemetry-opt-in-census.md`.

## What we can already know (no code)

| Signal | Source | 2026-07-19 baseline |
|---|---|---|
| Downloads/month | pypistats.org API | 11,943 (1,632/wk) |
| Downloads by version | pepy.tech / PyPI BigQuery | (site/API) |
| Unique cloners /14d | GitHub traffic API | 335 (2,610 clones) |
| Stars/forks | GitHub API | 5 / 1 |
| Brew installs | tap has NO public analytics (homebrew-core only) | n/a |
| Docker pulls | Docker Hub API (if image published) | n/a |

GitHub traffic is a rolling 14-day window — history is lost unless snapshotted.

## Phases

**Phase 0 — passive adoption pipeline (zero client code).**
`scripts/adoption_snapshot.py` pulls pypistats recent+system, GitHub
repo/traffic (clones, views), Docker Hub pulls; emits one JSON line.
`.github/workflows/adoption-stats.yml` runs it nightly and appends to
`data/adoption.jsonl` on the `metrics` branch (main stays clean).
Acceptance: two consecutive scheduled runs append rows; script runs locally
with `GITHUB_TOKEN` and degrades per-source (a failing source records an
`error` field, never kills the row).

**Phase 1 — opt-in census client (this branch).**
`distil/census.py`: consent state (`~/.distil/census`), install id
(`~/.distil/install-id`, minted on consent, deleted on `off`), payload
builder (version, os, arch, python, days_active, tokens/dollars saved totals
from `ledger.summary()`), sender (urllib POST, 1.5s, fail-silent), 24h
throttle (`~/.distil/census-last`). Hard gates in one place:
`DO_NOT_TRACK=1` / `DISTIL_NO_TELEMETRY=1` > stored consent > default OFF.
CLI: `distil census on|off|status|show` (`show` prints the exact JSON, sends
nothing). Onboard asks once (interactive only, default No). Ping fires from
the wrap/proxy shutdown flush. `TELEMETRY.md` documents schema + gates;
README/THREAT_MODEL get the honest qualifier.
Acceptance: unit tests prove (1) no consent → no network even with endpoint
set, (2) DNT beats stored consent, (3) payload contains only allowed keys —
schema-frozen test, (4) `off` deletes install-id, (5) throttle honored,
(6) send failure never raises. Full suite + gates PASS.

**Phase 2 — ingest + public aggregates (needs a deploy decision).**
Tiny ingest service (Cloudflare Worker or equivalent, code in
`packaging/census-worker/`): validates schema, rate-limits, discards IPs,
upserts by install id, publishes rolled-up JSON (installs by version, actives
by week, Σ tokens/$ saved) to a public URL. Docs site "adoption" panel reads
it. Deploy is a human step (infra creds); until then census sends fail open.

**Phase 3 — community savings board.**
Wire the existing signed `SavingsAggregate` (`distil/telemetry.py`) through
the same endpoint (`distil federated-leaderboard --submit`), render the
Headroom-style fleet counter ("N tokens saved across M instances") on the
site — opt-in only.

**Phase 4 — later.**
Version-check update notifier on CLI start (24h throttle, benefit-first,
`DISTIL_NO_UPDATE_CHECK`), doubling as census transport for opted-in users;
OTel exporter for enterprise fleets.

## Non-goals

- No content, prompts, hashes-of-content, file paths, or key material — ever.
- No opt-out phone-home. No scarf-js-style post-install ping.
- No per-request events; the census is one small JSON per day maximum.
