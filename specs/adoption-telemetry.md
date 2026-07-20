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

**Phase 2 — ingest + public aggregates. IMPLEMENTED (deploy = human step).**
`packaging/census-worker/` (Vercel function, zero-dep): strict validation
(1 KB cap, key allowlist, numeric ceilings), stores nothing, forwards to
`repository_dispatch("census")`. `census-ingest.yml` RE-validates
(`scripts/census_validate.py`, defense in depth) and appends to
`data/census.jsonl` on the `metrics` branch — the datastore is a public git
branch, so the whole pipeline is auditable. `scripts/census_rollup.py`
(nightly, inside adoption-stats.yml) dedupes latest-per-install-id, drops
out-of-ceiling rows, emits `data/aggregates.json` + shields endpoint badges.
Remaining human step: mint the fine-grained PAT + `vercel deploy` + DNS
(runbook: packaging/census-worker/README.md). Until then, sends fail open.

**Phase 3 — community savings board. IMPLEMENTED via the census.**
Community Σ tokens/$ ride the census rollup (latest ledger totals per
install id) — README endpoint badges + the live adoption strip on
docs/index.html render them. The HMAC-signed `SavingsAggregate` federation
(`distil/telemetry.py`) remains the org-internal path (file-based, unchanged).

**Phase 4 — update notifier. IMPLEMENTED.**
`distil/updatecheck.py`: wrap/proxy check PyPI ≤1/24h in a daemon thread,
one stderr line when behind, `DISTIL_NO_UPDATE_CHECK=1` opts out, disclosed
in TELEMETRY.md. (OTel exporter for enterprise fleets stays future work.)

## Non-goals

- No content, prompts, hashes-of-content, file paths, or key material — ever.
- No opt-out phone-home. No scarf-js-style post-install ping.
- No per-request events; the census is one small JSON per day maximum.
