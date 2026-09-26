# Agentic operating routines

Distil's own maintenance work — adoption reporting, issue triage, docs drift, cost
refresh, releases, launch reminders — run as **scripts that do the deterministic part**
and **model prompts that do the judgment part**, never the reverse. Every routine below
produces a draft; nothing here posts, merges, releases, or spends without the
maintainer saying go.

## What runs when

| Routine | Schedule | Trigger | Reads | Writes | Approval point |
|---|---|---|---|---|---|
| [Weekly adoption report](../ops/routines/weekly-adoption-report.md) | Weekly (Mon) | `ops-adoption-report.yml` cron / manual | `gh api` (read-only), pypistats | Workflow artifact (md+json) + a prose note | None to read; a recommended action is a suggestion |
| [Issue triage](../ops/routines/issue-triage.md) | On new external issues | Manual/subagent | Issue body, `docs/claims.json`, source | Draft label/repro-plan/reply/fix-plan | Applying the label, posting the reply, opening the branch |
| [Claims/docs drift check](../ops/routines/claims-docs-drift-check.md) | Every relevant PR + pre-release | CI already runs the tests; routine runs on demand | `tests/test_claims_coverage.py`, `test_site_claims.py`, `test_docs_commands_parse.py` | Draft PR plan on failure | Opening the PR |
| [Monthly cost-truth refresh](../ops/routines/monthly-cost-truth-refresh.md) | Monthly | Manual only | `benchmarks/cost_truth` free subcommands | Updated estimate + a spend plan | Every live dollar, every month |
| [Two-week release train](../ops/routines/two-week-release-train.md) | Every 2 weeks | Manual/calendar | `CHANGELOG.md`, 7 version files, packaging tests | Draft release PR (branch + diff) | Merging the PR; tagging (`scripts/release.sh`) |
| [Launch-calendar nudger](../ops/routines/launch-calendar-nudger.md) | Weekly | Manual | `launch/*/PLAN.md` (local, gitignored) if present | A "due this week" note in-chat | N/A (read-only reminder) |

## What this may never do

- **Never post, comment, or open an issue/PR anywhere.** Every routine above stops at a
  draft. `gh` is used read-only throughout (`gh api`, `gh issue view`, `gh pr view`) —
  no routine calls `gh issue comment`, `gh pr create`, `gh pr merge`, or
  `gh issue edit --add-label`.
- **Never merge or tag a release.** `scripts/release.sh` (existing, unchanged) is the
  only thing that pushes `main` and pushes a `v*` tag, and it runs interactively with
  its own confirmation prompts — no routine here invokes it non-interactively.
- **Never spend money without approval, each time.** The cost-truth routine refreshes
  free estimates monthly; the live driver (once it exists) is a separate, explicitly
  approved action, not a standing schedule.
- **Never call `claude` or any paid API** from inside a routine or its workflow. The one
  GitHub Actions workflow this program adds (`ops-adoption-report.yml`) calls only
  `gh api` (read) and `pypistats.org` (unauthenticated, free) and uploads an artifact —
  no secrets beyond the optional pre-existing `TRAFFIC_TOKEN`.
- **Never widen write scope to get a convenience.** Issue triage doesn't add
  `issues: write` to a workflow just to auto-label; the release train doesn't add a
  service account that could merge its own PR. If a routine's draft output would be
  more useful posted automatically, that's a deliberate scope change for the
  maintainer to decide, not a default this program takes for itself.

## How the maintainer approves

Every routine's output lands in one of two places:

1. **A workflow artifact** (adoption report) — read it, or hand it to a subagent to
   summarize; nothing downstream happens until you ask for a next step.
2. **A chat response or a local, uncommitted branch/diff** (everything else) — review
   the diff, and if it's a PR-shaped output, run `gh pr create` yourself (or ask an
   agent to, in that turn, as an explicit instruction — not as this routine's default).

None of these routines chain into each other automatically. The claims/docs check
doesn't auto-open a PR because the release train ran; the release train doesn't
auto-tag because CI went green. Each stop is a place a human looks before the next
routine (or the same one, next cycle) picks the thread back up.

## Existing workflows this program reuses (not duplicates)

- **`adoption-stats.yml`** — nightly registry snapshot to the `metrics` branch (history
  only; GitHub's traffic API keeps just 14 days). The weekly adoption report is a
  *different* cadence and a *different* audience (a human-readable digest with deltas
  and maintainer-activity filtered out) — see `ops-adoption-report.yml`'s header comment
  for why it's a separate workflow rather than bolted onto the nightly cron.
- **`census-ingest.yml`** — the opt-in census rollup (`specs/adoption-telemetry.md`).
  Deliberately untouched and unreferenced by anything here: census consent is not the
  same as census contribution, and the community total has historically been dominated
  by very few contributing machines. Conflating that opt-in, content-free total with
  *external* adoption signals (stars, forks, traffic, downloads, external issues/PRs —
  none of which need consent because none of them run code on a user's machine) is
  exactly the mistake this program's adoption report is designed not to repeat (see
  `scripts/ops/adoption_report.py`'s docstring).
- **`sdlc-qa.yml`** — the PR test gate; `tests/test_claims_coverage.py`,
  `test_site_claims.py`, and `test_docs_commands_parse.py` already run under it on every
  PR. The claims/docs-drift routine is the *prompt* for reading a failure and drafting
  the fix, not a new check.
