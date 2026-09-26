# Routine: weekly adoption report

| | |
|---|---|
| **Schedule** | Weekly, Monday. Automated half runs in CI (`.github/workflows/ops-adoption-report.yml`, Monday 03:17 UTC); the model half runs whenever the maintainer or a subagent picks up the artifact. |
| **Trigger** | `ops-adoption-report.yml` cron, or `workflow_dispatch`, or manually: `python3 scripts/ops/adoption_report.py` |
| **Inputs** | `gh api` (read-only: repo, releases, stargazers, traffic, collaborators, issues) + `pypistats.org` (no auth). No census data, no `~/.distil`. |
| **Outputs** | `adoption-report.md` + `adoption-report.json` (workflow artifact, 90-day retention) *plus* a short prose note (below) that the model writes when asked to read the artifact. |
| **Approval point** | None required to *read* the report — it is data, not an action. If the note recommends an action (open an issue, change a docs claim, adjust the launch calendar) that recommendation is a draft; the maintainer decides whether to act on it. |

## What the script measures

External-only signals, deliberately excluding what `docs/adoption.html` already shows
(census totals — opt-in, structurally dominated by one machine, see
`specs/adoption-telemetry.md`) and excluding the maintainer's own GitHub activity:

- stars (+ delta since the report window, from stargazer timestamps)
- forks
- traffic: views/uniques, clones/uniques (14-day rolling — GitHub's own window)
- top referrers (14d)
- issues/PRs opened by non-maintainer, non-bot accounts (maintainer = current repo
  collaborators; a stale collaborator list only ever *undercounts* "external", it never
  manufactures external activity that isn't there)
- PyPI downloads, without mirrors, split by OS (`Darwin`/`Windows` as a human-install
  proxy, `Linux` as a CI-install proxy — pypistats' own `null` category, unclassified
  downloads with no reported OS, is relabeled `unknown_or_bot` so it is never silently
  folded into either proxy)
- release count, total and since the window

## The model's job (judgment only)

The script does not interpret its own numbers — that is the one part worth spending a
model on. Given `adoption-report.md`/`.json` and the *previous* week's artifact (fetch
from the most recent prior `ops-adoption-report` run, or from
`data/adoption.jsonl` on the `metrics` branch if the artifact has expired):

1. **What moved and why** — 3-6 sentences, plain language. Name the single largest
   mover in each section (e.g. "clones uniques +40% — likely the #182 claims-gate post
   on HN, top referrer is news.ycombinator.com"). If a section's `error` field is set,
   say so; do not silently omit it or guess a number to fill the gap.
2. **Top 3 actions** — ranked, one line each, addressed to the maintainer. Each action
   must point at something the *external* numbers actually show (a referrer spike worth
   a follow-up post, a contributor whose issue is stale, a platform with near-zero
   downloads worth a compatibility check) — never invent an action from a flat week.
   A flat week's honest top action can be "nothing stands out; here is what to watch
   next week."

Never post this note anywhere (no issue comment, no PR, no Slack). Write it to a file
next to the artifact, or hand it back in the response — the maintainer chooses whether
and where it goes public.

## Reuse, not duplication

`scripts/ops/adoption_report.py` reuses `scripts/adoption_snapshot.py`'s HTTP retry/UA
helper (`_get`) rather than re-implementing backoff. It does **not** touch the `metrics`
branch or `adoption-stats.yml`'s nightly append — that pipeline is the raw-history
snapshot; this is the weekly human-facing digest, kept separate because the cadence and
shape differ (see the workflow file's header comment for the full reasoning).
