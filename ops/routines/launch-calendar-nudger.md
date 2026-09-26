# Routine: launch-calendar nudger

| | |
|---|---|
| **Schedule** | Weekly (pair with the adoption-report routine's cadence — same day is fine, they're independent). |
| **Trigger** | Manual, or run alongside the weekly adoption report. |
| **Inputs** | `launch/2026-10/PLAN.md` if it exists locally (it's gitignored — `launch/` is listed in `.gitignore` as "local launch/marketing drafts (not public)" — so this file may simply not be present on a given machine or in CI); otherwise nothing to read, and the routine says so rather than fabricating a calendar. |
| **Outputs** | A short "what's due this week" note: items from the plan whose date falls in the next 7 days, plus anything already overdue. |
| **Approval point** | This routine only reads and reminds. It never drafts launch copy, never posts anywhere, never edits the plan file — a launch calendar is inherently forward-facing marketing content and stays entirely in the maintainer's hands. |

## Why "otherwise a repo copy... without drafts" wasn't built

The task description offered an alternative: keep a repo-tracked copy of the calendar
with drafts stripped, for when `launch/2026-10/PLAN.md` isn't present. That means
maintaining a second file in sync with a gitignored one, by hand, forever — the kind of
duplication this whole program exists to avoid. `launch/` is gitignored on purpose (no
public repo should carry pre-announcement drafts), and CI has no access to it, so a
repo-tracked shadow copy would only ever serve the *local* case where the real file is
already sitting right there. **Skipped**: if a repo-tracked stripped calendar is wanted
later, add a `launch/CALENDAR.md` (dates + item names only, no draft copy) that the
maintainer updates by hand alongside `PLAN.md` — a script can't tell "which parts are
drafts" without a schema `PLAN.md` doesn't currently have.

## The model's job

1. Check for `launch/2026-10/PLAN.md` (and any sibling `launch/<month>/PLAN.md` —
   glob `launch/*/PLAN.md`, since the routine outlives any one month).
   ```bash
   ls launch/*/PLAN.md 2>/dev/null
   ```
2. **Not present:** say so plainly — "no local launch plan found; nothing to nudge
   about this week" — and stop. Do not guess dates, do not invent a calendar.
3. **Present:** parse the dated items (whatever structure `PLAN.md` actually uses —
   read it fresh each run rather than assuming a fixed schema, since it's
   maintainer-authored and free to change shape). Report:
   - **Overdue**: items dated before today with no marked-done status.
   - **Due this week**: items dated within the next 7 days.
   - Nothing more — no summary of future items, no restating the whole plan back.
4. Hand the note back in the response. Never write it into the (gitignored) `launch/`
   tree, and never surface plan *content* (copy drafts, embargo details) anywhere
   outside this direct response — the note is for the maintainer's eyes in this
   conversation, not a file that could end up committed or shared by accident.
