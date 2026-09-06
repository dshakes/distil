# Claims ledger

**Rule: no number on the site without an entry in `docs/claims.json`.**

Any figure, percentage, or comparative statement on the docs site — a
benchmark result, a cost saving, a pass rate — gets one entry in
`docs/claims.json`. `tests/test_site_claims.py` enforces two things every CI
run:

1. **Each entry's locator still matches the live page.** An entry names a
   `page` (or list of pages), an optional `anchor` (an HTML `id`), and a
   `snippet` (a verbatim substring of the page's source). If the page changes
   and the snippet no longer appears, the test fails — the claim moved,
   changed, or was quietly deleted, and the ledger is now lying.
2. **The total entry count is frozen.** Adding or removing a claim without
   touching the ledger fails the count check in
   `tests/test_site_claims.py::test_claim_count_is_frozen`. Bump
   `EXPECTED_ENTRY_COUNT` in that file in the *same commit* that edits
   `claims.json`, so the diff is reviewable instead of silent.

## Statuses

- **`verified`** — backed by a named test, script, or report, and currently
  accurate. The `note` field says what backs it.
- **`stale`** — backed, but the underlying run predates a recent change and
  needs a re-run. Not wrong, just dated; the `note` says what changed
  underneath it.
- **`wrong`** — currently on the page and currently incorrect. The `note`
  explains why and whether a fix is tracked. A `wrong` entry does **not**
  authorize editing the number as part of an unrelated PR — the value itself
  changes in a dedicated, reviewed fix, not as a side effect of ledger
  bookkeeping.
- **`unsourced`** — a number with no named artifact backing it on the page
  itself (no linked benchmark script, no named corpus, no reproduction
  command). Not necessarily false — often it's real but under-cited —  but a
  reader has no way to check it without the ledger entry.

## Adding a claim

1. Find (or add) the number on the page.
2. Add an entry to `docs/claims.json` with a stable `anchor` (prefer an
   existing heading `id`) or a `snippet` that is unlikely to be touched by
   unrelated edits — a distinctive phrase next to the number, not just the
   number alone if the number recurs elsewhere on the page.
3. Bump `EXPECTED_ENTRY_COUNT` in `tests/test_site_claims.py`.
4. Run `pytest tests/test_site_claims.py` before committing.

## Why snippets, not line numbers

Line numbers drift with every unrelated edit to a page — a single inserted
paragraph shifts every claim below it. A snippet or heading anchor keeps
working regardless of what else on the page changes, and fails loudly,
pointing at the right claim, when the cited text actually moves or is
removed.
