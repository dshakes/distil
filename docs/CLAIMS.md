# Claims ledger

**Rule: no number on the site without an entry in `docs/claims.json`.**

Any figure, percentage, or comparative statement on the docs site — a
benchmark result, a cost saving, a pass rate — gets one entry in
`docs/claims.json`. Two test modules enforce it every CI run.

`tests/test_site_claims.py` — ledger → page:

1. **Each entry's locator still resolves.** An entry names a `page` (or list of
   pages), an optional `anchor` (an HTML `id`), and a `snippet` (a verbatim
   substring of the page's source). Every listed page must exist, and the
   locator must still resolve on at least one of them. A claim repeated on six
   surfaces is worded differently on each, so the locator is required
   *somewhere* rather than everywhere — it still fails when the claim leaves
   the site.
2. **The total entry count is frozen.** Adding or removing a claim without
   touching the ledger fails the count check in
   `tests/test_site_claims.py::test_claim_count_is_frozen`. Bump
   `EXPECTED_ENTRY_COUNT` in that file in the *same commit* that edits
   `claims.json`, so the diff is reviewable instead of silent.

`tests/test_claims_coverage.py` — page → ledger, and ledger → artifact. Both
directions were missing until 2026-09-15, which is how `docs/llms.txt`,
`plugins/`, and one trust card on `index.html` carried numbers no entry named
while CI stayed green, and how one entry could cite a directory that held no
artifact while another printed 52.3% from a log that said 47.9%:

3. **Reverse coverage.** Every percentage and multiplier on `README.md`,
   `docs/*.html`, `docs/llms.txt` and `plugins/**/*.md` must appear in some
   entry that names that page. Sample terminal blocks, fenced code, `<script>`,
   `<style>`, `<svg>` and statistical notation (`95% CI`, `α=0.10`) are
   stripped before the scan. Anything left that is genuinely not a claim goes
   in `NON_CLAIMS` with a specific reason — never to make a real result pass.
   Pages not yet cleared carry a frozen `LEDGER_DEBT` set that may only shrink;
   a *new* number on those pages still fails.
4. **Artifacts are opened.** For every entry with an `artifact`, the path must
   exist and a directory must contain files. If the entry also carries
   `values`, each value must appear in the artifact — JSON fields are compared
   as both fractions and percentages, so a page's `36.8%` matches a stored
   `0.368`. An entry that genuinely cannot be machine-checked sets
   `"check": "manual"` and must say why in `check_reason`.

## Fields beyond the locator

- **`artifact`** — repo-relative path to the file or directory the number comes
  from. Prefer the exact file over its directory; a directory is checked for
  being non-empty, which is weaker.
- **`values`** — the number strings this entry covers, e.g.
  `["83.2%", "47.9%"]`. This is what lets one entry cover a whole table, and
  it is what the artifact check reads.
- **`check": "manual"` + `check_reason`** — for a figure that is derived (a
  ratio of two columns), computed at report time, or produced by a command
  whose output was never committed. Say which; "hard to check" is not a reason.

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
3. Cite the `artifact` it comes from and list its `values`, or set
   `"check": "manual"` with a `check_reason`.
4. Bump `EXPECTED_ENTRY_COUNT` in `tests/test_site_claims.py`.
5. Run `pytest tests/test_site_claims.py tests/test_claims_coverage.py`
   before committing.

## Why snippets, not line numbers

Line numbers drift with every unrelated edit to a page — a single inserted
paragraph shifts every claim below it. A snippet or heading anchor keeps
working regardless of what else on the page changes, and fails loudly,
pointing at the right claim, when the cited text actually moves or is
removed.
