# Routine: claims/docs drift check

| | |
|---|---|
| **Schedule** | Every PR that touches `docs/`, `CHANGELOG.md`, `distil/cli.py`, or a public number's source data (already covered by CI on every push — see "What already runs it" below). Run by hand before a release with `make test` or the three commands directly. |
| **Trigger** | `git diff` touching one of the paths above, or on demand before `scripts/release.sh`. |
| **Inputs** | `pytest tests/test_claims_coverage.py tests/test_site_claims.py tests/test_docs_commands_parse.py -v`, plus a manual docs-vs-CLI pass for anything the automated check can't parse (prose claims about behavior, not just commands). |
| **Outputs** | A pass/fail report. On fail: a **DRAFT PR plan** (title, the exact diff, which claim/command it fixes) — never an opened PR. |
| **Approval point** | Opening the PR. This routine prepares the diff and the plan; `gh pr create` is a write action and stays with the maintainer (or a follow-up step they explicitly approve). |

## What already runs it (reuse, don't duplicate)

Three tests already exist and do the mechanical half of this job — this routine is the
prompt that runs them, reads the failure, and turns it into a plan, not a
reimplementation:

- **`tests/test_claims_coverage.py`** — every entry in `docs/claims.json` (the ledger of
  every numeric/comparative claim on the site) has a valid status and a frozen count,
  so a claim can't be silently added or removed.
- **`tests/test_site_claims.py`** — the other direction: every ledger entry still points
  at a real snippet/anchor on the page it names. Together the two catch both "a claim
  drifted off the page" and "a claim was added/removed without anyone noticing."
- **`tests/test_docs_commands_parse.py`** — every `distil …` command printed anywhere in
  the docs is parsed (never executed) against `distil.cli.build_parser()`. A docs
  runbook with a typo'd flag fails here before a user ever pastes it.

```bash
make test   # runs the full suite, including all three; or scope it:
uv run --with pytest --with pillow python -m pytest \
    tests/test_claims_coverage.py tests/test_site_claims.py tests/test_docs_commands_parse.py -q
```

## The model's job

1. Run the three tests (or `make test` if a fuller signal is wanted). Read failures
   verbatim — never paraphrase a stack trace into "looks fine."
2. For a **claims failure**: find the real current value (re-run the benchmark/script
   `docs/claims.json` cites as the claim's source — most entries name one), and draft
   the exact `docs/claims.json` + page edit that makes it agree, or flag it `stale` if
   the underlying number needs a maintainer re-measurement first (e.g. a benchmark that
   costs money to re-run — hand off to the cost-truth routine, don't re-run it here).
3. For a **docs-commands failure**: draft the one-line fix to the doc (the flag/command
   that doesn't parse), quoting `distil --help` or the relevant subcommand's `--help`
   as the source of truth — never guess a flag name.
4. Package the result as a **draft PR plan**: a title, a one-paragraph body (what
   drifted, what the fix restores), and the literal diff. Do not run `gh pr create`.
   If nothing drifted, say so — a clean run needs no plan.

## Guardrails

- This check is free, offline, and fast by design (the docstring of
  `test_docs_commands_parse.py` says so explicitly) — never let this routine spend
  money or call a live API to "double check" a claim; that's the cost-truth routine's
  job, gated separately.
- A claim's frozen `EXPECTED_ENTRY_COUNT` changing is a deliberate, reviewed edit, never
  an artifact of this routine auto-bumping a number to make a test pass.
