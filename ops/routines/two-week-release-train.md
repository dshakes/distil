# Routine: two-week release train

| | |
|---|---|
| **Schedule** | Every two weeks (matches the cadence the recent tags show — 1.52.0 → 1.53.0rc1 → 1.54.0). |
| **Trigger** | Manual, or a calendar reminder two weeks after the last GA tag: `git tag --sort=-creatordate \| head -1`. |
| **Inputs** | `CHANGELOG.md`'s `[Unreleased]` section, `pyproject.toml`'s current `version`, the 7 version-location files (below), `tests/test_packaging_smoke.py`, `tests/test_claims_coverage.py` / `test_site_claims.py` / `test_docs_commands_parse.py`. |
| **Outputs** | A **draft release PR**: version bump across all 7 locations, `CHANGELOG.md`'s `[Unreleased]` section moved into a dated `[x.y.z]` entry, and the docs-check results attached. |
| **Approval point** | Everything past "open the PR" is the maintainer's: reviewing, merging, and — per `scripts/release.sh`'s own model — tagging (`git push` of the `v*` tag is what triggers CI's build+publish). This routine never runs `git push`, never runs `scripts/release.sh` (which pushes to `main` directly under `--dry-run`-off), and never creates the tag. |

## The 7 version locations (verified against `tests/test_packaging_smoke.py`'s guards)

1. `pyproject.toml` — `version = "..."`
2. `CITATION.cff` — `version:`
3. `server.json` — top-level `"version"`
4. `server.json` — the nested package block's `"version"` (two hits in one file, easy
   to bump one and miss the other — `test_packaging_smoke.py` guards both)
5. `packaging/npm/package.json` — `"version"`
6. `plugins/distil/.claude-plugin/plugin.json` — `"version"`
7. `packaging/helm/distil-gateway/Chart.yaml` — `appVersion` (NOT `version:`, which
   tracks the chart's own version, not the distil release it defaults to — this is the
   one that drifted silently through all of 1.42.0)

`homebrew-tap`'s `distil.rb` deliberately lags — the tap is the canonical copy, this
repo's copy is a staging artifact `scripts/release.sh` patches *after* the tag exists
(step 5/6 of that script, sha256-pinned to the pushed tag's tarball) — do not bump it
here; there is nothing to bump yet before the tag exists.

## The model's job

1. **Confirm there's something to ship.** `git log --oneline v<last>..HEAD -- distil/
   scripts/ docs/` — if `[Unreleased]` in `CHANGELOG.md` is empty and there's no
   unreleased commit of substance, the honest output is "nothing to ship this cycle,"
   not a version bump for its own sake.
2. **Pick the version.** SemVer, keyed off the actual changes in `[Unreleased]`
   (breaking → major, feature → minor, fix-only → patch) — the last several releases
   have all bumped minor even for fix-heavy cycles (project convention favors minor
   over patch for a public tool), so default to minor unless the diff is a pure hotfix.
3. **Bump all 7 locations** to the new version, in one diff.
4. **Move `[Unreleased]` → `[x.y.z] — YYYY-MM-DD — <short theme>`** in `CHANGELOG.md`,
   leaving a fresh empty `## [Unreleased]` above it. Match the existing voice (see the
   last few entries — plain declarative sentences, "Added/Fixed/Research" style
   sub-bullets, no marketing adjectives).
5. **Run the docs checks** (delegate to `ops/routines/claims-docs-drift-check.md`'s
   three tests) and `python3 scripts/build_agent_tables.py --check` — a version bump
   that also changed a preset or a doc'd command must not ship with a stale table.
   `uvx ruff@0.15.10 check . && uvx ruff@0.15.10 format --check .` and
   `uv run --with pytest --with pillow python -m pytest tests/test_packaging_smoke.py -q`
   are the release-blocking gates; paste their actual output into the PR plan, not a
   summary.
6. **Draft the PR** (title `release: x.y.z — <theme>`, body = the new CHANGELOG
   section) as a local branch + diff. Do not push it, do not open it with `gh pr
   create` — hand the branch name and diff to the maintainer.
7. **Stop.** The maintainer reviews, merges, then runs `scripts/release.sh` themselves
   (which pushes `main` + the tag and lets CI publish) — that script already exists and
   already does exactly this; this routine's job ends at "PR ready to review."
