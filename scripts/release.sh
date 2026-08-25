#!/usr/bin/env bash
# Distil release driver — push, tag, and let CI publish; refresh the Homebrew sha256.
#
# PUBLISHING MODEL (industry standard): this repo publishes to PyPI via
# .github/workflows/release.yml using PyPA Trusted Publishing (OIDC) — there is NO
# API token anywhere. You push a v* tag; CI gates, builds, attaches GitHub release
# artifacts, and (when enabled) publishes to PyPI. So this script's job is to push
# main + the tag; CI does the upload. No twine, no laptop token.
#
# ONE-TIME SETUP on pypi.org (per the footer of release.yml), then never again:
#   1. pypi.org → distil-llm → Publishing → add a Trusted Publisher:
#        owner dshakes · repo distil · workflow release.yml · environment pypi
#   2. GitHub → repo → Settings → Variables → set  PUBLISH_TO_PYPI = true
#      (until set, a tag still ships GitHub artifacts but skips the PyPI upload.)
#
# Run from the repo root on a clean `main`:
#   ./scripts/release.sh
#
# Prereqs: uv on PATH (build/smoke-test) · push rights to origin · gh CLI (optional,
# to watch the release run). No twine and no PyPI token required.
#
# Flags:
#   --dry-run        print every action, change nothing (no push, no tag)
#   --skip-tests     skip the pytest gate (NOT recommended)
#   --local-publish  ALSO upload from this machine via `uv publish` (escape hatch for
#                    when Trusted Publishing isn't set up; needs UV_PUBLISH_TOKEN).
#                    Leave OFF if CI publishes — double-upload of a version fails.
set -euo pipefail

DRY_RUN=0
SKIP_TESTS=0
LOCAL_PUBLISH=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --local-publish) LOCAL_PUBLISH=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

# ---- pretty + safety helpers -------------------------------------------------
bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '\033[36m›\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
die()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

run() {  # echo + execute, or just echo under --dry-run
  printf '\033[90m$ %s\033[0m\n' "$*"
  [ "$DRY_RUN" -eq 1 ] && return 0
  eval "$*"
}

confirm() {  # confirm "message" ; aborts the step on anything but y/Y
  [ "$DRY_RUN" -eq 1 ] && { info "[dry-run] would ask: $1"; return 1; }
  printf '\033[33m? %s [y/N] \033[0m' "$1"
  read -r reply
  [[ "$reply" =~ ^[yY]$ ]]
}

# ---- derive version from the source of truth ---------------------------------
VERSION="$(grep -E '^version *= *"' pyproject.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
[ -n "$VERSION" ] || die "could not read version from pyproject.toml"
TAG="v$VERSION"
TARBALL_URL="https://github.com/dshakes/distil/archive/refs/tags/${TAG}.tar.gz"

# rc = soak candidate (see RELEASING.md): GitHub release is marked prerelease by
# CI, no Homebrew bump, no Docker image. Promote by re-tagging the same commit
# with the final version after the soak window.
IS_RC=0
case "$VERSION" in *rc*) IS_RC=1 ;; esac

bold "Distil release — ${TAG}$([ "$DRY_RUN" -eq 1 ] && echo '  (dry-run)')"
echo

# ---- preflight ---------------------------------------------------------------
bold "1/6  Preflight"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || die "not on main (on '$BRANCH')"
ok "on main"

[ -z "$(git status --porcelain)" ] || die "working tree is dirty — commit or stash first"
ok "working tree clean"

# version agreement across every surface that ships a version string
# (distil/__init__.py is single-sourced from pyproject since 5e1173b — no literal to check)
CITE_V="$(grep -E '^version:' CITATION.cff | sed -E 's/.*: *([0-9][^ ]*).*/\1/')"
# npm and the Claude plugin manifest use SemVer, so an rc is `X.Y.Z-rc.N` there
# while PEP 440 surfaces use `X.Y.ZrcN`. Derive both spellings once.
SEMVER_V="$(printf '%s' "$VERSION" | sed -E 's/rc([0-9]+)$/-rc.\1/')"

if [ "$IS_RC" -eq 1 ]; then
  info "rc release — CITATION.cff stays at the last citable final ($CITE_V)"
  # Everything EXCEPT CITATION still gets checked. Skipping these on rcs recreated
  # the very blind spot the checks were added for: plugin.json sat at 1.8.6 for 22
  # releases because nothing checked it, and an unchecked rc is how it starts again.
  PLUGIN_V="$(grep -E '^\s*"version":' plugins/distil/.claude-plugin/plugin.json | head -1 | sed -E 's/.*"version": *"([^"]+)".*/\1/')"
  [ "$PLUGIN_V" = "$SEMVER_V" ] || die "plugin.json is $PLUGIN_V, expected $SEMVER_V (SemVer spelling of $VERSION)"
  SERVER_V="$(grep -E '^\s*"version":' server.json | head -1 | sed -E 's/.*"version": *"([^"]+)".*/\1/')"
  [ "$SERVER_V" = "$VERSION" ] || die "server.json is $SERVER_V, pyproject is $VERSION"
  NPM_V="$(grep -E '^\s*"version":' packaging/npm/package.json | head -1 | sed -E 's/.*"version": *"([^"]+)".*/\1/')"
  [ "$NPM_V" = "$SEMVER_V" ] || die "packaging/npm/package.json is $NPM_V, expected $SEMVER_V"
  # release.yml skips the container build for rc tags, so an rc appVersion would
  # point a default `helm install` at an image that is never published.
  CHART_V="$(grep -E '^appVersion:' packaging/helm/distil-gateway/Chart.yaml | sed -E 's/.*: *"?([^"]+)"?.*/\1/')"
  case "$CHART_V" in
    *rc*) die "Chart.yaml appVersion is $CHART_V — an rc image is never built; pin it to the last GA" ;;
  esac
  ok "rc versions consistent (plugin.json, server.json, npm as $SEMVER_V; chart pinned to $CHART_V)"
else
  [ "$CITE_V" = "$VERSION" ] || die "CITATION.cff is $CITE_V, pyproject is $VERSION"
  # The Claude Code plugin manifest is a shipped surface too — it sat at 1.8.6 for
  # 22 releases because nothing checked it. server.json likewise feeds the MCP
  # registry listing.
  PLUGIN_V="$(grep -E '^\s*"version":' plugins/distil/.claude-plugin/plugin.json | head -1 | sed -E 's/.*"version": *"([^"]+)".*/\1/')"
  [ "$PLUGIN_V" = "$VERSION" ] || die "plugin.json is $PLUGIN_V, pyproject is $VERSION"
  SERVER_V="$(grep -E '^\s*"version":' server.json | head -1 | sed -E 's/.*"version": *"([^"]+)".*/\1/')"
  [ "$SERVER_V" = "$VERSION" ] || die "server.json is $SERVER_V, pyproject is $VERSION"
  # The npm wrapper carries a version badge in the README and nothing checked it:
  # it sat at 1.27.0 in-repo while the registry served 1.25.0 and PyPI was on
  # 1.33.1 — a badge advertising a build nine releases old.
  NPM_V="$(grep -E '^\s*"version":' packaging/npm/package.json | head -1 | sed -E 's/.*"version": *"([^"]+)".*/\1/')"
  [ "$NPM_V" = "$VERSION" ] || die "packaging/npm/package.json is $NPM_V, pyproject is $VERSION"
  ok "version $VERSION consistent (pyproject, CITATION, plugin.json, server.json, npm)"
fi

git rev-parse "$TAG" >/dev/null 2>&1 && die "tag $TAG already exists — bump the version or delete the tag"
ok "tag $TAG is free"

if [ "$IS_RC" -eq 1 ]; then
  # rc: the changelog entry is written once, under the final version it soaks for
  FINAL="${VERSION%%rc*}"
  grep -q "## \[$FINAL\]\|## \[$VERSION\]" CHANGELOG.md \
    || die "CHANGELOG.md has no '## [$FINAL]' entry (rc soaks for the final)"
  ok "CHANGELOG has the [$FINAL] entry this rc soaks for"
else
  grep -q "## \[$VERSION\]" CHANGELOG.md || die "CHANGELOG.md has no '## [$VERSION]' entry"
  ok "CHANGELOG has a [$VERSION] entry"
fi

if [ "$SKIP_TESTS" -eq 1 ]; then
  info "skipping tests (--skip-tests)"
else
  # Check the venv can actually RUN the suite before blaming the code. Without
  # this, a venv with no pytest reports "tests failed — not releasing", which
  # sends you hunting through a green codebase for a phantom regression.
  if ! .venv/bin/python -c "import pytest" >/dev/null 2>&1; then
    die ".venv has no pytest — the test gate cannot run (see RELEASING.md, 'the .venv the preflight tests run in')"
  fi
  # The optional extras matter MORE than pytest itself, because missing them fails
  # silently: those tests importorskip themselves and the gate goes green having
  # covered less than CI. Cutting 1.47.0rc1 read 6 skipped where CI reads 2 — four
  # tests, at-rest crypto among them, never ran.
  missing=""
  for mod in cryptography opentelemetry PIL aiohttp; do
    .venv/bin/python -c "import $mod" >/dev/null 2>&1 || missing="$missing $mod"
  done
  [ -n "$missing" ] && die ".venv is missing:$missing — those tests would silently skip, making this gate weaker than CI (see RELEASING.md)"

  info "running test suite…"
  run ".venv/bin/python -m pytest -q" || die "tests failed — not releasing"
  ok "tests pass"
fi

# clean-install smoke test: build the wheel, install it into a throwaway venv with
# no index, and confirm the entrypoint + bundled corpus actually work.
#
# DISTIL_HOME points at the throwaway too. A temp venv is not a machine: if it
# touches the real ~/.distil it inherits this machine's install id and consent,
# and anything it reports lands on the public adoption board as though someone
# were running that version. It also keeps a smoke-test bench off the real
# savings ledger.
info "clean-install smoke test…"
SMOKE="$(mktemp -d)"
trap 'rm -rf "$SMOKE"' EXIT
run "uv build --wheel -o '$SMOKE/wheel'"
if [ "$DRY_RUN" -eq 0 ]; then
  WHEEL="$(ls "$SMOKE"/wheel/distil_llm-"$VERSION"-py3-none-any.whl)"
  python3 -m venv "$SMOKE/venv"
  "$SMOKE/venv/bin/pip" install -q --no-index "$WHEEL"
  export DISTIL_HOME="$SMOKE/home"
  GOT="$("$SMOKE/venv/bin/distil" --version | awk '{print $2}')"
  [ "$GOT" = "$VERSION" ] || die "clean install reports $GOT, expected $VERSION"
  "$SMOKE/venv/bin/distil" bench >/dev/null || die "distil bench failed on a clean install"
  unset DISTIL_HOME
  ok "clean install works: distil $VERSION, bench gate runs"
fi
echo

# ---- push main ---------------------------------------------------------------
bold "2/6  Push main"
if confirm "push 'main' to origin?"; then
  run "git push origin main"
  ok "main pushed"
else
  info "skipped push"
fi
echo

# ---- tag (this is what triggers the release CI) ------------------------------
bold "3/6  Tag $TAG  →  triggers release.yml (gate, GitHub artifacts, PyPI publish)"
if confirm "create and push annotated tag $TAG?"; then
  run "git tag -a '$TAG' -m 'Distil $TAG'"
  run "git push origin '$TAG'"
  ok "$TAG pushed — CI is now building + publishing"
else
  info "skipped tag — the PyPI publish and Homebrew sha256 below need the tag; re-run later"
fi
echo

# ---- PyPI publish ------------------------------------------------------------
bold "4/6  PyPI publish (distil-llm $VERSION)"
if [ "$LOCAL_PUBLISH" -eq 1 ]; then
  info "--local-publish set: uploading from this machine via uv publish"
  if confirm "build + uv publish $VERSION now? (irreversible; do NOT also let CI publish the same version)"; then
    command -v uv >/dev/null 2>&1 || die "uv not found"
    [ -n "${UV_PUBLISH_TOKEN:-}" ] || die "set UV_PUBLISH_TOKEN=pypi-… (token from pypi.org)"
    run "rm -rf dist/distil_llm-$VERSION*"
    run "uv build -o dist"
    run "uv publish dist/distil_llm-$VERSION*"
    ok "uploaded distil-llm $VERSION to PyPI"
  else
    info "skipped local publish"
  fi
else
  info "Trusted Publishing: the release.yml 'publish-pypi' job uploads via OIDC — no token here."
  info "It runs only if repo variable PUBLISH_TO_PYPI=true and the PyPI pending publisher exists."
  if command -v gh >/dev/null 2>&1 && [ "$DRY_RUN" -eq 0 ]; then
    info "watch it:  gh run watch --repo dshakes/distil \$(gh run list --repo dshakes/distil -w release -L1 --json databaseId -q '.[0].databaseId')"
    info "verify after:  pip index versions distil-llm   # or check https://pypi.org/p/distil-llm"
  else
    info "watch the run at https://github.com/dshakes/distil/actions/workflows/release.yml"
  fi
fi
echo

# ---- refresh Homebrew sha256 -------------------------------------------------
bold "5/6  Refresh Homebrew formula sha256"
FORMULA="packaging/homebrew/distil.rb"
if [ "$IS_RC" -eq 1 ]; then
  info "rc release — skipping Homebrew (brew serves finals only)"
elif confirm "fetch $TAG tarball and patch the sha256 in $FORMULA?"; then
  if [ "$DRY_RUN" -eq 1 ]; then
    info "[dry-run] would curl $TARBALL_URL and shasum it"
  else
    info "fetching $TARBALL_URL …"
    SHA="$(curl -fsSL "$TARBALL_URL" | shasum -a 256 | awk '{print $1}')" \
      || die "could not fetch the tag tarball — is $TAG pushed yet?"
    [ "${#SHA}" -eq 64 ] || die "unexpected sha256: $SHA"
    # update sha + url + version in the formula — and the comment that says which
    # tag the sha belongs to. Leaving that comment unpatched let it drift 18
    # releases behind the sha it describes, which is worse than no comment.
    sed -i.bak -E \
      -e "s|sha256 \"[^\"]*\"|sha256 \"$SHA\"|" \
      -e "s|/tags/v[0-9][^\"]*\.tar\.gz|/tags/${TAG}.tar.gz|" \
      -e "s|version \"[0-9][^\"]*\"|version \"$VERSION\"|" \
      -e "s|(# sha256 is for the )v[0-9][^ ]*( source tarball)|\1${TAG}\2|" \
      "$FORMULA"
    rm -f "$FORMULA.bak"
    ok "patched $FORMULA → sha256 $SHA"
    if confirm "commit + push the formula bump?"; then
      run "git add '$FORMULA'"
      run "git commit -m 'release($VERSION): pin Homebrew sha256 to $TAG tarball'"
      run "git push origin main"
      ok "formula bump pushed"
    fi
  fi
else
  info "skipped Homebrew refresh"
fi
echo

# ---- done --------------------------------------------------------------------
bold "6/6  Done — manual follow-ups"
cat <<EOF
  • Homebrew TAP: this repo's $FORMULA is the source copy. Copy it into the actual
    tap repo so 'brew install dshakes/tap/distil' serves $VERSION:
        dshakes/homebrew-tap → Formula/distil.rb
  • GitHub release notes: paste the [$VERSION] section of CHANGELOG.md into
        https://github.com/dshakes/distil/releases/new?tag=$TAG
  • Verify the live install:  pipx install distil-llm && distil --version   # → $VERSION
  • Docs site redeploys from main via the pages workflow — confirm it's green.
EOF
ok "release driver finished"
