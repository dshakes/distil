#!/bin/sh
# Install distil (https://github.com/dshakes/distil) as an isolated uv tool.
#
#   curl -LsSf https://dshakes.github.io/distil/install.sh | sh
#   curl -LsSf https://dshakes.github.io/distil/install.sh | DISTIL_VERSION=1.54.0 sh
#
# What it does, and nothing else:
#   1. if `uv` is missing, runs Astral's official uv installer (https://astral.sh/uv/install.sh);
#      that installer is the only thing that may add a line to your shell profile
#   2. `uv tool install --upgrade distil-llm` (or the pinned DISTIL_VERSION)
#   3. prints the next step: `distil setup`
# No sudo. Safe to re-run: a second run upgrades in place.
set -eu

usage() {
	cat <<'EOF'
Install distil as an isolated uv tool.

Usage: install.sh [--help]

Environment:
  DISTIL_VERSION   install exactly this version (e.g. 1.54.0); default: latest

Installs uv first if it is missing, using Astral's official installer.
Re-running upgrades in place. Next step after install: distil setup
EOF
}

die() {
	printf 'distil install: error: %s\n' "$*" >&2
	exit 1
}

case "${1:-}" in
-h | --help)
	usage
	exit 0
	;;
"") ;;
*)
	usage >&2
	die "unknown argument: $1"
	;;
esac

spec=distil-llm
if [ -n "${DISTIL_VERSION:-}" ]; then
	case "$DISTIL_VERSION" in
	*[!0-9A-Za-z.+-]*) die "DISTIL_VERSION must look like 1.54.0, got: $DISTIL_VERSION" ;;
	esac
	spec="distil-llm==$DISTIL_VERSION"
fi

if ! command -v uv >/dev/null 2>&1; then
	command -v curl >/dev/null 2>&1 ||
		die "uv is not installed and curl is missing; install uv by hand: https://docs.astral.sh/uv/getting-started/installation/"
	echo "distil install: uv not found; installing it with Astral's official installer"
	curl -LsSf https://astral.sh/uv/install.sh | sh || die "the uv installer failed (see its output above)"
	# The installer edits your profile for future shells; this one needs the path now.
	PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$HOME/.cargo/bin:$PATH"
	command -v uv >/dev/null 2>&1 || die "uv was installed but is not on PATH; open a new shell and re-run"
fi

echo "distil install: uv tool install --upgrade $spec"
uv tool install --upgrade "$spec" || die "uv tool install $spec failed (see uv's output above)"

if command -v distil >/dev/null 2>&1; then
	echo
	echo "distil is installed. Next step:"
	echo "  distil setup"
else
	echo
	echo "distil is installed, but uv's tool directory is not on your PATH yet."
	echo "Run 'uv tool update-shell', open a new shell, then:"
	echo "  distil setup"
fi
