#!/usr/bin/env bash
# Build a single-file, dependency-free executable (distil.pyz).
# Works because the distil core is stdlib-only. The corpus is staged inside the
# archive and resolved at runtime via DISTIL_CORPUS.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="dist/distil.pyz"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp -r distil "$STAGE/distil"
cp -r corpus "$STAGE/corpus"

# Stamp the version as an importable module. The zipapp has no dist-info, and the
# pyproject fallback in distil/__init__.py can't read_text() inside an archive —
# so without this the pyz reports "0+source" (it did, through 1.33.0). zipimport
# can import, so a generated module is the one thing that survives the archive.
PY="$(command -v python3 || command -v python)"
"$PY" - "$STAGE/distil/_version.py" <<'STAMP'
import pathlib, re, sys
pp = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
v = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', pp).group(1)
pathlib.Path(sys.argv[1]).write_text(f'__version__ = "{v}"\n', encoding="utf-8")
print(f"stamped pyz version {v}")
STAMP

# __main__ so `python distil.pyz <cmd>` works; point the corpus at the archive's
# extracted sibling via a tiny launcher that sets DISTIL_CORPUS next to the pyz.
cat > "$STAGE/__main__.py" <<'PY'
import os, sys
# zipapp can't open() files inside the archive; ship corpus alongside the pyz and
# point DISTIL_CORPUS at the directory that *contains* the .pyz file.
archive = os.path.dirname(os.path.abspath(__file__))   # .../distil.pyz
base = os.path.dirname(archive)                         # dir holding the .pyz
os.environ.setdefault("DISTIL_CORPUS", os.path.join(base, "corpus"))
from distil.cli import main
sys.exit(main())
PY

mkdir -p dist
PY="$(command -v python3 || command -v python)"
"$PY" -m zipapp "$STAGE" -p "/usr/bin/env python3" -o "$OUT"
# Drop the corpus next to the pyz so bundled-corpus commands work out of the box.
rm -rf dist/corpus && cp -r corpus dist/corpus
chmod +x "$OUT"
echo "built $OUT  (+ dist/corpus)"
echo "run: DISTIL_CORPUS=dist/corpus python $OUT bench"
