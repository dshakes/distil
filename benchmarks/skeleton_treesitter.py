"""Tree-sitter vs brace-heuristic code skeletons, measured on real source.

    uv run --extra code python benchmarks/skeleton_treesitter.py [--out PATH]

For each file: characters kept by the zero-dependency brace heuristic (what distil
does without the ``[code]`` extra) and by the tree-sitter skeleton (with it), plus
**signature retention** — the share of the file's function signatures (first line of
every NAMED function/method the grammar finds; callbacks and closures are body
detail) that survive verbatim in the skeleton. A
skeleton that is small because it dropped signatures is not a better skeleton.

Inputs are the repo's own non-Python source plus a fixed set of public files at
pinned tags, fetched once into a temp cache and identified in the artifact by URL +
sha256 (third-party source is not committed). Offline re-runs reuse the cache.
Deterministic: no sampling, no model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from distil import skeleton as sk  # noqa: E402

PINNED = {
    "go": "https://raw.githubusercontent.com/golang/go/go1.23.0/src/strings/builder.go",
    "rust": "https://raw.githubusercontent.com/rust-lang/rust/1.80.0/library/alloc/src/string.rs",
    "java": "https://raw.githubusercontent.com/google/guava/v33.0.0/guava/src/com/google/common/base/Strings.java",
    "ruby": "https://raw.githubusercontent.com/rails/rails/v7.1.0/activesupport/lib/active_support/inflector/methods.rb",
    "c": "https://raw.githubusercontent.com/redis/redis/7.2.0/src/sds.c",
    "cpp": "https://raw.githubusercontent.com/google/leveldb/1.23/db/db_impl.cc",
    "typescript": "https://raw.githubusercontent.com/microsoft/TypeScript/v5.4.5/src/compiler/core.ts",
    "javascript": "https://raw.githubusercontent.com/expressjs/express/4.19.2/lib/router/index.js",
}
LOCAL = {
    "rust": "rust/distil-core/src/lib.rs",
    "typescript": "examples/js_ai_sdk_middleware.ts",
    "javascript": "docs/site.js",
}


def _fetch(url: str, cache: Path) -> str:
    p = cache / hashlib.sha256(url.encode()).hexdigest()[:16]
    if not p.exists():
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 — pinned https URLs
            p.write_bytes(r.read())
    return p.read_text(encoding="utf-8")


def _signatures(lang: str, text: str) -> list[str]:
    parser = sk._ts_parser(lang)
    if parser is None:
        return []
    tree = parser.parse(text.encode())  # type: ignore[attr-defined]
    lines = text.split("\n")
    out, stack = [], [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type in sk._TS_FUNCTIONS and n.type not in sk._TS_ANONYMOUS:
            out.append(lines[n.start_point[0]])
        stack.extend(n.children)
    return out


def measure(lang: str, source: str, text: str) -> dict:
    heur = sk.generic_code_skeleton(text) or text
    ts = sk.treesitter_skeleton(text) or text
    sigs = [s for s in _signatures(lang, text) if s.strip()]

    def kept(skel: str) -> float | None:
        if not sigs:
            return None
        have = set(skel.split("\n"))
        return round(sum(s in have for s in sigs) / len(sigs), 4)

    return {
        "lang": lang,
        "source": source,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chars": len(text),
        "functions": len(sigs),
        "heuristic_chars": len(heur),
        "treesitter_chars": len(ts),
        "heuristic_reduction_pct": round(100 * (1 - len(heur) / len(text)), 2),
        "treesitter_reduction_pct": round(100 * (1 - len(ts) / len(text)), 2),
        "heuristic_signatures_kept": kept(heur),
        "treesitter_signatures_kept": kept(ts),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks/results/skeleton_treesitter.json"))
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "distil-ts-bench"))
    args = ap.parse_args(argv)
    if not sk.treesitter_available():
        print("install the extra first: uv run --extra code python " + __file__)
        return 2
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    rows = [measure(lang, url, _fetch(url, cache)) for lang, url in PINNED.items()]
    rows += [
        measure(lang, rel, (ROOT / rel).read_text(encoding="utf-8")) for lang, rel in LOCAL.items()
    ]
    tot = sum(r["chars"] for r in rows)
    summary = {
        "files": len(rows),
        "chars": tot,
        "heuristic_reduction_pct": round(
            100 * (1 - sum(r["heuristic_chars"] for r in rows) / tot), 2
        ),
        "treesitter_reduction_pct": round(
            100 * (1 - sum(r["treesitter_chars"] for r in rows) / tot), 2
        ),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "files": rows}, indent=2) + "\n")
    for r in rows:
        print(
            f"{r['lang']:<11} {r['chars']:>7} chars  heuristic -{r['heuristic_reduction_pct']:5.1f}% "
            f"(sigs {r['heuristic_signatures_kept']})  tree-sitter -{r['treesitter_reduction_pct']:5.1f}% "
            f"(sigs {r['treesitter_signatures_kept']})  {r['source'].rsplit('/', 1)[-1]}"
        )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
