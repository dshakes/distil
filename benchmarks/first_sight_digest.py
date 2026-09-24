"""First-sight reduction of the live digest path, per content class, on REAL originals.

Reads the local restore store (~/.distil/restore: the most recent blocks the live
proxy actually digested) in memory, runs the adapter's `_compress_tool_result_text`
on each block with disk writes and the census stubbed out, and writes AGGREGATES
ONLY — counts and token sums, never content — to a JSON artifact. The baseline is
the digest as of ``--baseline-ref`` (loaded from git), so before/after run on one
snapshot of the store.

    uv run --python 3.12 --with cryptography python benchmarks/first_sight_digest.py \
        --baseline-ref origin/main --out benchmarks/results/2026-09-24/first_sight_digest.json

Content classes are a shape heuristic over the text (this script only; the proxy is
content-free and never classifies live traffic this way). Tokens are distil's
deterministic heuristic tokenizer, the same one the adapter's reject-if-bigger uses.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from distil import mcp_server
from distil.adapters import anthropic as A
from distil.compress import keep_policy as K
from distil.compress import tier1 as T1
from distil.tokenizer import HeuristicTokenizer

_GREP = re.compile(r"^[\w./-]+:\d+[:-]")
_CATN = re.compile(r"^\s*\d+[\t→]")
_TS = re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d|^\[?\d\d:\d\d:\d\d")
_CODE = re.compile(
    r"^\s*(def |class |import |from \S+ import|function |const |let |fn |func |pub |#include|package )"
)
_MARKER = re.compile(r"^<< \+\d+ lines(?:, handle=[0-9a-f]{8})? >>$")


def content_class(t: str) -> str:
    s = t.lstrip()
    if "<html" in t[:2000].lower() or "<!doctype" in t[:200].lower():
        return "html"
    if s[:1] in "[{":
        try:
            json.loads(s)
            return "json"
        except ValueError:
            pass
    lines = t.splitlines() or [""]
    n = len(lines)
    kind = K.classify(t)
    if kind is K.ContentKind.DIFF:
        return "diff"
    if kind is K.ContentKind.LOG:
        return "test/build"
    if kind is K.ContentKind.TRACEBACK:
        return "traceback"
    if sum(bool(_GREP.match(x)) for x in lines) > 0.5 * n:
        return "search"
    if sum(bool(_CATN.match(x)) for x in lines) > 0.5 * n:
        return "file-read"
    if sum(bool(_CODE.match(x)) for x in lines) > 0.05 * n:
        return "code"
    if sum(bool(_TS.search(x)) for x in lines) > 0.3 * n:
        return "log"
    if sum(x.lstrip()[:1] in "{[" for x in lines) > 0.5 * n:
        return "jsonl"
    return "other"


def _baseline_digest(ref: str) -> Callable[..., tuple[str, bool]]:
    src = subprocess.run(
        ["git", "show", f"{ref}:distil/compress/tier1.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("distil.compress._tier1_baseline", f.name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Path(f.name).unlink()
    return mod.digest  # type: ignore[no-any-return]


def _run(texts: list[str], digest: Callable[..., tuple[str, bool]]) -> dict[str, Any]:
    tok = HeuristicTokenizer()
    A._tier1_digest = digest
    agg: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    markers = [0, 0]
    for t in texts:
        out = A._compress_tool_result_text(t, A.RestoreStore())
        a, b = tok.count(t), tok.count(out)
        for k in (content_class(t), "ALL"):
            agg[k][0] += 1
            agg[k][1] += a
            agg[k][2] += b
        for ln in out.splitlines():
            if _MARKER.match(ln):
                markers[0] += 1
                markers[1] += tok.count(ln)
    return {
        "by_class": {
            k: {"n": n, "orig_tokens": a, "out_tokens": b, "reduction": round(1 - b / a, 4)}
            for k, (n, a, b) in sorted(agg.items())
        },
        "line_markers": {"count": markers[0], "tokens": markers[1]},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline-ref", default="origin/main")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    # Measurement must not write: no restore persistence, no census.
    A._record_restore = lambda h, o: True  # type: ignore[assignment]
    A._census = lambda b, t: None  # type: ignore[assignment]
    A._census_tokens = lambda b, n: None  # type: ignore[assignment]
    d = mcp_server._restore_dir()
    texts = [t for p in sorted(d.iterdir()) if (t := mcp_server._read_restore_text(p))]
    if not texts:
        print(f"no restore originals under {d}", file=sys.stderr)
        return 1
    res = {
        "source": "local restore store (most recent live-digested blocks), aggregates only",
        "tokenizer": "heuristic",
        "baseline_ref": args.baseline_ref,
        "before": _run(texts, _baseline_digest(args.baseline_ref)),
        "after": _run(texts, T1.digest),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    b, a = res["before"]["by_class"]["ALL"], res["after"]["by_class"]["ALL"]
    print(
        f"{a['n']} blocks: {b['reduction']:.2%} -> {a['reduction']:.2%} "
        f"(digest output {b['out_tokens']} -> {a['out_tokens']} tokens)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
