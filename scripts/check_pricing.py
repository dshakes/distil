#!/usr/bin/env python3
"""Compare distil's model table with the providers' published pricing — maintainer tool.

Manually run, never in CI or tests (it reads live web pages). The drift canary
(tests/test_pricing_canary.py) checks that every model id distil's code uses has a
row; this checks that each row's numbers still match the price list.

For every ``distil.pricing.CATALOG`` row it finds the model's display name on
Anthropic's pricing page ("claude-opus-4-8" → "Claude Opus 4.8") and reads the next
five ``$N / MTok`` figures in that table row — base input, 5m cache write, 1h cache
write, cache hit, output — and compares them with the row's derived prices. For the
ids distil routes but deliberately leaves unpriced (``pricing.UNPRICED``) it prints the
first figures published next to the id on OpenAI's page, as information only.

Exit 0 when every row matches, 1 on any mismatch or a row the page does not list.

Usage:
    python3 scripts/check_pricing.py                 # fetch both pages
    python3 scripts/check_pricing.py --anthropic saved.html [--openai saved.html]
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil import pricing  # noqa: E402 — after the repo-root path insert

ANTHROPIC_URL = "https://docs.claude.com/en/docs/about-claude/pricing"
OPENAI_URL = "https://openai.com/api/pricing/"
_PRICE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*MTok", re.I)
_DOLLARS = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
_TOL = 1e-6


def page_text(raw: str) -> str:
    """Visible text of an HTML page, whitespace collapsed (tables become one line)."""
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).strip()


def display_name(model_id: str) -> str:
    """``claude-sonnet-4-6`` → ``Claude Sonnet 4.6``; ``claude-fable-5`` → ``Claude Fable 5``."""
    family, *ver = model_id.split("-")[1:]
    return f"Claude {family.capitalize()} {'.'.join(ver)}"


#: Column order of the page's model table, by layout. The page has carried both: the
#: older one (input, 5m write, 1h write, hit, output) and, from 2026-10, input and output
#: first under a "Base tokens / Prompt caching" header.
_OLD_ORDER = (0, 1, 2, 3, 4)
_NEW_ORDER = (0, 4, 1, 2, 3)  # page position -> expected() index
_NEW_HEADER = re.compile(r"Input\s+Output\s+5m writes\s+1h writes", re.I)


def published(text: str, model_id: str) -> tuple[float, ...] | None:
    """The five $/MTok figures of the model's table row, in :func:`expected`'s order, or
    None if not listed.

    The name must not run on into a longer version ("Opus 4" must not match "Opus 4.8"),
    and must start a table row: the first price follows the name within a short run of
    text that names no other model (a footnote glyph, "(limited availability)", or a
    one-line description). The page names models in navigation and prose first, and in
    later tables (batch, fast mode) whose first price is not the base input price.
    """
    name = re.escape(display_name(model_id)) + r"(?![.\d])"
    m = re.search(name + r"(?:(?!Claude )[^$]){0,120}?(?=\$)", text)
    if m is None:
        return None
    figures = [float(x) for x in _PRICE.findall(text[m.end() : m.end() + 400])[:5]]
    if len(figures) != 5:
        return None
    order = _NEW_ORDER if _NEW_HEADER.search(text, 0, m.start()) else _OLD_ORDER
    out = [0.0] * 5
    for pos, idx in enumerate(order):
        out[idx] = figures[pos]
    return tuple(out)


def expected(p: pricing.Pricing) -> tuple[float, ...]:
    per_m = 1_000_000
    return (
        p.input_per_mtok,
        p.cache_write * per_m,
        p.cache_write_1h * per_m,
        p.cache_read * per_m,
        p.output_per_mtok,
    )


def check_anthropic(text: str) -> list[str]:
    problems = []
    labels = ("input", "5m write", "1h write", "cache hit", "output")
    for mid, p in pricing.CATALOG.items():
        got = published(text, mid)
        if got is None:
            problems.append(f"{mid}: '{display_name(mid)}' not found with 5 prices on the page")
            continue
        want = expected(p)
        diffs = [
            f"{lab} {w:g} vs published {g:g}"
            for lab, w, g in zip(labels, want, got)
            if abs(w - g) > _TOL
        ]
        print(f"  {mid:<22} {'ok' if not diffs else 'MISMATCH'}")
        problems += [f"{mid}: {d}" for d in diffs]
    return problems


def report_unpriced(text: str) -> None:
    for mid in sorted(pricing.UNPRICED):
        i = text.lower().find(mid.lower())
        figs = _DOLLARS.findall(text[i : i + 300]) if i >= 0 else []
        shown = ", ".join(f"${f}" for f in figs[:3]) or "not found"
        print(f"  {mid:<22} unpriced in distil; published near the id: {shown}")


def _read(path: str | None, url: str) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(url, headers={"User-Agent": "distil-check-pricing"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — fixed https URL
        return resp.read().decode("utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--anthropic", metavar="HTML", help="a saved copy of the Anthropic page")
    ap.add_argument("--openai", metavar="HTML", help="a saved copy of the OpenAI page")
    args = ap.parse_args(argv)

    print(f"Anthropic ({args.anthropic or ANTHROPIC_URL})")
    problems = check_anthropic(page_text(_read(args.anthropic, ANTHROPIC_URL)))
    print(f"OpenAI ({args.openai or OPENAI_URL})")
    try:
        report_unpriced(page_text(_read(args.openai, OPENAI_URL)))
    except OSError as exc:  # informational only: never fail the check on this page
        print(f"  could not read the OpenAI page: {exc}")
    for p in problems:
        print(f"! {p}")
    print("all rows match" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
