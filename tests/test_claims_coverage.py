"""The other half of the claims gate: pages -> ledger, and ledger -> artifacts.

``tests/test_site_claims.py`` checks that every ledger entry still points at a real
place on a real page. That direction alone lets two failures through, and the
2026-09-15 audit found both live on ``main``:

1. **No reverse coverage.** ``docs/CLAIMS.md`` states the rule in bold -- *no
   number on the site without an entry in docs/claims.json* -- and nothing
   enforced it. ``docs/llms.txt``, ``plugins/`` and ``docs/index.html``'s trust
   card carried numbers no entry named, and CI stayed green.
2. **No artifact validation.** The ``artifact`` field was declared and never
   read, so an entry could cite a directory that held no artifact
   (``v-reread-delta-codebench``), or a page could print 52.3% while the log it
   is presented from said 47.9% (``docs/benchmark.html``).

This module closes both. It is deliberately blunt: it compares number *strings*,
because a number a reader can quote is a number the ledger has to name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CLAIMS_PATH = DOCS / "claims.json"

# Pages the scan reads. Entries in claims.json spell their `page` relative to
# docs/, so anything outside docs/ is reached with `../` -- the spelling the
# ledger already used for README.md.
SCAN_FILES: tuple[str, ...] = (
    *sorted(f"docs/{p.name}" for p in DOCS.glob("*.html")),
    "docs/llms.txt",
    "README.md",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "plugins").rglob("*.md")),
)

# docs/changelog.html is generated from CHANGELOG.md by
# scripts/build_changelog_page.py. Every number in it is inside a dated release
# section and describes what was true at that release -- a historical record, not
# a current claim. Withdrawn numbers kept there on purpose are covered by
# v-changelog-1130-withdrawn. Scanning it would demand a ledger entry per
# release note.
EXCLUDED_FILES: dict[str, str] = {
    "docs/changelog.html": "generated from CHANGELOG.md; every entry is dated and historical",
}

# A number is a claim unless it is one of these. Keep this list short, keep every
# reason specific, and never add a row here to make a real number pass.
NON_CLAIMS: dict[str, dict[str, str]] = {
    "docs/index.html": {
        "53%": "53.1% rounded, in the same sentence that dates the run",
        "10×": "Anthropic's published cache-read discount, not a distil measurement",
        "2×": "quotes the cache-busting incident's headline, covered by v-cache-aware-vs-naive",
        "87%": "quotes someone else's marketing claim in order to refuse it",
        "99%": "the status line's glyph threshold (distil/cli.py), a constant not a result",
        "2.1×": "named only to say the 0.27.0-era ratio does not survive the re-run",
    },
    "docs/benchmark.html": {
        "53%": "53.1% rounded, in the same sentence that dates the run",
        "20%": "budget-ladder step size on 5-turn trajectories, a config consequence",
        "50%": "an anecdote about a corpus that was discarded, not a published result",
        "90%": "budget-ladder step arithmetic on 5-turn trajectories, a config consequence",
        "2.1×": "named only to say the 0.27.0-era ratio does not survive the re-run",
        "40%": "39.7% rounded, in the same sentence that dates it",
        "91%": "~91% rounds the 2026-06-23 block's 91.1%, covered by v-benchmark-codebench",
        "47%": "47.0% rounded, covered by v-curve-lossless",
    },
    "docs/architecture.html": {
        "20%": "the validate harness's default control-group split, a flag value not a result",
        "0.1×": "Anthropic's published cache-read price ratio, not a distil measurement",
    },
    "docs/concepts.html": {
        "53%": "53.1% rounded, in the same sentence that dates the run",
        "0.1×": "Anthropic's published cache-read price ratio, not a distil measurement",
        "1.25×": "Anthropic's published cache-write price ratio, not a distil measurement",
        "10×": "Anthropic's published cache-read discount, not a distil measurement",
        "87%": "quotes someone else's marketing claim in order to refuse it",
        "81%": "~81% rounds the offline standings' 80.5%, covered by u-benchmark-offline-standings",
    },
    "docs/faq.html": {
        "7\u00d7": '"roughly 7x safer" restates v-provider-compaction\'s 92.5%-vs-12.5% ratio',
        "3×": "`--shadow 1.0` issues three replays per request; arithmetic, not a measurement",
        "99%": "the status line's glyph threshold (distil/cli.py), a constant not a result",
    },
    "docs/getting-started.html": {
        "2%": "the default shadow sample rate, a flag value not a result",
        "10×": "Anthropic's published cache-read discount, not a distil measurement",
        "3×": "`--shadow 1.0` issues three replays per request; arithmetic, not a measurement",
        "99%": "the status line's glyph threshold (distil/cli.py), a constant not a result",
    },
    "docs/output.html": {},
    "docs/benchmarks.html": {},
    "docs/llms.txt": {},
    "README.md": {
        "10×": "Anthropic's published cache-read discount, not a distil measurement",
        "2×": "quotes the cache-busting incident's headline, covered by v-cache-aware-vs-naive",
        "3×": "`--shadow 1.0` issues three replays per request; arithmetic, not a measurement",
        "40%": "a status-line example in the state table, illustrative like the rest of that table",
        "99%": "the status line's glyph threshold (distil/cli.py), a constant not a result",
    },
    "plugins/distil/README.md": {
        "99%": "the status line's glyph threshold (distil/cli.py), a constant not a result",
    },
    "docs/integrations.html": {
        "99%": "the status line's glyph threshold (distil/cli.py), a constant not a result",
        "95%": "the status line's other glyph threshold, named in the same sentence as 99%",
        "40%": "a status-line example in the state table, illustrative like the rest of that table",
    },
    "plugins/distil/commands/distil-stats.md": {
        "55%": "a rendering example inside a prompt template (`███░ 55% trimmed`)",
    },
    "plugins/distil/commands/distil-shadow.md": {
        "10%": "the shadow sample rate the command sets, a flag value not a result",
    },
}

# Pages that still carry numbers no ledger entry names. This is DEBT, not an
# allowlist: it is frozen, it may only shrink, and a NEW uncovered number on any
# of these pages still fails the test below. Clearing a page means adding its
# entries to docs/claims.json and deleting its row here.
LEDGER_DEBT: dict[str, frozenset[str]] = {
    "docs/adapters.html": frozenset(["0.1×"]),
    "docs/benchmark-independent.html": frozenset(["0.0%"]),
    "docs/cache-contract.html": frozenset(["0%", "0.0%", "0.1×", "1.25×", "2×"]),
    "docs/cache.html": frozenset(["50%", "75%", "90%"]),
    "docs/cli.html": frozenset(["10%", "12.5%", "2%", "39%", "3×", "92.5%", "99%"]),
    "docs/compare.html": frozenset(["20%", "4.8%"]),
    "docs/corpus.html": frozenset(["18.1%", "22.8%", "24.9%", "25.5%", "25.7%", "32.6%", "35.3%"]),
    "docs/evals.html": frozenset(["0%", "100%", "16%", "4×", "52%", "56%", "7%", "86%"]),
    "docs/learn-compression.html": frozenset(
        ["0.10×", "0.1×", "1.0×", "1.25×", "10×", "20×", "26.8%", "26×", "3×", "4×", "6×"]
    ),
    "docs/learn-distil.html": frozenset(["0.1×", "1.0×", "10×", "90%"]),
    "docs/learn-tokens.html": frozenset(["2×", "30%", "37%", "5×"]),
    "docs/provider-compaction.html": frozenset(
        ["27.8%", "36.7%", "5×", "7.5%", "95%", "98.6%", "99.5%"]
    ),
    "docs/research.html": frozenset(
        [
            "0%",
            "0.0%",
            "1%",
            "1.4%",
            "100%",
            "100.0%",
            "11.4%",
            "11.6%",
            "12.0%",
            "12.5%",
            "14.0%",
            "14.4%",
            "15.7%",
            "16.0%",
            "18%",
            "18.0%",
            "2%",
            "2.4%",
            "2.5%",
            "20.0%",
            "23.2%",
            "26%",
            "26.0%",
            "27.8%",
            "28.4%",
            "28.6%",
            "28.8%",
            "29%",
            "3.5%",
            "3.9%",
            "31%",
            "32%",
            "32.4%",
            "32.6%",
            "32.7%",
            "34%",
            "35.0%",
            "36.6%",
            "36.7%",
            "36.8%",
            "39.2%",
            "4.0%",
            "4.2%",
            "41.1%",
            "43.5%",
            "43.8%",
            "48%",
            "5%",
            "5.5%",
            "5.6%",
            "5.7%",
            "52%",
            "52.0%",
            "53%",
            "54.0%",
            "55.5%",
            "58%",
            "60%",
            "60.0%",
            "7.0%",
            "7.5%",
            "71.5%",
            "73.9%",
            "8.0%",
            "8.4%",
            "8.5%",
            "86%",
            "91%",
            "92.5%",
            "93.9%",
            "95%",
            "95.0%",
            "95.4%",
            "96%",
            "96.7%",
            "98.6%",
            "98.8%",
            "99.5%",
        ]
    ),
    "docs/subscription.html": frozenset(
        ["0%", "0.0%", "0.00%", "25%", "33%", "33.3%", "96%", "99%"]
    ),
    "docs/techniques.html": frozenset(
        [
            "0%",
            "0.1×",
            "1,000×",
            "100%",
            "10×",
            "12.8%",
            "28.8%",
            "32.4%",
            "36.8%",
            "42%",
            "44%",
            "5.6%",
            "50%",
            "50.6%",
            "62%",
            "70%",
            "83.2%",
            "85%",
            "96%",
        ]
    ),
    "docs/which-mode.html": frozenset(["0%", "0.00%", "33%", "99%"]),
}


def _load_claims() -> list[dict]:
    return json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))["entries"]


def _pages(entry: dict) -> list[str]:
    page = entry["page"]
    return [page] if isinstance(page, str) else list(page)


def _page_keys(scan_file: str) -> set[str]:
    """Every spelling a claims.json `page` field may use for this file."""
    if scan_file.startswith("docs/"):
        return {scan_file[len("docs/") :], scan_file}
    return {f"../{scan_file}", scan_file}


_STRIP = (
    re.compile(r"<style\b.*?</style>", re.S | re.I),
    re.compile(r"<script\b.*?</script>", re.S | re.I),
    re.compile(r"<svg\b.*?</svg>", re.S | re.I),
    re.compile(r"<!--.*?-->", re.S),
    # Sample terminal blocks and fenced code: illustrative by construction, and
    # the ledger already carries `u-` entries saying so for the ones that matter.
    re.compile(r"<pre\b.*?</pre>", re.S | re.I),
    re.compile(r"```.*?```", re.S),
    re.compile(r"<[a-zA-Z/!][^>]*>", re.S),
)

# Statistical notation: a confidence level, a significance level or an interval
# bound is the shape of the claim, not the claim. Stripped before tokenising so
# "95% CI" never has to appear in the ledger.
_NOTATION = (
    re.compile(r"\b(?:9\d(?:\.\d)?)\s*%\s*(?:CI|confidence|conf\.)", re.I),
    re.compile(r"(?:at|with)\s+9\d\s*%", re.I),
    re.compile(r"[αδ]\s*=\s*0?\.\d+"),
    re.compile(r"≤\s*\d+(?:\.\d+)?\s*%\s*(?:@|at)?\s*9\d\s*%"),
    re.compile(r"≤\s*5\s*%"),
    re.compile(r"95%\s*CI", re.I),
)

# A percentage or a multiplier. `(?!\d)` keeps 1024x1024 (an image size) from
# reading as a 1024x multiplier.
_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:%|×)(?!\d)")


def _visible_numbers(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    for pattern in _STRIP:
        text = pattern.sub(" ", text)
    for pattern in _NOTATION:
        text = pattern.sub(" ", text)
    return {m.group(0).replace(" ", "").replace("&nbsp;", "") for m in _TOKEN.finditer(text)}


def _covered_numbers(entries: list[dict], page_keys: set[str]) -> str:
    """One blob of every entry that names this page, for substring lookup."""
    return " ".join(
        json.dumps(e, ensure_ascii=False) for e in entries if page_keys & set(_pages(e))
    )


@pytest.mark.parametrize("scan_file", [f for f in SCAN_FILES if f not in EXCLUDED_FILES])
def test_every_public_number_is_named_by_the_ledger(scan_file: str):
    """docs/CLAIMS.md's rule, enforced instead of documented.

    If this fails, a number is on a public page that no claims.json entry
    mentions. Fix it by adding the number to the entry that already covers its
    table (a `values` list is fine), by adding a new entry with its artifact, or
    -- if it is genuinely not a claim -- by adding one commented row to
    NON_CLAIMS with a specific reason. Never add a row to make a real result
    pass.
    """
    path = ROOT / scan_file
    entries = _load_claims()
    blob = _covered_numbers(entries, _page_keys(scan_file))
    non_claims = NON_CLAIMS.get(scan_file, {})
    debt = LEDGER_DEBT.get(scan_file, frozenset())

    uncovered = sorted(
        tok
        for tok in _visible_numbers(path)
        if tok not in blob and tok not in non_claims and tok not in debt
    )
    assert not uncovered, (
        f"{scan_file} shows {len(uncovered)} number(s) that no docs/claims.json entry "
        f"naming this page mentions: {uncovered}"
    )


def test_ledger_debt_and_non_claims_do_not_rot():
    """Both escape hatches must stay true, or they become the stale thing.

    A token listed as debt or as a non-claim that is no longer on the page is a
    leftover: it silently widens the gate for whatever number lands there next.
    """
    stale: list[str] = []
    for scan_file in SCAN_FILES:
        if scan_file in EXCLUDED_FILES:
            continue
        visible = _visible_numbers(ROOT / scan_file)
        for tok in LEDGER_DEBT.get(scan_file, frozenset()):
            if tok not in visible:
                stale.append(f"{scan_file}: LEDGER_DEBT {tok!r} is no longer on the page")
        for tok in NON_CLAIMS.get(scan_file, {}):
            if tok not in visible:
                stale.append(f"{scan_file}: NON_CLAIMS {tok!r} is no longer on the page")
    assert not stale, "remove these rows from tests/test_claims_coverage.py:\n" + "\n".join(stale)


def test_every_non_claim_carries_a_reason():
    empty = [
        f"{page}:{tok}"
        for page, rows in NON_CLAIMS.items()
        for tok, reason in rows.items()
        if not reason.strip()
    ]
    assert not empty, f"NON_CLAIMS rows without a reason: {empty}"


# ---------------------------------------------------------------- artifacts


def _numbers_in(artifact: Path) -> set[str]:
    """Every number the artifact states, as a string, for substring comparison.

    A JSON field may hold a fraction (0.368) where the page prints a percentage
    (36.8%), so both forms are produced. Applied per file, so a directory of
    per-condition score files is checked the same way a single file is.
    """
    files = (
        sorted(p for p in artifact.rglob("*") if p.is_file()) if artifact.is_dir() else [artifact]
    )
    forms: set[str] = set()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        forms.update(m.group(0) for m in re.finditer(r"-?\d+(?:\.\d+)?", text))
        if path.suffix != ".json":
            continue
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue  # a .json that will not parse is caught by the existence test
        scalars: list[float] = []

        def walk(node):
            if isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                scalars.append(float(node))

        walk(payload)
        for n in scalars:
            for scaled in (n, n * 100, abs(n), abs(n) * 100):
                forms.add(f"{scaled:.1f}")
                forms.add(f"{scaled:.2f}")
                forms.add(f"{round(scaled):d}")
    return forms


def _value_body(value: str) -> str:
    """'83.2%' -> '83.2'; '1,000x' -> '1000'."""
    return value.rstrip("%×").replace(",", "").strip()


@pytest.mark.parametrize("entry", _load_claims(), ids=lambda e: e["id"])
def test_claim_artifact_exists_and_states_its_values(entry: dict):
    """The `artifact` field stops being decoration.

    Three states were live in the ledger and all three passed CI before this
    test: an artifact path that did not exist, an artifact that contradicted the
    page, and a headline claim with no artifact at all.
    """
    manual = entry.get("check") == "manual"
    values = entry.get("values") or []
    artifact_field = entry.get("artifact")

    if manual:
        assert entry.get("check_reason", "").strip(), (
            f"{entry['id']}: check=manual needs a check_reason saying why it cannot be "
            "machine-checked"
        )
        return

    if values:
        assert artifact_field, (
            f"{entry['id']}: carries values {values} but no artifact. Either cite the file "
            'they come from, or mark the entry "check": "manual" with a reason.'
        )

    if not artifact_field:
        return

    artifact = ROOT / artifact_field
    assert artifact.exists(), f"{entry['id']}: artifact {artifact_field} does not exist"
    if artifact.is_dir():
        assert any(p.is_file() for p in artifact.rglob("*")), (
            f"{entry['id']}: artifact directory {artifact_field} contains no files"
        )

    if not values:
        return

    # Source files cited as artifacts (harness.py, provenance.py, shadow.py) back
    # a mechanism, not a measurement; a number check against them is meaningless.
    if artifact.is_file() and artifact.suffix in {".py", ".md"}:
        return

    stated = _numbers_in(artifact)
    missing = [v for v in values if _value_body(v) not in stated]
    assert not missing, (
        f"{entry['id']}: {artifact_field} does not state {missing} "
        f"(claimed on {', '.join(_pages(entry))})"
    )
