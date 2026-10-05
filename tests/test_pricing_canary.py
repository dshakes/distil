"""Drift canary for the model table (``distil/pricing.py``).

Every model id written into distil's own code — a default, a config it writes for a
client, a probe it sends — must name a row of the table exactly (a dated snapshot or a
Bedrock/Vertex spelling of one is fine). An id that only resolves through the
``resolve()`` prefix fallback is a NEW model being priced as an older one; an id that
does not resolve at all is billed at whatever each caller's fallback does. Both fail
here, so adding a model means adding its row.

And an unknown model is handled explicitly everywhere it is priced: ``resolve()`` says
None, and each caller has a stated fallback (unweighted tokens, None, or $0 tagged
"(unpriced)") — never a silent zero dollars passed off as a real price.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from distil import pricing

_PKG = Path(pricing.__file__).parent
# A literal that is a whole model id: family, then something carrying a version digit.
# (Client and tool names — "claude-desktop", "claude-cli" — carry no digit.)
_MODEL_ID = re.compile(r"(claude|gpt|gemini|o\d)-[a-z0-9.-]*\d[a-z0-9.-]*")


def _normalise(mid: str) -> str:
    """The table key *mid* names: Bedrock prefix and Vertex/snapshot suffixes dropped."""
    mid = mid.removeprefix("anthropic.").split("@", 1)[0]
    return re.sub(r"-\d{8}$", "", mid)


def _model_literals() -> list[tuple[str, int, str]]:
    found = []
    for path in sorted(_PKG.rglob("*.py")):
        if path.name == "pricing.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _MODEL_ID.fullmatch(node.value):
                    found.append((str(path.relative_to(_PKG.parent)), node.lineno, node.value))
    return found


def test_the_scan_sees_the_code() -> None:
    """Guard the canary itself: an empty scan would pass vacuously."""
    ids = {m for _, _, m in _model_literals()}
    assert "claude-sonnet-5-5" in ids  # the replay grader's default
    assert "claude-sonnet-4-6" in ids  # the VS Code entry distil writes


def test_every_model_id_in_the_code_is_in_the_table() -> None:
    known = set(pricing.CATALOG) | set(pricing.UNPRICED)
    missing = [
        f"{where}:{line}: {mid!r}"
        for where, line, mid in _model_literals()
        if _normalise(mid) not in known
    ]
    assert not missing, (
        "model ids used in code but absent from distil/pricing.py — add a CATALOG row "
        "(or an UNPRICED entry, for a model distil routes but must not price):\n"
        + "\n".join(missing)
    )


def test_table_rows_are_self_consistent() -> None:
    for key, p in pricing.CATALOG.items():
        assert p.name == key
        assert p.input_per_mtok > 0 and p.output_per_mtok >= p.input_per_mtok
        # Opus 5.5 and Fable 5.1 publish a cheaper cache hit; every other row is 0.1x.
        read = {"claude-opus-5-5": 0.05, "claude-fable-5-1": 0.025}.get(key, pricing.CACHE_READ_MULT)
        assert (p.cache_read_mult, p.cache_write_mult, p.cache_write_1h_mult) == (
            read,
            pricing.CACHE_WRITE_5M_MULT,
            pricing.CACHE_WRITE_1H_MULT,
        )
    assert pricing.DEFAULT_MODEL in pricing.CATALOG
    assert not set(pricing.UNPRICED) & set(pricing.CATALOG)


@pytest.mark.parametrize(
    "wire, row",
    [
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("anthropic.claude-opus-4-8", "claude-opus-4-8"),
        ("claude-opus-4-8@20260101", "claude-opus-4-8"),
        ("claude-sonnet-5-5", "claude-sonnet-5-5"),  # its own row, not sonnet-5's
    ],
)
def test_resolve_spellings(wire: str, row: str) -> None:
    p = pricing.resolve(wire)
    assert p is not None and p.name == row


@pytest.mark.parametrize("mid", [None, "", "gemini-2.5-pro", "mystery-model", *pricing.UNPRICED])
def test_unknown_and_unpriced_models_resolve_to_none(mid: str | None) -> None:
    assert pricing.resolve(mid) is None
    with pytest.raises(KeyError):
        pricing.get(mid or "")


def test_unknown_model_fallbacks_are_explicit(tmp_path: Path) -> None:
    """Each pricing caller's documented answer for a model it cannot price."""
    from distil.proxy import billed_input_equiv
    from distil.runtime import RuntimeSavings
    from distil.savings_screen import Tokens, request_cost

    usage = {"input_tokens": 100, "cache_read_input_tokens": 50, "output_tokens": 999}
    # The savings ledger's unit: unweighted input tokens (never Claude-rate weights).
    assert billed_input_equiv(usage, "mystery-model") == 150
    assert billed_input_equiv(usage, pricing.DEFAULT_MODEL) != 150
    # Dollar views: "cannot price" is None, not $0.
    assert request_cost("mystery-model", Tokens(100, 50, 0, 10)) is None
    # The runtime ledger keeps the tokens, books $0 and labels the row as unpriced.
    rs = RuntimeSavings(ledger_path=tmp_path / "ledger.jsonl")
    rs.record(1000, 400, model="mystery-model")
    assert rs.dollars_saved == 0.0
    assert rs.flush()
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["model"] == "mystery-model (unpriced)"


def test_check_pricing_script_parses_a_saved_page(tmp_path: Path) -> None:
    """scripts/check_pricing.py on a canned page — offline; the script is never run live
    in tests."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_pricing", _PKG.parent / "scripts" / "check_pricing.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def row(mid: str, figs: tuple[float, ...]) -> str:
        cells = "".join(f"<td>${f:g} / MTok</td>" for f in figs)
        return f"<tr><td>{mod.display_name(mid)}</td>{cells}</tr>"

    rows = {mid: mod.expected(p) for mid, p in pricing.CATALOG.items()}
    good = "<table>" + "".join(row(m, f) for m, f in rows.items()) + "</table>"
    assert mod.check_anthropic(mod.page_text(good)) == []

    drifted = dict(rows, **{"claude-opus-4-8": (6.0, 7.5, 12.0, 0.6, 30.0)})
    bad = "<table>" + "".join(row(m, f) for m, f in drifted.items()) + "</table>"
    problems = mod.check_anthropic(mod.page_text(bad))
    assert problems and all(p.startswith("claude-opus-4-8:") for p in problems)

    page = tmp_path / "anthropic.html"
    page.write_text(good, encoding="utf-8")
    openai = tmp_path / "openai.html"
    openai.write_text("<p>gpt-5.2 Input $1.00 Output $2.00</p>", encoding="utf-8")
    assert mod.main(["--anthropic", str(page), "--openai", str(openai)]) == 0
