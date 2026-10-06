"""One registry for every DISTIL_* variable; the docs table and test hooks must agree with it."""

from __future__ import annotations

import re
from pathlib import Path

from distil import _testing, envvars

_ROOT = Path(__file__).resolve().parent.parent
_NAME = re.compile(r"DISTIL_[A-Z0-9_]+")


def test_every_variable_in_code_is_registered() -> None:
    used = {
        m for p in (_ROOT / "distil").rglob("*.py") for m in _NAME.findall(p.read_text("utf-8"))
    }
    used = {u for u in used if not u.endswith("_")}  # DISTIL_ prefixes in prose/f-strings
    missing = used - {n for n, *_ in envvars.VARS}
    assert not missing, (
        f"unregistered DISTIL_* variables: {sorted(missing)}; add to distil/envvars.py"
    )


def test_docs_table_matches_registry() -> None:
    doc = (_ROOT / "docs" / "cli.html").read_text("utf-8")
    assert envvars.render_table() in doc, "run: python -m distil.envvars --write"


def test_test_hooks_are_inert_without_the_switch(monkeypatch) -> None:
    monkeypatch.setenv("DISTIL_HOTSWAP_TEST_FAIL_READY", "/tmp/x")
    monkeypatch.delenv("DISTIL_TESTING", raising=False)
    assert _testing.hook("DISTIL_HOTSWAP_TEST_FAIL_READY") is None
    monkeypatch.setenv("DISTIL_TESTING", "1")
    assert _testing.hook("DISTIL_HOTSWAP_TEST_FAIL_READY") == "/tmp/x"
