"""Tests for the content-aware skeleton digest (distil/skeleton.py)."""

from __future__ import annotations

import re

from distil.skeleton import (
    code_skeleton,
    generic_code_skeleton,
    smart_digest,
    text_window,
)

SAMPLE = '''\
import os
import sys
from typing import Any

CONST = 42


def helper(x: int) -> int:
    """Add one to x."""
    y = x + 1
    z = y * 2
    return z


class Widget:
    """A widget."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.parts = []

    def render(self) -> str:
        out = []
        for p in self.parts:
            out.append(str(p))
        return "".join(out)
'''


def test_code_skeleton_keeps_structure_drops_bodies():
    sk = code_skeleton(SAMPLE)
    assert sk is not None
    # Imports, constants, signatures, class header all survive.
    assert "import os" in sk
    assert "CONST = 42" in sk
    assert "def helper(x: int) -> int:" in sk
    assert "class Widget:" in sk
    assert "def __init__(self, name: str) -> None:" in sk
    assert "def render(self) -> str:" in sk
    # Bodies are elided.
    assert "z = y * 2" not in sk
    assert "for p in self.parts:" not in sk
    assert "..." in sk
    # Docstring hints kept.
    assert "Add one to x." in sk
    # Actually smaller.
    assert len(sk) < len(SAMPLE)


def test_code_skeleton_navigable_lists_every_symbol():
    sk = code_skeleton(SAMPLE)
    for sym in ("helper", "Widget", "__init__", "render"):
        assert sym in sk, f"skeleton must name {sym} so the agent can expand it"


def test_code_skeleton_none_on_non_python():
    assert code_skeleton("this is { not ] python (((") is None


def test_code_skeleton_none_when_no_bodies():
    # Pure imports/assignments: nothing to elide, skeleton wouldn't help.
    assert code_skeleton("import os\nx = 1\ny = 2\n") is None


def test_text_window_keeps_head_and_tail():
    # The traceback failure: the exception lives at the END.
    tb = "Traceback (most recent call last):\n" + ("  frame\n" * 200) + "ValueError: boom"
    w = text_window(tb, head=80, tail=80)
    assert "Traceback" in w  # head
    assert "ValueError: boom" in w  # tail — head-only truncation would drop this
    assert len(w) < len(tb)


def test_text_window_short_text_untouched():
    assert text_window("short", head=400, tail=200) == "short"


def test_smart_digest_routes_code_vs_text():
    assert "def helper" in smart_digest(SAMPLE)  # code path
    tb = "ERROR\n" + ("x\n" * 500) + "AssertionError: nope"
    d = smart_digest(tb)
    assert "AssertionError: nope" in d  # text path keeps the tail


def test_smart_digest_nested_functions_do_not_break():
    src = "def outer():\n    a = 1\n    def inner():\n        return 2\n    return inner()\n"
    sk = code_skeleton(src)
    # Outer signature kept, its whole body (incl. the closure) elided to one '...'.
    assert sk is not None
    assert "def outer():" in sk
    assert sk.count("...") == 1
    assert "return inner()" not in sk


def test_skeleton_is_a_certified_ladder_level():
    from distil.conformal import default_ladder

    names = [n for n, _ in default_ladder()]
    assert "skeleton" in names and "protect+skeleton" in names


def test_skeleton_certificate_reversible_and_recovers_decisions():
    from benchmarks.skeleton_certificate import decision_equivalence, reversibility_check

    rev = reversibility_check()
    assert rev["reversible_pct"] == 100.0  # byte-exact recovery contract
    de = decision_equivalence()
    # Reversible tier: raw digest is lossy, but WITH recovery it is decision-equivalent.
    assert de["recovered_decision_change_pct"] == 0.0
    assert de["raw_decision_change_pct"] > de["recovered_decision_change_pct"]


# --- generic (non-Python) brace-block skeleton -----------------------------

_TS = """import { db } from './db';

export class UserService {
  async findById(id: string): Promise<User | null> {
    const row = await this.repo.get(id);
    if (!row) return null;
    return mapRow(row);
  }
}
"""


def test_generic_skeleton_folds_braced_code_and_keeps_signatures():
    sk = generic_code_skeleton(_TS)
    assert sk is not None and len(sk) < len(_TS)
    # signatures + structure kept, body elided
    assert "export class UserService {" in sk
    assert "async findById(id: string): Promise<User | null> {" in sk
    assert "const row" not in sk and "mapRow(row)" not in sk
    assert "..." in sk


def test_generic_skeleton_defers_on_non_code_and_unbalanced():
    assert generic_code_skeleton("just prose, no braces here") is None
    assert generic_code_skeleton("plain { text } with no code keyword") is None
    # unbalanced braces (mid-edit / truncated) → bail, never corrupt
    assert generic_code_skeleton("func f() {\n  a();\n  b();\n  c();\n") is None
    # a closing brace with no matching open (depth goes negative) → bail
    assert generic_code_skeleton("func f() }\n  a();\n  b();\n  c();\n") is None


def test_generic_skeleton_ignores_braces_in_strings_and_comments():
    src = (
        "function f() {\n"
        '  const a = "a } brace in a string";  // and a { in a comment\n'
        "  const b = 2;\n"
        "  const c = 3;\n"
        "}\n"
    )
    sk = generic_code_skeleton(src)
    assert sk is not None
    assert "function f() {" in sk and sk.rstrip().endswith("}")


def test_generic_skeleton_handles_block_comments_and_escapes():
    # a /* */ block comment spanning lines with braces inside, and an escaped
    # quote inside a string before a brace — neither may unbalance the counter.
    src = (
        "function f() {\n"
        "  /* a block comment\n"
        "     with a } brace across lines */\n"
        '  const a = "he said \\" then { ";\n'
        "  const b = 2;\n"
        "  const c = 3;\n"
        "}\n"
    )
    sk = generic_code_skeleton(src)
    assert sk is not None and "function f() {" in sk
    assert sk.rstrip().endswith("}")  # balanced despite the stray braces


def test_generic_skeleton_short_body_not_elided():
    # a 1-line body is below min_run → left intact (no false elision)
    src = "func g() {\n  return 1;\n}\n"
    assert generic_code_skeleton(src) is None


def test_smart_digest_routes_nonpython_code_through_generic():
    out = smart_digest(_TS)
    assert "handle=" in out  # something was elided → recoverable marker present
    assert "findById" in out  # signature survived the digest


def test_generic_skeleton_reversible_through_tier1():
    from distil.compress.tier1 import Tier1Reversible
    from distil.trajectory import Block, Kind

    b = Block(id="b1", kind=Kind.TOOL_OUTPUT, text=_TS)
    res = Tier1Reversible().compress([b])
    comp = res.blocks[0].text
    assert "handle=" in comp and len(comp) < len(_TS)
    h = re.search(r"handle=([0-9a-f]{8})", comp).group(1)
    assert res.restore[h] == _TS  # byte-exact original recoverable
