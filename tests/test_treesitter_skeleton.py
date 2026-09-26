"""Tree-sitter code skeletons (the optional ``distil-llm[code]`` extra).

The real-grammar tests skip when the extra is absent; the stub-parser tests below
exercise the same elision logic everywhere, and the fallback tests prove the
heuristic is used, unchanged, when no grammar is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from distil import skeleton as sk
from distil.compress.tier1 import Tier1Reversible
from distil.trajectory import Block, Kind

GO = """package main

import "fmt"

// Add sums two ints.
func Add(a int, b int) int {
\tc := a + b
\tfmt.Println(c)
\treturn c
}

type S struct{ n int }

func (s *S) Inc() {
\ts.n++
\ts.n++
\ts.n++
}
"""

RUBY = """class Greeter
  def initialize(name)
    @name = name
    @count = 0
  end

  def greet(times)
    times.times { |i| puts "hi #{@name} #{i}" }
    @count += times
    @count
  end
end
"""

RUST = """use std::fmt;

pub fn area(w: u32, h: u32) -> u32 {
    let a = w * h;
    println!("{}", a);
    a
}

impl fmt::Display for P {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)?;
        Ok(())
    }
}
"""

TS = """export interface Opts { n: number }

export function run(o: Opts): number {
  const x = o.n * 2;
  console.log(x);
  return x;
}

const f = (a: string) => {
  const b = a.trim();
  return b.length;
};
"""

JAVA = """package a.b;

public class Box {
    private int n;

    public int twice(int k) {
        int r = k * 2;
        n += r;
        return r;
    }
}
"""

CPP = """#include <vector>

namespace geo {
int sum(const std::vector<int>& v) {
    int s = 0;
    for (int x : v) s += x;
    return s;
}
}
"""


def _grammar(mod: str) -> None:
    pytest.importorskip("tree_sitter")
    pytest.importorskip(mod)


@pytest.mark.parametrize(
    "src,mod,kept,gone",
    [
        (GO, "tree_sitter_go", ["func Add(a int, b int) int {", "// Add sums"], "fmt.Println"),
        (RUBY, "tree_sitter_ruby", ["def greet(times)", "  end", "class Greeter"], "@count +="),
        (RUST, "tree_sitter_rust", ["pub fn area(w: u32, h: u32) -> u32 {", "impl"], "println!"),
        (TS, "tree_sitter_typescript", ["export function run(o: Opts): number {"], "console.log"),
        (JAVA, "tree_sitter_java", ["public int twice(int k) {", "private int n;"], "n += r"),
        (CPP, "tree_sitter_cpp", ["int sum(const std::vector<int>& v) {"], "for (int x"),
    ],
    ids=["go", "ruby", "rust", "ts", "java", "cpp"],
)
def test_real_parse_keeps_signatures_elides_bodies(src, mod, kept, gone):
    _grammar(mod)
    out = sk.treesitter_skeleton(src)
    assert out is not None and len(out) < len(src)
    for k in kept:
        assert k in out
    assert gone not in out
    assert sk.treesitter_skeleton(src) == out  # deterministic


def test_ruby_is_new_ground_for_the_heuristic():
    """The brace heuristic cannot see Ruby at all; the grammar can."""
    _grammar("tree_sitter_ruby")
    assert sk.generic_code_skeleton(RUBY) is None
    assert sk.best_code_skeleton(RUBY) is not None


def test_broken_code_falls_back():
    _grammar("tree_sitter_go")
    broken = GO.replace("return c\n}", "return c\n")  # unbalanced: parse has errors
    assert sk.treesitter_skeleton(broken) is None


def test_tier1_skeleton_stays_recoverable():
    _grammar("tree_sitter_go")
    blk = Block(id="b1", kind=Kind.TOOL_OUTPUT, text=GO * 3)
    res = Tier1Reversible().compress([blk])
    (h,) = res.restore
    assert res.restore[h] == GO * 3
    assert f"handle={h}" in res.blocks[0].text


# ------------------------------------------------ extra-independent: stub parser


@dataclass
class _Node:
    type: str
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    text: bytes = b""
    children: list[_Node] = field(default_factory=list)
    body: _Node | None = None
    has_error: bool = False
    is_missing: bool = False
    start_byte: int = 0
    end_byte: int = 0

    def child_by_field_name(self, name: str) -> _Node | None:
        return self.body if name == "body" else None


@dataclass
class _Tree:
    root_node: _Node


class _Parser:
    def __init__(self, root: _Node | None) -> None:
        self.root = root

    def parse(self, data: bytes) -> _Tree:
        if self.root is None:
            raise ValueError("boom")
        return _Tree(self.root)


SRC = "fn a() {\n  x;\n  y;\n  z;\n}\nfn b() {\n  q;\n}\n"


def _fake_tree() -> _Node:
    body_a = _Node("block", (0, 7), (4, 1), b"{...}")
    fn_a = _Node("function_item", (0, 0), (4, 1), children=[body_a], body=body_a)
    body_b = _Node("block", (5, 7), (7, 1), b"{...}")  # one inner line: not worth it
    fn_b = _Node("function_item", (5, 0), (7, 1), children=[body_b], body=body_b)
    return _Node("source_file", (0, 0), (8, 0), children=[fn_a, fn_b])


@pytest.fixture
def only_rust(monkeypatch):
    def use(parser):
        monkeypatch.setattr(sk, "_ts_parsers", {k: None for k in sk._TS_GRAMMARS})
        sk._ts_parsers["rust"] = parser

    return use


def test_stub_elides_outermost_multiline_bodies(only_rust):
    only_rust(_Parser(_fake_tree()))
    assert sk.treesitter_skeleton(SRC) == "fn a() {\n  ...\n}\nfn b() {\n  q;\n}\n"
    assert sk.treesitter_available()


def test_stub_ruby_style_body(only_rust):
    body = _Node("body_statement", (1, 2), (3, 5), b"a")
    meth = _Node("method", (0, 0), (4, 3), children=[body], body=body)
    only_rust(_Parser(_Node("program", (0, 0), (5, 0), children=[meth])))
    src = "fn r(x)\n  a\n  b\n  c\nend\n"  # the rust hint nominates the stub parser
    assert sk.treesitter_skeleton(src) == "fn r(x)\n  ...\nend\n"


@pytest.mark.parametrize(
    "parser",
    [
        _Parser(None),
        _Parser(
            _Node(
                "source_file",
                (0, 0),
                (8, 0),
                has_error=True,
                children=[_Node("ERROR", (0, 0), (8, 0), end_byte=len(SRC))],
            )
        ),
        _Parser(_Node("source_file", (0, 0), (1, 0))),
    ],
    ids=["raises", "has-error", "no-functions"],
)
def test_stub_declines(only_rust, parser):
    only_rust(parser)
    assert sk.treesitter_skeleton(SRC) is None


def test_stub_one_line_body_not_elided(only_rust):
    body = _Node("body_statement", (0, 8), (0, 9), b"a")
    meth = _Node("method", (0, 0), (0, 12), children=[body], body=body)
    only_rust(_Parser(_Node("program", (0, 0), (1, 0), children=[meth])))
    assert sk.treesitter_skeleton("fn r(x) a end\nfn s\n") is None


def test_absent_extra_means_the_heuristic_unchanged(monkeypatch):
    monkeypatch.setattr(sk, "_ts_parsers", {k: None for k in sk._TS_GRAMMARS})
    assert not sk.treesitter_available()
    assert sk.treesitter_skeleton(GO) is None
    assert sk.best_code_skeleton(GO) == sk.generic_code_skeleton(GO)
    assert sk.treesitter_skeleton("x" * (sk._TS_MAX_BYTES + 1) + "\n") is None
    assert sk.treesitter_skeleton("no newline") is None


def test_parser_loader_handles_a_missing_grammar(monkeypatch):
    monkeypatch.setattr(sk, "_ts_parsers", {})
    monkeypatch.setitem(sk._TS_GRAMMARS, "go", ("no_such_grammar_module", "language"))
    assert sk._ts_parser("go") is None
    assert sk._ts_parsers["go"] is None  # cached, not retried per block


def _fn(kind, row0, row1, children=(), has_error=False):
    body = _Node("block", (row0, 5), (row1, 1), b"{", children=list(children))
    return _Node(kind, (row0, 0), (row1, 1), children=[body], body=body, has_error=has_error)


def test_named_inner_function_keeps_the_outer_body_open(only_rust):
    """A factory/IIFE/describe wrapper must not swallow every signature under it."""
    inner = _fn("function_item", 1, 5)
    only_rust(
        _Parser(
            _Node("source_file", (0, 0), (7, 0), children=[_fn("function_item", 0, 6, [inner])])
        )
    )
    src = "fn outer() {\n fn inner() {\n  a;\n  b;\n  c;\n }\n}\n"
    assert sk.treesitter_skeleton(src) == "fn outer() {\n fn inner() {\n  ...\n }\n}\n"


def test_anonymous_callbacks_go_with_the_body(only_rust):
    cb = _fn("arrow_function", 1, 5)
    only_rust(
        _Parser(_Node("source_file", (0, 0), (7, 0), children=[_fn("function_item", 0, 6, [cb])]))
    )
    src = "fn outer() {\n xs.map(x => {\n  a;\n  b;\n  c;\n })\n}\n"
    assert sk.treesitter_skeleton(src) == "fn outer() {\n ...\n}\n"


def test_function_with_a_parse_error_is_left_whole(only_rust):
    bad = _fn("function_item", 0, 4, has_error=True)
    only_rust(
        _Parser(_Node("source_file", (0, 0), (8, 0), children=[bad, _fake_tree().children[1]]))
    )
    assert sk.treesitter_skeleton(SRC) is None


def test_recursion_limit_falls_back(only_rust, monkeypatch):
    only_rust(_Parser(_fake_tree()))

    def deep(node):
        raise RecursionError

    monkeypatch.setattr(sk, "_ts_error_bytes", deep)
    assert sk.treesitter_skeleton(SRC) is None


def test_every_installed_grammar_loads():
    """An ABI mismatch (grammar newer than the tree-sitter runtime) fails silently into
    the heuristic; this is the check that notices. pyproject caps versions for 3.9."""
    import importlib.util

    pytest.importorskip("tree_sitter")
    sk._ts_parsers.clear()
    for lang, (mod, _fn) in sk._TS_GRAMMARS.items():
        if importlib.util.find_spec(mod) is not None:
            assert sk._ts_parser(lang) is not None, lang
