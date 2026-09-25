"""Content-aware skeleton digest — a structure-preserving, reversible compressor.

The reversible tier's job is to make a *peripheral* context block small while keeping it
**navigable** (the agent can see what's there and recover the part it needs). Plain
head-truncation fails at both: it shows an arbitrary first-N chars of a file (you cannot
tell which functions exist) and it *drops the tail of a traceback* — exactly where the
exception and failing assertion live.

This module produces a skeleton instead:

* **Python source** (the SWE-bench regime): keep every ``import``, every class/def
  **signature** (with decorators and the first docstring line), and elide function
  *bodies* to ``...``. The agent sees the full structure — which symbols exist, where —
  and can ``distil_expand`` the one block it needs. Reconstructed via :mod:`ast`; falls
  back cleanly on syntax errors / partial files.
* **Tracebacks & test output**: keep the head *and the tail* (the exception, the
  ``file:line``, the failing assertion), collapsing the quiet middle.
* **Anything else**: head+tail window rather than head-only.

Every transform is deterministic, stdlib-only (no model, no network — auditable and
safe to run on untrusted context), and **lossy only at the surface**: the caller keeps
the original behind a content handle, so the block is fully recoverable (the reversible
tier's contract). This is the digest the certified relevance-gate digests periphery with.
"""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Any

# Markers an agent (or a human) can grep for; kept short to not eat the savings.
_ELIDED = "..."  # body placeholder, emitted at the body's indentation


def _handle(text: str) -> str:
    """8-hex SHA-256 prefix — mirrors tier1._handle so a digest marker resolves
    against the same restore key (build_restore / RestoreStore) the caller records."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _docstring_first_line(node: ast.AST) -> str | None:
    """First physical line of a node's docstring, if it has one (kept as a hint)."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        doc = first.value.value.strip().splitlines()
        if doc:
            return doc[0].strip()
    return None


def _function_body_ranges(tree: ast.AST) -> list[tuple[int, int, int, str | None]]:
    """For each function whose enclosing scope is *not* another function, return
    ``(body_first_line, end_line, col_offset, docstring_first_line)`` — the line span to
    elide, the indentation to place ``...`` at, and a docstring hint to keep. Methods
    (in a class) are included; closures (in a function) are not — eliding the outer body
    already removes them.
    """
    ranges: list[tuple[int, int, int, str | None]] = []

    def visit(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not in_function:
                body = child.body
                doc = _docstring_first_line(child)
                # Body starts after the signature (and after the docstring, if kept).
                first_stmt = body[1] if (doc and len(body) > 1) else body[0]
                start = first_stmt.lineno
                end = child.end_lineno or start
                # Elide only when the body starts BELOW the signature line —
                # a one-liner (`def f(): pass`) shares its line with the def,
                # and eliding it would erase the signature itself.
                if end >= start > child.lineno:
                    ranges.append((start, end, first_stmt.col_offset, doc))
                # Descend with in_function=True so closures inside are not double-counted.
                visit(child, True)
            else:
                visit(
                    child,
                    in_function or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)),
                )

    visit(tree, False)
    return ranges


def code_skeleton(text: str) -> str | None:
    """Python skeleton: signatures + imports kept, bodies elided to ``...``.

    Returns ``None`` when the text is not parseable Python (caller should fall back), or
    when the skeleton would not actually be smaller than the original.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None

    ranges = _function_body_ranges(tree)
    if not ranges:
        return None  # no function bodies to elide — skeleton wouldn't help

    lines = text.splitlines()
    # Map each body's first line -> (end, indent, doc) for one-pass emission.
    elide_start = {start: (end, col, doc) for (start, end, col, doc) in ranges}
    out: list[str] = []
    i = 1  # 1-based line numbers (ast convention)
    n = len(lines)
    while i <= n:
        if i in elide_start:
            end, col, doc = elide_start[i]
            pad = " " * col
            if doc:
                out.append(f'{pad}"""{doc}"""')
            out.append(f"{pad}{_ELIDED}")
            i = end + 1  # skip the elided body
        else:
            out.append(lines[i - 1])
            i += 1
    skeleton = "\n".join(out)
    return skeleton if len(skeleton) < len(text) else None


# --------------------------------------------------------------------------- #
# Language-agnostic brace-block skeleton — the non-Python half of the code
# compressor. Python has a real parser (ast, above); C-family / JS / TS / Go /
# Rust / Java / Swift / Kotlin are brace-delimited, so their structure is
# recoverable WITHOUT a native parser: keep every line that opens or closes a
# block (the signatures and braces), elide the runs of pure body statements
# between them. Zero dependency by design — distil stays a pure-Python install;
# tree-sitter would buy higher fidelity at the cost of a mandatory native
# grammar, which is the very thing we don't force. Conservative: a block whose
# braces don't balance (unparseable / mid-edit) is left intact — we save less,
# never corrupt — and the byte-exact original is always one expand() away.
# --------------------------------------------------------------------------- #

_CODE_HINT = re.compile(
    r"\b(?:function|func|def|class|struct|impl|interface|public|private|"
    r"static|void|const|let|var|fn|type|enum|namespace|package|import)\b"
)


def _brace_depths(text: str) -> list[tuple[int, int]] | None:
    """Per-line ``(depth_before, depth_after)`` counting ``{}`` while skipping
    braces inside strings and comments. Returns None if braces never balance to
    zero (not clean braced code) or nesting goes negative (a ``}`` with no open)."""
    depth = 0
    per_line: list[tuple[int, int]] = []
    in_block_comment = False
    for line in text.split("\n"):
        before = depth
        i, n = 0, len(line)
        in_str: str | None = None  # active string quote char, else None
        while i < n:
            ch = line[i]
            two = line[i : i + 2]
            if in_block_comment:
                if two == "*/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            if in_str is not None:
                if ch == "\\":  # escape — skip next char
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue
            if two == "//" or ch == "#":  # line comment (C-family // and shell/py #)
                break
            if two == "/*":
                in_block_comment = True
                i += 2
                continue
            if ch in "\"'`":
                in_str = ch
                i += 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    return None  # unbalanced — bail, leave the block intact
            i += 1
        per_line.append((before, depth))
    if depth != 0 or in_block_comment:
        return None
    return per_line


def generic_code_skeleton(text: str, min_run: int = 3) -> str | None:
    """Brace-delimited code skeleton for non-Python source. Keeps every line that
    opens or closes a block (signatures + braces) and every top-level line; elides
    maximal runs of >= ``min_run`` pure-body lines (all at depth >= 1, changing no
    brace) into a single ``...`` at the body's indentation.

    Returns None when the text isn't clean braced code, carries no code hint, or
    wouldn't actually shrink. Reversible: the caller keeps the byte-exact original
    behind a content handle (elided bodies are recovered by ``expand()``)."""
    if "{" not in text or not _CODE_HINT.search(text):
        return None
    depths = _brace_depths(text)
    if depths is None:
        return None
    lines = text.split("\n")
    # A line is STRUCTURAL if it sits at top level, or opens/closes a block
    # (its depth changes) — those carry the signatures and braces we keep. A
    # non-structural line is a pure body statement (depth stays >= 1).
    structural = [before == 0 or after == 0 or after != before for (before, after) in depths]
    out: list[str] = []
    i, n, changed = 0, len(lines), False
    while i < n:
        if structural[i]:
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < n and not structural[j]:
            j += 1
        if j - i >= min_run:
            indent = len(lines[i]) - len(lines[i].lstrip())
            out.append(" " * indent + _ELIDED)
            changed = True
        else:
            out.extend(lines[i:j])
        i = j
    skeleton = "\n".join(out)
    return skeleton if changed and len(skeleton) < len(text) else None


# --------------------------------------------------------------------------- #
# Real parses, when the optional ``distil-llm[code]`` extra is installed. The
# brace heuristic above cannot see Ruby at all (no braces) and gives up on any
# file whose braces it cannot balance; a grammar can. Uses the official
# per-language tree-sitter grammar wheels — bundled, offline, no download at
# parse time — and falls back to the heuristic whenever they are absent, the
# language is not recognised, or the parse has errors. Python keeps ``ast``: it is
# already a real parser and needs no extra. Deterministic: same text, same output.
# --------------------------------------------------------------------------- #

#: language -> (grammar module, language function). ``tsx`` is left out on purpose:
#: the TypeScript grammar parses plain JS too, and JSX-heavy files fall back.
_TS_GRAMMARS: dict[str, tuple[str, str]] = {
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "c": ("tree_sitter_c", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "javascript": ("tree_sitter_javascript", "language"),
}

#: Cheap text hints deciding which grammars are worth trying, in this order. A hint
#: only nominates a candidate; the parse (error-free) is what accepts it.
_TS_HINTS: dict[str, re.Pattern[str]] = {
    "go": re.compile(r"^package \w+\s*$(?s:.*)^func ", re.M),
    "rust": re.compile(r"^\s*(?:pub(?:\([\w:]+\))? )?(?:async )?fn \w+", re.M),
    "java": re.compile(
        r"^\s*(?:public |private |protected )?(?:abstract |final |static )*"
        r"(?:class|interface|enum|record) \w+",
        re.M,
    ),
    "cpp": re.compile(r"^\s*#include\b|\bnamespace \w+|\btemplate\s*<|\bstd::", re.M),
    "c": re.compile(r"^\s*#(?:include|define|ifndef)\b", re.M),
    "ruby": re.compile(r"^\s*def \w+[?!]?(?:\(.*\))?\s*$(?s:.*)^\s*end\s*$", re.M),
    "typescript": re.compile(
        r"\binterface \w+|\btype \w+ =|:\s*(?:string|number|boolean|void|any)\b"
    ),
    "javascript": re.compile(
        r"\bfunction\b|=>|\bconst \w+ =|\brequire\(|module\.exports|^export ", re.M
    ),
}

#: Node types whose ``body`` field is a function body (elided); everything else —
#: classes, modules, namespaces, impl blocks — is structure and stays.
_TS_FUNCTIONS = frozenset(
    {
        "function_declaration",  # js/ts/go
        "generator_function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "method_declaration",  # go/java
        "constructor_declaration",
        "func_literal",
        "function_item",  # rust
        "function_definition",  # c/cpp
        "method",  # ruby
        "singleton_method",
    }
)

_TS_MAX_BYTES = 1_000_000  # a parse is linear, but a pathological blob is not code
_ts_parsers: dict[str, Any] = {}


def _ts_parser(lang: str) -> Any:
    """A cached parser for *lang*, or None when tree-sitter or that grammar is absent."""
    if lang in _ts_parsers:
        return _ts_parsers[lang]
    parser = None
    try:
        import importlib

        from tree_sitter import Language, Parser

        mod_name, fn = _TS_GRAMMARS[lang]
        parser = Parser(Language(getattr(importlib.import_module(mod_name), fn)()))
    except (ImportError, AttributeError, TypeError, ValueError):
        parser = None  # extra not installed, or an incompatible tree-sitter API
    _ts_parsers[lang] = parser
    return parser


def treesitter_available() -> bool:
    """True when at least one grammar of the ``[code]`` extra is importable."""
    return any(_ts_parser(lang) is not None for lang in _TS_GRAMMARS)


def _ts_body_span(node: Any) -> tuple[int, int] | None:
    """0-based inclusive rows strictly inside *node*'s function body, or None."""
    body = node.child_by_field_name("body")
    if body is None:
        return None
    if body.text[:1] == b"{":
        lo, hi = body.start_point[0] + 1, body.end_point[0] - 1
    elif body.start_point[0] > node.start_point[0] and body.end_point[0] < node.end_point[0]:
        lo, hi = body.start_point[0], body.end_point[0]  # ruby: def … end
    else:
        return None
    # Two or more lines, or the "..." costs what it saves.
    return (lo, hi) if hi - lo >= 1 else None


#: Anonymous functions. A body holding only these (callbacks, closures) is elided
#: whole; a body holding a NAMED function stays open so that name stays visible.
_TS_ANONYMOUS = frozenset({"arrow_function", "function_expression", "func_literal"})


def _ts_elisions(node: Any) -> tuple[list[tuple[int, int]], bool]:
    """``(row spans to elide under node, whether a named function lies under it)``.

    A function body is elided whole unless it contains a named function — then its
    named children are elided instead, which is what keeps a factory closure, a
    ``describe``-style wrapper or an IIFE module from swallowing every signature in
    the file. Subtrees with parse errors are never elided: we save less, never cut a
    body we could not delimit."""
    if node.type == "ERROR":
        return [], False
    inner: list[tuple[int, int]] = []
    named = False
    for child in node.children:
        spans, child_named = _ts_elisions(child)
        inner += spans
        named = named or child_named
    kind = node.type
    if kind not in _TS_FUNCTIONS:
        return inner, named
    is_named = named or kind not in _TS_ANONYMOUS
    if named or node.has_error:
        return inner, is_named
    span = _ts_body_span(node)
    return ([span] if span else inner), is_named


def _ts_error_bytes(node: Any) -> int:
    if node.type == "ERROR" or node.is_missing:
        return max(1, node.end_byte - node.start_byte)
    if not node.has_error:
        return 0
    return sum(_ts_error_bytes(c) for c in node.children)


#: A parse more than this share ERROR is the wrong grammar (or not code), not a file
#: with a few macros the grammar cannot see through.
_TS_MAX_ERROR_SHARE = 0.05


def treesitter_skeleton(text: str) -> str | None:
    """Signatures and structure kept, function bodies elided to ``...`` — from a real
    parse. None when the extra is absent, no grammar parses the text cleanly, or the
    skeleton would not be smaller (the caller then falls back to the heuristic)."""
    if len(text) > _TS_MAX_BYTES or "\n" not in text:
        return None
    data = text.encode("utf-8", "surrogatepass")
    for lang, hint in _TS_HINTS.items():
        if not hint.search(text):
            continue
        parser = _ts_parser(lang)
        if parser is None:
            continue
        try:
            tree = parser.parse(data)
        except (ValueError, TypeError):
            continue
        root = tree.root_node
        try:
            if _ts_error_bytes(root) > _TS_MAX_ERROR_SHARE * len(data):
                continue  # a guessed language that does not parse is not this language
            spans = sorted(_ts_elisions(root)[0])
        except RecursionError:
            continue  # nesting deeper than the stack: leave it to the heuristic
        if not spans:
            continue
        lines = text.split("\n")
        out: list[str] = []
        i = 0
        for lo, hi in spans:
            if lo < i:
                continue
            out.extend(lines[i:lo])
            first = lines[lo]
            out.append(first[: len(first) - len(first.lstrip())] + _ELIDED)
            i = hi + 1
        out.extend(lines[i:])
        skeleton = "\n".join(out)
        if len(skeleton) < len(text):
            return skeleton
    return None


def best_code_skeleton(text: str) -> str | None:
    """The best skeleton available: Python ``ast``, then a tree-sitter parse (when the
    ``[code]`` extra is installed), then the zero-dependency brace heuristic."""
    return code_skeleton(text) or treesitter_skeleton(text) or generic_code_skeleton(text)


def _info(line: str) -> int:
    """Lexical informativeness proxy: count of distinct alphanumeric tokens (len>2).
    A stand-in for the self-information score extractive compressors rank lines by."""
    toks = {
        w for w in "".join(c if c.isalnum() else " " for c in line.lower()).split() if len(w) > 2
    }
    return len(toks)


def text_window(
    text: str, *, head: int = 400, tail: int = 200, keep_head: int = 6, keep_tail: int = 4
) -> str:
    """Salience-aware window for non-code blocks.

    Head-only truncation drops the end (tracebacks/assertions); a plain head+tail window
    drops *buried* high-signal lines (an error or decision line in the middle of noisy
    output). This keeps the first ``keep_head`` and last ``keep_tail`` lines as structural
    anchors, plus the most-informative middle lines (by :func:`_info`) up to a char budget
    of ``head + tail`` — in original order — so a decision/error line survives wherever it
    sits. Recovery stays byte-exact via the caller's content handle."""
    if len(text) <= head + tail:
        return text
    lines = text.split("\n")
    if len(lines) <= keep_head + keep_tail + 1:
        omitted = len(text) - head - tail
        return f"{text[:head]}\n... [{omitted} chars elided] ...\n{text[-tail:]}"

    keep: set[int] = set(range(keep_head)) | set(range(len(lines) - keep_tail, len(lines)))
    budget = head + tail - sum(len(lines[i]) for i in keep)
    middle = sorted(
        range(keep_head, len(lines) - keep_tail),
        key=lambda i: _info(lines[i]),
        reverse=True,
    )
    for i in middle:
        if budget <= 0:
            break
        keep.add(i)
        budget -= len(lines[i]) + 1

    out: list[str] = []
    prev = -1
    for i in sorted(keep):
        if i != prev + 1:
            out.append("... [elided] ...")
        out.append(lines[i])
        prev = i
    result = "\n".join(out)
    return result if len(result) < len(text) else text


def smart_digest(text: str, *, head: int = 400, tail: int = 200) -> str:
    """Best available structure-preserving digest of one context block.

    Code → skeleton (signatures kept, bodies elided), then a head/tail window over
    the skeleton so the grade's ``head``/``tail`` budget actually bites on code:
    without windowing the skeleton, a light grade (large budget) and a heavy grade
    (small budget) produce the *identical* skeleton, collapsing the graded gate to a
    binary one on source. Non-code → a head+tail window directly.

    Deterministic and lossy *only at the surface*: when anything is elided the digest
    ends with a single ``handle=<8hex>`` marker keying the byte-exact original, so the
    caller (which records that same handle) can recover it. If nothing is elided the
    text is returned unchanged — no marker, no empty recoverability promise.
    """
    sk = best_code_skeleton(text)
    base = sk if sk is not None else text
    body = text_window(base, head=head, tail=tail)
    if body == text:
        return text
    return f"{body}\n<<distil elided, handle={_handle(text)}>>"


def _selfcheck() -> None:  # pragma: no cover — run via `python -m distil.skeleton`
    """Runnable check for the brace-skeleton parser path (ponytail: parser => one check)."""
    js = (
        "import x from 'y';\n"
        "function add(a, b) {\n"
        "  const s = a + b;  // a { in a comment does not count\n"
        '  const t = "a } string brace";\n'
        "  return s + t;\n"
        "}\n"
        "class Foo {\n"
        "  method() {\n"
        "    doA();\n"
        "    doB();\n"
        "    doC();\n"
        "  }\n"
        "}\n"
    )
    sk = generic_code_skeleton(js)
    assert sk is not None and len(sk) < len(js), "should fold braced code"
    assert "function add(a, b) {" in sk and "class Foo {" in sk and "method() {" in sk, (
        "signatures kept"
    )
    assert "doB();" not in sk, "3-line body elided"
    # brace inside string/comment must not unbalance the depth counter
    assert _brace_depths(js) is not None, "string/comment braces skipped"
    # unbalanced braces => bail, never corrupt
    assert generic_code_skeleton("func f() {\n  a();\n  b();\n") is None
    # no code hint / no braces => defer
    assert generic_code_skeleton("just prose, no braces here at all") is None
    # Python defers to the ast skeleton (generic returns None on hint-less {}, but
    # code_skeleton owns .py via smart_digest ordering)
    print("skeleton self-check: OK")


if __name__ == "__main__":  # pragma: no cover
    _selfcheck()
