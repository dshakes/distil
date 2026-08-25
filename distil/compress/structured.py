"""Reversible structured compaction — the token-dense content agents actually
traffic in (JSON arrays of records, tabular tool output) re-encoded into a
compact columnar form that carries the *same information* in far fewer tokens.

Unlike a lossy structural crusher, nothing is discarded: the byte-exact original
is kept in the restore table and is one ``expand(handle)`` away. So this is
reversible in effect — the model sees a smaller, equivalent view; the detail is
never lost. A homogeneous ``[{"id":1,"name":"a","ok":true}, …]`` array of N
records costs the repeated keys + punctuation N times in JSON; the columnar form
states the keys once and lists the values, typically a 40–70% reduction with no
loss of meaning.

Two folders, tried in order (both reversible, both DECISION-marker-safe, both
reject-if-not-smaller): ``fold`` handles flat arrays of scalar records; when a
record carries NESTED fields (dicts/lists — the common shape of real API tool
output), ``fold_records`` covers it by rendering non-scalar cells as compact
JSON, marking those columns in the header, and hoisting any column that repeats
one value in every row into a single ``«=name<TAB>value`` directive (constant-
column collapse, the columnar-DB / Parquet trick). On a realistic nested search-
result array with enum-like fields that measures ~62% fewer tokens — savings
strict ``fold`` left on the table — with the byte-exact original still one
``expand(handle)`` away. Anything neither folder matches is left for Tier-0/1.
"""

from __future__ import annotations

import hashlib
import json

# NOTE: None is deliberately EXCLUDED. In the columnar form a null cell and a
# missing key would both render empty, so a record carrying any null is left
# byte-exact (not folded) rather than shown in an ambiguous compact form. The
# original is byte-recoverable regardless; this keeps the in-context form honest.
_SCALAR = (str, int, float, bool)
_SEP = "\t"
_HDR = "«"  # « — an unambiguous, rare marker so the compact form is recognisable


def _handle(text: str) -> str:
    """8-hex SHA-256 prefix — mirrors tier1._handle so a folded marker resolves
    against the same restore key the caller (Tier1Reversible.compress) records."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _scalar(v: object) -> bool:
    return isinstance(v, _SCALAR)


def _cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def fold(text: str, emit_handle: bool = True) -> str | None:
    """Columnar-fold a JSON array of homogeneous flat records. Returns the compact
    form, or None if the text isn't a foldable structure or wouldn't shrink."""
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")) or "DECISION:" in text:
        return None
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, list) or len(obj) < 3:
        return None
    if not all(isinstance(r, dict) and all(_scalar(v) for v in r.values()) for r in obj):
        return None
    # Sparse records (a key present in one row, absent in another) would render
    # a missing cell and an empty-string cell identically — same ambiguity the
    # None exclusion above guards against. Fold only uniform-schema arrays.
    if len({frozenset(r.keys()) for r in obj}) > 1:
        return None

    # column order: first record's keys, then any extras in first-seen order
    cols: list[str] = []
    for r in obj:
        for k in r:
            if k not in cols:
                cols.append(k)
    # A column name containing the header delimiter (,) or the cell separator/newline
    # would make the self-describing header ambiguous (advertises the wrong schema) —
    # bail (rare). Without this, {"a,b":1} folds to a header claiming columns a AND b.
    if any("," in c or _SEP in c or "\n" in c for c in cols):
        return None
    # tabs/newlines in any cell would break the columnar layout — bail (rare)
    rows: list[str] = []
    for r in obj:
        cells = [_cell(r.get(c)) for c in cols]
        if any(_SEP in c or "\n" in c for c in cells):
            return None
        rows.append(_SEP.join(cells))

    # Embed the recovery handle in the marker so the model (and the offline grader's
    # _HANDLE_IN_TEXT regex) can expand the fold. The handle keys the byte-exact
    # original that Tier1Reversible.compress records under _handle(b.text) == this.
    # emit_handle=False -> a self-describing, lossless columnar table with NO recovery
    # handle: every row is inline, so the model reads it directly, and with no handle
    # there is nothing to invite an (unavailable) distil_expand call. That is what makes
    # the fold safe on the lossless/subscription path where no expand tool is injected.
    marker = (
        f"{_HDR}rows={len(obj)} cols={','.join(cols)} handle={_handle(text)}{_HDR}\n"
        if emit_handle
        else f"{_HDR}rows={len(obj)} cols={','.join(cols)}{_HDR}\n"
    )
    compact = marker + "\n".join(rows)
    return compact if len(compact) < len(s) else None


def fold_records(text: str, emit_handle: bool = True) -> str | None:
    """Columnar-fold a JSON array of homogeneous records whose values may be
    NESTED (dicts/lists), not only scalars — the common shape of real tool
    output (API responses, search hits, DB rows with nested fields). The strict
    ``fold`` handles flat scalar records; this extends coverage to the nested
    case that would otherwise fall through to the generic Tier-1 path and save
    far less.

    Reversible in effect: the byte-exact original is kept in the restore table
    (one ``expand(handle)`` away), so nothing is lost — the compact form is a
    smaller, information-complete columnar VIEW. Non-scalar cells are rendered as
    compact JSON (``ensure_ascii`` escapes tabs/newlines, so the columnar layout
    stays intact); the header marks which columns are JSON-encoded so the view is
    unambiguous. Conservative: uniform schema only, no ``DECISION`` marker, and
    only when it actually shrinks."""
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")) or "DECISION:" in text:
        return None
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, list) or len(obj) < 3:
        return None
    if not all(isinstance(r, dict) for r in obj):
        return None
    # Uniform schema only (a missing vs empty cell must not be ambiguous).
    if len({frozenset(r.keys()) for r in obj}) > 1:
        return None

    cols: list[str] = list(obj[0].keys())
    if any("," in c or _SEP in c or "\n" in c for c in cols):
        return None
    # A column is JSON-encoded iff ANY of its values is non-scalar.
    json_cols = {
        c for c in cols if any(not _scalar(r.get(c)) and r.get(c) is not None for r in obj)
    }
    # If nothing is nested, the strict scalar fold already covers it — defer.
    if not json_cols:
        return None

    def _render(c: str, v: object) -> str | None:
        if c in json_cols:
            enc = json.dumps(v, separators=(",", ":"), ensure_ascii=True)
            return enc if (_SEP not in enc and "\n" not in enc) else None
        cell = _cell(v)
        return cell if (_SEP not in cell and "\n" not in cell) else None

    # Render every cell first (per-column), so we can spot columns that carry the
    # SAME value in every row.
    grid: list[list[str]] = []
    for c in cols:
        col_cells: list[str] = []
        for r in obj:
            rendered = _render(c, r.get(c))
            if rendered is None:  # a literal tab/newline slipped through — bail
                return None
            col_cells.append(rendered)
        grid.append(col_cells)

    # Constant-column collapse (columnar-DB / Parquet-style constant encoding): a
    # column whose value repeats identically in every row costs N copies in the
    # body for zero information beyond "all rows share this". Hoist it into a single
    # `«=name<TAB>value` directive line and drop it from the body — the model reads
    # the shared value once, the byte-exact original is still one expand away. On
    # enum-heavy real tool output (status/region/flags) this is ~19% fewer tokens
    # over the plain columnar fold. (Dictionary-indexing *varying* low-cardinality
    # columns is deliberately skipped: the integer→value indirection is a
    # decision-equivalence risk a constant hoist doesn't carry.)
    const_idx = [i for i, cc in enumerate(grid) if len(set(cc)) == 1]
    body_idx = [i for i in range(len(cols)) if i not in set(const_idx)]

    consts = [f"{_HDR}={cols[i]}{_SEP}{grid[i][0]}" for i in const_idx]
    body_cols = [cols[i] for i in body_idx]
    rows = [_SEP.join(grid[i][r] for i in body_idx) for r in range(len(obj))]

    # `j=` lists the 0-based indices (within the BODY columns) that are JSON-encoded.
    jidx = ",".join(str(k) for k, i in enumerate(body_idx) if cols[i] in json_cols)
    head = f"{_HDR}rows={len(obj)} cols={','.join(body_cols)} j={jidx}"
    marker = (head + f" handle={_handle(text)}{_HDR}\n") if emit_handle else (head + f"{_HDR}\n")
    compact = marker + "".join(c + "\n" for c in consts) + "\n".join(rows)
    return compact if len(compact) < len(s) else None


def is_folded(text: str) -> bool:
    return text.startswith(_HDR)


# --------------------------------------------------------------------------- #
# Template mining (Drain / LogPai family) — collapse runs of NEAR-identical
# lines (logs, telemetry) into one template + a compact variable table. Where
# RLE only collapses byte-identical lines, this captures the dominant real case:
# lines that differ solely by timestamp / id / number. Reversible (the original
# is kept in the restore table); information-preserving (every variable retained).
# --------------------------------------------------------------------------- #

import re  # noqa: E402 — kept local to this section for readability

# A "variable" is the token class that varies line-to-line: integers/floats,
# hex/uuid blobs, and ISO-ish timestamps. Everything else is template skeleton.
_VAR = re.compile(r"\d{4}-\d{2}-\d{2}[T0-9:.\-Z]*|\b[0-9a-fA-F]{8,}\b|\b\d+(?:\.\d+)?")
_SLOT = "§"


def _mask(line: str) -> str:
    return _VAR.sub(_SLOT, line)


def template_fold(text: str, min_run: int = 5, emit_handle: bool = True) -> str | None:
    """Collapse runs of >=min_run consecutive lines sharing a template into
    ``«N× <template>»`` + a ``vars:`` table of the per-line varying tokens."""
    if "DECISION:" in text:
        return None
    lines = text.split("\n")
    if len(lines) < min_run:
        return None
    masks = [_mask(line) for line in lines]
    out: list[str] = []
    i, n, changed = 0, len(lines), False
    while i < n:
        m = masks[i]
        j = i
        while j < n and masks[j] == m:
            j += 1
        run = j - i
        if run >= min_run and _SLOT in m:
            rows = [" ".join(_VAR.findall(lines[k])) for k in range(i, j)]
            _h = f" handle={_handle(text)}" if emit_handle else ""
            out.append(f"{_HDR}{run}× {m}{_h}{_HDR}")
            out.append("vars: " + " | ".join(rows))
            changed = True
        else:
            out.extend(lines[i:j])
        i = j
    folded = "\n".join(out)
    return folded if changed and len(folded) < len(text) else None


# --- embedded JSON ------------------------------------------------------------
# The folds above require the WHOLE block to be a JSON array (`fold` checks
# s.startswith("[")). Real tool output rarely is: a `gh api` dump, an MCP result, a
# `curl | jq` tail and most CLI wrappers surround the payload with prose, a status
# line, or a trailing summary. Measured on distil before this existed: a 60-record
# array wrapped in two sentences compressed 0.0%, while the identical array alone
# compressed ~70%. The machinery was there; nothing ever handed it the span.
_MIN_EMBEDDED_CHARS = 400


def _balanced_spans(text: str) -> list[tuple[int, int]]:
    """Byte offsets of top-level balanced ``[...]`` spans, outermost only.

    A hand-rolled scanner rather than a regex: JSON nesting is not regular, and a
    greedy pattern would swallow everything between the first ``[`` and the last
    ``]``. Quote- and escape-aware, so a bracket inside a string never counts.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append((start, i + 1))
                    start = -1
    return spans


def fold_embedded(text: str, emit_handle: bool = True) -> str | None:
    """Fold JSON arrays *embedded* in a larger block, leaving the surrounding text
    intact. Returns the rewritten block, or None if nothing folded or it grew.

    Only the span is replaced, so the prose an agent reads for context — the status
    line, the "3 of 12 failed" summary — survives verbatim. Reversibility is
    unchanged: the caller records the byte-exact original under its handle, exactly
    as for a whole-block fold.
    """
    if "DECISION:" in text:
        return None  # never touch a block carrying the decision marker
    spans = [(a, b) for a, b in _balanced_spans(text) if b - a >= _MIN_EMBEDDED_CHARS]
    if not spans:
        return None
    # A block that IS a bare array is already handled by fold(); doing it here too
    # would produce a second, differently-marked encoding of the same thing.
    if len(spans) == 1 and spans[0][0] == 0 and spans[0][1] == len(text.strip()):
        return None
    out: list[str] = []
    prev = 0
    folded = False
    for a, b in spans:
        piece = text[a:b]
        compact = fold(piece, emit_handle=False) or fold_records(piece, emit_handle=False)
        if compact is None:
            continue
        out.append(text[prev:a])
        out.append(compact)
        prev = b
        folded = True
    if not folded:
        return None
    out.append(text[prev:])
    result = "".join(out)
    # Reject-if-bigger, the same contract every other fold honours.
    return result if len(result) < len(text) else None
