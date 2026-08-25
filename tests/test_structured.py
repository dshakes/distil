"""Reversible structured compaction (columnar fold) — savings AND reversibility."""

from __future__ import annotations

import json

from distil.compress.structured import fold, is_folded, template_fold
from distil.compress.tier1 import Tier1Reversible
from distil.fidelity import verify_reversible
from distil.trajectory import Block, Kind, Stability

_ARRAY = json.dumps(
    [{"id": i, "name": f"svc-{i}", "status": "active", "ok": True} for i in range(20)], indent=2
)


def test_fold_columnarises_and_shrinks():
    out = fold(_ARRAY)
    assert out is not None and is_folded(out)
    assert len(out) < len(_ARRAY) * 0.7  # meaningful reduction
    # information preserved: every record's name still present in the compact form
    for i in range(20):
        assert f"svc-{i}" in out


def test_fold_skips_non_foldable():
    assert fold("just some prose, not json") is None
    assert fold(json.dumps({"a": 1})) is None  # not an array
    assert fold(json.dumps([1, 2, 3])) is None  # not records
    assert fold(json.dumps([{"a": 1}, {"a": 2}])) is None  # < 3 records
    # never fold a block carrying a decision signal
    assert fold('[{"a":1},{"a":2},{"a":3}]\nDECISION: act') is None


def test_fold_skips_nested_objects():
    nested = json.dumps([{"a": {"deep": 1}} for _ in range(5)])
    assert fold(nested) is None  # only flat scalar records fold


def test_template_fold_collapses_near_identical_lines():
    logs = "\n".join(
        f"2026-06-22T10:{i:02d}:00Z INFO request id=req-{1000 + i} handled in {i}ms"
        for i in range(20)
    )
    out = template_fold(logs)
    assert out is not None and is_folded(out)
    assert len(out) < len(logs) * 0.8  # template stated once → real reduction
    # every varying value is retained (information-preserving): ids 1009 & 1019
    assert "1009" in out and "1019" in out


def test_template_fold_skips_unique_and_decision_lines():
    assert template_fold("a\nb\nc\nd\ne") is None  # no shared template
    assert template_fold("x 1\nx 2\nx 3\nx 4\nx 5\nDECISION: act") is None


def test_tier1_fold_is_byte_reversible():
    b = Block(id="obs", kind=Kind.TOOL_OUTPUT, text=_ARRAY, stability=Stability.VOLATILE)
    result = Tier1Reversible().compress([b])
    assert is_folded(result.blocks[0].text)
    assert len(result.blocks[0].text) < len(_ARRAY)
    # the byte-exact original is recoverable from the restore table
    rep = verify_reversible([b], result)
    assert rep.lossless
    assert _ARRAY in set(result.restore.values())


# --------------------------------------------------------------------------- #
# Nested-record fold (fold_records) — extends coverage to tool outputs whose
# records carry nested fields, the case strict `fold` skips.
# --------------------------------------------------------------------------- #

from distil.compress.structured import fold_records  # noqa: E402

_NESTED = json.dumps(
    [
        {
            "id": i,
            "name": f"svc-{i}",
            "labels": {"tier": "gold", "region": "us-east"},
            "ports": [80, 443],
            "ok": True,
        }
        for i in range(20)
    ],
    indent=2,
)


def test_fold_records_folds_nested_and_shrinks():
    out = fold_records(_NESTED)
    assert out is not None and is_folded(out)
    assert len(out) < len(_NESTED) * 0.75  # keys+structure stated once → real reduction
    # labels/ports/ok are identical in every row → constant-collapse hoists them
    # into `«=` directive lines, leaving only the varying id,name in the body.
    assert out.split("\n")[0].startswith("«rows=20 cols=id,name ")
    assert "«=labels\t" in out and "«=ports\t" in out and "«=ok\ttrue" in out
    # every value is still preserved in the compact view (information-complete)
    assert "us-east" in out and "443" in out


def test_fold_records_json_column_stays_in_body_when_it_varies():
    # a nested column that VARIES row-to-row is not constant → stays in the body,
    # and the header marks its (body-relative) index in j=.
    varied = json.dumps([{"id": i, "meta": {"k": i}, "ok": True} for i in range(4)])
    out = fold_records(varied)
    assert out is not None and is_folded(out)
    head = out.split("\n")[0]
    assert "cols=id,meta" in head and " j=1" in head  # meta is body col 1, JSON-encoded
    assert "«=ok\ttrue" in out  # ok is constant → hoisted


def test_fold_records_defers_when_flat():
    # all-scalar records are the strict fold's job — fold_records returns None
    flat = json.dumps([{"a": 1, "b": "x"} for _ in range(5)])
    assert fold_records(flat) is None


def test_fold_records_safety_guards():
    # non-uniform schema, DECISION marker, and too-short arrays never fold
    assert fold_records(json.dumps([{"a": {"x": 1}}, {"b": {"y": 2}}, {"a": {"z": 3}}])) is None
    assert fold_records('[{"a":{"x":1}},{"a":{"y":2}},{"a":{"z":3}}] DECISION: act') is None
    assert fold_records(json.dumps([{"a": {"x": 1}}])) is None  # < 3 records
    # not an array / not JSON / a list with a non-dict element → bail
    assert fold_records('{"a": {"x": 1}}') is None  # object, not array
    assert fold_records("[not json") is None
    assert fold_records(json.dumps([{"a": {"x": 1}}, {"a": {"y": 2}}, 7])) is None  # non-dict row
    # a column name carrying the header delimiter would make the header ambiguous → bail
    assert fold_records(json.dumps([{"a,b": {"x": i}} for i in range(3)])) is None
    # a literal tab in a *scalar* column (alongside a nested column) breaks the layout → bail
    tabbed = [{"nested": {"x": i}, "note": "a\tb"} for i in range(3)]
    assert fold_records(json.dumps(tabbed)) is None


def test_fold_records_is_byte_reversible_through_tier1():
    b = Block(id="obs", kind=Kind.TOOL_OUTPUT, text=_NESTED, stability=Stability.VOLATILE)
    result = Tier1Reversible().compress([b])
    assert is_folded(result.blocks[0].text)
    assert len(result.blocks[0].text) < len(_NESTED)
    rep = verify_reversible([b], result)
    assert rep.lossless  # byte-exact original recoverable from the restore table
    assert _NESTED in set(result.restore.values())


def test_fold_records_lossless_path_has_no_handle():
    # emit_handle=False (subscription/lossless): a self-describing table, no handle
    out = fold_records(_NESTED, emit_handle=False)
    assert out is not None and "handle=" not in out and is_folded(out)


# --- embedded JSON ------------------------------------------------------------
def _records(n: int) -> str:
    import json as _json

    return _json.dumps(
        [{"id": i, "name": f"pod-{i}", "ready": True, "restarts": 0} for i in range(n)]
    )


def test_an_array_wrapped_in_prose_now_folds():
    """The shape every whole-block fold missed, and the one real tool output takes.

    `fold` requires the entire block to be a JSON array, but a `gh api` dump, an MCP
    result or a `curl | jq` tail arrives wrapped in a status line or a summary.
    Measured before this existed: 0.0% on an array with two sentences around it.
    """
    from distil.compress.structured import fold_embedded

    blob = "Fetched the deployment:\n\n" + _records(60) + "\n\nAll healthy."
    out = fold_embedded(blob)
    assert out is not None
    assert len(out) < len(blob)
    # The prose the agent reads for context must survive byte-for-byte.
    assert out.startswith("Fetched the deployment:")
    assert out.endswith("All healthy.")


def test_a_bare_array_is_left_to_the_whole_block_fold():
    """No double-encoding: a block that IS an array belongs to fold(), which is
    strictly better and differently marked."""
    from distil.compress.structured import fold_embedded

    assert fold_embedded(_records(40)) is None


def test_brackets_inside_strings_do_not_confuse_the_scanner():
    """JSON nesting is not regular — a regex would swallow from the first [ to the
    last ]. The scanner is quote- and escape-aware."""
    from distil.compress.structured import _balanced_spans

    text = 'log: "an [unclosed bracket" then ' + _records(5) + " done"
    spans = _balanced_spans(text)
    assert len(spans) == 1, "the bracket inside the string must not open a span"
    a, b = spans[0]
    assert text[a] == "[" and text[b - 1] == "]"


def test_a_decision_marker_block_is_never_touched():
    """The certificate's oracle marker must reach the grader intact."""
    from distil.compress.structured import fold_embedded

    assert fold_embedded("DECISION: run_tests\n" + _records(60)) is None


def test_it_refuses_when_the_result_would_not_shrink():
    """Reject-if-bigger, the contract every other fold honours."""
    from distil.compress.structured import fold_embedded

    assert fold_embedded("prefix " + '[{"a":1},{"a":2},{"a":3}]' + " suffix") is None


def test_embedded_fold_is_reachable_from_the_live_adapter():
    """The adapter keeps its own fold chain, so a transform wired only into tier1
    would compress in tests and 0% in production — that exact split once left HTML
    at 0.0% on real traffic."""
    from distil.adapters.anthropic import compress_messages

    blob = "$ gh api repos/x/y/issues\n" + _records(50) + "\nShowing 50 of 128."
    msgs = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": blob}],
        },
        {"role": "user", "content": "next"},
        {"role": "user", "content": "next"},
    ]
    out, _store = compress_messages(msgs, verbatim=False)
    got = out[1]["content"][0]["content"]
    assert len(got) < len(blob), "the adapter path must fold embedded arrays too"
    assert "gh api repos/x/y/issues" in got, "surrounding text must survive"
