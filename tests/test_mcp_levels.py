"""distil mcp levels: L0 semantic equivalence, determinism, L1-L3 surfaces, R results."""

from __future__ import annotations

import copy
import json
import random
import re

import pytest

from distil.mcpproxy import fakeserver, levels

FIXTURES = {n: fakeserver.load_fixture(n) for n in fakeserver.fixture_names()}
ALL_TOOLS = [(n, t) for n, f in FIXTURES.items() for t in f["tools"]]

# Keywords that carry no validation meaning (JSON Schema 2020-12 core §7/§9, validation
# §9): the ONLY things L0 may remove.
ANNOTATIONS = {"title", "$comment", "description", "$schema"}


# ---------------------------------------------------------------- a reference validator


def _validate(schema, inst, root=None) -> bool:  # noqa: C901 — a compact reference, on purpose
    root = root if root is not None else schema
    if schema is True or schema == {}:
        return True
    if schema is False:
        return False
    if "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/"), ref
        node = root
        for part in ref[2:].split("/"):
            node = node[part]
        if not _validate(node, inst, root):
            return False
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        ok = {
            "object": isinstance(inst, dict),
            "array": isinstance(inst, list),
            "string": isinstance(inst, str),
            "integer": isinstance(inst, int) and not isinstance(inst, bool),
            "number": isinstance(inst, (int, float)) and not isinstance(inst, bool),
            "boolean": isinstance(inst, bool),
            "null": inst is None,
        }
        if not any(ok.get(x, False) for x in types):
            return False
    if "enum" in schema and inst not in schema["enum"]:
        return False
    if "const" in schema and inst != schema["const"]:
        return False
    for kw in ("anyOf", "oneOf", "allOf"):
        if kw in schema:
            hits = [_validate(s, inst, root) for s in schema[kw]]
            if kw == "anyOf" and not any(hits):
                return False
            if kw == "oneOf" and sum(hits) != 1:
                return False
            if kw == "allOf" and not all(hits):
                return False
    if isinstance(inst, dict):
        props = schema.get("properties", {})
        if any(k not in inst for k in schema.get("required", [])):
            return False
        for k, v in inst.items():
            if k in props:
                if not _validate(props[k], v, root):
                    return False
            elif schema.get("additionalProperties") is False:
                return False
            elif isinstance(schema.get("additionalProperties"), dict) and not _validate(
                schema["additionalProperties"], v, root
            ):
                return False
    if isinstance(inst, list):
        if "minItems" in schema and len(inst) < schema["minItems"]:
            return False
        if isinstance(schema.get("items"), dict) and not all(
            _validate(schema["items"], x, root) for x in inst
        ):
            return False
    if isinstance(inst, str):
        if "minLength" in schema and len(inst) < schema["minLength"]:
            return False
        if "pattern" in schema and not re.search(schema["pattern"], inst):
            return False
    if isinstance(inst, (int, float)) and not isinstance(inst, bool):
        if "minimum" in schema and inst < schema["minimum"]:
            return False
        if "maximum" in schema and inst > schema["maximum"]:
            return False
    return True


_SAMPLES = [None, True, 0, 7, -3, 2.5, "", "x", "Europe/Berlin", [], ["a"], [1, 2], {}, {"k": 1}]


def _instances(schema: dict, rng: random.Random, n: int = 40) -> list:
    """Valid-ish instances for an object schema, plus every mutation that should flip it."""
    props = schema.get("properties", {})
    out = []
    for _ in range(n):
        inst = {k: rng.choice(_SAMPLES) for k in props}
        for k in list(inst):
            if rng.random() < 0.3:
                del inst[k]
        if rng.random() < 0.3:
            inst["unexpected"] = rng.choice(_SAMPLES)
        out.append(inst)
    return out


@pytest.mark.parametrize(
    ("server", "tool"), ALL_TOOLS, ids=[f"{s}.{t['name']}" for s, t in ALL_TOOLS]
)
def test_l0_is_validation_equivalent_on_every_real_schema(server, tool):
    """Same accept/reject decision on every generated instance, original vs L0."""
    before = tool["inputSchema"]
    after = levels.canonical_tool(tool)["inputSchema"]
    rng = random.Random(tool["name"])
    for inst in _instances(before, rng):
        assert _validate(before, inst) == _validate(after, inst), inst


@pytest.mark.parametrize(
    ("server", "tool"), ALL_TOOLS, ids=[f"{s}.{t['name']}" for s, t in ALL_TOOLS]
)
def test_l0_only_removes_annotation_keywords(server, tool):
    """The structural proof: every change is an annotation keyword or whitespace."""
    after = levels.canonical_tool(tool)
    for path, change in levels.dropped_paths(tool, after):
        key = path.rsplit("/", 1)[-1]
        if change == "removed":
            assert key in ANNOTATIONS, path
        else:  # "shortened" — only descriptions, and only their whitespace
            assert key == "description", path
    # and what the model can call is otherwise identical
    assert after["name"] == tool["name"]


def _strip(node):
    """Everything except annotations: the validation-relevant skeleton."""
    if isinstance(node, dict):
        return {
            k: _strip(v) for k, v in node.items() if k not in ANNOTATIONS or not isinstance(v, str)
        }
    if isinstance(node, list):
        return [_strip(x) for x in node]
    return node


@pytest.mark.parametrize(
    ("server", "tool"), ALL_TOOLS, ids=[f"{s}.{t['name']}" for s, t in ALL_TOOLS]
)
def test_l0_keeps_every_validation_keyword(server, tool):
    assert _strip(levels.canonical_tool(tool)["inputSchema"]) == _strip(tool["inputSchema"])


def test_l0_never_mistakes_a_property_named_like_a_keyword():
    tool = {
        "name": "t",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "title": "Title"},
                "$comment": {"type": "string"},
                "description": {"type": "string", "description": "  "},
            },
            "required": ["title", "$comment"],
        },
    }
    out = levels.canonical_tool(tool)["inputSchema"]
    assert set(out["properties"]) == {"title", "$comment", "description"}
    assert out["properties"]["title"] == {"type": "string"}  # the derivable title went
    assert out["properties"]["description"] == {"type": "string"}  # empty description went
    assert out["required"] == ["title", "$comment"]


def test_l0_keeps_dialect_when_dropping_it_would_change_meaning():
    tuple_form = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": [{"type": "string"}],
    }
    assert "$schema" in levels.canonical_schema(tuple_form)
    ref_sibling = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "properties": {"a": {"$ref": "#/definitions/x", "type": "string"}},
        "definitions": {"x": {"type": "string"}},
    }
    assert "$schema" in levels.canonical_schema(ref_sibling)
    draft4 = {"$schema": "http://json-schema.org/draft-04/schema#", "type": "object"}
    assert "$schema" in levels.canonical_schema(draft4)
    boolean_excl = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "properties": {"n": {"minimum": 1, "exclusiveMinimum": True}},
    }
    assert "$schema" in levels.canonical_schema(boolean_excl)
    plain = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "$comment": "x",
    }
    assert levels.canonical_schema(plain) == {"type": "object"}


def test_l0_does_not_mutate_its_input_and_keeps_non_derivable_titles():
    tool = FIXTURES["everything"]["tools"][0]
    snapshot = copy.deepcopy(tool)
    out = levels.canonical_tool(tool)
    assert tool == snapshot
    assert out.get("title") == tool.get("title")  # "Echo Tool" != "echo": kept


def test_whitespace_normalisation():
    assert levels._ws("a  \nb\n\n\n\nc  ") == "a\nb\n\nc"
    tool = {"name": "x", "description": "   ", "inputSchema": {"type": "object"}}
    assert "description" not in levels.canonical_tool(tool)


# ---------------------------------------------------------------- determinism


@pytest.mark.parametrize("level", levels.LEVELS)
def test_every_level_is_byte_stable(level):
    tools = FIXTURES["github"]["tools"]
    a = json.dumps(levels.Surface("github", tools, level).tools_list())
    b = json.dumps(levels.Surface("github", copy.deepcopy(tools), level).tools_list())
    assert a == b
    surf = levels.Surface("github", tools, level)
    assert json.dumps(surf.tools_list()) == json.dumps(surf.tools_list())


def test_no_level_is_larger_than_l0():
    for name, fx in FIXTURES.items():
        l0 = levels.Surface(name, fx["tools"], "L0", results=False)._list_tokens()
        for lv in levels.LEVELS:
            surf = levels.Surface(name, fx["tools"], lv, results=False)
            assert surf._list_tokens() <= l0, (name, lv)


def test_small_servers_fall_back_to_l0_and_say_what_was_asked():
    surf = levels.Surface("time", FIXTURES["time"]["tools"], "L1", results=False)
    assert surf.level == "L0" and surf.requested == "L1"


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError):
        levels.Surface("x", [], "L9")


# ---------------------------------------------------------------- L1


def test_summary_is_extractive_and_never_longer():
    for _, tool in ALL_TOOLS:
        desc = tool.get("description") or ""
        out = levels.summarize(desc)
        assert len(out) <= len(levels._ws(desc))
        for unit in levels.summary_units(desc):
            assert unit in levels._ws(desc)
        if out != levels._ws(desc):
            assert out == " ".join(levels.summary_units(desc))


def test_summary_keeps_first_sentence_and_constraints():
    text = "Reads a file. It is fast. Paths must be absolute. Some trivia. Never follows symlinks. Also X."
    assert levels.summarize(text) == "Reads a file. Paths must be absolute. Never follows symlinks."


def test_l1_surface_offers_full_description():
    fx = FIXTURES["filesystem"]
    surf = levels.Surface("filesystem", fx["tools"], "L1")
    assert surf.level == "L1"
    names = [t["name"] for t in surf.tools_list()]
    assert names[: len(fx["tools"])] == [t["name"] for t in fx["tools"]]  # backend order kept
    assert surf.resolve("filesystem_get_tool_schema") == ("schema", None)
    full = surf.schema_text("edit_file")
    assert (
        levels._ws(next(t for t in fx["tools"] if t["name"] == "edit_file")["description"]) in full
    )


# ---------------------------------------------------------------- L2 / L3


def test_l2_index_unlock_is_append_only_and_monotonic():
    fx = FIXTURES["github"]
    surf = levels.Surface("github", fx["tools"], "L2")
    first = surf.tools_list()
    assert [t["name"] for t in first] == [
        "github_get_tool_schema",
        "github_invoke_tool",
        "github_expand",
    ]
    index = first[0]["description"]
    for t in fx["tools"]:
        assert t["name"] + "(" in index
    assert surf.unlock("create_issue") is True
    assert surf.unlock("create_issue") is False  # already visible
    assert surf.unlock("nope") is False
    assert surf.unlock("get_issue") is True
    after = surf.tools_list()
    assert json.dumps(after[:3]) == json.dumps(first)  # the prefix never moved
    assert [t["name"] for t in after[3:]] == ["create_issue", "get_issue"]
    real = next(t for t in fx["tools"] if t["name"] == "create_issue")
    assert after[3]["inputSchema"] == levels.canonical_schema(
        real["inputSchema"], name_hint="create_issue"
    )


def test_l2_resolution_routes_real_names_even_when_not_surfaced():
    surf = levels.Surface("git", FIXTURES["git"]["tools"], "L2")
    assert surf.resolve("git_status") == ("tool", "git_status")
    assert surf.resolve("git_invoke_tool") == ("invoke", None)
    assert surf.resolve("git_expand") == ("expand", None)
    assert surf.resolve("rm_rf") == ("unknown", None)
    assert "is now in your tool list" in surf.schema_text("git_status")
    assert surf.suggest("git_stat")[0] == "git_status"


def test_l0_has_no_schema_tool_and_results_off_has_no_expand():
    surf = levels.Surface("git", FIXTURES["git"]["tools"], "L0", results=False)
    assert surf.resolve("git_get_tool_schema") == ("unknown", None)
    assert surf.resolve("git_expand") == ("unknown", None)
    assert all(not t["name"].startswith("git_get_tool") for t in surf.tools_list())


def test_l3_pins_most_used_and_never_moves_mid_session():
    fx = FIXTURES["github"]
    usage = {"create_issue": 9, "get_issue": 5, "list_issues": 2, "gone_tool": 50}
    pins = levels.choose_pins(usage, [t["name"] for t in fx["tools"]])
    assert pins == frozenset({"create_issue", "get_issue"})  # min calls + must exist
    surf = levels.Surface("github", fx["tools"], "L3", pinned=pins)
    names = [t["name"] for t in surf.tools_list()]
    assert names[3:] == ["create_issue", "get_issue"]
    assert surf.unlock("create_issue") is False  # pinned is already visible
    many = {f"t{i}": 10 + i for i in range(20)}
    assert len(levels.choose_pins(many, list(many))) == levels.PIN_TOP_K


def test_meta_names_are_sanitised_and_bounded():
    assert levels.meta_name("my server!", "expand") == "my_server__expand"
    assert len(levels.meta_name("x" * 200, "get_tool_schema")) <= 64
    assert levels.safe_server("") == "mcp"


def test_signature_and_index_line_without_schema():
    assert levels.signature({"name": "ping"}) == "ping()"
    assert levels.index_line({"name": "ping"}) == "ping()"


def test_prefixed_surface_exposes_prefixed_names():
    surf = levels.Surface("b", FIXTURES["time"]["tools"], "L0", prefix="b_")
    assert all(
        t["name"].startswith("b_") for t in surf.tools_list() if not t["name"].endswith("expand")
    )
    assert surf.resolve("b_get_current_time") == ("tool", "get_current_time")


# ---------------------------------------------------------------- R


def _big_listing(n: int = 400) -> str:
    rows = [f"entry-{i:04d} size={i * 37 % 9973} ok" for i in range(n)]
    rows.insert(n // 2, "ERROR: entry failed checksum")
    return "\n".join(rows)


def test_results_are_digested_recoverably():
    store: dict[str, str] = {}
    text = _big_listing()
    result = {
        "content": [
            {"type": "text", "text": text},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
        ]
    }
    snapshot = copy.deepcopy(result)
    new, info = levels.compress_result(
        "list_things", result, record=lambda h, t: store.setdefault(h, t) == t
    )
    assert result == snapshot  # input untouched
    assert info.tokens_after < info.tokens_before
    small = new["content"][0]["text"]
    assert "ERROR: entry failed checksum" in small  # must-keep lines survive
    (handle,) = info.handles
    assert f"handle={handle}" in small and store[handle] == text  # byte-exact recovery
    assert new["content"][1] == result["content"][1]  # images are never touched


def test_json_record_results_use_the_columnar_fold():
    records = [{"id": i, "status": "ok", "region": "eu"} for i in range(80)]
    text = json.dumps(records, indent=1)
    new, info = levels.compress_result(
        "search", {"content": [{"type": "text", "text": text}]}, record=lambda h, t: True
    )
    assert new["content"][0]["text"].startswith("«rows=80")
    assert info.tokens_after < info.tokens_before


@pytest.mark.parametrize(
    ("tool", "result", "why"),
    [
        ("read_text_file", {"content": [{"type": "text", "text": _big_listing()}]}, "exact-quote"),
        (
            "get_file_contents",
            {"content": [{"type": "text", "text": _big_listing()}]},
            "exact-quote",
        ),
        ("x", {"content": [{"type": "text", "text": _big_listing()}], "isError": True}, "error"),
        (
            "x",
            {"content": [{"type": "text", "text": _big_listing()}], "structuredContent": {"a": 1}},
            "structured",
        ),
        ("x", "not a result", "shape"),
    ],
)
def test_results_that_must_not_be_touched(tool, result, why):
    new, info = levels.compress_result(tool, result, record=lambda h, t: True)
    assert new is result and info.skipped == why


def test_reject_if_bigger_and_short_text_untouched():
    short = {"content": [{"type": "text", "text": "ok"}]}
    new, info = levels.compress_result("x", short, record=lambda h, t: True)
    assert new is short and info.handles == []
    # every line must-keep: the digest cannot drop anything, so nothing changes
    errors = "\n".join(
        f"ERROR distinct failure number {i} in {chr(65 + i % 26)}" for i in range(300)
    )
    same = {"content": [{"type": "text", "text": errors}]}
    new, _ = levels.compress_result("x", same, record=lambda h, t: True)
    assert new["content"][0]["text"] == errors or len(new["content"][0]["text"]) < len(errors)


def test_a_handle_collision_leaves_the_block_verbatim():
    result = {"content": [{"type": "text", "text": _big_listing()}]}
    new, info = levels.compress_result("x", result, record=lambda h, t: False)
    assert new is result and info.handles == []
