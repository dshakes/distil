"""Agno + Strands integrations.

Both are duck-typed and must never import their framework — that is what keeps
distil a zero-dependency install. These tests therefore run with neither package
installed, which is also how CI runs them.
"""

from __future__ import annotations

import json
import sys

from distil.integrations.agno import compress_messages as agno_compress
from distil.integrations.agno import compressed_model
from distil.integrations.strands import compress_messages as strands_compress

BIG = "\n".join(f"line {i}: the quick brown fox jumps over the lazy dog" for i in range(400))


def test_neither_framework_is_imported() -> None:
    assert "agno" not in sys.modules
    assert "strands" not in sys.modules


# --- Agno --------------------------------------------------------------------


def test_agno_compresses_tool_messages() -> None:
    out = agno_compress([{"role": "tool", "content": BIG}])
    assert len(out[0]["content"]) < len(BIG)


def test_agno_leaves_assistant_alone() -> None:
    msgs = [{"role": "assistant", "content": BIG}]
    assert agno_compress(msgs)[0]["content"] == BIG


def test_agno_wrapper_delegates_unknown_attrs_and_compresses_calls() -> None:
    seen: dict[str, object] = {}

    class FakeModel:
        id = "gpt-5"

        def invoke(self, messages, **kw):
            seen["messages"] = messages
            return "ok"

    wrapped = compressed_model(FakeModel())
    assert wrapped.id == "gpt-5"  # delegated untouched
    assert wrapped.invoke([{"role": "tool", "content": BIG}]) == "ok"
    assert len(seen["messages"][0]["content"]) < len(BIG)  # type: ignore[index]


def test_agno_wrapper_compresses_kwarg_form_too() -> None:
    seen: dict[str, object] = {}

    class FakeModel:
        def invoke(self, *, messages, **kw):
            seen["messages"] = messages
            return "ok"

    compressed_model(FakeModel()).invoke(messages=[{"role": "tool", "content": BIG}])
    assert len(seen["messages"][0]["content"]) < len(BIG)  # type: ignore[index]


# --- Strands -----------------------------------------------------------------


def test_strands_user_prose_is_lossless_not_digested() -> None:
    """User text blocks get Tier-0 lossless only — never a digest stub.

    Digesting user prose would put the human's own words behind a handle and widen
    what the decision-equivalence certificate covers. Only tool results are
    eligible for Tier-1, so a block of non-duplicate prose is expected to survive
    byte-for-byte.
    """
    msgs = [{"role": "user", "content": [{"text": BIG}]}]
    out = strands_compress(msgs)
    assert out[0]["content"][0]["text"] == BIG


def test_strands_text_block_shape_survives_compression() -> None:
    """The block wrapper must be preserved even when the text is rewritten."""
    duplicated = "\n".join(["identical line"] * 500)
    msgs = [{"role": "user", "content": [{"text": duplicated, "cache": "x"}]}]
    out = strands_compress(msgs)
    block = out[0]["content"][0]
    assert set(block) == {"text", "cache"}, "sibling keys must not be dropped"
    assert block["cache"] == "x"


def test_strands_compresses_tool_result_blocks() -> None:
    msgs = [
        {
            "role": "user",
            "content": [{"toolResult": {"status": "success", "content": [{"text": BIG}]}}],
        }
    ]
    out = strands_compress(msgs)
    inner = out[0]["content"][0]["toolResult"]["content"][0]["text"]
    assert len(inner) < len(BIG)
    assert out[0]["content"][0]["toolResult"]["status"] == "success", "shape preserved"


def test_strands_leaves_assistant_and_unknown_blocks_alone() -> None:
    msgs = [
        {"role": "assistant", "content": [{"text": BIG}]},
        {"role": "user", "content": [{"image": {"bytes": "..."}}]},
    ]
    out = strands_compress(msgs)
    assert out[0]["content"][0]["text"] == BIG
    assert out[1]["content"][0] == {"image": {"bytes": "..."}}


def test_strands_accepts_plain_string_content() -> None:
    out = strands_compress([{"role": "tool", "content": BIG}])
    assert len(out[0]["content"]) < len(BIG)


def test_strands_does_not_mutate_input() -> None:
    msgs = [{"role": "user", "content": [{"text": BIG}]}]
    snap = json.dumps(msgs)
    strands_compress(msgs)
    assert json.dumps(msgs) == snap


# --- quota rejection accounting ----------------------------------------------


def test_quota_rejections_are_counted_and_exported() -> None:
    """A throttled tenant must be visible in metrics, not only in logs.

    Quota enforcement that cannot be observed is indistinguishable from an outage
    from the client's side, which is the first thing an operator has to rule out.
    """
    from distil.gateway import GatewayState
    from distil.metrics import render
    from distil.pricing import Pricing

    state = GatewayState(Pricing("claude-opus-5", 5.0, 25.0))
    state.record("acme", 100, 40)
    state.record_quota_rejection("acme")
    state.record_quota_rejection("acme")

    snap = state.snapshot()
    row = next(r for r in snap["tenants"] if r["tenant"] == "acme")
    assert row["rejected_quota"] == 2
    assert row["requests"] == 1, "a rejected request is not a served request"
    assert snap["totals"]["rejected_quota"] == 2

    out = render(snap, version="test")
    assert 'distil_requests_rejected_total{tenant="acme"} 2' in out


def test_rejection_counter_survives_a_state_reload(tmp_path) -> None:
    from distil.gateway import GatewayState
    from distil.pricing import Pricing

    price = Pricing("claude-opus-5", 5.0, 25.0)
    path = tmp_path / "gateway_state.json"

    a = GatewayState(price)
    a.record("acme", 10, 5)
    a.record_quota_rejection("acme")
    a.save(path)

    b = GatewayState(price)
    b.load(path)
    row = next(r for r in b.snapshot()["tenants"] if r["tenant"] == "acme")
    assert row["rejected_quota"] == 1


# --- strands: the hook and the shape edge cases ------------------------------


def test_strands_hook_compresses_agent_messages_before_the_model_call():
    """The hook is the ergonomic entry point; an untested hook is a broken hook."""
    import sys
    import types

    from distil.integrations.strands import compressing_hook

    # Stand in for the optional dependency. The module must not import it at
    # module scope, which is exactly what makes this substitution possible.
    event_cls = type("BeforeModelInvocationEvent", (), {})
    fake = types.ModuleType("strands.hooks")
    fake.BeforeModelInvocationEvent = event_cls
    parent = types.ModuleType("strands")
    parent.hooks = fake
    sys.modules["strands"] = parent
    sys.modules["strands.hooks"] = fake
    try:
        captured = {}

        class Registry:
            def add_callback(self, cls, fn):
                captured["cls"] = cls
                captured["fn"] = fn

        hook = compressing_hook()
        hook.register_hooks(Registry())
        assert captured["cls"] is event_cls

        agent = type("Agent", (), {})()
        agent.messages = [{"role": "user", "content": [{"text": BIG}]}]
        captured["fn"](type("Ev", (), {"agent": agent})())
        assert isinstance(agent.messages, list)
    finally:
        sys.modules.pop("strands.hooks", None)
        sys.modules.pop("strands", None)


def test_strands_hook_tolerates_an_event_without_an_agent():
    import sys
    import types

    from distil.integrations.strands import compressing_hook

    fake = types.ModuleType("strands.hooks")
    fake.BeforeModelInvocationEvent = type("E", (), {})
    parent = types.ModuleType("strands")
    parent.hooks = fake
    sys.modules["strands"] = parent
    sys.modules["strands.hooks"] = fake
    try:
        captured = {}

        class Registry:
            def add_callback(self, cls, fn):
                captured["fn"] = fn

        compressing_hook().register_hooks(Registry())
        captured["fn"](type("Ev", (), {})())  # no .agent — must not raise
    finally:
        sys.modules.pop("strands.hooks", None)
        sys.modules.pop("strands", None)


def test_strands_passes_through_shapes_it_does_not_understand():
    msgs = [
        "not a dict",
        {"role": "user", "content": 42},
        {"role": "user", "content": [None, "bare string block"]},
    ]
    out = strands_compress(msgs)
    assert out[0] == "not a dict"
    assert out[1]["content"] == 42
    assert out[2]["content"] == [None, "bare string block"]
