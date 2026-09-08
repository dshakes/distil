"""LlamaIndex integration.

Duck-typed/framework-free — this test file runs with llama_index not installed,
which is also how CI runs it.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from typing import Any

from distil.integrations.llamaindex import (
    DistilLLM,
    DistilNodePostprocessor,
    compress_tool_result,
    compressing_tool,
)

BIG = "\n".join(f"line {i}: the quick brown fox jumps over the lazy dog" for i in range(400))
# Tool-result content gets the reversible Tier-1 digest (shrinks any large blob).
# Plain user/system chat content only gets Tier-0 lossless folds, which need
# actual redundancy (an exact-duplicate run) to have anything to collapse.
REPETITIVE = "\n".join("same repeated line here" for _ in range(400))


def test_llama_index_is_not_imported() -> None:
    assert "llama_index" not in sys.modules


# --- fakes (duck-typed pydantic-v2-shaped node/message objects) -------------


class _FakeTextNode:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeNodeWithScore:
    def __init__(self, text: str, score: float = 1.0) -> None:
        self.node = _FakeTextNode(text)
        self.score = score


class _FakeImageNode:
    """No `.text` field — must pass through untouched."""


class _FakeChatMessage:
    def __init__(self, role: str, content: str | None, *, multi_block: bool = False) -> None:
        self.role = role
        self._content = content
        self._multi_block = multi_block

    @property
    def content(self) -> str | None:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        if self._multi_block:
            raise ValueError("ChatMessage contains multiple blocks, use 'ChatMessage.blocks'.")
        self._content = value


class _FakeRole:
    """A MessageRole-style enum member: `.value` holds the plain string."""

    def __init__(self, value: str) -> None:
        self.value = value


# --- DistilNodePostprocessor -------------------------------------------------


def test_node_postprocessor_shrinks_big_node_text() -> None:
    pp = DistilNodePostprocessor()
    nodes = [_FakeNodeWithScore(BIG)]
    out = pp.postprocess_nodes(nodes)
    assert len(out[0].node.text) < len(BIG)
    assert out[0] is nodes[0]  # mutated in place, same NodeWithScore object


def test_node_postprocessor_leaves_short_text_alone() -> None:
    pp = DistilNodePostprocessor()
    nodes = [_FakeNodeWithScore("short")]
    out = pp.postprocess_nodes(nodes)
    assert out[0].node.text == "short"


def test_node_postprocessor_passes_through_nodes_without_text() -> None:
    pp = DistilNodePostprocessor()

    class _NoNode:
        node = _FakeImageNode()

    n = _NoNode()
    out = pp.postprocess_nodes([n])
    assert out[0] is n


def test_node_postprocessor_accepts_query_bundle_and_query_str() -> None:
    pp = DistilNodePostprocessor()
    out = pp.postprocess_nodes([_FakeNodeWithScore(BIG)], query_bundle=object(), query_str="q")
    assert len(out[0].node.text) < len(BIG)


def test_node_postprocessor_async_matches_sync() -> None:
    pp = DistilNodePostprocessor()
    out = asyncio.run(pp.apostprocess_nodes([_FakeNodeWithScore(BIG)]))
    assert len(out[0].node.text) < len(BIG)


def test_node_postprocessor_class_name() -> None:
    assert DistilNodePostprocessor().class_name() == "DistilNodePostprocessor"


# --- compress_tool_result / compressing_tool --------------------------------


def test_compress_tool_result_shrinks_big_output() -> None:
    assert len(compress_tool_result(BIG)) < len(BIG)


def test_compressing_tool_wraps_sync_function() -> None:
    def get_data() -> str:
        return BIG

    wrapped = compressing_tool(get_data)
    assert len(wrapped()) < len(BIG)
    assert wrapped.__name__ == "get_data"


def test_compressing_tool_wraps_async_function() -> None:
    async def get_data() -> str:
        return BIG

    result = asyncio.run(compressing_tool(get_data)())
    assert len(result) < len(BIG)


def test_compressing_tool_passes_through_non_string_return() -> None:
    def get_data() -> dict:
        return {"a": 1}

    assert compressing_tool(get_data)() == {"a": 1}


def test_compressing_tool_preserves_signature_for_schema_builders() -> None:
    """FunctionTool.from_defaults(fn=...) builds its JSON schema from the callable's
    signature/annotations, not from *args/**kwargs — a wrapper that loses either
    yields a broken tool schema, which looks like LlamaIndex is broken, not distil."""

    def get_weather(city: str, units: str = "metric") -> str:
        """Get the weather for a city."""
        return BIG

    wrapped = compressing_tool(get_weather)
    assert inspect.signature(wrapped) == inspect.signature(get_weather)
    params = inspect.signature(wrapped).parameters
    assert list(params) == ["city", "units"]
    assert params["units"].default == "metric"


# --- DistilLLM ---------------------------------------------------------------


class _FakeLLM:
    metadata = {"model_name": "fake"}

    def __init__(self) -> None:
        self.seen_messages: Any = None
        self.seen_prompt: Any = None

    def chat(self, messages: list[Any], **kw: Any) -> str:
        self.seen_messages = messages
        return "ok"

    async def achat(self, messages: list[Any], **kw: Any) -> str:
        self.seen_messages = messages
        return "ok"

    def complete(self, prompt: str, **kw: Any) -> str:
        self.seen_prompt = prompt
        return "ok"

    def stream_chat(self, messages: list[Any], **kw: Any) -> Any:
        self.seen_messages = messages

        def _gen() -> Any:
            yield "chunk"

        return _gen()


def test_distil_llm_delegates_unknown_attrs() -> None:
    assert DistilLLM(_FakeLLM()).metadata == {"model_name": "fake"}


def test_distil_llm_compresses_chat_messages_positional() -> None:
    fake = _FakeLLM()
    wrapped = DistilLLM(fake)
    msg = _FakeChatMessage("user", REPETITIVE)
    assert wrapped.chat([msg]) == "ok"
    assert len(fake.seen_messages[0].content) < len(REPETITIVE)


def test_distil_llm_compresses_chat_messages_kwarg() -> None:
    fake = _FakeLLM()
    wrapped = DistilLLM(fake)
    msg = _FakeChatMessage("user", REPETITIVE)
    wrapped.chat(messages=[msg])
    assert len(fake.seen_messages[0].content) < len(REPETITIVE)


def test_distil_llm_never_rewrites_assistant_messages() -> None:
    fake = _FakeLLM()
    wrapped = DistilLLM(fake)
    msg = _FakeChatMessage(_FakeRole("assistant"), BIG)
    wrapped.chat([msg])
    assert fake.seen_messages[0].content == BIG


def test_distil_llm_leaves_multi_block_messages_untouched() -> None:
    fake = _FakeLLM()
    wrapped = DistilLLM(fake)
    # REPETITIVE (not BIG) so a compressed value actually differs from the
    # original — otherwise the no-op short-circuit would never exercise the
    # setter's ValueError at all, and this test would pass for the wrong reason.
    msg = _FakeChatMessage("user", REPETITIVE, multi_block=True)
    wrapped.chat([msg])  # must not raise even though the setter would refuse
    assert fake.seen_messages[0].content == REPETITIVE


def test_distil_llm_leaves_non_string_content_untouched() -> None:
    fake = _FakeLLM()
    wrapped = DistilLLM(fake)
    msg = _FakeChatMessage("user", None)
    wrapped.chat([msg])
    assert fake.seen_messages[0].content is None


def test_distil_llm_compresses_complete_prompt() -> None:
    fake = _FakeLLM()
    wrapped = DistilLLM(fake)
    assert wrapped.complete(REPETITIVE) == "ok"
    assert len(fake.seen_prompt) < len(REPETITIVE)


def test_distil_llm_stream_chat_returns_real_generator() -> None:
    fake = _FakeLLM()
    wrapped = DistilLLM(fake)
    msg = _FakeChatMessage("user", REPETITIVE)
    chunks = list(wrapped.stream_chat([msg]))
    assert chunks == ["chunk"]
    assert len(fake.seen_messages[0].content) < len(REPETITIVE)


def test_distil_llm_repr() -> None:
    assert "distil-compressed" in repr(DistilLLM(_FakeLLM()))


# --- DistilLLM: the type check LlamaIndex actually performs ------------------
#
# LlamaIndex validates an `llm=` argument by TYPE, in two places that both
# reduce to isinstance against the LLM base class: `resolve_llm()` — which
# `index.as_query_engine(llm=...)` and `Settings.llm = ...` both route
# through — ends in `assert isinstance(llm, LLM)`
# (llama_index/core/llms/utils.py), and every Pydantic component with an
# `llm: LLM` field (FunctionAgent among them) rejects a non-instance with
# "Input should be a valid dictionary or instance of LLM". A delegating
# wrapper object fails both, and registering as a virtual subclass does not
# help — Pydantic disables register()-based isinstance support and warns so.
# Measured against llama-index-core 0.14.24 on 2026-09-08.
#
# Reproduced below WITHOUT the dependency: llama_index isn't installed in CI
# (see test_llama_index_is_not_imported), so an importorskip'd version of this
# would silently never run — the failure mode this repo treats as no test at
# all. The isinstance property is what both mechanisms come down to.


class _FakeLLMBase:
    """Stands in for ``llama_index.core.llms.LLM``."""


class _FakeVendorLLM(_FakeLLMBase):
    """Stands in for a concrete LLM (``OpenAI``, ``Anthropic``, ...)."""

    def __init__(self) -> None:
        self.seen_messages: Any = None

    def chat(self, messages: list[Any], **kw: Any) -> str:
        self.seen_messages = messages
        return "ok"


def _resolve_llm(llm: Any) -> Any:
    """What llama_index/core/llms/utils.py does before handing the LLM on."""
    assert isinstance(llm, _FakeLLMBase)
    return llm


class _FakeQueryEngine:
    """A component with a declared ``llm`` field, built the way a user builds
    one: ``index.as_query_engine(llm=DistilLLM(...))``."""

    def __init__(self, *, llm: Any) -> None:
        self.llm = _resolve_llm(llm)


def test_distil_llm_passes_the_type_check_a_component_performs() -> None:
    vendor = _FakeVendorLLM()
    wrapped = DistilLLM(vendor)

    engine = _FakeQueryEngine(llm=wrapped)  # the assert inside is the real gate
    assert isinstance(wrapped, _FakeLLMBase), "resolve_llm()'s isinstance assert would fail"
    assert isinstance(wrapped, _FakeVendorLLM), "a pydantic `llm: LLM` field would reject this"
    assert engine.llm is wrapped

    # ...and it is still a compressing wrapper, not just the right type.
    engine.llm.chat([_FakeChatMessage("user", REPETITIVE)])
    assert len(vendor.seen_messages[0].content) < len(REPETITIVE)


def test_distil_llm_does_not_re_type_the_callers_own_object() -> None:
    """The caller keeps handing their own LLM to other code; wrapping must not
    mutate it into a compressing subclass behind their back."""
    vendor = _FakeVendorLLM()
    wrapped = DistilLLM(vendor)

    assert wrapped is not vendor
    assert type(vendor) is _FakeVendorLLM


class _SlottedLLM:
    """No ``__dict__``, so its instances cannot be re-typed into a subclass
    that has one — the fallback's reason to exist."""

    __slots__ = ("seen_messages",)

    def __init__(self) -> None:
        self.seen_messages: Any = None

    def chat(self, messages: list[Any], **kw: Any) -> str:
        self.seen_messages = messages
        return "ok"


def test_distil_llm_falls_back_to_delegation_for_a_class_it_cannot_re_type() -> None:
    """A duck-typed object we can't subclass-and-retype must keep working by
    delegation rather than raising — it just won't pass a type check."""
    llm = _SlottedLLM()
    wrapped = DistilLLM(llm)

    assert not isinstance(wrapped, _SlottedLLM), "this is the delegating fallback, by construction"
    assert wrapped.chat([_FakeChatMessage("user", REPETITIVE)]) == "ok"
    assert len(llm.seen_messages[0].content) < len(REPETITIVE)
