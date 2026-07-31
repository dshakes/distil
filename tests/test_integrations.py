"""In-process LiteLLM + LangChain integration helpers (framework-free core)."""

from __future__ import annotations

from distil.integrations import langchain as dlc
from distil.integrations import litellm as dll

BIG = "\n".join(f"row {i}: value_{i} status=ok detail=lorem" for i in range(40))


# --- LiteLLM --------------------------------------------------------------- #


def test_litellm_compress_compresses_messages():
    # Trailing turns keep the tool message out of the recency-exempt window (the
    # adapter keeps the most recent turns verbatim) so it still digests.
    kwargs = {
        "model": "x",
        "messages": [
            {"role": "tool", "content": BIG},
            {"role": "user", "content": "next"},
            {"role": "user", "content": "next"},
        ],
    }
    out = dll.compress(kwargs)
    assert out["messages"][0]["content"] != BIG
    assert len(out["messages"][0]["content"]) < len(BIG)
    assert out["model"] == "x"


def test_litellm_compress_strips_distil_flag():
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "distil_verbatim": True}
    out = dll.compress(kwargs)
    assert "distil_verbatim" not in out


def test_litellm_verbatim_does_not_digest():
    kwargs = {"messages": [{"role": "tool", "content": BIG}], "distil_verbatim": True}
    out = dll.compress(kwargs)
    assert "<< +" not in out["messages"][0]["content"]  # no digest stub


def test_litellm_non_list_messages_untouched():
    kwargs = {"messages": "nope"}
    assert dll.compress(kwargs) == kwargs


# --- LangChain (duck-typed) ------------------------------------------------ #


class _Msg:
    """Minimal stand-in for a pydantic-v2 LangChain message."""

    def __init__(self, type_, content):
        self.type = type_
        self.content = content

    def model_copy(self, update):
        return _Msg(self.type, update.get("content", self.content))


def test_langchain_tool_message_digested_dict():
    msgs = [{"type": "tool", "content": BIG}]
    out = dlc.compress_messages(msgs)
    assert out[0]["content"] != BIG and len(out[0]["content"]) < len(BIG)


def test_langchain_ai_message_never_rewritten():
    minifiable = '{"a":   1,    "b": 2}'
    out = dlc.compress_messages([{"type": "ai", "content": minifiable}])
    assert out[0]["content"] == minifiable


def test_langchain_object_messages_via_model_copy():
    out = dlc.compress_messages([_Msg("tool", BIG)])
    assert isinstance(out[0], _Msg)
    assert out[0].content != BIG and len(out[0].content) < len(BIG)


def test_langchain_verbatim():
    out = dlc.compress_messages([{"type": "tool", "content": BIG}], verbatim=True)
    assert "<< +" not in out[0]["content"]


def test_langchain_non_string_content_untouched():
    blocks = [{"type": "text", "text": "x"}]
    out = dlc.compress_messages([{"type": "human", "content": blocks}])
    assert out[0]["content"] is blocks


# --- LangGraph hook -------------------------------------------------------- #


def test_langgraph_compress_state_dict():
    from distil.integrations.langgraph import compress_state

    state = {"messages": [{"type": "tool", "content": BIG}], "other": 42}
    out = compress_state(state)
    assert out["other"] == 42  # untouched
    assert out["messages"][0]["content"] != BIG  # compressed
    assert len(out["messages"][0]["content"]) < len(BIG)


def test_langgraph_pre_model_hook_returns_only_messages():
    from distil.integrations.langgraph import pre_model_hook

    hook = pre_model_hook(verbatim=True)
    upd = hook({"messages": [{"type": "human", "content": "hi"}], "x": 1})
    assert set(upd.keys()) == {"messages"}  # merges back only messages


def test_langgraph_state_without_messages_untouched():
    from distil.integrations.langgraph import compress_state

    assert compress_state({"foo": "bar"}) == {"foo": "bar"}
    assert pre_model_hook_empty()


def pre_model_hook_empty():
    from distil.integrations.langgraph import pre_model_hook

    return pre_model_hook()({"foo": "bar"}) == {}


# ---------------------------------------------------------------------------
# The langchain-distil wrapper contract.
#
# `langchain-distil` is published separately and listed in LangChain's own community
# middleware integrations, but it is a thin wrapper: it imports these three symbols
# from this repo and re-exports them. Rename or re-signature any of them and the
# published package breaks at import time, for users who installed it on LangChain's
# recommendation — with nothing in this repo going red. Same failure class as the
# server.json registry entry and the npm badge: a promise nothing verifies.
# ---------------------------------------------------------------------------


def test_langchain_distil_imports_still_resolve() -> None:
    """The exact import lines in langchain_distil/__init__.py."""
    from distil.integrations.langchain import compress_messages
    from distil.integrations.langgraph import compress_state, pre_model_hook

    assert callable(compress_messages)
    assert callable(compress_state)
    assert callable(pre_model_hook)


def test_langchain_distil_call_signatures_are_stable() -> None:
    """The wrapper calls these with keyword arguments; losing one breaks it."""
    import inspect

    from distil.integrations.langchain import compress_messages
    from distil.integrations.langgraph import compress_state, pre_model_hook

    assert "verbatim" in inspect.signature(compress_messages).parameters
    for fn in (compress_state, pre_model_hook):
        params = inspect.signature(fn).parameters
        assert "verbatim" in params and "key" in params, fn.__name__


def test_langchain_distil_wrapper_behaviour() -> None:
    """Reproduce what `as_runnable()` does — compress a message list — without
    requiring langchain-core, which the wrapper imports lazily for exactly this
    reason."""
    from distil.integrations.langchain import compress_messages

    class _Msg:
        def __init__(self, content: str, type_: str) -> None:
            self.content = content
            self.type = type_

        def model_copy(self, update: dict) -> "_Msg":
            return _Msg(update.get("content", self.content), self.type)

    big = "\n".join(f"2026-07-30 INFO worker-{i} latency_ms={i} ok" for i in range(300))
    out = compress_messages([_Msg("check the logs", "human"), _Msg(big, "tool")])
    assert len(out) == 2
    assert len(out[-1].content) < len(big), "tool message should have been digested"
    assert out[0].content == "check the logs", "human message must not be rewritten"
