"""LlamaIndex integration — compress retrieved context and outgoing LLM calls.

Like :mod:`distil.integrations.autogen`, this module is **duck-typed and never
imports llama_index**: it works on anything exposing the shapes described below, so
distil stays a zero-dependency install and a LlamaIndex version bump cannot break it.

API shapes verified against ``docs.llamaindex.ai`` (stable API reference,
``llama_index.core.postprocessor.types.BaseNodePostprocessor`` and
``llama_index.core.base.llms.base.BaseLLM``) and the ``run-llama/llama_index``
source on GitHub (``llama-index-core``, main branch) on 2026-09-06:

* A node postprocessor is any object with ``postprocess_nodes(nodes, query_bundle=None,
  query_str=None)`` / ``apostprocess_nodes(...)`` returning ``List[NodeWithScore]`` —
  callers (``RetrieverQueryEngine`` and friends) call these two methods directly, with
  no ``isinstance`` check, so a plain duck-typed object slots into
  ``node_postprocessors=[...]`` exactly like a real ``BaseNodePostprocessor``.
* ``NodeWithScore.node`` is a (usually) mutable ``TextNode`` with a plain ``.text: str``
  field — retrieved context is exactly the "large recoverable blob" shape the reversible
  Tier-1 digest exists for, so it is compressed the same way tool output is elsewhere in
  this package.
* ``FunctionTool.from_defaults(fn=..., async_fn=...)`` wraps a plain callable (sync or
  async) for use by LlamaIndex agents — wrapping that callable is the seam for
  compressing a tool's return value before the agent ever sees it.
* An ``llm=`` argument, unlike a node postprocessor, IS type-checked: ``resolve_llm()``
  asserts ``isinstance(llm, LLM)`` and Pydantic components declare ``llm: LLM``. That is
  why ``DistilLLM`` re-types rather than wraps — see its docstring (re-measured against
  llama-index-core 0.14.24 on 2026-09-08).
* ``BaseLLM.chat/achat/stream_chat/astream_chat`` take ``messages: Sequence[ChatMessage]``;
  ``.complete/.acomplete/.stream_complete/.astream_complete`` take a plain string
  ``prompt``. ``ChatMessage.content`` is a property backed by ``.blocks`` (a list of
  content blocks) — its setter only accepts a single ``TextBlock``, so a multi-block
  (e.g. multimodal) message is left untouched here.

Three integration points, in increasing order of intrusiveness::

    # 1. Compress retrieved nodes before they reach the LLM's context window.
    from distil.integrations.llamaindex import DistilNodePostprocessor

    engine = index.as_query_engine(node_postprocessors=[DistilNodePostprocessor()])

    # 2. Compress a tool's return value before an agent sees it.
    from distil.integrations.llamaindex import compressing_tool

    tool = FunctionTool.from_defaults(fn=compressing_tool(get_weather))

    # 3. Wrap an LLM so every outgoing chat/completion call is compressed.
    from distil.integrations.llamaindex import DistilLLM

    llm = DistilLLM(OpenAI(model="gpt-5"))

Restore handles from any of the three land in the same on-disk store the proxy and
the other integrations use, so a handle minted here is expandable anywhere.
"""

from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable
from typing import Any

from ..api import compress_messages as _compress

__all__ = [
    "DistilNodePostprocessor",
    "compress_tool_result",
    "compressing_tool",
    "DistilLLM",
]


def _compress_text(text: str, *, role: str, verbatim: bool) -> str:
    """Compress one string by routing it through the public API as a 1-message list."""
    out = _compress([{"role": role, "content": text}], verbatim=verbatim).messages
    new = out[0].get("content") if isinstance(out[0], dict) else None
    return new if isinstance(new, str) else text


def compress_tool_result(text: str, *, verbatim: bool = False) -> str:
    """Compress one tool-output string via the reversible Tier-1 digest."""
    return _compress_text(text, role="tool", verbatim=verbatim)


def compressing_tool(func: Callable[..., Any], *, verbatim: bool = False) -> Callable[..., Any]:
    """Wrap a plain callable so its string return value is compressed.

    Works whether *func* is sync or async — ``FunctionTool.from_defaults`` accepts
    either as ``fn=``/``async_fn=``, and the wrapper preserves that (an async *func*
    gets an async wrapper, so ``inspect.iscoroutinefunction`` still sees through it).

    ``FunctionTool`` builds its JSON schema from *func*'s signature, so the wrapper
    carries ``functools.wraps`` (name/doc/annotations/``__wrapped__``) AND an explicit
    ``__signature__``. Non-string returns pass through untouched.
    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def _awrapped(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)
            return (
                compress_tool_result(result, verbatim=verbatim)
                if isinstance(result, str)
                else result
            )

        _awrapped.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
        return _awrapped

    @functools.wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        return (
            compress_tool_result(result, verbatim=verbatim) if isinstance(result, str) else result
        )

    _wrapped.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    return _wrapped


class DistilNodePostprocessor:
    """Duck-typed ``BaseNodePostprocessor`` that compresses retrieved node text.

    Drop into ``node_postprocessors=[...]`` on any query engine or retriever. Only
    ``TextNode``-shaped nodes (a plain mutable ``.text: str``) are touched; anything
    else (image nodes, an unfamiliar node type) passes through unchanged.
    """

    def __init__(self, *, verbatim: bool = False) -> None:
        self._verbatim = verbatim

    def _compress_one(self, node_with_score: Any) -> Any:
        node = getattr(node_with_score, "node", None)
        text = getattr(node, "text", None)
        if node is None or not isinstance(text, str):
            return node_with_score
        new_text = compress_tool_result(text, verbatim=self._verbatim)
        if new_text != text:
            node.text = new_text
        return node_with_score

    def postprocess_nodes(
        self, nodes: list[Any], query_bundle: Any = None, query_str: Any = None
    ) -> list[Any]:
        return [self._compress_one(n) for n in nodes]

    async def apostprocess_nodes(
        self, nodes: list[Any], query_bundle: Any = None, query_str: Any = None
    ) -> list[Any]:
        return self.postprocess_nodes(nodes, query_bundle=query_bundle, query_str=query_str)

    def class_name(self) -> str:  # matches BaseComponent.class_name(), harmless if unused
        return "DistilNodePostprocessor"

    def __repr__(self) -> str:  # pragma: no cover - diagnostic affordance
        return "<DistilNodePostprocessor>"


def _role_str(role: Any) -> str:
    return str(getattr(role, "value", role) or "").lower()


def _compress_chat_message(m: Any, *, verbatim: bool) -> None:
    """Compress one ``ChatMessage`` in place, skipping what isn't safely rewritable."""
    role = _role_str(getattr(m, "role", ""))
    if role == "assistant":
        return  # never rewrite the model's own words
    content = getattr(m, "content", None)
    if not isinstance(content, str) or not content:
        return  # no blocks, or a multimodal/unknown block shape
    new_text = _compress_text(content, role=role or "user", verbatim=verbatim)
    if new_text == content:
        return
    try:
        m.content = new_text
    except ValueError:
        pass  # multi-block message — the .content setter refuses, leave it untouched


def compress_chat_messages(messages: list[Any], *, verbatim: bool = False) -> list[Any]:
    """Compress a ``ChatMessage`` list in place and return it.

    ``ChatMessage`` is mutated directly rather than copied: it has no framework-wide
    ``model_copy``-safe reconstruction path once ``.blocks`` is involved, and mutating
    ``.content`` is exactly the API LlamaIndex itself provides for this.
    """
    for m in messages:
        _compress_chat_message(m, verbatim=verbatim)
    return messages


# Method names a LlamaIndex LLM exposes for a completion call.
_CHAT_METHODS = ("chat", "achat", "stream_chat", "astream_chat")
_COMPLETE_METHODS = ("complete", "acomplete", "stream_complete", "astream_complete")
_CALL_METHODS = _CHAT_METHODS + _COMPLETE_METHODS


def _compress_in(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any], verbatim: bool
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Compress whichever argument carries the messages/prompt (kwarg or first positional)."""
    if name in _CHAT_METHODS:
        if isinstance(kwargs.get("messages"), list):
            kwargs = {
                **kwargs,
                "messages": compress_chat_messages(kwargs["messages"], verbatim=verbatim),
            }
        elif args and isinstance(args[0], list):
            args = (compress_chat_messages(args[0], verbatim=verbatim), *args[1:])
        return args, kwargs
    # _COMPLETE_METHODS: a plain string prompt, positional or as `prompt=`.
    if isinstance(kwargs.get("prompt"), str):
        kwargs = {
            **kwargs,
            "prompt": _compress_text(kwargs["prompt"], role="user", verbatim=verbatim),
        }
    elif args and isinstance(args[0], str):
        args = (_compress_text(args[0], role="user", verbatim=verbatim), *args[1:])
    return args, kwargs


class _DelegatingLLM:
    """Fallback: delegate every attribute to the wrapped LLM, compressing the
    chat/completion calls on the way through.

    Reached only when the LLM's class can't be subclassed or its instance
    can't be re-typed (see ``DistilLLM``). Correct for anything duck-typed,
    but it is NOT an instance of the wrapped class, so a caller that checks
    the type rejects it.
    """

    def __init__(self, llm: Any, *, verbatim: bool = False) -> None:
        self._llm = llm
        self._verbatim = verbatim

    def __getattr__(self, name: str) -> Any:
        # Only reached for names absent from this instance's own __dict__, so
        # self._llm / self._verbatim above never recurse through here.
        attr = getattr(self._llm, name)
        if name not in _CALL_METHODS or not callable(attr):
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            a, kw = _compress_in(name, args, kwargs, self._verbatim)
            return attr(*a, **kw)

        return _wrapped

    def __repr__(self) -> str:
        return f"<distil-compressed {self._llm!r}>"


def _compressing_override(name: str, inner: Callable[..., Any], verbatim: bool) -> Any:
    """One method for the subclass below: compress the outgoing messages/prompt,
    then call the ORIGINAL instance's bound method — never ``self``'s, which is
    this override, so nothing re-enters."""
    if inspect.iscoroutinefunction(inner):

        async def _amethod(self: Any, *args: Any, **kwargs: Any) -> Any:
            a, kw = _compress_in(name, args, kwargs, verbatim)
            return await inner(*a, **kw)

        return functools.wraps(inner)(_amethod)

    def _method(self: Any, *args: Any, **kwargs: Any) -> Any:
        a, kw = _compress_in(name, args, kwargs, verbatim)
        return inner(*a, **kw)

    return functools.wraps(inner)(_method)


def DistilLLM(llm: Any, *, verbatim: bool = False) -> Any:  # noqa: N802 — called like the class it replaces
    """Wrap a LlamaIndex ``LLM`` so outgoing chat/completion calls are compressed::

        llm = DistilLLM(OpenAI(model="gpt-5"))
        query_engine = index.as_query_engine(llm=llm)

    Returns *llm* re-typed as a transparent subclass of its own class, rather
    than a wrapper object around it. Delegation alone is not enough, because
    LlamaIndex validates an ``llm=`` argument by TYPE — measured against
    llama-index-core 0.14.24 on 2026-09-08:

    * ``resolve_llm()``, which both ``index.as_query_engine(llm=...)`` and
      ``Settings.llm = ...`` go through, ends in ``assert isinstance(llm, LLM)``
      (``llama_index/core/llms/utils.py``);
    * every Pydantic component with an ``llm: LLM`` field (``FunctionAgent``
      among them) rejects a non-instance outright — *Input should be a valid
      dictionary or instance of LLM*.

    Registering as a virtual subclass does not help: Pydantic disables
    ``register()``-based ``isinstance`` support and warns that it does.
    Subclassing whatever class we were handed satisfies both checks for any
    LLM implementation, first-party or not, and still imports nothing.

    A shallow copy of *llm* carries its state into the subclass, and the
    compressing overrides call the ORIGINAL instance's bound methods, so what
    runs is the wrapped model's own behaviour. Every other attribute
    (``metadata``, ``callback_manager``, ``class_name``, ...) is the copy's,
    untouched. Streaming methods return the real generator unmodified — only
    the outgoing messages/prompt are compressed, never the model's reply.

    Anything whose class can't be subclassed, or whose instance can't be
    re-typed, falls back to plain delegation (``_DelegatingLLM``): a
    duck-typed object keeps working, at the cost of failing a type check.
    """
    try:
        ns: dict[str, Any] = {"__repr__": lambda self: f"<distil-compressed {llm!r}>"}
        for name in _CALL_METHODS:
            inner = getattr(llm, name, None)
            if callable(inner):
                ns[name] = _compressing_override(name, inner, verbatim)
        # ponytail: a throwaway subclass per wrapped instance — an app wraps its
        # LLM once, so caching by (class, verbatim) would buy nothing.
        subclass = type(f"Distil{type(llm).__name__}", (type(llm),), ns)
        retyped = copy.copy(llm)
        retyped.__class__ = subclass
        return retyped
    except Exception:  # noqa: BLE001 — any class we can't re-type: stay duck-typed
        return _DelegatingLLM(llm, verbatim=verbatim)
