"""Microsoft AutoGen integration — compress tool output and model-client input.

Like :mod:`distil.integrations.agno`, this module is **duck-typed and never
imports autogen**: it works on anything exposing the shapes described below, so
distil stays a zero-dependency install and an AutoGen version bump cannot break it.

API shapes verified against ``microsoft.github.io/autogen`` (stable docs,
``autogen_core.models`` / ``autogen_core.tools``, autogen-core 0.4+/0.7+) on
2026-09-04:

* ``ChatCompletionClient.create(messages, *, tools=..., ...)`` and
  ``create_stream(messages, *, ...)`` take ``messages: Sequence[LLMMessage]``
  where ``LLMMessage = SystemMessage | UserMessage | AssistantMessage |
  FunctionExecutionResultMessage`` (all pydantic ``BaseModel``, discriminated by
  a ``.type`` field).
* ``SystemMessage.content`` and ``UserMessage.content`` are (usually) plain
  strings; ``AssistantMessage.content`` is the model's own text or tool calls —
  never rewritten here, matching every other integration in this package.
* ``FunctionExecutionResultMessage.content`` is a ``list[FunctionExecutionResult]``,
  each with a ``.content: str`` field — this is where a tool's output lands, and
  the reversible Tier-1 digest applies to each item's string.
* ``FunctionTool(func, description=...)`` wraps a plain callable (sync or async);
  wrapping *that* callable is the seam for compressing a tool's return value
  before it is ever packed into a ``FunctionExecutionResult``.

Two integration points, in increasing order of intrusiveness::

    # 1. Compress one tool's output as it is produced.
    from distil.integrations.autogen import compress_tool_result

    async def get_weather(city: str) -> str:
        return compress_tool_result(f"... huge forecast for {city} ...")

    tool = FunctionTool(get_weather, description="Get the weather for a city")

    # 2. Wrap a model client so every outgoing call is compressed.
    from distil.integrations.autogen import DistilModelClient

    client = DistilModelClient(OpenAIChatCompletionClient(model="gpt-4o"))
    agent = AssistantAgent("assistant", model_client=client)

Restore handles from either path land in the same on-disk store the proxy and
the other integrations use, so a handle minted here is expandable anywhere.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from ..api import compress_messages as _compress

__all__ = ["compress_messages", "compress_tool_result", "compressing_tool", "DistilModelClient"]


def _compress_text(text: str, *, role: str, verbatim: bool) -> str:
    """Compress one string by routing it through the public API as a 1-message list."""
    out = _compress([{"role": role, "content": text}], verbatim=verbatim).messages
    new = out[0].get("content") if isinstance(out[0], dict) else None
    return new if isinstance(new, str) else text


def compress_tool_result(text: str, *, verbatim: bool = False) -> str:
    """Compress one tool-output string via the reversible Tier-1 digest.

    This is the string that ends up as a ``FunctionExecutionResult.content`` —
    call it from inside a tool function, or use :func:`compressing_tool` to wrap
    the function itself.
    """
    return _compress_text(text, role="tool", verbatim=verbatim)


def compressing_tool(func: Callable[..., Any], *, verbatim: bool = False) -> Callable[..., Any]:
    """Wrap a ``FunctionTool`` callable so its string return value is compressed.

    Works whether *func* is sync or async — ``FunctionTool`` supports both, and
    the wrapper preserves that (an async *func* gets an async wrapper, so
    ``inspect.iscoroutinefunction`` still sees through it correctly)::

        tool = FunctionTool(compressing_tool(get_weather), description="...")

    Non-string returns (the function's own contract may return something else
    entirely) pass through untouched.

    ``FunctionTool`` builds its JSON schema from *func*'s signature and type
    annotations, not from ``*args, **kwargs`` — so the wrapper carries
    ``functools.wraps`` (name/doc/annotations/``__wrapped__``) AND an explicit
    ``__signature__``, for whichever of the two a schema builder reads.
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


def _with_field(obj: Any, **updates: Any) -> Any:
    """Return a copy of *obj* with the given fields replaced — pydantic v1/v2 shapes."""
    if hasattr(obj, "model_copy"):  # pydantic v2 (current autogen-core)
        return obj.model_copy(update=updates)
    if hasattr(obj, "copy"):  # pydantic v1
        try:
            return obj.copy(update=updates)
        except TypeError:
            pass
    return obj  # unknown immutable shape — leave it rather than risk corruption


def compress_messages(messages: list[Any], *, verbatim: bool = False) -> list[Any]:
    """Return a new list of AutoGen ``LLMMessage`` objects with content compressed.

    ``FunctionExecutionResultMessage`` items get the reversible Tier-1 digest (one
    per ``FunctionExecutionResult.content`` string); ``SystemMessage``/``UserMessage``
    string content gets Tier-0 lossless; ``AssistantMessage`` — the model's own
    words — is never rewritten. Non-string/unknown content passes through untouched.
    """
    out: list[Any] = []
    for m in messages:
        t = str(getattr(m, "type", "") or "")
        if t == "AssistantMessage":
            out.append(m)  # never rewrite the model's own words
            continue
        if t == "FunctionExecutionResultMessage":
            results = getattr(m, "content", None)
            if not isinstance(results, list):
                out.append(m)
                continue
            changed = False
            new_results = []
            for r in results:
                text = getattr(r, "content", None)
                if not isinstance(text, str):
                    new_results.append(r)
                    continue
                new_text = compress_tool_result(text, verbatim=verbatim)
                changed = changed or new_text != text
                new_results.append(r if new_text == text else _with_field(r, content=new_text))
            out.append(_with_field(m, content=new_results) if changed else m)
            continue
        content = getattr(m, "content", None)
        if not isinstance(content, str):
            out.append(m)  # UserMessage image-block list, or an unknown shape
            continue
        role = "system" if t == "SystemMessage" else "user"
        new_text = _compress_text(content, role=role, verbatim=verbatim)
        out.append(m if new_text == content else _with_field(m, content=new_text))
    return out


# Method names a ChatCompletionClient exposes for a completion call.
_CALL_METHODS = ("create", "create_stream")


def _compress_in(args: tuple[Any, ...], kwargs: dict[str, Any], verbatim: bool) -> tuple:
    """Compress whichever argument carries the message list (kwarg or first positional)."""
    if isinstance(kwargs.get("messages"), list):
        kwargs = {**kwargs, "messages": compress_messages(kwargs["messages"], verbatim=verbatim)}
    elif args and isinstance(args[0], list):
        args = (compress_messages(args[0], verbatim=verbatim), *args[1:])
    return args, kwargs


class DistilModelClient:
    """Wrap an AutoGen ``ChatCompletionClient`` so outgoing messages are compressed.

    Duck-typed against the shape ``create(messages, ...)`` / ``create_stream(messages,
    ...)``; every other attribute (``model_info``, ``capabilities``, ``count_tokens``,
    ``close``, ...) is delegated untouched, so this is safe to hand anywhere the real
    client is expected::

        client = DistilModelClient(OpenAIChatCompletionClient(model="gpt-4o"))
        agent = AssistantAgent("assistant", model_client=client)

    ``create`` stays a plain (non-async) wrapper around the real coroutine function:
    calling it returns the same coroutine/async-generator the real client would have
    returned, so ``await``/``async for`` on the result work exactly as the framework
    expects.
    """

    def __init__(self, client: Any, *, verbatim: bool = False) -> None:
        self._client = client
        self._verbatim = verbatim

    def __getattr__(self, name: str) -> Any:
        # Only reached for names absent from this instance's own __dict__, so
        # self._client / self._verbatim above never recurse through here.
        attr = getattr(self._client, name)
        if name not in _CALL_METHODS or not callable(attr):
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            a, kw = _compress_in(args, kwargs, self._verbatim)
            return attr(*a, **kw)

        return _wrapped

    def __repr__(self) -> str:  # pragma: no cover - diagnostic affordance
        return f"<distil-compressed {self._client!r}>"
