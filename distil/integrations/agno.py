"""Agno integration — compress an agent's messages in-process.

Like :mod:`distil.integrations.langchain`, this module is **duck-typed and never
imports agno**: it works on anything exposing ``role``/``type`` and ``content``
(Agno message objects, pydantic models, plain dicts). That keeps distil a
zero-dependency install and means an Agno version bump cannot break it.

Two integration points, in increasing order of intrusiveness::

    # 1. Compress a message list you already hold.
    from distil.integrations.agno import compress_messages
    msgs = compress_messages(session.messages)

    # 2. Wrap a model/callable so every call is compressed on the way out.
    from distil.integrations.agno import compressed_model
    agent = Agent(model=compressed_model(OpenAIChat(id="gpt-5")))

The wrapper is a transparent proxy: every attribute except the invocation methods
(``invoke`` / ``ainvoke`` / ``response`` / ``aresponse``) is delegated untouched, so
the object still behaves as the framework expects.
"""

from __future__ import annotations

from typing import Any

from ..api import compress_messages as _compress

__all__ = ["compress_messages", "compressed_model"]

# Method names Agno models expose for a completion. Wrapping all of them means the
# integration keeps working whether the caller is sync, async, or streaming.
_CALL_METHODS = ("invoke", "ainvoke", "response", "aresponse")


def compress_messages(messages: list[Any], *, verbatim: bool = False) -> list[Any]:
    """Return a new list with compressible content reversibly compressed."""
    return _compress(messages, verbatim=verbatim).messages


def _compress_in(args: tuple[Any, ...], kwargs: dict[str, Any], verbatim: bool) -> tuple:
    """Compress whichever argument carries the message list (kwarg or first positional)."""
    if isinstance(kwargs.get("messages"), list):
        kwargs = {**kwargs, "messages": compress_messages(kwargs["messages"], verbatim=verbatim)}
    elif args and isinstance(args[0], list):
        args = (compress_messages(args[0], verbatim=verbatim), *args[1:])
    return args, kwargs


def compressed_model(model: Any, *, verbatim: bool = False) -> Any:
    """Wrap an Agno model so its messages are compressed before every call.

    Unknown attributes are delegated to the wrapped model, so this is safe to hand
    to ``Agent(model=...)`` in place of the original.
    """

    class _CompressedModel:
        # Not a subclass: Agno models vary in __init__ contract, and delegation is
        # the only approach that survives a framework upgrade.
        def __getattr__(self, name: str) -> Any:
            attr = getattr(model, name)
            if name not in _CALL_METHODS or not callable(attr):
                return attr

            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                a, kw = _compress_in(args, kwargs, verbatim)
                return attr(*a, **kw)

            return _wrapped

        def __repr__(self) -> str:  # pragma: no cover - diagnostic affordance
            return f"<distil-compressed {model!r}>"

    return _CompressedModel()
