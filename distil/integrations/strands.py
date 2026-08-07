"""Strands Agents integration — compress an agent's messages in-process.

Duck-typed and framework-free, exactly like :mod:`distil.integrations.langchain`:
this module never imports ``strands``, so distil stays a zero-dependency install
and a Strands release cannot break it.

Strands messages are ``{"role": ..., "content": [blocks]}`` where each block is a
dict — commonly ``{"text": "..."}`` for prose and ``{"toolResult": {...}}`` for tool
output. That block shape is handled here natively, in addition to the plain-string
``content`` form the other integrations accept::

    from distil.integrations.strands import compress_messages

    agent.messages = compress_messages(agent.messages)

Or hook it into the agent loop so it happens automatically::

    from distil.integrations.strands import compressing_hook

    agent = Agent(model=..., hooks=[compressing_hook()])

Tool results get the reversible Tier-1 digest; prose gets Tier-0 lossless; the
model's own turns are never rewritten.
"""

from __future__ import annotations

from typing import Any

from ..api import compress_messages as _compress

__all__ = ["compress_messages", "compressing_hook"]


def _compress_text(text: str, *, role: str, verbatim: bool) -> str:
    """Compress one string by routing it through the public API as a 1-message list."""
    out = _compress([{"role": role, "content": text}], verbatim=verbatim).messages
    new = out[0].get("content") if isinstance(out[0], dict) else None
    return new if isinstance(new, str) else text


def _compress_block(block: Any, *, verbatim: bool) -> Any:
    """Compress a single Strands content block, preserving its shape."""
    if not isinstance(block, dict):
        return block
    if isinstance(block.get("text"), str):
        new = _compress_text(block["text"], role="user", verbatim=verbatim)
        return block if new == block["text"] else {**block, "text": new}
    tr = block.get("toolResult")
    if isinstance(tr, dict) and isinstance(tr.get("content"), list):
        inner = [
            (
                {**b, "text": _compress_text(b["text"], role="tool", verbatim=verbatim)}
                if isinstance(b, dict) and isinstance(b.get("text"), str)
                else b
            )
            for b in tr["content"]
        ]
        return {**block, "toolResult": {**tr, "content": inner}}
    return block


def compress_messages(messages: list[Any], *, verbatim: bool = False) -> list[Any]:
    """Return a new message list with Strands content blocks compressed.

    Handles both the block form (``content`` is a list of dicts) and the plain
    string form. Assistant turns pass through untouched.
    """
    out: list[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        if str(m.get("role", "")).lower() == "assistant":
            out.append(m)  # never rewrite the model's own words
            continue
        content = m.get("content")
        if isinstance(content, list):
            blocks = [_compress_block(b, verbatim=verbatim) for b in content]
            out.append(m if blocks == content else {**m, "content": blocks})
        elif isinstance(content, str):
            new = _compress_text(content, role=str(m.get("role", "user")), verbatim=verbatim)
            out.append(m if new == content else {**m, "content": new})
        else:
            out.append(m)
    return out


def compressing_hook(*, verbatim: bool = False) -> Any:
    """Return a Strands hook that compresses ``agent.messages`` before each model call.

    Duck-typed against the Strands hook protocol (an object with
    ``register_hooks(registry)``); it imports the event class lazily from
    ``strands`` only when actually registered, so importing this module never
    requires Strands to be installed.
    """

    class _DistilCompressionHook:
        def register_hooks(self, registry: Any, **_: Any) -> None:
            from strands.hooks import BeforeModelInvocationEvent  # local: optional dep

            def _on_before(event: Any) -> None:
                agent = getattr(event, "agent", None)
                if agent is None:
                    return  # no agent on the event — nothing to compress
                msgs = getattr(agent, "messages", None)
                if isinstance(msgs, list):
                    agent.messages = compress_messages(msgs, verbatim=verbatim)

            registry.add_callback(BeforeModelInvocationEvent, _on_before)

    return _DistilCompressionHook()
