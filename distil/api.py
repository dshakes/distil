"""Public library API — embed distil in your own agent.

The proxy (``distil proxy`` / ``distil wrap``) stays the zero-config path: it needs
no code change and sees the agent's *intent* alongside its output, which is what
makes query-aware keeps possible. This module is for the other case — you are
building the agent, you already hold the message list, and you want to compress it
in-process before the model call::

    from distil import compress_messages, expand_handle

    result = compress_messages(messages)
    print(f"{result.saved_pct:.1f}% smaller")
    response = client.messages.create(model=..., messages=result.messages)

    original = expand_handle(result.handles[0])   # byte-exact, any time, any process

Message shapes are duck-typed: plain ``{"role": ..., "content": ...}`` dicts
(OpenAI / Anthropic wire format) and objects exposing ``.role``/``.type`` and
``.content`` (LangChain ``BaseMessage``, pydantic models) both work, so this module
never imports a framework.

**What gets compressed.** Tool/function results get the reversible Tier-1 digest —
elided spans are replaced by a short stub carrying an 8-hex handle, and the original
bytes are persisted so :func:`expand_handle` recovers them exactly. User/system text gets
Tier-0 lossless transforms only. The model's own turns (``assistant``/``ai``) are
never rewritten. Non-string content (image blocks, tool-use structures) passes
through untouched.

**Reversibility is the contract.** Every digest is recoverable via :func:`expand_handle`
for as long as the restore store retains it (``DISTIL_RESTORE_TTL_DAYS``, default
14). Pass ``verbatim=True`` to disable Tier-1 digests entirely and take only the
in-context-lossless transforms — lower savings, nothing behind a handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CompressionResult", "compress_messages", "expand_handle"]


@dataclass(frozen=True)
class CompressionResult:
    """What :func:`compress_messages` returns.

    ``messages`` is a new list — the input is never mutated. Unchanged messages are
    passed through by identity, so ``result.messages[i] is messages[i]`` holds
    wherever nothing was compressed.
    """

    messages: list[Any]
    tokens_before: int
    tokens_after: int
    handles: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def saved_pct(self) -> float:
        """Percent smaller, 0.0 when there was nothing to measure.

        Can legitimately be ~0 on small payloads: savings come from *large* tool
        output, and a request that is mostly system prompt has nothing to fold.
        """
        if self.tokens_before <= 0:
            return 0.0
        return 100.0 * (self.tokens_before - self.tokens_after) / self.tokens_before


def _role(m: Any) -> str:
    if isinstance(m, dict):
        return str(m.get("role") or m.get("type") or "")
    return str(getattr(m, "role", "") or getattr(m, "type", "") or "")


def _content(m: Any) -> Any:
    if isinstance(m, dict):
        return m.get("content")
    return getattr(m, "content", None)


def _with_content(m: Any, new_text: str) -> Any:
    """Return a copy of *m* with replaced content, across message shapes."""
    if isinstance(m, dict):
        return {**m, "content": new_text}
    if hasattr(m, "model_copy"):  # pydantic v2
        return m.model_copy(update={"content": new_text})
    if hasattr(m, "copy"):  # pydantic v1 — dict.copy() takes no kwargs
        try:
            return m.copy(update={"content": new_text})
        except TypeError:
            pass
    return m  # unknown immutable shape — leave it rather than risk corruption


def compress_messages(
    messages: list[Any],
    *,
    verbatim: bool = False,
    tokenizer: str = "heuristic",
) -> CompressionResult:
    """Compress a message list in-process.

    :param messages: OpenAI/Anthropic-style dicts or duck-typed message objects.
    :param verbatim: Tier-0 lossless transforms only — no digest stubs, no handles.
        Use for flat-rate/subscription billing or out-of-distribution traffic where
        the model must reason over real bytes rather than recover them.
    :param tokenizer: ``"heuristic"`` (default, offline, no key) or ``"anthropic"``
        for billing-grade counts (requires the ``anthropic`` extra and credentials).
    :returns: a :class:`CompressionResult`. Never raises on unusual message shapes —
        anything it does not understand is passed through unchanged.
    """
    # Imported lazily: `import distil` must stay cheap (the CLI's --version path
    # imports this package), and the compression stack pulls in the adapter tree.
    from .adapters.anthropic import (
        RestoreStore,
        _compress_text_content,
        _compress_tool_result_text,
        _keep_tls,
    )
    from .tokenizer import resolve as _resolve_tokenizer

    tok = _resolve_tokenizer(tokenizer)

    def _count(m: Any) -> int:
        c = _content(m)
        return tok.count(c) if isinstance(c, str) else 0

    before = sum(_count(m) for m in messages)

    _keep_tls.fn = None
    try:
        store = RestoreStore()
        out: list[Any] = []
        for m in messages:
            content = _content(m)
            if not isinstance(content, str):
                out.append(m)  # image block, tool_use struct, None — pass through
                continue
            role = _role(m).lower()
            if role in ("assistant", "ai"):
                out.append(m)  # never rewrite the model's own words
                continue
            if role in ("tool", "function"):
                new_text = _compress_tool_result_text(content, store, verbatim)
            else:
                new_text = _compress_text_content(content, store, verbatim)
            out.append(m if new_text == content else _with_content(m, new_text))
    finally:
        _keep_tls.fn = None

    after = sum(_count(m) for m in out)
    return CompressionResult(
        messages=out,
        tokens_before=before,
        tokens_after=after,
        handles=sorted(store.handles),
    )


def expand_handle(handle: str) -> str | None:
    """Recover the original text behind an 8-hex digest *handle*.

    Reads the on-disk restore store, so it works across processes and restarts —
    the handle from a proxy session expands here, and vice versa. Returns ``None``
    for an unknown, expired, or malformed handle rather than raising.
    """
    from .mcp_server import load_restore

    return load_restore(handle)
