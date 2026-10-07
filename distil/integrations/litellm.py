"""LiteLLM integration — compress requests in-process, no proxy required.

Two ways to use distil with LiteLLM:

1. **Proxy (zero code):** point LiteLLM at ``distil proxy`` via ``api_base``.
2. **In-process (this module):** compress the messages right before the call::

       from distil.integrations import litellm as distil_litellm

       resp = distil_litellm.completion(
           model="claude-opus-4-8",
           messages=[...],
           distil_verbatim=True,   # optional; Tier-0 only, no digest stubs
       )

:func:`compress` is the pure, framework-free core (returns new completion kwargs
with the ``messages`` reversibly compressed); :func:`completion` / :func:`acompletion`
are thin wrappers that hand the compressed kwargs to the real ``litellm``.
"""

from __future__ import annotations

from typing import Any

from .litellm_hook import _digest_allowed, compress_request


def compress(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``litellm.completion`` kwargs with ``messages`` compressed.

    A ``distil_verbatim=True`` kwarg (consumed here, not forwarded) selects the
    verbatim, in-context-lossless mode (Tier-0 only, no digest). Non-list
    ``messages`` are returned untouched.

    ``litellm.completion`` takes OpenAI Chat-shaped messages whatever the provider, so
    this is the Chat Completions adapter — the hook's own path — not the Anthropic one,
    which read a list-shaped ``role:"tool"`` message as user text. And the digest is
    the hook's opt-in under the same policy: refused on a subscription session.
    """
    messages = kwargs.get("messages")
    if not isinstance(messages, list):
        return kwargs
    verbatim = bool(kwargs.get("distil_verbatim", False))
    new = {k: v for k, v in kwargs.items() if k != "distil_verbatim"}
    out = compress_request(new, "completion", digest=not verbatim and _digest_allowed())
    return out[0] if out is not None else new


def completion(**kwargs: Any) -> Any:
    """Drop-in for ``litellm.completion`` that compresses the request first."""
    import litellm  # lazy: optional dependency

    return litellm.completion(**compress(kwargs))


async def acompletion(**kwargs: Any) -> Any:
    """Async drop-in for ``litellm.acompletion`` that compresses the request first."""
    import litellm  # lazy: optional dependency

    return await litellm.acompletion(**compress(kwargs))
