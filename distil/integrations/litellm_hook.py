"""LiteLLM Proxy hook — compress requests before LiteLLM routes them.

LiteLLM's extension point for rewriting a request on the *proxy* is
``CustomLogger.async_pre_call_hook(user_api_key_dict, cache, data, call_type)``:
it runs once per request, before routing, and a returned ``dict`` replaces the
payload. This module is that hook, reusing the serving adapters so a request
compressed here is compressed exactly as it would be by ``distil proxy``::

    # litellm config.yaml
    litellm_settings:
      callbacks: distil.integrations.litellm_hook.proxy_handler_instance

or, to choose options in code::

    from distil.integrations.litellm_hook import DistilCompressionHook

    proxy_handler_instance = DistilCompressionHook(digest=False)

Contract:

* **Lossless-only by default** — Tier-0 transforms only (``verbatim=True``), so
  the model sees semantically identical content. ``digest=True`` (or
  ``DISTIL_LITELLM_DIGEST=1`` for the YAML form) opts into Tier-1 digests.
  A LiteLLM hook cannot inject the ``distil_expand`` tool, so a digest stub sent
  from here has no in-conversation recovery path; use the sidecar ``distil proxy``
  when you want digest mode. The opt-in warns once, and is refused (lossless-only,
  with a warning) on a subscription machine — the same policy the proxy applies.
* **Fail-open** — any error, unknown call type or unrecognized shape returns the
  original request untouched.
* **Content-free** — message text is never logged or stored by this module; only
  token counts reach the savings ledger (same rows ``distil proxy`` writes, in ``DISTIL_HOME``).
* **Optional dependency** — ``litellm`` is imported lazily, only when the hook
  class is first requested (``pip install 'distil-llm[litellm]'``).
  :class:`HookCore` holds all the logic and needs no litellm.

Covered ``call_type`` values: ``completion`` / ``acompletion`` (OpenAI Chat shape,
which LiteLLM uses for every provider) and ``anthropic_messages`` (the native
``/v1/messages`` route), plus ``responses`` / ``aresponses`` (the Responses API's
``input`` array). Embedding calls pass through unchanged.

LiteLLM's SDK-side ``litellm.callbacks`` list does not run ``async_pre_call_hook``
(that is proxy-only); for in-process SDK use see
:mod:`distil.integrations.litellm`.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
from pathlib import Path
from typing import Any

from ..adapters.anthropic import compress_messages
from ..adapters.openai import compress_chat_completions
from ..runtime import RuntimeSavings

__all__ = ["HookCore", "compress_request"]

_log = logging.getLogger("distil.litellm")  # exception *types* only, never content

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        _log.warning(msg)


def _digest_allowed() -> bool:
    """May the opt-in digest run here? Same policy as the proxy: never on a
    subscription — and with no distil_expand to make it recoverable, there is no
    ``--expand``-style exception either."""
    from ..policy import may_compress_lossy, session_auth_mode

    if may_compress_lossy(session_auth_mode(False)):
        _warn_once(
            "irrecoverable",
            "distil litellm: digest=True sends Tier-1 stubs the model cannot recover "
            "(no distil_expand from a LiteLLM hook); use `distil proxy` for a "
            "recoverable digest",
        )
        return True
    _warn_once(
        "subscription",
        "distil litellm: digest refused on a subscription session; running lossless-only",
    )
    return False


_CHAT_CALLS = frozenset({"completion", "acompletion"})
_ANTHROPIC_CALLS = frozenset({"anthropic_messages"})
# LiteLLM's ``/v1/responses`` route (``litellm.responses`` / ``aresponses``): the body's
# ``input`` array, the shape Codex speaks.
_RESPONSES_CALLS = frozenset({"responses", "aresponses"})


def _count(messages: list[dict[str, Any]]) -> int:
    from ..proxy import _count_messages  # same heuristic the proxy ledger rows use

    return _count_messages(messages)


def compress_request(
    data: dict[str, Any], call_type: str, *, digest: bool = False
) -> tuple[dict[str, Any], int, int] | None:
    """Return ``(new_data, tokens_before, tokens_after)``, or ``None`` if untouched.

    Pure and framework-free. Raises nothing the caller must handle beyond what the
    adapters raise; :meth:`HookCore.apply` wraps it fail-open.
    """
    if call_type in _RESPONSES_CALLS:
        items = data.get("input")
        if not isinstance(items, list):
            return None  # a bare-string input has no tool output to compress
        from ..adapters.openai import compress_responses_input, count_responses_tokens

        new_items, _store = compress_responses_input(items, verbatim=not digest)
        return (
            {**data, "input": new_items},
            count_responses_tokens(items),
            count_responses_tokens(new_items),
        )
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None
    if call_type in _CHAT_CALLS:
        compressed, _store = compress_chat_completions(messages, verbatim=not digest)
    elif call_type in _ANTHROPIC_CALLS:
        compressed, _store = compress_messages(messages, verbatim=not digest)
    else:
        return None
    return {**data, "messages": compressed}, _count(messages), _count(compressed)


class HookCore:
    """The hook's logic, independent of LiteLLM (so it is testable without it)."""

    def __init__(self, *, digest: bool = False, ledger_path: Path | str | None = None) -> None:
        # The YAML form's DISTIL_LITELLM_DIGEST arrives here too, so one guard covers both.
        self.digest = digest and _digest_allowed()
        self.savings = RuntimeSavings(
            mode="digest" if self.digest else "verbatim",
            ledger_path=Path(ledger_path) if ledger_path else None,
        )
        atexit.register(self._flush)

    def _flush(self) -> None:
        try:
            self.savings.flush()
        except Exception as e:  # noqa: BLE001 — exit path must never raise
            _log.debug("distil litellm: ledger flush failed (%s)", type(e).__name__)

    def apply(self, data: dict[str, Any], call_type: str | None) -> dict[str, Any]:
        """Return the request to send: compressed, or *data* itself on any failure."""
        try:
            out = compress_request(data, str(call_type or ""), digest=self.digest)
            if out is None:
                return data
            new_data, before, after = out
            self.savings.record(before, after, str(data.get("model") or ""))
            self.savings.maybe_flush()
            return new_data
        except Exception as e:  # noqa: BLE001 — compression must never break a request
            _log.debug("distil litellm: fail-open (%s)", type(e).__name__)
            return data


def _build_hook_class() -> Any:
    from litellm.integrations.custom_logger import CustomLogger

    class DistilCompressionHook(CustomLogger):  # type: ignore[misc]
        """LiteLLM Proxy ``CustomLogger`` that compresses requests via distil."""

        def __init__(
            self, digest: bool = False, ledger_path: Path | str | None = None, **kwargs: Any
        ) -> None:
            super().__init__(**kwargs)
            self.core = HookCore(digest=digest, ledger_path=ledger_path)

        async def async_pre_call_hook(
            self, user_api_key_dict: Any, cache: Any, data: dict[str, Any], call_type: Any
        ) -> dict[str, Any]:
            # CPU-bound: keep the proxy's event loop free.
            return await asyncio.to_thread(self.core.apply, data, call_type)

    return DistilCompressionHook


_cache: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Build the hook class / default instance on first access (lazy litellm import)."""
    if name not in ("DistilCompressionHook", "proxy_handler_instance"):
        raise AttributeError(name)
    if "cls" not in _cache:
        try:
            _cache["cls"] = _build_hook_class()
        except ImportError as e:
            raise ImportError(
                "distil's LiteLLM hook needs litellm: pip install 'distil-llm[litellm]'"
            ) from e
    if name == "DistilCompressionHook":
        return _cache["cls"]
    if "inst" not in _cache:
        _cache["inst"] = _cache["cls"](
            digest=os.environ.get("DISTIL_LITELLM_DIGEST", "") not in ("", "0")
        )
    return _cache["inst"]
