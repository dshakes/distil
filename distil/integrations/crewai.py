"""CrewAI integration — compress tool output before it re-enters the agent's context.

Preferred path is still the proxy (``LLM(base_url=...)``, see docs/integrations.html).
This adds the in-process half: CrewAI feeds every tool's return value back to the
model, so a chatty tool bloats every later turn. Wrap the tools once::

    from distil.integrations.crewai import wrap_tools

    agent = Agent(role="...", tools=wrap_tools([search_db, read_log]))

It is **duck-typed and never imports crewai**: a tool is anything with ``_run``
(CrewAI ``BaseTool``). Each tool is copied (``model_copy``/``copy.copy``) and the copy's
``_run`` replaced, so the caller's tool, name, description and schema are untouched
and the result is still a real ``BaseTool`` that ``Agent(tools=...)`` accepts.
Outputs use the reversible Tier-1 digest; non-string results pass through.
"""

from __future__ import annotations

import copy
import functools
from typing import Any

from ..adapters.anthropic import RestoreStore, _compress_tool_result_text, _keep_tls

__all__ = ["compress_tool_output", "wrap_tools"]


def compress_tool_output(text: str, *, verbatim: bool = False) -> str:
    """Compress one tool result string (returned unchanged if it doesn't shrink)."""
    _keep_tls.fn = None
    try:
        return _compress_tool_result_text(text, RestoreStore(), verbatim)
    finally:
        _keep_tls.fn = None


def _wrap_one(tool: Any, verbatim: bool) -> Any:
    if not callable(getattr(tool, "_run", None)):
        raise TypeError(f"{type(tool).__name__} has no _run(); not a CrewAI tool")
    clone = tool.model_copy() if hasattr(tool, "model_copy") else copy.copy(tool)
    inner = clone._run

    @functools.wraps(inner)
    def _run(*args: Any, **kwargs: Any) -> Any:
        out = inner(*args, **kwargs)
        return compress_tool_output(out, verbatim=verbatim) if isinstance(out, str) else out

    # object.__setattr__: pydantic models reject setting a non-field attribute.
    object.__setattr__(clone, "_run", _run)

    ainner = getattr(clone, "_arun", None)
    if callable(ainner):  # async execution calls _arun and would skip _run entirely

        @functools.wraps(ainner)
        async def _arun(*args: Any, **kwargs: Any) -> Any:
            out = await ainner(*args, **kwargs)
            return compress_tool_output(out, verbatim=verbatim) if isinstance(out, str) else out

        object.__setattr__(clone, "_arun", _arun)
    return clone


def wrap_tools(tools: list[Any], *, verbatim: bool = False) -> list[Any]:
    """Return copies of *tools* whose string results are compressed."""
    return [_wrap_one(t, verbatim) for t in tools]
