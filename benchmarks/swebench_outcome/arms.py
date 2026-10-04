"""Pluggable arms. An arm is a name plus up to five independent hooks; the agent loop never
branches on the arm's name.

- `transform`:    messages -> messages | (messages, store); applied before every model call.
                  `store` (if any) is kept for the arm's tool `handlers`.
- `tools`/`handlers`: extra tool definitions and the code that answers them.
- `params`/`betas`:   extra `messages.create` kwargs; any `betas` route the call through
                  `client.beta.messages.create(betas=[...])`, the SDK's documented mechanism.
- `wrap_env`:     (env, stats) -> env; the tool-output boundary (e.g. RTK rewrites bash commands).
- `on_response`:  (response, stats) -> None; harvest per-call arm telemetry into `stats`.
- `meta`:         library + version (+ config), copied into every result row.

"plain" has no hooks, so its requests are exactly the baseline's.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

from distil.expand import EXPAND_TOOL, EXPAND_TOOL_NAME

from .env import Env
from .names import ARM_NAMES, BASELINE

__all__ = ["ARM_NAMES", "BASELINE"]  # re-exported for callers that already import arms

Transform = Callable[[list[dict[str, Any]]], Any]
Handler = Callable[[dict[str, Any], list[Any], dict[str, Any]], tuple[str, bool]]


class ArmUnavailable(RuntimeError):
    """An arm's real dependency is missing or has the wrong version. Raised before any spend."""


@dataclass(frozen=True)
class Arm:
    name: str
    transform: Transform | None = None
    tools: tuple[dict[str, Any], ...] = ()
    handlers: Mapping[str, Handler] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    betas: tuple[str, ...] = ()
    wrap_env: Callable[[Env, dict[str, Any]], Env] | None = None
    on_response: Callable[[Any, dict[str, Any]], None] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


def pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def parse_arms(spec: str) -> tuple[str, ...]:
    names = tuple(s.strip() for s in spec.split(",") if s.strip())
    bad = [n for n in names if n not in ARM_NAMES]
    if bad or not names or len(set(names)) != len(names):
        raise SystemExit(
            f"--arms must be a comma list of distinct names from {', '.join(ARM_NAMES)}; got {spec!r}"
        )
    return names


def plain_arm() -> Arm:
    return Arm("plain")


def _expand(inp: dict[str, Any], stores: list[Any], res: dict[str, Any]) -> tuple[str, bool]:
    res["expand_calls"] += 1
    for s in reversed(stores):
        try:
            return str(s.expand(str(inp.get("handle", "")))), False
        except KeyError:
            continue
    res["expand_misses"] += 1
    return f"error: no original found for handle {inp.get('handle')!r}", True


def distil_arm(compress: Transform | None = None) -> Arm:
    """The served transform, `compress_messages`, plus the `distil_expand` tool it needs."""

    def transform(messages: list[dict[str, Any]]) -> Any:
        if compress is not None:
            return compress(messages)
        from distil.adapters.anthropic import compress_messages

        return compress_messages(messages)

    return Arm(
        "distil",
        transform=transform,
        tools=(EXPAND_TOOL,),
        handlers={EXPAND_TOOL_NAME: _expand},
        meta={"library": "distil", "version": pkg_version("distil-llm")},
    )


# Anthropic context editing (docs: build-with-claude/context-editing). Verified shape: beta header
# `context-management-2025-06-27`, `context_management={"edits": [{"type":
# "clear_tool_uses_20250919", trigger, keep, clear_at_least, ...}]}`, response
# `context_management.applied_edits[{type, cleared_tool_uses, cleared_input_tokens}]`.
CM_BETA = "context-management-2025-06-27"
CM_EDIT = "clear_tool_uses_20250919"
# The documented defaults (trigger 100k input tokens, keep 3) never fire here: the committed
# 300-task runs average ~3.9k prompt tokens per step (p90 ~5.9k). The harness defaults below are
# scaled to that range so the strategy actually engages; they are a harness choice, not
# Anthropic's defaults, and are recorded in every row's arm_meta.
CM_TRIGGER, CM_KEEP, CM_CLEAR_AT_LEAST = 3000, 2, 1000


def provider_cm_arm(
    trigger: int = CM_TRIGGER, keep: int = CM_KEEP, clear_at_least: int = CM_CLEAR_AT_LEAST
) -> Arm:
    if min(trigger, keep, clear_at_least) < 1:
        raise SystemExit("provider-cm: trigger, keep and clear_at_least must be >= 1")
    edit = {
        "type": CM_EDIT,
        "trigger": {"type": "input_tokens", "value": trigger},
        "keep": {"type": "tool_uses", "value": keep},
        "clear_at_least": {"type": "input_tokens", "value": clear_at_least},
    }

    def on_response(resp: Any, stats: dict[str, Any]) -> None:
        stats.setdefault("applied_calls", 0)
        stats.setdefault("cleared_tool_uses", 0)
        stats.setdefault("cleared_input_tokens", 0)
        cm = getattr(resp, "context_management", None)
        edits = getattr(cm, "applied_edits", None) or []
        if edits:
            stats["applied_calls"] += 1
        for e in edits:
            stats["cleared_tool_uses"] += getattr(e, "cleared_tool_uses", 0) or 0
            stats["cleared_input_tokens"] += getattr(e, "cleared_input_tokens", 0) or 0

    return Arm(
        "provider-cm",
        params={"context_management": {"edits": [edit]}},
        betas=(CM_BETA,),
        on_response=on_response,
        meta={
            "library": "anthropic",
            "version": pkg_version("anthropic"),
            "beta": CM_BETA,
            "edit": CM_EDIT,
            "trigger_input_tokens": trigger,
            "keep_tool_uses": keep,
            "clear_at_least_input_tokens": clear_at_least,
        },
    )


def get_arm(name: str, compress: Transform | None = None) -> Arm:
    """Arms that need no external dependency. The others are built by `cli.build_arms`."""
    if name == "plain":
        return plain_arm()
    if name == "distil":
        return distil_arm(compress)
    raise KeyError(f"arm {name!r} needs options; build it with cli.build_arms")


def resolve_arm(arm: str | Arm, compress: Transform | None = None) -> Arm:
    if isinstance(arm, Arm):
        return arm
    return get_arm(arm, compress)
