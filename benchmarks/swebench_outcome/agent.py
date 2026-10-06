"""Minimal tool-use coding agent. Arms differ only through the hooks on `arms.Arm`."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .arms import Arm, resolve_arm
from .env import Env, EnvError

PRICES = {  # $/MTok (in, out); mirrors benchmarks/model_migration_eval.py PRICES
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
}
OUT_CAP = 20_000
_USAGE = (
    ("input", "input_tokens"),
    ("output", "output_tokens"),
    ("cache_write", "cache_creation_input_tokens"),
    ("cache_read", "cache_read_input_tokens"),
)
SYSTEM = (
    "You are an expert software engineer. A repository is checked out at /testbed and you must "
    "resolve the GitHub issue you are given by editing source files (not tests). Explore with "
    "bash, edit with the editor tool, and run code to check your fix. When done, stop and "
    "reply with a short summary; your changes are collected automatically."
)


def direct_client() -> Any:
    """Always https://api.anthropic.com: a distil-wrapped shell exports ANTHROPIC_BASE_URL to the
    local proxy, and the harness must not inherit it (same reason as model_migration_eval)."""
    import anthropic

    base = os.environ.get("ANTHROPIC_BASE_URL")
    if base and "api.anthropic.com" not in base:
        print(f"note: ignoring ANTHROPIC_BASE_URL={base}; calling api.anthropic.com directly")
    # An explicit timeout: the SDK refuses a non-streaming request whose max_tokens implies more
    # than 10 minutes (> ~21k) unless the client timeout is set, and --max-tokens may exceed it.
    return anthropic.Anthropic(base_url="https://api.anthropic.com", max_retries=4, timeout=1800.0)


def price(model: str, pin: float | None = None, pout: float | None = None) -> tuple[float, float]:
    if pin is not None and pout is not None:
        return pin, pout
    base = model.split("@")[0]
    if base not in PRICES:
        raise KeyError(f"no price for {model}; pass --price-in/--price-out ($/MTok)")
    return PRICES[base]


def cost_usd(u: dict[str, int], pin: float, pout: float) -> float:
    """Cache tokens at the standard 1.25x write / 0.1x read multipliers."""
    inp = u["input"] + 1.25 * u["cache_write"] + 0.1 * u["cache_read"]
    return (inp * pin + u["output"] * pout) / 1e6


class Budget:
    def __init__(self, limit: float | None):
        self.limit, self.spent = limit, 0.0

    def add(self, usd: float) -> None:
        self.spent += usd

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.spent >= self.limit


@dataclass
class Cfg:
    model: str = "claude-sonnet-5-5"
    effort: str = "medium"
    max_steps: int = 40
    task_timeout: float = 1800.0
    max_tokens: int = 16_000
    pin: float | None = None
    pout: float | None = None


def build_tools(arm: str | Arm) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {"type": "bash_20250124", "name": "bash"},
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
    ]
    return [*tools, *resolve_arm(arm).tools]


def _cap(s: str) -> str:
    if len(s) <= OUT_CAP:
        return s
    return (
        s[: OUT_CAP // 2] + f"\n...[{len(s) - OUT_CAP} chars truncated]...\n" + s[-OUT_CAP // 2 :]
    )


def run_editor(env: Env, inp: dict[str, Any]) -> str:
    cmd, path = inp.get("command"), inp.get("path", "")
    try:
        if cmd == "view":
            text = env.read_file(path)
            lines = text.splitlines()
            lo, hi = (inp.get("view_range") or [1, len(lines)])[:2]
            hi = len(lines) if hi == -1 else hi
            return "\n".join(f"{i:6}\t{lines[i - 1]}" for i in range(lo, min(hi, len(lines)) + 1))
        if cmd == "create":
            env.write_file(path, inp.get("file_text", ""))
            return f"created {path}"
        if cmd == "str_replace":
            text, old = env.read_file(path), inp.get("old_str", "")
            n = text.count(old)
            if n != 1:
                return f"error: old_str matched {n} times in {path}; it must match exactly once"
            env.write_file(path, text.replace(old, inp.get("new_str", ""), 1))
            return f"edited {path}"
        if cmd == "insert":
            lines = env.read_file(path).splitlines(True)
            at = int(inp.get("insert_line", 0))
            lines[at:at] = [inp.get("insert_text", "") + "\n"]
            env.write_file(path, "".join(lines))
            return f"inserted into {path}"
        return f"error: unsupported command {cmd!r}"
    except FileNotFoundError:
        return f"error: {path} does not exist"


def with_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of *messages* with an ephemeral cache breakpoint on the newest block."""
    if not messages:
        return messages
    last = dict(messages[-1])
    content = last["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    last["content"] = blocks
    return [*messages[:-1], last]


def _param(block: Any) -> dict[str, Any]:
    return block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else dict(block)


def run_agent(
    client: Any,
    env: Env,
    task: dict[str, Any],
    arm: str | Arm,
    cfg: Cfg,
    budget: Budget,
    clock: Callable[[], float] = time.monotonic,
    compress: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Returns a result record. `failure_class` is None for a graded outcome.

    `compress` overrides the distil arm's transform (tests); other arms ignore it."""
    import anthropic

    arm = resolve_arm(arm, compress)
    pin, pout = price(cfg.model, cfg.pin, cfg.pout)
    use = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    res: dict[str, Any] = {
        "instance_id": task["instance_id"],
        "arm": arm.name,
        "arm_meta": dict(arm.meta),
        "model": cfg.model,
        "effort": cfg.effort,
        "failure_class": None,
        "error": None,
        "steps": 0,
        "expand_calls": 0,
        "expand_misses": 0,
    }
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Resolve this issue:\n\n" + task["problem_statement"]}
    ]
    tools, stores, t0 = build_tools(arm), [], clock()
    arm_stats: dict[str, Any] = {}
    run_env = env
    stop = "step_limit"
    transcript: list[Any] = list(messages)
    try:
        if arm.wrap_env:
            run_env = arm.wrap_env(env, arm_stats)
        while res["steps"] < cfg.max_steps:
            if budget.exhausted:
                res["failure_class"], stop = "budget_stopped", "budget"
                break
            if clock() - t0 > cfg.task_timeout:
                res["failure_class"], stop = "timeout", "timeout"
                break
            # Every arm caches like a real agent (Claude Code marks the newest turn); without
            # it every step re-pays tools + system + history at full price, which inflated
            # the distil arm's injected expand-tool definition in the first powered run.
            send = with_cache_breakpoint(messages)
            if arm.transform:
                out = arm.transform(send)
                send, store = out if isinstance(out, tuple) else (out, None)
                if store is not None:
                    stores.append(store)
            kwargs: dict[str, Any] = dict(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                system=SYSTEM,
                tools=tools,
                messages=send,
                thinking={"type": "adaptive"},
                output_config={"effort": cfg.effort},
                tool_choice={"type": "auto"},
            )
            kwargs.update(arm.params)
            api = arm.client or client
            create = api.messages.create
            if arm.betas:
                kwargs["betas"] = list(arm.betas)
                create = api.beta.messages.create
            try:
                resp = create(**kwargs)
            except anthropic.APIError as e:
                res["failure_class"], res["error"], stop = "api_error", repr(e)[:500], "api_error"
                break
            res["steps"] += 1
            if arm.on_response:
                arm.on_response(resp, arm_stats)
            d = {k: getattr(resp.usage, a, 0) or 0 for k, a in _USAGE}
            for k, v in d.items():
                use[k] += v
            budget.add(cost_usd(d, pin, pout))
            assistant = {"role": "assistant", "content": [_param(b) for b in resp.content]}
            messages.append(assistant)
            transcript.append(assistant)
            uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not uses:
                stop = resp.stop_reason or "end_turn"
                break
            results = []
            for b in uses:
                out, err = _dispatch(b, run_env, stores, res, arm)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": _cap(out),
                        **({"is_error": True} if err else {}),
                    }
                )
            user = {"role": "user", "content": results}
            messages.append(user)
            transcript.append(user)
    except EnvError as e:
        res["failure_class"], res["error"], stop = "env_error", str(e)[:500], "env_error"
    except Exception as e:  # noqa: BLE001 - recorded, never scored as unresolved
        res["failure_class"], res["error"], stop = "internal_error", repr(e)[:500], "internal_error"
    patch = ""
    if res["failure_class"] not in ("env_error", "internal_error"):
        try:
            patch = run_env.diff()
        except EnvError as e:
            res["failure_class"], res["error"] = "env_error", str(e)[:500]
    if res["failure_class"] is None and not patch.strip():
        res["failure_class"] = "gave_up"
    res.update(
        stop=stop,
        patch=patch,
        wall_s=round(clock() - t0, 2),
        usage=use,
        cost_usd=round(cost_usd(use, pin, pout), 6),
        transcript=transcript,
    )
    if arm_stats:
        res["arm_stats"] = arm_stats
    return res


def _dispatch(
    b: Any, env: Env, stores: list[Any], res: dict[str, Any], arm: Arm
) -> tuple[str, bool]:
    name, inp = b.name, b.input
    if name == "bash":
        if inp.get("restart"):
            return "restarted (each command runs in a fresh shell at /testbed)", False
        return _cap(env.exec(inp.get("command", ""))), False
    if name == "str_replace_based_edit_tool":
        out = run_editor(env, inp)
        return out, out.startswith("error")
    if name in arm.handlers:
        return arm.handlers[name](inp, stores, res)
    return f"error: unknown tool {name}", True
