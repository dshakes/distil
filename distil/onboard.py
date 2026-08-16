"""``distil onboard`` — one command that sets up distil and guides you to use it.

Detects your environment (OS, package managers, agent CLIs, install method, the
optional anthropic extra, Claude Code + subscription), wires the savings status
line, and prints a next-steps guide tailored to what it found — how to route your
agent, validate outcomes with shadow mode, watch savings, and re-verify. Works on
macOS and Windows; mutating actions are gated.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .doctor import subscription_mode

# Agent CLIs we know how to route, in priority order.
_AGENTS = [
    ("claude", "Claude Code"),
    ("codex", "Codex"),
    ("gemini", "Gemini CLI"),
    ("opencode", "OpenCode"),
    ("qwen", "Qwen Code"),
    ("goose", "goose"),
    ("grok", "Grok CLI"),
    ("openhands", "OpenHands"),
]
_MANAGERS = ("pipx", "uv", "brew", "scoop", "pip")

# Per-agent proxy presets for `distil wrap`.  Maps argv[0] basename → (env_var,
# upstream, label).  Sourced from each agent's published SDK/env-var contract:
#   claude       — Anthropic SDK honours ANTHROPIC_BASE_URL (Anthropic SDK docs).
#   codex        — OpenAI Codex CLI uses the OpenAI SDK which reads OPENAI_BASE_URL;
#                  the SDK appends /v1 itself so no suffix needed here (OpenAI SDK docs).
#   gemini       — Gemini CLI honours GOOGLE_GEMINI_BASE_URL (verified: distil statusline
#                  already checks this var, and Gemini CLI changelog confirms it).
#   aider        — defaults to OpenAI mode; OPENAI_BASE_URL is the primary override
#                  (aider docs: --openai-api-base / OPENAI_BASE_URL).  Users routing
#                  Claude models should pass --env-var ANTHROPIC_BASE_URL explicitly.
#   opencode     — OpenAI-compatible mode reads OPENAI_API_KEY + OPENAI_BASE_URL from
#                  the environment, and env takes precedence over its config file
#                  (OpenCode provider docs).
#   qwen         — Qwen Code calls LLMs through the OpenAI SDK and documents
#                  OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL explicitly
#                  (Qwen Code README).
#   goose        — Block's goose reads OPENAI_HOST for the endpoint (NOT
#                  OPENAI_BASE_URL, which it ignores) plus OPENAI_BASE_PATH,
#                  defaulting to "v1/chat/completions" — a path distil's proxy
#                  serves, so the default needs no change (goose provider docs).
#   grok         — superagent-ai/grok-cli resolves its endpoint as (1) an explicit
#                  argument, (2) GROK_BASE_URL, (3) the default https://api.x.ai/v1.
#                  Note the upstream carries the /v1 itself, unlike the OpenAI SDK.
#   openhands    — reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL, but ONLY when run
#                  with `--override-with-envs`. Without that flag it ignores the
#                  environment entirely and reads ~/.openhands/settings.json, so a
#                  preset alone routes NOTHING while reporting success — the exact
#                  failure the note below is about. `wrap` therefore checks argv and
#                  says so; see _warn_if_env_ignored in cli.py.
#   cursor-agent — env var not publicly documented; left out rather than guessing.
#                  Use --env-var to configure manually.
#
# DELIBERATELY ABSENT: cursor, copilot, cline, continue, windsurf. These are IDE
# extensions, not CLIs — there is no argv to wrap and no documented env-var
# contract, so `wrap` cannot reach them and a guessed variable would silently
# route nothing while reporting success. Their supported path is the always-on
# proxy plus the editor's own "custom base URL"/OpenAI-compatible setting; see
# docs/IDE-AGENTS.md. Adding a preset here without a published contract is how
# you ship a lie that looks like a feature.
AGENT_PRESETS: dict[str, tuple[str, str, str]] = {
    # cmd_name: (env_var, upstream_base_url, human_label)
    "claude": ("ANTHROPIC_BASE_URL", "https://api.anthropic.com", "Claude Code"),
    "codex": ("OPENAI_BASE_URL", "https://api.openai.com", "Codex CLI"),
    "gemini": (
        "GOOGLE_GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com",
        "Gemini CLI",
    ),
    "aider": ("OPENAI_BASE_URL", "https://api.openai.com", "aider"),
    "opencode": ("OPENAI_BASE_URL", "https://api.openai.com", "OpenCode"),
    "qwen": ("OPENAI_BASE_URL", "https://api.openai.com", "Qwen Code"),
    "goose": ("OPENAI_HOST", "https://api.openai.com", "goose"),
    "grok": ("GROK_BASE_URL", "https://api.x.ai/v1", "Grok CLI"),
    "openhands": ("LLM_BASE_URL", "https://api.openai.com", "OpenHands"),
}


@dataclass
class Env:
    os_name: str
    managers: list[str] = field(default_factory=list)
    agents: list[tuple[str, str]] = field(default_factory=list)  # (cmd, label)
    has_anthropic: bool = False
    has_api_key: bool = False
    subscription: bool = False
    installed_version: str = ""
    method: str = "pip"  # how distil is installed: pipx | uv | uvx | pip


def detect() -> Env:
    from . import __version__

    return Env(
        os_name=platform.system() or "unknown",
        managers=[m for m in _MANAGERS if shutil.which(m)],
        agents=[(c, n) for c, n in _AGENTS if shutil.which(c)],
        has_anthropic=importlib.util.find_spec("anthropic") is not None,
        has_api_key=bool(os.environ.get("ANTHROPIC_API_KEY")),
        subscription=subscription_mode(),
        installed_version=__version__,
        method=install_method(),
    )


def install_method() -> str:
    """How the running distil is installed — drives the right upgrade/uninstall
    command. THE single source of truth (offboard, upgrade, doctor all use it).
    Checks the package path AND the resolved `distil` binary path, because a
    Homebrew install lives in a Cellar venv the package path alone can miss."""
    from . import __file__ as pkg_file

    paths = [(pkg_file or "")]
    exe = shutil.which("distil")
    if exe:
        paths.append(str(Path(exe).resolve()))
    blob = " ".join(paths).replace(os.sep, "/").lower()
    if "/cellar/" in blob or "/homebrew/" in blob or "/usr/local/" in blob:
        return "homebrew"
    if "/pipx/" in blob:
        return "pipx"
    if "/uv/tools/" in blob:
        return "uv"
    if "/uv/" in blob or "/.cache/uv/" in blob:
        return "uvx"  # ephemeral run — nothing persistent to upgrade
    return "pip"


def upgrade_command(method: str) -> str:
    return {
        "homebrew": "brew upgrade distil",
        "pipx": "pipx upgrade distil-llm",
        "uv": "uv tool upgrade distil-llm",
        "uvx": "uvx --from distil-llm@latest distil onboard   # uvx runs the latest each time",
        "pip": "pip install --upgrade distil-llm   # inside your venv (PEP 668)",
    }.get(method, "pip install --upgrade distil-llm")


def uninstall_command(method: str) -> str:
    """How to uninstall distil for the way it was installed (used by offboard)."""
    return {
        "homebrew": "brew uninstall dshakes/tap/distil && brew untap dshakes/tap",
        "pipx": "pipx uninstall distil-llm",
        "uv": "uv tool uninstall distil-llm",
        "uvx": "# uvx runs ephemerally — nothing persistent to uninstall",
        "pip": "pip uninstall distil-llm   # inside your venv (PEP 668)",
    }.get(method, "pip uninstall distil-llm")


def latest_pypi_version(timeout: float = 2.5) -> str | None:
    """Latest distil-llm version on PyPI, or None if offline / the check fails."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/distil-llm/json", timeout=timeout
        ) as resp:
            return json.load(resp)["info"]["version"]
    except Exception:  # noqa: BLE001 — offline / DNS / timeout: just skip the check
        return None


def _ver_tuple(s: str) -> tuple[tuple[int, ...], bool]:
    import re

    nums = re.findall(r"\d+", s.split("+")[0])[:3]
    base = tuple(int(n) for n in nums) + (0,) * (3 - len(nums))
    is_pre = bool(re.search(r"(dev|rc|a|b)\d*", s))
    return base, is_pre


def is_outdated(installed: str, latest: str | None) -> bool:
    """True if a newer *released* version than ``installed`` is available."""
    if not latest:
        return False
    if installed == latest:
        return False  # nothing is newer than itself, pre-release or not
    bi, pre_i = _ver_tuple(installed)
    bl, _pre_l = _ver_tuple(latest)
    if bi != bl:
        return bi < bl
    return pre_i  # same base number, but ours is a pre-release of it → older


def best_install_command(managers: list[str]) -> str:
    """The recommended way to (re)install distil persistently on this machine."""
    if "pipx" in managers:
        return "pipx install distil-llm"
    if "uv" in managers:
        return "uv tool install distil-llm"
    if "brew" in managers:
        return "brew install pipx && pipx install distil-llm"
    if "scoop" in managers:  # Windows
        return "scoop install pipx && pipx install distil-llm"
    return "python -m pip install --user pipx && pipx install distil-llm"


def next_steps(env: Env) -> list[tuple[str, str, str]]:
    """Tailored guide as (title, command, note) rows."""
    agent = env.agents[0][0] if env.agents else "claude"
    steps: list[tuple[str, str, str]] = []

    if not env.agents:
        steps.append(
            (
                "Install a coding agent",
                "# e.g. Claude Code, Codex, or Gemini CLI",
                "no agent CLI detected on PATH — install one, then re-run distil onboard",
            )
        )

    # Routing — mode depends on billing.
    if env.subscription:
        steps.append(
            (
                "Route your agent (subscription-safe)",
                f"distil wrap --lossless-only -- {agent}",
                "flat-rate plan: real lossless token savings — minifies JSON, collapses"
                " duplicate runs, ToS-safe (no lossy digest, no tool injection). Won't"
                " lower a flat-rate bill, but fewer tokens per turn = more context"
                " headroom and rate-limit room.",
            )
        )
        steps.append(
            (
                "Shrink tool output in-process (Claude Code)",
                "distil hook --install",
                "PostToolUse hook: Claude Code compresses its own tool results before"
                " reading them — a first-party extension point, so no proxy and no"
                " credentials involved. Wins on verbose JSON and repetitive logs"
                " (measured 25-99%); prose and unique-line output are left alone.",
            )
        )
    else:
        steps.append(
            (
                "Route your agent",
                f"distil wrap --expand -- {agent}",
                "metered key: aggressive reversible digest; the model recovers detail on demand",
            )
        )

    steps.append(
        (
            "Validate it preserved your outcomes (shadow)",
            f"distil wrap --shadow 0.1 -- {agent}",
            "runs 10% of requests twice and checks the next action is unchanged — then: distil shadow-stats",
        )
    )
    steps.append(
        (
            "Watch your savings",
            "distil dashboard",
            "live terminal view (or distil leaderboard for a snapshot)",
        )
    )
    steps.append(
        (
            "Run the test gate anytime",
            "distil bench",
            "corpus-wide non-inferiority gate — no API key needed",
        )
    )
    steps.append(
        (
            "Re-verify your setup",
            "distil doctor",
            "ledger, shadow, proxy self-test, wiring",
        )
    )
    if not env.has_anthropic:
        steps.append(
            (
                "Optional: live grading / billing-grade tokens",
                "pipx inject distil-llm anthropic   # then set ANTHROPIC_API_KEY",
                "only needed for --runner/--tokenizer anthropic",
            )
        )
    return steps


def report(env: Env, latest: str | None) -> dict:
    """A structured snapshot for an agent to reason over (`distil onboard --json`).

    Pure facts + recommendations — no actions taken. The intelligence (deciding,
    asking, running steps) lives in the agent/skill that consumes this."""
    outdated = is_outdated(env.installed_version, latest)
    return {
        "os": env.os_name,
        "agents": [c for c, _ in env.agents],
        "primary_agent": env.agents[0][0] if env.agents else None,
        "package_managers": env.managers,
        "billing": "subscription" if env.subscription else "metered",
        "installed_version": env.installed_version,
        "latest_version": latest,
        "upgrade_available": outdated,
        "install_method": env.method,
        "upgrade_command": upgrade_command(env.method) if outdated else None,
        "anthropic_extra": env.has_anthropic,
        "api_key": env.has_api_key,
        "best_install_command": best_install_command(env.managers),
        "next_steps": [{"title": t, "command": cmd, "note": n} for t, cmd, n in next_steps(env)],
    }
