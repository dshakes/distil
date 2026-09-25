"""The four arms as data: pinned install, documented Claude Code integration, savings claim.

Nothing here executes. ``plan`` prints these so a reviewer (and the competitor maintainers
the protocol invites) can check the exact commands *before* a dollar is spent. Every field
marked ``verified=False`` is a claim about a third-party CLI that the live preflight
(protocol §6.3, the "chain proof") must confirm; ``plan`` lists them and the live runner
refuses to start while any remain.

Versions are the latest releases on 2026-09-25 and are frozen with the protocol. They are
not bumped mid-study, even if a bug is found (Quesma hit an ``rtk find`` loop fixed one
release after their run; the protocol's answer is to report it, not to swap versions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

CLAUDE_CODE_VERSION = "2.1.282"  # npm @anthropic-ai/claude-code, latest 2026-09-25
HARBOR_VERSION = "0.23.0"  # PyPI harbor, sha256 8747400d…5c37 (wheel)
TASK_SUITE = "terminal-bench@2.1"  # Harbor dataset id — confirm spelling at freeze


@dataclass(frozen=True)
class Arm:
    name: str
    version: str | None
    #: argv steps, run on the host with cwd = the arm's isolated tool dir ``{tools}``.
    install: tuple[tuple[str, ...], ...]
    #: how Claude Code is launched. ``{meter}`` = the neutral meter's base URL;
    #: ``{claude}`` = the claude argv (``-p <instruction> --model M ...``).
    launch: tuple[str, ...]
    #: env the arm needs so ITS upstream is the meter (never the provider directly).
    upstream_env: dict[str, str] = field(default_factory=dict)
    #: the tool's own savings report, read from the run's isolated state after the run.
    claim: tuple[str, ...] = ()
    claim_unit: str = ""
    verified: bool = False
    notes: str = ""


ARMS: dict[str, Arm] = {
    "control": Arm(
        name="control",
        version=None,
        install=(),
        launch=("{claude}",),
        upstream_env={"ANTHROPIC_BASE_URL": "{meter}"},
        verified=True,
        notes="Claude Code direct to the meter. No compressor.",
    ),
    "rtk": Arm(
        name="rtk",
        version="0.50.0",
        install=(
            (
                "curl",
                "-fsSLo",
                "{tools}/rtk.tar.gz",
                "https://github.com/rtk-ai/rtk/releases/download/v0.50.0/"
                "rtk-x86_64-unknown-linux-musl.tar.gz",
            ),
            (
                "sh",
                "-c",
                "echo 'bc2b8902b0d9c796c82ef45f16ae2307e17757afeca5ee156235a3dc7bda5f89  "
                "{tools}/rtk.tar.gz' | sha256sum -c -",
            ),
            ("tar", "-xzf", "{tools}/rtk.tar.gz", "-C", "{tools}"),
            # inside the task container, with HOME = the run's fresh home:
            ("{tools}/rtk", "init", "-g"),
        ),
        launch=("{claude}",),
        upstream_env={"ANTHROPIC_BASE_URL": "{meter}"},
        claim=("{tools}/rtk", "gain"),
        claim_unit="rtk-tokens (output bytes removed / 4, per RTK's docs)",
        verified=False,
        notes="Claude Code PreToolUse hook rewrites Bash commands; no proxy, so claude talks "
        "to the meter directly. Static musl binary: installing it does not perturb the task "
        "container's Python/Node. Verify: `rtk init -g` hook flags, `rtk gain` JSON output.",
    ),
    "headroom": Arm(
        name="headroom",
        version="0.38.0",
        install=(
            ("uv", "venv", "--python", "3.12", "{tools}/venv"),
            # hashes for the full closure are generated at freeze with
            # `uv pip compile --generate-hashes` and committed next to the protocol
            (
                "uv",
                "pip",
                "install",
                "--python",
                "{tools}/venv/bin/python",
                "--require-hashes",
                "-r",
                "{tools}/headroom-0.38.0.requirements.txt",
            ),
        ),
        launch=("{tools}/venv/bin/headroom", "wrap", "{claude}"),
        upstream_env={"ANTHROPIC_TARGET_API_URL": "{meter}"},
        claim=("{tools}/venv/bin/headroom", "savings", "--json"),
        claim_unit="headroom-reported tokens and USD",
        verified=False,
        notes="Documented integration `headroom wrap claude`. UNVERIFIED: the env/flag that "
        "points headroom's upstream at a custom URL, wrap's argv passthrough syntax, and the "
        "savings CLI. The chain proof fails closed if the meter sees no traffic.",
    ),
    "distil": Arm(
        name="distil",
        version="1.52.0",
        install=(
            ("uv", "venv", "--python", "3.12", "{tools}/venv"),
            (
                "uv",
                "pip",
                "install",
                "--python",
                "{tools}/venv/bin/python",
                "--require-hashes",
                "-r",
                "{tools}/distil-1.52.0.requirements.txt",
            ),
        ),
        launch=("{tools}/venv/bin/distil", "wrap", "--upstream", "{meter}", "{claude}"),
        upstream_env={"DISTIL_HOME": "{state}/distil"},
        claim=("{tools}/venv/bin/distil", "stats", "--json"),
        claim_unit="distil-reported tokens and USD",
        verified=False,
        notes="The PUBLISHED wheel (PyPI distil-llm), never the authors' working tree. "
        "`--upstream` exists on `distil wrap` (distil/cli.py); verify it accepts an http "
        "loopback URL and that `stats --json` reads the isolated DISTIL_HOME.",
    ),
}


def unverified() -> list[str]:
    return [a.name for a in ARMS.values() if not a.verified]


def render(template: tuple[str, ...], claude: list[str] | None = None, **subs: str) -> list[str]:
    """Fill ``{name}`` placeholders; a bare ``{claude}`` expands to the whole claude argv."""
    out: list[str] = []
    for part in template:
        if part == "{claude}":
            out.extend(claude or [])
        else:
            out.append(part.format(**subs))
    return out
