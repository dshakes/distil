"""The four arms: pinned artifacts, documented Claude Code integration, savings claim.

Every tool runs INSIDE the task container, installed and launched exactly as its own docs
say (protocol Amendment 1). This module only builds strings — bash scripts the Harbor
agent (``harbor_agent.py``) executes in the container, and the local executor
(``live.LocalExecutor``) executes against the mock upstream in tests. Nothing here runs a
competitor binary.

Every ``evidence`` string cites the primary source the spec was checked against on
2026-09-25 (wheel/tarball downloaded, hashed, unpacked and READ — never executed).
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path

LOCKS = Path(__file__).resolve().parent / "locks"

# --------------------------------------------------------------------------- pins

CLAUDE_CODE_VERSION = "2.1.282"  # npm @anthropic-ai/claude-code latest, 2026-09-25
HARBOR_VERSION = "0.23.0"  # PyPI harbor; wheel sha256 8747400dbb2a5e22…
TASK_DATASET = "terminal-bench/terminal-bench-2-1"  # Harbor hub id (89 tasks); resolved version is recorded in the run manifest

#: Downloaded once on the host by ``live.prepare_tools``, sha256-verified, then bind-mounted
#: read-only into every task container at ``HOST_MOUNT``. Name -> (url, sha256).
ARTIFACTS: dict[str, tuple[str, str]] = {
    "uv.tar.gz": (
        "https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-unknown-linux-musl.tar.gz",
        "db7278c9f57981338fddff1fb250e11964bc0a4fafcb9eed8303fdb117dc067b",
    ),
    "rtk.tar.gz": (
        "https://github.com/rtk-ai/rtk/releases/download/v0.50.0/rtk-x86_64-unknown-linux-musl.tar.gz",
        "bc2b8902b0d9c796c82ef45f16ae2307e17757afeca5ee156235a3dc7bda5f89",
    ),
}
UV_DIR_IN_TARBALL = "uv-x86_64-unknown-linux-musl"

CT = "/opt/cost-truth"  # everything the harness adds to a container lives here
HOST_MOUNT = f"{CT}/host"  # read-only: artifacts + locks
UV_CACHE_MOUNT = f"{CT}/uv-cache"  # read-write, shared across containers (uv locks it)
#: Serena is resolved by ``uvx`` at run time (unpinned in headroom's spec); this pins the
#: resolution to the freeze date so it cannot drift mid-study.
UV_EXCLUDE_NEWER = "2026-09-25T00:00:00Z"

CANARY_MAX_TOKENS = 8
CANARY_USER_PREFIX = "cost-truth-canary-"

#: Identical for every arm (protocol Amendment 1, §B): each tool's own wrap sets
#: ENABLE_TOOL_SEARCH=true because Claude Code turns tool-search deferral OFF behind any
#: non-first-party ANTHROPIC_BASE_URL — and the meter makes EVERY arm non-first-party.
#: Setting it everywhere restores first-party behaviour for control/rtk instead of letting
#: the proxy arms be credited for undoing a meter artifact. Both tools keep an existing value.
COMMON_ENV = {
    "ENABLE_TOOL_SEARCH": "true",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_AUTOUPDATER": "1",
    "IS_SANDBOX": "1",  # Harbor's claude-code agent: allows bypassPermissions as root
    "FORCE_AUTO_BACKGROUND_TASKS": "1",  # Harbor's claude-code agent sets both
    "ENABLE_BACKGROUND_TASKS": "1",
}


@dataclass(frozen=True)
class Arm:
    name: str
    version: str | None
    verified: bool
    evidence: tuple[str, ...]
    claim_unit: str = ""


ARMS: dict[str, Arm] = {
    "control": Arm("control", None, True, ("Claude Code direct to the meter.",)),
    "rtk": Arm(
        "rtk",
        "0.50.0",
        True,
        (
            "release tarball rtk-x86_64-unknown-linux-musl.tar.gz sha256 bc2b8902…5f89 matches "
            "the release's checksums.txt and our own download",
            "README v0.50.0: Claude Code integration is `rtk init -g` (PreToolUse hook, native "
            "binary `rtk hook claude`); `--auto-patch` is the documented non-interactive (CI) form",
            "src/hooks/init.rs: with stdin not a TTY the settings.json patch prompt DEFAULTS TO NO, "
            "so plain `rtk init -g` in a headless container installs no hook — `--auto-patch` is required",
            "src/hooks/init.rs writes $HOME/.claude/settings.json (dirs::home_dir; CLAUDE_CONFIG_DIR "
            "is not consulted) — so the harness must NOT relocate CLAUDE_CONFIG_DIR as Harbor's "
            "stock agent does, or the hook is silently ignored",
            "src/hooks/constants.rs: hook command is literally `rtk hook claude` -> rtk must be on PATH",
            "src/analytics/gain.rs: `rtk gain --format json` -> {summary:{total_saved,...}}",
            "src/main.rs: `rtk rewrite <cmd>` exists (single source of truth for hooks) -> canary check",
        ),
        claim_unit="rtk tokens (output bytes removed / 4)",
    ),
    "headroom": Arm(
        "headroom",
        "0.38.0",
        True,
        (
            "PyPI headroom_ai-0.38.0-cp310-abi3-manylinux_2_28_x86_64.whl sha256 941d1f0c…7db7, "
            "unpacked and read (not executed)",
            'README 0.38.0 install: `uv tool install --python 3.13 "headroom-ai[all]"`; integration '
            "`headroom wrap claude`; print mode `headroom wrap claude -- -p` (click "
            "ignore_unknown_options, claude args after `--`)",
            "cli/wrap.py _start_proxy: proxy subprocess env = os.environ.copy(); proxy/server.py:6420 "
            "reads ANTHROPIC_TARGET_API_URL; providers/registry.py resolves it first -> the edge meter "
            "IS reachable: ANTHROPIC_TARGET_API_URL=<meter>. No fallback needed.",
            "cli/wrap.py claude(): also registers the headroom MCP retrieve tool and the Serena MCP "
            "(`uvx --from serena-agent serena start-mcp-server ...`, only if uvx is on PATH) at user "
            "scope, writes .claude/settings.local.json in cwd, sets ENABLE_TOOL_SEARCH=true unless set",
            "cli/savings.py: `headroom savings --json` -> report.to_dict() "
            "(lifetime.tokens_saved, cost_effective_usd, cost_usd) from ~/.headroom/savings_events.jsonl",
            "hash-locked closure (171 pkgs incl. torch + CUDA wheels) resolved with --no-build: "
            "locks/headroom-ai-0.38.0-all.cp313-x86_64-manylinux_2_28.txt",
        ),
        claim_unit="headroom tokens_saved / cost_effective_usd",
    ),
    "distil": Arm(
        "distil",
        "1.54.0",
        True,
        (
            "PyPI distil_llm-1.54.0-py3-none-any.whl sha256 c124150f…2cb4 (the published wheel, "
            "never the authors' tree); stdlib-only, lock has one line",
            "distil/cli.py: `distil wrap --upstream URL -- claude ...` (argparse REMAINDER, leading "
            "`--` stripped at cli.py:3251); proxy forwards to `_upstream + path` (http allowed)",
            "distil/onboard.py claude preset sets ENABLE_TOOL_SEARCH=true via setdefault",
            "API-key auth in a fresh HOME is not subscription_mode -> default digest tier (not the "
            "subscription lossless-only default)",
            "`distil stats --json` is the leaderboard alias (cli.py:4434), reads $DISTIL_HOME",
        ),
        claim_unit="distil tokens / USD saved",
    ),
}

CLAUDE_ARGS = (
    "--verbose",
    "--output-format=stream-json",
    "--permission-mode=bypassPermissions",
)


def unverified() -> list[str]:
    return [a.name for a in ARMS.values() if not a.verified]


# --------------------------------------------------------------------------- scripts


def _stepper(logs_dir: str, arm: str) -> list[str]:
    """Bash prelude: ``step NAME CMD...`` runs CMD with its output in a per-step log.

    On failure it prints the last 20 lines to stderr, writes ``install.json`` (arm, step,
    exit code, arch) to the logs dir Harbor syncs to the host, and exits with CMD's code —
    so an install problem reaches the preflight report instead of a 7 MB exception blob.
    Step logs are package-manager output only: no prompt, no task content.
    """
    q = shlex.quote(logs_dir)
    return [
        "set -uo pipefail",
        f'ct_logs={q}/install; mkdir -p "$ct_logs"',
        'ct_report() { printf \'{"arm": "%s", "step": "%s", "exit": %s, "arch": "%s"}\\n\' '
        f'{shlex.quote(arm)} "$1" "$2" "$(uname -m)" > {q}/install.json; }}',
        'step() { local name="$1"; shift; "$@" > "$ct_logs/$name.log" 2>&1 && return 0; '
        'local rc=$?; echo "cost-truth: install step $name FAILED (exit $rc)" >&2; '
        'tail -n 20 "$ct_logs/$name.log" >&2; ct_report "$name" "$rc"; exit "$rc"; }',
    ]


def install_script(arm: str, logs_dir: str = "/logs/agent", root: str = CT) -> str:
    """Bash, run as root in the container after Claude Code is installed. x86_64 only.

    Only the arm's own dirs are chmod-ed: the host mounts under ``root`` are read-only
    (a recursive chmod over them was the first live failure, 2026-09-25).
    """
    host = f"{root}/host"
    lines = [
        *_stepper(logs_dir, arm),
        'step arch test "$(uname -m)" = x86_64',
        f"step mkdir mkdir -p {root}/bin",
    ]
    if arm in ("headroom", "distil"):
        py, lock = (
            ("3.13", "headroom-ai-0.38.0-all.cp313-x86_64-manylinux_2_28.txt")
            if arm == "headroom"
            else ("3.12", "distil-llm-1.54.0.cp312-x86_64-manylinux_2_28.txt")
        )
        uv_env = f"env UV_CACHE_DIR={root}/uv-cache UV_PYTHON_INSTALL_DIR={root}/python"
        lines += [
            f"step uv-unpack tar -xzf {host}/uv.tar.gz -C {root}",
            f"step uv-install install -m 0755 {root}/{UV_DIR_IN_TARBALL}/uv {root}/{UV_DIR_IN_TARBALL}/uvx {root}/bin/",
            f"step venv {uv_env} {root}/bin/uv venv --python {py} {root}/{arm}",
            f"step pip {uv_env} {root}/bin/uv pip install --python {root}/{arm}/bin/python "
            f"--require-hashes --no-deps -r {host}/locks/{lock}",
            f"step chmod chmod -R a+rX {root}/{arm} {root}/python {root}/bin",
        ]
    elif arm == "rtk":
        lines += [
            f"step rtk-unpack tar -xzf {host}/rtk.tar.gz -C /usr/local/bin rtk",
            "step rtk-chmod chmod 0755 /usr/local/bin/rtk",
        ]
    lines.append("ct_report installed 0")
    return "\n".join(lines) + "\n"


def agent_setup_script(arm: str, logs_dir: str = "/logs/agent") -> str:
    """Bash, run as the agent user before the task: per-user configuration the docs prescribe."""
    if arm == "rtk":
        return "\n".join(
            [
                *_stepper(logs_dir, arm),
                _path_prefix(arm, CT),
                "step rtk-init rtk init -g --auto-patch",
                "",
            ]
        )
    return "true\n"


def arm_env(arm: str, meter_url: str, root: str = CT) -> dict[str, str]:
    """Env for the launch. Every arm's LAST hop before the provider is the meter."""
    env = dict(COMMON_ENV)
    if arm in ("control", "rtk"):
        env["ANTHROPIC_BASE_URL"] = meter_url
    elif arm == "headroom":
        env["ANTHROPIC_TARGET_API_URL"] = meter_url
        env["UV_EXCLUDE_NEWER"] = UV_EXCLUDE_NEWER  # pins uvx's Serena resolution
        env["UV_CACHE_DIR"] = f"{root}/uv-cache"
        env["UV_PYTHON_INSTALL_DIR"] = f"{root}/python"
    elif arm == "distil":
        env["CT_UPSTREAM"] = meter_url
    else:
        raise KeyError(arm)
    return env


def _path_prefix(arm: str, root: str) -> str:
    extra = {"headroom": f"{root}/headroom/bin:{root}/bin:", "distil": f"{root}/distil/bin:"}.get(
        arm, ""
    )
    return f'export PATH="{extra}$HOME/.local/bin:$PATH"'


def launcher(arm: str, claude_args: list[str]) -> list[str]:
    """The documented way each tool launches Claude Code."""
    if arm in ("control", "rtk"):
        return ["claude", *claude_args]
    if arm == "headroom":
        return ["headroom", "wrap", "claude", "--", *claude_args]
    if arm == "distil":
        return ["distil", "wrap", "--upstream", "$CT_UPSTREAM", "--", "claude", *claude_args]
    raise KeyError(arm)


def _render(argv: list[str]) -> str:
    # "$CT_UPSTREAM" must stay a shell expansion; everything else is quoted literally.
    return " ".join('"$CT_UPSTREAM"' if a == "$CT_UPSTREAM" else shlex.quote(a) for a in argv)


def launch_script(arm: str, model: str, logs_dir: str, root: str = CT) -> str:
    """Bash, run as the agent user: pipe the instruction (in $CT_INSTRUCTION) into the arm."""
    argv = launcher(arm, [*CLAUDE_ARGS, "--model", model, "--print"])
    return "\n".join(
        [
            "set -o pipefail",
            _path_prefix(arm, root),
            'ct_instruction="$CT_INSTRUCTION"; unset CT_INSTRUCTION',
            f'printf "%s" "$ct_instruction" | {_render(argv)} 2>&1 | tee {shlex.quote(logs_dir)}/claude-code.txt',
        ]
    )


def post_run_script(arm: str, logs_dir: str, root: str = CT) -> str:
    """Bash, agent user, after the run: the tool's own savings claim + the session transcript."""
    claim = {
        "rtk": "rtk gain --format json",
        "headroom": "headroom savings --json",
        "distil": "distil stats --json",
    }.get(arm)
    q = shlex.quote(logs_dir)
    lines = [_path_prefix(arm, root), f"mkdir -p {q}/sessions"]
    if claim:
        lines.append(f"{claim} > {q}/claim.json 2> {q}/claim.err || true")
    lines.append(f"cp -r $HOME/.claude/projects {q}/sessions/ 2>/dev/null || true")
    return "\n".join(lines) + "\n"


def canary_script(arm: str, model: str, logs_dir: str, nonce: str, root: str = CT) -> str:
    """Bash, agent user: prove this arm's traffic reaches the meter THROUGH the tool.

    A shim named ``claude`` is put first on PATH. The tool launches it exactly as it would
    launch Claude Code; the shim records the base URL it was handed and sends ONE tiny
    request (``max_tokens`` = 8, a nonce in ``metadata.user_id``) to it. ``claude mcp ...``
    (headroom registers MCP servers through the CLI) is passed to the real binary.
    The host then requires: the meter saw the nonce, and — for proxy arms — the URL the
    shim got is NOT the meter (so the request crossed the tool to get there).
    """
    q = shlex.quote(logs_dir)
    body = json.dumps(
        {
            "model": model,
            "max_tokens": CANARY_MAX_TOKENS,
            "metadata": {"user_id": CANARY_USER_PREFIX + nonce},
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        }
    )
    shim = "\n".join(
        [
            "#!/usr/bin/env bash",
            'case "${1:-}" in mcp|--version|doctor) exec "$HOME/.local/bin/claude" "$@";; esac',
            f'printf \'{{"base_url": "%s"}}\' "$ANTHROPIC_BASE_URL" > {q}/canary.json',
            'curl -sS --max-time 120 -o /dev/null -w "%{http_code}" -X POST "${ANTHROPIC_BASE_URL%/}/v1/messages" '
            '-H "content-type: application/json" -H "anthropic-version: 2023-06-01" '
            f'-H "x-api-key: $ANTHROPIC_API_KEY" --data {shlex.quote(body)} > {q}/canary.status',
        ]
    )
    lines = [
        "set -uo pipefail",
        f"mkdir -p {root}/canary {q}",
        f"cat > {root}/canary/claude <<'CT_SHIM'\n{shim}\nCT_SHIM",
        f"chmod 0755 {root}/canary/claude",
        _path_prefix(arm, root),
        f'export PATH="{root}/canary:$PATH"',
        f"{_render(launcher(arm, ['--print']))} < /dev/null > {q}/canary.out 2>&1",
        f"echo $? > {q}/canary.exit",
    ]
    if arm == "rtk":
        lines += [
            f'grep -q "rtk hook claude" "$HOME/.claude/settings.json" && echo 1 > {q}/rtk.hook || echo 0 > {q}/rtk.hook',
            f'rtk rewrite "git status" > {q}/rtk.rewrite 2>&1 || true',
        ]
    return "\n".join(lines) + "\n"
