"""Offline STAND-INS for the local (mock-upstream) executor. Not the competitors.

The live run executes the real, pinned tools inside task containers. Tests cannot (we do
not run competitor code on the host, and never run ``claude``), so the local executor puts
these on PATH instead:

* ``claude``   — a scripted multi-turn client (``mock.scripted_agent``) that talks to
                 whatever ``ANTHROPIC_BASE_URL`` it was handed; ``claude mcp ...`` exits 0.
* ``headroom`` — ``wrap claude -- ARGS``: a pass-through proxy whose upstream is
                 ``ANTHROPIC_TARGET_API_URL`` (the verified headroom contract), then runs
                 ``claude ARGS`` against it; ``savings --json`` prints an empty report.
* ``rtk``      — ``init -g --auto-patch`` writes the hook entry into
                 ``$HOME/.claude/settings.json``; ``rewrite CMD`` prints ``rtk CMD``;
                 ``gain --format json`` prints an empty summary.

distil is NOT stood in: the local executor runs the real ``distil wrap`` from this tree.
They prove the harness's plumbing (scripts, env, chain, canary, claims); they say nothing
about what any tool saves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.cost_truth import meter as m
from benchmarks.cost_truth import mock


def _claude(argv: list[str]) -> int:
    if argv[:1] in (["mcp"], ["--version"]):
        return 0
    sys.stdin.read()  # the instruction; the script is keyed on task/seed/arm instead
    model = argv[argv.index("--model") + 1] if "--model" in argv else "claude-sonnet-5"
    res = mock.scripted_agent(
        os.environ["ANTHROPIC_BASE_URL"],
        os.environ["CT_TASK"],
        int(os.environ["CT_SEED"]),
        os.environ["CT_ARM"],
        model,
    )
    Path(os.environ["CT_RESULT"]).write_text(
        json.dumps({"solved": res.solved, "claim": res.claimed_tokens_saved})
    )
    return 0


def _headroom(argv: list[str]) -> int:
    if argv[:1] == ["savings"]:
        print(json.dumps({"lifetime": {"tokens_saved": 0}, "stand_in": True}))
        return 0
    assert argv[:2] == ["wrap", "claude"], argv
    args = argv[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    cfg = m.MeterConfig(
        upstream=os.environ["ANTHROPIC_TARGET_API_URL"],
        log_path=Path(os.environ["HOME"]) / ".headroom-standin.jsonl",
        spend=m.SpendMeter(1e9),
        run_id="headroom-standin",
    )
    with m.UsageMeter(cfg) as proxy:  # a pass-through proxy; its own log is discarded
        env = {**os.environ, "ANTHROPIC_BASE_URL": proxy.base_url}
        return subprocess.call(["claude", *args], env=env)


def _rtk(argv: list[str]) -> int:
    if argv[:1] == ["init"]:
        settings = Path(os.environ["HOME"]) / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        hook = {"matcher": "Bash", "hooks": [{"type": "command", "command": "rtk hook claude"}]}
        settings.write_text(json.dumps({"hooks": {"PreToolUse": [hook]}}))
        return 0
    if argv[:1] == ["rewrite"]:
        print("rtk " + " ".join(argv[1:]))
        return 0
    if argv[:1] == ["gain"]:
        print(json.dumps({"summary": {"total_saved": 0}, "stand_in": True}))
        return 0
    return 2


TOOLS = {"claude": _claude, "headroom": _headroom, "rtk": _rtk}


def install(bin_dir: Path, repo_root: Path) -> None:
    """Write executable stand-ins for every tool into ``bin_dir``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in TOOLS:
        p = bin_dir / name
        p.write_text(
            f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(repo_root)!r})\n"
            "from benchmarks.cost_truth.standins import TOOLS\n"
            f"sys.exit(TOOLS[{name!r}](sys.argv[1:]))\n"
        )
        p.chmod(0o755)
