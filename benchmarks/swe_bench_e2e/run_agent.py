#!/usr/bin/env python3
"""Run the coding agent (aider) on SWE-bench instances and emit prediction patches.

For each instance and each condition (full / distil_trunc500 / llmlingua2) this:

1. checks out the instance ``repo`` at ``base_commit`` in an isolated git worktree
   (one bare-ish clone per repo, cached and reused across its instances);
2. starts the E7 compression proxy for that condition (``full`` => transparent) and
   points aider's Anthropic client at it via ``ANTHROPIC_BASE_URL``;
3. runs ``aider`` non-interactively on the ``problem_statement`` (no oracle files — the
   agent must localise from the issue text alone, the honest SWE-bench setting);
4. captures the working-tree diff as the ``model_patch`` and writes one prediction row
   plus the proxy's per-instance compression/usage stats.

Agent: **aider** (chosen over SWE-agent/OpenHands for lowest setup friction — a single
isolated CLI, no per-instance Docker, produces a clean git diff). Model
``anthropic/claude-sonnet-4-6`` at ``temperature 0``. The Anthropic API has no request
seed, so determinism comes from ``temperature 0``; the only deliberate randomness in
Phase 5 is the seed-1729 instance sample.

Predictions are written incrementally to ``predictions/<condition>.jsonl`` and skipped
on resume, so an interrupted run never loses completed work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.swe_bench_e2e.compress_proxy import COMPRESSORS, CompressStats, serve

ROOT = Path(__file__).resolve().parents[2]
MODEL = "anthropic/claude-sonnet-4-6"
TEMPERATURE = 0.0
AIDER_BIN = os.environ.get("AIDER_BIN", str(Path.home() / ".local/bin/aider"))
CONDITIONS = ("full", "distil_trunc500", "llmlingua2")

# git worktree add/remove mutate shared state under a clone's .git; serialise per-clone.
_CLONE_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)
_CLONE_LOCKS_GUARD = threading.Lock()


def _clone_lock(clone: Path) -> threading.Lock:
    with _CLONE_LOCKS_GUARD:
        return _CLONE_LOCKS[str(clone)]


# --------------------------------------------------------------------------- #
# Repo / worktree management
# --------------------------------------------------------------------------- #
def _run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


def ensure_clone(repo: str, cache_dir: Path) -> Path:
    """Clone ``owner/name`` once into ``cache_dir/owner__name`` (full history)."""
    dest = cache_dir / repo.replace("/", "__")
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    res = _run(["git", "clone", "--quiet", url, str(dest)], timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(f"clone {repo} failed: {res.stderr[-500:]}")
    return dest


def make_worktree(clone: Path, base_commit: str, work_root: Path) -> Path:
    """Create a detached worktree of ``clone`` at ``base_commit``."""
    work_root.mkdir(parents=True, exist_ok=True)
    wt = Path(tempfile.mkdtemp(dir=str(work_root)))
    # Ensure the commit is present (older clones may lack it only if shallow; ours is full).
    res = _run(
        ["git", "worktree", "add", "--detach", str(wt), base_commit],
        cwd=clone,
        timeout=600,
    )
    if res.returncode != 0:
        # fetch the specific commit then retry
        _run(["git", "fetch", "--quiet", "origin", base_commit], cwd=clone, timeout=600)
        res = _run(
            ["git", "worktree", "add", "--detach", str(wt), base_commit],
            cwd=clone,
            timeout=600,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"worktree add @ {base_commit} failed: {res.stderr[-500:]}"
            )
    return wt


def remove_worktree(clone: Path, wt: Path) -> None:
    _run(["git", "worktree", "remove", "--force", str(wt)], cwd=clone, timeout=120)
    if wt.exists():
        shutil.rmtree(wt, ignore_errors=True)


# aider's own scratch files + our prompt file must never leak into the model_patch.
_EXCLUDES = [":(exclude).aider*", ":(exclude).e7_problem.txt", ":(exclude)**/.aider*"]


def capture_patch(wt: Path) -> str:
    """Return the working-tree diff (incl. new files) as a SWE-bench model_patch.

    aider's chat/history/cache files and our prompt file are excluded so the patch
    contains only the agent's source edits.
    """
    _run(["git", "add", "-A", "-N", "--", ".", *_EXCLUDES], cwd=wt, timeout=120)
    res = _run(
        ["git", "diff", "--no-color", "--", ".", *_EXCLUDES], cwd=wt, timeout=120
    )
    return res.stdout


# --------------------------------------------------------------------------- #
# Aider invocation
# --------------------------------------------------------------------------- #
def run_aider(
    wt: Path, problem: str, base_url: str, api_key: str, timeout: float
) -> dict[str, Any]:
    """Run aider one-shot on the problem statement inside ``wt``. Returns run metadata."""
    prompt_file = wt / ".e7_problem.txt"
    prompt_file.write_text(problem)
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = api_key
    env["ANTHROPIC_BASE_URL"] = (
        base_url  # litellm/anthropic honour this; proxy forwards upstream
    )
    env["AIDER_ANTHROPIC_API_BASE"] = base_url
    env.setdefault(
        "OPENAI_API_KEY", "sk-unused"
    )  # aider sometimes probes; keep it quiet
    cmd = [
        AIDER_BIN,
        "--model",
        MODEL,
        # aider 0.86.2 doesn't recognise claude-sonnet-4-6 and would fall back to the
        # "whole" edit format (model rewrites entire files -> 25-45k-token outputs that
        # take 600-900s and time out on large repos). Force concise search/replace diffs:
        # ~10-50x smaller outputs, far faster, more reliable, identical across conditions.
        "--edit-format",
        "diff",
        "--no-auto-commits",  # aider edits the worktree but never commits; we capture the diff
        "--no-dirty-commits",
        "--yes-always",
        "--no-check-update",
        "--no-show-model-warnings",
        "--no-stream",
        "--map-tokens",
        "2048",
        # aider 0.86.2 sends temperature=0 for claude-sonnet-4-6 by default
        # (models.send_completion: use_temperature=True -> temperature=0); the
        # Anthropic API exposes no request seed, so determinism is temperature-driven.
        "--message-file",
        str(prompt_file),
    ]
    t0 = time.time()
    try:
        res = _run(cmd, cwd=wt, env=env, timeout=timeout)
        status = "ok" if res.returncode == 0 else f"exit{res.returncode}"
        tail = (
            (res.stdout or "")[-2000:] + "\n--STDERR--\n" + (res.stderr or "")[-1000:]
        )
    except subprocess.TimeoutExpired:
        status, tail = "timeout", "aider timed out"
    finally:
        prompt_file.unlink(missing_ok=True)
    return {"status": status, "seconds": round(time.time() - t0, 1), "log_tail": tail}


# --------------------------------------------------------------------------- #
# Per-instance driver
# --------------------------------------------------------------------------- #
def run_instance(
    inst: dict[str, Any],
    condition: str,
    api_key: str,
    cache_dir: Path,
    work_root: Path,
    timeout: float,
) -> dict[str, Any]:
    iid = inst["instance_id"]
    clone = ensure_clone(inst["repo"], cache_dir)
    with _clone_lock(clone):
        wt = make_worktree(clone, inst["base_commit"], work_root)
    # The problem statement is the task, not "file content the agent reads" — never
    # compress it (would handicap B/C for the wrong reason). Pass it as the protected text.
    httpd, state = serve(
        compressor=COMPRESSORS[condition], protect=inst["problem_statement"]
    )
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        meta = run_aider(wt, inst["problem_statement"], base_url, api_key, timeout)
        patch = capture_patch(wt)
    finally:
        httpd.shutdown()
        with _clone_lock(clone):
            remove_worktree(clone, wt)
    stats: CompressStats = state.stats
    return {
        "instance_id": iid,
        "model_name_or_path": f"distil-e7-{condition}",
        "model_patch": patch,
        "_condition": condition,
        "_agent": "aider",
        "_model": MODEL,
        "_temperature": TEMPERATURE,
        "_run": meta,
        "_compress": asdict(stats),
        "_empty_patch": not patch.strip(),
    }


def load_sample(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data["instances"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample",
        type=Path,
        default=ROOT / "docs/paper/results/swe_e2e/sample_full.json",
    )
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs/paper/results/swe_e2e")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / ".e7_cache/repos")
    ap.add_argument("--work-root", type=Path, default=ROOT / ".e7_cache/work")
    ap.add_argument(
        "--timeout", type=float, default=900.0, help="per-instance aider timeout (s)"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only first N instances (smoke)"
    )
    ap.add_argument(
        "--only", type=str, default=None, help="comma-separated instance_ids to run"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent instances (each: own proxy + worktree)",
    )
    args = ap.parse_args()

    from dotenv import dotenv_values

    api_key = dotenv_values(ROOT / ".env").get("ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY", ""
    )
    if not api_key:
        raise SystemExit("no ANTHROPIC_API_KEY available (.env or env)")

    instances = load_sample(args.sample)
    if args.only:
        want = set(args.only.split(","))
        instances = [i for i in instances if i["instance_id"] in want]
    if args.limit:
        instances = instances[: args.limit]

    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"{args.condition}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["instance_id"])

    todo = [i for i in instances if i["instance_id"] not in done]
    print(
        f"condition={args.condition}: {len(todo)} to run, {len(done)} already done, "
        f"workers={args.workers}",
        flush=True,
    )

    # Pre-clone every unique repo serially so the parallel phase only does worktree adds
    # (clone-from-scratch is the one git op that races badly across threads).
    for repo in sorted({i["repo"] for i in todo}):
        ensure_clone(repo, args.cache_dir)

    write_lock = threading.Lock()
    done_count = [0]

    def work(inst: dict[str, Any]) -> None:
        iid = inst["instance_id"]
        try:
            row = run_instance(
                inst,
                args.condition,
                api_key,
                args.cache_dir,
                args.work_root,
                args.timeout,
            )
        except Exception as e:  # noqa: BLE001 — record failure, keep going
            row = {
                "instance_id": iid,
                "model_name_or_path": f"distil-e7-{args.condition}",
                "model_patch": "",
                "_condition": args.condition,
                "_error": str(e)[:500],
                "_empty_patch": True,
            }
        c = row.get("_compress", {})
        with write_lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done_count[0] += 1
            print(
                f"[{args.condition} {done_count[0]}/{len(todo)}] {iid} "
                f"status={row.get('_run', {}).get('status', 'err')} "
                f"empty={row['_empty_patch']} "
                f"blk={c.get('blocks_compressed', 0)}/{c.get('blocks_seen', 0)} "
                f"in={c.get('usage_input_tokens', 0)} out={c.get('usage_output_tokens', 0)} "
                f"t={row.get('_run', {}).get('seconds', 0)}s",
                flush=True,
            )

    if args.workers <= 1:
        for inst in todo:
            work(inst)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))


if __name__ == "__main__":
    main()
