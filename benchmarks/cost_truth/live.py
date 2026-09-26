"""The live driver: approval gate, tool preparation, canary preflight, paired dispatch.

``python -m benchmarks.cost_truth live --phase pilot --i-approve-spend 92.07``

Order of operations (each step fails closed):

1. **Approval** — ``--i-approve-spend`` must equal the phase's pre-registered hard cap to the
   cent; any arm spec still unverified refuses the run.
2. **Prepare** — pinned artifacts are downloaded once into the tools dir and sha256-checked;
   the hash-locked requirement files are copied next to them. The dir is bind-mounted
   read-only into every task container.
3. **Canary preflight** — one tiny request per arm, launched through the arm's real
   integration, must reach the meter; for proxy arms the URL Claude Code was handed must NOT
   be the meter (so the request crossed the tool). RTK must also show its hook installed and
   its rewriter answering. Any failure aborts before a single task runs.
4. **Dispatch** — the pre-registered schedule, cache-TTL spacing, concurrency, one hard cap
   shared by every meter. Each run gets its own meter on its own port.

Executors: ``HarborExecutor`` (live: ``harbor trial start`` per run, the real tools in the
task container) and ``LocalExecutor`` (tests: the same scripts on the host against the mock
upstream, with stand-ins for claude/headroom/rtk and the real ``distil wrap``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from benchmarks.cost_truth import analysis as an
from benchmarks.cost_truth import arms
from benchmarks.cost_truth import meter as m
from benchmarks.cost_truth import runner as rn

REPO_ROOT = Path(__file__).resolve().parents[2]
ANTHROPIC_API = "https://api.anthropic.com"

#: Pre-registered design (protocol §7). tasks x seeds x arms per model.
DESIGN: dict[str, dict[str, Any]] = {
    "pilot": {"model": "claude-sonnet-5", "tasks": 10, "seeds": 2},
    "primary": {"model": "claude-sonnet-5", "tasks": 89, "seeds": 5},
    "replication": {"model": "claude-haiku-4-5", "tasks": 89, "seeds": 3},
}
#: One Terminal-Bench 2.1 attempt under Claude Code (assumption, protocol §9).
PROFILE = {"turns": 30, "prefix": 22_000, "new_per_turn": 1_800, "out_per_turn": 450}
SAFETY = 1.25
SCHEDULE_SEED = 20260925  # committed at freeze (protocol §5.2)
PLATFORM = "linux/amd64"
AGENT_SETUP_TIMEOUT_S = 3600


class ApprovalError(RuntimeError):
    """The run was not approved for exactly this phase's cap."""


class PreflightError(RuntimeError):
    """A pinned artifact, an arm spec, or the canary chain proof failed."""


def phase_cap(phase: str) -> float:
    d = DESIGN[phase]
    per = an.price_tokens(d["model"], an.attempt_tokens(**PROFILE))
    return round(per * d["tasks"] * d["seeds"] * len(rn.ARMS) * SAFETY, 2)


def check_approval(phase: str, approved: str | None) -> float:
    """Return the cap iff ``approved`` is exactly this phase's cap (to the cent)."""
    if phase not in DESIGN:
        raise ApprovalError(f"unknown phase {phase!r}; one of {sorted(DESIGN)}")
    cap = phase_cap(phase)
    if approved is None:
        raise ApprovalError(
            f"refusing to spend: pass --i-approve-spend {cap:.2f} (the {phase} hard cap)"
        )
    try:
        value = float(approved)
    except ValueError:
        raise ApprovalError(f"--i-approve-spend must be a number, got {approved!r}") from None
    if not math.isfinite(value) or round(value, 2) != cap:
        raise ApprovalError(
            f"--i-approve-spend {approved} does not match the {phase} hard cap ${cap:.2f}"
        )
    if arms.unverified():
        raise ApprovalError(f"arm specs not verified: {', '.join(arms.unverified())}")
    return cap


# --------------------------------------------------------------------------- tools


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - pinned https URLs, sha256-checked after


def prepare_tools(
    tools_dir: Path, fetch: Callable[[str, Path], None] = _download
) -> dict[str, str]:
    """Download (once) and verify every pinned artifact; copy the locks. Returns name->sha.

    Layout: ``host/`` (artifacts + locks) is mounted READ-ONLY; ``uv-cache/`` is a sibling
    mounted read-write. Keeping the cache out of ``host/`` keeps the read-only mount small
    and immutable.
    """
    host = tools_dir / "host"
    host.mkdir(parents=True, exist_ok=True)
    (tools_dir / "uv-cache").mkdir(exist_ok=True)
    out: dict[str, str] = {}
    for name, (url, want) in arms.ARTIFACTS.items():
        dest = host / name
        if not dest.exists():
            fetch(url, dest)
        got = sha256(dest)
        if got != want:
            dest.unlink()
            raise PreflightError(f"{name}: sha256 {got} != pinned {want}; deleted")
        out[name] = got
    shutil.copytree(arms.LOCKS, host / "locks", dirs_exist_ok=True)
    return out


# --------------------------------------------------------------------------- outcomes


@dataclass
class Outcome:
    status: str  # one of runner.COUNTED + runner.EXCLUDED, before meter-based overrides
    claim: dict[str, Any] | None = None
    agent_wall_s: float | None = None
    detail: str = ""
    install: dict[str, Any] | None = None


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))  # py<3.11 rejects "Z"


def classify_harbor(result: dict[str, Any]) -> Outcome:
    """Map a Harbor ``result.json`` to a protocol §10 status."""
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward", min(rewards.values()) if rewards else 0)
    info = result.get("exception_info") or {}
    exc = info.get("exception_type") or ""
    msg = (info.get("exception_message") or "").strip().splitlines()
    if exc and msg:  # the last line only, capped: Harbor embeds whole command outputs here
        exc = f"{exc}: {msg[-1][:300]}"
    ex = result.get("agent_execution") or {}
    wall = None
    if ex.get("started_at") and ex.get("finished_at"):
        wall = (_ts(ex["finished_at"]) - _ts(ex["started_at"])).total_seconds()
    if reward is not None and float(reward) >= 1:
        return Outcome("solved", agent_wall_s=wall, detail=exc)
    if exc.startswith("AgentTimeoutError"):
        return Outcome("timeout", agent_wall_s=wall, detail=exc)
    if not ex.get("started_at"):  # never reached the agent: build/setup/install failure
        return Outcome("infra_error", detail=exc or "agent never started")
    return Outcome("failed", agent_wall_s=wall, detail=exc)


def read_install(logs_dir: Path) -> dict[str, Any] | None:
    """``install.json`` from the in-container step wrapper, plus the failing step's tail."""
    try:
        doc = json.loads((logs_dir / "install.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    if doc.get("exit"):
        log = logs_dir / "install" / f"{doc.get('step')}.log"
        lines = log.read_text(errors="replace").splitlines() if log.exists() else []
        doc["tail"] = [ln[:300] for ln in lines[-20:]]
    return doc


def read_claim(logs_dir: Path) -> dict[str, Any] | None:
    p = logs_dir / "claim.json"
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def verify_canary(
    arm: str, meter_log: list[dict[str, Any]], logs_dir: Path, meter_url: str
) -> list[str]:
    """Reasons the chain proof failed; empty means proven."""
    problems = []
    inst = read_install(logs_dir)
    if inst and inst.get("exit"):
        problems.append(f"install step {inst.get('step')!r} failed with exit {inst.get('exit')}")
    if not any(r.get("canary") and r.get("status") == 200 for r in meter_log):
        problems.append("meter never saw the canary request (or the provider refused it)")
    try:
        handed = json.loads((logs_dir / "canary.json").read_text())["base_url"].rstrip("/")
    except (OSError, ValueError, KeyError):
        handed = ""
        problems.append("the claude shim was never launched by the arm")
    proxy_arm = arm in ("headroom", "distil")
    if handed and proxy_arm and handed == meter_url.rstrip("/"):
        problems.append("claude was pointed straight at the meter: the tool's proxy was bypassed")
    if handed and not proxy_arm and handed != meter_url.rstrip("/"):
        problems.append(f"claude was pointed at {handed}, not the meter")
    if arm == "rtk":

        def read(name: str) -> str:
            p = logs_dir / name
            return p.read_text().strip() if p.exists() else ""

        if read("rtk.hook") != "1":
            problems.append("rtk hook not in $HOME/.claude/settings.json")
        rw = read("rtk.rewrite")
        if not rw.startswith("rtk"):
            problems.append(f"`rtk rewrite 'git status'` did not rewrite (got {rw[:40]!r})")
    return problems


# --------------------------------------------------------------------------- executors


class Executor(Protocol):
    def run(
        self, spec: rn.RunSpec, meter_url: str, logs_dir: Path, mode: str, nonce: str
    ) -> Outcome: ...


class LocalExecutor:
    """Runs the SAME arm scripts on the host with stand-ins; for tests and dry runs only."""

    def __init__(self, work: Path, distil_bin: Path) -> None:
        self.work, self.distil_bin = work, distil_bin

    def run(
        self, spec: rn.RunSpec, meter_url: str, logs_dir: Path, mode: str, nonce: str
    ) -> Outcome:
        from benchmarks.cost_truth import standins

        run_dir = self.work / f"{spec.run_id}.{mode}"
        env = rn.isolated_env(run_dir, dict(os.environ), {})
        home, root = Path(env["HOME"]), run_dir / "ct"
        standins.install(home / ".local" / "bin", REPO_ROOT)
        (root / "headroom" / "bin").mkdir(parents=True)
        (root / "distil" / "bin").mkdir(parents=True)
        (root / "headroom" / "bin" / "headroom").symlink_to(home / ".local" / "bin" / "headroom")
        (root / "distil" / "bin" / "distil").symlink_to(self.distil_bin)
        logs_dir.mkdir(parents=True, exist_ok=True)
        result = run_dir / "result.json"
        env.update(arms.arm_env(spec.arm, meter_url, root=str(root)))
        env.update(
            {
                "ANTHROPIC_API_KEY": "dry-run-no-key",
                "CT_TASK": spec.task,
                "CT_SEED": str(spec.seed),
                "CT_ARM": spec.arm,
                "CT_RESULT": str(result),
                "DISTIL_NO_UPDATE_CHECK": "1",
            }
        )

        def sh(script: str, extra: dict[str, str] | None = None) -> int:
            with (run_dir / "exec.log").open("a") as log:
                return subprocess.call(
                    ["bash", "-c", script],
                    env={**env, **(extra or {})},
                    cwd=run_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

        if mode == "canary":
            sh(arms.agent_setup_script(spec.arm, str(logs_dir)))
            sh(arms.canary_script(spec.arm, spec.model, str(logs_dir), nonce, root=str(root)))
            return Outcome("solved")
        t0 = time.monotonic()
        sh(arms.agent_setup_script(spec.arm, str(logs_dir)))
        code = sh(
            arms.launch_script(spec.arm, spec.model, str(logs_dir), root=str(root)),
            {"CT_INSTRUCTION": "solve"},
        )
        wall = time.monotonic() - t0
        sh(arms.post_run_script(spec.arm, str(logs_dir), root=str(root)))
        try:
            solved = bool(json.loads(result.read_text())["solved"])
        except (OSError, ValueError, KeyError):
            return Outcome("failed", read_claim(logs_dir), wall, f"agent exit {code}, no result")
        return Outcome(
            "solved" if solved else "failed", read_claim(logs_dir), wall, f"agent exit {code}"
        )


class HarborExecutor:
    """Live: one ``harbor trial start`` per run, real tools inside the task container."""

    def __init__(
        self, tasks_dir: Path, trials_dir: Path, tools_dir: Path, url_host: str, harbor: list[str]
    ) -> None:
        self.tasks_dir, self.trials_dir, self.tools_dir = tasks_dir, trials_dir, tools_dir
        self.url_host, self.harbor = url_host, harbor

    def argv(self, spec: rn.RunSpec, meter_url: str, mode: str, nonce: str) -> list[str]:
        mounts = [
            {
                "type": "bind",
                "source": str(self.tools_dir / "host"),
                "target": arms.HOST_MOUNT,
                "read_only": True,
            },
            {
                "type": "bind",
                "source": str(self.tools_dir / "uv-cache"),
                "target": arms.UV_CACHE_MOUNT,
            },
        ]
        return [
            *self.harbor,
            "trial",
            "start",
            "--path",
            str(self.tasks_dir / spec.task),
            "--agent",  # an import path; --agent-import-path is deprecated in harbor 0.23
            "benchmarks.cost_truth.harbor_agent:CostTruthAgent",
            "--model",
            f"anthropic/{spec.model}",
            "--agent-kwarg",
            f"ct_arm={spec.arm}",
            "--agent-kwarg",
            f"ct_meter={meter_url}",
            "--agent-kwarg",
            f"ct_mode={mode}",
            "--agent-kwarg",
            f"ct_nonce={nonce}",
            "--trial-name",
            f"{spec.run_id}.{mode}",
            "--trials-dir",
            str(self.trials_dir),
            "--mounts",
            json.dumps(mounts),
            "--allow-agent-host",
            self.url_host,
            # installs are not an outcome; under amd64 emulation Claude Code's own install
            # alone took 4.5 min, and Harbor's 360 s default killed two arms (2026-09-25)
            "--agent-setup-timeout",
            str(AGENT_SETUP_TIMEOUT_S),
        ]

    def run(
        self, spec: rn.RunSpec, meter_url: str, logs_dir: Path, mode: str, nonce: str
    ) -> Outcome:
        # Every task container is linux/amd64 for every arm: 85 of 89 Terminal-Bench 2.1
        # images are amd64-only, and the other 4 must not run natively while the rest are
        # emulated (Amendment 2). Harbor passes os.environ through to docker compose.
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "DOCKER_DEFAULT_PLATFORM": PLATFORM}
        proc = subprocess.run(
            self.argv(spec, meter_url, mode, nonce), env=env, capture_output=True, text=True
        )
        trial = self.trials_dir / f"{spec.run_id}.{mode}"
        try:
            result = json.loads((trial / "result.json").read_text())
        except (OSError, ValueError):
            return Outcome("infra_error", detail=f"harbor exit {proc.returncode}, no result.json")
        agent_logs = trial / "agent"
        if agent_logs.exists() and agent_logs != logs_dir:
            shutil.copytree(agent_logs, logs_dir, dirs_exist_ok=True)
        out = classify_harbor(result)
        out.claim = read_claim(logs_dir)
        out.install = read_install(logs_dir)
        return out


# --------------------------------------------------------------------------- dispatch


@dataclass
class LiveConfig:
    phase: str
    cap_usd: float
    out_dir: Path
    upstream: str = ANTHROPIC_API
    bind: str = "127.0.0.1"
    url_host: str = "host.docker.internal"
    concurrency: int = 4
    min_gap_s: float = rn.MIN_GAP_S
    tasks: list[str] | None = None  # the frozen manifest; pilot samples from it
    seeds: list[int] | None = None
    #: agent logs hold transcripts (content): keep them out of the tracked results dir
    logs_root: Path | None = None


def pilot_tasks(manifest: list[str], n: int, seed: int = SCHEDULE_SEED) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(manifest), n))


def _metered(
    cfg: LiveConfig, spend: m.SpendMeter, spec: rn.RunSpec, executor: Executor, mode: str
) -> tuple[Outcome, list[dict[str, Any]], Path, str]:
    nonce = uuid.uuid4().hex if mode == "canary" else ""
    log_path = cfg.out_dir / "meter" / f"{spec.run_id}.{mode}.jsonl"
    logs_dir = (cfg.logs_root or cfg.out_dir / "agent-logs") / f"{spec.run_id}.{mode}"
    mcfg = m.MeterConfig(
        cfg.upstream, log_path, spend, spec.run_id, bind=cfg.bind, canary_nonce=nonce or None
    )
    with m.UsageMeter(mcfg) as meter:
        url = f"http://{cfg.url_host}:{meter.port}"
        try:
            out = executor.run(spec, url, logs_dir, mode, nonce)
        except (OSError, subprocess.SubprocessError) as e:
            out = Outcome("arm_crash", detail=f"{type(e).__name__}: {e}")
    return out, m.read_log(log_path), logs_dir, url


def preflight_canaries(
    cfg: LiveConfig, spend: m.SpendMeter, executor: Executor, task: str, model: str
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for arm in rn.ARMS:
        spec = rn.RunSpec(task, -1, arm, model, -1)
        out, log, logs_dir, url = _metered(cfg, spend, spec, executor, "canary")
        problems = verify_canary(arm, log, logs_dir, url)
        report[arm] = {
            "ok": not problems,
            "problems": problems,
            "trial_status": out.status,
            "trial_detail": out.detail,
            "install": out.install,
            "cost_usd": sum(r["cost_usd"] for r in log),
        }
    (cfg.out_dir / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    bad = {a: r["problems"] for a, r in report.items() if not r["ok"]}
    if bad:
        raise PreflightError(f"canary chain proof failed: {bad}")
    return report


def run_phase(
    cfg: LiveConfig, executor: Executor, preflight_only: bool = False
) -> list[dict[str, Any]]:
    d = DESIGN[cfg.phase]
    tasks = cfg.tasks or []
    if cfg.phase == "pilot" and len(tasks) > d["tasks"]:
        tasks = pilot_tasks(tasks, d["tasks"])
    seeds = cfg.seeds if cfg.seeds is not None else list(range(d["seeds"]))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    spend = m.SpendMeter(cfg.cap_usd)
    preflight_canaries(cfg, spend, executor, tasks[0], d["model"])
    if preflight_only:
        return []

    pending = rn.build_schedule(tasks, seeds, rn.ARMS, d["model"], SCHEDULE_SEED)
    runs: list[dict[str, Any]] = []
    last_end: dict[str, float] = {}
    lock = threading.Lock()
    stop = threading.Event()
    retried: set[str] = set()

    def one(spec: rn.RunSpec) -> None:
        out, log, _, _ = _metered(cfg, spend, spec, executor, "run")
        rec = rn.summarise_run(
            spec, out.status, log, wall_s=out.agent_wall_s, claim=out.claim, detail=out.detail[:200]
        )
        with lock:
            last_end[spec.task] = time.monotonic()
            if rec["status"] == "budget_stop" or spend.exhausted:
                stop.set()
            if rec["status"] == "infra_error" and spec.run_id not in retried:
                retried.add(spec.run_id)  # protocol §10: re-run once at the end of the schedule
                pending.append(spec)
            runs.append(rec)
            with (cfg.out_dir / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    inflight: dict[Future[None], str] = {}
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        while (pending or inflight) and not stop.is_set():
            with lock:
                busy = set(inflight.values())
                now = time.monotonic()
                eligible = [r for r in pending if r.task not in busy]
                i = (
                    rn.next_eligible(eligible, last_end, now, cfg.min_gap_s)
                    if len(inflight) < cfg.concurrency
                    else None
                )
                if i is not None:
                    spec = eligible[i]
                    pending.remove(spec)
                    inflight[pool.submit(one, spec)] = spec.task
            for fut in [f for f in inflight if f.done()]:
                fut.result()
                del inflight[fut]
            if i is None:
                time.sleep(0.05)
        for fut in inflight:
            fut.result()

    manifest = {
        "mode": "live",
        "phase": cfg.phase,
        "model": d["model"],
        "cap_usd": cfg.cap_usd,
        "spent_usd": round(spend.spent, 6),
        "truncated": stop.is_set(),
        "tasks": tasks,
        "seeds": seeds,
        "arms": list(rn.ARMS),
        "schedule_seed": SCHEDULE_SEED,
        "min_gap_s": cfg.min_gap_s,
        "versions": {a.name: a.version for a in arms.ARMS.values()},
        "claude_code": arms.CLAUDE_CODE_VERSION,
        "harbor": arms.HARBOR_VERSION,
    }
    (cfg.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return runs


def pilot_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """What the pilot is FOR (protocol §7) — and nothing that compares arms."""
    by: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        if r["status"] in rn.COUNTED and r["cost_usd"] > 0:
            by.setdefault((r["task"], r["arm"]), []).append(math.log(r["cost_usd"]))
    groups = [v for v in by.values() if len(v) > 1]
    sigma_w = (
        math.sqrt(statistics.fmean([statistics.variance(v) for v in groups]))
        if groups
        else float("nan")
    )
    blocks: dict[tuple[str, int], list[bool]] = {}
    for r in runs:
        if r["status"] in rn.COUNTED:
            blocks.setdefault((r["task"], r["seed"]), []).append(r["solved"])
    discord = [len(set(v)) > 1 for v in blocks.values() if len(v) > 1]
    counted = [r for r in runs if r["status"] in rn.COUNTED]

    def med(key: str) -> float:
        return statistics.median([r["usage"][key] for r in counted]) if counted else float("nan")

    return {
        "sigma_within_log_cost": sigma_w,
        "block_discordance": sum(discord) / len(discord) if discord else float("nan"),
        "median_tokens_per_attempt": {
            k: med(k) for k in (*m.USAGE_KEYS, "cache_creation_1h_input_tokens")
        },
        "median_turns": statistics.median([r["turns"] for r in counted])
        if counted
        else float("nan"),
        "any_1h_cache_writes": any(
            r["usage"]["cache_creation_1h_input_tokens"] > 0 for r in counted
        ),
        "mde_ratio_89x5_at_this_sigma": an.mde_ratio(89, sigma_w, 0.15, 5)
        if groups
        else float("nan"),
        "note": "pooled over arms on purpose: the pilot does not compare arms (protocol §7)",
    }
