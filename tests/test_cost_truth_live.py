"""Cost-truth live driver: approval gate, pinned tools, arm scripts, canary chain proof,
Harbor glue, and the whole live pipeline end to end against the MOCK upstream ($0)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from benchmarks.cost_truth import arms
from benchmarks.cost_truth import live
from benchmarks.cost_truth import meter as m
from benchmarks.cost_truth import mock
from benchmarks.cost_truth import runner as rn
from benchmarks.cost_truth.__main__ import main

ESTIMATE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "results"
    / "cost_truth"
    / "cost_estimate.json"
)
DISTIL_BIN = Path(sys.executable).parent / "distil"
needs_bash = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("curl") is None, reason="bash+curl"
)

# --------------------------------------------------------------------------- approval gate


def test_pilot_cap_matches_the_published_estimate() -> None:
    est = json.loads(ESTIMATE.read_text())
    for phase in live.DESIGN:
        assert live.phase_cap(phase) == est["phases"][phase]["hard_cap_usd"]
    assert live.phase_cap("pilot") == 92.07


@pytest.mark.parametrize("approved", [None, "50", "92.06", "92.08", "nan", "inf", "92.07USD", ""])
def test_approval_refuses_anything_but_the_exact_cap(approved: str | None) -> None:
    with pytest.raises(live.ApprovalError):
        live.check_approval("pilot", approved)


@pytest.mark.parametrize("approved", ["92.07", "92.070", " 92.07"])
def test_approval_accepts_the_exact_cap(approved: str) -> None:
    assert live.check_approval("pilot", approved) == 92.07


def test_approval_refuses_the_cap_of_another_phase() -> None:
    with pytest.raises(live.ApprovalError, match="replication"):
        live.check_approval("replication", str(live.phase_cap("pilot")))


def test_approval_refuses_while_any_arm_is_unverified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(arms.ARMS, "rtk", arms.Arm("rtk", "0.50.0", False, ()))
    with pytest.raises(live.ApprovalError, match="rtk"):
        live.check_approval("pilot", "92.07")


def test_live_cli_refuses_before_touching_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("must not run anything before approval")

    monkeypatch.setattr(subprocess, "call", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(live, "prepare_tools", boom)
    assert main(["live", "--phase", "pilot"]) == 2
    assert "--i-approve-spend 92.07" in capsys.readouterr().err
    assert main(["live", "--phase", "pilot", "--i-approve-spend", "100"]) == 2
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["live", "--phase", "pilot", "--i-approve-spend", "92.07"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


# --------------------------------------------------------------------------- pinned tools


def test_prepare_tools_verifies_hashes_and_copies_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = hashlib.sha256(b"abc").hexdigest()
    monkeypatch.setattr(arms, "ARTIFACTS", {"t.tgz": ("https://example.invalid/t", good)})
    got = live.prepare_tools(tmp_path, fetch=lambda url, dest: dest.write_bytes(b"abc"))
    assert got == {"t.tgz": good}
    assert (
        tmp_path / "host" / "locks" / "distil-llm-1.54.0.cp312-x86_64-manylinux_2_28.txt"
    ).exists()
    assert (tmp_path / "uv-cache").is_dir()
    (tmp_path / "host" / "t.tgz").write_bytes(b"tampered")
    with pytest.raises(live.PreflightError, match="sha256"):
        live.prepare_tools(tmp_path, fetch=lambda url, dest: dest.write_bytes(b"x"))
    assert not (tmp_path / "host" / "t.tgz").exists()  # a bad artifact never survives to be mounted


def test_locks_are_hash_pinned_to_the_verified_wheels() -> None:
    d = (arms.LOCKS / "distil-llm-1.54.0.cp312-x86_64-manylinux_2_28.txt").read_text()
    assert (
        "distil-llm==1.54.0" in d
        and "c124150fcfe7bf450c28045422cf976bac1a9d395494f64a544c3c6920142cb4" in d
    )
    h = (arms.LOCKS / "headroom-ai-0.38.0-all.cp313-x86_64-manylinux_2_28.txt").read_text()
    assert (
        "headroom-ai==0.38.0" in h
        and "941d1f0c0aa0754bd4959bba0b212d1554a56c179c0643dd7852a4b9d95c7db7" in h
    )
    pkgs = [ln for ln in h.splitlines() if "==" in ln and not ln.startswith("#")]
    assert len(pkgs) > 100 and all(
        ln.rstrip().endswith("\\") for ln in pkgs
    )  # every pin carries hashes


# --------------------------------------------------------------------------- arm scripts


@needs_bash
@pytest.mark.parametrize("arm", rn.ARMS)
def test_every_arm_script_is_valid_bash(arm: str) -> None:
    for script in (
        arms.install_script(arm),
        arms.agent_setup_script(arm),
        arms.launch_script(arm, "claude-sonnet-5", "/logs/agent"),
        arms.post_run_script(arm, "/logs/agent"),
        arms.canary_script(arm, "claude-sonnet-5", "/logs/agent", "abc"),
    ):
        assert subprocess.run(["bash", "-n", "-c", script], capture_output=True).returncode == 0, (
            script
        )


def test_arm_env_routes_every_arm_through_the_meter_last() -> None:
    for arm in rn.ARMS:
        env = arms.arm_env(arm, "http://meter:1")
        assert env["ENABLE_TOOL_SEARCH"] == "true"  # identical for all arms (Amendment 1 §B)
        assert "http://meter:1" in env.values()
    assert arms.arm_env("control", "http://meter:1")["ANTHROPIC_BASE_URL"] == "http://meter:1"
    hr = arms.arm_env("headroom", "http://meter:1")
    assert hr["ANTHROPIC_TARGET_API_URL"] == "http://meter:1" and "ANTHROPIC_BASE_URL" not in hr
    assert "ANTHROPIC_BASE_URL" not in arms.arm_env("distil", "http://meter:1")


def test_launchers_are_the_documented_integrations() -> None:
    assert arms.launcher("headroom", ["-p"]) == ["headroom", "wrap", "claude", "--", "-p"]
    assert arms.launcher("distil", ["-p"])[:6] == [
        "distil",
        "wrap",
        "--upstream",
        "$CT_UPSTREAM",
        "--",
        "claude",
    ]
    assert "rtk init -g --auto-patch" in arms.agent_setup_script(
        "rtk"
    )  # headless: no prompt, no hook
    assert '--upstream "$CT_UPSTREAM"' in arms.launch_script("distil", "m", "/l")
    s = arms.launch_script("control", "claude-sonnet-5", "/l")
    assert "--model claude-sonnet-5 --print" in s and "unset CT_INSTRUCTION" in s
    assert "CLAUDE_CONFIG_DIR" not in s  # RTK's hook lives in $HOME/.claude


def test_install_scripts_use_only_hash_locked_mounted_inputs() -> None:
    for arm in ("headroom", "distil"):
        s = arms.install_script(arm)
        assert (
            "--require-hashes --no-deps" in s
            and f"{arms.HOST_MOUNT}/locks/" in s
            and "uname -m" in s
        )
        assert "uv venv --managed-python" in s  # never the task image's own interpreter
    assert f"{arms.HOST_MOUNT}/rtk.tar.gz" in arms.install_script("rtk")
    assert "tar" not in arms.install_script("control")


# --------------------------------------------------------------------------- canary


def test_meter_flags_the_canary_without_logging_the_nonce(tmp_path: Path) -> None:
    log = tmp_path / "l.jsonl"
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 8,
        "metadata": {"user_id": "cost-truth-canary-n0nce"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    with (
        mock.MockUpstream() as up,
        m.UsageMeter(m.MeterConfig(up.url, log, m.SpendMeter(1), "r", canary_nonce="n0nce")) as mt,
    ):
        req = urllib.request.Request(
            mt.base_url + "/v1/messages",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    rec = m.read_log(log)[0]
    assert rec["canary"] is True and "n0nce" not in log.read_text()


def _logs(tmp_path: Path, base_url: str, **files: str) -> Path:
    (tmp_path / "canary.json").write_text(json.dumps({"base_url": base_url}))
    for k, v in files.items():
        (tmp_path / k.replace("_", ".")).write_text(v)
    return tmp_path


def test_verify_canary_rules(tmp_path: Path) -> None:
    seen = [{"canary": True, "status": 200}]
    meter = "http://h:1"
    assert live.verify_canary("control", seen, _logs(tmp_path, meter), meter) == []
    assert live.verify_canary("distil", seen, _logs(tmp_path, "http://127.0.0.1:9"), meter) == []
    assert "bypassed" in live.verify_canary("distil", seen, _logs(tmp_path, meter), meter)[0]
    assert (
        "never saw"
        in live.verify_canary("headroom", [], _logs(tmp_path, "http://127.0.0.1:9"), meter)[0]
    )
    assert (
        live.verify_canary(
            "rtk", seen, _logs(tmp_path, meter, rtk_hook="1", rtk_rewrite="rtk git status"), meter
        )
        == []
    )
    probs = live.verify_canary(
        "rtk", seen, _logs(tmp_path, meter, rtk_hook="0", rtk_rewrite="git status"), meter
    )
    assert len(probs) == 2


# --------------------------------------------------------------------------- harbor glue


def test_classify_harbor_results() -> None:
    started = {"started_at": "2026-09-25T10:00:00", "finished_at": "2026-09-25T10:05:00"}
    solved = live.classify_harbor(
        {"verifier_result": {"rewards": {"reward": 1}}, "agent_execution": started}
    )
    assert solved.status == "solved" and solved.agent_wall_s == 300.0
    assert (
        live.classify_harbor(
            {"verifier_result": {"rewards": {"reward": 0}}, "agent_execution": started}
        ).status
        == "failed"
    )
    to = {"exception_info": {"exception_type": "AgentTimeoutError"}, "agent_execution": started}
    assert live.classify_harbor(to).status == "timeout"
    assert (
        live.classify_harbor({"exception_info": {"exception_type": "RuntimeError"}}).status
        == "infra_error"
    )


def test_harbor_argv_passes_arm_as_kwargs_not_env(tmp_path: Path) -> None:
    ex = live.HarborExecutor(
        tmp_path / "tasks",
        tmp_path / "trials",
        tmp_path / "tools",
        "host.docker.internal",
        ["harbor"],
    )
    argv = ex.argv(
        rn.RunSpec("fix-git", 0, "rtk", "claude-sonnet-5", 0),
        "http://host.docker.internal:5",
        "run",
        "",
    )
    s = " ".join(argv)
    assert "--agent benchmarks.cost_truth.harbor_agent:CostTruthAgent" in s
    assert "--agent-kwarg ct_arm=rtk" in s and "--ae" not in argv and "--agent-env" not in argv
    assert (
        "--model anthropic/claude-sonnet-5" in s and "--allow-agent-host host.docker.internal" in s
    )
    mounts = json.loads(argv[argv.index("--mounts") + 1])
    assert mounts[0]["read_only"] is True and mounts[0]["target"] == arms.HOST_MOUNT
    assert mounts[0]["source"].endswith("/tools/host") and mounts[1]["source"].endswith(
        "/tools/uv-cache"
    )
    assert argv[argv.index("--agent-setup-timeout") + 1] == str(live.AGENT_SETUP_TIMEOUT_S)


def test_harbor_agent_runs_the_arm_scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("harbor")
    import asyncio

    from benchmarks.cost_truth.harbor_agent import CostTruthAgent

    class Res:
        def __init__(self, stdout: str = "") -> None:
            self.return_code, self.stdout, self.stderr = 0, stdout, ""

    class FakeEnv:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def exec(self, command: str, **kw: Any) -> Res:
            self.calls.append({"command": command, **kw})
            return Res(f"{arms.CLAUDE_CODE_VERSION} (Claude Code)")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    agent = CostTruthAgent(
        logs_dir=tmp_path,
        model_name="anthropic/claude-sonnet-5",
        ct_arm="headroom",
        ct_meter="http://h:9",
    )
    env = FakeEnv()
    asyncio.run(agent.install(env))  # type: ignore[arg-type]
    asyncio.run(agent.run("fix the build", env, None))  # type: ignore[arg-type]
    cmds = [c["command"] for c in env.calls]
    assert any(arms.install_script("headroom").strip() in c for c in cmds)
    launch = next(c for c in env.calls if "headroom wrap claude" in c["command"])
    assert (
        launch["env"]["ANTHROPIC_TARGET_API_URL"] == "http://h:9"
        and launch["env"]["CT_INSTRUCTION"] == "fix the build"
    )
    assert "CLAUDE_CONFIG_DIR" not in launch["env"] and "headroom" not in json.dumps(
        launch["env"]
    ).replace("wrap", "")
    assert any(
        "headroom savings --json" in c for c in cmds
    )  # claim collected even on the happy path
    with pytest.raises(ValueError, match="ct_meter"):
        CostTruthAgent(logs_dir=tmp_path, model_name="anthropic/claude-sonnet-5", ct_arm="rtk")


# --------------------------------------------------------------------------- end to end on the mock


def _cfg(tmp_path: Path, up: mock.MockUpstream, tasks: list[str]) -> live.LiveConfig:
    return live.LiveConfig(
        "pilot",
        5.0,
        tmp_path / "out",
        upstream=up.url,
        url_host="127.0.0.1",
        concurrency=3,
        min_gap_s=0.0,
        tasks=tasks,
        seeds=[0],
    )


@needs_bash
@pytest.mark.skipif(not DISTIL_BIN.exists(), reason="distil console script not installed")
def test_live_pipeline_end_to_end_against_the_mock(tmp_path: Path) -> None:
    with mock.MockUpstream() as up:
        runs = live.run_phase(
            _cfg(tmp_path, up, ["t0", "t1"]), live.LocalExecutor(tmp_path / "work", DISTIL_BIN)
        )
    out = tmp_path / "out"
    pre = json.loads((out / "preflight.json").read_text())
    assert all(pre[a]["ok"] for a in rn.ARMS), pre  # incl. the REAL distil wrap traversal
    assert len(runs) == 2 * len(rn.ARMS)
    assert all(
        r["status"] in ("solved", "failed") and r["cost_usd"] > 0 and r["requests"] >= 4
        for r in runs
    )
    claims = {r["arm"]: r["claim"] for r in runs}
    assert claims["control"] is None
    assert "total_tokens_saved" in claims["distil"]  # the real `distil stats --json`
    assert claims["rtk"]["summary"] == {"total_saved": 0} and claims["headroom"]["stand_in"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["spent_usd"] == pytest.approx(
        sum(r["cost_usd"] for r in runs) + sum(pre[a]["cost_usd"] for a in rn.ARMS), abs=1e-5
    )
    s = live.pilot_summary(runs)
    assert s["median_turns"] > 0 and "sigma_within_log_cost" in s


@needs_bash
def test_canary_catches_a_tool_that_is_bypassed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_launcher, real_env = arms.launcher, arms.arm_env

    def bypass_launcher(arm: str, args: list[str]) -> list[str]:
        return ["claude", *args] if arm == "headroom" else real_launcher(arm, args)

    def bypass_env(arm: str, meter_url: str, root: str = arms.CT) -> dict[str, str]:
        env = real_env(arm, meter_url, root)
        if arm == "headroom":
            env["ANTHROPIC_BASE_URL"] = (
                meter_url  # misconfigured: claude talks to the meter directly
            )
        return env

    monkeypatch.setattr(arms, "launcher", bypass_launcher)
    monkeypatch.setattr(arms, "arm_env", bypass_env)
    monkeypatch.setattr(rn, "ARMS", ("control", "headroom"))
    with mock.MockUpstream() as up, pytest.raises(live.PreflightError, match="bypassed"):
        live.run_phase(
            _cfg(tmp_path, up, ["t0"]), live.LocalExecutor(tmp_path / "work", DISTIL_BIN)
        )


@needs_bash
def test_canary_catches_an_unreachable_meter(tmp_path: Path) -> None:
    with mock.MockUpstream() as up:
        cfg = _cfg(tmp_path, up, ["t0"])
        cfg.url_host = (
            "cost-truth.invalid"  # wrong host for the containers: nothing reaches the meter
        )
        with pytest.raises(live.PreflightError, match="never saw"):
            live.run_phase(cfg, live.LocalExecutor(tmp_path / "work", DISTIL_BIN))
    assert up.requests == 0


# --------------------------------------------------------------------------- loud installs (2026-09-25)


@needs_bash
def test_install_step_failure_is_loud_and_reported(tmp_path: Path) -> None:
    script = "\n".join(
        [
            *arms._stepper(str(tmp_path), "distil"),
            "step ok true",
            "step pip sh -c 'echo resolving; exit 7'",
            "echo unreachable",
        ]
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert proc.returncode == 7 and "unreachable" not in proc.stdout
    assert "install step pip FAILED (exit 7)" in proc.stderr and "resolving" in proc.stderr
    inst = live.read_install(tmp_path)
    assert (
        inst is not None
        and inst["step"] == "pip"
        and inst["exit"] == 7
        and inst["tail"] == ["resolving"]
    )
    probs = live.verify_canary("distil", [], tmp_path, "http://m:1")
    assert probs[0] == "install step 'pip' failed with exit 7"


def test_install_script_never_chmods_the_read_only_mount() -> None:
    for arm in ("headroom", "distil"):
        chmod = [
            ln
            for ln in arms.install_script(arm).splitlines()
            if "chmod" in ln and not ln.startswith(("ct_", "step()"))
        ]
        assert chmod and all(
            arms.HOST_MOUNT not in ln and f"-R a+rX {arms.CT}\n" not in ln + "\n" for ln in chmod
        )


def test_harbor_executor_pins_amd64_for_every_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, str]] = []

    def fake_run(argv: list[str], env: dict[str, str], **kw: Any) -> Any:
        seen.append(env)
        return subprocess.CompletedProcess(argv, 1, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ex = live.HarborExecutor(tmp_path, tmp_path / "trials", tmp_path / "tools", "h", ["harbor"])
    for arm in rn.ARMS:
        out = ex.run(
            rn.RunSpec("t", 0, arm, "claude-sonnet-5", 0), "http://h:1", tmp_path / arm, "run", ""
        )
        assert out.status == "infra_error"
    assert all(e["DOCKER_DEFAULT_PLATFORM"] == "linux/amd64" for e in seen) and len(seen) == len(
        rn.ARMS
    )


def test_classify_keeps_only_the_last_line_of_harbor_messages() -> None:
    blob = "stdout: " + "x" * 10_000 + "\nchmod: Read-only file system\nexit status 1"
    out = live.classify_harbor(
        {"exception_info": {"exception_type": "ApiRateLimitError", "exception_message": blob}}
    )
    assert out.status == "infra_error" and out.detail == "ApiRateLimitError: exit status 1"


@needs_bash
@pytest.mark.skipif(not DISTIL_BIN.exists(), reason="distil console script not installed")
def test_preflight_only_runs_the_canaries_and_nothing_else(tmp_path: Path) -> None:
    with mock.MockUpstream() as up:
        runs = live.run_phase(
            _cfg(tmp_path, up, ["t0", "t1"]),
            live.LocalExecutor(tmp_path / "work", DISTIL_BIN),
            preflight_only=True,
        )
        assert runs == [] and up.requests == len(rn.ARMS)  # exactly one tiny request per arm
    pre = json.loads((tmp_path / "out" / "preflight.json").read_text())
    assert all(pre[a]["ok"] and pre[a]["trial_status"] for a in rn.ARMS)


def test_already_spent_must_fit_under_the_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    assert (
        main(["live", "--phase", "pilot", "--i-approve-spend", "92.07", "--already-spent", "95"])
        == 2
    )
    assert "--already-spent" in capsys.readouterr().err
