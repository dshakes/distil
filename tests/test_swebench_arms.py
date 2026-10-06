"""Offline tests for the pluggable arms of benchmarks/swebench_outcome: plain, distil, rtk,
selective, provider-cm. No API, network, docker, GPT-2, spaCy or RTK binary needed: each arm's
real dependency is replaced by a scripted fake at its documented boundary."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import socket
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

anthropic = pytest.importorskip("anthropic", reason="anthropic SDK not installed")

from benchmarks.swebench_outcome import agent as ag  # noqa: E402
from benchmarks.swebench_outcome import arm_rtk, arm_selective, arms, cli, grade, report, run  # noqa: E402
from benchmarks.swebench_outcome.env import EnvError, FakeEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TASK = {"instance_id": "a__b-1", "problem_statement": "fix foo"}
CFG = ag.Cfg()


def tu(i, name, **inp):
    return NS(
        type="tool_use",
        id=i,
        name=name,
        input=inp,
        model_dump=lambda **_: {"type": "tool_use", "id": i, "name": name, "input": inp},
    )


def txt(t):
    return NS(type="text", text=t, model_dump=lambda **_: {"type": "text", "text": t})


def resp(blocks, stop="tool_use", tin=1000, tout=100, cm=None):
    r = NS(
        content=blocks,
        stop_reason=stop,
        usage=NS(
            input_tokens=tin,
            output_tokens=tout,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    if cm is not None:
        r.context_management = NS(applied_edits=cm)
    return r


class FakeClient:
    """Records every call, separately for the stable and the beta endpoint."""

    def __init__(self, script):
        self.script, self.calls, self.beta_calls = list(script), [], []
        self.messages = NS(create=self._create)
        self.beta = NS(messages=NS(create=self._create_beta))

    def _next(self):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _create(self, **kw):
        self.calls.append(kw)
        return self._next()

    def _create_beta(self, **kw):
        self.beta_calls.append(kw)
        return self._next()


EDIT = [
    resp([tu("t1", "bash", command="cat f.py")]),
    resp(
        [
            tu(
                "t2",
                "str_replace_based_edit_tool",
                command="str_replace",
                path="f.py",
                old_str="x = 1",
                new_str="x = 2",
            )
        ]
    ),
    resp([txt("done")], stop="end_turn"),
]


def go(script, arm, env=None, cfg=None, budget=None, **kw):
    env = env or FakeEnv({"f.py": "x = 1\n"}, exec_fn=lambda c: "OUTPUT")
    c = FakeClient(script)
    r = ag.run_agent(c, env, TASK, arm, cfg or CFG, budget or ag.Budget(None), **kw)
    return r, c, env


# ---------------------------------------------------------------------------- plain / distil


def test_plain_request_is_the_baseline_and_untouched_by_any_hook():
    r, c, env = go(EDIT, "plain")
    assert not c.beta_calls and len(c.calls) == 3
    assert set(c.calls[0]) == {
        "model",
        "max_tokens",
        "system",
        "tools",
        "messages",
        "thinking",
        "output_config",
        "tool_choice",
    }
    assert [t["name"] for t in c.calls[0]["tools"]] == ["bash", "str_replace_based_edit_tool"]
    assert env.cmds == ["cat f.py"] and not env.raw_cmds and not env.installed
    assert r["arm"] == "plain" and r["arm_meta"] == {} and "arm_stats" not in r


def test_distil_arm_records_library_and_version_and_accepts_a_name_or_an_arm():
    a = arms.get_arm("distil")
    assert a.meta["library"] == "distil" and "version" in a.meta
    assert [t["name"] for t in ag.build_tools("distil")] == [
        "bash",
        "str_replace_based_edit_tool",
        "distil_expand",
    ]
    r, *_ = go([resp([txt("x")], stop="end_turn")], a)
    assert r["arm_meta"]["library"] == "distil"
    with pytest.raises(KeyError, match="needs options"):
        arms.get_arm("rtk")


def test_parse_arms():
    assert arms.parse_arms("plain, distil,rtk") == ("plain", "distil", "rtk")
    for bad in ("", "plain,plain", "plain,nope"):
        with pytest.raises(SystemExit, match="--arms"):
            arms.parse_arms(bad)


# ---------------------------------------------------------------------------- provider-cm


def test_provider_cm_uses_the_documented_beta_call_and_leaves_plain_alone():
    edits = [NS(type=arms.CM_EDIT, cleared_tool_uses=2, cleared_input_tokens=1500)]
    script = [EDIT[0], resp([tu("t2", "bash", command="ls")], cm=edits), EDIT[2]]
    r, c, _ = go(script, arms.provider_cm_arm(trigger=2500, keep=1, clear_at_least=500))
    assert not c.calls and len(c.beta_calls) == 3  # never the stable endpoint
    kw = c.beta_calls[0]
    assert kw["betas"] == ["context-management-2025-06-27"]
    assert kw["context_management"] == {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 2500},
                "keep": {"type": "tool_uses", "value": 1},
                "clear_at_least": {"type": "input_tokens", "value": 500},
            }
        ]
    }
    assert kw["thinking"] == {"type": "adaptive"} and kw["tool_choice"] == {"type": "auto"}
    assert r["arm_stats"] == {
        "applied_calls": 1,
        "cleared_tool_uses": 2,
        "cleared_input_tokens": 1500,
    }
    m = r["arm_meta"]
    assert (m["library"], m["beta"], m["edit"]) == ("anthropic", arms.CM_BETA, arms.CM_EDIT)
    assert (m["trigger_input_tokens"], m["keep_tool_uses"]) == (2500, 1)
    # the same script through plain sends no context_management and no betas
    _, c2, _ = go(EDIT, "plain")
    assert "context_management" not in c2.calls[0] and "betas" not in c2.calls[0]


def test_provider_cm_validates_its_config():
    with pytest.raises(SystemExit, match=">= 1"):
        arms.provider_cm_arm(trigger=0)


def test_sdk_accepts_the_beta_shape():
    """The SDK's own signature has the params we pass (guards against a rename upstream)."""
    import inspect

    sig = inspect.signature(anthropic.Anthropic(api_key="x").beta.messages.create)
    assert {"betas", "context_management"} <= set(sig.parameters)


# ---------------------------------------------------------------------------- rtk

REWRITES = {
    "git status": (0, "rtk git status\n"),
    "cat f.py": (3, "rtk read f.py\n"),
    "echo hi": (1, ""),
    "rm -rf x": (2, ""),
}


def rtk_raw(argv):
    if argv[0] == arm_rtk.CONTAINER_PATH:
        return 0, f"rtk {arm_rtk.RTK_VERSION}\n", ""
    assert argv[:5] == ["env", "-u", "RTK_REWRITE_HOST", "rtk", "rewrite"]
    rc, out = REWRITES.get(argv[5], (127, ""))
    return rc, out, "boom" if rc == 127 else ""


def test_rtk_rewrite_follows_the_hook_exit_protocol():
    env = FakeEnv(raw_fn=rtk_raw)
    assert arm_rtk.rewrite(env, "git status") == "rtk git status"  # exit 0: allow rule
    assert arm_rtk.rewrite(env, "cat f.py") == "rtk read f.py"  # exit 3: ask / no rule
    assert arm_rtk.rewrite(env, "echo hi") == "echo hi"  # exit 1: no RTK equivalent
    assert arm_rtk.rewrite(env, "rm -rf x") == "rm -rf x"  # exit 2: deny rule, unchanged
    with pytest.raises(EnvError, match=r"exit 127.*boom"):  # never a silent pass-through
        arm_rtk.rewrite(env, "unknown")


def test_rtk_arm_rewrites_bash_installs_the_pinned_binary_and_leaves_the_editor_alone():
    env = FakeEnv({"f.py": "x = 1\n"}, exec_fn=lambda c: f"ran:{c}", raw_fn=rtk_raw)
    binary = Path("/fake/rtk")
    r, c, env = go(EDIT, arms.resolve_arm(arm_rtk.rtk_arm(binary)), env=env)
    assert env.installed == [(binary, "/usr/local/bin/rtk")]
    assert env.cmds == ["rtk read f.py"]  # what the container actually ran
    # ...and the model saw that command's output, not the original's
    seen = c.calls[1]["messages"][-1]["content"][0]["content"]
    assert seen == "ran:rtk read f.py"
    assert "-x = 1" in r["patch"] and "+x = 2" in r["patch"]  # editor path unaffected
    assert r["arm_stats"] == {"commands": 1, "rewritten": 1}
    assert r["arm_meta"]["library"] == "rtk" and r["arm_meta"]["version"] == "0.51.0"
    assert r["arm_meta"]["license"] == "Apache-2.0" and r["failure_class"] is None
    assert len(c.calls) == 3 and not c.beta_calls
    assert c.calls[0]["tools"] == ag.build_tools("plain")  # same tools, same system prompt


def test_rtk_wrong_binary_version_is_an_env_error_not_a_silent_plain_run():
    env = FakeEnv(raw_fn=lambda argv: (0, "rtk 0.0.1\n", ""))
    r, c, _ = go(EDIT, arm_rtk.rtk_arm(Path("/fake/rtk")), env=env)
    assert r["failure_class"] == "env_error" and "pinned rtk 0.51.0" in r["error"]
    assert not c.calls  # failed before any (paid) call


def _tarball(payload=b"\x7fELF fake rtk", name="rtk"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_rtk_verifies_the_pinned_sha256_and_extracts(tmp_path):
    data = _tarball()
    urls = []

    def opener(url, timeout):
        urls.append(url)
        return _Resp(data)

    p = arm_rtk.fetch_rtk(tmp_path, opener, hashlib.sha256(data).hexdigest())
    assert p.read_bytes() == b"\x7fELF fake rtk" and p.stat().st_mode & 0o111
    assert (
        urls == [arm_rtk.RTK_URL]
        and "v0.51.0" in urls[0]
        and "x86_64-unknown-linux-musl" in urls[0]
    )
    arm_rtk.fetch_rtk(tmp_path, opener, hashlib.sha256(data).hexdigest())
    assert len(urls) == 1  # cached tarball is re-verified, not re-downloaded


def test_fetch_rtk_rejects_a_tampered_download_and_other_failures(tmp_path):
    with pytest.raises(arms.ArmUnavailable, match="pinned"):
        arm_rtk.fetch_rtk(tmp_path, lambda u, timeout: _Resp(_tarball()))  # real pin != fake data
    assert not (tmp_path / arm_rtk.RTK_ASSET).exists()  # deleted, not trusted next time

    def offline(url, timeout):
        raise OSError("no route")

    with pytest.raises(arms.ArmUnavailable, match="--rtk-bin"):
        arm_rtk.fetch_rtk(tmp_path / "b", offline)
    data = _tarball(name="not-rtk")
    with pytest.raises(arms.ArmUnavailable, match="no `rtk` binary"):
        arm_rtk.fetch_rtk(
            tmp_path / "c", lambda u, timeout: _Resp(data), hashlib.sha256(data).hexdigest()
        )


def test_docker_env_install_and_exec_raw_build_the_expected_docker_calls(tmp_path, monkeypatch):
    from benchmarks.swebench_outcome import env as envmod

    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(
            argv, 0, stdout="ok\n", stderr=b"" if isinstance(kw.get("input"), bytes) else ""
        )

    monkeypatch.setattr(envmod.subprocess, "run", fake_run)
    e = envmod.DockerEnv.__new__(envmod.DockerEnv)
    e.name, e.workdir = "swo-x", "/testbed"
    src = tmp_path / "rtk"
    src.write_bytes(b"\x7fELF")
    e.install_file(src, "/usr/local/bin/rtk")
    argv, kw = calls[0]
    assert argv[:5] == ["docker", "exec", "-i", "swo-x", "sh"] and argv[-2:] == [
        "_",
        "/usr/local/bin/rtk",
    ]
    assert "chmod 755" in argv[6] and kw["input"] == b"\x7fELF"  # streamed, owned by container root
    assert e.exec_raw(["rtk", "--version"]) == (0, "ok\n", "")
    assert calls[1][0] == ["docker", "exec", "-w", "/testbed", "swo-x", "rtk", "--version"]

    monkeypatch.setattr(
        envmod.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"read-only fs"),
    )
    with pytest.raises(EnvError, match="install /usr/local/bin/rtk: read-only fs"):
        e.install_file(src, "/usr/local/bin/rtk")

    def hang(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(envmod.subprocess, "run", hang)
    with pytest.raises(EnvError, match="install"):
        e.install_file(src, "/usr/local/bin/rtk")
    with pytest.raises(EnvError, match="install"):
        e.install_file(tmp_path / "missing", "/usr/local/bin/rtk")


def test_rtk_pin_matches_the_release_checksum_file_format():
    assert len(arm_rtk.RTK_SHA256) == 64 and int(arm_rtk.RTK_SHA256, 16)
    assert arm_rtk.RTK_URL.endswith(f"v{arm_rtk.RTK_VERSION}/{arm_rtk.RTK_ASSET}")


# ---------------------------------------------------------------------------- selective

BIG = "The quick brown fox jumps over the lazy dog. " * 20  # 900 chars


def fake_reduce(calls):
    def reduce(text):
        calls.append(text)
        return text[: len(text) // 2]

    return reduce


def msgs(content):
    return [
        {"role": "user", "content": "issue"},
        {"role": "assistant", "content": [{"type": "text", "text": BIG}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "a", "content": content}],
        },
    ]


def test_selective_prunes_only_long_tool_results_memoises_and_never_mutates_input():
    calls = []
    arm = arm_selective.selective_arm(fake_reduce(calls))
    original = msgs(BIG)
    snapshot = copy.deepcopy(original)
    out = arm.transform(original)
    assert original == snapshot  # input untouched
    assert out[0] == original[0] and out[1] == original[1]  # prompt + assistant text: not touched
    pruned = out[2]["content"][0]["content"]
    assert 0 < len(pruned) < len(BIG)
    n = len(calls)
    assert n == len(arm_selective.chunks(BIG)) > 1
    assert arm.transform(msgs(BIG)) == out and len(calls) == n  # memoised: byte-stable prefix
    short = arm.transform(msgs("ok"))
    assert short[2]["content"][0]["content"] == "ok" and len(calls) == n  # below MIN_CHARS


def test_selective_handles_block_content_empty_and_not_smaller_results():
    arm = arm_selective.selective_arm(lambda t: "")
    blocks = [{"type": "text", "text": BIG}, {"type": "image", "source": {}}]
    out = arm.transform(msgs(blocks))[2]["content"][0]["content"]
    assert out[0] == {"type": "text", "text": arm_selective.EMPTY} and out[1] == blocks[1]
    keep = arm_selective.selective_arm(lambda t: t + "!")  # never grows a result
    assert keep.transform(msgs(BIG))[2]["content"][0]["content"] == BIG
    plain = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a"}]}]
    assert arm.transform(plain) == plain  # tool_result without content


def test_selective_chunks_cut_on_lines_and_split_huge_lines():
    text = "\n".join(["a" * 300] * 5) + "\n" + "b" * 2000
    cs = arm_selective.chunks(text, 800)
    assert "".join(cs) == text and all(len(c) <= 800 for c in cs)
    assert arm_selective.chunks("") == []


def test_selective_arm_runs_in_the_agent_loop_and_records_metadata():
    calls = []
    big_out = BIG + "\n" + BIG
    env = FakeEnv({"f.py": "x = 1\n"}, exec_fn=lambda c: big_out)
    r, c, _ = go(EDIT, arm_selective.selective_arm(fake_reduce(calls)), env=env)
    sent = c.calls[1]["messages"][-1]["content"][0]["content"]
    assert len(sent) < len(big_out) and calls
    m = r["arm_meta"]
    assert (m["library"], m["version"], m["lm"]) == ("selective-context", "0.1.4", "gpt2")
    assert (m["reduce_ratio"], m["reduce_level"]) == (0.35, "phrase")
    _, c2, _ = go(EDIT, "plain", env=FakeEnv({"f.py": "x = 1\n"}, exec_fn=lambda c: big_out))
    assert c2.calls[1]["messages"][-1]["content"][0]["content"] == big_out  # plain: verbatim


class FakeProc:
    def __init__(self, lines):
        self.stdin, self.stdout = io.StringIO(), io.StringIO("".join(ln + "\n" for ln in lines))
        self.killed = False

    def poll(self):
        return 1

    def kill(self):
        self.killed = True

    def wait(self):
        return 0


HELLO = json.dumps({"ready": True, "versions": {"selective-context": "0.1.4", "spacy": "3.2.0"}})


def test_selective_worker_client_protocol_and_failures():
    p = FakeProc([HELLO, json.dumps({"context": "short"}), json.dumps({"error": "KeyError: x"})])
    w = arm_selective.SelectiveWorker("py", popen=lambda *a, **k: p)
    assert w.versions["spacy"] == "3.2.0"
    assert w.reduce("some text") == "short"
    assert json.loads(p.stdin.getvalue().splitlines()[0]) == {
        "text": "some text",
        "ratio": 0.35,
        "level": "phrase",
    }
    with pytest.raises(RuntimeError, match="failed on a 3-char chunk: KeyError"):
        w.reduce("abc")
    w.close()
    assert p.killed

    bad = FakeProc([json.dumps({"error": "ModuleNotFoundError: selective_context"})])
    with pytest.raises(arms.ArmUnavailable, match="ModuleNotFoundError.*pip install"):
        arm_selective.SelectiveWorker("py", popen=lambda *a, **k: bad)
    assert bad.killed
    old = FakeProc([json.dumps({"ready": True, "versions": {"selective-context": "0.1.3"}})])
    with pytest.raises(arms.ArmUnavailable, match="pinned selective-context==0.1.4, found 0.1.3"):
        arm_selective.SelectiveWorker("py", popen=lambda *a, **k: old)
    with pytest.raises(arms.ArmUnavailable, match="worker exited"):
        arm_selective.SelectiveWorker("py", popen=lambda *a, **k: FakeProc([]))

    def nope(*a, **k):
        raise FileNotFoundError("no such python")

    with pytest.raises(arms.ArmUnavailable, match="cannot start 'py'"):
        arm_selective.SelectiveWorker("py", popen=nope)


def test_the_real_worker_script_speaks_the_protocol_against_a_stub_library(tmp_path):
    """Runs selective_worker.py for real; only `selective_context` itself is a stub."""
    stub = tmp_path / "selective_context.py"
    stub.write_text(
        "print('Loading dependencies...')\n"
        "class SelectiveContext:\n"
        "    def __init__(self, model_type, lang):\n"
        "        assert (model_type, lang) == ('gpt2', 'en')\n"
        "    def __call__(self, text, reduce_ratio, reduce_level):\n"
        "        if text == 'boom':\n"
        "            raise ValueError('too long')\n"
        "        return text.upper(), []\n"
    )
    env = {"PYTHONPATH": str(tmp_path), "PATH": ""}
    p = subprocess.Popen(
        [sys.executable, str(arm_selective.WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        hello = json.loads(p.stdout.readline())
        assert hello["ready"] is True and set(hello["versions"]) >= {"selective-context", "torch"}
        for text, want in (
            ("abc", {"context": "ABC"}),
            ("boom", {"error": "ValueError: too long"}),
        ):
            p.stdin.write(json.dumps({"text": text, "ratio": 0.35, "level": "phrase"}) + "\n")
            p.stdin.flush()
            assert json.loads(p.stdout.readline()) == want  # stub's print() did not leak in
    finally:
        p.stdin.close()
        assert p.wait(timeout=30) == 0


def test_the_real_worker_reports_a_missing_library_and_the_client_turns_it_into_a_clear_error(
    tmp_path,
):
    with pytest.raises(
        arms.ArmUnavailable, match="selective:.*pip install selective-context==0.1.4"
    ):
        arm_selective.SelectiveWorker(sys.executable, popen=_popen_without_library(tmp_path))


def _popen_without_library(tmp_path):
    def popen(argv, **kw):
        # an empty PYTHONPATH + isolated mode guarantees `import selective_context` fails
        return subprocess.Popen([argv[0], "-I", *argv[1:]], **kw)

    return popen


# ---------------------------------------------------------------------------- plan / reuse / run


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _rows(arm, n, cost, effort="medium", model="claude-sonnet-5-5", steps=4):
    return [
        {
            "instance_id": f"i{k}",
            "arm": arm,
            "model": model,
            "effort": effort,
            "failure_class": None,
            "steps": steps,
            "cost_usd": cost,
            "patch": "p",
            "usage": {"input": 10, "output": 2},
        }
        for k in range(n)
    ]


def test_estimate_arms_measured_vs_proxy_and_reuse(tmp_path):
    _write(tmp_path / "results.jsonl", _rows("plain", 4, 0.02) + _rows("distil", 4, 0.03))
    cal = run.calibrate(tmp_path / "results.jsonl", CFG)
    assert cal["mean"] == pytest.approx({"plain": 0.02, "distil": 0.03}) and not cal["mismatch"]
    est = run.estimate_arms(10, ("plain", "distil", "rtk"), cal, {"plain": 4})
    assert est["arms"]["plain"]["total_usd"] == pytest.approx(0.02 * 6)  # 4 reused rows are free
    assert est["arms"]["distil"]["total_usd"] == pytest.approx(0.30)
    assert est["arms"]["rtk"]["per_task_usd"] == pytest.approx(0.02 * run.UNMEASURED_FACTOR)
    assert "proxy" in est["arms"]["rtk"]["basis"] and "measured" in est["arms"]["plain"]["basis"]
    assert est["total_usd"] == pytest.approx(0.12 + 0.30 + 0.30)
    assert run.calibrate(tmp_path / "results.jsonl", ag.Cfg(effort="low"))["mismatch"]
    _write(tmp_path / "only_distil.jsonl", _rows("distil", 1, 0.1))
    with pytest.raises(SystemExit, match="no plain arm"):
        run.estimate_arms(1, ("rtk",), run.calibrate(tmp_path / "only_distil.jsonl", CFG))
    _write(tmp_path / "empty.jsonl", [])
    with pytest.raises(SystemExit, match="no calibration results"):
        run.calibrate(tmp_path / "empty.jsonl", CFG)


def test_plan_arms_is_calibrated_from_the_committed_300_task_run_and_touches_no_network(
    tmp_path, monkeypatch, capsys
):
    def boom(*a, **k):
        raise AssertionError("network/API/arm dependency touched by plan")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(run, "load_records", boom)
    monkeypatch.setattr(arm_rtk, "fetch_rtk", boom)
    monkeypatch.setattr(arm_selective, "make_selective", boom)
    cal = ROOT / "benchmarks" / "results" / "swebench-outcome-300-medium"
    plain = [r for r in run.read_results(cal / "results.jsonl") if r["arm"] == "plain"]
    assert len(plain) == 300
    idf = tmp_path / "ids.txt"
    idf.write_text("\n".join(r["instance_id"] for r in plain))
    every = "plain,distil,rtk,selective,provider-cm"
    reuse = [f"--reuse-arm=plain={cal}", f"--reuse-arm=distil={cal}"]
    rc = cli.main(
        ["plan", "--instances", str(idf), "--arms", every, "--calibrate", str(cal), *reuse]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "PLACEHOLDER" not in out
    assert "plain: $0.0284/task x 0 tasks = $0.00 (measured, n=" in out
    assert "rtk: $0.0426/task x 300 tasks" in out and "proxy: plain mean x1.5" in out
    assert "ESTIMATE $" in out and "reused arms cost nothing" in out


def test_plan_warns_when_the_calibration_ran_a_different_effort(tmp_path, capsys):
    _write(tmp_path / "results.jsonl", _rows("plain", 2, 0.02, effort="low"))
    (tmp_path / "ids.txt").write_text("i0\ni1\n")
    cli.main(
        ["plan", "--instances", str(tmp_path / "ids.txt"), "--arms", "plain,rtk"]
        + ["--calibrate", str(tmp_path)]
    )
    assert "WARNING: calibration ran claude-sonnet-5-5@low, not claude-sonnet-5-5@medium" in (
        capsys.readouterr().out
    )


def test_reuse_arm_copies_rows_and_transcripts_tags_them_and_keeps_them_out_of_the_budget(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _write(src / "results.jsonl", _rows("plain", 3, 1.0) + _rows("distil", 3, 5.0))
    (src / "transcripts" / "plain").mkdir(parents=True)
    (src / "transcripts" / "plain" / "i0.json").write_text("[]")
    assert run.reuse_arm(out, "plain", src, ["i0", "i1", "zzz"], CFG) == 2
    assert run.reuse_arm(out, "plain", src, ["i0", "i1"], CFG) == 0  # idempotent
    rows = run.read_results(out / "results.jsonl")
    assert [r["instance_id"] for r in rows] == ["i0", "i1"]
    assert all(r["reused_from"] == str(src) and r["arm"] == "plain" for r in rows)
    assert (out / "transcripts" / "plain" / "i0.json").exists()
    assert run.read_manifest(out) == {"plain": {"from": str(src), "rows": 2}}
    _write(out / "results.jsonl", [*rows, {**_rows("rtk", 1, 0.5)[0], "instance_id": "i9"}])
    assert run.spent_in(out / "results.jsonl") == pytest.approx(0.5)  # reused $2.00 not counted
    with pytest.raises(SystemExit, match="refusing to mix"):
        run.reuse_arm(tmp_path / "o2", "plain", src, ["i0"], ag.Cfg(effort="low"))
    with pytest.raises(SystemExit, match="no 'rtk' rows"):
        run.reuse_arm(tmp_path / "o3", "rtk", src, ["i0"], CFG)
    for bad in ("plain", "nope=x", "plain="):
        with pytest.raises(SystemExit, match="ARM=DIR"):
            cli._reuse_pairs([bad])


def _built_arms(tmp_path):
    """Every arm, with its real dependency replaced by a fake at the documented boundary."""
    calls = []
    return {
        "plain": arms.plain_arm(),
        "distil": arms.distil_arm(lambda m: (list(m), NS(expand=lambda h: "ORIGINAL"))),
        "rtk": arm_rtk.rtk_arm(Path("/fake/rtk")),
        "selective": arm_selective.selective_arm(fake_reduce(calls)),
        "provider-cm": arms.provider_cm_arm(),
    }, calls


def test_smoke_every_arm_end_to_end_with_fake_env_and_client(tmp_path):
    """`run` offline: all five arms through run_all, rotating order, one results row each."""
    built, sel_calls = _built_arms(tmp_path)
    big = BIG + "\n" + BIG

    def factory(_):
        return FakeEnv(
            {"f.py": "x = 1\n"},
            exec_fn=lambda c: big if "f.py" in c else "OUTPUT",
            raw_fn=rtk_raw,
        )

    class Client(FakeClient):
        def __init__(self):
            super().__init__([])

        def _next(self):
            n = len(self.calls) + len(self.beta_calls)
            return EDIT[(n - 1) % 3]

    c = Client()
    names = tuple(built)
    s = run.run_all(
        [{"instance_id": f"i{k}", "problem_statement": "p"} for k in range(5)],
        tmp_path,
        c,
        factory,
        CFG,
        ag.Budget(None),
        tuple(built.values()),
    )
    assert s["ran"] == 25
    rows = run.read_results(tmp_path / "results.jsonl")
    assert {r["arm"] for r in rows} == set(names) and all(r["failure_class"] is None for r in rows)
    assert all("-x = 1" in r["patch"] for r in rows)
    first = [next(r["arm"] for r in rows if r["instance_id"] == f"i{k}") for k in range(5)]
    assert first == list(names)  # rotation: every arm leads exactly once in five tasks
    by = {a: [r for r in rows if r["arm"] == a] for a in names}
    assert all(r["arm_stats"]["rewritten"] == 1 for r in by["rtk"])
    assert all(r["arm_meta"]["library"] == "selective-context" for r in by["selective"])
    assert all(r["arm_stats"]["cleared_tool_uses"] == 0 for r in by["provider-cm"])
    assert len(c.beta_calls) == 15 and len(c.calls) == 60  # only provider-cm used the beta call
    assert sel_calls  # selective pruned the rtk-sized output for its own arm only
    assert (tmp_path / "transcripts" / "provider-cm" / "i0.json").exists()
    summary = report.analyse(
        rows,
        [{"instance_id": r["instance_id"], "arm": r["arm"], "status": "resolved"} for r in rows],
    )
    assert list(summary["arms"]) == list(names) and set(summary["comparisons"]) == set(names) - {
        "plain"
    }


def test_cli_run_refuses_before_spending_when_an_arm_dependency_is_missing(tmp_path, monkeypatch):
    f = tmp_path / "ids.txt"
    f.write_text("a__b-1\n")

    def boom(*a, **k):
        raise AssertionError("must fail before any API client / dataset load")

    monkeypatch.setattr(cli, "direct_client", boom)
    monkeypatch.setattr(cli, "load_records", boom)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "no-such-python"))
    base = ["run", "--instances", str(f), "--budget-usd", "1", "--i-understand-this-costs-money"]
    assert cli.main([*base, "--arms", "plain,selective", "--out", str(tmp_path / "o")]) == 2
    monkeypatch.setattr(
        arm_rtk, "fetch_rtk", lambda: (_ for _ in ()).throw(arms.ArmUnavailable("rtk: offline"))
    )
    assert cli.main([*base, "--arms", "plain,rtk", "--out", str(tmp_path / "o")]) == 2
    assert cli.main([*base, "--arms", "rtk", "--rtk-bin", str(tmp_path / "missing")]) == 2


def test_cli_run_reuses_arms_and_only_runs_the_rest(tmp_path, monkeypatch, capsys):
    from benchmarks.swebench_outcome import env as envmod

    src, out = tmp_path / "src", tmp_path / "out"
    _write(src / "results.jsonl", _rows("plain", 2, 1.0))
    (tmp_path / "ids.txt").write_text("i0\ni1\n")
    seen = {}

    def fake_run_all(tasks, o, client, factory, cfg, budget, arms_):
        seen.update(tasks=[t["instance_id"] for t in tasks], arms=[a.name for a in arms_])
        seen["spent"] = budget.spent
        return {"ran": 0}

    monkeypatch.setattr(cli, "run_all", fake_run_all)
    monkeypatch.setattr(cli, "direct_client", lambda: object())
    monkeypatch.setattr(
        cli, "load_records", lambda ids: [{"instance_id": i, "problem_statement": "p"} for i in ids]
    )
    monkeypatch.setattr(envmod, "DockerEnv", object)
    base = ["run", "--instances", str(tmp_path / "ids.txt"), "--out", str(out)]
    base += ["--budget-usd", "9", "--i-understand-this-costs-money", "--arms", "plain,provider-cm"]
    assert cli.main([*base, "--reuse-arm", f"plain={src}"]) == 0
    assert seen["arms"] == ["provider-cm"] and seen["spent"] == 0  # reused plain cost not charged
    assert "reused 2 plain rows" in capsys.readouterr().out
    assert cli.main([*base[:-2], "--arms", "plain", "--reuse-arm", f"plain={src}"]) == 0
    assert "nothing to run" in capsys.readouterr().out


# ---------------------------------------------------------------------------- grade / report


def test_grade_carries_over_reused_arm_grades(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _write(src / "grades.jsonl", [{"instance_id": "i0", "arm": "plain", "status": "resolved"}] * 1)
    _write(src / "results.jsonl", _rows("plain", 2, 1.0))
    run.reuse_arm(out, "plain", src, ["i0", "i1"], CFG)
    results = run.read_results(out / "results.jsonl")
    got = grade.carry_over(out, results, ("plain", "rtk"))
    assert got == [{"instance_id": "i0", "arm": "plain", "status": "resolved"}]
    assert grade.carry_over(out, results, ("rtk",)) == []  # arm not requested
    (src / "grades.jsonl").unlink()
    assert grade.carry_over(out, results, None) == []  # nothing to carry: it will be graded


def _multi(arms_, solved):
    res, grades = [], []
    for i in range(120):
        for a in arms_:
            res.append(
                {
                    "instance_id": f"i{i}",
                    "arm": a,
                    "failure_class": None,
                    "cost_usd": 1.0,
                    "steps": 3,
                    "usage": {"input": 5, "output": 1},
                    "arm_meta": {"library": a + "-lib", "version": "9.9"},
                    **({"reused_from": "somewhere"} if a == "plain" else {}),
                }
            )
            grades.append(
                {
                    "instance_id": f"i{i}",
                    "arm": a,
                    "status": "resolved" if solved(a, i) else "unresolved",
                }
            )
    return res, grades


def test_report_compares_every_arm_with_plain_and_prints_a_summary_table():
    def solved(a, i):
        return i < 60 or (a == "rtk" and i < 66) or (a == "selective" and i >= 100 and False)

    res, grades = _multi(("plain", "rtk", "selective", "provider-cm"), solved)
    a = report.analyse(res, grades)
    assert list(a["arms"]) == ["plain", "rtk", "selective", "provider-cm"]
    assert a["n_common"] == 120 and a["arms"]["rtk"]["resolved"] == 66
    c = a["comparisons"]
    assert (c["rtk"]["base_only"], c["rtk"]["arm_only"], c["rtk"]["n_pairs"]) == (0, 6, 120)
    assert c["rtk"]["diff"] == pytest.approx(0.05) and c["selective"]["diff"] == 0
    assert c["rtk"]["mcnemar_p"] == pytest.approx(2 * (1 / 2**6))
    assert (a["plain_only"], a["distil_only"]) == (0, 6)  # headline = first challenger (no distil)
    md = report.markdown(a, {"plain": {"from": "benchmarks/results/x", "rows": 120}})
    assert md.startswith("# SWE-bench Lite outcome eval (head-to-head)")
    assert "## rtk vs plain (120 pairs)" in md and "## provider-cm vs plain (120 pairs)" in md
    assert "| rtk | rtk-lib | 9.9 | 66/120 |" in md and "| plain | plain-lib | 9.9 | 60/120 |" in md
    assert "baseline" in md and "**reused** from `benchmarks/results/x` (120 rows" in md
    assert "3 comparisons against one baseline" in md and "0.0167" in md
    assert "reused" not in report.markdown(a).split("| arm |")[0]  # only when a manifest says so


def test_report_pairwise_sets_differ_when_an_arm_is_incomplete_and_needs_plain():
    res, grades = _multi(("plain", "rtk"), lambda a, i: i < 60)
    res = [
        r for r in res if not (r["arm"] == "rtk" and r["instance_id"] >= "i9")
    ]  # rtk stops early
    grades = [g for g in grades if not (g["arm"] == "rtk" and g["instance_id"] >= "i9")]
    a = report.analyse(res, grades)
    assert a["comparisons"]["rtk"]["n_pairs"] == a["n_common"] == a["arms"]["rtk"]["n"]
    with pytest.raises(SystemExit, match="needs the 'plain' arm"):
        report.analyse(res, grades, arms=("rtk",))
    only_plain = report.analyse(res, grades, arms=("plain",))
    assert only_plain["comparisons"] == {} and only_plain["n_pairs"] == 0
    assert "| plain |" in report.markdown(only_plain)
    assert report.arm_order({"zeta", "rtk", "plain"}) == ["plain", "rtk", "zeta"]


@pytest.mark.parametrize(
    "run_dir", sorted((ROOT / "benchmarks" / "results").glob("swebench-outcome-*"))
)
def test_committed_runs_still_re_report_identically(run_dir):
    """The refactor must not move a number: the generated part of every committed report.md
    (hand-written run notes follow it) is reproduced byte for byte from results + grades."""
    rp, gp = run_dir / "report.md", run_dir / "grades.jsonl"
    if not (rp.exists() and gp.exists()):
        pytest.skip("run has no committed report/grades")
    grades = [json.loads(ln) for ln in gp.read_text().splitlines()]
    text = rp.read_text()
    # a head-to-head report names its arms on an "Arms:" line; older ones are plain vs distil
    arms = ("plain", "distil")
    for ln in text.splitlines():
        if ln.startswith("Arms: "):
            arms = tuple(x.strip() for x in ln[6:].split(".")[0].split(","))
            break
    a = report.analyse(run.read_results(run_dir / "results.jsonl"), grades, 0.05, arms)
    manifest = run_dir / run.REUSE_MANIFEST
    reuse = json.loads(manifest.read_text()) if manifest.exists() else None
    assert text.startswith(report.markdown(a, reuse))


def test_selective_arm_records_what_it_pruned():
    """A head-to-head arm must leave evidence that it ran: how many tool results it pruned
    and the characters before/after, as RTK and provider-cm do."""
    from benchmarks.swebench_outcome.arm_selective import selective_arm

    arm = selective_arm(lambda t: t[: len(t) // 2])
    big = "x" * 1000
    msgs = [
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": big},
                {"type": "tool_result", "tool_use_id": "b", "content": "short"},
            ],
        },
    ]
    arm.transform(msgs)
    stats: dict = {}
    arm.on_response(None, stats)
    assert stats["tool_results"] == 2 and stats["pruned_results"] == 1
    assert stats["chars_before"] == 1005 and stats["chars_after"] < stats["chars_before"]
