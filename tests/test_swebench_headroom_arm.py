"""Tests for the `headroom` arm of benchmarks/swebench_outcome: the real Headroom proxy in front of the
Anthropic API. Offline: fakes for the process boundary, and (importorskip) the REAL pinned
`headroom-ai[proxy]` against a local stub upstream, so no Anthropic call is ever made."""

from __future__ import annotations

import http.server
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

anthropic = pytest.importorskip("anthropic", reason="anthropic SDK not installed")

from benchmarks.swebench_outcome import agent as ag  # noqa: E402
from benchmarks.swebench_outcome import arm_headroom as hr  # noqa: E402
from benchmarks.swebench_outcome import arms, cli, run  # noqa: E402
from benchmarks.swebench_outcome.env import FakeEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TASK = {"instance_id": "a__b-1", "problem_statement": "fix foo"}


class FakeProc:
    def __init__(self, code=None):
        self.code, self.returncode, self.killed = code, code, False

    def poll(self):
        return self.code

    def terminate(self):
        self.code = self.returncode = 0

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.code


def _ok(version=hr.HR_VERSION):
    return lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout=json.dumps({"headroom-ai": version}) + "\n", stderr=""
    )


def test_names_and_arm_meta_record_the_pinned_version():
    assert "headroom" in arms.ARM_NAMES and arms.parse_arms("plain,headroom") == (
        "plain",
        "headroom",
    )
    a = hr.headroom_arm("http://127.0.0.1:1", lambda u: NS(base_url=u), {})
    assert a.client.base_url == "http://127.0.0.1:1" and not a.transform and not a.tools
    assert a.meta["library"] == "headroom-ai" and a.meta["version"] == hr.HR_VERSION == "0.40.0"


def test_probe_refuses_missing_broken_or_wrong_version(tmp_path):
    with pytest.raises(arms.ArmUnavailable, match="cannot run"):
        hr.probe(str(tmp_path / "no-such-python"))

    def bad(*a, **k):
        return subprocess.CompletedProcess(
            a, 1, stdout="", stderr="ImportError: x\nModuleNotFoundError: fastapi"
        )

    with pytest.raises(
        arms.ArmUnavailable, match="not importable.*fastapi.*headroom-ai\\[proxy\\]"
    ):
        hr.probe(sys.executable, run=bad)
    with pytest.raises(arms.ArmUnavailable, match="pinned headroom-ai==0.40.0, found 0.38.0"):
        hr.probe(sys.executable, run=_ok("0.38.0"))
    assert hr.probe(sys.executable, run=_ok())["headroom-ai"] == hr.HR_VERSION


def test_proxy_is_started_like_headroom_wrap_and_isolated_from_the_real_home(monkeypatch):
    monkeypatch.setattr(hr, "probe", lambda p: {"headroom-ai": hr.HR_VERSION})
    monkeypatch.setenv("HEADROOM_WRAP_OWNED", "1")
    seen = {}

    def popen(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return FakeProc()

    monkeypatch.setattr(
        hr.urllib.request, "urlopen", lambda *a, **k: NS(status=200, __enter__=None)
    )
    monkeypatch.setattr(hr.HeadroomProxy, "_wait_live", lambda self, t: None)
    p = hr.HeadroomProxy("py", port=4321, popen=popen)
    assert seen["cmd"][:6] == ["py", "-m", "headroom.cli", "proxy", "--port", "4321"]
    assert "--workers" in seen["cmd"] and "--no-subscription-tracking" in seen["cmd"]
    env = seen["env"]
    assert env["HEADROOM_AGENT_TYPE"] == "claude" and env["HEADROOM_STACK"] == "wrap_claude"
    assert (
        "HEADROOM_WRAP_OWNED" not in env
    )  # that flag makes the proxy exit without a `wrap` client
    assert env["HOME"] == env["HEADROOM_WORKSPACE_DIR"] != str(Path.home())
    assert p.url == "http://127.0.0.1:4321"
    p.close()
    assert p.proc.poll() == 0


def test_a_proxy_that_dies_before_it_is_live_refuses_with_its_log(monkeypatch):
    monkeypatch.setattr(hr, "probe", lambda p: {"headroom-ai": hr.HR_VERSION})

    def popen(cmd, stdout, **kw):
        stdout.write("boom: no module named headroom._core\n")
        stdout.flush()
        return FakeProc(code=3)

    with pytest.raises(arms.ArmUnavailable, match=r"exited \(code 3\).*headroom._core"):
        hr.HeadroomProxy("py", popen=popen)


def test_a_proxy_that_never_comes_up_is_stopped_and_refuses(monkeypatch):
    monkeypatch.setattr(hr, "probe", lambda p: {"headroom-ai": hr.HR_VERSION})
    proc = FakeProc()
    with pytest.raises(arms.ArmUnavailable, match="not live after"):
        hr.HeadroomProxy("py", port=1, popen=lambda *a, **k: proc, timeout=0.2)
    assert proc.code == 0  # terminated, not leaked


def test_agent_sends_a_headroom_arm_through_its_own_client_and_never_the_direct_one():
    class C:
        def __init__(self, tag):
            self.tag, self.calls = tag, []
            self.messages = NS(create=self.create)

        def create(self, **kw):
            self.calls.append(kw)
            if self.tag == "direct":
                raise AssertionError("headroom arm used the direct client")
            blk = NS(
                type="text", text="done", model_dump=lambda **_: {"type": "text", "text": "done"}
            )
            u = NS(
                input_tokens=10,
                output_tokens=1,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
            return NS(content=[blk], stop_reason="end_turn", usage=u)

    proxy_client = C("proxy")
    arm = arms.Arm("headroom", client=proxy_client, meta={"version": hr.HR_VERSION})
    r = ag.run_agent(
        C("direct"), FakeEnv({"f.py": "x = 1\n"}), TASK, arm, ag.Cfg(), ag.Budget(None)
    )
    assert len(proxy_client.calls) == 1
    assert r["arm"] == "headroom" and r["arm_meta"]["version"] == "0.40.0"


def test_cli_run_refuses_before_spend_when_headroom_is_missing_and_closes_earlier_arms(
    tmp_path, monkeypatch
):
    f = tmp_path / "ids.txt"
    f.write_text("a__b-1\n")

    def boom(*a, **k):
        raise AssertionError("must fail before any API client / dataset load")

    monkeypatch.setattr(cli, "direct_client", boom)
    monkeypatch.setattr(cli, "load_records", boom)
    base = ["run", "--instances", str(f), "--budget-usd", "1", "--i-understand-this-costs-money"]
    rc = cli.main(
        [*base, "--arms", "plain,headroom", "--headroom-python", str(tmp_path / "nope"),
         "--out", str(tmp_path / "o")]
    )  # fmt: skip
    assert rc == 2
    # an arm that started a process must be closed when a later arm refuses
    closed = []
    import benchmarks.swebench_outcome.arm_selective as sel

    monkeypatch.setattr(
        sel,
        "make_selective",
        lambda py: (arms.Arm("selective"), NS(close=lambda: closed.append(1))),
    )
    monkeypatch.setattr(
        hr, "make_headroom", lambda py: (_ for _ in ()).throw(arms.ArmUnavailable("hr"))
    )
    assert cli.main([*base, "--arms", "selective,headroom", "--out", str(tmp_path / "o")]) == 2
    assert closed == [1]


def test_plan_calibrate_prices_headroom_as_an_unmeasured_arm_from_the_committed_run(
    tmp_path, capsys
):
    cal = ROOT / "benchmarks" / "results" / "swebench-outcome-300-medium"
    plain = [r for r in run.read_results(cal / "results.jsonl") if r["arm"] == "plain"]
    idf = tmp_path / "ids.txt"
    idf.write_text("\n".join(r["instance_id"] for r in plain))
    rc = cli.main(
        ["plan", "--instances", str(idf), "--arms", "plain,distil,headroom", "--calibrate", str(cal),
         f"--reuse-arm=plain={cal}", f"--reuse-arm=distil={cal}", "--effort", "medium", "--max-steps", "60"]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert (
        rc == 0 and "headroom: $0.0426/task x 300 tasks" in out and "proxy: plain mean x1.5" in out
    )


# ------------------------------------------------------------ the real Headroom, stub upstream


class _Upstream(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        _Upstream.seen.append((self.path, json.loads(body)))
        out = json.dumps(
            {
                "id": "msg_x", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        ).encode()  # fmt: skip
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_real_headroom_proxy_receives_the_arms_requests_and_forwards_upstream(monkeypatch):
    """Skipped unless headroom-ai[proxy]==0.40.0 is importable here. Starts the REAL proxy process,
    sends a request through the arm's client, and checks a stub upstream got it from Headroom."""
    pytest.importorskip("headroom", reason="headroom-ai not installed")
    pytest.importorskip("fastapi", reason="headroom-ai[proxy] not installed")
    try:
        hr.probe(sys.executable)
    except arms.ArmUnavailable as e:
        pytest.skip(str(e))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Upstream.seen.clear()
    up = f"http://127.0.0.1:{srv.server_address[1]}"
    proxy = hr.HeadroomProxy(sys.executable, extra_env={"ANTHROPIC_TARGET_API_URL": up})
    try:
        arm = hr.headroom_arm(
            proxy.url, lambda u: anthropic.Anthropic(base_url=u, api_key="k", max_retries=0), {}
        )
        rows = [
            {"id": i, "status": "ok", "path": f"/src/m{i}.py", "msg": "line %d" % i}
            for i in range(400)
        ]
        r = arm.client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=16,
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": json.dumps(rows)}
                    ],
                },
            ],
        )
        assert r.content[0].text == "ok"
        with urllib.request.urlopen(proxy.url + "/livez", timeout=5) as h:
            assert json.loads(h.read())["service"] == "headroom-proxy"
    finally:
        proxy.close()
        srv.shutdown()
    assert [p for p, _ in _Upstream.seen] == ["/v1/messages"]  # reached upstream only via Headroom
