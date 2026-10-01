"""Offline tests for benchmarks/swebench_outcome. No API, network, docker, or swebench needed."""

from __future__ import annotations

import json
import socket
from types import SimpleNamespace as NS

import pytest

anthropic = pytest.importorskip("anthropic", reason="anthropic SDK not installed")


from benchmarks.swebench_outcome import agent as ag  # noqa: E402
from benchmarks.swebench_outcome import cli, grade, report, run, stats  # noqa: E402
from benchmarks.swebench_outcome.env import EnvError, FakeEnv  # noqa: E402


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


def resp(blocks, stop="tool_use", tin=1000, tout=100):
    return NS(
        content=blocks,
        stop_reason=stop,
        usage=NS(
            input_tokens=tin,
            output_tokens=tout,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


TASK = {"instance_id": "a__b-1", "problem_statement": "fix foo"}
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


def go(script, arm="plain", env=None, cfg=None, budget=None, **kw):
    env = env or FakeEnv({"f.py": "x = 1\n"})
    c = FakeClient(script)
    r = ag.run_agent(c, env, TASK, arm, cfg or ag.Cfg(), budget or ag.Budget(None), **kw)
    return r, c, env


def test_loop_edits_and_yields_patch():
    r, c, env = go(EDIT)
    assert r["failure_class"] is None and r["steps"] == 3
    assert "-x = 1" in r["patch"] and "+x = 2" in r["patch"]
    assert env.cmds == ["cat f.py"]
    kw = c.calls[0]
    assert kw["tools"][:2] == [
        {"type": "bash_20250124", "name": "bash"},
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
    ]
    assert len(kw["tools"]) == 2 and kw["thinking"] == {"type": "adaptive"}
    assert kw["tool_choice"] == {"type": "auto"}
    assert r["usage"]["input"] == 3000 and r["cost_usd"] > 0


def test_distil_arm_compresses_and_expands():
    stores = []

    class Store:
        def expand(self, h):
            if h != "deadbeef":
                raise KeyError(h)
            return "ORIGINAL"

    def fake_compress(msgs):
        stores.append(len(msgs))
        return list(msgs), Store()

    script = [
        resp([tu("t1", "distil_expand", handle="deadbeef")]),
        resp([tu("t2", "distil_expand", handle="00000000")]),
        resp([txt("x")], stop="end_turn"),
    ]
    r, c, _ = go(script, arm="distil", compress=fake_compress)
    assert stores == [1, 3, 5]  # once before every model call
    assert any(t["name"] == "distil_expand" for t in c.calls[0]["tools"])
    tr = c.calls[1]["messages"][-1]["content"][0]
    assert tr["content"] == "ORIGINAL"
    assert r["expand_calls"] == 2 and r["expand_misses"] == 1


def test_real_compress_messages_is_the_default():
    from distil.adapters.anthropic import compress_messages

    r, c, _ = go([resp([txt("x")], stop="end_turn")], arm="distil")
    assert (
        c.calls[0]["messages"]
        == compress_messages([{"role": "user", "content": "Resolve this issue:\n\nfix foo"}])[0]
    )
    assert r["failure_class"] == "gave_up"


def test_plain_arm_has_no_expand_tool_and_sends_messages_as_is():
    _, c, _ = go(EDIT)
    assert all(t["name"] != "distil_expand" for t in c.calls[0]["tools"])
    assert c.calls[0]["messages"][0]["content"].endswith("fix foo")


def test_budget_stop():
    b = ag.Budget(0.0001)
    r, c, _ = go(EDIT, budget=b)
    assert r["failure_class"] == "budget_stopped" and r["steps"] == 1 and len(c.calls) == 1


def test_failure_classes():
    r, *_ = go([anthropic.APIError("boom", request=NS(), body=None)])
    assert r["failure_class"] == "api_error"
    r, *_ = go(EDIT, cfg=ag.Cfg(task_timeout=-1))
    assert r["failure_class"] == "timeout"
    r, *_ = go([resp([txt("no idea")], stop="end_turn")])
    assert r["failure_class"] == "gave_up"
    r, *_ = go([resp([tu("t", "bash", command="x")])] * 2, cfg=ag.Cfg(max_steps=2))
    assert r["stop"] == "step_limit" and r["failure_class"] == "gave_up"  # no edits, no patch

    class Boom(FakeEnv):
        def exec(self, cmd, timeout=120):
            raise EnvError("container died")

    r, *_ = go([resp([tu("t", "bash", command="x")])], env=Boom())
    assert r["failure_class"] == "env_error"


def test_editor_rejects_ambiguous_replace():
    env = FakeEnv({"f": "a\na\n"})
    out = ag.run_editor(
        env, {"command": "str_replace", "path": "f", "old_str": "a", "new_str": "b"}
    )
    assert out.startswith("error") and env.files["f"] == "a\na\n"


def _tasks(n):
    return [{"instance_id": f"i{k}", "problem_statement": "p"} for k in range(n)]


def test_resume_skips_completed_and_budget_persists(tmp_path):
    def factory(_):
        return FakeEnv({"f.py": "x = 1\n"})

    c1 = FakeClient(EDIT * 2)
    s = run.run_all(_tasks(1), tmp_path, c1, factory, ag.Cfg(), ag.Budget(None))
    assert s["ran"] == 2
    c2 = FakeClient(EDIT * 2)
    s = run.run_all(
        _tasks(2),
        tmp_path,
        c2,
        factory,
        ag.Cfg(),
        ag.Budget(None),
        compress=lambda m: (list(m), NS(expand=lambda h: "")),
    )
    assert s["skipped"] == 2 and s["ran"] == 2
    rows = run.read_results(tmp_path / "results.jsonl")
    assert len(rows) == 4 and "transcript" not in rows[0]
    assert (tmp_path / "transcripts" / "plain" / "i0.json").exists()
    assert [r["arm"] for r in rows if r["instance_id"] == "i1"] == ["distil", "plain"]  # alternates


def test_run_all_stops_on_budget_without_recording_partial(tmp_path):
    s = run.run_all(
        _tasks(2),
        tmp_path,
        FakeClient(EDIT * 4),
        lambda _: FakeEnv({"f.py": "x = 1\n"}),
        ag.Cfg(),
        ag.Budget(0.0001),
        compress=lambda m: (list(m), NS(expand=None)),
    )
    assert s["stopped_on_budget"] == 1 and s["ran"] == 0
    assert run.read_results(tmp_path / "results.jsonl") == []


def test_env_factory_failure_is_classed(tmp_path):
    def bad(_):
        raise RuntimeError("no image")

    run.run_all(_tasks(1), tmp_path, FakeClient([]), bad, ag.Cfg(), ag.Budget(None))
    assert {r["failure_class"] for r in run.read_results(tmp_path / "results.jsonl")} == {
        "env_error"
    }


def test_stats_hand_computed():
    lo, hi = stats.wilson(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)
    assert stats.mcnemar_exact(10, 4) == pytest.approx(2 * 1471 / 16384)
    assert stats.mcnemar_exact(0, 0) == 1.0 and stats.mcnemar_exact(5, 5) == 1.0
    d, lo, hi = stats.paired_diff(10, 4, 100)  # distil worse by 6 pts
    assert d == pytest.approx(-0.06)
    assert (lo, hi) == pytest.approx(
        (-0.06 - 1.959964 * 0.036932, -0.06 + 1.959964 * 0.036932), abs=1e-4
    )
    assert stats.sample_size_noninferiority(0.05, 0.15) == 471


def test_report_pairs_and_verdict():
    res, grades = [], []
    for i in range(40):  # 40 pairs, identical outcomes except 1 each way
        for arm in ("plain", "distil"):
            res.append(
                {
                    "instance_id": f"i{i}",
                    "arm": arm,
                    "failure_class": None,
                    "cost_usd": 1.0,
                    "steps": 5,
                    "usage": {"input": 10, "output": 2},
                    "expand_calls": 1 if arm == "distil" else 0,
                }
            )
            solved = i < 20 or (arm == "plain" and i == 20) or (arm == "distil" and i == 21)
            grades.append(
                {
                    "instance_id": f"i{i}",
                    "arm": arm,
                    "status": "resolved" if solved else "unresolved",
                }
            )
    res.append({"instance_id": "x", "arm": "plain", "failure_class": "api_error", "cost_usd": 0})
    a = report.analyse(res, grades)
    assert (a["n_pairs"], a["plain_only"], a["distil_only"], a["both"]) == (40, 1, 1, 20)
    assert (
        a["arms"]["distil"]["expand_calls"] == 40
        and a["arms"]["plain"]["classes"]["api_error"] == 1
    )
    assert a["verdict"] == "INCONCLUSIVE"  # n=40 cannot clear a 5pt margin
    assert "INCONCLUSIVE" in report.markdown(a)


def test_parse_report_and_predictions(tmp_path):
    rep = {
        "resolved_ids": ["a"],
        "unresolved_ids": ["b"],
        "empty_patch_ids": ["c"],
        "error_ids": ["d"],
    }
    assert grade.parse_report(rep) == {
        "a": "resolved",
        "b": "unresolved",
        "c": "unresolved",
        "d": "grader_error",
    }
    rows = [
        {"instance_id": "a", "arm": "plain", "failure_class": None, "patch": "p"},
        {"instance_id": "b", "arm": "plain", "failure_class": "api_error", "patch": "p"},
    ]
    paths = grade.write_predictions(rows, tmp_path)
    (line,) = paths["plain"].read_text().splitlines()
    assert json.loads(line) == {
        "instance_id": "a",
        "model_name_or_path": "swo-plain",
        "model_patch": "p",
    }


def test_grade_skips_without_swebench(tmp_path):
    pytest.importorskip("swebench", reason="swebench not installed (pip install swebench)")
    pytest.importorskip("docker", reason="docker SDK not installed")


def test_grade_errors_clearly_without_swebench(tmp_path, monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "swebench":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(SystemExit, match="swebench is not installed"):
        grade.grade(tmp_path)


def test_plan_makes_no_network_calls(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("network/API touched by plan")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(run, "load_records", boom)
    f = tmp_path / "ids.txt"
    f.write_text("a__b-1\na__b-2\n# c\na__b-3\n")
    assert cli.main(["plan", "--instances", str(f), "--limit", "2", "--seed", "1"]) == 0
    text = capsys.readouterr().out
    assert "tasks=2" in text and "ESTIMATE" in text and "PLACEHOLDER" in text
    assert run.select_ids(["a", "b", "c", "d"], 1, None) == run.select_ids(
        ["d", "c", "b", "a"], 1, None
    )


def test_run_refuses_without_flags(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("a__b-1\n")
    assert cli.main(["run", "--instances", str(f), "--out", str(tmp_path)]) == 2
    assert (
        cli.main(["run", "--instances", str(f), "--budget-usd", "1", "--out", str(tmp_path)]) == 2
    )
