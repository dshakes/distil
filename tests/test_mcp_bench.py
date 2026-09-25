"""distil mcp bench: the pre-registered harness, end to end in dry-run mode (no spend)."""

from __future__ import annotations

import argparse
import io
import json
import urllib.error

import pytest

from distil import cli
from distil.mcpproxy import bench, fakeserver


def test_tasks_are_deterministic_and_well_formed():
    a = bench.generate_tasks(300, seed=3)
    b = bench.generate_tasks(300, seed=3)
    assert [(t.prompt, t.gold) for t in a] == [(t.prompt, t.gold) for t in b]
    assert len({t.prompt for t in a}) == 300
    fx = bench.fixtures()
    for t in a:
        names = {x["name"] for x in fx[t.server]["tools"]}
        assert set(t.tools) <= names, t
        assert "{" not in t.prompt
        schema = next(x for x in fx[t.server]["tools"] if x["name"] == t.tools[0])["inputSchema"]
        assert set(t.gold) <= set(schema.get("properties", {})), t
        assert bench._schema_ok(fx[t.server], t.tools[0], t.gold), t  # gold is itself valid
    assert bench.generate_tasks(300, seed=4)[5].prompt != a[5].prompt


def test_result_tasks_hide_the_answer_in_long_output():
    for t in bench.generate_result_tasks(30):
        assert t.answer in t.result and len(t.result.splitlines()) > 50


def test_scoring():
    t = bench.Task("t", "git", ("git_log",), "p", {"repo_path": "/r", "max_count": 5})
    assert bench.score(t, "git_log", {"repo_path": "/r", "max_count": 5.0, "extra": 1}) == {
        "selection": 1,
        "args": 1,
    }
    assert bench.score(t, "git_log", {"repo_path": "/r", "max_count": "5"}) == {
        "selection": 1,
        "args": 1,
    }
    assert bench.score(t, "git_log", {"repo_path": "/r"}) == {"selection": 1, "args": 0}
    assert bench.score(t, "git_status", {"repo_path": "/r", "max_count": 5}) == {
        "selection": 0,
        "args": 0,
    }
    assert bench.score(t, None, None) == {"selection": 0, "args": 0}
    assert bench._eq(True, "true") and not bench._eq(True, 1) and bench._eq(["a", "b"], ["b", "a"])
    assert not bench._eq(5, "five") and not bench._eq(["a"], ["a", "a"])


def test_required_n_matches_the_protocol():
    assert bench.required_n(bench.P_DISCORDANT) <= bench.N_TOOL_TASKS
    assert bench.required_n(bench.P_DISCORDANT_R) <= bench.N_RESULT_TASKS
    assert bench.required_n(0.06, margin=0.02, alpha=0.05, power=0.8) == 928


def test_compare_detects_equivalence_and_harm():
    base = [1] * 950 + [0] * 50
    same = compare = bench.compare(base, list(base))
    assert same["non_inferior"] and same["mean_diff"] == 0
    worse = [0] * 80 + base[80:]
    compare = bench.compare(base, worse)
    assert not compare["non_inferior"] and compare["losses"] == 80
    with pytest.raises(ValueError):
        bench.compare([1], [1, 0])


def test_fixed_sequence_stops_at_the_first_failure():
    good = [1] * 1000
    bad = [0] * 100 + [1] * 900
    arms = {
        "raw": {"selection": good, "args": good},
        "L0": {"selection": good, "args": good},
        "L1": {"selection": bad, "args": good},
        "L2": {"selection": good, "args": good},
    }
    v = bench.verdicts(arms)
    assert v["L0"]["status"] == "certified" and v["L1"]["status"] == "failed"
    assert v["L2"]["status"] == "not-tested"
    noisy = [1 if i % 5 else 0 for i in range(1000)]
    shifted = noisy[1:] + noisy[:1]  # same rate, 40% discordant: underpowered
    v = bench.verdicts(
        {"raw": {"selection": noisy, "args": noisy}, "L0": {"selection": shifted, "args": shifted}}
    )
    assert v["L0"]["status"] in ("inconclusive", "failed")


@pytest.fixture(scope="module")
def oracle_run():
    return bench.run(model=bench.scripted("oracle"), n_tools=96, n_results=30)


def test_dry_run_oracle_certifies_the_plumbing(oracle_run):
    s = oracle_run["summary"]
    for arm in ("raw", "L0", "L1", "L2", "L3"):
        assert s["arms"][arm]["selection"] == 1.0 and s["arms"][arm]["args"] == 1.0, arm
        assert s["arms"][arm]["success"] == 1.0, arm
    assert s["arms"]["raw"]["extra_round_trips"] == 0
    assert s["arms"]["L2"]["mean_round_trips"] == 2.0  # schema fetch, then the real tool
    assert s["arms"]["L3"]["mean_round_trips"] < 2.0  # pinned tools skip the fetch
    assert (
        s["arms"]["L2"]["tools_tokens"]
        < s["arms"]["L0"]["tools_tokens"]
        < s["arms"]["raw"]["tools_tokens"]
    )
    assert s["results"]["R"]["correct"] == 1.0 and s["results"]["R"]["mean_expands"] > 0
    assert s["results"]["R"]["mean_input_tokens"] < s["results"]["raw"]["mean_input_tokens"]
    assert all(v["status"] == "certified" for v in oracle_run["verdicts"].values())


def test_dry_run_catches_a_degraded_level():
    res = bench.run(
        model=bench.scripted("degraded", degrade={"L2": 0.3}),
        n_tools=200,
        n_results=6,
        arms=("raw", "L2"),
    )
    assert res["verdicts"]["L2"]["status"] == "failed"
    assert res["summary"]["arms"]["L2"]["selection"] < res["summary"]["arms"]["raw"]["selection"]


def test_dry_run_invoke_fallback_and_no_expand():
    res = bench.run(model=bench.scripted("invoke"), n_tools=48, n_results=9, arms=("raw", "L2"))
    assert res["summary"]["arms"]["L2"]["selection"] == 1.0
    res = bench.run(model=bench.scripted("no-expand"), n_tools=4, n_results=30, arms=("raw",))
    assert res["summary"]["results"]["R"]["correct"] < 1.0
    assert res["verdicts"]["R"]["status"] == "failed"


def test_cost_estimate(oracle_run):
    runs = bench.run(model=bench.scripted("oracle"), n_tools=48, n_results=6)["_runs"]
    est = bench.estimate_cost(runs["tools"], runs["results"])
    assert set(est["models"]) == set(bench.LIVE_MODELS)
    for m, row in est["models"].items():
        assert 0 < row["usd_cached"] < row["usd_no_cache"], m
    h, o = est["models"]["claude-haiku-4-5"], est["models"]["claude-opus-5"]
    assert h["usd_no_cache"] < o["usd_no_cache"]


class _Resp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def test_live_model_is_capped_and_parses_tool_use(monkeypatch):
    sent = []
    replies = iter(
        [
            urllib.error.HTTPError("u", 529, "overloaded", {}, io.BytesIO()),
            {
                "content": [
                    {"type": "tool_use", "name": "git_status", "input": {"repo_path": "/r"}}
                ],
                "usage": {"input_tokens": 1_000_000, "output_tokens": 10},
            },
            {"content": [{"type": "text", "text": "E1234"}], "usage": {"input_tokens": 10}},
        ]
    )

    def opener(req, timeout):
        sent.append(json.loads(req.data))
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return _Resp(r)

    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    m = bench.AnthropicModel("claude-haiku-4-5", budget_usd=1.0, api_key="k", opener=opener)
    tools = fakeserver.load_fixture("git")["tools"][:2]
    turn = m(tools, [{"role": "user", "text": "status?"}], None)
    assert turn.name == "git_status" and turn.args == {"repo_path": "/r"}
    assert (
        sent[-1]["tools"][-1]["cache_control"] == {"type": "ephemeral"}
        and sent[-1]["temperature"] == 0
    )
    assert m.spent >= 1.0
    with pytest.raises(bench.BudgetExceeded):
        m(tools, [], None)
    m.budget = 100
    assert m([], [{"role": "user", "text": "q"}], None).text == "E1234"


def test_live_model_needs_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        bench.AnthropicModel("claude-haiku-4-5", 1.0)


def test_live_model_gives_up_on_client_errors():
    def opener(req, timeout):
        raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO())

    m = bench.AnthropicModel("claude-haiku-4-5", budget_usd=1.0, api_key="k", opener=opener)
    with pytest.raises(urllib.error.HTTPError):
        m([], [{"role": "user", "text": "q"}], None)


def test_history_to_messages():
    msgs = bench._to_messages(
        [
            {"role": "user", "text": "q"},
            {"role": "call", "name": "t", "args": {}},
            {"role": "tool", "text": "r"},
        ]
    )
    assert msgs[1]["content"][0]["id"] == msgs[2]["content"][0]["tool_use_id"]


def _ns(tmp_path, **kw):
    base = dict(
        mcp_cmd="bench",
        live=False,
        model=None,
        budget_usd=None,
        behavior="oracle",
        n=48,
        n_results=6,
        seed=0,
        out=str(tmp_path),
    )
    return argparse.Namespace(**{**base, **kw})


def test_cli_dry_run_writes_results_and_estimate(tmp_path, capsys):
    assert cli.cmd_mcp(_ns(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "no API calls" in out and "cost estimate (nothing spent)" in out
    res = json.loads((tmp_path / "dryrun-oracle.json").read_text())
    assert res["mode"].startswith("dry-run") and "_runs" not in res and "certificate" not in res
    assert set(json.loads((tmp_path / "cost_estimate.json").read_text())["models"]) == set(
        bench.LIVE_MODELS
    )
    assert cli.cmd_mcp(_ns(tmp_path, behavior="noisy")) == 0
    assert not (tmp_path / "cost_estimate.json").read_text() == ""


def test_cli_live_refuses_without_cap_or_over_budget(tmp_path, capsys, monkeypatch):
    assert cli.cmd_mcp(_ns(tmp_path, live=True)) == 2
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="gpt-9", budget_usd=5)) == 2
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="claude-opus-5", budget_usd=0.01)) == 1
    assert "nothing was sent" in capsys.readouterr().err


def test_cli_live_run_with_a_stubbed_model(tmp_path, monkeypatch, capsys):
    class Stub:
        def __init__(self, model, budget):
            self.spent = 0.5
            self.inner = bench.scripted("oracle")

        def __call__(self, *a):
            return self.inner(*a)

    monkeypatch.setattr(bench, "AnthropicModel", Stub)
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="claude-haiku-4-5", budget_usd=1000)) == 0
    res = json.loads((tmp_path / "live-claude-haiku-4-5.json").read_text())
    assert res["mode"] == "live" and res["spent_usd"] == 0.5

    class Broke(Stub):
        def __call__(self, *a):
            raise bench.BudgetExceeded("cap")

    monkeypatch.setattr(bench, "AnthropicModel", Broke)
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="claude-haiku-4-5", budget_usd=1000)) == 1

    def nokey(*a):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(bench, "AnthropicModel", nokey)
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="claude-haiku-4-5", budget_usd=1000)) == 1
