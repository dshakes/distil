"""distil mcp bench: the pre-registered harness, end to end in dry-run mode (no spend)."""

from __future__ import annotations

import argparse
import io
import json
import urllib.error

import pytest

from distil import cli, pricing
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


def test_live_model_is_metered_and_parses_tool_use(monkeypatch):
    sent = []
    replies = iter(
        [
            urllib.error.HTTPError("u", 529, "overloaded", {"retry-after": "3"}, io.BytesIO()),
            {
                "content": [
                    {"type": "tool_use", "name": "git_status", "input": {"repo_path": "/r"}}
                ],
                "usage": {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 2000,
                    "cache_read_input_tokens": 50_000,
                    "output_tokens": 10,
                },
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

    slept = []
    monkeypatch.setattr(bench.time, "sleep", slept.append)
    m = bench.AnthropicModel("claude-haiku-4-5", budget_usd=1.0, api_key="k", opener=opener)
    tools = fakeserver.load_fixture("git")["tools"][:2]
    turn = m(tools, [{"role": "user", "text": "status?"}], None)
    assert turn.name == "git_status" and turn.args == {"repo_path": "/r"}
    assert (
        sent[-1]["tools"][-1]["cache_control"] == {"type": "ephemeral"}
        and sent[-1]["temperature"] == 0
    )
    assert slept == [3.0]  # retry-after honoured
    # Every attempt is on the meter: the failed one at its estimated input cost, the
    # billed one at usage x list price (cache write / read priced separately).
    fail, ok = m.meter.calls
    assert fail["status"] == 529 and fail["estimated"] and fail["usd"] > 0
    p = pricing.get("claude-haiku-4-5")
    billed = 100 * p.input + 2000 * p.cache_write + 50_000 * p.cache_read + 10 * p.output
    assert ok == {
        "arm": "",
        "task": "",
        "attempt": 1,
        "status": 200,
        "in": 100,
        "cw": 2000,
        "cr": 50_000,
        "out": 10,
        "usd": round(billed, 6),
    }
    assert m.spent == pytest.approx(fail["usd"] + billed) and m.meter.reserved == pytest.approx(0)
    assert m([], [{"role": "user", "text": "q"}], None).text == "E1234"
    m.meter.ceiling = m.spent  # no headroom left: the next call is refused unsent
    before = len(sent)
    with pytest.raises(bench.BudgetExceeded):
        m(tools, [{"role": "user", "text": "q"}], None)
    assert len(sent) == before


def test_retry_after_is_bounded():
    def err(h):
        return urllib.error.HTTPError("u", 429, "rl", h, io.BytesIO())

    assert bench._retry_after(err({}), 0) == 2.0
    assert bench._retry_after(err({"retry-after": "junk"}), 1) == 4.0
    assert bench._retry_after(err({"retry-after": "999"}), 0) == 60.0


def test_retries_are_refused_at_the_ceiling(monkeypatch):
    """A retry is a new attempt: it must pass the meter too, or the ceiling leaks."""
    n = []

    def opener(req, timeout):
        n.append(1)
        raise urllib.error.HTTPError("u", 503, "down", {}, io.BytesIO())

    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    m = bench.AnthropicModel("claude-haiku-4-5", budget_usd=1.0, api_key="k", opener=opener)
    with pytest.raises(urllib.error.HTTPError):
        m([], [{"role": "user", "text": "q"}], None)
    assert len(n) == bench.AnthropicModel.ATTEMPTS and len(m.meter.calls) == len(n)
    tiny = bench.AnthropicModel("claude-haiku-4-5", budget_usd=1e-9, api_key="k", opener=opener)
    with pytest.raises(bench.BudgetExceeded):
        tiny([], [{"role": "user", "text": "q"}], None)


def test_transport_errors_retry_then_abort(monkeypatch):
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)

    def opener(req, timeout):
        raise TimeoutError("slow")

    m = bench.AnthropicModel("claude-haiku-4-5", budget_usd=1.0, api_key="k", opener=opener)
    with pytest.raises(TimeoutError):
        m([], [{"role": "user", "text": "q"}], None)
    assert [c["status"] for c in m.meter.calls] == ["TimeoutError"] * 4


def _fake_api(usage):
    """An opener that answers like the API: the oracle's decision, fixed usage."""
    oracle = bench.scripted("oracle")
    tasks = {t.prompt: t for t in bench.generate_tasks(48, 0)}
    rtasks = {t.prompt: t for t in bench.generate_result_tasks(6, 0)}
    sent = []

    def opener(req, timeout):
        body = json.loads(req.data)
        sent.append(body)
        if req.full_url.endswith("count_tokens"):
            return _Resp({"input_tokens": 1234})
        prompt = body["messages"][0]["content"]
        task = tasks.get(prompt) or rtasks[prompt]
        hist = [{"role": "user", "text": prompt}]
        for msg in body["messages"][1:]:
            for b in msg["content"] if isinstance(msg["content"], list) else []:
                if b["type"] == "tool_use":
                    hist.append({"role": "call", "name": b["name"], "args": b["input"]})
                elif b["type"] == "tool_result":
                    hist.append({"role": "tool", "text": b["content"]})
        tools = [{"name": t["name"]} for t in body["tools"]]
        turn = oracle(tools, hist, ("x", task))
        content = (
            [{"type": "tool_use", "name": turn.name, "input": turn.args}]
            if turn.name
            else [{"type": "text", "text": turn.text}]
        )
        return _Resp({"content": content, "usage": usage})

    return opener, sent


@pytest.mark.parametrize("ceiling", [0.05, 0.25])
def test_meter_stops_a_mock_live_run_at_a_tiny_ceiling(ceiling):
    """The dry run of the ceiling. $0.05 is below one raw-arm call's worst case, so
    nothing is sent; $0.25 stops the tool family mid-run. Either way: no verdicts."""
    usage = {"input_tokens": 200, "cache_read_input_tokens": 9000, "output_tokens": 60}
    opener, sent = _fake_api(usage)
    m = bench.AnthropicModel("claude-haiku-4-5", budget_usd=ceiling, api_key="k", opener=opener)
    res = bench.run(model=m, n_tools=48, n_results=6, workers=4, stop_at_failure=True)
    assert res["stopped"]["family"] == "tools"
    assert {v["status"] for v in res["verdicts"].values()} == {"no-verdict"}
    assert m.spent <= ceiling and m.meter.reserved == pytest.approx(0)
    assert (m.spent > 0) == (ceiling == 0.25)
    assert len(m.meter.calls) == len(sent)  # every call that went out is on the meter
    assert sum(c["usd"] for c in m.meter.calls) == pytest.approx(m.spent)


def test_mock_live_run_under_budget_completes_and_meters_everything():
    opener, sent = _fake_api({"input_tokens": 50, "output_tokens": 5})
    m = bench.AnthropicModel("claude-haiku-4-5", budget_usd=50, api_key="k", opener=opener)
    res = bench.run(model=m, n_tools=48, n_results=6, workers=4, stop_at_failure=True)
    assert res["stopped"] is None and res["verdicts"]["R"]["status"] != "no-verdict"
    assert len(m.meter.calls) == len(sent) and m.spent == pytest.approx(
        len(sent) * m.meter.cost({"input_tokens": 50, "output_tokens": 5})
    )
    assert m.preflight({"raw": [{"name": "a"}]}) == {"raw": 1234}


def test_budget_stop_in_R_keeps_tool_verdicts():
    calls = {"n": 0}
    inner = bench.scripted("oracle")

    def model(tools, history, ctx):
        if isinstance(ctx[1], bench.ResultTask):
            raise bench.BudgetExceeded("cap")
        calls["n"] += 1
        return inner(tools, history, ctx)

    res = bench.run(model=model, n_tools=48, n_results=6)
    assert res["stopped"]["family"] == "R" and res["verdicts"]["R"]["status"] == "no-verdict"
    assert res["verdicts"]["L0"]["status"] != "no-verdict"


def test_stop_at_failure_skips_later_levels():
    model = bench.scripted("degraded", degrade={"L0": 0.5})
    res = bench.run(model=model, n_tools=48, n_results=6, stop_at_failure=True)
    assert res["verdicts"]["L0"]["status"] == "failed"
    assert {res["verdicts"][lv]["status"] for lv in ("L1", "L2", "L3")} == {"not-tested"}
    assert set(res["summary"]["arms"]) == {"raw", "L0"}


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
        workers=2,
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
    assert "expected cost" in capsys.readouterr().err

    class Unmetered:
        def __init__(self, model, budget):
            self.meter = None

    monkeypatch.setattr(bench, "AnthropicModel", Unmetered)
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="claude-haiku-4-5", budget_usd=85)) == 1
    assert "no spend meter" in capsys.readouterr().err
    assert not list(tmp_path.glob("live_*"))


def test_cli_live_run_with_a_mock_api(tmp_path, monkeypatch, capsys):
    opener, _ = _fake_api({"input_tokens": 50, "output_tokens": 5})
    real = bench.AnthropicModel
    monkeypatch.setattr(
        bench, "AnthropicModel", lambda m, b: real(m, b, api_key="k", opener=opener)
    )
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="claude-haiku-4-5", budget_usd=85)) == 0
    res = json.loads((tmp_path / "live_claude-haiku-4-5.json").read_text())
    calls = (tmp_path / "live_calls_claude-haiku-4-5.jsonl").read_text().splitlines()
    assert res["mode"] == "live" and res["stopped"] is None and "_runs" not in res
    assert res["api_attempts"] == len(calls) > 0 and res["preflight_input_tokens"]["raw"] == 1234
    assert res["spent_usd"] == pytest.approx(sum(json.loads(c)["usd"] for c in calls), abs=1e-3)
    by_arm = res["spend_by_arm"]
    assert {"tools:raw", "tools:L0", "results:raw", "results:R"} <= set(by_arm)
    assert sum(r["attempts"] for r in by_arm.values()) == len(calls)
    # the ceiling: a run it stops still writes its (content-free) spend record
    monkeypatch.setattr(
        bench, "estimate_cost", lambda *a: {"models": {"claude-haiku-4-5": {"usd_cached": 0}}}
    )
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="claude-haiku-4-5", budget_usd=0.06)) == 1
    res = json.loads((tmp_path / "live_claude-haiku-4-5.json").read_text())
    assert res["stopped"] and 0 < res["spent_usd"] <= 0.06
    assert "spend ceiling" in capsys.readouterr().err


def test_cli_live_abort_and_missing_key(tmp_path, monkeypatch, capsys):
    def opener(req, timeout):
        raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO())

    real = bench.AnthropicModel
    monkeypatch.setattr(
        bench, "AnthropicModel", lambda m, b: real(m, b, api_key="k", opener=opener)
    )
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="claude-haiku-4-5", budget_usd=85)) == 1
    assert "aborted" in json.loads((tmp_path / "live_claude-haiku-4-5.json").read_text())

    def nokey(*a):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(bench, "AnthropicModel", nokey)
    assert cli.cmd_mcp(_ns(tmp_path, live=True, model="claude-haiku-4-5", budget_usd=85)) == 1
