"""--tier batch: Message Batches result mapping and tier-tagged cost. Fake client, no API."""

from __future__ import annotations

import importlib.util
from itertools import count
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from distil.replay.anthropic_runner import AnthropicRunner, BatchClient

ROOT = Path(__file__).resolve().parent.parent


class FakeBatches:
    """Answers each request with its own (shuffled) draw number; first poll is not ended."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.created: list[list[dict]] = []
        self.polls = 0
        self.draw = count()
        self.fail = fail or set()

    def create(self, requests):
        self.created.append(requests)
        return NS(id=f"b{len(self.created)}", processing_status="in_progress")

    def retrieve(self, bid):
        self.polls += 1
        return NS(id=bid, processing_status="ended")

    def results(self, bid):
        reqs = self.created[int(bid[1:]) - 1]
        for r in reversed(reqs):  # order must not matter: mapping is by custom_id
            if r["custom_id"] in self.fail:
                res = NS(type="errored", error=NS(error=NS(type="api_error", message="boom")))
            else:
                text = f"{r['params']['messages'][0]['content']}#{next(self.draw)}"
                res = NS(type="succeeded", message=NS(content=[NS(type="text", text=text)]))
            yield NS(custom_id=r["custom_id"], result=res)


def _batch_client(batches: FakeBatches, **kw) -> BatchClient:
    return BatchClient(NS(messages=NS(batches=batches)), poll_s=0, **kw)


def _ask(client, content):
    r = client.messages.create(model="m", messages=[{"role": "user", "content": content}])
    return "".join(b.text for b in r.content)


def test_results_map_by_custom_id_and_repeats_are_independent_draws() -> None:
    fb = FakeBatches()
    out = _batch_client(fb).run_all(
        {
            "a": lambda c: (_ask(c, "q1"), _ask(c, "q1")),  # full_a / full_b style repeat
            "b": lambda c: _ask(c, "q2"),
        }
    )
    assert len(fb.created) == 1 and len(fb.created[0]) == 3  # one batch, 3 distinct ids
    a1, a2 = out["a"]
    assert a1.startswith("q1#") and a2.startswith("q1#") and a1 != a2
    assert out["b"].startswith("q2#")
    assert fb.polls == 1


def test_dependent_round_takes_a_second_batch() -> None:
    fb = FakeBatches()
    out = _batch_client(fb).run_all({"t": lambda c: _ask(c, "round2:" + _ask(c, "round1"))})
    assert out["t"].startswith("round2:round1#")
    assert [len(b) for b in fb.created] == [2, 1]  # pass 1 also sends the placeholder-built round 2


def test_failed_request_surfaces_per_task_and_budget_is_enforced() -> None:
    probe = _batch_client(FakeBatches())
    probe.run_all({"x": lambda c: _ask(c, "q")})
    cid = next(iter(probe.results))
    out = _batch_client(FakeBatches(fail={cid})).run_all({"x": lambda c: _ask(c, "q")})
    assert "errored" in str(out["x"])

    with pytest.raises(SystemExit, match="budget"):
        _batch_client(FakeBatches(), max_requests=1).run_all(
            {"x": lambda c: (_ask(c, "q1"), _ask(c, "q2"))}
        )


def test_nondeterministic_requests_stop_at_the_pass_cap() -> None:
    fb, n = FakeBatches(), count()
    with pytest.raises(SystemExit, match="converge"):
        _batch_client(fb).run_all({"x": lambda c: _ask(c, f"q{next(n)}")})
    assert len(fb.created) == BatchClient.MAX_PASSES


def test_anthropic_runner_decides_through_the_batch_client() -> None:
    from distil.replay.prompts import canonical
    from distil.trajectory import Block, Kind, Stability

    class ToolBatches(FakeBatches):
        def results(self, bid):
            for r in self.created[int(bid[1:]) - 1]:
                msg = NS(content=[NS(type="tool_use", input={"action": "refund", "target": "o1"})])
                yield NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=msg))

    fb = ToolBatches()
    runner = AnthropicRunner(model="m", effort=None, samples=3)
    blocks = [Block("b0", Kind.USER, "refund order o1", Stability.VOLATILE, True)]
    out = _batch_client(fb).run_all(
        {"k": lambda c: setattr(runner, "_client", c) or runner.decide(blocks)}
    )
    assert out["k"] == canonical("refund", "o1")
    assert len(fb.created[0]) == 3  # three majority-vote samples, three distinct requests


def _bench(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefetch_keeps_paid_decisions_when_one_request_fails(tmp_path, monkeypatch) -> None:
    from distil.trajectory import Block, Kind, Stability

    class OneBad(FakeBatches):
        def results(self, bid):
            for r in self.created[int(bid[1:]) - 1]:
                if "bad" in r["params"]["messages"][0]["content"]:
                    res = NS(type="errored", error=NS(error=NS(type="api_error", message="boom")))
                else:
                    msg = NS(content=[NS(type="tool_use", input={"action": "a", "target": "t"})])
                    res = NS(type="succeeded", message=msg)
                yield NS(custom_id=r["custom_id"], result=res)

    prove = _bench("prove")
    monkeypatch.setattr(prove, "CACHE_DIR", tmp_path)
    runner = AnthropicRunner(model="m", effort=None, samples=1)
    cache = prove.DecisionCache(runner, "t", _batch_client(OneBad()))
    reqs = [
        ([Block("b0", Kind.USER, text, Stability.VOLATILE, True)], None) for text in ("ok", "bad")
    ]
    with pytest.raises(SystemExit, match="boom"):
        cache.prefetch(reqs)
    assert len(prove.DecisionCache(runner, "t").store) == 1  # the good decision reached disk
    assert runner._client is None  # the real client is restored


def test_cost_is_tagged_by_tier() -> None:
    mme = _bench("model_migration_eval")
    u = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 500_000}
    rt = mme.cost_usd("claude-sonnet-5-5", u)
    assert rt > 0
    assert mme.cost_usd("claude-sonnet-5-5", u, "batch") == pytest.approx(rt * 0.5)
    assert mme.cost_usd("claude-sonnet-5-5", u, "realtime") == rt
