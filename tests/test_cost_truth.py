"""Cost-truth benchmark harness: meter, cap, pairing, isolation, analysis math, dry run."""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

from benchmarks.cost_truth import analysis as an
from benchmarks.cost_truth import arms as arms_mod
from benchmarks.cost_truth import meter as m
from benchmarks.cost_truth import mock
from benchmarks.cost_truth import runner as rn
from benchmarks.cost_truth.__main__ import PROFILE, SAFETY, estimate

# --------------------------------------------------------------------------- pricing


def test_usage_cost_known_answer_with_1h_split() -> None:
    usage = {
        "input_tokens": 1000,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 4000,
        "cache_creation": {"ephemeral_5m_input_tokens": 3000, "ephemeral_1h_input_tokens": 1000},
        "output_tokens": 500,
    }
    # sonnet-5 $3/$15: 1000*3 + 2000*0.3 + 3000*3.75 + 1000*6 + 500*15, per Mtok
    assert m.usage_cost("claude-sonnet-5", usage) == pytest.approx(0.02835)


def test_usage_cost_without_split_bills_writes_at_5m_rate() -> None:
    usage = {"cache_creation_input_tokens": 1_000_000}
    assert m.usage_cost("claude-haiku-4-5", usage) == pytest.approx(1.25)


def test_unpriced_model_is_refused_not_billed_zero() -> None:
    with pytest.raises(m.UnpricedModel):
        m.usage_cost("gpt-nope", {"input_tokens": 1})


def test_worst_case_bounds_any_real_bill() -> None:
    body = 40_000  # bytes
    worst = m.worst_case_cost("claude-sonnet-5", body, 8192)
    # every byte a 1h write and every max_token emitted is the most a request can bill
    real = m.usage_cost(
        "claude-sonnet-5",
        {
            "cache_creation_input_tokens": body,
            "cache_creation": {"ephemeral_1h_input_tokens": body},
            "output_tokens": 8192,
        },
    )
    assert worst >= real


# --------------------------------------------------------------------------- spend cap


def test_spend_meter_reserve_settle_and_refuse() -> None:
    s = m.SpendMeter(1.0)
    r = s.reserve(0.6)
    with pytest.raises(m.BudgetExceeded):
        s.reserve(0.5)  # 0.6 reserved + 0.5 > 1.0
    s.settle(r, 0.1)
    assert s.spent == pytest.approx(0.1) and s.reserved == pytest.approx(0.0)
    s.reserve(0.5)  # fits now that the worst case was released
    assert s.exhausted


def test_spend_meter_never_exceeds_cap_under_concurrency() -> None:
    s = m.SpendMeter(10.0)

    def worker() -> None:
        for _ in range(200):
            try:
                r = s.reserve(0.05)
            except m.BudgetExceeded:
                return
            s.settle(r, 0.05)

    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert s.spent <= s.cap + 1e-9
    assert s.reserved == pytest.approx(0.0, abs=1e-9)


def test_spend_meter_rejects_nonpositive_cap() -> None:
    with pytest.raises(ValueError):
        m.SpendMeter(0)


# --------------------------------------------------------------------------- SSE usage


def test_sse_usage_survives_arbitrary_chunk_boundaries() -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "model": "claude-sonnet-5",
                "usage": {"input_tokens": 5, "cache_read_input_tokens": 900, "output_tokens": 1},
            },
        },
        {"type": "content_block_delta", "delta": {"text": "hello"}},
        {"type": "message_delta", "usage": {"output_tokens": 77}},
    ]
    raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    for step in (1, 7, len(raw)):
        p = m.SSEUsage()
        for i in range(0, len(raw), step):
            p.feed(raw[i : i + step])
        assert p.model == "claude-sonnet-5"
        assert p.usage == {"input_tokens": 5, "cache_read_input_tokens": 900, "output_tokens": 77}


# --------------------------------------------------------------------------- meter over HTTP


def _post(url: str, body: dict[str, Any]) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url + "/v1/messages",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": "sk-secret-do-not-log"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


_BODY = {
    "model": "claude-sonnet-5",
    "max_tokens": 1024,
    "system": "SECRET-SYSTEM",
    "messages": [{"role": "user", "content": "SECRET-CONTENT " * 50}],
}


@pytest.mark.parametrize("stream", [False, True])
def test_meter_bills_from_provider_usage_and_logs_no_content(tmp_path: Path, stream: bool) -> None:
    spend = m.SpendMeter(10.0)
    log = tmp_path / "meter.jsonl"
    with mock.MockUpstream() as up, m.UsageMeter(m.MeterConfig(up.url, log, spend, "r1")) as meter:
        status, body = _post(meter.base_url, {**_BODY, "stream": stream})
        assert status == 200
        assert (
            (b"message_start" in body)
            if stream
            else (json.loads(body)["content"][0]["text"] == "ok")
        )
    recs = m.read_log(log)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["usage"]["cache_creation_input_tokens"] > 0 and rec["usage"]["output_tokens"] > 1
    assert rec["cost_usd"] == pytest.approx(m.usage_cost("claude-sonnet-5", rec["usage"]), rel=1e-6)
    assert spend.spent == pytest.approx(
        rec["cost_usd"], rel=1e-6
    ) and spend.reserved == pytest.approx(0.0)
    text = log.read_text()
    for secret in ("SECRET", "sk-secret", "hello", '"ok"'):
        assert secret not in text


def test_meter_refuses_over_cap_before_reaching_provider(tmp_path: Path) -> None:
    spend = m.SpendMeter(0.001)  # below any request's worst case
    log = tmp_path / "meter.jsonl"
    with mock.MockUpstream() as up, m.UsageMeter(m.MeterConfig(up.url, log, spend, "r1")) as meter:
        status, _ = _post(meter.base_url, _BODY)
        assert status == 402
        assert up.requests == 0
    assert spend.spent == 0.0
    assert m.read_log(log)[0]["note"] == "budget_exceeded"


def test_meter_refuses_unpriced_model(tmp_path: Path) -> None:
    with (
        mock.MockUpstream() as up,
        m.UsageMeter(m.MeterConfig(up.url, tmp_path / "l.jsonl", m.SpendMeter(5), "r")) as meter,
    ):
        status, _ = _post(meter.base_url, {**_BODY, "model": "mystery-model"})
        assert status == 400 and up.requests == 0


def test_meter_upstream_error_bills_nothing_and_releases_reservation(tmp_path: Path) -> None:
    spend = m.SpendMeter(5.0)
    log = tmp_path / "l.jsonl"
    with mock.MockUpstream() as up, m.UsageMeter(m.MeterConfig(up.url, log, spend, "r")) as meter:
        up.fail_next = 1
        status, _ = _post(meter.base_url, _BODY)
        assert status == 529
    rec = m.read_log(log)[0]
    assert rec["status"] == 529 and rec["usage"] is None and rec["cost_usd"] == 0.0
    assert spend.spent == 0.0 and spend.reserved == pytest.approx(0.0)


def test_meter_unreachable_upstream_is_a_502_not_a_hang(tmp_path: Path) -> None:
    spend = m.SpendMeter(5.0)
    with m.UsageMeter(
        m.MeterConfig("http://127.0.0.1:9", tmp_path / "l.jsonl", spend, "r", timeout_s=2)
    ) as meter:
        status, _ = _post(meter.base_url, _BODY)
    assert status == 502 and spend.reserved == pytest.approx(0.0)


# --------------------------------------------------------------------------- pairing / randomisation


def test_schedule_pairs_every_block_with_every_arm_once() -> None:
    tasks, seeds = [f"t{i}" for i in range(9)], [0, 1, 2]
    s = rn.build_schedule(tasks, seeds, rn.ARMS, "claude-sonnet-5", 7)
    assert len(s) == len(tasks) * len(seeds) * len(rn.ARMS)
    blocks: dict[tuple[str, int], list[str]] = {}
    for r in s:
        blocks.setdefault((r.task, r.seed), []).append(r.arm)
    assert all(sorted(v) == sorted(rn.ARMS) for v in blocks.values())
    assert len({r.run_id for r in s}) == len(s)


def test_schedule_is_reproducible_and_randomised() -> None:
    a = rn.build_schedule(["x", "y", "z"], [0, 1], rn.ARMS, "m", 1)
    assert a == rn.build_schedule(["x", "y", "z"], [0, 1], rn.ARMS, "m", 1)
    assert [r.run_id for r in a] != [
        r.run_id for r in rn.build_schedule(["x", "y", "z"], [0, 1], rn.ARMS, "m", 2)
    ]


def test_arm_order_within_blocks_is_balanced() -> None:
    s = rn.build_schedule([f"t{i}" for i in range(400)], [0], rn.ARMS, "m", 3)
    first = [s[i].arm for i in range(0, len(s), len(rn.ARMS))]
    for arm in rn.ARMS:
        assert 0.18 < first.count(arm) / len(first) < 0.32


def test_next_eligible_enforces_cache_ttl_spacing() -> None:
    p = [rn.RunSpec("a", 0, "rtk", "m", 0), rn.RunSpec("b", 0, "rtk", "m", 1)]
    assert rn.next_eligible(p, {"a": 100.0}, 200.0, 360.0) == 1  # a is still inside the TTL
    assert rn.next_eligible(p, {"a": 100.0}, 460.0, 360.0) == 0
    assert rn.next_eligible(p, {"a": 100.0, "b": 150.0}, 200.0, 360.0) is None


# --------------------------------------------------------------------------- isolation


def test_isolated_env_drops_operator_state_and_points_inside_run_dir(tmp_path: Path) -> None:
    base = {
        "PATH": "/usr/bin",
        "HOME": "/Users/op",
        "ANTHROPIC_BASE_URL": "http://leak",
        "DISTIL_HOME": "/Users/op/.distil",
        "HEADROOM_MODE": "x",
        "ANTHROPIC_API_KEY": "k",
    }
    env = rn.isolated_env(tmp_path / "r1", base, {"ANTHROPIC_BASE_URL": "http://meter"})
    assert env["PATH"] == "/usr/bin" and env["ANTHROPIC_BASE_URL"] == "http://meter"
    assert "HEADROOM_MODE" not in env and "ANTHROPIC_API_KEY" not in env
    for k in ("HOME", "CLAUDE_CONFIG_DIR", "DISTIL_HOME", "XDG_CONFIG_HOME", "TMPDIR"):
        assert Path(env[k]).is_relative_to(tmp_path / "r1") and Path(env[k]).is_dir()
    other = rn.isolated_env(tmp_path / "r2", base, {})
    assert other["HOME"] != env["HOME"]


def test_isolated_env_refuses_a_reused_run_dir(tmp_path: Path) -> None:
    rn.isolated_env(tmp_path / "r", {}, {})
    with pytest.raises(RuntimeError, match="not fresh"):
        rn.isolated_env(tmp_path / "r", {}, {})


def test_summarise_run_classifies_budget_stop_and_broken_chain() -> None:
    spec = rn.RunSpec("t", 0, "rtk", "claude-sonnet-5", 0)
    assert rn.summarise_run(spec, "solved", [])["status"] == "arm_crash"
    assert (
        rn.summarise_run(spec, "failed", [{"status": 402, "usage": None}])["status"]
        == "budget_stop"
    )


# --------------------------------------------------------------------------- analysis math


def _run(
    task: str, seed: int, arm: str, cost: float, solved: bool, status: str | None = None
) -> dict[str, Any]:
    return {
        "task": task,
        "seed": seed,
        "arm": arm,
        "model": "claude-sonnet-5",
        "cost_usd": cost,
        "solved": solved,
        "status": status or ("solved" if solved else "failed"),
        "turns": 3,
        "first_request_cache_read": 0,
        "claim": {"tokens_saved": 10},
        "usage": {
            "input_tokens": 0,
            "cache_read_input_tokens": int(cost * 1e6),
            "cache_creation_input_tokens": 0,
            "output_tokens": 0,
        },
    }


def test_quantile_known_answers() -> None:
    v = [1.0, 2.0, 3.0, 4.0]
    assert an.quantile(v, 0.0) == 1.0 and an.quantile(v, 1.0) == 4.0 and an.quantile(v, 0.5) == 2.5
    assert math.isnan(an.quantile([], 0.5))
    assert an.quantile([1.0, math.inf], 1.0) == math.inf


def test_exact_half_cost_same_success_is_ratio_half_with_degenerate_ci() -> None:
    runs = []
    for t in ("a", "b", "c"):
        for s in (0, 1):
            runs += [_run(t, s, "control", 2.0, True), _run(t, s, "x", 1.0, True)]
    res = an.analyze(runs, ["control", "x"], b=500)
    c = res["comparisons"]["x"]
    assert c["usd_per_solved_ratio"] == pytest.approx(0.5)
    assert c["ratio_ci"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert c["success_diff"] == 0.0 and c["noninferior"] and c["tost_equivalent"]
    assert c["cost_verdict"] == "cheaper" and c["verdict"] == "saves"
    assert c["secondary"]["task_mean_log_ratio"] == pytest.approx(math.log(0.5))
    assert c["secondary"]["total_usd_ratio"] == pytest.approx(0.5)
    assert c["secondary"]["leave_one_task_out_ratio"] == [pytest.approx(0.5), pytest.approx(0.5)]


def test_usd_per_solved_hand_computed() -> None:
    runs = [
        _run("a", 0, "control", 1.0, True),
        _run("a", 1, "control", 1.0, False),
        _run("a", 0, "x", 1.0, True),
        _run("a", 1, "x", 2.0, True),
    ]
    res = an.analyze(runs, ["control", "x"], b=50)
    assert res["per_arm"]["control"]["usd_per_solved"] == pytest.approx(2.0)  # $2 / 1 solve
    assert res["per_arm"]["x"]["usd_per_solved"] == pytest.approx(1.5)  # $3 / 2 solves
    assert res["comparisons"]["x"]["usd_per_solved_ratio"] == pytest.approx(0.75)
    assert res["comparisons"]["x"]["success_diff"] == pytest.approx(0.5)


def test_cheap_arm_that_never_solves_is_not_a_saving() -> None:
    runs = []
    for t in ("a", "b", "c", "d"):
        runs += [_run(t, 0, "control", 1.0, True), _run(t, 0, "x", 0.01, False)]
    c = an.analyze(runs, ["control", "x"], b=200)["comparisons"]["x"]
    assert math.isinf(c["usd_per_solved_ratio"]) and c["cost_verdict"] == "dearer"
    assert not c["noninferior"] and c["verdict"] == "not shown"


def test_non_attributable_failure_drops_the_block_for_every_arm() -> None:
    runs = [
        _run("a", 0, "control", 1.0, True),
        _run("a", 0, "x", 1.0, True, status="infra_error"),
        _run("b", 0, "control", 1.0, True),
        _run("b", 0, "x", 0.5, True),
    ]
    res = an.analyze(runs, ["control", "x"], b=50)
    assert res["blocks"] == 1 and res["excluded_runs"] == 1
    assert res["per_arm"]["control"]["attempts"] == 1


def test_arm_crash_is_counted_against_the_arm() -> None:
    runs = [_run("a", 0, "control", 1.0, True), _run("a", 0, "x", 0.3, False, status="arm_crash")]
    res = an.analyze(runs, ["control", "x"], b=50)
    assert res["blocks"] == 1 and res["per_arm"]["x"]["cost_usd"] == pytest.approx(0.3)


def test_mcnemar_secondary_uses_loss_and_gain_counts() -> None:
    runs = []
    for i in range(20):
        runs += [_run(f"t{i}", 0, "control", 1.0, True), _run(f"t{i}", 0, "x", 1.0, i != 0)]
    mc = an.analyze(runs, ["control", "x"], b=50)["comparisons"]["x"]["secondary"][
        "mcnemar_cert_margin"
    ]
    assert mc["n"] == 20 and mc["delta"] == pytest.approx(-0.05)


def test_warm_start_sensitivity_reprices_first_request_reads() -> None:
    runs = [_run("a", 0, "control", 1.0, True), _run("a", 0, "x", 1.0, True)]
    runs[1]["first_request_cache_read"] = 1_000_000  # x inherited a warm prefix
    c = an.analyze(runs, ["control", "x"], b=20)["comparisons"]["x"]
    # 1M tokens moved from read ($0.30) to write ($3.75): x now costs $4.45 vs $1
    assert c["secondary"]["warm_start_ratio"] == pytest.approx(4.45)


# --------------------------------------------------------------------------- power / estimate


def test_power_formulas_invert_and_match_textbook() -> None:
    n = an.n_tasks_for_mde(0.9, 0.6, 0.15, 5)
    assert an.mde_ratio(n, 0.6, 0.15, 5) >= 0.9  # n tasks detect a 10% cut...
    assert an.mde_ratio(n - 1, 0.6, 0.15, 5) < 0.9  # ...and n-1 do not
    z = NormalDist().inv_cdf(1 - 0.05 / 3) + NormalDist().inv_cdf(0.8)
    assert an.ni_margin(445, 0.1) == pytest.approx(z * math.sqrt(0.1 / 445))


def test_estimate_is_consistent() -> None:
    est = estimate()
    tok = an.attempt_tokens(**PROFILE)
    for ph in est["phases"].values():
        per = an.price_tokens(ph["model"], tok)
        assert ph["hard_cap_usd"] == pytest.approx(per * ph["runs_total"] * SAFETY, abs=0.01)
        assert ph["usd_per_attempt_if_no_cache"] > ph["usd_per_attempt_cached"]
    assert est["total_hard_cap_usd"] == pytest.approx(
        sum(p["hard_cap_usd"] for p in est["phases"].values()), abs=0.02
    )


# --------------------------------------------------------------------------- arms


def test_arm_specs_are_pinned_verified_and_cite_evidence() -> None:
    assert set(arms_mod.ARMS) == set(rn.ARMS)
    assert arms_mod.unverified() == []
    assert arms_mod.ARMS["distil"].version == "1.54.0"  # the released wheel, not rc/dev
    assert all(a.evidence for a in arms_mod.ARMS.values())
    for a in rn.ARMS:  # every arm is metered
        assert "http://m" in json.dumps(arms_mod.arm_env(a, "http://m"))


# --------------------------------------------------------------------------- dry run end to end


def test_dry_run_end_to_end(tmp_path: Path) -> None:
    runs = rn.dry_run(tmp_path, ["t0", "t1"], [0], "claude-sonnet-5", cap_usd=5.0)
    assert len(runs) == 2 * len(rn.ARMS)
    assert all(r["status"] in ("solved", "failed") and r["cost_usd"] > 0 for r in runs)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["synthetic"] is True
    assert manifest["spent_usd"] == pytest.approx(sum(r["cost_usd"] for r in runs), abs=1e-5)
    assert not any((tmp_path / "runs").iterdir())  # per-run homes are gone
    # same-task runs never overlap the cache TTL on the virtual clock
    by_task: dict[str, list[float]] = {}
    for r in runs:
        by_task.setdefault(r["task"], []).append(r["virtual_start_s"])
    for starts in by_task.values():
        starts.sort()
        assert all(b - a >= 90.0 + rn.MIN_GAP_S for a, b in zip(starts, starts[1:]))
    logs = "".join(p.read_text() for p in (tmp_path / "meter").iterdir())
    assert "make the tests pass" not in logs and "xxxx" not in logs
    res = an.analyze(runs, list(rn.ARMS), b=100)
    assert set(res["comparisons"]) == {"rtk", "headroom", "distil"}


def test_dry_run_hard_cap_stops_the_schedule(tmp_path: Path) -> None:
    runs = rn.dry_run(tmp_path, ["t0", "t1", "t2"], [0, 1], "claude-sonnet-5", cap_usd=0.2)
    assert runs[-1]["status"] == "budget_stop"
    assert len(runs) < 3 * 2 * len(rn.ARMS)
    assert json.loads((tmp_path / "manifest.json").read_text())["spent_usd"] <= 0.2
