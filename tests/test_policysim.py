"""benchmarks/policysim: hand-computed cache costs on synthetic trajectories, and the
invariants the simulator's conclusions rest on. Synthetic content only."""

from __future__ import annotations

import json
import random

import pytest

from benchmarks.policysim import calibrate as cal
from benchmarks.policysim import costmodel as cm
from benchmarks.policysim import policies as pol
from benchmarks.policysim.sim import allocate_hidden, simulate
from benchmarks.policysim.trajectory import Trajectory, blocks

# One regex piece per word, so token counts are hand-computable: "w w w" = 3 tokens.
TM = cm.TokenModel(scale=1.0, per_block=0.0)


def words(n: int, w: str = "w") -> str:
    return " ".join([w] * n)


def traj(results: list[int], overhead: int = 1000, gap: float = 10.0, **kw: object) -> Trajectory:
    """user(task) -> [assistant tool_use, user tool_result] * len(results) -> assistant."""
    msgs: list[dict[str, object]] = [{"role": "user", "content": words(100, "task")}]
    for i, n in enumerate(results):
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {}}],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": words(n)}],
            }
        )
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
    ends = [i + 1 for i, m in enumerate(msgs) if m["role"] == "user"]
    t = Trajectory(
        id="syn",
        source="synthetic",
        arm="plain",
        model="claude-sonnet-5-5",
        messages=msgs,  # type: ignore[arg-type]
        request_ends=ends,
        times=[k * gap for k in range(len(ends))],
        overhead_tokens=overhead,
        **kw,  # type: ignore[arg-type]
    )
    return t


def _tool_use_tokens(i: int) -> float:
    return TM.count("bash" + json.dumps({}))


def test_anthropic_plain_hand_computed() -> None:
    t = traj([200, 300])
    r = simulate(t, pol.Policy(), cm.Anthropic(framing=0.0), TM)
    p1 = 1000 + 100
    p2 = p1 + _tool_use_tokens(0) + 200
    p3 = p2 + _tool_use_tokens(1) + 300
    assert r.usage.write_5m == pytest.approx(p3)  # P1 + (P2-P1) + (P3-P2)
    assert r.usage.read == pytest.approx(p1 + p2)
    assert r.usage.input == 0
    # $ at claude-sonnet-5-5: $2 in, read 0.1x, write 1.25x
    assert r.usd == pytest.approx((p3 * 1.25 + (p1 + p2) * 0.1) * 2 / 1e6)


def test_plain_reproduces_billed_usage_on_fixture() -> None:
    """Billed usage for this fixture follows from the documented rules alone; the plain
    policy must reproduce it exactly (the calibration identity, without the tokenizer)."""
    sizes = [50, 400, 80, 1200, 30]
    t = traj(sizes, overhead=1500)
    prefixes, p = [], 1500 + 100.0
    prefixes.append(p)
    for i, n in enumerate(sizes):
        p += _tool_use_tokens(i) + n
        prefixes.append(p)
    billed = {"cache_write": prefixes[-1], "cache_read": sum(prefixes[:-1]), "input": 2 * 6}
    r = simulate(t, pol.Policy(), cm.Anthropic(framing=2.0), TM)
    assert r.usage.write_5m == pytest.approx(billed["cache_write"])
    assert r.usage.read == pytest.approx(billed["cache_read"])
    assert r.usage.input == pytest.approx(billed["input"])


def test_below_minimum_is_not_cached() -> None:
    t = traj([10], overhead=100)  # ~212 tokens < 512 (Sonnet 5.5 minimum)
    r = simulate(t, pol.Policy(), cm.Anthropic(framing=0.0), TM)
    assert r.usage.read == r.usage.write_5m == 0
    assert r.usage.input > 0


def test_ttl_expiry_forces_a_full_rewrite() -> None:
    warm = simulate(traj([200, 200], gap=200.0), pol.Policy(), cm.Anthropic(framing=0.0), TM)
    cold = simulate(traj([200, 200], gap=400.0), pol.Policy(), cm.Anthropic(framing=0.0), TM)
    assert warm.usage.read > 0
    assert cold.usage.read == 0
    assert cold.usage.write_5m > warm.usage.write_5m


def test_one_hour_ttl_survives_a_long_gap() -> None:
    t = traj([200, 200], gap=1000.0, ttl="1h")
    r = simulate(t, pol.Policy(), cm.Anthropic(framing=0.0), TM)
    assert r.usage.read > 0 and r.usage.write_5m == 0 and r.usage.write_1h > 0


def test_rewrite_beyond_the_20_block_lookback_misses_entirely() -> None:
    prov = cm.Anthropic(framing=0.0)
    base = [cm.Seg(f"k{i}", 100.0, None) for i in range(30)]
    prov.request(base, 0.0, "5m")
    near = [*base[:29], cm.Seg("changed", 100.0, None), cm.Seg("new", 100.0, None)]
    u = prov.request(near, 1.0, "5m")  # old breakpoint (#29) gone; #28 is not an entry
    assert u.read == 0
    prov.reset()
    prov.request(base, 0.0, "5m")
    grow = [*base, *[cm.Seg(f"n{i}", 10.0, None) for i in range(25)]]
    u = prov.request(grow, 1.0, "5m")  # entry is 26 positions back: outside the window
    assert u.read == 0
    prov.reset()
    prov.request(base, 0.0, "5m")
    run = [*base, *[cm.Seg(f"r{i}", 10.0, "tool_result") for i in range(25)]]
    u = prov.request(run, 1.0, "5m")  # a tool_result run is ONE position
    assert u.read == 3000


def test_openai_rounds_to_128_and_gemini_needs_4096() -> None:
    segs = [cm.Seg("a", 1000.0, None), cm.Seg("b", 300.0, None)]
    o = cm.provider("openai-5.4")
    o.request(segs, 0.0, "5m")
    u = o.request([*segs, cm.Seg("c", 50.0, None)], 1.0, "5m")
    assert u.read == 1280 and u.write_5m == 0  # 1300 floored to 128s; no write charge
    o6 = cm.provider("openai-5.6")
    o6.request(segs, 0.0, "5m")
    u = o6.request([*segs, cm.Seg("c", 50.0, None)], 1.0, "5m")
    assert u.read == 1300 and u.write_5m == 50  # exact boundary; remainder written
    g = cm.provider("gemini-implicit")
    g.request(segs, 0.0, "5m")
    assert g.request([*segs, cm.Seg("c", 50.0, None)], 1.0, "5m").read == 0


def test_prefix_cache_prorates_a_partly_shared_block() -> None:
    o = cm.provider("openai-5.6")
    o.request([cm.Seg("a", 2000.0, None), cm.Seg("x", 100.0, None, "abcdefghij")], 0.0, "5m")
    u = o.request([cm.Seg("a", 2000.0, None), cm.Seg("y", 100.0, None, "abcdeZZZZZ")], 1.0, "5m")
    assert u.read == pytest.approx(2050.0)


def test_gemini_explicit_charges_storage() -> None:
    g = cm.provider("gemini-explicit", every=1)
    segs = [cm.Seg("a", 10_000.0, None)]
    g.request(segs, 0.0, "5m")
    u = g.request([*segs, cm.Seg("b", 10.0, None)], 3600.0, "5m")
    assert u.read == 10_000
    assert u.storage_usd == pytest.approx(10_000 * 4.50 / 1e6)


def _random_history(rng: random.Random, n: int) -> Trajectory:
    lines = ["PASSED tests/test_x.py::test_a", "ok", "Traceback (most recent call last):"]
    lines += ["ERROR: boom", "x = 1", '{"a": 1,   "b": [1, 2, 3]}', "", "same", "same", "same"]
    sizes = []
    msgs: list[dict[str, object]] = [{"role": "user", "content": "fix it"}]
    for i in range(n):
        cmd = rng.choice(["pytest -q", "git status", "cat f.py", "grep -rn x ."])
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {"command": cmd}}
                ],
            }
        )
        body = "\n".join(rng.choice(lines) for _ in range(rng.randint(1, 400)))
        sizes.append(len(body))
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": body}],
            }
        )
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
    ends = [i + 1 for i, m in enumerate(msgs) if m["role"] == "user"]
    return Trajectory(
        "r",
        "synthetic",
        "plain",
        "claude-sonnet-5-5",
        msgs,
        ends,
        [0.0] * len(ends),  # type: ignore[arg-type]
    )


ENTRY = [
    pol.EntryPolicy("entry-lossless", pol.tier0),
    pol.EntryPolicy("entry-digest", pol.entry_digest),
    pol.EntryPolicy("rtk-like", pol.rtk_like),
    pol.EntryPolicy("trunc-200", pol.truncate(200)),
]


@pytest.mark.parametrize("policy", ENTRY, ids=lambda p: p.name)
def test_entry_policies_never_change_an_already_sent_byte(policy: pol.Policy) -> None:
    for seed in range(5):
        t = _random_history(random.Random(seed), 8)
        policy.reset()
        prev: list[str] = []
        for end in t.request_ends:
            sent = [json.dumps(m, sort_keys=True) for m in policy.step(t.messages[:end], 0, 300)]
            assert sent[: len(prev)] == prev
            prev = sent
        # ...so under Anthropic rules every request is a pure extension of the last one:
        # the whole trajectory writes exactly the final prompt once, nothing twice.
        r = simulate(t, policy, cm.Anthropic(framing=0.0), TM)
        last = policy.step(t.messages[: t.request_ends[-1]], 0, 300)
        policy.reset()
        final = sum(s.tokens for s in cm.segments(last, TM, {}, 0))
        assert r.usage.write_5m == pytest.approx(final)


def test_entry_policies_keep_every_must_keep_line() -> None:
    t = _random_history(random.Random(7), 10)
    for p in ENTRY:
        r = simulate(t, p, cm.Anthropic(framing=0.0), TM)
        assert r.violations == 0, p.name


def test_history_rewrites_cost_cache_writes() -> None:
    """The mechanism under test: rewriting already-cached history re-pays the write."""
    t = _random_history(random.Random(3), 12)
    entry = simulate(t, pol.EntryPolicy("e", pol.tier0), cm.Anthropic(framing=0.0), TM)
    window = simulate(t, pol.Window(pol.tier0, 2, 2), cm.Anthropic(framing=0.0), TM)
    assert window.usage.write_5m > entry.usage.write_5m


def test_string_content_and_its_text_block_are_the_same_bytes() -> None:
    a = cm.segments([{"role": "user", "content": "hi"}], TM, {}, 0)
    b = cm.segments(
        [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {}}]}],
        TM,
        {},
        0,
    )
    assert [s.key for s in a] == [s.key for s in b]


def test_hidden_thinking_is_the_billed_output_minus_visible_text() -> None:
    t = traj([10])
    t.messages[1]["content"] = [  # type: ignore[index]
        {"type": "thinking", "thinking": "", "signature": "s" * 30},
        *blocks(t.messages[1]),
    ]
    t.messages[-1]["content"] = [  # type: ignore[index]
        {"type": "thinking", "thinking": "", "signature": "s" * 10},
        {"type": "text", "text": "done"},
    ]
    visible = _tool_use_tokens(0) + 1
    t.output_tokens = int(visible + 400)
    allocate_hidden(t, TM)
    assert sum(t.hidden.values()) == 400
    assert t.hidden["s" * 30] == 300


def test_fit_recovers_known_token_model() -> None:
    """Billed totals generated from a known (overhead, scale, per_block) are recovered."""
    true = cm.TokenModel(scale=1.4, per_block=20.0)
    trajs = []
    rng = random.Random(1)
    for i in range(30):
        t = traj([rng.randint(50, 3000) for _ in range(rng.randint(2, 8))], overhead=1600)
        t.id = f"s{i}"
        t.overhead_tokens = 1600
        r = simulate(t, pol.Policy(), cm.Anthropic(framing=2.0), true)
        t.billed = {
            "input": round(r.usage.input),
            "cache_write": round(r.usage.write_5m),
            "cache_read": round(r.usage.read),
            "output": 0,
        }
        trajs.append(t)
    tm, overhead, framing = cal.fit(trajs)
    assert tm.scale == pytest.approx(1.4, rel=0.01)
    assert tm.per_block == pytest.approx(20.0, rel=0.05)
    assert overhead == pytest.approx(1600, rel=0.01)
    assert framing == pytest.approx(2.0)
    rep = cal.evaluate(trajs, tm, framing)
    assert abs(rep["usd"]["aggregate_error"]) < 0.005


def test_policy_names_are_unique() -> None:
    names = [s.name for s in pol.build()]
    assert len(names) == len(set(names))
