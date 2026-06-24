"""Tests for the real-trace adapters + the prove.py proof harness.

These lock in the de-circularization work: τ-/SWE-bench traces load into the
trajectory model with NO planted DECISION markers, the smoke runner distinguishes
recoverable (safe) from irrecoverable (unsafe) compression, and the
frontier/coverage machinery produces a sound out-of-sample certificate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from distil.compress.adaptive import byte_exact
from distil.conformal import default_ladder
from distil.replay import realtrace
from distil.replay.smoke_runner import SmokeRunner
from distil.trajectory import Stability

FIX = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures"


def _load_prove():
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "prove.py"
    spec = importlib.util.spec_from_file_location("prove", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #


def test_tau_adapter_loads_no_markers():
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    assert len(entries) >= 6
    for e in entries:
        assert e.domain == "tau-bench"
        assert e.trajectory.turns
        # no planted DECISION: oracle anywhere — that is the whole point
        for t in e.trajectory.turns:
            assert all("DECISION:" not in b.text for b in t.blocks)
            # cacheable prefix invariant: volatile blocks come last
            kinds = [b.stability is Stability.VOLATILE for b in t.blocks]
            assert kinds == sorted(kinds, key=lambda v: v)  # all False then all True
            assert any(b.stability is Stability.VOLATILE for b in t.blocks)


def test_swe_adapter_loads_with_resolution():
    entries = realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    assert len(entries) >= 4
    assert all(e.domain == "swe-bench" for e in entries)
    # resolution status is carried for the downstream task-success metric
    statuses = [realtrace.resolved_status(e) for e in entries]
    assert any(s is True for s in statuses) and any(s is False for s in statuses)


def test_gold_actions_present_and_canonical():
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    gold = realtrace.gold_actions(entries)
    assert gold
    for g in gold.values():
        # fingerprint is the same {action,target} JSON the live runner emits
        assert g.fingerprint.startswith('{"action":')


def test_structural_validation_clean():
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    entries += realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    assert realtrace.validate_real(entries) == []


# --------------------------------------------------------------------------- #
# smoke runner — recoverable vs irrecoverable
# --------------------------------------------------------------------------- #


def test_smoke_runner_is_not_marker_based():
    # the smoke runner must NOT be the circular DECISION-marker oracle
    from distil.replay.runner import DeterministicRunner

    assert SmokeRunner().name != DeterministicRunner().name
    assert getattr(SmokeRunner(), "evidential", True) is False


def test_byte_exact_preserves_decision_truncation_can_flip():
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    runner = SmokeRunner()

    def truncate120(blocks, turn):
        return [
            b.copy_with(b.text[:120]) if b.stability is Stability.VOLATILE else b for b in blocks
        ]

    flips_byte = flips_trunc = record_turns = 0
    for e in entries:
        for t in e.trajectory.turns:
            base = runner.decide(t.blocks)
            if base == "<no-record>":
                continue
            record_turns += 1
            flips_byte += int(runner.decide(byte_exact(t.blocks, t.index)) != base)
            flips_trunc += int(runner.decide(truncate120(t.blocks, t.index)) != base)
    assert record_turns > 0
    assert flips_byte == 0  # byte-exact never changes a decision
    assert flips_trunc > 0  # aggressive truncation drops the load-bearing record


# --------------------------------------------------------------------------- #
# harness end-to-end
# --------------------------------------------------------------------------- #


def _matrix():
    prove = _load_prove()
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    entries += realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    gold = realtrace.gold_actions(entries)
    ladder = default_ladder()

    class _Cache:
        def __init__(self):
            self.r = SmokeRunner()

        def decide(self, blocks, restore=None):
            return self.r.decide(blocks)

        def prefetch(self, requests, workers=1):
            pass

    return prove, prove.build_matrix(entries, _Cache(), ladder, gold), ladder


def test_frontier_lossless_safe_aggressive_unsafe():
    prove, matrix, ladder = _matrix()
    rows = {r["level"]: r for r in prove.e1_frontier(matrix, ladder)}
    # the reversible digest saves tokens at ZERO decision change...
    assert rows["lossless"]["savings"] > 0.05
    assert rows["lossless"]["decision_change"] == 0.0
    assert rows["byte-exact"]["decision_change"] == 0.0
    # ...while blind aggressive truncation changes decisions
    assert rows["truncate@120"]["decision_change"] > 0.0
    assert rows["truncate@120"]["savings"] > rows["lossless"]["savings"]


def test_certificate_holds_out_of_sample():
    prove, matrix, ladder = _matrix()
    # at an α the sample can support, the held-out coverage must meet the 1-δ target
    cov = prove.e2_coverage(matrix, ladder, alpha=0.2, delta=0.05, method="ltt", reps=200, seed=0)
    assert cov["certified_frac"] > 0.5
    assert cov["empirical_coverage"] >= 0.95  # the guarantee, validated out-of-sample
    assert cov["mean_realized_risk"] <= 0.2
    assert cov["mean_test_savings"] > 0.05


def test_tiny_sample_refuses_tight_alpha():
    # honesty property: too few calibration turns ⇒ refuse to certify a tight α
    prove, matrix, ladder = _matrix()
    cov = prove.e2_coverage(matrix, ladder, alpha=0.01, delta=0.05, method="ltt", reps=50, seed=0)
    assert cov["certified_frac"] == 0.0


def test_task_success_metric_safe_levels_hold_baseline():
    prove, matrix, ladder = _matrix()
    e4 = prove.e4_task_success(matrix, ladder, seed=0, boot=200)
    assert e4 is not None and e4["n"] >= 6
    rows = {r["level"]: r for r in e4["levels"]}
    # safe levels keep the full baseline success-rate; aggressive truncation erodes it
    assert rows["lossless"]["retained_success"] == e4["baseline_success"]
    assert rows["byte-exact"]["retained_success"] == e4["baseline_success"]
    assert rows["truncate@120"]["retained_success"] < e4["baseline_success"]
    assert rows["lossless"]["savings"] > 0.05


# --------------------------------------------------------------------------- #
# runners — fingerprint rendering / parsing (no network)
# --------------------------------------------------------------------------- #


def test_parse_fingerprint_handles_fences_and_prose():
    from distil.replay.prompts import parse_fingerprint

    canon = '{"action":"refund","target":"a1"}'  # action normalized, target case-folded
    assert parse_fingerprint('{"action": "refund", "target": "A1"}') == canon
    assert parse_fingerprint('```json\n{"action":"Refund","target":"A1"}\n```') == canon
    assert parse_fingerprint('Sure! {"target":"A1","action":"refund"} done.') == canon
    assert parse_fingerprint("no json here") == "<no-decision>"
    # paraphrase of the same tool is NOT a decision change
    assert parse_fingerprint('{"action":"search_flights","target":"x"}') == parse_fingerprint(
        '{"action":"SearchFlights","target":"x"}'
    )


def test_render_keeps_system_prefix_separate():
    from distil.replay.prompts import render

    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    blocks = entries[0].trajectory.turns[-1].blocks
    system, user = render(blocks)
    assert "customer-support agent" in system  # stable system prompt
    assert "[tool_output]" in user or "[user]" in user  # observations go to the user turn


def test_claude_cli_result_extraction():
    from distil.replay.claude_cli_runner import ClaudeCliRunner

    envelope = (
        '{"type":"result","result":"```json\\n{\\"action\\":\\"x\\",\\"target\\":\\"y\\"}\\n```"}'
    )
    assert ClaudeCliRunner._result_text(envelope) == '```json\n{"action":"x","target":"y"}\n```'
    # non-envelope stdout falls through unchanged
    assert ClaudeCliRunner._result_text("plain text") == "plain text"


def test_openai_runner_parses_mocked_response(monkeypatch):
    from distil.replay.openai_runner import OpenAIRunner

    class _Resp:
        def __init__(self, payload):
            self._b = payload.encode()

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    payload = '{"choices":[{"message":{"content":"{\\"action\\":\\"edit\\",\\"target\\":\\"src/x.py\\"}"}}]}'
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(payload))
    runner = OpenAIRunner("local-model", base_url="http://x/v1")
    entries = realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    blocks = entries[0].trajectory.turns[0].blocks
    assert runner.decide(blocks) == '{"action":"edit","target":"src/x.py"}'


# --------------------------------------------------------------------------- #
# fetch_real converters
# --------------------------------------------------------------------------- #


def _load_fetch():
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "fetch_real.py"
    spec = importlib.util.spec_from_file_location("fetch_real", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tau_result_normalization():
    fr = _load_fetch()
    rec = {
        "task_id": "retail_7",
        "success": True,
        "traj": [
            {"role": "system", "content": "support agent"},
            {"role": "user", "content": "refund A1"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "credit", "arguments": {"id": "A1"}}}],
            },
        ],
    }
    ep = fr.normalize_tau_episode(rec, 0)
    assert ep["id"] == "retail_7" and ep["reward"] == 1 and len(ep["messages"]) == 3
    assert fr.normalize_tau_episode({"id": "x"}, 1) is None  # no messages → dropped


def test_swe_patch_target_parsing():
    fr = _load_fetch()
    patch = (
        "--- a/src/foo.py\n+++ b/src/foo.py\n@@\n-x\n+y\n--- a/t/bar.py\n+++ b/t/bar.py\n@@\n-a\n+b"
    )
    assert fr.parse_patch_targets(patch) == ["src/foo.py", "t/bar.py"]
    assert fr.parse_patch_targets("") == []


def test_swe_localization_episode_round_trips(tmp_path):
    import json

    fr = _load_fetch()
    patch = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,2 +1,2 @@\n-bad\n+good"
    inst = {"instance_id": "p__b1", "problem_statement": "foo is wrong", "patch": patch}
    ep = fr.build_swe_localization_episode(inst, ["src/zzz.py", "src/qqq.py"])
    p = tmp_path / "swe.json"
    p.write_text(json.dumps([ep]))
    entries = realtrace.load_swe_bench(p)
    assert len(entries) == 1
    gold = realtrace.gold_actions(entries)
    edit = [g for g in gold.values() if g.action == "edit"][0]
    assert edit.target == "src/foo.py"  # ground truth = the file the gold patch edits
    assert realtrace.validate_real(entries) == []


# --------------------------------------------------------------------------- #
# shuffled-position corpus variant (removes the recency/tail-truncation confound)
# --------------------------------------------------------------------------- #


def _search_hits(ep):
    """The list of 'FILE …' result blocks in the edit-step observation, in order."""
    obs = ep["trajectory"][1]["observation"]
    return obs.split("->\n", 1)[1].split("\n\n")


def _gold_index(ep):
    """Index of the gold hit (the one carrying a real hunk, not the distractor marker)
    among the search hits, plus the total number of hits."""
    hits = _search_hits(ep)
    for i, h in enumerate(hits):
        lines = h.split("\n")
        if len(lines) < 2 or lines[1].strip() != "(no obvious relation to the issue)":
            return i, len(hits)
    raise AssertionError("no gold hit found")


def _synthetic_instance(n):
    patch = f"--- a/src/mod{n}.py\n+++ b/src/mod{n}.py\n@@ -1,2 +1,2 @@\n-bad{n}\n+good{n}"
    return {
        "instance_id": f"proj__bug-{n}",
        "problem_statement": f"issue {n}",
        "patch": patch,
    }


_DISTRACTORS = [f"src/d{i}.py" for i in range(6)]  # 6 distractors → 7 hits total


def test_swe_localization_gold_last_by_default():
    fr = _load_fetch()
    ep = fr.build_swe_localization_episode(_synthetic_instance(0), _DISTRACTORS)
    idx, n = _gold_index(ep)
    assert n == 7 and idx == 6  # original construction: gold appended LAST


def test_swe_localization_shuffle_is_deterministic():
    fr = _load_fetch()
    inst = _synthetic_instance(0)
    a = fr.build_swe_localization_episode(inst, _DISTRACTORS, gold_seed=1729)
    b = fr.build_swe_localization_episode(inst, _DISTRACTORS, gold_seed=1729)
    assert a == b  # same seed + instance → byte-identical
    c = fr.build_swe_localization_episode(inst, _DISTRACTORS, gold_seed=42)
    # a different seed should generally move the needle (not a hard guarantee per
    # instance, but the constructions must not be the same object/text by accident)
    assert isinstance(c, dict)


def test_swe_localization_shuffle_preserves_content():
    """Shuffled corpus is identical to the gold-last build except the gold hit's
    position: same hit set, same gold target, same problem statement, same action."""
    fr = _load_fetch()
    for n in range(20):
        inst = _synthetic_instance(n)
        base = fr.build_swe_localization_episode(inst, _DISTRACTORS)
        shuf = fr.build_swe_localization_episode(inst, _DISTRACTORS, gold_seed=1729)
        assert set(_search_hits(shuf)) == set(_search_hits(base))  # same content
        assert shuf["trajectory"][1]["action"] == base["trajectory"][1]["action"]
        assert shuf["problem_statement"] == base["problem_statement"]
        assert shuf["system"] == base["system"] and shuf["info"] == base["info"]
        assert shuf["trajectory"][0] == base["trajectory"][0]  # search step untouched


def test_swe_localization_shuffle_position_roughly_uniform():
    """Across many instances the gold hit's position is spread over all slots, not
    biased toward last — the whole point of the variant. Deterministic (fixed seed)."""
    import collections

    fr = _load_fetch()
    positions = []
    for n in range(700):
        ep = fr.build_swe_localization_episode(_synthetic_instance(n), _DISTRACTORS, gold_seed=1729)
        idx, total = _gold_index(ep)
        assert total == 7
        positions.append(idx)
    counts = collections.Counter(positions)
    assert set(counts) == set(range(7))  # every slot used, including 0 (first)
    # last slot is not the mode; distribution is broadly flat (~100 expected per slot)
    assert counts[6] < len(positions) * 0.25
    assert min(counts.values()) > len(positions) * 0.07
    assert abs(sum(positions) / len(positions) - 3.0) < 0.6  # uniform mean over 0..6


# --------------------------------------------------------------------------- #
# expand-aware grading (the with-expand frontier)
# --------------------------------------------------------------------------- #


def test_build_restore_matches_digest_handles():
    from distil.compress.strategies import distil
    from distil.compress.tier1 import _handle
    from distil.replay.expand_runner import build_restore
    from distil.trajectory import Block, Kind, Stability

    # a verbose tool output whose load-bearing record sits in the MIDDLE → folded
    lines = [f"log line {i} status=ok lat={i}" for i in range(6)]
    lines.insert(3, "RECORD status=delivered condition=opened amount=88.50 days_since=11")
    obs = Block("obs", Kind.TOOL_OUTPUT, "\n".join(lines), Stability.VOLATILE)
    blocks = [Block("sys", Kind.SYSTEM, "policy", Stability.STABLE), obs]
    restore = build_restore(blocks)
    assert _handle(obs.text) in restore  # handle is content-addressed → reproducible
    compressed = distil(blocks, 0)
    folded = [b for b in compressed if "handle=" in b.text]
    assert folded, "the digest should have folded the verbose middle behind a handle"


def test_expand_loop_recovers_a_folded_decision():
    from distil.compress.strategies import distil
    from distil.replay import prompts
    from distil.replay.expand_runner import ExpandAwareRunner, build_restore
    from distil.replay.smoke_runner import SmokeRunner
    from distil.trajectory import Block, Kind, Stability

    # record buried in the middle so the reversible digest folds it out of view
    noise = [f"log {i} kind=heartbeat status=ok shard={i}" for i in range(8)]
    noise.insert(4, "RECORD status=delivered condition=opened amount=42.00 days_since=20")
    obs = Block("obs", Kind.TOOL_OUTPUT, "\n".join(noise), Stability.VOLATILE)
    blocks = [Block("sys", Kind.SYSTEM, "support policy", Stability.STABLE), obs]

    smoke = SmokeRunner()
    er = ExpandAwareRunner(smoke)
    restore = build_restore(blocks)
    compressed = distil(blocks, 0)
    assert any("handle=" in b.text for b in compressed)  # record is folded

    base = er.decide(blocks, restore)  # uncompressed (no handles → direct decide)
    # no-expand: decide on the folded text without recovery
    s, u = prompts.decision_prompt(compressed)
    no_expand = prompts.parse_fingerprint(smoke._raw(s, u))
    # with-expand: the loop recovers the folded record, then decides
    with_expand = er.decide(compressed, restore)

    assert no_expand != base  # folding hid the load-bearing record → decision flips
    assert with_expand == base  # recovery restores byte-exact content → decision preserved


# --------------------------------------------------------------------------- #
# structured forced-tool grading
# --------------------------------------------------------------------------- #


def test_fingerprint_from_args_canonicalizes():
    from distil.replay.prompts import canonical, fingerprint_from_args

    assert fingerprint_from_args({"action": "Search_Flights", "target": "NYC SEA"}) == canonical(
        "search_flights", "nyc sea"
    )
    assert fingerprint_from_args('{"action":"edit","target":"a.py"}') == canonical("edit", "a.py")
    assert fingerprint_from_args({"nope": 1}) == "<no-decision>"


def test_openai_forced_tool_path(monkeypatch):
    from distil.replay.openai_runner import OpenAIRunner
    from distil.replay.prompts import canonical

    runner = OpenAIRunner("local", base_url="http://x/v1")
    # server returns a forced function call with structured args (no prose)
    tool_body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "record_decision",
                                "arguments": '{"action":"Issue_Refund","target":"A1"}',
                            }
                        }
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(OpenAIRunner, "_post", lambda self, payload: tool_body)
    entries = realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    blocks = entries[0].trajectory.turns[0].blocks
    assert runner.decide(blocks) == canonical("issue_refund", "A1")


def test_openai_tool_fallback_to_text(monkeypatch):
    import urllib.error

    from distil.replay.openai_runner import OpenAIRunner
    from distil.replay.prompts import canonical

    runner = OpenAIRunner("local", base_url="http://x/v1")
    text_body = {"choices": [{"message": {"content": '{"action":"edit","target":"f.py"}'}}]}

    def fake_post(self, payload):
        if "tools" in payload:  # server can't do tools
            raise urllib.error.URLError("no tools")
        return text_body

    monkeypatch.setattr(OpenAIRunner, "_post", fake_post)
    entries = realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    blocks = entries[0].trajectory.turns[0].blocks
    assert runner.decide(blocks) == canonical("edit", "f.py")  # fell back to free-text


def test_anthropic_runner_returns_canonical():
    from types import SimpleNamespace

    from distil.replay.anthropic_runner import AnthropicRunner
    from distil.replay.prompts import canonical

    class _FakeMessages:
        def create(self, **kw):
            blk = SimpleNamespace(
                type="tool_use", input={"action": "Search Flights", "target": "X"}
            )
            return SimpleNamespace(content=[blk])

    client = SimpleNamespace(messages=_FakeMessages())
    runner = AnthropicRunner(client=client)
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    blocks = entries[0].trajectory.turns[-1].blocks
    assert runner.decide(blocks) == canonical("searchflights", "x")  # normalized, not raw


# --------------------------------------------------------------------------- #
# competitor / structural baselines (head-to-head)
# --------------------------------------------------------------------------- #


def _load_baselines_mod():
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "baselines.py"
    spec = importlib.util.spec_from_file_location("baselines", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_baselines_load_and_compress():
    bl = _load_baselines_mod()
    rungs = dict(bl.load_baselines(include_real=False))
    assert {
        "truncate@500",
        "recency-window@500",
        "keep-last-3-turns",
        "recomp-extractive",
        "selective-context",
    } <= set(rungs)
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    blocks = entries[1].trajectory.turns[-1].blocks  # verbose volatile obs + history
    tot0 = sum(len(b.text) for b in blocks)

    # volatile-only baselines: shrink the tail, leave the cacheable prefix byte-identical
    for name in (
        "truncate@500",
        "recency-window@500",
        "recomp-extractive",
        "selective-context",
    ):
        out = rungs[name](blocks, 0)
        pre_in = [b.text for b in blocks if b.stability is not Stability.VOLATILE]
        pre_out = [b.text for b in out if b.stability is not Stability.VOLATILE]
        assert pre_in == pre_out
        assert sum(len(b.text) for b in out) <= tot0

    # keep-last-k-turns is the cache-breaking memory baseline: it rewrites HISTORY,
    # keeping only the most recent k entries (test on a synthetic long history)
    from distil.trajectory import (
        Block,
        Kind,
    )  # Stability already imported at module level

    hist = Block(
        "h",
        Kind.HISTORY,
        "\n\n".join(f"[msg{i}] turn {i}" for i in range(10)),
        Stability.SETTLING,
    )
    synth = [
        Block("s", Kind.SYSTEM, "policy", Stability.STABLE),
        hist,
        Block("o", Kind.TOOL_OUTPUT, "obs", Stability.VOLATILE),
    ]
    out = rungs["keep-last-3-turns"](synth, 0)
    hist_out = next(b.text for b in out if b.kind is Kind.HISTORY)
    assert hist_out.count("[msg") == 3  # only the last 3 entries survive
    assert "earlier turns dropped" in hist_out


def test_longllmlingua_threads_question_through(monkeypatch):
    """Regression: the LongLLMLingua adapter must pass a non-empty ``question=`` to
    ``compress_prompt`` — its rank_method="longllmlingua" path asserts without one, and
    the bare ``except`` would silently swallow that into a misleading no-op (0% savings)
    row. We inject a fake ``llmlingua`` so the real Llama-2 backbone never loads, then
    assert the standing (non-volatile) context reaches the call as the question. Against
    the pre-fix code (no ``question=``) ``recorded["question"]`` is unset → this fails.
    """
    import sys
    import types

    from distil.trajectory import Block, Kind, Stability

    recorded: dict = {}

    class _Recorder:
        def __init__(self, *a, **k):
            pass

        def compress_prompt(self, *a, **k):
            recorded.update(k)
            recorded["pos_args"] = a
            return {"compressed_prompt": "x"}  # strictly shorter → _map_volatile keeps it

    fake = types.ModuleType("llmlingua")
    fake.PromptCompressor = _Recorder
    monkeypatch.setitem(sys.modules, "llmlingua", fake)

    bl = _load_baselines_mod()
    strat = bl.longllmlingua(rate=0.5)
    assert strat is not None  # fake loads → factory returns a strategy

    blocks = [
        Block(
            "sys",
            Kind.SYSTEM,
            "ISSUE: the refund flow double-charges customers",
            Stability.STABLE,
        ),
        Block("obs", Kind.TOOL_OUTPUT, "log line " * 200, Stability.VOLATILE),
    ]
    out = strat(blocks, 0)

    # the question was actually threaded through (this is what the bug dropped)…
    assert recorded.get("question"), "question= was not passed to compress_prompt"
    # …and it is the standing context (the SYSTEM issue), not the volatile tail compressed
    assert "refund flow double-charges" in recorded["question"]
    assert "log line" not in recorded["question"]
    # and the volatile block was compressed using it
    assert any(b.text == "x" for b in out)


def test_strip_question_recovers_compressed_context():
    """Unit: LLMLingua assembles ``compressed_prompt`` = compressed-context + the
    verbatim question; we must splice the question back out so the block can shrink."""
    bl = _load_baselines_mod()
    q = "SYSTEM POLICY\nISSUE: the bug report, quite long"
    assert bl._strip_question("kept ctx\n\n" + q, q) == "kept ctx"  # question after
    assert bl._strip_question(q + "\n\nkept ctx", q) == "kept ctx"  # question before
    assert bl._strip_question("no question here", q) == "no question here"  # absent → as-is


def test_longllmlingua_strips_question_so_block_shrinks(monkeypatch):
    """Regression for the silent no-op: ``compress_prompt`` returns the compressed
    context with the (uncompressed) question appended. The OLD code returned that whole
    string as the new volatile block, which — being the long question plus context —
    was bigger than the original, so reject-if-bigger discarded it every time (0%
    savings that read as 'no compression'). After the fix the question is spliced out,
    the block actually shrinks, and ``_map_volatile`` keeps it."""
    import sys
    import types

    from distil.trajectory import Block, Kind, Stability

    long_question = "SYSTEM POLICY " * 50  # non-volatile context, deliberately long

    class _Fake:
        def __init__(self, *a, **k):
            pass

        def compress_prompt(self, *a, **k):
            # mimic real output: a SHORT compressed context + the verbatim question
            return {"compressed_prompt": "tiny ctx\n\n" + k["question"]}

    fake = types.ModuleType("llmlingua")
    fake.PromptCompressor = _Fake
    monkeypatch.setitem(sys.modules, "llmlingua", fake)

    bl = _load_baselines_mod()
    strat = bl.longllmlingua(rate=0.5)
    blocks = [
        Block("s", Kind.SYSTEM, long_question, Stability.STABLE),
        Block("o", Kind.TOOL_OUTPUT, "X" * 400, Stability.VOLATILE),  # 400-char tail
    ]
    out = strat(blocks, 0)
    vol = next(b.text for b in out if b.id == "o")
    assert vol == "tiny ctx"  # question spliced out, compressed context kept
    assert len(vol) < 400  # the block actually shrank (the no-op bug is gone)
    assert long_question.strip() not in vol  # the question did not leak into the block


def test_head_to_head_distil_certifies_aggressive_baseline_does_not():
    prove = _load_prove()
    bl = _load_baselines_mod()
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    entries += realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    gold = realtrace.gold_actions(entries)
    ladder = default_ladder()
    baselines = bl.load_baselines(include_real=False)

    class _Cache:
        def __init__(self):
            self.r = SmokeRunner()

        def decide(self, blocks, restore=None):
            return self.r.decide(blocks)

        def prefetch(self, requests, workers=1):
            pass

    matrix = prove.build_matrix(entries, _Cache(), ladder, gold, baselines=baselines)
    rows = {r["method"]: r for r in prove.head_to_head(matrix, ladder, alpha=0.2, delta=0.05)}
    assert "recomp-extractive" in rows and "lossless" in rows  # both graded together
    assert rows["lossless"]["certifies"] is True  # reversible digest holds the decision
    # an aggressive lossy baseline saves more but flips decisions → cannot certify
    assert rows["recomp-extractive"]["savings"] > rows["lossless"]["savings"]
    assert rows["recomp-extractive"]["certifies"] is False


# --------------------------------------------------------------------------- #
# report → LaTeX filler
# --------------------------------------------------------------------------- #


def _load_r2l():
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "report_to_latex.py"
    spec = importlib.util.spec_from_file_location("report_to_latex", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_to_latex_fragments():
    r2l = _load_r2l()
    rep = {
        "frontier": [
            {"level": "lossless", "n": 10, "decision_change": 0.0, "savings": 0.16},
            {"level": "truncate@120", "n": 10, "decision_change": 0.5, "savings": 0.30},
        ],
        "head_to_head": [
            {
                "method": "lossless",
                "kind": "distil",
                "savings": 0.16,
                "decision_change": 0.0,
                "certifies": True,
            },
            {
                "method": "llmlingua-2",
                "kind": "baseline",
                "savings": 0.40,
                "decision_change": 0.2,
                "certifies": False,
            },
        ],
        "coverage": {
            "alpha": 0.05,
            "delta": 0.05,
            "method": "ltt",
            "reps": 500,
            "certified_frac": 1.0,
            "empirical_coverage": 0.99,
            "target_coverage": 0.95,
            "mean_realized_risk": 0.01,
            "mean_test_savings": 0.16,
        },
        "task_success": {
            "n": 50,
            "baseline_success": 0.8,
            "levels": [
                {
                    "level": "lossless",
                    "savings": 0.16,
                    "retained_success": 0.8,
                    "ci_low": 0.7,
                    "ci_high": 0.9,
                }
            ],
        },
        "shift": [
            {
                "held_out_domain": "retail",
                "certified": "lossless",
                "realized_risk": 0.02,
                "savings": 0.15,
                "held_within_alpha": True,
            }
        ],
    }
    macros = r2l.macros(rep)
    assert "\\renewcommand{\\HLsavings}{16.0\\%}" in macros
    h2h = r2l.head_to_head(rep)
    assert "llmlingua-2" in h2h and "\\checkmark" in h2h and "$\\times$" in h2h
    assert "\\begin{tabular}" in h2h and "\\bottomrule" in h2h
    # underscores in method names must be LaTeX-escaped (\_)
    assert r2l._tex("truncate@500_x") == "truncate@500\\_x"
    for fn in (r2l.frontier, r2l.coverage, r2l.task_success, r2l.shift):
        out = fn(rep)
        assert out and "\\" in out  # produced LaTeX, didn't crash


def test_e5_macros_prefix_and_skips_absent_methods():
    r2l = _load_r2l()
    rep = {
        "head_to_head": [
            {
                "method": "recency-window@500",
                "kind": "baseline",
                "savings": 0.16,
                "decision_change": 0.085,
                "certifies": True,
            },
            {
                "method": "longllmlingua",
                "kind": "baseline",
                "savings": 0.057,
                "decision_change": 0.035,
                "certifies": True,
            },
            # llmlingua-2, recomp, lossless, truncate@120, byte-exact all ABSENT here
        ],
        "coverage": {"empirical_coverage": 0.993},
    }
    out = r2l.e5_macros(rep, prefix="Orig")
    # cited methods present in the report get prefixed DC + Sav macros
    assert "\\renewcommand{\\OrigRecencyDC}{8.5\\%}" in out
    assert "\\renewcommand{\\OrigRecencySav}{16.0\\%}" in out
    assert "\\renewcommand{\\OrigLongLLDC}{3.5\\%}" in out
    assert "\\renewcommand{\\OrigCoverage}{99.3\\%}" in out
    # methods absent from this report are silently skipped (no macro emitted)
    assert "LLMtwo" not in out and "TruncShort" not in out and "ByteExact" not in out
    # a different prefix renames every macro
    assert "\\ShufRecencyDC" in r2l.e5_macros(rep, prefix="Shuf")


# --- honest-denominator + guards (added with the real-run hardening) ---------- #


def test_frontier_reports_effective_and_trivial_fields():
    prove, matrix, ladder = _matrix()
    rows = {r["level"]: r for r in prove.e1_frontier(matrix, ladder)}
    for r in rows.values():
        # every row carries the honest denominator so no-op turns can't inflate equivalence
        assert "effective_n" in r and "trivial_frac" in r and "decision_change_effective" in r
        assert 0.0 <= r["trivial_frac"] <= 1.0
        assert r["effective_n"] <= r["n"]
    # byte-exact changes no text on any turn → 100% trivial, 0 effective turns
    assert rows["byte-exact"]["trivial_frac"] == 1.0
    assert rows["byte-exact"]["effective_n"] == 0


def test_e4_flags_nonevidential_degenerate_outcomes():
    prove, matrix, ladder = _matrix()
    # force a degenerate outcome label set (all True) — mirrors swe-hf resolved=True
    for rec in matrix.values():
        rec["success"] = True
    e4 = prove.e4_task_success(matrix, ladder, seed=0, boot=50)
    assert e4 is not None and e4["outcome_evidential"] is False


def test_report_to_latex_refuses_smoke_run(tmp_path):
    import json
    import subprocess
    import sys

    rep = tmp_path / "smoke.json"
    rep.write_text(json.dumps({"args": {"runner": "smoke", "samples": 1}, "frontier": []}))
    script = Path(__file__).resolve().parent.parent / "benchmarks" / "report_to_latex.py"
    out = subprocess.run([sys.executable, str(script), str(rep)], capture_output=True, text=True)
    assert out.returncode != 0
    assert "smoke" in (out.stderr + out.stdout).lower()
