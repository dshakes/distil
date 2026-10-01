"""Offline checks of benchmarks/model_migration_eval.py — the harness that picked the
live certifier. Only its pure parts and its fake-client wiring: no API calls."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mme():
    spec = importlib.util.spec_from_file_location(
        "model_migration_eval", ROOT / "benchmarks" / "model_migration_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fp(action: str, target: str) -> str:
    from distil.replay.prompts import canonical

    return canonical(action, target)


def test_act_reads_the_action_or_passes_through(mme) -> None:
    assert mme._act(_fp("Search Flights", "SFO")) == "searchflights"
    assert mme._act(mme.NO) == mme.NO


def test_grade_exact_vs_action_equivalence(mme) -> None:
    a = _fp("refund", "order 1")
    arms = {
        "full_a": a,
        "full_b": _fp("refund", "order one"),  # paraphrased target: action-equal only
        "distil": a,
        "expand_structured": _fp("cancel", "order 1"),  # a real action flip
        "trunc": _fp("ask", "user"),
        "served": a,
        "served_expand": mme.NO,  # undecided never counts as a match
    }
    g = mme.grade({"arms": arms, "gold": _fp("refund", "x")}, ref_fp=a)
    assert g["equiv_distil"] == g["equiv_distil_act"] == 1
    assert (g["self_consist"], g["self_consist_act"]) == (0, 1)
    assert (g["equiv_expand"], g["equiv_expand_act"]) == (0, 0)
    assert g["equiv_served_act"] == 1 and g["equiv_served_expand_act"] == 0
    assert g["agree_gold_act"] == 1 and g["agree_ref"] == 1
    assert g["decided"] == 1 and g["trunc_detect"] == 1
    # baseline rows grade agreement against their own second draw
    assert mme.grade({"arms": arms}, None)["agree_ref"] == 0
    # nothing decided: every equivalence is 0, and it is counted as undecided
    none = mme.grade({"arms": {k: mme.NO for k in arms}}, None)
    assert none["decided"] == 0 and none["equiv_distil"] == 0 and none["trunc_detect"] == 0


def test_cost_usd(mme) -> None:
    u = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
    }
    # sonnet-5-5: $2 in, $10 out; cache write 1.25x, read 0.1x
    assert mme.cost_usd("claude-sonnet-5-5", u) == pytest.approx(2 + 1 + 2.5 + 0.2)
    # a dated serving id prices as its base model
    assert mme.cost_usd("claude-opus-4-8-20260101", u) == pytest.approx(5 + 2.5 + 6.25 + 0.5)
    with pytest.raises(KeyError):
        mme.cost_usd("gpt-4o", u)


def test_same_model(mme) -> None:
    assert mme._same_model("claude-sonnet-5-5", "claude-sonnet-5-5")
    assert mme._same_model("claude-sonnet-5-5-20260901", "claude-sonnet-5-5")
    assert mme._same_model("claude-sonnet-5-5@2026-09-01", "claude-sonnet-5-5")
    # a different model that merely shares a prefix is a serving substitution
    assert not mme._same_model("claude-sonnet-5-5-fast", "claude-sonnet-5-5")
    assert not mme._same_model("claude-sonnet-5", "claude-sonnet-5-5")


@pytest.fixture(scope="module")
def case(mme):
    return mme.load_cases()[0]


def test_run_case_oracle_wiring(mme, case) -> None:
    run = mme.run_case(case, mme.FakeClient("oracle", "claude-sonnet-5-5"), "claude-sonnet-5-5")
    assert set(run["arms"]) == {"full_a", "full_b", "distil", "expand_structured", "trunc"}
    g = mme.grade(run, None)
    assert g["equiv_distil"] == g["equiv_expand"] == g["self_consist"] == g["decided"] == 1
    assert run["n_calls"] >= 5 and run["usage"]["input_tokens"] == 1000 * run["n_calls"]
    assert run["model"] == "claude-sonnet-5-5"


def test_run_case_null_wiring(mme, case) -> None:
    run = mme.run_case(case, mme.FakeClient("null", "m"), "m")
    g = mme.grade(run, None)
    assert g["decided"] == 0 and g["equiv_distil"] == 0 and g["self_consist"] == 0


def test_run_case_flip_wiring(mme, case) -> None:
    random.seed(7)
    run = mme.run_case(case, mme.FakeClient("flip", "m"), "m")
    g = mme.grade(run, None)
    # a fresh random target per call: exact equivalence fails, the action still matches
    assert g["equiv_distil"] == 0 and g["equiv_distil_act"] == 1


def test_run_case_rejects_a_substituted_model(mme, case) -> None:
    # A classified error passes through AnthropicRunner._create unchanged, so main()
    # files it as serving_substitution rather than a generic API error.
    with pytest.raises(mme.ServingMismatch, match="served claude-haiku-4-5 != requested") as e:
        mme.run_case(case, mme.FakeClient("oracle", "claude-haiku-4-5"), "claude-sonnet-5-5")
    assert e.value.failure_class == "serving_substitution"


def test_serve_handles_are_restorable(mme) -> None:
    from distil.replay.expand_runner import _expand_blocks

    trajs = sorted(mme.SWE_DIR.glob("*.traj"))
    if not trajs:
        pytest.skip("SWE-agent cache absent — run model_migration_eval.py --fetch-swe-agent")
    turn, _, req = max(mme._swe_turns(trajs[0]), key=lambda x: len(x[2]["pairs"]))
    sblocks, restore = mme.serve(req)
    handles = sorted({h for b in sblocks for h in re.findall(r"handle=([0-9a-f]{8})", b.text)})
    assert handles, "a long coding turn should have digested tool output"
    assert set(handles) <= set(restore)
    back = _expand_blocks(sblocks, handles, restore)
    assert [b.text for b in back] == [b.text for b in turn.blocks]
    json.dumps(restore)  # plain str -> str, persistable with a result row


def test_fetch_verified_rejects_a_bad_download(mme, tmp_path, monkeypatch) -> None:
    import hashlib
    import urllib.request

    def fake(url, dest):
        Path(dest).write_bytes(b"corrupt")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake)
    dest = tmp_path / "x.json"
    with pytest.raises(SystemExit):
        mme._fetch_verified("u", dest, "0" * 64)
    assert not dest.exists() and not list(tmp_path.iterdir())
    mme._fetch_verified("u", dest, hashlib.sha256(b"corrupt").hexdigest())
    assert dest.read_bytes() == b"corrupt"
