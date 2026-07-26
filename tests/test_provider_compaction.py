"""Offline tests for the provider-compaction (Anthropic context-editing) A/B.

Everything runs against a scripted fake client — no API key, no network. The
live path is the same code with a real ``anthropic.Anthropic()``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distil.certify.provider_compaction import (
    CONTEXT_MGMT_BETA,
    Case,
    OpenAIArms,
    ProviderArms,
    edit_config,
    load_episodes,
    run_experiment,
    to_case,
)

FIXTURE = Path(__file__).parent.parent / "benchmarks" / "fixtures" / "tau_bench_sample.json"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """Mimics the SDK response surface the harness reads: model_dump()."""

    def __init__(self, body: dict) -> None:
        self._body = body

    def model_dump(self) -> dict:
        return self._body


def _tool_resp(name: str, inp: dict, applied_edits: list | None = None) -> dict:
    body: dict = {"content": [{"type": "tool_use", "id": "t1", "name": name, "input": inp}]}
    if applied_edits is not None:
        body["context_management"] = {"applied_edits": applied_edits}
    return body


_FIRED = [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 2, "cleared_input_tokens": 900}]


class _FakeClient:
    """Scripted beta.messages.create. ``script`` maps arm -> list of bodies
    consumed in order (cycled); arm is "edited" when context_management is sent.
    Records every request for assertions."""

    def __init__(self, script: dict[str, list[dict]], on_call=None) -> None:
        self._script = {k: list(v) for k, v in script.items()}
        self.requests: list[dict] = []
        self._on_call = on_call
        self.beta = self
        self.messages = self

    def create(self, **kw):
        if self._on_call:
            self._on_call(kw)
        self.requests.append(kw)
        arm = "edited" if "context_management" in kw else "baseline"
        queue = self._script[arm]
        body = queue.pop(0) if len(queue) > 1 else queue[0]
        return _FakeResponse(body)


def _case() -> Case:
    return Case(
        case_id="c1",
        system="sys",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_001", "name": "get", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_001", "content": "ok"}],
            },
        ],
        tools=[{"name": "get", "description": "d", "input_schema": {"type": "object"}}],
    )


# --------------------------------------------------------------------------- #
# Episode conversion
# --------------------------------------------------------------------------- #


def test_fixture_episodes_convert_to_valid_cases():
    episodes = load_episodes(FIXTURE)
    assert episodes, "fixture missing"
    cases = [c for c in (to_case(e, i) for i, e in enumerate(episodes)) if c]
    assert cases, "no fixture episode produced a usable case"
    for c in cases:
        # history ends on a user turn and contains at least one tool pair
        assert c.messages[-1]["role"] == "user"
        flat = [b for m in c.messages if isinstance(m["content"], list) for b in m["content"]]
        uses = {b["id"] for b in flat if b["type"] == "tool_use"}
        results = {b["tool_use_id"] for b in flat if b["type"] == "tool_result"}
        assert results and results <= uses, "every tool_result must pair with a tool_use"
        assert c.tools, "tool menu must be declared or the model cannot act"


def test_to_case_cuts_before_final_assistant_decision():
    episodes = load_episodes(FIXTURE)
    case = to_case(episodes[0], 0)
    assert case is not None
    # the trace's own final assistant action must NOT be in the history
    last = case.messages[-1]
    assert last["role"] == "user"
    assert any(b["type"] == "tool_result" for b in last["content"])


def test_to_case_rejects_episode_without_tool_history():
    ep = {
        "id": "x",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    assert to_case(ep, 0) is None


def test_to_case_declares_tools_seen_only_in_calls():
    episodes = load_episodes(FIXTURE)
    case = to_case(episodes[0], 0)
    assert case is not None
    names = {t["name"] for t in case.tools}
    assert "get_order" in names  # only appears via tool_calls, not the declared menu


def test_load_episodes_rejects_non_list_json(tmp_path):
    p = tmp_path / "e.json"
    p.write_text('{"not": "a list"}')
    assert load_episodes(p) == []


def test_tool_input_accepts_json_string_and_rejects_garbage():
    from distil.certify.provider_compaction import _tool_input

    assert _tool_input({"a": "1"}) == {"a": "1"}
    assert _tool_input('{"a": "1"}') == {"a": "1"}
    assert _tool_input('["not", "an", "object"]') == {}
    assert _tool_input("not json") == {}
    assert _tool_input(42) == {}


def test_to_case_defensive_branches():
    # messages not a list
    assert to_case({"messages": "nope"}, 0) is None
    # non-dict messages, nameless tool calls, orphan tool results, empty assistant
    ep = {
        "id": "messy",
        "messages": [
            "garbage",
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "orphan result before any call"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {}}]},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f"}}]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "the decision to cut"},
        ],
    }
    case = to_case(ep, 0)
    assert case is not None
    assert case.system == "sys"
    flat = [b for m in case.messages if isinstance(m["content"], list) for b in m["content"]]
    assert [b["type"] for b in flat] == ["tool_use", "tool_result"]


def test_to_case_keeps_assistant_text_and_breaks_trim_on_user_tail():
    ep = {
        "id": "textful",
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "let me check",  # text block alongside the call
                "tool_calls": [{"function": {"name": "f", "arguments": {"q": "1"}}}],
            },
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "g"}}]},
            {"role": "user", "content": "actually wait"},  # user tail after the dangling call
            {"role": "assistant", "content": "final decision"},
        ],
    }
    case = to_case(ep, 0)
    # the dangling g-call cannot be trimmed past the user tail -> unusable case
    assert case is None


def test_missing_anthropic_package_is_a_clean_message(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)  # import raises ImportError
    with pytest.raises(SystemExit, match="'anthropic' package"):
        ProviderArms().observe(_case(), None)


def test_to_case_trims_dangling_tool_use():
    # the last assistant call has no tool_result before the cut point — an
    # unpaired tool_use is an API error, so it must be trimmed away
    ep = {
        "id": "dangling",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f"}}]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "g"}}]},
            {"role": "assistant", "content": "final decision"},
        ],
    }
    case = to_case(ep, 0)
    assert case is not None
    flat = [b for m in case.messages if isinstance(m["content"], list) for b in m["content"]]
    uses = [b["id"] for b in flat if b["type"] == "tool_use"]
    results = [b["tool_use_id"] for b in flat if b["type"] == "tool_result"]
    assert uses == results  # no dangling tool_use survives


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #


def test_ensure_client_failure_is_a_clean_message(monkeypatch):
    import sys
    import types

    stub = types.ModuleType("anthropic")

    class _Boom:
        def __init__(self):
            raise RuntimeError("no key")

    stub.Anthropic = _Boom
    monkeypatch.setitem(sys.modules, "anthropic", stub)
    with pytest.raises(SystemExit, match="could not initialise"):
        ProviderArms().observe(_case(), None)


def test_api_failure_is_a_clean_message():
    class _Exploding:
        def __init__(self):
            self.beta = self
            self.messages = self

        def create(self, **kw):
            raise ValueError("rate limited")

    arms = ProviderArms(client=_Exploding())
    arms._RETRIES = 1  # don't sit through backoff in tests
    with pytest.raises(SystemExit, match="API call failed"):
        arms.observe(_case(), None)


def test_transient_api_failure_is_retried(monkeypatch):
    from distil.certify import provider_compaction as pc

    naps: list[float] = []
    monkeypatch.setattr(pc.time, "sleep", naps.append)

    class _Flaky:
        def __init__(self):
            self.beta = self
            self.messages = self
            self.failures = 2

        def create(self, **kw):
            if self.failures:
                self.failures -= 1
                raise ValueError("500 server_error")  # transient upstream blip
            return _FakeResponse(_tool_resp("a", {}))

    obs = ProviderArms(client=_Flaky()).observe(_case(), None)
    assert obs.signature.startswith("tool:")
    assert naps == [8.0, 16.0]  # bounded backoff, then success — the run survives


def test_observe_sends_beta_and_edits_only_on_edited_arm():
    client = _FakeClient(
        {"baseline": [_tool_resp("a", {})], "edited": [_tool_resp("a", {}, _FIRED)]}
    )
    arms = ProviderArms(client=client)
    edits = edit_config(trigger_tokens=500, keep_tool_uses=0)
    base = arms.observe(_case(), None)
    edited = arms.observe(_case(), edits)
    assert client.requests[0]["betas"] == [CONTEXT_MGMT_BETA]
    assert "context_management" not in client.requests[0]
    assert client.requests[1]["context_management"] == edits
    assert not base.fired
    assert edited.fired and edited.cleared_input_tokens == 900
    assert base.signature.startswith("tool:")


def test_applied_edits_without_cleared_tokens_is_not_fired():
    empty = [
        {"type": "clear_tool_uses_20250919", "cleared_tool_uses": 0, "cleared_input_tokens": 0}
    ]
    client = _FakeClient(
        {"baseline": [_tool_resp("a", {})], "edited": [_tool_resp("a", {}, empty)]}
    )
    arms = ProviderArms(client=client)
    obs = arms.observe(_case(), edit_config(trigger_tokens=500, keep_tool_uses=0))
    assert not obs.fired


def test_majority_vote_takes_modal_signature():
    client = _FakeClient(
        {"baseline": [_tool_resp("a", {}), _tool_resp("b", {}), _tool_resp("b", {})], "edited": []}
    )
    arms = ProviderArms(client=client)
    obs = arms.majority(_case(), None, votes=3)
    from distil.shadow import decision_signature

    assert obs.signature == decision_signature(_tool_resp("b", {}))


def test_live_call_budget_is_a_hard_stop():
    client = _FakeClient({"baseline": [_tool_resp("a", {})], "edited": []})
    arms = ProviderArms(client=client, max_calls=2)
    arms.observe(_case(), None)
    arms.observe(_case(), None)
    with pytest.raises(SystemExit):
        arms.observe(_case(), None)


# --------------------------------------------------------------------------- #
# OpenAI arm
# --------------------------------------------------------------------------- #


class _FakeOpenAI:
    """Scripted responses.create; records requests, cycles bodies per arm."""

    def __init__(self, script: dict[str, list[dict]]) -> None:
        self._script = {k: list(v) for k, v in script.items()}
        self.requests: list[dict] = []
        self.responses = self

    def create(self, **kw):
        self.requests.append(kw)
        arm = "edited" if "context_management" in kw else "baseline"
        queue = self._script[arm]
        body = queue.pop(0) if len(queue) > 1 else queue[0]
        return _FakeResponse(body)


def _oa_call(name: str, args: str, compacted: bool = False) -> dict:
    out: list = []
    if compacted:
        out.append({"type": "compaction", "encrypted_content": "opaque"})
    out.append({"type": "function_call", "call_id": "c1", "name": name, "arguments": args})
    return {"output": out}


def test_openai_input_conversion_maps_tool_pairs():
    from distil.certify.provider_compaction import _responses_input, _responses_tools

    items = _responses_input(_case())
    assert items[0] == {"role": "user", "content": "hi"}
    assert items[1]["type"] == "function_call" and items[1]["call_id"] == "toolu_001"
    assert json.loads(items[1]["arguments"]) == {}
    assert items[2] == {"type": "function_call_output", "call_id": "toolu_001", "output": "ok"}
    tools = _responses_tools(_case())
    assert tools == [
        {"type": "function", "name": "get", "description": "d", "parameters": {"type": "object"}}
    ]


def test_openai_observe_fires_on_compaction_item_and_matches_signature():
    client = _FakeOpenAI(
        {
            "baseline": [_oa_call("get", '{"x": "1"}')],
            "edited": [_oa_call("get", '{"x": "1"}', compacted=True)],
        }
    )
    arms = OpenAIArms(client=client)
    edits = arms.edits(trigger_tokens=100, keep_tool_uses=0)
    assert edits == [{"type": "compaction", "compact_threshold": 100}]
    base = arms.observe(_case(), None)
    edited = arms.observe(_case(), edits)
    assert "context_management" not in client.requests[0]
    assert client.requests[1]["context_management"] == edits
    assert not base.fired and edited.fired
    # same decision through compaction -> identical signature (shadow-normalized)
    assert base.signature == edited.signature
    assert base.signature.startswith("tool:")


def test_openai_text_answer_signature_and_experiment_end_to_end(tmp_path):
    text_resp = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hm"}]}]
    }
    client = _FakeOpenAI(
        {
            "baseline": [_oa_call("get", "{}")],
            "edited": [
                {
                    "output": [
                        {"type": "compaction", "encrypted_content": "x"},
                        text_resp["output"][0],
                    ]
                }
            ],
        }
    )
    arms = OpenAIArms(client=client)
    report = run_experiment([_case()] * 4, arms, tmp_path, votes=1, min_fired=2)
    assert report.protocol["target"] == "openai-compaction"
    assert report.cases_fired == 4
    assert report.ab_change_rate == 1.0  # tool_use -> text is a decision change
    assert "OpenAI server-side compaction" in report.statement


def test_openai_signature_none_and_text_blocks():
    from distil.certify.provider_compaction import _responses_input, _responses_signature

    assert _responses_signature([]) == "none"
    assert _responses_signature([{"type": "message", "content": []}]) == "text"
    # assistant text blocks in the history map to plain role messages
    case = Case(
        case_id="t",
        system="s",
        messages=[
            {"role": "assistant", "content": [{"type": "text", "text": "thinking aloud"}]},
            {"role": "user", "content": "go"},
        ],
        tools=[],
    )
    items = _responses_input(case)
    assert items[0] == {"role": "assistant", "content": "thinking aloud"}


def test_openai_client_failures_are_clean_messages(monkeypatch):
    import sys
    import types

    class _Exploding:
        def __init__(self):
            self.responses = self

        def create(self, **kw):
            raise ValueError("quota")

    arms = OpenAIArms(client=_Exploding())
    arms._RETRIES = 1  # don't sit through backoff in tests
    with pytest.raises(SystemExit, match="OpenAI API call failed"):
        arms.observe(_case(), None)

    monkeypatch.setitem(sys.modules, "openai", None)  # import raises ImportError
    with pytest.raises(SystemExit, match="'openai' package"):
        OpenAIArms().observe(_case(), None)

    stub = types.ModuleType("openai")

    class _Boom:
        def __init__(self):
            raise RuntimeError("no key")

    stub.OpenAI = _Boom
    monkeypatch.setitem(sys.modules, "openai", stub)
    with pytest.raises(SystemExit, match="could not initialise the OpenAI client"):
        OpenAIArms().observe(_case(), None)


def test_cli_openai_dry_run(capsys):
    from distil.cli import main

    rc = main(["certify-provider", str(FIXTURE), "--provider", "openai", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "openai-compaction" in out and "gpt-5.2" in out


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #


def test_run_experiment_writes_protocol_before_first_call(tmp_path):
    seen = []

    def on_call(_kw):
        seen.append((tmp_path / "protocol.json").exists())

    client = _FakeClient(
        {"baseline": [_tool_resp("a", {})], "edited": [_tool_resp("a", {}, _FIRED)]},
        on_call=on_call,
    )
    arms = ProviderArms(client=client)
    run_experiment([_case()], arms, tmp_path, votes=1, min_fired=1)
    assert seen and all(seen), "the protocol must be pre-registered before any live call"


def test_killed_run_resumes_without_re_buying_finished_cases(tmp_path, monkeypatch):
    # the provider 500s partway through and stays down long enough to exhaust the
    # retries; the cases already paid for must survive the kill
    monkeypatch.setattr("distil.certify.provider_compaction.time.sleep", lambda _s: None)
    boom = {"n": 0}

    def on_call(_kw):
        boom["n"] += 1
        if boom["n"] > 6:  # mid-run, after two full cases (3 arms x 1 vote each)
            raise RuntimeError("upstream 500")

    spec = {"baseline": [_tool_resp("a", {})], "edited": [_tool_resp("a", {}, _FIRED)]}
    cases = [_case() for _ in range(4)]

    arms = ProviderArms(client=_FakeClient(spec, on_call=on_call))
    with pytest.raises(SystemExit):
        run_experiment(cases, arms, tmp_path, votes=1, min_fired=1)
    ledger = (tmp_path / "cases.jsonl").read_text().splitlines()
    assert len(ledger) == 2, "each finished case is flushed as it completes"

    created_before = json.loads((tmp_path / "protocol.json").read_text())["created"]
    resumed = ProviderArms(client=_FakeClient(spec))
    report = run_experiment(cases, arms=resumed, out_dir=tmp_path, votes=1, min_fired=1)

    assert resumed.calls_made == 6, "only the two unfinished cases are re-bought"
    assert report.calls_made == 12, "the report still reports the full experiment cost"
    assert report.cases_total == 4
    assert json.loads((tmp_path / "protocol.json").read_text())["created"] == created_before, (
        "a resume keeps the original pre-registration timestamp"
    )


def test_changed_protocol_orphans_the_ledger(tmp_path):
    spec = {"baseline": [_tool_resp("a", {})], "edited": [_tool_resp("a", {}, _FIRED)]}
    cases = [_case() for _ in range(2)]
    run_experiment(cases, ProviderArms(client=_FakeClient(spec)), tmp_path, votes=1, min_fired=1)

    # same directory, different pre-registered trigger -> nothing may be replayed
    rerun = ProviderArms(client=_FakeClient(spec))
    run_experiment(cases, rerun, tmp_path, votes=1, min_fired=1, trigger_tokens=99999)
    assert rerun.calls_made == 6, "a parameter change must not reuse the old experiment's cases"


def test_report_same_decision_certifies_and_counts_fired(tmp_path):
    client = _FakeClient(
        {"baseline": [_tool_resp("a", {"x": "1"})], "edited": [_tool_resp("a", {"x": "1"}, _FIRED)]}
    )
    arms = ProviderArms(client=client)
    cases = [_case() for _ in range(20)]
    report = run_experiment(cases, arms, tmp_path, votes=1, alpha=0.2, delta=0.1, min_fired=10)
    assert report.cases_fired == 20
    assert report.ab_change_rate == 0.0
    assert report.aa_change_rate == 0.0
    assert report.certified and not report.underpowered
    assert report.calls_made == 60  # 3 arms x 1 vote x 20 cases
    assert report.total_cleared_input_tokens == 20 * 900
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["protocol"]["sig_version"] >= 3
    assert "CERTIFIED" in saved["statement"]


def test_decision_change_on_edited_arm_is_counted(tmp_path):
    client = _FakeClient(
        {"baseline": [_tool_resp("a", {})], "edited": [_tool_resp("DIFFERENT", {}, _FIRED)]}
    )
    arms = ProviderArms(client=client)
    report = run_experiment([_case()] * 4, arms, tmp_path, votes=1, alpha=0.1, min_fired=2)
    assert report.ab_change_rate == 1.0
    assert not report.certified
    assert "NOT certified" in report.statement


def test_unfired_cases_are_excluded_from_ab_sample(tmp_path):
    # editing never fires -> no A/B evidence -> underpowered, never certified
    client = _FakeClient({"baseline": [_tool_resp("a", {})], "edited": [_tool_resp("b", {})]})
    arms = ProviderArms(client=client)
    report = run_experiment([_case()] * 4, arms, tmp_path, votes=1, min_fired=2)
    assert report.cases_fired == 0
    assert report.underpowered and not report.certified
    assert "UNDERPOWERED" in report.statement
    # the (b != a) disagreement must NOT leak into the A/B rate: those cases carry no evidence
    assert report.ab_change_rate == 0.0


def test_aa_noise_floor_adjusts_ab_rate(tmp_path):
    # baseline flips between two decisions (hot model) -> A/A disagreement absorbs it
    client = _FakeClient(
        {
            "baseline": [_tool_resp("a", {}), _tool_resp("b", {}), _tool_resp("a", {})],
            "edited": [_tool_resp("a", {}, _FIRED)],
        }
    )
    arms = ProviderArms(client=client)
    report = run_experiment([_case()] * 2, arms, tmp_path, votes=1, min_fired=1)
    assert report.aa_change_rate > 0.0
    assert report.adjusted_change_rate <= report.ab_change_rate or report.ab_change_rate == 0.0


def test_cli_live_path_with_patched_arm(tmp_path, capsys, monkeypatch):
    from distil.certify import provider_compaction as pc
    from distil.cli import main

    fired_obs = pc.ArmObservation(signature="tool:same", fired=True, cleared_input_tokens=50)

    def fake_observe(self, case, edits):
        self.calls_made += 1
        return fired_obs if edits is not None else pc.ArmObservation("tool:same", False, 0)

    monkeypatch.setattr(pc.ProviderArms, "observe", fake_observe)
    rc = main(
        [
            "certify-provider",
            str(FIXTURE),
            "--n",
            "2",
            "--votes",
            "1",
            "--alpha",
            "0.5",
            "--min-fired",
            "1",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "provider-compaction certificate" in out
    assert (tmp_path / "out" / "protocol.json").exists()
    assert (tmp_path / "out" / "report.json").exists()


def test_cli_rejects_episodes_without_tool_history(tmp_path, capsys):
    from distil.cli import main

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "x", "messages": [{"role": "user", "content": "hi"}]}]))
    rc = main(["certify-provider", str(bad), "--dry-run"])
    assert rc == 2
    assert "no usable episodes" in capsys.readouterr().out


def test_cli_dry_run_makes_no_calls(capsys):
    from distil.cli import main

    rc = main(["certify-provider", str(FIXTURE), "--dry-run", "--votes", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No calls made" in out
    assert "3 arms" in out
