"""LiteLLM Proxy hook: logic tests that need no litellm, plus real-class tests
that skip cleanly when it is absent."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from distil.integrations import litellm_hook as lh

# Pretty-printed JSON: Tier-0 (lossless) re-serializes it compactly.
_JSON = json.dumps({"rows": [{"id": i, "name": f"n{i}"} for i in range(60)]}, indent=4)
# Long line-oriented output: only Tier-1 (digest) shrinks this materially.
_LOG = "\n".join(
    f"{i}: module_{i * 7919 % 1000} load took {i * 31 % 977} ms id={i * 104729}" for i in range(400)
)


def _anthropic(result: str) -> list[dict]:
    msgs: list[dict] = []
    for i in range(6):  # enough turns that the oldest results fall outside the recency window
        body = result if i == 0 else "ok"
        msgs += [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": body}],
            },
        ]
    return msgs


def _chat(result: str) -> list[dict]:
    msgs: list[dict] = []
    for i in range(6):
        body = result if i == 0 else "ok"
        msgs += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "sh", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"c{i}", "content": body},
        ]
    return msgs


@pytest.fixture
def core(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_SESSION", "test-litellm")
    return lh.HookCore(ledger_path=tmp_path / "savings.jsonl")


def test_chat_call_is_compressed_losslessly_by_default(core):
    data = {"model": "gpt-4o", "messages": _chat(_JSON), "stream": True}
    out = core.apply(data, "acompletion")
    assert out is not data and out["stream"] is True and out["model"] == "gpt-4o"
    assert len(out["messages"][1]["content"]) < len(_JSON)
    assert "n59" in out["messages"][1]["content"]  # nothing dropped
    assert data["messages"][1]["content"] == _JSON  # input not mutated


def test_anthropic_messages_call_is_compressed(core):
    out = core.apply(
        {"model": "claude-opus-4-8", "messages": _anthropic(_JSON)}, "anthropic_messages"
    )
    assert len(out["messages"][1]["content"][0]["content"]) < len(_JSON)


def test_default_never_digests_but_digest_opt_in_does(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_SESSION", "test-litellm")
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    data = {"model": "gpt-4o", "messages": _chat(_LOG)}
    lossless = lh.HookCore(ledger_path=tmp_path / "a.jsonl").apply(data, "completion")
    digest = lh.HookCore(digest=True, ledger_path=tmp_path / "b.jsonl").apply(data, "completion")
    assert lossless["messages"][1]["content"] == _LOG
    assert len(digest["messages"][1]["content"]) < len(_LOG)


def test_subscription_refuses_digest_and_stays_lossless(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_SESSION", "test-litellm")
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    monkeypatch.setattr(lh, "_warned", set())
    data = {"model": "gpt-4o", "messages": _chat(_LOG)}
    with caplog.at_level("WARNING", logger="distil.litellm"):
        hooks = [lh.HookCore(digest=True, ledger_path=tmp_path / f"{i}.jsonl") for i in "ab"]
    assert all(h.digest is False and h.savings.mode == "verbatim" for h in hooks)
    assert hooks[0].apply(data, "completion")["messages"][1]["content"] == _LOG
    refused = [r for r in caplog.records if "subscription" in r.getMessage()]
    assert len(refused) == 1


def test_payg_digest_warns_once_that_stubs_are_irrecoverable(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_SESSION", "test-litellm")
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    monkeypatch.setattr(lh, "_warned", set())
    with caplog.at_level("WARNING", logger="distil.litellm"):
        for name in ("a", "b"):
            assert lh.HookCore(digest=True, ledger_path=tmp_path / f"{name}.jsonl").digest
        lh.HookCore(ledger_path=tmp_path / "c.jsonl")  # lossless: no warning
    warned = [r.getMessage() for r in caplog.records]
    assert len(warned) == 1 and "distil proxy" in warned[0]


@pytest.mark.parametrize(
    "data,call_type",
    [
        ({"model": "m", "input": "x"}, "embedding"),
        ({"model": "m", "messages": "not a list"}, "completion"),
        ({"model": "m", "messages": _chat(_JSON)}, "aresponses"),
        ({"model": "m", "messages": _chat(_JSON)}, None),
        ({"model": "m"}, "completion"),
    ],
)
def test_unrecognized_requests_pass_through_untouched(core, data, call_type):
    assert core.apply(data, call_type) is data


def test_fail_open_on_adapter_error(core, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret-prompt-text")

    monkeypatch.setattr(lh, "compress_chat_completions", boom)
    data = {"model": "m", "messages": _chat(_JSON)}
    assert core.apply(data, "completion") is data


def test_failure_does_not_log_content(core, monkeypatch, caplog):
    monkeypatch.setattr(lh, "compress_chat_completions", lambda *a, **k: 1 / 0)
    with caplog.at_level("DEBUG", logger="distil.litellm"):
        core.apply({"model": "m", "messages": _chat("PRIVATE-PROMPT")}, "completion")
    assert "PRIVATE-PROMPT" not in caplog.text and "ZeroDivisionError" in caplog.text


def test_savings_land_in_the_ledger_content_free(tmp_path, core):
    core.apply({"model": "claude-opus-4-8", "messages": _anthropic(_JSON)}, "anthropic_messages")
    core.savings.flush()  # first call already flushed (max_age); no-op if so
    rows = [json.loads(line) for line in (tmp_path / "savings.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["mode"] == "verbatim"
    assert rows[0]["distil_input_tokens"] < rows[0]["baseline_input_tokens"]
    assert rows[0]["model"] == "claude-opus-4-8" and "rows" not in json.dumps(rows[0])


def test_lazy_names_raise_helpful_error_without_litellm(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)  # makes `import litellm` ImportError
    monkeypatch.setattr(lh, "_cache", {})
    with pytest.raises(ImportError, match=r"distil-llm\[litellm\]"):
        lh.DistilCompressionHook  # noqa: B018
    with pytest.raises(AttributeError):
        lh.nope  # noqa: B018


def test_real_litellm_hook_class(tmp_path, monkeypatch):
    pytest.importorskip("litellm")
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_SESSION", "test-litellm")
    monkeypatch.setattr(lh, "_cache", {})
    from litellm.integrations.custom_logger import CustomLogger

    hook = lh.DistilCompressionHook(ledger_path=tmp_path / "savings.jsonl")
    assert isinstance(hook, CustomLogger)
    data = {"model": "gpt-4o", "messages": _chat(_JSON)}
    out = asyncio.run(hook.async_pre_call_hook(None, None, data, "acompletion"))
    assert len(out["messages"][1]["content"]) < len(_JSON)
    monkeypatch.delenv("DISTIL_LITELLM_DIGEST", raising=False)
    assert lh.proxy_handler_instance.core.digest is False


# After the lazy-import test: importing litellm here would mask its ImportError.
def test_env_digest_opt_in_goes_through_the_subscription_guard(tmp_path, monkeypatch):
    pytest.importorskip("litellm")
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_SESSION", "test-litellm")
    monkeypatch.setenv("DISTIL_LITELLM_DIGEST", "1")
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    monkeypatch.setattr(lh, "_cache", {})
    monkeypatch.setattr(lh, "_warned", set())
    assert lh.proxy_handler_instance.core.digest is False


def test_responses_calls_compress_the_input_array():
    """LiteLLM's Responses route carries Codex-shaped ``input`` items, not ``messages``."""
    big = "\n".join(f"row {i}: value_{i} status=ok detail=lorem" for i in range(60))
    data = {
        "model": "gpt-5.2",
        "input": [
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": big},
        ],
    }
    for call_type in ("responses", "aresponses"):
        out = lh.compress_request(data, call_type, digest=True)
        assert out is not None
        new, before, after = out
        assert after < before and "handle=" in new["input"][1]["output"]
    assert lh.compress_request({"input": "hi"}, "responses") is None


def test_in_process_wrapper_uses_the_chat_adapter_and_the_policy(monkeypatch):
    """``litellm.completion`` messages are Chat-shaped for every provider: a list-shaped
    tool message is tool output (Tier-1), and a subscription session never digests."""
    from distil.integrations import litellm as dll

    big = "\n".join(f"row {i}: value_{i} status=ok detail=lorem" for i in range(60))
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": big}]},
    ]
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    out = dll.compress({"model": "gpt-5.2", "messages": msgs})
    assert "handle=" in out["messages"][1]["content"][0]["text"]
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    out = dll.compress({"model": "gpt-5.2", "messages": msgs})
    assert "handle=" not in json.dumps(out["messages"])
