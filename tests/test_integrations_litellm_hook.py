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
    data = {"model": "gpt-4o", "messages": _chat(_LOG)}
    lossless = lh.HookCore(ledger_path=tmp_path / "a.jsonl").apply(data, "completion")
    digest = lh.HookCore(digest=True, ledger_path=tmp_path / "b.jsonl").apply(data, "completion")
    assert lossless["messages"][1]["content"] == _LOG
    assert len(digest["messages"][1]["content"]) < len(_LOG)


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
