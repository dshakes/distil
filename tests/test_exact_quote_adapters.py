"""The exact-quote guarantee has to hold on every request shape, not just Anthropic's.

Codex reads files over the Responses API and Gemini over generateContent; an agent whose
edits silently stop matching is the same failure whichever wire format it speaks. Before
this, only the Messages path had the exemption at all.
"""

from __future__ import annotations

import json

from distil.adapters.anthropic import compress_messages, take_quote_hazard
from distil.adapters.gemini import compress_generate_request
from distil.adapters.openai import compress_chat_completions, compress_responses_input

SRC = "\n".join(
    f"def handler_{i}(request):\n    payload = request.json()\n    return {{'ok': True, 'n': {i}}}"
    for i in range(40)
)
QUOTE = "def handler_7(request):\n    payload = request.json()\n    return {'ok': True, 'n': 7}"
NOISE = "\n".join(f"2026-09-04 10:00:{i:02d} INFO worker ok id={i}" for i in range(60))


# --------------------------------------------------------------------------- Anthropic


def _anthropic_session(command: str, *, filler: int = 6) -> list[dict]:
    msgs: list[dict] = [
        {"role": "user", "content": "refactor the handlers"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "r1", "name": "Bash", "input": {"command": command}}
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": SRC}]},
    ]
    for i in range(filler):
        msgs += [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"n{i}",
                        "name": "Bash",
                        "input": {"command": "pytest -q"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"n{i}", "content": NOISE}],
            },
        ]
    return msgs


def _result(out: list[dict], tool_use_id: str) -> str:
    for m in out:
        for blk in m.get("content") or ():
            if isinstance(blk, dict) and blk.get("tool_use_id") == tool_use_id:
                return str(blk.get("content"))
    raise AssertionError(f"no tool_result for {tool_use_id}")


def test_shell_read_survives_byte_exact_however_old() -> None:
    """`cat file` is a file read. Digesting it makes the next Edit unmatchable — the
    failure that ends a run with the agent saying "done" and the disk untouched."""
    out, _store = compress_messages(_anthropic_session("cat /app/handlers.py"))
    assert _result(out, "r1") == SRC
    assert QUOTE in _result(out, "r1")


def test_ordinary_shell_output_beside_it_still_digests() -> None:
    """The exemption reads the command, so the Bash tool is not exempted wholesale."""
    out, _store = compress_messages(_anthropic_session("cat /app/handlers.py"))
    assert "handle=" in _result(out, "n0"), "routine test output should still digest"


def test_a_piped_read_is_not_treated_as_a_file_read() -> None:
    out, _store = compress_messages(_anthropic_session("cat /app/handlers.py | grep def"))
    assert "handle=" in _result(out, "r1")


def test_quote_hazard_is_counted_and_reported() -> None:
    msgs = _anthropic_session("cat /app/handlers.py")
    msgs.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "e1",
                    "name": "Edit",
                    "input": {"old_string": QUOTE, "new_string": QUOTE + "  # x"},
                }
            ],
        }
    )
    compress_messages(msgs)
    assert take_quote_hazard() == {"survived": 1, "lost": 0}


def test_no_edit_in_the_history_means_no_hazard_record() -> None:
    """Most requests carry no literal-match edit; those must not fabricate a count."""
    compress_messages(_anthropic_session("cat /app/handlers.py"))
    assert take_quote_hazard() is None


# --------------------------------------------------------------------------- OpenAI


def test_chat_completions_exempts_a_shell_read() -> None:
    messages: list[dict] = [
        {"role": "user", "content": "refactor"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "cat /app/handlers.py"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": SRC},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": "pytest"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": NOISE},
    ]
    out, _store = compress_chat_completions(messages)
    assert out[2]["content"] == SRC
    assert "handle=" in out[4]["content"], "ordinary output must still digest"


def test_responses_input_exempts_a_shell_read() -> None:
    items: list[dict] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        {
            "type": "function_call",
            "call_id": "f1",
            "name": "shell",
            "arguments": json.dumps({"command": "sed -n '1,200p' /app/handlers.py"}),
        },
        {"type": "function_call_output", "call_id": "f1", "output": SRC},
        {
            "type": "function_call",
            "call_id": "f2",
            "name": "shell",
            "arguments": json.dumps({"command": "pytest -q"}),
        },
        {"type": "function_call_output", "call_id": "f2", "output": NOISE},
    ]
    out, _store = compress_responses_input(items)
    assert out[2]["output"] == SRC
    assert "handle=" in out[4]["output"]


# --------------------------------------------------------------------------- Gemini


def test_gemini_exempts_a_shell_read_matched_positionally() -> None:
    """Gemini gives a functionCall no id, so call and response are paired by name+order."""
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "refactor"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "bash", "args": {"command": "cat /app/handlers.py"}}}
                ],
            },
            {
                "role": "user",
                "parts": [{"functionResponse": {"name": "bash", "response": {"stdout": SRC}}}],
            },
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "bash", "args": {"command": "pytest -q"}}}],
            },
            {
                "role": "user",
                "parts": [{"functionResponse": {"name": "bash", "response": {"stdout": NOISE}}}],
            },
        ]
    }
    out, _store = compress_generate_request(body)
    contents = out["contents"]
    kept = contents[2]["parts"][0]["functionResponse"]["response"]["stdout"]
    other = contents[4]["parts"][0]["functionResponse"]["response"]["stdout"]
    assert kept == SRC
    assert "handle=" in other, "the second call was not a read and must still digest"


def test_another_adapter_on_the_same_thread_clears_the_hazard_count() -> None:
    """The proxy reads the counter per request off the handler's own thread, so an
    Anthropic request's count must not be reported against the next OpenAI one."""
    msgs = _anthropic_session("cat /app/handlers.py")
    msgs.append(
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "e1", "name": "Edit", "input": {"old_string": QUOTE}}
            ],
        }
    )
    compress_messages(msgs)
    assert take_quote_hazard() is not None
    compress_chat_completions([{"role": "user", "content": "unrelated"}])
    assert take_quote_hazard() is None
    compress_responses_input([])
    assert take_quote_hazard() is None
    compress_generate_request({"contents": []})
    assert take_quote_hazard() is None
