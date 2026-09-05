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


def _bash_call(tid: str, command: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": command}}],
    }


def _tool_result(tid: str, content: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": content}],
    }


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


def test_a_cd_prefixed_read_is_still_a_read() -> None:
    """`cd /repo && cat main.py` is the commonest way an agent reads a file. Requiring
    every stage to be a reader refused it, which digested the quote."""
    out, _store = compress_messages(_anthropic_session("cd /app && cat handlers.py"))
    assert _result(out, "r1") == SRC


def test_the_census_names_which_rule_froze_the_block() -> None:
    """`tool_result_shell_read` is the new rule's cost, reported apart from 1.49.0's."""
    from distil.adapters.anthropic import take_census

    compress_messages(_anthropic_session("cat /app/handlers.py"))
    census = take_census() or {}
    assert census.get("tool_result_shell_read", 0) > 0
    assert census.get("tool_result_exact_quote", 0) == 0


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


# --------------------------------------------------------------------------- contracts


def _list_shaped_read(command: str) -> list[dict]:
    """A read whose tool_result carries content as a LIST of parts, as Claude Code sends."""
    msgs = _anthropic_session(command)
    msgs[2]["content"][0]["content"] = [{"type": "text", "text": SRC}]
    return msgs


def test_a_list_shaped_exact_quote_block_is_still_censused() -> None:
    """The census's whole value is that it accounts for the entire payload.

    An exempt block counted as zero tokens is worse than no census: it is in the payload,
    it is protected on purpose, and the report would attribute it to nothing.
    """
    from distil.adapters.anthropic import take_census
    from distil.proxy import _count_messages

    msgs = _list_shaped_read("cat /app/handlers.py")
    compress_messages(msgs)
    census = take_census() or {}
    assert census.get("tool_result_shell_read", 0) > 0
    total, payload = sum(census.values()), _count_messages(msgs)
    assert abs(total - payload) / payload < 0.005, f"census {total} vs payload {payload}"


def test_a_list_shaped_tool_message_is_censused_on_the_openai_path_too() -> None:
    from distil.adapters.anthropic import take_census

    messages: list[dict] = [
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
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": SRC}]},
    ]
    compress_chat_completions(messages)
    assert (take_census() or {}).get("tool_result_shell_read", 0) > 0


def _is_image(block: object) -> bool:
    return isinstance(block, dict) and block.get("type") == "image"


def test_the_quote_hazard_retry_does_not_swallow_the_images(monkeypatch) -> None:
    """The vision deduper elides an image it has already SEEN, so it is per-pass state.

    Left alive across the widen-on-miss retry it would have seen every image during the
    first pass, so the second pass elides all of them and the model receives none at all.
    Verified against the pre-fix shape: 0 images survive, versus 1 here.
    """
    monkeypatch.setenv("DISTIL_VISION", "1")
    img = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo=" * 400},
    }
    # A re-read of the same path whose content CHANGED: it supersedes the first read, which
    # therefore digests, which loses the Edit's quote — the miss that forces the retry.
    renamed = SRC.replace("handler_7", "renamed_7")
    msgs: list[dict] = [
        {"role": "user", "content": "refactor"},
        _bash_call("r1", "cat /app/handlers.py"),
        _tool_result("r1", SRC),
        {"role": "user", "content": [dict(img), dict(img)]},
    ]
    for i in range(4):
        msgs += [_bash_call(f"n{i}", "pytest -q"), _tool_result(f"n{i}", NOISE)]
    msgs += [
        _bash_call("r2", "cat /app/handlers.py"),
        _tool_result("r2", renamed),
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "e1", "name": "Edit", "input": {"old_string": QUOTE}}
            ],
        },
    ]
    out, _store = compress_messages(msgs)
    assert _result(out, "r1") == SRC, "the retry must have widened the exemption"
    assert take_quote_hazard() == {"survived": 1, "lost": 0}
    images = [b for m in out for b in (m.get("content") or []) if _is_image(b)]
    assert len(images) == 1, "the second pass must not elide every image as already-seen"
