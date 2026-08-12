"""Exhaustiveness of the eligibility census, across every adapter and block kind.

The census exists to explain a savings number. A PARTIAL census is worse than none:
the unattributed tokens are invisible, so whatever bucket happens to be largest silently
inherits the blame. The original version of this invariant covered only the Anthropic
text path, and review caught two holes it could not see — the OpenAI adapter's own
assistant/user walking, and image blocks (which the baseline counts by pixel area).
"""

from __future__ import annotations

import base64
import io

import pytest

from distil.adapters.anthropic import compress_messages, take_census
from distil.adapters.openai import compress_chat_completions, compress_responses_input
from distil.proxy import _count_messages

_LOG = "\n".join(f"2026-08-11 INFO processed record {i} in {i % 7}ms" for i in range(60))
_PROSE = " ".join(f"Reasoning sentence number {i}." for i in range(150))


def _assert_exhaustive(census: dict[str, int] | None, payload: int, label: str) -> None:
    assert census, f"{label}: no census recorded"
    total = sum(census.values())
    assert abs(total - payload) / max(payload, 1) < 0.005, (
        f"{label}: census {total} != payload {payload} — "
        f"{payload - total} tokens unattributed. buckets={census}"
    )


def test_anthropic_messages_are_fully_attributed() -> None:
    msgs = []
    for turn in range(5):
        msgs.append({"role": "user", "content": "continue"})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": _PROSE}]})
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{turn}", "content": _LOG}],
            }
        )
    compress_messages(msgs, verbatim=False)
    _assert_exhaustive(take_census(), _count_messages(msgs), "anthropic")


def test_openai_chat_completions_are_fully_attributed() -> None:
    """The hole review found: this adapter opens a census and shares the tool_result
    helper, but walks assistant/user text itself."""
    msgs = []
    for turn in range(5):
        msgs.append({"role": "user", "content": "continue"})
        msgs.append({"role": "assistant", "content": _PROSE})
        msgs.append({"role": "tool", "tool_call_id": f"t{turn}", "content": _LOG})
    compress_chat_completions(msgs, verbatim=False)
    _assert_exhaustive(take_census(), _count_messages(msgs), "openai chat")


def test_openai_list_content_parts_are_fully_attributed() -> None:
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": _PROSE}]},
        {"role": "assistant", "content": [{"type": "text", "text": _PROSE}]},
        {"role": "tool", "tool_call_id": "t", "content": [{"type": "text", "text": _LOG}]},
    ]
    compress_chat_completions(msgs, verbatim=False)
    _assert_exhaustive(take_census(), _count_messages(msgs), "openai parts")


def _png(width: int, height: int) -> str:
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (11, 22, 33)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_image_blocks_are_attributed_on_the_same_basis_as_the_baseline() -> None:
    """Images are billed by pixel area, not base64 length. If the census counted them as
    text the totals would be on different scales and exhaustiveness could never hold —
    which is why `vision.block_tokens` is shared with `proxy._count_messages`."""
    data = _png(320, 240)
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": data},
                },
                {"type": "text", "text": _PROSE},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": _PROSE}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": _LOG}],
        },
    ]
    compress_messages(msgs, verbatim=False)
    census = take_census()
    _assert_exhaustive(census, _count_messages(msgs), "images")
    assert (census or {}).get("image_kept", 0) > 0, "image tokens landed in no image bucket"


def test_openai_responses_api_input_is_fully_attributed() -> None:
    """The Responses API is a separate entry point with its own walking — and its own
    census to open. A hole here would be invisible to the Chat Completions tests."""
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": _PROSE}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": _PROSE}],
        },
        {"type": "function_call_output", "call_id": "c1", "output": _LOG},
    ]
    compress_responses_input(items, verbatim=False)
    census = take_census()
    assert census, "responses API recorded no census"
    assert census.get("assistant_text", 0) > 0, "assistant output_text was not attributed"
    assert census.get("user_text", 0) > 0, "user input_text was not attributed"
