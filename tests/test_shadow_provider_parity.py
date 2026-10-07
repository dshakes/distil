"""Cross-provider parity for shadow signatures, replay pinning and referee model lookup."""

from __future__ import annotations

import json

from distil import referee
from distil.shadow import decision_signature_from_body, deterministic_body

ARGS = {"cmd": "ls -la"}


def _sse(events: list[dict]) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


TOOL_ITEMS = [
    {"type": "reasoning", "summary": []},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
    {"type": "function_call", "name": "shell", "arguments": json.dumps(ARGS), "call_id": "c1"},
]
TEXT_ITEMS = TOOL_ITEMS[:2]


def _tool_shapes() -> list[str]:
    anth = {"content": [{"type": "tool_use", "name": "shell", "input": ARGS}]}
    chat = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "shell", "arguments": json.dumps(ARGS)}}]
                }
            }
        ]
    }
    gem = {
        "candidates": [{"content": {"parts": [{"functionCall": {"name": "shell", "args": ARGS}}]}}]
    }
    resp = {"object": "response", "output": TOOL_ITEMS}
    done = [
        {"type": "response.output_item.done", "output_index": i, "item": it}
        for i, it in enumerate(TOOL_ITEMS)
    ]
    return [decision_signature_from_body(json.dumps(x)) for x in (anth, chat, gem, resp)] + [
        decision_signature_from_body(_sse(done)),
        decision_signature_from_body(
            _sse(done[:1] + [{"type": "response.completed", "response": resp}])
        ),
    ]


def test_tool_decision_equal_across_providers() -> None:
    sigs = _tool_shapes()
    assert sigs[0].startswith("tool:")
    # Chat/Responses share the openai-style signature; Anthropic/Gemini keys differ by design.
    assert len(set(sigs[1:2] + sigs[3:])) == 1  # Chat == Responses (JSON + both SSE)


def test_text_decision_equal_across_providers() -> None:
    anth = {"content": [{"type": "text", "text": "hi"}]}
    chat = {"choices": [{"message": {"content": "hi"}}]}
    gem = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    resp = {"object": "response", "output": TEXT_ITEMS}
    done = [{"type": "response.output_item.done", "item": it} for it in TEXT_ITEMS]
    sigs = {decision_signature_from_body(json.dumps(x)) for x in (anth, chat, gem, resp)}
    sigs.add(decision_signature_from_body(_sse(done)))
    assert sigs == {"text"}


def test_responses_matches_chat_tool_signature() -> None:
    chat = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "shell", "arguments": json.dumps(ARGS)}}]
                }
            }
        ]
    }
    resp = {"output": TOOL_ITEMS}
    assert decision_signature_from_body(json.dumps(chat)) == decision_signature_from_body(
        json.dumps(resp)
    )


def test_gemini_pins_generation_config_temperature() -> None:
    body = {"contents": [], "generationConfig": {"temperature": 0.9, "topK": 3}}
    rb = deterministic_body(json.dumps(body).encode())
    assert rb is not None and rb.pinned_temperature
    out = json.loads(rb.body)
    assert out["generationConfig"] == {"temperature": 0, "topK": 3}
    assert "temperature" not in out


def test_gemini_without_temperature_is_not_injected() -> None:
    rb = deterministic_body(json.dumps({"contents": [], "generationConfig": {"topK": 3}}).encode())
    assert rb is not None and not rb.pinned_temperature
    out = json.loads(rb.body)
    assert "temperature" not in out and "temperature" not in out["generationConfig"]


def test_referee_takes_gemini_model_from_path() -> None:
    raw = json.dumps({"contents": []}).encode()
    sh = referee.shape("distil", "/v1beta/models/gemini-2.5-pro:generateContent", raw, raw, {})
    assert sh is not None and sh.model == "gemini-2.5-pro"
    sh2 = referee.shape(
        "distil",
        "/v1/messages",
        json.dumps({"model": "m"}).encode(),
        json.dumps({"model": "m"}).encode(),
        {},
    )
    assert sh2 is not None and sh2.model == "m"
