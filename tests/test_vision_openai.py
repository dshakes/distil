"""Vision duplicate elision (ADR 0003), ported to the OpenAI adapter.

Mirrors tests/test_vision.py's Anthropic coverage for both OpenAI shapes: Chat
Completions ``image_url`` parts and the Responses API's flat ``input_image``
items. The transform, the certificate gate, and the census reasons are shared
with Anthropic via ``compress.vision`` — only the block shape differs here.
"""

from __future__ import annotations

import base64
import json
import struct

import pytest

from distil.adapters.openai import (
    compress_chat_completions,
    compress_responses_input,
    count_responses_tokens,
)
from distil.compress import vision
from distil.proxy import _count_messages


def _png(w: int, h: int, pad: int = 4096) -> bytes:
    ihdr = b"IHDR" + struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + ihdr + b"\x00\x00\x00\x00" + b"\x00" * pad


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _data_uri(raw: bytes) -> str:
    return f"data:image/png;base64,{_b64(raw)}"


def _chat_image_part(raw: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": _data_uri(raw), "detail": "auto"}}


def _input_image_item(raw: bytes) -> dict:
    return {"type": "input_image", "image_url": _data_uri(raw), "detail": "auto"}


@pytest.fixture
def certified(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_VISION", raising=False)
    cert = tmp_path / "certificates" / "vision.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(json.dumps({"strategy": "vision", "non_inferior": True}))
    return tmp_path


def _msg(role: str, part: dict) -> dict:
    return {"role": role, "content": [part]}


# ---------------------------------------------------------------------------
# Chat Completions — image_url
# ---------------------------------------------------------------------------


def test_chat_first_occurrence_is_never_touched(certified):
    part = _chat_image_part(_png(800, 600))
    out, _store = compress_chat_completions([_msg("user", part)])
    assert out[0]["content"][0] == part


def test_chat_repeat_is_elided_and_recovers_byte_exact(certified):
    raw = _png(1024, 1024)
    part = _chat_image_part(raw)
    messages = [
        _msg("user", json.loads(json.dumps(part))),
        {"role": "assistant", "content": "looking"},
        _msg("user", json.loads(json.dumps(part))),
    ]
    out, store = compress_chat_completions(messages)

    assert out[0]["content"][0]["type"] == "image_url", "first occurrence must survive"
    stub = out[2]["content"][0]
    assert stub["type"] == "text", "the repeat should have become a reference"
    assert "distil:image" in stub["text"]

    handle = stub["text"].split("handle=")[1].split()[0]
    recovered = json.loads(store.expand(handle))
    assert recovered == part["image_url"], "expand did not return the original image_url object"
    payload = recovered["url"].split(";base64,", 1)[1]
    assert base64.b64decode(payload) == raw, "image bytes did not round-trip"


def test_chat_distinct_images_are_never_elided(certified):
    a, b = _chat_image_part(_png(1024, 1024)), _chat_image_part(_png(512, 512))
    messages = [_msg("user", a), {"role": "assistant", "content": "x"}, _msg("user", b)]
    out, _ = compress_chat_completions(messages)
    assert out[0]["content"][0]["type"] == "image_url"
    assert out[2]["content"][0]["type"] == "image_url", "a different image was wrongly elided"


def test_chat_verbatim_mode_never_elides(certified):
    part = _chat_image_part(_png(1024, 1024))
    messages = [
        _msg("user", json.loads(json.dumps(part))),
        _msg("user", json.loads(json.dumps(part))),
    ]
    out, _ = compress_chat_completions(messages, verbatim=True)
    assert all(m["content"][0]["type"] == "image_url" for m in out)


def test_chat_tiny_images_are_left_alone(certified):
    tiny = _chat_image_part(b"\x89PNG" + b"x")
    messages = [_msg("user", json.loads(json.dumps(tiny))) for _ in range(2)]
    out, _ = compress_chat_completions(messages)
    assert all(m["content"][0]["type"] == "image_url" for m in out)


def test_chat_malformed_image_url_passes_through(certified):
    weird = [
        {"type": "image_url"},  # no image_url object
        {"type": "image_url", "image_url": "not-a-dict"},
        {"type": "image_url", "image_url": {}},
    ]
    out, _ = compress_chat_completions([{"role": "user", "content": weird}])
    assert out[0]["content"] == weird


def test_chat_url_sources_are_never_deduped(certified):
    """A plain (non-data) URL is not proof of identical bytes — same rule as Anthropic."""
    part = {"type": "image_url", "image_url": {"url": "https://example.com/shot.png"}}
    messages = [_msg("user", dict(part)) for _ in range(2)]
    out, _ = compress_chat_completions(messages)
    assert all(m["content"][0]["type"] == "image_url" for m in out)


def test_chat_disabled_without_a_certificate(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_VISION", raising=False)
    monkeypatch.setattr(vision, "_shipped_certificate_path", lambda: tmp_path / "absent.json")
    part = _chat_image_part(_png(1024, 1024))
    messages = [_msg("user", part), _msg("user", json.loads(json.dumps(part)))]
    out, _ = compress_chat_completions(messages)
    assert out == messages, "images were touched with no certificate present"


def test_chat_census_is_exhaustive_for_images(certified):
    from distil.adapters.anthropic import take_census

    raw = _png(1024, 1024)
    part = _chat_image_part(raw)
    messages = [
        _msg("user", json.loads(json.dumps(part))),
        {"role": "assistant", "content": "prose"},
        _msg("user", json.loads(json.dumps(part))),
    ]
    compress_chat_completions(messages)
    census = take_census()
    payload = _count_messages(messages)
    total = sum(census.values())
    assert abs(total - payload) / max(payload, 1) < 0.005, (
        f"census {total} != payload {payload}, buckets={census}"
    )
    assert census.get("image_elided", 0) > 0


# ---------------------------------------------------------------------------
# Responses API — input_image
# ---------------------------------------------------------------------------


def _resp_msg(role: str, item: dict) -> dict:
    return {"type": "message", "role": role, "content": [item]}


def test_responses_first_occurrence_is_never_touched(certified):
    item = _input_image_item(_png(800, 600))
    out, _store = compress_responses_input([_resp_msg("user", item)])
    assert out[0]["content"][0] == item


def test_responses_repeat_is_elided_and_recovers_byte_exact(certified):
    raw = _png(1024, 1024)
    item = _input_image_item(raw)
    items = [
        _resp_msg("user", json.loads(json.dumps(item))),
        _resp_msg("assistant", {"type": "output_text", "text": "looking"}),
        _resp_msg("user", json.loads(json.dumps(item))),
    ]
    out, store = compress_responses_input(items)

    assert out[0]["content"][0]["type"] == "input_image"
    stub = out[2]["content"][0]
    assert stub["type"] == "input_text", "the repeat should have become a reference"
    assert "distil:image" in stub["text"]

    handle = stub["text"].split("handle=")[1].split()[0]
    recovered = json.loads(store.expand(handle))
    assert recovered == {"url": item["image_url"]}
    payload = recovered["url"].split(";base64,", 1)[1]
    assert base64.b64decode(payload) == raw


def test_responses_verbatim_mode_never_elides(certified):
    item = _input_image_item(_png(1024, 1024))
    items = [
        _resp_msg("user", json.loads(json.dumps(item))),
        _resp_msg("user", json.loads(json.dumps(item))),
    ]
    out, _ = compress_responses_input(items, verbatim=True)
    assert all(i["content"][0]["type"] == "input_image" for i in out)


def test_responses_malformed_image_passes_through(certified):
    weird = [
        {"type": "input_image"},  # no image_url
        {"type": "input_image", "image_url": None},
        {"type": "input_image", "file_id": "file-abc"},  # file reference, no url/bytes
    ]
    out, _ = compress_responses_input([_resp_msg("user", w) for w in weird][:1])
    # Each malformed shape individually passes through untouched.
    for w in weird:
        out, _ = compress_responses_input([_resp_msg("user", w)])
        assert out[0]["content"][0] == w


def test_count_responses_tokens_counts_input_image(certified):
    raw = _png(1024, 1024)
    item = _input_image_item(raw)
    before = count_responses_tokens([_resp_msg("user", item)])
    assert before >= 1000, "image item counted as ~free"


def test_responses_census_is_exhaustive_for_images(certified):
    from distil.adapters.anthropic import take_census

    raw = _png(1024, 1024)
    item = _input_image_item(raw)
    items = [
        _resp_msg("user", json.loads(json.dumps(item))),
        _resp_msg("assistant", {"type": "output_text", "text": "prose"}),
        _resp_msg("user", json.loads(json.dumps(item))),
    ]
    compress_responses_input(items)
    census = take_census()
    payload = count_responses_tokens(items)
    total = sum(census.values())
    assert abs(total - payload) / max(payload, 1) < 0.005, (
        f"census {total} != payload {payload}, buckets={census}"
    )
    assert census.get("image_elided", 0) > 0


# ---------------------------------------------------------------------------
# Cache contract: elision must be byte-stable across separate calls (turns),
# since a stub that differed run-to-run would bust the provider's prefix cache
# the same way the sliding recency window did.
# ---------------------------------------------------------------------------


def test_chat_elision_is_byte_stable_across_turns(certified):
    raw = _png(1024, 1024)
    part = _chat_image_part(raw)
    messages = [
        _msg("user", json.loads(json.dumps(part))),
        {"role": "assistant", "content": "x"},
        _msg("user", json.loads(json.dumps(part))),
    ]
    out1, _ = compress_chat_completions([json.loads(json.dumps(m)) for m in messages])
    out2, _ = compress_chat_completions([json.loads(json.dumps(m)) for m in messages])
    assert out1[2]["content"][0]["text"] == out2[2]["content"][0]["text"]


def test_responses_elision_is_byte_stable_across_turns(certified):
    raw = _png(1024, 1024)
    item = _input_image_item(raw)
    items = [
        _resp_msg("user", json.loads(json.dumps(item))),
        _resp_msg("assistant", {"type": "output_text", "text": "x"}),
        _resp_msg("user", json.loads(json.dumps(item))),
    ]
    out1, _ = compress_responses_input([json.loads(json.dumps(i)) for i in items])
    out2, _ = compress_responses_input([json.loads(json.dumps(i)) for i in items])
    assert out1[2]["content"][0]["text"] == out2[2]["content"][0]["text"]
