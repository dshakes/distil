"""Vision duplicate elision (ADR 0003), ported to the Gemini adapter.

Mirrors tests/test_vision.py's Anthropic coverage for Gemini's two image
shapes: ``inlineData`` (base64 bytes) and ``fileData`` (a ``fileUri``). The
transform, the certificate gate, and the census reasons are shared with
Anthropic via ``compress.vision`` — only the part shape differs here.
"""

from __future__ import annotations

import base64
import json
import struct

import pytest

from distil.adapters.gemini import compress_generate_request, count_tokens
from distil.compress import vision


def _png(w: int, h: int, pad: int = 4096) -> bytes:
    ihdr = b"IHDR" + struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + ihdr + b"\x00\x00\x00\x00" + b"\x00" * pad


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _inline_part(raw: bytes) -> dict:
    return {"inlineData": {"mimeType": "image/png", "data": _b64(raw)}}


def _file_part(uri: str) -> dict:
    return {"fileData": {"mimeType": "image/png", "fileUri": uri}}


@pytest.fixture
def certified(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_VISION", raising=False)
    cert = tmp_path / "certificates" / "vision.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(json.dumps({"strategy": "vision", "non_inferior": True}))
    return tmp_path


def _content(role: str, part: dict) -> dict:
    return {"role": role, "parts": [part]}


def _body(contents: list[dict]) -> dict:
    return {"contents": contents}


# ---------------------------------------------------------------------------
# inlineData
# ---------------------------------------------------------------------------


def test_inline_first_occurrence_is_never_touched(certified):
    part = _inline_part(_png(800, 600))
    out, _store = compress_generate_request(_body([_content("user", part)]))
    assert out["contents"][0]["parts"][0] == part


def test_inline_repeat_is_elided_and_recovers_byte_exact(certified):
    raw = _png(1024, 1024)
    part = _inline_part(raw)
    contents = [
        _content("user", json.loads(json.dumps(part))),
        _content("model", {"text": "looking"}),
        _content("user", json.loads(json.dumps(part))),
    ]
    out, store = compress_generate_request(_body(contents))

    first = out["contents"][0]["parts"][0]
    assert "inlineData" in first, "first occurrence must survive"
    stub = out["contents"][2]["parts"][0]
    assert "text" in stub, "the repeat should have become a reference"
    assert "distil:image" in stub["text"]

    handle = stub["text"].split("handle=")[1].split()[0]
    recovered = json.loads(store.expand(handle))
    assert recovered == part["inlineData"], "expand did not return the original inlineData object"
    assert base64.b64decode(recovered["data"]) == raw, "image bytes did not round-trip"


def test_inline_distinct_images_are_never_elided(certified):
    a, b = _inline_part(_png(1024, 1024)), _inline_part(_png(512, 512))
    contents = [_content("user", a), _content("model", {"text": "x"}), _content("user", b)]
    out, _ = compress_generate_request(_body(contents))
    assert "inlineData" in out["contents"][0]["parts"][0]
    assert "inlineData" in out["contents"][2]["parts"][0], "a different image was wrongly elided"


def test_inline_verbatim_mode_never_elides(certified):
    part = _inline_part(_png(1024, 1024))
    contents = [
        _content("user", json.loads(json.dumps(part))),
        _content("user", json.loads(json.dumps(part))),
    ]
    out, _ = compress_generate_request(_body(contents), verbatim=True)
    assert all("inlineData" in c["parts"][0] for c in out["contents"])


def test_inline_tiny_images_are_left_alone(certified):
    tiny = _inline_part(b"\x89PNG" + b"x")
    contents = [_content("user", json.loads(json.dumps(tiny))) for _ in range(2)]
    out, _ = compress_generate_request(_body(contents))
    assert all("inlineData" in c["parts"][0] for c in out["contents"])


def test_inline_model_role_is_never_touched(certified):
    """We never rewrite the model's own words/parts, images included."""
    part = _inline_part(_png(1024, 1024))
    contents = [
        _content("model", json.loads(json.dumps(part))),
        _content("model", json.loads(json.dumps(part))),
    ]
    out, _ = compress_generate_request(_body(contents))
    assert all("inlineData" in c["parts"][0] for c in out["contents"])


def test_malformed_inline_data_passes_through(certified):
    weird = [
        {"inlineData": "not-a-dict"},
        {"inlineData": {}},
        {"inlineData": None},
    ]
    for w in weird:
        out, _ = compress_generate_request(_body([_content("user", w)]))
        assert out["contents"][0]["parts"][0] == w


def test_disabled_without_a_certificate(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_VISION", raising=False)
    monkeypatch.setattr(vision, "_shipped_certificate_path", lambda: tmp_path / "absent.json")
    part = _inline_part(_png(1024, 1024))
    contents = [_content("user", part), _content("user", json.loads(json.dumps(part)))]
    body = _body(contents)
    out, _ = compress_generate_request(body)
    assert out is body, "images were touched with no certificate present"


def test_inline_census_is_exhaustive_for_images(certified):
    from distil.adapters.anthropic import take_census

    raw = _png(1024, 1024)
    part = _inline_part(raw)
    contents = [
        _content("user", json.loads(json.dumps(part))),
        _content("model", {"text": "prose"}),
        _content("user", json.loads(json.dumps(part))),
    ]
    compress_generate_request(_body(contents))
    census = take_census()
    payload = count_tokens(_body(contents))
    total = sum(census.values())
    assert abs(total - payload) / max(payload, 1) < 0.005, (
        f"census {total} != payload {payload}, buckets={census}"
    )
    assert census.get("image_elided", 0) > 0


# ---------------------------------------------------------------------------
# fileData
# ---------------------------------------------------------------------------


def test_file_data_with_a_data_uri_is_deduped(certified):
    """fileUri carrying an embedded data: URI is legitimate proof of identity."""
    raw = _png(1024, 1024)
    uri = f"data:image/png;base64,{_b64(raw)}"
    part = _file_part(uri)
    contents = [
        _content("user", json.loads(json.dumps(part))),
        _content("model", {"text": "x"}),
        _content("user", json.loads(json.dumps(part))),
    ]
    out, store = compress_generate_request(_body(contents))
    assert "fileData" in out["contents"][0]["parts"][0]
    stub = out["contents"][2]["parts"][0]
    assert "text" in stub
    handle = stub["text"].split("handle=")[1].split()[0]
    recovered = json.loads(store.expand(handle))
    assert recovered == {"url": uri}


def test_file_data_plain_url_is_never_deduped(certified):
    """A bare (non-data) fileUri is not proof of identical bytes."""
    part = _file_part("https://files.example.com/shot.png")
    contents = [_content("user", json.loads(json.dumps(part))) for _ in range(2)]
    out, _ = compress_generate_request(_body(contents))
    assert all("fileData" in c["parts"][0] for c in out["contents"])


def test_malformed_file_data_passes_through(certified):
    weird = [
        {"fileData": {"mimeType": "image/png"}},  # no fileUri
        {"fileData": {"fileUri": 12345}},  # wrong type
    ]
    for w in weird:
        out, _ = compress_generate_request(_body([_content("user", w)]))
        assert out["contents"][0]["parts"][0] == w


def test_count_tokens_counts_inline_and_file_data(certified):
    raw = _png(1024, 1024)
    inline_body = _body([_content("user", _inline_part(raw))])
    assert count_tokens(inline_body) >= 1000, "inline image counted as ~free"

    uri = f"data:image/png;base64,{_b64(raw)}"
    file_body = _body([_content("user", _file_part(uri))])
    assert count_tokens(file_body) >= 1000, "fileData image counted as ~free"


# ---------------------------------------------------------------------------
# Cache contract: elision must be byte-stable across separate calls (turns).
# ---------------------------------------------------------------------------


def test_inline_elision_is_byte_stable_across_turns(certified):
    raw = _png(1024, 1024)
    part = _inline_part(raw)
    contents = [
        _content("user", json.loads(json.dumps(part))),
        _content("model", {"text": "x"}),
        _content("user", json.loads(json.dumps(part))),
    ]
    out1, _ = compress_generate_request(_body([json.loads(json.dumps(c)) for c in contents]))
    out2, _ = compress_generate_request(_body([json.loads(json.dumps(c)) for c in contents]))
    assert out1["contents"][2]["parts"][0]["text"] == out2["contents"][2]["parts"][0]["text"]
