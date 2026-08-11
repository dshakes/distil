"""Downscaling images: off by default, inert without a codec, never silently lossy.

`vision.py` refuses downscaling because the model sees different pixels and nobody
can say whether the answer changed. This transform is allowed to exist only because
it changes that situation: the original goes into the RestoreStore and the
replacement carries a handle, which puts it in the same category as the Tier-1 text
digest rather than the category vision.py rejects.

Every test here defends one leg of that argument. If any of them fails, the feature
has become the thing the module docstring says it is not.
"""

from __future__ import annotations

import base64
import io
import json

import pytest

from distil.compress import vision_scale as vs

try:  # the codec is an optional extra; most of this file runs without it
    from PIL import Image

    HAVE_PIL = True
except ImportError:  # pragma: no cover - exercised on installs without [image]
    HAVE_PIL = False

needs_pil = pytest.mark.skipif(not HAVE_PIL, reason="requires the optional [image] extra")


def _png(width: int, height: int) -> bytes:
    assert HAVE_PIL
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _source(raw: bytes) -> dict:
    return {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(raw).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# The gate — the entire argument for letting the transform exist
# ---------------------------------------------------------------------------


def test_it_ships_disabled(monkeypatch, tmp_path) -> None:
    """No certificate, no downscaling. This is the default a user gets."""
    monkeypatch.delenv("DISTIL_VISION_DOWNSCALE", raising=False)
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    assert vs.enabled() is False


def test_no_bundled_certificate_is_inherited(monkeypatch, tmp_path) -> None:
    """`vision` ships a maintainer certificate; this deliberately does not.

    Byte-identical elision is provably the same image, so one corpus can speak
    for another. Losing pixel detail is a claim about whether *your* screenshots
    still drive the same decisions, which nobody else's corpus can answer.
    """
    monkeypatch.delenv("DISTIL_VISION_DOWNSCALE", raising=False)
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from pathlib import Path as _P

    shipped = _P(vs.__file__).resolve().parent.parent / "certificates"
    assert not (shipped / "vision-downscale.json").exists(), (
        "a bundled certificate would switch pixel loss on for every user by default"
    )
    assert vs.enabled() is False


def test_a_local_passing_certificate_enables_it(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DISTIL_VISION_DOWNSCALE", raising=False)
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    cert = tmp_path / "certificates" / "vision-downscale.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(json.dumps({"strategy": "vision-downscale", "non_inferior": True}))
    assert vs.enabled() is True


def test_a_certificate_for_another_strategy_does_not_count(monkeypatch, tmp_path) -> None:
    """The elision certificate must not silently authorise downscaling."""
    monkeypatch.delenv("DISTIL_VISION_DOWNSCALE", raising=False)
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    cert = tmp_path / "certificates" / "vision-downscale.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(json.dumps({"strategy": "vision", "non_inferior": True}))
    assert vs.enabled() is False


def test_a_stated_failure_keeps_it_off(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DISTIL_VISION_DOWNSCALE", raising=False)
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    cert = tmp_path / "certificates" / "vision-downscale.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(json.dumps({"strategy": "vision-downscale", "non_inferior": False}))
    assert vs.enabled() is False


def test_enabled_without_a_codec_is_not_active(monkeypatch) -> None:
    """An enabled feature with no codec must not half-work."""
    monkeypatch.setenv("DISTIL_VISION_DOWNSCALE", "1")
    monkeypatch.setattr(vs, "available", lambda: False)
    assert vs.enabled() is True
    assert vs.active() is False, "active() must require BOTH the gate and the codec"


# ---------------------------------------------------------------------------
# The transform itself
# ---------------------------------------------------------------------------


@needs_pil
def test_a_large_image_is_reduced_to_the_ceiling() -> None:
    out = vs.downscale(_png(2400, 1200), max_edge=1024)
    assert out is not None
    _data, (w, h) = out
    assert max(w, h) == 1024
    assert w / h == pytest.approx(2.0, rel=0.02), "aspect ratio must be preserved"


@needs_pil
def test_a_small_image_is_left_alone() -> None:
    """Below the threshold the note costs about what the resize saves, and a
    downscaled icon is worse than useless."""
    assert vs.downscale(_png(800, 600)) is None


def test_an_undecodable_payload_is_left_alone() -> None:
    """A corrupt image must never take down a request."""
    assert vs.downscale(b"not an image at all") is None


@needs_pil
def test_a_saving_too_small_to_justify_detail_loss_is_declined() -> None:
    monkeypatchless = _source(_png(1250, 40))  # over the edge threshold, tiny area
    assert vs.plan(monkeypatchless) is None


def test_a_url_source_is_never_touched() -> None:
    """distil does not hold those bytes and does not fetch them."""
    assert vs.plan({"type": "url", "url": "https://example.com/x.png"}) is None


# ---------------------------------------------------------------------------
# Not silently lossy — the property the whole design rests on
# ---------------------------------------------------------------------------


def test_the_note_states_the_change_and_the_way_back() -> None:
    text = vs.note_text("abcd1234", (2400, 1200), (1024, 512), 900)
    assert "2400x1200" in text and "1024x512" in text, "the magnitude must be stated"
    assert "handle=abcd1234" in text, "the way back must be in the house grammar"
    assert "distil_expand" in text


@needs_pil
def test_the_adapter_emits_the_image_and_its_note_as_a_pair(monkeypatch) -> None:
    """End to end: a downscaled image is never emitted alone.

    Alone it would be a silent alteration — indistinguishable, to the model, from
    an image that was always that size. The note is what makes the loss visible
    and recoverable, so the pair is the unit.
    """
    monkeypatch.setenv("DISTIL_VISION", "1")
    monkeypatch.setenv("DISTIL_VISION_DOWNSCALE", "1")
    from distil.adapters import anthropic as ad

    raw = _png(2400, 1200)
    # The image must sit OUTSIDE the recency window: distil never alters the
    # freshest turns, so a lone message is protected and nothing would happen.
    msgs = [
        {"role": "user", "content": [{"type": "image", "source": _source(raw)}]},
        *(
            m
            for _ in range(6)
            for m in (
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                {"role": "user", "content": [{"type": "text", "text": "next"}]},
            )
        ),
    ]
    out, store = ad.compress_messages(msgs)

    blocks = out[0]["content"]
    assert len(blocks) == 2, f"expected an image+note pair, got {[b.get('type') for b in blocks]}"
    assert blocks[0]["type"] == "image"
    assert blocks[1]["type"] == "text" and "distil_expand" in blocks[1]["text"]

    # The image actually shrank...
    shrunk = base64.b64decode(blocks[0]["source"]["data"])
    assert len(shrunk) < len(raw)

    # ...and the ORIGINAL is recoverable byte-for-byte through the handle.
    handle = blocks[1]["text"].split("handle=")[1].split(" ")[0].rstrip(">").strip()
    recovered = json.loads(store.expand(handle))
    assert base64.b64decode(recovered["data"]) == raw, "the original was not recoverable"


@needs_pil
def test_disabled_is_byte_for_byte_the_old_behaviour(monkeypatch) -> None:
    """With the gate off the adapter must be indistinguishable from before."""
    monkeypatch.setenv("DISTIL_VISION", "1")
    monkeypatch.delenv("DISTIL_VISION_DOWNSCALE", raising=False)
    from distil.adapters import anthropic as ad

    src = _source(_png(2400, 1200))
    msgs = [
        {"role": "user", "content": [{"type": "image", "source": src}]},
        *(
            m
            for _ in range(6)
            for m in (
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                {"role": "user", "content": [{"type": "text", "text": "next"}]},
            )
        ),
    ]
    out, _ = ad.compress_messages(msgs)
    blocks = out[0]["content"]
    assert len(blocks) == 1 and blocks[0]["source"]["data"] == src["data"]


@needs_pil
def test_the_freshest_turn_is_never_downscaled(monkeypatch) -> None:
    """Recency outranks the saving, and it does so for pixels too.

    distil never makes an agent reason blind over its most recent input. That
    rule predates this transform and applies to it automatically — this test
    exists because that was discovered by accident rather than by design, and an
    inherited guarantee nobody asserts is one refactor from being lost.
    """
    monkeypatch.setenv("DISTIL_VISION", "1")
    monkeypatch.setenv("DISTIL_VISION_DOWNSCALE", "1")
    from distil.adapters import anthropic as ad

    src = _source(_png(2400, 1200))
    msgs = [{"role": "user", "content": [{"type": "image", "source": src}]}]
    out, _ = ad.compress_messages(msgs)
    blocks = out[0]["content"]
    assert len(blocks) == 1, "the newest turn's image was altered"
    assert blocks[0]["source"]["data"] == src["data"], "byte-identical, or recency is broken"


@needs_pil
def test_a_screenshot_inside_a_tool_result_is_spliced_not_nested(monkeypatch) -> None:
    """The path that actually carries screenshots, and the one that shipped broken.

    Computer-use and browser tools return images inside a tool_result's content
    list. That loop assembles its own list, so the image+note pair has to be
    spliced there too — appending it whole nests a list inside `content` and the
    provider rejects the request. Every block in `content` must be a dict.
    """
    monkeypatch.setenv("DISTIL_VISION", "1")
    monkeypatch.setenv("DISTIL_VISION_DOWNSCALE", "1")
    from distil.adapters import anthropic as ad

    raw = _png(2400, 1200)
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "image", "source": _source(raw)}],
                }
            ],
        },
        *(
            m
            for _ in range(6)
            for m in (
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                {"role": "user", "content": [{"type": "text", "text": "next"}]},
            )
        ),
    ]
    out, store = ad.compress_messages(msgs)
    inner = out[0]["content"][0]["content"]

    assert all(isinstance(b, dict) for b in inner), (
        f"a list was nested inside tool_result content — malformed request: {inner}"
    )
    assert [b["type"] for b in inner] == ["image", "text"], (
        f"expected the image+note pair spliced in order, got {[b.get('type') for b in inner]}"
    )
    handle = inner[1]["text"].split("handle=")[1].split(" ")[0].rstrip(">").strip()
    assert base64.b64decode(json.loads(store.expand(handle))["data"]) == raw
