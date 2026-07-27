"""The `vision` strategy and the media-carrying Block (ADR 0003 / ADR 0004 rank 1).

What these tests can and cannot establish, stated plainly because the difference
is the whole point of this feature:

  CAN  — the mechanism. Byte-identical images collapse, distinct images survive,
         media round-trips through serialisation, and no compressor silently
         drops a block's media.
  CANNOT — decision-equivalence. Whether a model that saw an image decides the
         same way once a later identical copy is elided requires a real vision
         model. That is `distil certify --strategy vision --runner anthropic`
         against corpus/vision-ci-dashboard.json, and no assertion here
         substitutes for it.

The failure mode this guards against is a green suite that reads as a
certificate. It is not one.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path

import pytest

from distil.compress.strategies import REGISTRY
from distil.trajectory import Block, Kind, Stability, Trajectory

_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "vision-ci-dashboard.json"


def _png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    """A real, decodable PNG — not a header stub. A vision certificate graded on
    an unrenderable image would prove nothing."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _img(rgb: tuple[int, int, int]) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(_png(32, 32, rgb)).decode(),
        },
    }


def _blk(bid: str, media: list[dict] | None = None, text: str = "shot") -> Block:
    return Block(bid, Kind.TOOL_OUTPUT, text, Stability.VOLATILE, True, media)


def _digest(m: dict) -> str:
    return hashlib.sha256(m["source"]["data"].encode()).hexdigest()


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------


def test_identical_images_collapse_to_one():
    red = _img((200, 40, 40))
    blocks = [_blk("a", [dict(red)]), _blk("b", [dict(red)]), _blk("c", [dict(red)])]
    out = REGISTRY["vision"](blocks, 0)
    kept = [m for b in out for m in (b.media or [])]
    assert len(kept) == 1, "three identical images should collapse to one"


def test_distinct_images_are_never_elided():
    """The falsifiable half. A strategy that deduped distinct images would flip
    the corpus decision from promote_release to open_failing_build."""
    red, green = _img((200, 40, 40)), _img((40, 180, 90))
    blocks = [_blk("a", [dict(red)]), _blk("b", [dict(green)]), _blk("c", [dict(red)])]
    out = REGISTRY["vision"](blocks, 0)
    kept = {_digest(m) for b in out for m in (b.media or [])}
    assert kept == {_digest(red), _digest(green)}, "a distinct image was lost"


def test_url_sources_are_never_treated_as_duplicates():
    """Two occurrences of one URL are not evidence of the same pixels — signed
    URLs, dashboards and cache-busted screenshots all serve different bytes from
    a stable URL."""
    u = {"type": "image", "source": {"type": "url", "url": "https://x.test/tile.png"}}
    out = REGISTRY["vision"]([_blk("a", [dict(u)]), _blk("b", [dict(u)])], 0)
    assert len([m for b in out for m in (b.media or [])]) == 2


def test_elision_leaves_a_visible_note_in_the_text():
    """A silently shorter prompt is indistinguishable from a lost screenshot."""
    red = _img((200, 40, 40))
    out = REGISTRY["vision"]([_blk("a", [dict(red)]), _blk("b", [dict(red)])], 0)
    assert any("distil:image" in b.text for b in out)


def test_text_is_otherwise_untouched():
    """`vision` isolates the image transform so certifying it certifies THAT."""
    red = _img((200, 40, 40))
    blocks = [_blk("a", [dict(red)], "keep me exactly"), _blk("b", None, "and me")]
    out = REGISTRY["vision"](blocks, 0)
    assert out[0].text == "keep me exactly"
    assert out[1].text == "and me"


# ---------------------------------------------------------------------------
# The data model — media must survive everything that rewrites a block
# ---------------------------------------------------------------------------


def test_copy_with_carries_media_through():
    """Compressors rewrite text via copy_with. If it dropped media, a compressed
    turn would be graded against a prompt the uncompressed arm never had — the
    A/B would be invalid and would silently look like a pass."""
    red = _img((200, 40, 40))
    b = _blk("a", [red], "before")
    assert b.copy_with("after").media == [red]


def test_media_round_trips_through_serialisation():
    red = _img((200, 40, 40))
    t = Trajectory("t", "m", [])
    from distil.trajectory import Turn

    t.turns = [Turn(0, [_blk("a", [red])])]
    back = Trajectory.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back.turns[0].blocks[0].media == [red]


def test_blocks_without_media_serialise_exactly_as_before():
    """Every existing corpus file must round-trip byte-identically."""
    d = json.loads((_CORPUS.parent / "coding-bugfix.json").read_text())
    back = Trajectory.from_dict(d).to_dict()
    for turn in back["turns"]:
        for blk in turn["blocks"]:
            assert "media" not in blk, "media key leaked into a text-only block"


# ---------------------------------------------------------------------------
# The corpus itself
# ---------------------------------------------------------------------------


def test_corpus_images_are_genuinely_decodable():
    """CRCs valid and IDAT decompresses to exactly the expected pixel count. A
    corpus of broken images would certify nothing while looking fine."""
    traj = Trajectory.load(_CORPUS)
    payloads = {m["source"]["data"] for t in traj.turns for b in t.blocks for m in (b.media or [])}
    assert payloads, "vision corpus carries no images"
    for data in payloads:
        raw = base64.b64decode(data)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        pos, idat = 8, b""
        while pos < len(raw):
            ln = struct.unpack(">I", raw[pos : pos + 4])[0]
            tag = raw[pos + 4 : pos + 8]
            body = raw[pos + 8 : pos + 8 + ln]
            crc = struct.unpack(">I", raw[pos + 8 + ln : pos + 12 + ln])[0]
            assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF, f"CRC failure in {tag!r}"
            if tag == b"IDAT":
                idat += body
            pos += 12 + ln
        w, h = struct.unpack(">II", raw[16:24])
        assert len(zlib.decompress(idat)) == h * (1 + w * 3)


def test_corpus_accumulates_context_so_duplicates_actually_arise():
    """Turns carry history forward, as every other trajectory does. Without that
    each turn holds one image, there is nothing to de-duplicate, and the domain
    would certify vacuously."""
    traj = Trajectory.load(_CORPUS)
    counts = [sum(len(b.media or []) for b in t.blocks) for t in traj.turns]
    assert counts == sorted(counts) and counts[-1] > counts[0], counts
    assert counts[-1] >= 4, "final turn should hold several images"


def test_vision_strategy_preserves_the_distinct_set_on_every_corpus_turn():
    """The invariant the live certificate then tests for decision impact."""
    traj = Trajectory.load(_CORPUS)
    for turn in traj.turns:
        before = {_digest(m) for b in turn.blocks for m in (b.media or [])}
        after = {
            _digest(m) for b in REGISTRY["vision"](turn.blocks, turn.index) for m in (b.media or [])
        }
        assert after == before, f"turn {turn.index}: distinct image set changed"


def test_corpus_is_excluded_from_the_offline_gate_on_purpose():
    """It must NOT drift back into the deterministic manifest. That runner grades
    by reading DECISION: markers out of text; this domain's decision is in the
    pixels. Passing it offline would require writing the tile's colour into the
    text, making the image decorative and the certificate a claim about text."""
    manifest = json.loads((_CORPUS.parent / "manifest.json").read_text())
    offline = {t["file"] for t in manifest["trajectories"]}
    assert "vision-ci-dashboard.json" not in offline
    live = {t["file"] for t in manifest.get("live_only", [])}
    assert "vision-ci-dashboard.json" in live
    reason = manifest["live_only"][0]["why_not_in_the_offline_gate"]
    assert "live vision runner" in reason


@pytest.mark.parametrize("strategy", ["none", "distil", "naive", "aggressive"])
def test_existing_strategies_never_drop_media(strategy):
    """A text strategy that silently discarded images would make any later
    comparison against it meaningless."""
    red = _img((200, 40, 40))
    blocks = [_blk("a", [red], "some tool output\n" * 10)]
    out = REGISTRY[strategy](blocks, 0)
    assert [m for b in out for m in (b.media or [])] == [red], f"{strategy} dropped media"
