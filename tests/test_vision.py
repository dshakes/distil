"""Vision content type (ADR 0003) — reversible duplicate elision, gated.

The properties that matter, in order: it is OFF until certified, it never
touches a first occurrence, and anything it does elide comes back byte-exact.
"""

from __future__ import annotations

import base64
import json
import struct

import pytest

from distil.adapters.anthropic import compress_messages
from distil.compress import vision


# ---------------------------------------------------------------------------
# Fixtures: real file headers, built by hand so there is no image dependency
# ---------------------------------------------------------------------------


def _png(w: int, h: int, pad: int = 4096) -> bytes:
    ihdr = b"IHDR" + struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + ihdr
        + b"\x00\x00\x00\x00"
        + b"\x00" * pad  # bulk, so the payload clears MIN_B64_CHARS
    )


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _image_block(raw: bytes) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _b64(raw)},
    }


@pytest.fixture
def certified(tmp_path, monkeypatch):
    """A fleet where `distil certify --strategy vision` has passed."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_VISION", raising=False)
    cert = tmp_path / "certificates" / "vision.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(json.dumps({"strategy": "vision", "non_inferior": True}))
    return tmp_path


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_disabled_without_a_certificate(tmp_path, monkeypatch):
    """ADR 0003: a new content type stays disabled until SOMETHING certifies it.

    The package now ships the maintainer's live certificate, so the default is
    enabled — but the gate is unchanged: remove every certificate and the adapter
    is byte-for-byte what it was before this module existed. That is what this
    asserts, by stripping the shipped one too.
    """
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_VISION", raising=False)
    monkeypatch.setattr(vision, "_shipped_certificate_path", lambda: tmp_path / "absent.json")
    assert vision.enabled() is False

    img = _image_block(_png(1024, 1024))
    messages = [
        {"role": "user", "content": [img]},
        {"role": "user", "content": [json.loads(json.dumps(img))]},
    ]
    out, _store = compress_messages(messages)
    assert out == messages, "images were touched with no certificate present"


def test_env_can_force_enable_and_hard_disable(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_VISION", "1")
    assert vision.enabled() is True
    # An explicit 0 wins even when a certificate exists — the fleet escape hatch.
    cert = tmp_path / "certificates" / "vision.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text("{}")
    monkeypatch.setenv("DISTIL_VISION", "0")
    assert vision.enabled() is False


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


def test_first_occurrence_is_never_touched(certified):
    """The model has to actually see the image at least once."""
    img = _image_block(_png(800, 600))
    out, _ = compress_messages([{"role": "user", "content": [img]}])
    assert out[0]["content"][0] == img


def test_repeat_is_elided_and_recovers_byte_exact(certified):
    raw = _png(1024, 1024)
    img = _image_block(raw)
    messages = [
        {"role": "user", "content": [json.loads(json.dumps(img))], "_t": 0},
        {"role": "assistant", "content": [{"type": "text", "text": "looking"}]},
        {"role": "user", "content": [json.loads(json.dumps(img))]},
        # trailing turns so the duplicate above is not inside the recency window
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "c"}]},
        {"role": "user", "content": [{"type": "text", "text": "d"}]},
    ]
    out, store = compress_messages(messages)

    assert out[0]["content"][0]["type"] == "image", "first occurrence must survive"
    stub = out[2]["content"][0]
    assert stub["type"] == "text", "the repeat should have become a reference"
    assert "distil:image" in stub["text"]

    handle = stub["text"].split("handle=")[1].split()[0]
    recovered = json.loads(store.expand(handle))
    assert recovered == img["source"], "expand did not return the exact original source"
    assert base64.b64decode(recovered["data"]) == raw, "image bytes did not round-trip"


def test_distinct_images_are_never_elided(certified):
    a, b = _image_block(_png(1024, 1024)), _image_block(_png(512, 512))
    messages = [
        {"role": "user", "content": [a]},
        {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
        {"role": "user", "content": [b]},
        {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
        {"role": "user", "content": [{"type": "text", "text": "z"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "w"}]},
        {"role": "user", "content": [{"type": "text", "text": "v"}]},
    ]
    out, _ = compress_messages(messages)
    assert out[0]["content"][0]["type"] == "image"
    assert out[2]["content"][0]["type"] == "image", "a different image was wrongly elided"


def test_verbatim_mode_never_elides(certified):
    img = _image_block(_png(1024, 1024))
    messages = [
        {"role": "user", "content": [json.loads(json.dumps(img))]},
        {"role": "user", "content": [json.loads(json.dumps(img))]},
    ]
    out, _ = compress_messages(messages, verbatim=True)
    assert all(m["content"][0]["type"] == "image" for m in out)


def test_screenshot_inside_a_tool_result_is_handled(certified):
    """Computer-use / browser screenshots arrive as an image block nested in a
    tool_result content list — the case this feature exists for."""
    raw = _png(1024, 1024)

    def tr():
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [_image_block(raw)]}
            ],
        }

    messages = [
        tr(),
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        tr(),
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "c"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "d"}]},
        {"role": "user", "content": [{"type": "text", "text": "e"}]},
    ]
    out, store = compress_messages(messages)
    first = out[0]["content"][0]["content"][0]
    second = out[2]["content"][0]["content"][0]
    assert first["type"] == "image"
    assert second["type"] == "text", "nested duplicate screenshot was not elided"
    handle = second["text"].split("handle=")[1].split()[0]
    assert base64.b64decode(json.loads(store.expand(handle))["data"]) == raw


def test_tiny_images_are_left_alone(certified):
    """An icon is cheaper inline than the stub that would replace it."""
    tiny = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _b64(b"\x89PNG" + b"x")},
    }
    messages = [
        {"role": "user", "content": [json.loads(json.dumps(tiny))]},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [json.loads(json.dumps(tiny))]},
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "c"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "d"}]},
        {"role": "user", "content": [{"type": "text", "text": "e"}]},
    ]
    out, _ = compress_messages(messages)
    assert all(
        blk["type"] == "image" for m in out for blk in m["content"] if blk.get("type") == "image"
    )
    assert out[2]["content"][0]["type"] == "image"


def test_malformed_blocks_pass_through(certified):
    """Never raise on a shape we do not recognize."""
    weird = [
        {"type": "image"},  # no source
        {"type": "image", "source": "not-a-dict"},
        {"type": "image", "source": {}},  # no data, no url
        {"type": "image", "source": {"data": 123}},
    ]
    out, _ = compress_messages([{"role": "user", "content": weird}])
    assert out[0]["content"] == weird


# ---------------------------------------------------------------------------
# Dimension parsing — savings accounting depends on it, so it is asserted
# ---------------------------------------------------------------------------


def test_image_dims_reads_real_headers():
    assert vision.image_dims(_png(1024, 768)) == (1024, 768)

    gif = b"GIF89a" + struct.pack("<HH", 640, 480)
    assert vision.image_dims(gif) == (640, 480)

    # JPEG: SOI, a APP0 segment to be skipped, then SOF0 carrying h,w.
    jpeg = (
        b"\xff\xd8"
        + b"\xff\xe0"
        + struct.pack(">H", 4)
        + b"\x00\x00"
        + b"\xff\xc0"
        + struct.pack(">H", 11)
        + b"\x08"
        + struct.pack(">HH", 300, 400)
        + b"\x03\x00\x00\x00"
    )
    assert vision.image_dims(jpeg) == (400, 300)

    webp = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8 " + b"\x00" * 10 + struct.pack("<HH", 200, 100)
    assert vision.image_dims(webp) == (200, 100)


def test_image_dims_returns_none_on_junk():
    for junk in (b"", b"not an image", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff" + b"\xff" * 20):
        assert vision.image_dims(junk) is None


def test_estimate_tokens_is_conservative_and_capped():
    assert vision.estimate_tokens(_png(1024, 1024)) == (1024 * 1024) // 750  # ~1398
    # Cap only bites well above that — a huge image cannot bill unbounded.
    assert vision.estimate_tokens(_png(4000, 4000)) == vision._MAX_IMAGE_TOKENS
    assert vision.estimate_tokens(_png(100, 100)) == (100 * 100) // 750
    # Unknown size must not be reported as free.
    assert vision.estimate_tokens(b"junk") == vision._UNKNOWN_SIZE_TOKENS
    assert vision.estimate_tokens(None) == vision._UNKNOWN_SIZE_TOKENS


def test_dedup_counts_what_it_saved():
    d = vision.ImageDedup()
    src = {"type": "base64", "media_type": "image/png", "data": _b64(_png(1024, 1024))}
    assert d.elide(src) is None  # first
    verdict = d.elide(json.loads(json.dumps(src)))
    assert verdict is not None
    _handle, _original, tokens = verdict
    assert tokens == (1024 * 1024) // 750
    assert d.elided == 1 and d.tokens_saved == tokens
    d.reset()
    assert d.elided == 0 and d.tokens_saved == 0


# ---------------------------------------------------------------------------
# Header-parser edge cases. A screenshot payload is attacker-influenced in a
# browser-automation agent, so every one of these must return rather than raise.
# ---------------------------------------------------------------------------


def test_jpeg_parser_survives_hostile_and_unusual_segment_chains():
    # A segment that under-points leaves the scanner on a non-FF byte; it must
    # walk forward to the next marker instead of giving up. (The SOI magic is
    # always FF D8 FF, so the filler has to sit after the first segment.)
    padded = (
        b"\xff\xd8\xff\xe0"
        + struct.pack(">H", 2)  # empty APP0
        + b"\x00"  # filler the scanner must step over
        + b"\xff\xc2"  # SOF2 (progressive) — still a frame header
        + struct.pack(">H", 11)
        + b"\x08"
        + struct.pack(">HH", 120, 240)
        + b"\x03\x00\x00\x00"
    )
    assert vision.image_dims(padded) == (240, 120)

    # A restart/standalone marker carries no length field — must not be read as one.
    standalone = (
        b"\xff\xd8"
        + b"\xff\xd0"  # RST0
        + b"\xff\xc0"
        + struct.pack(">H", 11)
        + b"\x08"
        + struct.pack(">HH", 60, 80)
        + b"\x03\x00\x00\x00"
    )
    assert vision.image_dims(standalone) == (80, 60)

    # A zero segment length would loop forever if trusted.
    looping = b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 0) + b"\x00" * 40
    assert vision.image_dims(looping) is None

    # DHT (0xC4) sits inside the SOF range but is NOT a frame header.
    not_a_frame = b"\xff\xd8" + b"\xff\xc4" + struct.pack(">H", 4) + b"\x00\x00" + b"\x00" * 20
    assert vision.image_dims(not_a_frame) is None


def test_webp_lossless_and_extended_variants():
    vp8l = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8L" + b"\x00" * 5 + struct.pack("<I", 0)
    assert vision.image_dims(vp8l + b"\x00" * 10) == (1, 1)

    vp8x = (
        b"RIFF"
        + b"\x00" * 4
        + b"WEBP"
        + b"VP8X"
        + b"\x00" * 8
        + (299).to_bytes(3, "little")
        + (199).to_bytes(3, "little")
    )
    assert vision.image_dims(vp8x) == (300, 200)

    # A RIFF container that is not WEBP, and an unknown WEBP chunk.
    assert vision.image_dims(b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 30) is None
    assert vision.image_dims(b"RIFF" + b"\x00" * 4 + b"WEBP" + b"XXXX" + b"\x00" * 30) is None


def test_zero_dimension_image_is_not_reported_as_free():
    assert vision.estimate_tokens(_png(0, 0)) == vision._UNKNOWN_SIZE_TOKENS


def test_enabled_survives_an_unreadable_home(monkeypatch, tmp_path):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_VISION", raising=False)

    def _boom(self, *args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(vision.Path, "read_text", _boom)
    # The property under test is that an unreadable home does not PROPAGATE, not
    # what the gate then decides — with both sources unreadable it must simply be
    # a clean False rather than an exception reaching the request path.
    monkeypatch.setattr(vision, "_shipped_certificate_path", lambda: tmp_path / "absent.json")
    assert vision.enabled() is False


def test_url_sources_are_never_deduped():
    """Two occurrences of the same URL are NOT evidence of the same image.

    Dashboards, signed URLs, cache-busted screenshots and auth-dependent
    resources all return different pixels from a stable URL. Keying on the URL
    would let a genuinely different second image be replaced by a stub asserting
    it is identical — and expand would then hand back the FIRST image's bytes.
    That is the reversibility contract broken, silently, so the URL case is
    refused outright rather than approximated.
    """
    d = vision.ImageDedup(min_b64_chars=4)
    src = {"type": "url", "url": "https://example.com/" + "a" * 40 + ".png"}
    assert d.elide(src) is None
    assert d.elide(dict(src)) is None, "a url repeat was elided without proof of identity"
    assert d.elided == 0


def test_unrecognized_and_oversized_stub_sources_are_left_alone():
    d = vision.ImageDedup()
    assert d.elide({"type": "base64"}) is None  # no data, no url
    assert d.elide({"data": 12345}) is None  # non-string payload

    # Reject-if-bigger, in TOKENS: a 10x10 image is ~1 token, the stub is ~50.
    # Replacing it would cost more than leaving it inline.
    small = vision.ImageDedup(min_b64_chars=1)
    src = {"data": _b64(_png(10, 10, pad=0))}
    assert small.elide(src) is None  # first sighting
    assert small.elide(dict(src)) is None, "a stub costlier than the image must be refused"
    assert small.elided == 0


def test_decode_b64_never_raises_on_garbage():
    assert vision._decode_b64(None) is None
    assert vision._decode_b64("") is None
    assert vision._decode_b64(12345) is None
    assert vision._decode_b64("!!!not base64!!!") is not None or True  # must not raise


# ---------------------------------------------------------------------------
# The certificate gate must actually read the certificate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload,why",
    [
        ("{}", "an empty object certifies nothing"),
        ('{"strategy": "vision"}', "named the strategy but carried no verdict"),
        ('{"strategy": "vision", "non_inferior": false}', "an explicit FAIL"),
        ('{"strategy": "tier1", "non_inferior": true}', "a certificate for another strategy"),
        ('{"strategy": "vision", "non_inferior": "yes"}', "a truthy string is not true"),
        ('{"strategy": "vision", "non_inferior": tru', "a truncated/interrupted write"),
        ("[]", "valid JSON, wrong shape"),
        ("", "an empty file"),
    ],
)
def test_gate_fails_closed_on_anything_short_of_a_real_certificate(
    tmp_path, monkeypatch, payload, why
):
    """File existence is not certification.

    The gate is the entire argument for letting a new content type exist, so it
    parses rather than stat()s. Failing closed costs compression; failing open
    puts an uncertified transform on live traffic.
    """
    cert = tmp_path / "certificates" / "vision.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(payload)
    # Asserted on the PARSER, not on enabled(): "this file does not certify" and
    # "the gate is closed" are different claims once a shipped certificate can be
    # fallen back to. The gate-level behaviour is pinned by the full state matrix
    # below.
    assert vision._certificate_is_valid(cert) is False, f"parser accepted {why}"


def test_gate_opens_only_on_a_real_certificate(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_VISION", raising=False)
    cert = tmp_path / "certificates" / "vision.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    for verdict in ("non_inferior", "certified", "passed"):
        cert.write_text(json.dumps({"strategy": "vision", verdict: True}))
        assert vision.enabled() is True, f"{verdict}=true should certify"


def test_gate_fails_closed_when_the_path_is_unreadable(tmp_path, monkeypatch):
    """A directory where the certificate should be, a permissions error, a
    vanished mount — none of these are a pass."""
    cert = tmp_path / "certificates" / "vision.json"
    cert.mkdir(parents=True)  # a DIRECTORY, not a file
    assert vision._certificate_is_valid(cert) is False


def test_proxy_counts_image_tokens_so_elision_scores_positive():
    """Savings telemetry must not go NEGATIVE on an elided image.

    Images bill by pixel area, but the proxy's counter only ever walked text
    keys — so an image contributed 0 to the "before" side while its replacement
    stub contributed real tokens to the "after" side. The feature would have
    reported itself as a loss.
    """
    from distil.proxy import _count_messages, _tokens_saved

    raw = _png(1024, 1024)
    before = [{"role": "user", "content": [_image_block(raw)]}]
    after = [
        {
            "role": "user",
            "content": [{"type": "text", "text": vision.reference_text("ab12cd34", 1398)}],
        }
    ]
    assert _count_messages(before) >= 1000, "image block counted as ~free"
    assert _tokens_saved(before, after) > 0, "eliding an image scored as no saving"

    # Nested in a tool_result — where screenshots actually arrive.
    nested = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": [_image_block(raw)]}
            ],
        }
    ]
    assert _count_messages(nested) >= 1000, "nested screenshot counted as ~free"


# ---------------------------------------------------------------------------
# The shipped (maintainer) certificate
# ---------------------------------------------------------------------------


def test_shipped_certificate_enables_vision_out_of_the_box(tmp_path, monkeypatch):
    """Certifying needs a live vision model and an API key, which most users of a
    stdlib-only library will never run. Shipping the maintainer's real result lets
    them inherit it rather than the feature being inert for everyone but its
    author. It is still a parsed certificate, not a skipped gate."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))  # no local certificate
    monkeypatch.delenv("DISTIL_VISION", raising=False)
    assert vision.enabled() is True
    assert vision._shipped_certificate_path().is_file()


def test_shipped_certificate_states_its_own_scope():
    """A certificate you inherit is worth what you can check about it. It must
    name the model, the corpus, the verdict, and how to reproduce it — otherwise
    'certified' is just a boolean someone typed."""
    doc = json.loads(vision._shipped_certificate_path().read_text())
    assert doc["strategy"] == "vision"
    assert doc["non_inferior"] is True
    for key in ("model", "trajectory", "runner", "match_rate", "reproduce", "scope"):
        assert doc.get(key), f"shipped certificate omits {key!r}"
    assert "certify --strategy vision" in doc["reproduce"]
    # It must NOT overclaim: the scope has to say this is not about your traffic.
    assert "does NOT certify your traffic" in doc["scope"]


def test_env_disable_beats_the_shipped_certificate(tmp_path, monkeypatch):
    """The escape hatch has to work even for a certificate the user did not
    install and cannot delete."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_VISION", "0")
    assert vision.enabled() is False


# ---------------------------------------------------------------------------
# The complete gate state space
#
# The fallback introduced a second certificate source, and the interaction of
# (env override x local verdict x shipped verdict) is exactly where a gate goes
# quietly wrong. Enumerated rather than sampled, so no combination is decided by
# accident and every cell is a stated intention.
# ---------------------------------------------------------------------------

_PASS = json.dumps({"strategy": "vision", "non_inferior": True})
_FAIL = json.dumps({"strategy": "vision", "non_inferior": False})
_CORRUPT = '{"strategy": "vision", "non_inferior": tru'
_OTHER = json.dumps({"strategy": "tier1", "non_inferior": True})
_MUTE = json.dumps({"strategy": "vision"})  # names it, states nothing


@pytest.mark.parametrize(
    "env,local,expected,rationale",
    [
        # An explicit operator decision outranks every certificate, both ways.
        ("0", None, False, "operator disabled it"),
        ("0", _PASS, False, "operator disable beats a local PASS"),
        ("1", None, True, "operator force-enabled, uncertified"),
        ("1", _FAIL, True, "operator override beats a local FAIL"),
        # No override: a local verdict is a decision about YOUR traffic and wins.
        (None, _PASS, True, "local PASS"),
        (
            None,
            _FAIL,
            False,
            "local FAIL must NOT fall through to the shipped PASS — that would let "
            "our corpus overrule your evidence about your own traffic",
        ),
        # No local VERDICT is an absence, not a failure: fall through.
        (None, None, True, "no local certificate -> shipped"),
        (None, _CORRUPT, True, "corrupt local file is an absence, not a FAIL"),
        (None, _OTHER, True, "a certificate for another strategy says nothing here"),
        (None, _MUTE, True, "names the strategy but states no verdict"),
    ],
)
def test_gate_state_matrix(tmp_path, monkeypatch, env, local, expected, rationale):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    if env is None:
        monkeypatch.delenv("DISTIL_VISION", raising=False)
    else:
        monkeypatch.setenv("DISTIL_VISION", env)
    if local is not None:
        cert = tmp_path / "certificates" / "vision.json"
        cert.parent.mkdir(parents=True, exist_ok=True)
        cert.write_text(local)
    assert vision.enabled() is expected, rationale


def test_gate_closes_completely_when_no_certificate_exists_anywhere(tmp_path, monkeypatch):
    """The pre-certification guarantee must still hold: strip BOTH sources and
    the adapter is byte-for-byte what it was before this feature existed."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_VISION", raising=False)
    monkeypatch.setattr(vision, "_shipped_certificate_path", lambda: tmp_path / "nope.json")
    assert vision.enabled() is False

    img = _image_block(_png(1024, 1024))
    messages = [
        {"role": "user", "content": [img]},
        {"role": "user", "content": [json.loads(json.dumps(img))]},
    ]
    out, _ = compress_messages(messages)
    assert out == messages


def test_verdict_is_tri_state_not_boolean(tmp_path):
    """The distinction the fallback depends on: absent/corrupt returns None (an
    absence, may fall through) while a stated failure returns False (a decision,
    may not)."""
    p = tmp_path / "c.json"
    assert vision._certificate_verdict(p) is None, "missing file"
    p.write_text(_CORRUPT)
    assert vision._certificate_verdict(p) is None, "corrupt file"
    p.write_text(_OTHER)
    assert vision._certificate_verdict(p) is None, "other strategy"
    p.write_text(_MUTE)
    assert vision._certificate_verdict(p) is None, "no verdict stated"
    p.write_text(_FAIL)
    assert vision._certificate_verdict(p) is False, "explicit FAIL"
    p.write_text(_PASS)
    assert vision._certificate_verdict(p) is True, "explicit PASS"
