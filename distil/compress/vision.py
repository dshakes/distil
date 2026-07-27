"""Vision blocks — reversible duplicate elision, gated on a certificate.

ADR 0003 records the largest quantifiable coverage gap: distil compresses the
text *around* an image block and leaves the block itself at full cost. A
1024x1024 image is roughly 1,400 input tokens, and an agent that screenshots a
UI, re-reads a diagram, or polls a dashboard pays that on every turn the block
stays in context.

What this module does — and deliberately does not do
----------------------------------------------------
The state of the art here is **downscaling** or dropping the provider's detail
level. That is lossy by construction: the model sees a different image and
nobody can say whether the answer changed. distil's contract forbids it, so
this module implements the one transform that is *byte-reversible*:

    the SECOND and later appearances of a byte-identical image become a short
    text reference; the first appearance is left completely untouched.

The model still sees the image — once. The elided appearances carry an 8-hex
handle into the RestoreStore, so ``distil_expand`` returns the exact original
``source`` object. Nothing is approximated, downscaled, or re-encoded.

This is not a hypothetical duplicate. A UI-automation loop screenshots between
actions and the page frequently has not changed; a research agent re-attaches
the same figure while reasoning about it. Those repeats are byte-identical and
are pure waste.

Why it is still gated
---------------------
Removing an inline image and leaving a reference *does* change what the model
sees, even though the bytes are recoverable. Per ADR 0003 decision 1 that makes
it a new content type entering through the existing gate: it ships **disabled**
until ``distil certify --strategy vision`` reports non-inferior, exactly like
every other strategy. ``enabled()`` is False until a certificate exists.

Zero runtime dependencies: image dimensions are read from the file header with
the stdlib, so no Pillow, and an unparseable header degrades to "unknown size"
rather than raising.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

# Anthropic bills vision at roughly (width x height) / 750 input tokens, capped
# by the provider's own resize at ~1568px on the long edge. Used for savings
# accounting only — never for a correctness decision — so an unknown size falls
# back to a deliberately CONSERVATIVE estimate rather than an optimistic one.
_TOKENS_PER_PIXEL_DIVISOR = 750
_MAX_IMAGE_TOKENS = 1600
_UNKNOWN_SIZE_TOKENS = 750

# Below this, eliding is not worth it: the reference stub itself costs tokens,
# and a tiny icon may genuinely be cheaper left inline. Measured on the base64
# payload length, which is what we can always see.
MIN_B64_CHARS = 512

# Rough chars-per-token for the reference stub's own cost. Only used to compare
# the stub against the image it would replace, so a coarse constant is enough —
# distil never bills from this, and the calibrated tokenizer is not on this path.
_CHARS_PER_TOKEN = 4


def _certificate_path() -> Path:
    """Where a passing ``distil certify --strategy vision`` run records itself.

    Resolved at call time (not import) so ``DISTIL_HOME`` can be pointed at a
    temp dir by tests and by the certification run itself — same convention as
    the savings ledger.
    """
    home = Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))
    return home / "certificates" / "vision.json"


def _shipped_certificate_path() -> Path:
    """The maintainer-signed certificate bundled with the package.

    Certifying requires a live vision model and an API key, which most users of a
    stdlib-only library will never run. Shipping the maintainer's result lets them
    inherit a real, reproducible certificate instead of the feature being inert
    for everyone but its author.

    What they inherit is scoped, and the file says so: non-inferiority of
    byte-identical duplicate elision on the bundled vision trajectory against one
    named model, with the exact command to reproduce it. It is not a claim about
    their traffic. A local certificate takes precedence, so certifying your own
    workload overrides ours, and ``DISTIL_VISION=0`` disables regardless.
    """
    return Path(__file__).resolve().parent.parent / "certificates" / "vision.json"


def enabled() -> bool:
    """True only when this content type has been certified non-inferior.

    ADR 0003: coverage is a gap worth closing, the gate is not negotiable. With
    no certificate this returns False and the adapter's behavior is byte-for-byte
    what it was before this module existed.

    ``DISTIL_VISION=1`` force-enables for the certification run itself and for
    tests — it is an opt-IN to *uncertified* behavior, so it is deliberately not
    documented as a user-facing tuning knob. ``DISTIL_VISION=0`` hard-disables
    even when a certificate is present, which is the escape hatch if a fleet
    wants the old behavior back without deleting state.

    Resolution order, most specific first:

      1. ``DISTIL_VISION`` — an explicit operator decision, either way.
      2. A LOCAL certificate under ``DISTIL_HOME`` — you certified your own
         workload, so your result outranks ours.
      3. The certificate shipped with the package — the maintainer's live run on
         the bundled vision trajectory. Real and reproducible, but scoped to that
         corpus and model; the file states its own scope.

    Every one of them still has to parse and carry an explicit passing verdict.
    Inheriting a certificate is not the same as skipping the gate.
    """
    override = os.environ.get("DISTIL_VISION")
    if override is not None:
        return override.strip() not in ("", "0", "false", "no")

    local = _certificate_verdict(_certificate_path())
    if local is not None:
        # An explicit local verdict is a DECISION and wins either way. Critically
        # that includes a local FAIL: if you certified your own workload and it
        # did not pass, silently falling back to the shipped PASS would use our
        # corpus to overrule your evidence about your traffic.
        return local
    # No local verdict at all (absent, corrupt, truncated, wrong strategy) is an
    # ABSENCE, not a failure — fall through to the shipped certificate.
    return _certificate_verdict(_shipped_certificate_path()) is True


def _certificate_is_valid(path: Path) -> bool:
    """Whether *path* holds a certificate that actually certifies THIS strategy.

    File existence is not certification. An empty ``{}``, a truncated write, a
    half-finished run, or a certificate for some other strategy would all pass an
    ``is_file()`` check and silently switch on a transform the ADR says ships
    only after a non-inferiority result. The gate is the entire argument for
    letting a new content type exist at all, so it parses.

    Requires the certificate to name this strategy AND to carry an explicit
    passing verdict. Fails closed on anything it cannot read or does not
    understand — a missing key, a false verdict, malformed JSON, an unreadable
    file. The cost of failing closed is that compression stays off; the cost of
    failing open is an uncertified transform on live traffic.
    """
    return _certificate_verdict(path) is True


def _certificate_verdict(path: Path) -> bool | None:
    """Tri-state read of a certificate: True = passes, False = explicit FAIL,
    None = no readable verdict at all.

    The three states matter because a shipped certificate now exists to fall back
    on. Collapsing "explicitly failed" into "no verdict" would mean a user whose
    own certification run FAILED silently inherits our PASS — our corpus
    overruling their evidence about their own traffic. An unreadable file is an
    absence and may fall through; a stated failure may not.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:  # absent, unreadable, a directory — no verdict
        return None
    try:
        doc = json.loads(raw)
    except (ValueError, TypeError):
        return None  # truncated or corrupt write — no verdict, not a failure
    if not isinstance(doc, dict):
        return None
    if doc.get("strategy") != "vision":
        return None  # a certificate for something else says nothing about this
    keys = ("non_inferior", "certified", "passed")
    if any(doc.get(k) is True for k in keys):
        return True
    if any(doc.get(k) is False for k in keys):
        return False  # a stated FAIL — a decision, and it wins
    return None  # named the strategy but stated no verdict


# ---------------------------------------------------------------------------
# Dimension parsing — stdlib header reads, never a decode
# ---------------------------------------------------------------------------


def image_dims(raw: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG/JPEG/GIF/WEBP header, or None if unreadable.

    Reads only the header — it never decodes pixels, so a hostile or truncated
    payload costs bounded work. Returns None rather than raising on anything it
    does not recognize; the caller treats that as "unknown size".
    """
    try:
        if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
            # IHDR is mandated to be the first chunk: width/height at byte 16.
            w, h = struct.unpack(">II", raw[16:24])
            return (int(w), int(h))

        if raw[:3] == b"\xff\xd8\xff":
            # Walk the segment chain to a Start-Of-Frame marker.
            i, n = 2, len(raw)
            while i + 9 < n:
                if raw[i] != 0xFF:
                    i += 1
                    continue
                marker = raw[i + 1]
                # SOF0-SOF15 carry the dimensions; C4/C8/CC are not frame headers.
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">HH", raw[i + 5 : i + 9])
                    return (int(w), int(h))
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2  # standalone markers carry no length field
                    continue
                seg_len = struct.unpack(">H", raw[i + 2 : i + 4])[0]
                if seg_len < 2:
                    return None  # malformed length would loop forever
                i += 2 + seg_len
            return None

        if raw[:6] in (b"GIF87a", b"GIF89a") and len(raw) >= 10:
            w, h = struct.unpack("<HH", raw[6:10])
            return (int(w), int(h))

        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP" and len(raw) >= 30:
            fmt = raw[12:16]
            if fmt == b"VP8 ":
                w, h = struct.unpack("<HH", raw[26:30])
                return (int(w) & 0x3FFF, int(h) & 0x3FFF)
            if fmt == b"VP8L":
                bits = struct.unpack("<I", raw[21:25])[0]
                return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
            if fmt == b"VP8X":
                w = int.from_bytes(raw[24:27], "little") + 1
                h = int.from_bytes(raw[27:30], "little") + 1
                return (w, h)
        return None
    except (struct.error, IndexError, ValueError):
        return None


def estimate_tokens(raw: bytes | None) -> int:
    """Input tokens an image block costs, for savings accounting.

    Unknown dimensions fall back to a conservative constant: this number feeds
    the savings ledger, and distil's rule is that an unverifiable number is
    reported low rather than flattering.
    """
    dims = image_dims(raw) if raw else None
    if dims is None:
        return _UNKNOWN_SIZE_TOKENS
    w, h = dims
    if w <= 0 or h <= 0:
        return _UNKNOWN_SIZE_TOKENS
    return max(1, min(_MAX_IMAGE_TOKENS, (w * h) // _TOKENS_PER_PIXEL_DIVISOR))


# ---------------------------------------------------------------------------
# Duplicate elision
# ---------------------------------------------------------------------------


def _handle(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def reference_text(handle: str, tokens: int) -> str:
    """The stub that replaces a repeated image.

    Says three things the model needs: this is an image it has ALREADY been
    shown, it is byte-identical, and it is recoverable. Without the first two an
    agent can conclude the image was withheld and re-request a screenshot, which
    would cost more than the elision saved.

    Uses the house marker grammar (``<< … handle=XXXXXXXX >>``) that tier1 emits
    and ``distil_expand`` documents, so the handle is delimited exactly the way
    every other digest's is — a trailing period here would be parsed as part of
    the handle.
    """
    return (
        f"<< distil:image — identical to an image already shown above in this "
        f"conversation, not repeated to save ~{tokens} input tokens; "
        f"recover the original with distil_expand, handle={handle} >>"
    )


class ImageDedup:
    """Per-request memory of image payloads already emitted verbatim.

    Scoped to a single ``compress_messages`` call: the message list *is* the
    conversation history, so a duplicate within it is a duplicate the model is
    about to be billed for twice. Deliberately NOT persisted across requests —
    a later request re-sends the whole history, and its own first occurrence
    must stay verbatim or the model would never see the image at all.
    """

    def __init__(self, min_b64_chars: int = MIN_B64_CHARS) -> None:
        self.min_b64_chars = min_b64_chars
        self._seen: set[str] = set()
        self.elided = 0
        self.tokens_saved = 0

    def reset(self) -> None:
        self._seen.clear()
        self.elided = 0
        self.tokens_saved = 0

    def note(self, source: dict[str, Any]) -> None:
        """Record a source we are leaving verbatim, without eliding it."""
        key = self._key(source)
        if key is not None:
            self._seen.add(key)

    @staticmethod
    def _key(source: dict[str, Any]) -> str | None:
        """Identity of an image payload, or None if we cannot prove identity.

        ONLY inline base64 payloads are keyed, because only there do we hold the
        actual bytes. URL sources are deliberately excluded: two occurrences of
        the same URL are not evidence of the same image. Dashboards, signed
        URLs, `?t=` cache-busted screenshots and auth-dependent resources all
        return different pixels from a stable URL, so keying on the URL would
        let a *different* second image be replaced by a stub asserting it is
        identical — a false claim, and a silent one, since expand would return
        the first image's bytes.

        That is the reversibility contract broken, not merely a missed saving,
        so the URL case is refused rather than approximated. Deduping it safely
        would need a verified content digest of the fetched bytes, which means
        fetching them, which this module does not do.
        """
        data = source.get("data")
        if isinstance(data, str) and data:
            return f"b64:{data}"
        return None

    def elide(self, source: dict[str, Any]) -> tuple[str, str, int] | None:
        """Decide the fate of one image ``source``.

        Returns ``(handle, original_json, tokens_saved)`` when this exact payload
        has already been emitted verbatim and is worth eliding; returns None to
        leave the block untouched — which is the answer for a first occurrence,
        a too-small payload, or anything unrecognized.
        """
        key = self._key(source)
        if key is None:
            return None

        payload = key.split(":", 1)[1]
        if len(payload) < self.min_b64_chars:
            # Too small to be worth a stub; remember it so a later identical
            # copy is still recognized as seen rather than treated as first.
            self._seen.add(key)
            return None

        if key not in self._seen:
            self._seen.add(key)
            return None  # first sighting always goes through verbatim

        original = json.dumps(source, sort_keys=True)
        handle = _handle(original)
        raw = _decode_b64(source.get("data"))
        tokens = estimate_tokens(raw)

        # Reject-if-bigger, the same invariant every other compressor obeys: a
        # stub that costs more than what it replaces is not a compression.
        #
        # Compared in TOKENS, not characters. An image block is billed by pixel
        # area, not by the length of its base64 payload, so a character-length
        # comparison measures neither side correctly — and for a url source it is
        # nonsense in the expensive direction: the url text is short while the
        # image the model actually processes is not, so a plain length test
        # refuses every url dedup no matter how large the image.
        stub_tokens = max(1, len(reference_text(handle, tokens)) // _CHARS_PER_TOKEN)
        if tokens <= stub_tokens:
            return None

        self.elided += 1
        self.tokens_saved += tokens
        return handle, original, tokens


def _decode_b64(data: Any) -> bytes | None:
    """Best-effort base64 decode for the dimension read. Never raises."""
    if not isinstance(data, str) or not data:
        return None
    import base64
    import binascii

    try:
        # Only the header is needed, so decode a prefix — this bounds the work on
        # a multi-megabyte screenshot instead of materializing the whole image.
        # 4096 base64 chars is ~3 KB, well past PNG/GIF/WEBP headers and past the
        # JPEG SOF in ordinary files. A JPEG carrying a large EXIF thumbnail can
        # push its SOF beyond that; dimensions then read as unknown, which costs
        # a conservative token estimate, never a wrong one.
        prefix = data[:4096]
        prefix = prefix[: len(prefix) - (len(prefix) % 4)]
        return base64.b64decode(prefix, validate=False)
    except (binascii.Error, ValueError):
        return None
