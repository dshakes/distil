"""Downscaling a first-occurrence image — lossy in context, recoverable outside it.

`vision.py` deliberately refuses downscaling, and its reasoning stands on its own
terms: resize the pixels and the model sees a different image, so nobody can say
whether the answer changed. This module does not overturn that. It changes one
thing about the situation — the original goes into the RestoreStore and the
replacement carries a handle — which moves downscaling into the same category as
the Tier-1 text digest rather than the category `vision.py` rejects.

The distinction that matters
----------------------------
distil's contract is not "never alter what the model sees". The text digest
alters it on every request. The contract is that nothing is *irrecoverably* lost:
a stub says what it replaced and how to get it back, and `distil_expand` returns
the exact bytes. Applied here:

    an oversized image is sent at reduced resolution, accompanied by a text note
    that says it was downscaled, from what to what, and the handle that returns
    the untouched original.

So the model can see it is looking at a reduced image and can ask for the full
one. That is strictly more than it knows about a downscale performed silently.

Where this is genuinely weaker than the text digest
---------------------------------------------------
An elided text span is *visibly* absent — the model reads a marker where content
used to be. A downscaled image is not: it looks like an image, and a model that
does not read the accompanying note may never realise detail is missing and so
never expand. That asymmetry is real, it cannot be engineered away from inside
this module, and it is exactly why this ships **disabled** behind its own
certificate rather than on by default.

Gate
----
Ships off. `enabled()` is False until `distil certify --strategy vision-downscale`
produces a local passing certificate. Unlike `vision`, NO certificate is bundled:
the maintainer's corpus cannot speak for whether losing pixel detail changes
decisions on someone else's screenshots. `DISTIL_VISION_DOWNSCALE=1` force-enables
for the certification run and tests — an opt-in to uncertified behaviour, not a
tuning knob.

Dependency
----------
Real resampling needs a real codec, which distil does not carry. Pillow is an
optional extra (`distil[image]`); without it `available()` is False and every
entry point is a no-op. distil's stdlib-only guarantee is unchanged for anyone
who does not opt in.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from .vision import _certificate_verdict, estimate_tokens, image_dims

#: Long-edge ceiling after downscaling. Providers already resize above ~1568px
#: and bill on the resized dimensions, so anything at or under that ceiling is
#: paid for as-is — the saving only starts below it. 1024 keeps a UI screenshot
#: legible (text in a 1024px-wide window is still readable) while cutting the
#: billed area of a 1568px image by roughly 2.3x.
DEFAULT_MAX_EDGE = 1024

#: Never touch an image already at or below this. Small images are cheap, the
#: accompanying note is not free, and a downscaled icon is worse than useless.
MIN_EDGE_TO_SCALE = 1200

#: Minimum token saving before it is worth altering the image at all. Below this
#: the note costs about what the resize saves and the model loses detail for
#: nothing.
MIN_TOKENS_SAVED = 120


def _certificate_path() -> Path:
    home = Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))
    return home / "certificates" / "vision-downscale.json"


def available() -> bool:
    """Whether a codec capable of resampling is installed."""
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        return False
    return True


def enabled() -> bool:
    """True only when downscaling has been certified non-inferior *locally*.

    No bundled certificate, deliberately: `vision`'s shipped certificate covers
    byte-identical elision, where the model still sees every image once and the
    bytes are provably the same. Losing pixel detail is a different claim, and
    whether it changes a decision depends on what the user screenshots — a
    maintainer's corpus cannot answer that for them.
    """
    override = os.environ.get("DISTIL_VISION_DOWNSCALE")
    if override is not None:
        return override.strip() not in ("", "0", "false", "no")
    return _certificate_verdict(_certificate_path(), "vision-downscale") is True


def active() -> bool:
    """Enabled AND able to run. Both, or the adapter must leave images alone."""
    return enabled() and available()


def downscale(raw: bytes, max_edge: int = DEFAULT_MAX_EDGE) -> tuple[bytes, tuple[int, int]] | None:
    """Resample *raw* so its long edge is at most *max_edge*.

    Returns (new_bytes, (w, h)) or None when the image should be left alone —
    already small enough, undecodable, or the re-encode came out no smaller
    (which happens: re-encoding a heavily-optimised PNG can grow it).
    """
    dims = image_dims(raw)
    if dims is None:
        return None  # unknown size: never guess at an image we cannot measure
    width, height = dims
    if max(width, height) < MIN_EDGE_TO_SCALE:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    scale = max_edge / float(max(width, height))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    try:
        with Image.open(io.BytesIO(raw)) as img:
            fmt = img.format or "PNG"
            resized = img.resize(new_size, Image.LANCZOS)
            out = io.BytesIO()
            # Keep the original container. Converting PNG->JPEG would shrink more
            # but silently discards alpha and adds block artefacts to text, which
            # is the opposite of what a UI screenshot needs.
            resized.save(out, format=fmt)
    except Exception:
        # A corrupt or exotic image must never take down a request. Leaving it
        # verbatim is always safe — the same rule the text digester follows.
        return None
    data = out.getvalue()
    if len(data) >= len(raw):
        return None
    return data, new_size


def note_text(handle: str, before: tuple[int, int], after: tuple[int, int], saved: int) -> str:
    """The text block that accompanies a downscaled image.

    Without this the transform would be silently lossy, which is the thing distil
    does not do. It states the alteration, its magnitude, and the way back, using
    the same `<< … handle=XXXXXXXX >>` grammar as every other digest so
    `distil_expand` parses it identically.
    """
    return (
        f"<< distil:image — the image above was downscaled from "
        f"{before[0]}x{before[1]} to {after[0]}x{after[1]} to save ~{saved} input "
        f"tokens; if you need detail that is not legible at this size, recover the "
        f"untouched original with distil_expand, handle={handle} >>"
    )


def plan(source: dict[str, Any], max_edge: int = DEFAULT_MAX_EDGE) -> tuple[bytes, Any, int] | None:
    """Decide whether to downscale one image `source`, and by how much.

    Returns (new_raw_bytes, (w, h), tokens_saved) or None to leave it alone.
    Only inline base64 payloads are considered: a URL source's bytes are not in
    hand, and distil does not fetch them.
    """
    data = source.get("data")
    if not isinstance(data, str) or not data:
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception:
        return None
    before = image_dims(raw)
    if before is None:
        return None
    result = downscale(raw, max_edge)
    if result is None:
        return None
    new_raw, after = result
    saved = estimate_tokens(raw) - estimate_tokens(new_raw)
    if saved < MIN_TOKENS_SAVED:
        return None
    return new_raw, after, saved
