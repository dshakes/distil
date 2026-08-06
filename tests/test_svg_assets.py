"""Every shipped SVG must be well-formed, and every animated one must be escapable.

Written after an audit found `hero-terminal.svg` carrying twelve animations and no
`prefers-reduced-motion` guard. It had been on the landing page since the redesign,
through an entire accessibility series, because nothing checked.

The keyTimes assertions are not pedantry: a list that does not start at 0 and end at 1,
or is not monotonic, makes the browser drop the whole animation or stutter the loop —
and a diagram that renders half-drawn is worse than a static one.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ASSETS = sorted((Path(__file__).resolve().parent.parent / "docs" / "assets").glob("*.svg"))

# Static on purpose. A favicon that moves is a bug, and every social platform renders
# an OpenGraph card as a still image — animating it buys nothing and risks the frame a
# scraper happens to capture being a half-drawn one.
INTENTIONALLY_STATIC = {"logo.svg", "logo-lockup.svg", "og.svg"}

# The OpenGraph card is never in a reading order. It is fetched by scrapers and rendered
# as a preview thumbnail beside link text that already says what the page is, so a name
# inside it reaches nobody. Every other asset is a document someone can open.
UNNAMED_BY_DESIGN = {"og.svg"}


def test_there_are_assets_to_check() -> None:
    """A glob that silently matches nothing turns this whole file into a no-op."""
    assert len(ASSETS) > 10


@pytest.mark.parametrize("path", ASSETS, ids=lambda p: p.name)
class TestEverySvg:
    def test_is_well_formed_xml(self, path: Path) -> None:
        ET.fromstring(path.read_text(encoding="utf-8"))

    def test_animation_can_be_turned_off(self, path: Path) -> None:
        """Motion without an escape hatch is an accessibility defect, not a style choice."""
        text = path.read_text(encoding="utf-8")
        if "<animate" not in text and "@keyframes" not in text:
            assert path.name in INTENTIONALLY_STATIC, (
                f"{path.name} is static — animate it, or add it to INTENTIONALLY_STATIC "
                "with the reason"
            )
            return
        assert "prefers-reduced-motion" in text, (
            f"{path.name} animates with no reduced-motion guard"
        )

    def test_keytimes_are_valid(self, path: Path) -> None:
        for kt in re.findall(r'keyTimes="([^"]+)"', path.read_text(encoding="utf-8")):
            values = [float(x) for x in kt.split(";")]
            assert values[0] == 0.0, f"keyTimes must start at 0: {kt}"
            assert values[-1] == 1.0, f"keyTimes must end at 1: {kt}"
            assert values == sorted(values), f"keyTimes must be non-decreasing: {kt}"

    def test_values_and_keytimes_agree(self, path: Path) -> None:
        """Unequal counts make the browser ignore the animation entirely — it fails by
        rendering the start state forever, which looks like a design choice."""
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"<animate\b[^>]*?>", text, re.S):
            tag = m.group(0)
            vals = re.search(r'values="([^"]+)"', tag)
            times = re.search(r'keyTimes="([^"]+)"', tag)
            if vals and times:
                assert len(vals.group(1).split(";")) == len(times.group(1).split(";")), (
                    f"{path.name}: values/keyTimes length mismatch in {tag[:90]}"
                )

    def test_has_an_accessible_name(self, path: Path) -> None:
        """A diagram carrying the argument must reach a screen reader too.

        `<img alt>` at the usage site already names these for a page reader, but an SVG
        is also a document people open directly — GitHub renders `docs/assets/*.svg` as
        a page — and there the alt attribute does not exist. The name belongs in the
        file, where it travels with it.
        """
        if path.name in UNNAMED_BY_DESIGN:
            return
        text = path.read_text(encoding="utf-8")
        assert "aria-label" in text or "aria-labelledby" in text, f"{path.name} has no name"
