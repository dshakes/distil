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
INTENTIONALLY_STATIC = {
    "logo.svg",
    "logo-lockup.svg",
    "og.svg",
    # A structural map of the four entry points and which compression tier each
    # reaches. Nothing about it happens over time — the reader compares boxes,
    # and motion would only pull the eye along one path as if it were the
    # recommended one. The other diagrams animate because they show a *process*.
    "integration-surface.svg",
    # A plot of measured results (distil bench --curve), regenerated from the corpus
    # rather than drawn. The reader compares four fixed points; nothing here happens
    # over time, and animating a measurement would imply a trend the data does not
    # contain. The other diagrams animate because they show a *process*.
    "curve.svg",
}

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


# --- diagrams must not out-date the code -------------------------------------
# A number baked into a diagram is the easiest claim in the project to forget:
# it is not grep-able as prose and nobody re-reads an SVG. `integration-surface`
# already drifted once — it said "6 agents" the day `goose` made it seven.


def _svg_text(path: Path) -> str:
    import re as _re

    raw = path.read_text(encoding="utf-8")
    inner = " ".join(_re.findall(r"<text[^>]*>(.*?)</text>", raw, _re.S))
    return _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", "", inner))


def test_no_diagram_claims_a_stale_agent_count():
    import re as _re

    from distil.onboard import AGENT_PRESETS

    actual = len(AGENT_PRESETS)
    for path in ASSETS:
        for claimed in _re.findall(r"(\d+)\s+agents\b", _svg_text(path)):
            assert int(claimed) == actual, (
                f"{path.name} says '{claimed} agents' but AGENT_PRESETS has {actual}"
            )


def test_no_diagram_names_an_agent_that_has_no_preset():
    """A diagram promising an agent distil cannot route is a support ticket."""
    import re as _re

    from distil.onboard import AGENT_PRESETS

    known = set(AGENT_PRESETS) | {
        # Named in diagrams as clients/providers rather than as wrap targets.
        "cursor",
        "copilot",
        "cline",
        "continue",
        "windsurf",
        "crewai",
        "autogen",
    }
    for path in ASSETS:
        for m in _re.findall(r"distil wrap -- ([a-z][a-z0-9-]*)", _svg_text(path)):
            assert m in known, f"{path.name} shows `distil wrap -- {m}` with no such preset"
