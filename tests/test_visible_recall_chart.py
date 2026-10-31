"""docs/assets/visible-recall.svg must match a fresh build and the committed results."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_visible_recall_chart", _ROOT / "scripts" / "build_visible_recall_chart.py"
)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)

_ROW = re.compile(
    r'data-arm="([^"]*)" data-visible="([^"]*)" data-true="([^"]*)" '
    r'data-reduction="([^"]*)" data-lost="([^"]*)"'
)


def test_chart_matches_fresh_build() -> None:
    """Regenerating must be a no-op. If this fails: python3 scripts/build_visible_recall_chart.py"""
    assert mod.OUT.read_text(encoding="utf-8") == mod.build()


def test_every_plotted_arm_equals_the_held_out_results() -> None:
    held = json.loads(mod.RESULTS.read_text(encoding="utf-8"))["held_out"]["arms"]
    rows = _ROW.findall(mod.OUT.read_text(encoding="utf-8"))
    plotted = [key for _, members in mod.TIERS for key, *_ in members]
    assert [r[0] for r in rows] == plotted
    for key, vis, true, red, lost in rows:
        a = held[key]["all"]
        assert float(vis) == round(a["visible_recall"], 4)
        assert float(true) == round(a["true_recall"], 4)
        assert float(red) == round(a["reduction"], 4)
        assert int(lost) == a["facts_lost"]


def test_chart_is_the_held_out_split_not_the_tuning_split() -> None:
    """The dev half is where the change was tuned; plotting it would show the fitted number."""
    data = json.loads(mod.RESULTS.read_text(encoding="utf-8"))
    n = data["held_out"]["arms"]["distil-default (new)"]["all"]["n"]
    assert f"held-out n={n}" in mod.OUT.read_text(encoding="utf-8")
