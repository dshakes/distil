"""The degradation curve runs, and says the thing it exists to say.

`distil bench` proves the default is non-inferior. That is a yes/no, and it hides the
shape of the tradeoff on either side of the operating point. These tests pin the shape:
reversible rungs hold full recall, the lossy rung buys savings by losing facts, and both
artefacts (results JSON, chart SVG) are actually written and well-formed.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from distil import curve
from distil.corpus import CORPUS_DIR, load_corpus


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    """A two-trajectory corpus — enough to exercise the macro average (which needs more
    than one domain to mean anything) without paying for all nine."""
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    picked = manifest["trajectories"][:2]
    assert len({t["domain"] for t in picked}) == 2, "need two domains for a macro average"
    out = tmp_path / "corpus"
    out.mkdir()
    for entry in picked:
        shutil.copy(CORPUS_DIR / entry["file"], out / entry["file"])
    (out / "manifest.json").write_text(json.dumps({"trajectories": picked}), encoding="utf-8")
    return out


def test_curve_runs_on_a_mini_corpus(mini_corpus: Path) -> None:
    points = curve.run(load_corpus(mini_corpus))
    assert [p.rung for p in points] == ["none", "byte-exact", "lossless", "aggressive"]
    for p in points:
        assert 0.0 <= p.recall <= 1.0
        assert p.latency_ms >= 0.0
        # Recall counts expand-recoverable facts, so it can only ever meet or exceed the
        # facts that stayed visible. The reverse would mean the recovery path invented one.
        assert p.recall >= p.visible_recall - 1e-9


def test_reversible_rungs_keep_every_fact(mini_corpus: Path) -> None:
    """The load-bearing claim of the whole ladder: everything at or below `lossless` is
    recoverable, so no fact is lost — only moved behind a handle."""
    points = {p.rung: p for p in curve.run(load_corpus(mini_corpus))}
    for name in ("none", "byte-exact", "lossless"):
        assert points[name].reversible
        assert points[name].lost_facts == 0, f"{name} lost facts but claims reversibility"
        assert points[name].recall == pytest.approx(1.0)


def test_the_aggressive_rung_is_where_the_curve_bends(mini_corpus: Path) -> None:
    """The point of drawing a curve at all. `aggressive` saves more than `lossless` and
    pays for it in facts — irrecoverably, since it issues no handle. If this ever stops
    being true the ladder has changed shape and the docs are wrong."""
    points = {p.rung: p for p in curve.run(load_corpus(mini_corpus))}
    agg, loss = points["aggressive"], points["lossless"]
    assert not agg.reversible
    assert agg.savings_pct > loss.savings_pct
    assert agg.lost_facts > 0
    assert agg.recall < loss.recall


def test_digest_moves_facts_behind_a_handle_rather_than_dropping_them(mini_corpus: Path) -> None:
    """The distinction the curve is drawn to show: `lossless` has strictly *lower* visible
    recall than `byte-exact` — it really did remove text the model can read — while total
    recall stays at 1.0 because every one of those facts is behind a handle."""
    points = {p.rung: p for p in curve.run(load_corpus(mini_corpus))}
    assert points["lossless"].visible_recall < points["byte-exact"].visible_recall
    assert points["lossless"].recall == pytest.approx(points["byte-exact"].recall)


def test_writes_results_json_and_a_wellformed_svg(mini_corpus: Path, tmp_path: Path) -> None:
    entries = load_corpus(mini_corpus)
    points = curve.run(entries)
    out = tmp_path / "results" / "curve.json"
    svg = tmp_path / "assets" / "curve.svg"
    curve.write(points, corpus_size=len(entries), results_path=out, svg_path=svg)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["corpus_trajectories"] == 2
    assert payload["generated"] and payload["version"]
    assert [p["rung"] for p in payload["points"]] == [s.name for s in curve.LADDER]

    root = ET.fromstring(svg.read_text(encoding="utf-8"))
    assert root.tag.endswith("svg")
    # The chart must carry the date and version it was generated from — an undated
    # benchmark chart on a docs page is indistinguishable from a stale one.
    text = svg.read_text(encoding="utf-8")
    assert payload["version"] in text and payload["generated"] in text
    assert "aggressive" in text and "not reversible" not in text.lower().split("aria-label")[0]


def test_cli_curve_writes_both_artifacts(mini_corpus: Path, tmp_path: Path) -> None:
    from distil.cli import main

    out = tmp_path / "curve.json"
    svg = tmp_path / "curve.svg"
    rc = main(
        [
            "bench",
            "--curve",
            "--corpus",
            str(mini_corpus),
            "--curve-out",
            str(out),
            "--curve-svg",
            str(svg),
        ]
    )
    assert rc == 0
    assert out.exists() and svg.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["points"]


def test_svg_scales_its_axis_to_the_observed_range() -> None:
    """A fixed 0-1 y-axis would collapse three rungs onto one line and hide the only
    comparison the chart exists to make, so the axis is scaled. Guard the scaling rather
    than the pixels: the lossy point must sit visibly below the reversible ones."""
    points = [
        curve.CurvePoint("none", "", 0.0, 1.0, 1.0, 0, True, 0.0),
        curve.CurvePoint("lossless", "", 47.0, 1.0, 0.77, 0, True, 1.0),
        curve.CurvePoint("aggressive", "", 65.0, 0.55, 0.55, 800, False, 1.0),
    ]
    svg = curve.render_svg(points, version="9.9.9", generated="2026-09-04")
    ET.fromstring(svg)
    assert "9.9.9" in svg
    assert "800 facts lost" in svg
    lo = 0.55 - 0.02
    assert curve._y(0.55, lo) > curve._y(1.0, lo) + 100, "lossy rung is not visibly separated"
