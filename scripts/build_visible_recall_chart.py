#!/usr/bin/env python3
"""Build docs/assets/visible-recall.svg from benchmarks/results/visible-recall/results.json.

Stdlib only, deterministic (fixed arm order, fixed float formats), so
tests/test_visible_recall_chart.py can regenerate it and compare byte for byte.

    python3 scripts/build_visible_recall_chart.py          # write
    python3 scripts/build_visible_recall_chart.py --check  # exit 1 if stale

The chart is the held-out half only (n=180 real tool outputs): the dev half is where
the change was tuned, so plotting it would show the fitted number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results" / "visible-recall" / "results.json"
OUT = ROOT / "docs" / "assets" / "visible-recall.svg"

FONT = "Inter,ui-sans-serif,Segoe UI,Roboto,sans-serif"
INK, MUT, DIM, GRID = "#f2f3f7", "#9aa1b3", "#7d8598", "#252c3e"
DISTIL, HEADROOM = "#8b7bff", "#a3d977"

# (tier heading, [(results.json arm key, label, colour, is the shipped-before row)])
TIERS: list[tuple[str, list[tuple[str, str, str, bool]]]] = [
    (
        "DEFAULT TIER",
        [
            ("distil-default (main)", "distil default · before", DISTIL, True),
            ("distil-default (new)", "distil default · this release", DISTIL, False),
            ("headroom-default", "Headroom 0.42 default", HEADROOM, False),
            ("headroom-coding", "Headroom 0.42 coding profile", HEADROOM, False),
        ],
    ),
    (
        "AGGRESSIVE TIER",
        [
            ("distil-aggressive (main)", "distil aggressive · before", DISTIL, True),
            ("distil-aggressive (new)", "distil aggressive · this release", DISTIL, False),
            ("headroom-agent90", "Headroom 0.42 agent-90", HEADROOM, False),
        ],
    ),
]

W = 1200
X0, X1 = 330, 900  # bar span = 0..100% recall
BAR_H, ROW, TIER_GAP = 22, 40, 34
TOP = 150


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _x(frac: float) -> float:
    return X0 + (X1 - X0) * frac


def build() -> str:
    held = json.loads(RESULTS.read_text(encoding="utf-8"))["held_out"]
    arms = held["arms"]
    n = arms["distil-default (new)"]["all"]["n"]
    rows: list[str] = []
    y = TOP
    for heading, members in TIERS:
        rows.append(
            f'<text x="48" y="{y + 4}" font-size="12" letter-spacing="2" font-weight="700" '
            f'fill="{MUT}">{heading}</text>'
        )
        y += 18
        for key, label, colour, before in members:
            a = arms[key]["all"]
            vis, true, red = a["visible_recall"], a["true_recall"], a["reduction"]
            lost = a["facts_lost"]
            fill_op = "0.38" if before else "1"
            weight = "500" if before else "700"
            rows.append(
                f'<g data-arm="{escape(key)}" data-visible="{vis:.4f}" data-true="{true:.4f}" '
                f'data-reduction="{red:.4f}" data-lost="{lost}">'
            )
            rows.append(
                f'<text x="{X0 - 14}" y="{y + 15}" font-size="13" font-weight="{weight}" '
                f'text-anchor="end" fill="{INK if not before else MUT}">{escape(label)}</text>'
            )
            # True recall: the outline (everything recoverable, in view or behind a handle).
            rows.append(
                f'<rect x="{X0}" y="{y}" width="{_x(true) - X0:.1f}" height="{BAR_H}" rx="5" '
                f'fill="none" stroke="{colour}" stroke-width="1.6" stroke-dasharray="4 3"/>'
            )
            # Visible recall: the solid bar (what the model reads without an expand).
            rows.append(
                f'<rect x="{X0}" y="{y}" width="{_x(vis) - X0:.1f}" height="{BAR_H}" rx="5" '
                f'fill="{colour}" fill-opacity="{fill_op}"/>'
            )
            lost_txt = f" · {lost} lost" if lost else ""
            rows.append(
                f'<text x="{X1 + 16}" y="{y + 15}" font-size="12.5" fill="{INK}">'
                f'<tspan font-weight="700">{_pct(vis)}</tspan>'
                f'<tspan fill="{MUT}"> visible · {_pct(true)} true{lost_txt}</tspan></text>'
            )
            rows.append(
                f'<text x="{X1 + 16}" y="{y + 30}" font-size="11" fill="{DIM}">'
                f"tokens removed {_pct(red)}</text>"
            )
            rows.append("</g>")
            y += ROW
        y += TIER_GAP - 12
    height = y + 88
    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = _x(frac)
        grid.append(
            f'<line x1="{gx:.1f}" y1="{TOP - 14}" x2="{gx:.1f}" y2="{y - 10}" stroke="{GRID}" '
            f'stroke-width="1"/>'
            f'<text x="{gx:.1f}" y="{TOP - 22}" font-size="11" text-anchor="middle" '
            f'fill="{DIM}">{int(frac * 100)}%</text>'
        )
    title = "Visible vs true fact recall, distil and Headroom 0.42"
    desc = (
        f"Held-out half of the offline head-to-head, n={n} real tool outputs. For each arm a "
        "solid bar is visible recall (facts left in the text the model reads) and a dashed "
        "outline is true recall (visible plus facts recoverable through distil_expand or "
        "Headroom's retrieval store). "
        + "; ".join(
            f"{label} {_pct(arms[k]['all']['visible_recall'])} visible, "
            f"{_pct(arms[k]['all']['true_recall'])} true, "
            f"{_pct(arms[k]['all']['reduction'])} tokens removed"
            for _, m in TIERS
            for k, label, _, _ in m
        )
        + ". Offline token and fact-recall measurements only; not a cost or task-success result."
    )
    legend_y = 102
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" font-family="{FONT}" role="img" '
        'aria-labelledby="vrTitle vrDesc">\n'
        f'  <title id="vrTitle">{escape(title)}</title>\n'
        f'  <desc id="vrDesc">{escape(desc)}</desc>\n'
        '  <defs><linearGradient id="vrbg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#0b0c13"/><stop offset="1" stop-color="#06070b"/>'
        "</linearGradient></defs>\n"
        f'  <rect width="{W}" height="{height}" fill="url(#vrbg)"/>\n'
        f'  <text x="48" y="52" font-size="24" font-weight="800" fill="{INK}">'
        f'What the model can see <tspan fill="{MUT}" font-weight="500" font-size="15">'
        f"— visible vs true fact recall, held-out n={n}</tspan></text>\n"
        f'  <g font-size="12" fill="{MUT}">'
        f'<rect x="48" y="{legend_y - 11}" width="26" height="13" rx="3" fill="{DISTIL}"/>'
        f'<text x="82" y="{legend_y}">visible recall (solid)</text>'
        f'<rect x="250" y="{legend_y - 11}" width="26" height="13" rx="3" fill="none" '
        f'stroke="{MUT}" stroke-dasharray="4 3"/>'
        f'<text x="284" y="{legend_y}">true recall (dashed: + recoverable)</text>'
        f'<rect x="530" y="{legend_y - 11}" width="13" height="13" rx="3" fill="{DISTIL}"/>'
        f'<text x="551" y="{legend_y}">distil</text>'
        f'<rect x="610" y="{legend_y - 11}" width="13" height="13" rx="3" fill="{HEADROOM}"/>'
        f'<text x="631" y="{legend_y}">Headroom 0.42.0</text>'
        f'<text x="770" y="{legend_y}">faded = distil before this release</text></g>\n'
        f"  {''.join(grid)}\n"
        "  " + "\n  ".join(rows) + "\n"
        f'  <text x="48" y="{height - 47}" font-size="11.5" fill="{DIM}">'
        "Offline, $0: same 360 real tool outputs and scorer as the Headroom 0.42 head-to-head; "
        "this is the held-out half. Tokens and fact recall only, not cost or task success.</text>\n"
        f'  <text x="48" y="{height - 30}" font-size="11.5" fill="{DIM}">'
        "“lost” = in neither the text nor the retrieval store; the head-to-head write-up shows "
        "some are a matcher artifact (error lines cut mid-token), not compression loss.</text>\n"
        f'  <text x="48" y="{height - 13}" font-size="11.5" fill="{DIM}">'
        "Source: benchmarks/results/visible-recall/results.json · "
        "built by scripts/build_visible_recall_chart.py</text>\n"
        "</svg>\n"
    )


def main(argv: list[str]) -> int:
    svg = build()
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != svg:
            print(f"{OUT.relative_to(ROOT)} is stale: run scripts/build_visible_recall_chart.py")
            return 1
        return 0
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
