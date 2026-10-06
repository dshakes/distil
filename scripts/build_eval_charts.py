#!/usr/bin/env python3
"""Build the eval bubble charts in docs/assets/ from committed result artifacts.

Stdlib only. Output is deterministic (sorted iteration, fixed float formats), so
tests/test_eval_charts.py can regenerate and compare byte for byte.

    python3 scripts/build_eval_charts.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
SUMMARY = ROOT / "benchmarks" / "results" / "model-migration" / "summary.json"
RESULTS = ROOT / "benchmarks" / "results"

# Site palette (docs/site.css and the existing SVGs in docs/assets).
BG, PANEL, GRID, AXIS = "#06070b", "#0c0e15", "#1b2030", "#252c3e"
INK, MUT, DIM = "#f2f3f7", "#9aa1b3", "#7d8598"
ACC, ACC2 = "#8b7bff", "#5ad1c9"
# One colour and legend label per outcome-eval arm (plain/distil keep their historical ones).
ARM_STYLE = {
    "plain": (ACC2, "plain agent"),
    "distil": (ACC, "distil-served agent"),
    "rtk": ("#f2b66d", "RTK (rewrite hook)"),
    "selective": ("#e5738e", "Selective Context"),
    "provider-cm": ("#6db7f2", "Anthropic context editing"),
    "distil-sh": ("#7fd1b9", "distil sh (rewrite at source)"),
}
EXTRA_COLORS = ("#a3d977", "#d98fe0", "#c9c9c9")  # arms outside ARM_STYLE, in sorted order
FONT = "Inter,ui-sans-serif,Segoe UI,Roboto,sans-serif"

W, H = 1200, 680
PL, PR, PT, PB = 100, 1140, 130, 520  # plot box
RMAX = 46.0


@dataclass(frozen=True)
class Bubble:
    label: str
    x: float
    y: float
    size: float
    color: str


@dataclass(frozen=True)
class Chart:
    filename: str
    title: str
    caption: str
    desc: str
    xlabel: str
    ylabel: str
    xfmt: str
    yfmt: str
    size_legend: str
    size_unit: str
    bubbles: list[Bubble]
    key: list[tuple[str, str]]  # colour legend entries (label, colour)


def certifier_bubbles() -> list[Bubble]:
    """One bubble per certifier setting, held-out test split."""
    variants = json.loads(SUMMARY.read_text())["certifier_migration_real"]["variants"]
    out = []
    for name in sorted(variants):
        v = variants[name]
        r = v["test_rates"]
        label = v["model"] + (f" {v['effort']}" if v["effort"] else "")
        out.append(
            Bubble(
                label,
                v["cost_per_case_usd"],
                r["equiv_expand_act"] * 100,
                r["self_consist_act"] * 100,
                ACC,
            )
        )
    return out


def _arm_color(arm: str, unknown: list[str]) -> str:
    if arm in ARM_STYLE:
        return ARM_STYLE[arm][0]
    return EXTRA_COLORS[unknown.index(arm) % len(EXTRA_COLORS)]


def _outcome_runs(results_dir: Path) -> list[tuple[Path, list[dict], list[dict]]]:
    def rows(p: Path) -> list[dict]:
        return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]

    def run_key(d: Path) -> tuple[int, int | str]:
        s = d.name.removeprefix("swebench-outcome-")
        return (0, int(s)) if s.isdigit() else (1, s)

    return [
        (d, rows(d / "results.jsonl"), rows(d / "grades.jsonl"))
        for d in sorted(results_dir.glob("swebench-outcome-*"), key=run_key)
        if (d / "results.jsonl").exists() and (d / "grades.jsonl").exists()
    ]


def _unknown_arms(runs: list[tuple[Path, list[dict], list[dict]]]) -> list[str]:
    return sorted({r["arm"] for _, res, _ in runs for r in res} - set(ARM_STYLE))


def outcome_bubbles(results_dir: Path = RESULTS) -> list[Bubble]:
    """One bubble per (run, arm) over every swebench-outcome-* run in *results_dir*."""
    sys.path.insert(0, str(ROOT))
    from benchmarks.swebench_outcome.report import analyse

    runs = _outcome_runs(results_dir)
    unknown = _unknown_arms(runs)
    out = []
    for d, res, grades in runs:
        a = analyse(res, grades)
        run = d.name.removeprefix("swebench-outcome-")
        # Runs made before the harness cached are not cost-comparable with later ones:
        # mark them from the data itself (no cache reads anywhere in the run).
        if not any((r.get("usage") or {}).get("cache_read") for r in res):
            run += " (uncached)"
        for arm, m in a["arms"].items():
            out.append(
                Bubble(
                    f"{run} {arm}",
                    # round: sum() of floats differs by an ulp across Python versions (3.12 changed it)
                    round(m["cost_usd"] / m["tasks"], 6),
                    m["rate"] * 100,
                    float(m["n"]),
                    _arm_color(arm, unknown),
                )
            )
    return out


def outcome_key(results_dir: Path = RESULTS) -> list[tuple[str, str]]:
    """Legend entries for the arms that appear in *results_dir*, canonical order first."""
    runs = _outcome_runs(results_dir)
    unknown = _unknown_arms(runs)
    present = {r["arm"] for _, res, _ in runs for r in res}
    order = [a for a in ARM_STYLE if a in present] + unknown
    return [(ARM_STYLE[a][1] if a in ARM_STYLE else a, _arm_color(a, unknown)) for a in order]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.floor(lo / step) * step
    ticks, t = [], first
    while t < hi + step * 0.999:
        ticks.append(round(t, 10))
        t += step
        if ticks[-1] >= hi:
            break
    return ticks


def _fmt(v: float, spec: str) -> str:
    return format(v, spec)


def render(c: Chart) -> str:
    bs = c.bubbles
    xt = _nice_ticks(min(b.x for b in bs) * 0.6, max(b.x for b in bs) * 1.15)
    yt = _nice_ticks(min(b.y for b in bs) - 3, max(b.y for b in bs) + 3)
    x0, x1, y0, y1 = xt[0], xt[-1], yt[0], yt[-1]
    sx = lambda v: PL + (v - x0) / (x1 - x0) * (PR - PL)  # noqa: E731
    sy = lambda v: PB - (v - y0) / (y1 - y0) * (PB - PT)  # noqa: E731
    smax = max(b.size for b in bs)
    rad = lambda s: RMAX * math.sqrt(s / smax)  # noqa: E731  (area ∝ size)

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="{FONT}" role="img" aria-labelledby="t d">',
        f'  <title id="t">{escape(c.title)}</title>',
        f'  <desc id="d">{escape(c.desc)}</desc>',
        f'  <rect width="{W}" height="{H}" fill="{BG}"/>',
        f'  <text x="48" y="52" fill="{INK}" font-size="25" font-weight="700">'
        f"{escape(c.title)}</text>",
        f'  <text x="48" y="78" fill="{MUT}" font-size="14">{escape(c.caption)}</text>',
        f'  <rect x="{PL}" y="{PT}" width="{PR - PL}" height="{PB - PT}" '
        f'fill="{PANEL}" stroke="{AXIS}"/>',
    ]
    for t in xt:
        o.append(f'  <line x1="{sx(t):.1f}" y1="{PT}" x2="{sx(t):.1f}" y2="{PB}" stroke="{GRID}"/>')
        o.append(
            f'  <text x="{sx(t):.1f}" y="{PB + 20}" text-anchor="middle" fill="{DIM}" '
            f'font-size="12">{_fmt(t, c.xfmt)}</text>'
        )
    for t in yt:
        o.append(f'  <line x1="{PL}" y1="{sy(t):.1f}" x2="{PR}" y2="{sy(t):.1f}" stroke="{GRID}"/>')
        o.append(
            f'  <text x="{PL - 10}" y="{sy(t) + 4:.1f}" text-anchor="end" fill="{DIM}" '
            f'font-size="12">{_fmt(t, c.yfmt)}</text>'
        )
    o.append(
        f'  <text x="{(PL + PR) / 2:.1f}" y="{PB + 48}" text-anchor="middle" fill="{MUT}" '
        f'font-size="13" font-weight="700">{escape(c.xlabel)}</text>'
    )
    o.append(
        f'  <text transform="translate(30 {(PT + PB) / 2:.1f}) rotate(-90)" '
        f'text-anchor="middle" fill="{MUT}" font-size="13" font-weight="700">'
        f"{escape(c.ylabel)}</text>"
    )
    for b in sorted(bs, key=lambda b: -b.size):  # big first so small stay visible
        cx, cy, r = sx(b.x), sy(b.y), rad(b.size)
        o.append(
            f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{b.color}" '
            f'fill-opacity="0.35" stroke="{b.color}" stroke-width="2" '
            f'data-label="{escape(b.label)}" data-x="{b.x!r}" data-y="{b.y!r}" '
            f'data-size="{b.size!r}"><title>{escape(b.label)}: '
            f"{_fmt(b.x, c.xfmt)}, {_fmt(b.y, c.yfmt)}, "
            f"{_fmt(b.size, c.yfmt)}</title></circle>"
        )
    # Labels: first free spot among right/left/above/below (est. text width, no overlaps).
    boxes = [
        (sx(b.x) - rad(b.size), sy(b.y) - rad(b.size), sx(b.x) + rad(b.size), sy(b.y) + rad(b.size))
        for b in bs
    ]
    placed: list[tuple[float, float, float, float]] = []
    for i, b in sorted(enumerate(bs), key=lambda p: -p[1].size):
        cx, cy, r = sx(b.x), sy(b.y), rad(b.size)
        w, h = 7.4 * len(b.label), 16
        cands = [
            (cx + r + 8, cy + 4, "start"),
            (cx - r - 8, cy + 4, "end"),
            (cx, cy - r - 8, "middle"),
            (cx, cy + r + 20, "middle"),
        ]
        for lx, ly_, anchor in cands:
            x_lo = lx - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
            box = (x_lo, ly_ - h + 3, x_lo + w, ly_ + 3)
            hit = any(
                box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                for q in placed + [bx for j, bx in enumerate(boxes) if j != i]
            )
            if not hit and PL < box[0] and box[2] < PR and PT < box[1] and box[3] < PB:
                break
        else:  # nothing free: fall back to the first candidate
            lx, ly_, anchor = cands[0]
            box = (lx, ly_ - h + 3, lx + w, ly_ + 3)
        placed.append(box)
        o.append(
            f'  <text x="{lx:.1f}" y="{ly_:.1f}" text-anchor="{anchor}" fill="{INK}" '
            f'font-size="13" paint-order="stroke" stroke="{BG}" stroke-width="3">'
            f"{escape(b.label)}</text>"
        )
    # Legend: size encoding + colour key, below the plot.
    ly = PB + 84
    o.append(f'  <circle cx="{PL + 14}" cy="{ly}" r="{rad(smax):.1f}" fill="none" stroke="{MUT}"/>')
    o.append(
        f'  <circle cx="{PL + 14}" cy="{ly + rad(smax) - rad(smax / 4):.1f}" '
        f'r="{rad(smax / 4):.1f}" fill="none" stroke="{MUT}"/>'
    )
    o.append(
        f'  <text x="{PL + 14 + RMAX + 14}" y="{ly + 4}" fill="{MUT}" font-size="13">'
        f"{escape(c.size_legend)}</text>"
    )
    kx = 700
    for i, (name, col) in enumerate(c.key):
        yy = ly - 8 + i * 22
        o.append(
            f'  <circle cx="{kx}" cy="{yy}" r="6" fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="2"/>'
        )
        o.append(
            f'  <text x="{kx + 14}" y="{yy + 4}" fill="{MUT}" font-size="13">{escape(name)}</text>'
        )
    o.append(
        f'  <text x="48" y="{H - 14}" fill="{DIM}" font-size="12">{escape(c.size_unit)}</text>'
    )
    o.append("</svg>")
    return "\n".join(o) + "\n"


def charts() -> list[Chart]:
    cb = certifier_bubbles()
    ob = outcome_bubbles()
    return [
        Chart(
            "eval-bubble-certifier.svg",
            "Certifier settings: cost, equivalence, self-consistency",
            "Held-out test split, tau-bench turns. Source: benchmarks/results/model-migration/summary.json",
            "Bubble chart, one bubble per certifier setting. x is cost per case in dollars, "
            "y is expand action-equivalence in percent, bubble area is self-consistency in percent. "
            + "; ".join(f"{b.label}: ${b.x:.4f}, {b.y:.1f}%, {b.size:.1f}%" for b in cb)
            + ".",
            "Cost per case (USD)",
            "Expand action-equivalence (%)",
            ".2f",
            ".0f",
            "Bubble area = self-consistency (%)",
            "Rows are the per-model/effort variants of certifier_migration_real, test_rates.",
            cb,
            [("certifier setting (model, effort)", ACC)],
        ),
        Chart(
            "eval-bubble-outcome.svg",
            "SWE-bench Lite outcome: cost vs resolved",
            "One bubble per run and arm; uncached runs are not cost-comparable. Source: benchmarks/results/swebench-outcome-*/",
            "Bubble chart, one bubble per run and arm. x is cost per task in dollars, "
            "y is resolved tasks in percent of paired tasks, bubble area is the number of paired tasks. "
            + "; ".join(f"{b.label}: ${b.x:.4f}, {b.y:.1f}%, {b.size:.0f} tasks" for b in ob)
            + ".",
            "Cost per task (USD)",
            "Resolved (%)",
            ".2f",
            ".0f",
            "Bubble area = number of paired tasks",
            "Analysis from benchmarks/swebench_outcome/report.py (graded outcomes only).",
            ob,
            outcome_key(),
        ),
    ]


def build() -> dict[str, str]:
    return {c.filename: render(c) for c in charts()}


def main() -> None:
    for name, svg in build().items():
        (ASSETS / name).write_text(svg)
        print(f"wrote docs/assets/{name}")


if __name__ == "__main__":
    main()
