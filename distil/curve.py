"""The degradation curve — what each rung of the ladder actually costs in recall.

`distil bench` answers "is the shipped strategy non-inferior?" with a yes. That is the
right question for a gate and the wrong one for a decision: it says nothing about what
sits on either side of the operating point. Someone choosing between lossless-only and
the digest, or wondering what the aggressive rung would buy, has had no measured answer
— only the claim that the default is safe.

This traces the whole ladder in one pass over the offline corpus. For each rung it
reports token savings, fact-level recall from the retention harness, whether the rung is
reversible, and how long it took. Plotting recall against savings gives the curve the
project has been describing in prose: three rungs sit at full recall and differ only in
savings, and the fourth buys its extra savings by losing facts outright.

Zero cost by construction — no API calls, no network, no model. It is a measurement of
the compressors themselves, so it can run on every commit.

The rungs are `compress.adaptive.PRODUCTION_LADDER`, plus the distinction that ladder
does not draw: `byte-exact` IS the lossless-only (subscription) operating point, since
Tier-0 is the only transform available when no recovery handle can be issued.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import retention
from .compress.adaptive import _aggressive
from .compress.base import CompressResult
from .compress.tier0 import Tier0Lossless
from .compress.tier1 import Tier1Reversible
from .corpus import CorpusEntry, load_corpus
from .tokenizer import DEFAULT as _TOK
from .trajectory import Block, Stability

_T0 = Tier0Lossless()
_T1 = Tier1Reversible()

# A rung takes the VOLATILE blocks of one turn and returns (blocks, restore). The restore
# map is what makes recall meaningful: retention scores a fact as *recoverable* only when
# it is present in the bytes behind a handle the compressed text actually carries, so a
# rung that folds without a handle correctly scores those facts as lost.
Rung = Callable[[list[Block]], CompressResult]


def _none(blocks: list[Block]) -> CompressResult:
    return CompressResult(blocks, {})


def _byte_exact(blocks: list[Block]) -> CompressResult:
    return _T0.compress(blocks)


def _lossless(blocks: list[Block]) -> CompressResult:
    """The shipped default: Tier-1 reversible digest, then Tier-0 over the result.
    Mirrors `compress.strategies.distil` and `retention.probe_trajectory`."""
    t1 = _T1.compress(blocks)
    t0 = _T0.compress(t1.blocks)
    return CompressResult(t0.blocks, {**t1.restore, **t0.restore})


def _aggressive_rung(blocks: list[Block]) -> CompressResult:
    """Head/tail truncation, ignoring decision-relevance, with NO recovery handle.
    The rung the dial only reaches when you relax `target_equivalence` below 1.0."""
    return CompressResult(_aggressive(blocks, 0), {})


@dataclass(frozen=True)
class RungSpec:
    name: str
    note: str
    reversible: bool
    fn: Rung


LADDER: list[RungSpec] = [
    RungSpec("none", "no compression — the reference", True, _none),
    RungSpec(
        "byte-exact", "Tier-0 only; the lossless-only / subscription point", True, _byte_exact
    ),
    RungSpec("lossless", "Tier-1 digest + Tier-0 — the shipped default", True, _lossless),
    RungSpec("aggressive", "head/tail truncation, no recovery handle", False, _aggressive_rung),
]


@dataclass(frozen=True)
class CurvePoint:
    rung: str
    note: str
    savings_pct: float
    recall: float  # macro-average across domains — cannot be swung by one fixture
    visible_recall: float  # before counting expand-recoverable facts
    lost_facts: int
    reversible: bool
    latency_ms: float


def measure(spec: RungSpec, entries: list[CorpusEntry]) -> CurvePoint:
    """Run one rung over the corpus and score it."""
    report = retention.RetentionReport()
    base_tok = comp_tok = 0
    elapsed = 0.0
    for entry in entries:
        for turn in entry.trajectory.turns:
            volatile = [b for b in turn.blocks if b.stability is Stability.VOLATILE]
            if not volatile:
                continue
            start = time.perf_counter()
            result = spec.fn(volatile)
            elapsed += time.perf_counter() - start
            by_id = {b.id: b.text for b in result.blocks}
            for original in volatile:
                text = by_id.get(original.id, "")
                base_tok += _TOK.count(original.text)
                comp_tok += _TOK.count(text)
                report.probes.append(
                    retention.probe_block(original, text, result.restore, domain=entry.domain)
                )
    macro = report.macro()
    return CurvePoint(
        rung=spec.name,
        note=spec.note,
        savings_pct=round(100.0 * (1.0 - comp_tok / base_tok), 2) if base_tok else 0.0,
        recall=round(macro.recall, 4),
        visible_recall=round(macro.visible_recall, 4),
        lost_facts=report.lost_facts,
        reversible=spec.reversible,
        latency_ms=round(1000.0 * elapsed, 2),
    )


def run(entries: list[CorpusEntry] | None = None) -> list[CurvePoint]:
    entries = entries if entries is not None else load_corpus()
    return [measure(spec, entries) for spec in LADDER]


def to_dict(points: list[CurvePoint], *, corpus_size: int) -> dict[str, object]:
    from . import __version__

    return {
        "generated": time.strftime("%Y-%m-%d", time.gmtime()),
        "version": __version__,
        "corpus_trajectories": corpus_size,
        "tokenizer": type(_TOK).__name__,
        "recall": "macro-average across corpus domains, counting expand-recoverable facts",
        "points": [asdict(p) for p in points],
    }


# --------------------------------------------------------------------------- rendering


def format_table(points: list[CurvePoint]) -> str:
    rows = [f"{'rung':<12}{'savings':>9}{'recall':>9}{'visible':>9}{'lost':>6}{'rev':>5}{'ms':>8}"]
    rows.append("-" * 58)
    for p in points:
        rows.append(
            f"{p.rung:<12}{p.savings_pct:>8.1f}%{p.recall:>9.3f}{p.visible_recall:>9.3f}"
            f"{p.lost_facts:>6}{'yes' if p.reversible else 'NO':>5}{p.latency_ms:>8.1f}"
        )
    return "\n".join(rows)


_W, _H = 720, 380
_L, _R, _T, _B = 64, 24, 40, 52  # margins


def _x(savings: float) -> float:
    return _L + (savings / 100.0) * (_W - _L - _R)


def _y(recall: float, lo: float) -> float:
    span = max(1.0 - lo, 1e-9)
    return _T + (1.0 - (recall - lo) / span) * (_H - _T - _B)


def render_svg(points: list[CurvePoint], *, version: str = "", generated: str = "") -> str:
    """A stdlib-only inline SVG of recall against savings.

    The y-axis is scaled to the observed recall range rather than pinned to 0-1. Three of
    the four rungs sit at or near 1.0, so a full-height axis would collapse them onto one
    line and hide the only comparison the chart exists to make.
    """
    lo = min(min(p.recall for p in points), 1.0) - 0.02
    lo = max(0.0, min(lo, 0.98))
    ordered = sorted(points, key=lambda p: p.savings_pct)
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{_x(p.savings_pct):.1f},{_y(p.recall, lo):.1f}"
        for i, p in enumerate(ordered)
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" role="img" '
        f'font-family="Inter,ui-sans-serif,Segoe UI,Roboto,sans-serif" '
        f'aria-label="Fact recall against token savings for each rung of the compression '
        f"ladder. Reversible rungs hold full recall; the aggressive rung buys savings by "
        f'losing facts.">',
        '<defs><linearGradient id="cvg" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#8b7bff"/><stop offset="1" stop-color="#5ad1c9"/>'
        "</linearGradient></defs>",
        f'<rect width="{_W}" height="{_H}" fill="#0b0c13"/>',
        f'<text x="{_L}" y="24" font-size="14" font-weight="700" fill="url(#cvg)">'
        "The degradation curve — fact recall vs token savings</text>",
    ]
    # gridlines + y labels
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = lo + frac * (1.0 - lo)
        yy = _y(val, lo)
        parts.append(
            f'<line x1="{_L}" y1="{yy:.1f}" x2="{_W - _R}" y2="{yy:.1f}" '
            f'stroke="#222839" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{_L - 8}" y="{yy + 4:.1f}" font-size="10" fill="#7b8298" '
            f'text-anchor="end">{val:.3f}</text>'
        )
    # x axis labels
    for pct in (0, 25, 50, 75, 100):
        xx = _x(pct)
        parts.append(
            f'<text x="{xx:.1f}" y="{_H - _B + 20}" font-size="10" fill="#7b8298" '
            f'text-anchor="middle">{pct}%</text>'
        )
    parts.append(
        f'<text x="{(_L + _W - _R) / 2:.1f}" y="{_H - 12}" font-size="11" fill="#9aa0b2" '
        f'text-anchor="middle">token savings</text>'
    )
    parts.append(
        f'<text x="16" y="{_H / 2:.1f}" font-size="11" fill="#9aa0b2" text-anchor="middle" '
        f'transform="rotate(-90 16 {_H / 2:.1f})">fact recall (macro)</text>'
    )
    parts.append(f'<path d="{path}" fill="none" stroke="url(#cvg)" stroke-width="2"/>')

    for p in ordered:
        cx, cy = _x(p.savings_pct), _y(p.recall, lo)
        # Reversible rungs are filled; the lossy rung is hollow and red — the one visual
        # distinction that matters more than its position on the curve.
        fill = "#5ad19a" if p.reversible else "#0b0c13"
        stroke = "#5ad19a" if p.reversible else "#ff6b6b"
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="2"><title>{p.rung}: {p.savings_pct:.1f}% savings, '
            f"recall {p.recall:.3f}, {'reversible' if p.reversible else 'NOT reversible'}"
            "</title></circle>"
        )
        anchor = "end" if cx > _W - _R - 80 else "start"
        dx = -10 if anchor == "end" else 10
        parts.append(
            f'<text x="{cx + dx:.1f}" y="{cy - 10:.1f}" font-size="11" font-weight="700" '
            f'fill="#dbe0ee" text-anchor="{anchor}">{p.rung}</text>'
        )
        if not p.reversible:
            parts.append(
                f'<text x="{cx + dx:.1f}" y="{cy + 18:.1f}" font-size="10" fill="#ff6b6b" '
                f'text-anchor="{anchor}">{p.lost_facts} facts lost · not reversible</text>'
            )

    stamp = " · ".join(x for x in (f"distil {version}" if version else "", generated) if x)
    if stamp:
        parts.append(
            f'<text x="{_W - _R}" y="24" font-size="10" fill="#7b8298" '
            f'text-anchor="end">{stamp}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write(
    points: list[CurvePoint],
    *,
    corpus_size: int,
    results_path: Path,
    svg_path: Path,
) -> None:
    payload = to_dict(points, corpus_size=corpus_size)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(
        render_svg(
            points,
            version=str(payload["version"]),
            generated=str(payload["generated"]),
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CurvePoint",
    "LADDER",
    "RungSpec",
    "format_table",
    "measure",
    "render_svg",
    "run",
    "to_dict",
    "write",
]
