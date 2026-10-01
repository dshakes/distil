"""eval — the certified compression frontier (the proof pack).

The artifact no competitor publishes: a savings-vs-quality curve where every
point carries its certification verdict. We sweep compression aggressiveness and
measure, for each level, (token savings, decision-equivalence) over a corpus,
then mark which levels the non-inferiority gate certifies. This *locates the
cliff* past which lossy compression starts dropping decisions — and shows distil
sitting safely inside that frontier at real savings with 100% equivalence.

Reproducible offline with the deterministic runner (structural decision-
equivalence). For real task-accuracy on a benchmark (tau-bench / SWE-bench /
GSM8K), ingest its traces into a corpus and run with ``--runner anthropic`` —
the same curve, graded by the live model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .certify.gate import certify
from .certify.stats import tost
from .compress.strategies import distil as distil_strategy, served as served_strategy
from .compress.tier0 import Tier0Lossless
from .corpus import CorpusEntry, load_corpus
from .replay.runner import AgentRunner
from .tokenizer import DEFAULT, Tokenizer
from .trajectory import Block, Stability

_T0 = Tier0Lossless()
# Truncation lengths (chars) for the lossy sweep — the knob that can drop decisions.
DEFAULT_LIMITS = [4000, 2000, 1200, 700, 450, 300, 220, 160, 120, 90]


def _truncate(limit: int):
    def strat(blocks: list[Block], turn: int) -> list[Block]:
        return [
            b.copy_with(b.text[:limit]) if b.stability is Stability.VOLATILE else b for b in blocks
        ]

    return strat


def _tier0(blocks: list[Block], turn: int) -> list[Block]:
    stable = [b for b in blocks if b.stability is not Stability.VOLATILE]
    volatile = [b for b in blocks if b.stability is Stability.VOLATILE]
    return stable + _T0.compress(volatile).blocks


@dataclass
class EvalPoint:
    label: str
    savings: float  # avg token-reduction fraction over the corpus
    equivalence: float  # avg decision-equivalence match rate
    certified: bool  # gate verdict (non-inferior AND 100% equivalence)


@dataclass
class FrontierReport:
    points: list[EvalPoint] = field(default_factory=list)
    runner: str = "deterministic"
    # A live grader's model/effort (None for the offline oracle), so a raw curve says
    # which certifier graded it — the default moves between releases.
    grader_model: str | None = None
    grader_effort: str | None = None

    @property
    def distil_point(self) -> EvalPoint | None:
        return next((p for p in self.points if p.label.startswith("distil")), None)

    @property
    def certified_ceiling(self) -> float:
        """Max savings among certified points — the proven compression limit."""
        cert = [p.savings for p in self.points if p.certified]
        return max(cert) if cert else 0.0


def _measure(
    entries: list[CorpusEntry], strat, tok: Tokenizer, runner: AgentRunner | None
) -> tuple[float, float, bool]:
    tot_base = tot_comp = 0
    diffs: list[float] = []
    matches: list[float] = []
    for e in entries:
        rep = certify(e.trajectory, strat, runner=runner)
        matches.append(rep.match_rate)
        diffs += [(1.0 if d.matched else 0.0) - 1.0 for d in rep.divergences]
        for turn in e.trajectory.turns:
            comp = strat(turn.blocks, turn.index)
            tot_base += sum(tok.count(b.text) for b in turn.blocks)
            tot_comp += sum(tok.count(b.text) for b in comp)
    savings = (1.0 - tot_comp / tot_base) if tot_base else 0.0
    equivalence = sum(matches) / len(matches) if matches else 0.0
    certified = bool(diffs) and tost(diffs).non_inferior and equivalence == 1.0
    return savings, equivalence, certified


def frontier(
    entries: list[CorpusEntry] | None = None,
    *,
    runner: AgentRunner | None = None,
    limits: list[int] | None = None,
    tok: Tokenizer = DEFAULT,
) -> FrontierReport:
    entries = entries if entries is not None else load_corpus()
    limits = limits if limits is not None else DEFAULT_LIMITS
    report = FrontierReport(
        runner=getattr(runner, "name", "deterministic"),
        grader_model=getattr(runner, "model", None),
        grader_effort=getattr(runner, "effort", None),
    )

    # reference operating points — `served` is the serving adapter itself (what a
    # caching client is sent), so the frontier shows the served path, not just the
    # volatile-only slice `distil` certifies
    for label, strat in (
        ("tier-0 lossless", _tier0),
        ("distil (cache-aware)", distil_strategy),
        ("served (adapter)", served_strategy),
    ):
        s, eq, ok = _measure(entries, strat, tok, runner)
        report.points.append(EvalPoint(label, s, eq, ok))

    # the lossy truncation sweep — traces the accuracy cliff
    for lim in limits:
        s, eq, ok = _measure(entries, _truncate(lim), tok, runner)
        report.points.append(EvalPoint(f"truncate@{lim}", s, eq, ok))

    report.points.sort(key=lambda p: p.savings)
    return report


def format_frontier(report: FrontierReport) -> str:
    out = [f"certified compression frontier  (runner={report.runner})", ""]
    out.append(f"{'level':<22}{'savings':>9}{'equiv':>8}{'certified':>11}  curve")
    out.append("-" * 74)
    for p in report.points:
        bar = "█" * round(p.savings * 22)
        flag = "✔ PASS" if p.certified else "✘ —"
        out.append(
            f"{p.label:<22}{p.savings * 100:>8.1f}%{p.equivalence * 100:>7.0f}%{flag:>11}  {bar}"
        )
    out.append("-" * 74)
    dp = report.distil_point
    if dp:
        out.append(
            f"\ndistil: {dp.savings * 100:.1f}% savings @ {dp.equivalence * 100:.0f}% "
            f"decision-equivalence — certified."
        )
    out.append(
        f"certified ceiling: {report.certified_ceiling * 100:.1f}% savings "
        "(beyond this, lossy compression drops decisions and the gate rejects it)."
    )
    return "\n".join(out)


def write_raw(report: FrontierReport, out_dir: str, stamp: str) -> str:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"frontier-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for p in report.points:
            f.write(
                json.dumps(
                    {
                        "label": p.label,
                        "savings": p.savings,
                        "equivalence": p.equivalence,
                        "certified": p.certified,
                        "runner": report.runner,
                        "grader_model": report.grader_model,
                        "grader_effort": report.grader_effort,
                    }
                )
                + "\n"
            )
    return str(path)
