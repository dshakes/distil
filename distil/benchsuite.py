"""The public-benchmark suite — external validity in tiers, at zero marginal cost.

Every number distil publishes elsewhere is graded against distil's own corpus. That
is rigorous and *unfalsifiable by a stranger*. This suite answers the other question:
does the compressor hold up on data whose ground truth someone else wrote?

Two design choices separate it from the benchmark suites this space usually ships.

**It costs nothing to run.** Grading is deterministic recall against a third-party
answer key, so a tier is a few HTTP GETs and some string work — no model in the loop,
no API key, no spend cap to blow. Suites that grade with an LLM judge cost real money
per tier, which makes them something you run before a launch rather than before a
merge; a gate you skip because it costs money is not a gate. Ours runs in CI.

**It refuses to average controls with evidence.** Benchmarks differ enormously in how
much *compressible payload* they carry, and tables usually hide it. A GSM8K case is a
one-line word problem: there is nothing to compress, so an unchanged score proves the
compressor left it alone — a control, not a demonstration. A BFCL case is a JSON tool
schema; a MS MARCO case is ten retrieved passages. Only those can show compression
quality. Both kinds belong here, and each row is labelled, because reporting a
thin-payload null result as evidence is how a suite looks stronger than its data.

Tiers are ordered by evidential value, not by cost:

  1  the payload an agent proxy actually risks breaking — tool schemas and retrieval
  2  the harder payloads: long-form narrative, code, multi-passage RAG
  3  the thin-payload controls, which bound the false-positive rate
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import datasets, retention

# Tier membership. Tier 1 leads with `bfcl` deliberately: tool-calling is the failure
# an agent proxy is most likely to cause and least likely to notice, because a QA
# benchmark never asks the model to *act*.
TIERS: dict[int, tuple[str, ...]] = {
    1: ("bfcl", "hotpotqa", "squad", "gsm8k"),
    2: ("msmarco", "narrativeqa", "codesearchnet", "humaneval"),
    3: ("truthfulqa", "mmlu", "arc", "triviaqa"),
}


@dataclass
class BenchRow:
    """One benchmark's result, carrying the context needed to read it honestly."""

    name: str
    payload: str  # "rich" | "thin"
    cases: int
    savings: float
    answer_recall: float
    support_recall: float
    lost: int
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "payload": self.payload,
            "cases": self.cases,
            "savings": round(self.savings, 4),
            "answer_recall": round(self.answer_recall, 4),
            "support_recall": round(self.support_recall, 4),
            "lost": self.lost,
            "error": self.error,
        }


@dataclass
class SuiteReport:
    tiers: list[int] = field(default_factory=list)
    rows: list[BenchRow] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def graded(self) -> list[BenchRow]:
        return [r for r in self.rows if r.ok]

    @property
    def failed(self) -> list[BenchRow]:
        return [r for r in self.rows if not r.ok]

    @property
    def evidence(self) -> list[BenchRow]:
        """Rich-payload rows — the only ones that can demonstrate compression quality."""
        return [r for r in self.graded if r.payload == "rich"]

    @property
    def controls(self) -> list[BenchRow]:
        return [r for r in self.graded if r.payload == "thin"]

    @property
    def total_lost(self) -> int:
        return sum(r.lost for r in self.graded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tiers": self.tiers,
            "duration_s": round(self.duration_s, 2),
            "rows": [r.to_dict() for r in self.rows],
            "evidence_benchmarks": len(self.evidence),
            "control_benchmarks": len(self.controls),
            "total_lost": self.total_lost,
            "failed": [r.name for r in self.failed],
        }


def run(
    tiers: list[int] | None = None,
    *,
    n: int | None = None,
    offline: bool = False,
    names: list[str] | None = None,
) -> SuiteReport:
    """Grade every benchmark in the requested tiers.

    A benchmark that cannot be loaded is recorded as a FAILED row, never omitted:
    a suite that silently drops what it could not fetch reports a clean sheet for a
    run that measured less than it claimed.
    """
    started = time.time()
    wanted = sorted(tiers or [1])
    selected = list(names) if names else [d for t in wanted for d in TIERS.get(t, ())]

    report = SuiteReport(tiers=wanted)
    for name in selected:
        try:
            payload = datasets.payload_class(name)
        except datasets.DatasetUnavailable as exc:
            report.rows.append(BenchRow(name, "?", 0, 0.0, 0.0, 0.0, 0, error=str(exc)))
            continue
        try:
            cases = datasets.load(name, n, offline=offline)
            graded = retention.score_dataset(cases, name)
            report.rows.append(
                BenchRow(
                    name=name,
                    payload=payload,
                    cases=len(cases),
                    savings=_num(graded, "savings"),
                    answer_recall=_num(graded, "answer_recall"),
                    support_recall=_num(graded, "support_recall"),
                    lost=int(
                        _num(graded, "support_facts")
                        - _num(graded, "support_facts") * _num(graded, "support_recall")
                        + 0.5
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - every failure must surface as a row
            report.rows.append(
                BenchRow(name, payload, 0, 0.0, 0.0, 0.0, 0, error=f"{type(exc).__name__}: {exc}")
            )
    report.duration_s = time.time() - started
    return report


def _num(graded: Any, field_name: str) -> float:
    """Read a metric off a `DatasetReport`.

    The per-compressor numbers live in the nested ``distil`` block, not at the top
    level; reading the top level returned 0.0 for everything and produced a table of
    zeros that still looked like a completed run. Falls back to the attribute so a
    shape change surfaces as a wrong number rather than an exception mid-suite.
    """
    payload: dict[str, Any] = graded.to_dict() if hasattr(graded, "to_dict") else {}
    nested = payload.get("distil")
    block: dict[str, Any] = nested if isinstance(nested, dict) else {}
    value = block.get(field_name, payload.get(field_name, getattr(graded, field_name, None)))
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def format_report(report: SuiteReport) -> str:
    if not report.rows:
        return "benchmark suite: nothing selected — nothing to grade."
    lines = [
        f"Public benchmark suite — tier(s) {', '.join(map(str, report.tiers))}, "
        f"{len(report.graded)}/{len(report.rows)} graded in {report.duration_s:.1f}s "
        f"(no API key, no spend)",
        "",
        f"  {'benchmark':<15}{'payload':<9}{'cases':>6}{'savings':>10}"
        f"{'answer':>9}{'support':>9}{'lost':>6}",
        "  " + "-" * 64,
    ]
    for row in report.rows:
        if not row.ok:
            lines.append(f"  {row.name:<15}{'—':<9}{'FAILED':>6}   {row.error[:40]}")
            continue
        lines.append(
            f"  {row.name:<15}{row.payload:<9}{row.cases:>6}{row.savings:>9.1%}"
            f"{row.answer_recall:>9.1%}{row.support_recall:>9.1%}{row.lost:>6}"
        )
    lines += [
        "  " + "-" * 64,
        "",
        f"evidence benchmarks (rich payload): {len(report.evidence)}   "
        f"controls (thin payload): {len(report.controls)}",
        "  a control's unchanged score shows the compressor left an incompressible",
        "  prompt alone. It is not evidence of compression quality — only the rich",
        "  rows can be that, and averaging the two overstates the suite.",
    ]
    if report.failed:
        lines.append(
            f"\n{len(report.failed)} benchmark(s) could not be graded: "
            f"{', '.join(r.name for r in report.failed)}"
        )
    return "\n".join(lines)
