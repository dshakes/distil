"""The public-benchmark suite — external validity in tiers, at zero marginal cost.

Every number distil publishes elsewhere is graded against distil's own corpus. That
is rigorous and *unfalsifiable by a stranger*. This suite answers the other question:
does the compressor hold up on data whose ground truth someone else wrote?

Two design choices separate it from the benchmark suites this space usually ships.

**It costs nothing to run.** Grading is deterministic recall against a third-party
answer key, so a tier is a few HTTP GETs and some string work — no model in the loop,
no API key, no spend cap to blow. Suites that grade with an LLM judge cost real money
per tier, which makes them something you run before a launch rather than before a
merge; a gate you skip because it costs money is not a gate. This one is wired
into `make gate` and the CI gate job.

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
    # Why the row failed. Only `unavailable` — a transport or cache problem — may be
    # downgraded to a warning: an outage is not a compression regression. A `bug`
    # (any other exception, or a schema drift that leaves no adaptable rows) is our
    # own defect and must never be waited out.
    failure: str = ""  # "" | "unavailable" | "bug"
    # What was asked for, when that differs from what was graded. A split can end
    # early or an adapter can drop malformed rows; either way `-n 100` graded on 7
    # cases is a different measurement, and hiding that makes a thin run look full.
    requested: int = 0
    # How many golds of each kind were actually GRADED. A recall over zero golds is
    # vacuously 1.0, so without these counts a dimension nobody measured looks like a
    # dimension that passed.
    answer_graded: int = 0
    support_facts: int = 0
    # Did compression actually do anything on this benchmark? `retention` calls a
    # recall of 100% over an identity function "the arithmetic of an identity
    # function", and the suite was dropping that signal — so a regression that turned
    # compression OFF returned savings=0, perfect recall, and passed every threshold.
    engaged: bool = True

    @property
    def short(self) -> bool:
        return bool(self.requested) and self.cases < self.requested

    @property
    def measured(self) -> bool:
        """True when at least one dimension had golds to grade."""
        return bool(self.answer_graded or self.support_facts)

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "payload": self.payload,
            "cases": self.cases,
            "requested": self.requested,
            "answer_graded": self.answer_graded,
            "support_facts": self.support_facts,
            "engaged": self.engaged,
            "short": self.short,
            "savings": round(self.savings, 4),
            "answer_recall": round(self.answer_recall, 4),
            "support_recall": round(self.support_recall, 4),
            "lost": self.lost,
            "error": self.error,
            "failure": self.failure,
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
        """Rich rows that actually MEASURED something.

        A rich row with cases but no golds of either kind grades nothing, yet renders
        as 100%/100% because a recall over zero is 1.0. Counting it as evidence
        recreated the vacuous pass this whole suite exists to refuse, so it is
        excluded here and surfaced as `unmeasured`.
        """
        return [r for r in self.graded if r.payload == "rich" and r.measured]

    @property
    def idle(self) -> list[BenchRow]:
        """Rich rows where compression did nothing. 100% recall over an identity
        function is arithmetic, not fidelity — and it would pass every threshold."""
        return [r for r in self.graded if r.payload == "rich" and r.cases and not r.engaged]

    @property
    def unmeasured(self) -> list[BenchRow]:
        """Rich rows that graded cases but had no golds to grade them against."""
        return [r for r in self.graded if r.payload == "rich" and r.cases and not r.measured]

    @property
    def controls(self) -> list[BenchRow]:
        return [r for r in self.graded if r.payload == "thin"]

    @property
    def collapsed(self) -> list[BenchRow]:
        """Rich benchmarks that graded cases and recalled NOTHING.

        A run like that is not a low score, it is a broken one — the compressor
        returned nothing usable, or the loader handed it nothing to grade. It must
        never exit 0 just because no threshold flag happened to be passed.
        """
        out: list[BenchRow] = []
        for r in self.evidence:
            if not r.cases:
                continue
            # Only dimensions with golds to grade can testify. `retention` returns a
            # recall of 1.0 over zero facts, so a SQuAD-shaped benchmark — which has
            # no support facts at all — reported support_recall=1.0 and escaped a
            # requirement that BOTH recalls be zero. That is the same blind spot just
            # fixed in the loss count, one property over.
            measured = []
            if r.answer_graded:
                measured.append(r.answer_recall)
            if r.support_facts:
                measured.append(r.support_recall)
            if measured and max(measured) <= 0.0:
                out.append(r)
        return out

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
            "collapsed": [r.name for r in self.collapsed],
            "unmeasured": [r.name for r in self.unmeasured],
            "idle": [r.name for r in self.idle],
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
            report.rows.append(
                BenchRow(name, "?", 0, 0.0, 0.0, 0.0, 0, error=str(exc), failure="bug")
            )
            continue
        try:
            # The effective request size: `n` when given, else the spec's own
            # default. Recording 0 for a default run meant `short` could never be
            # true there — and the default invocation is the one users are documented
            # to run, so the shortfall signal was missing exactly where it mattered.
            want = n or datasets.SPECS[name].default_n
            cases = datasets.load(name, n, offline=offline)
            graded = retention.score_dataset(cases, name)
            report.rows.append(
                BenchRow(
                    name=name,
                    payload=payload,
                    cases=len(cases),
                    requested=want,
                    engaged=bool(getattr(graded, "engaged", True)),
                    answer_graded=int(_num(graded, "answer_graded")),
                    support_facts=int(_num(graded, "support_facts")),
                    savings=_num(graded, "savings"),
                    answer_recall=_num(graded, "answer_recall"),
                    support_recall=_num(graded, "support_recall"),
                    # Unrecoverable golds, counting BOTH kinds. Support-only counting
                    # made the gate blind on any benchmark with `support=[]` — SQuAD
                    # among them — so an answer regression scored zero loss.
                    lost=int(
                        _num(graded, "support_facts") * (1 - _num(graded, "support_recall"))
                        + _num(graded, "answer_graded") * (1 - _num(graded, "answer_recall"))
                        + 0.5
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - every failure must surface as a row
            # A DatasetUnavailable from transport or an empty cache is availability.
            # Anything else — a TypeError in an adapter, a scoring bug — is ours, and
            # so is a DatasetUnavailable that means the upstream shape changed.
            # By TYPE, not by message. `DatasetSchemaError` is a subclass, so the
            # order matters: a schema/join failure is ours, a plain
            # `DatasetUnavailable` is availability, anything else is ours.
            availability = isinstance(exc, datasets.DatasetUnavailable) and not isinstance(
                exc, datasets.DatasetSchemaError
            )
            report.rows.append(
                BenchRow(
                    name,
                    payload,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0,
                    error=f"{type(exc).__name__}: {exc}",
                    failure="unavailable" if availability else "bug",
                )
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
        seen = f"{row.cases}/{row.requested}" if row.short else str(row.cases)
        lines.append(
            f"  {row.name:<15}{row.payload:<9}{seen:>6}{row.savings:>9.1%}"
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
    short = [r for r in report.graded if r.short]
    if short:
        lines.append(
            "\n"
            + ", ".join(f"{r.name} graded {r.cases}/{r.requested}" for r in short)
            + " — split ended early or rows were unadaptable; the number is real but"
            "\n  it is not the sample size you asked for."
        )
    if report.failed:
        lines.append(
            f"\n{len(report.failed)} benchmark(s) could not be graded: "
            f"{', '.join(r.name for r in report.failed)}"
        )
    return "\n".join(lines)
