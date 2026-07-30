"""Fact-level retention — recall, and the value of being reversible.

Every other gate distil ships answers a *block* question: `verify` proves the bytes
round-trip, `certify`/`shadow` prove the next action is unchanged. None answers the
question a user actually asks about a compressor: **of the load-bearing facts in my
tool output — the numbers, the paths, the error lines — how many are still in front
of the model?**

This module answers it, offline and at zero cost, with a tri-state per fact:

- ``retained``    — present in the compressed text, verbatim or after a legitimate
                    format conversion (JSON reshaped into a table/KV row).
- ``recoverable`` — absent from the compressed text, but provably reachable: the
                    block's digest marker carries a handle whose restore entry
                    *contains that fact*. One ``distil_expand`` call away.
- ``lost``        — absent, with no recovery path. The only bucket that can make an
                    agent answer wrong.

Two numbers fall out, and the gap between them is the whole product:

    visible recall = retained / total              what the model sees for free
    true recall    = (retained + recoverable) / total    the information bound

A lossy compressor has one number, because for it recoverable is always zero. The
gap is exactly what reversibility buys, quantified.

Recall is deliberately the headline, not F1. The loss is asymmetric: a dropped fact
is a wrong answer, while retained-but-unneeded content is merely fewer tokens saved.
Precision belongs to the savings number, which distil already reports everywhere.

Recoverability here is **block-scoped and verified** — the handle must appear in
that block's own compressed text, and the fact must appear in that handle's restore
bytes. It is never inferred from a marker sitting elsewhere in the prompt, so the
percentages are absolute, not merely comparative.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    _HAVE_FCNTL = False

from .compress.keep_policy import _ERR_RE
from .compress.tier0 import Tier0Lossless
from .compress.tier1 import Tier1Reversible
from .corpus import CorpusEntry, load_corpus
from .tokenizer import DEFAULT as _TOK
from .trajectory import Block, Kind, Stability

DIMENSIONS = ("numerics", "artifacts", "errors")

# A number with its key context ("retry_limit: 3", "port=8787", "invoices=88ms").
# Bare numbers are skipped: without a key they are unverifiable noise.
#
# The lookarounds and the trailing unit matter more than they look. Without them the
# match ends mid-token — "invoices=88" clipped out of "invoices=88ms", and junk like
# "T09:14" carved out of "2026-06-21T09:14:02Z". Both make bad probes: the first loses
# the unit and would false-match "invoices=889", the second is not a fact at all. They
# also cannot satisfy a right-boundary test, which is what surfaced this while
# tightening `_in_recovery` for the audit finding.
_NUMERIC_RE = re.compile(
    r"(?<![\w:.-])[A-Za-z_][\w.-]{0,24}\"?[ =:]{1,3}\d+(?:\.\d+)?[A-Za-z%]*(?![\w:.-])"
)
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")
_PATH_RE = re.compile(r"(?:~/|\.{1,2}/|/)?(?:[\w.-]+/){2,}[\w.@-]+")
# Requires at least one a-f, so decimal runs (timestamps, row counts) are not
# mistaken for content hashes.
_HEX_RE = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}\b")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_ARTIFACT_RES = (_URL_RE, _PATH_RE, _HEX_RE, _UUID_RE)

_HANDLE_RE = re.compile(r"handle=([0-9a-f]{8})")
# Collapse punctuation that format conversions rewrite, keeping the characters that
# make a path/url/hash what it is.
_NORMALIZE_RE = re.compile(r"[^\w./-]+")
_NUMERIC_SPLIT_RE = re.compile(r"(.+?)[\"' =:]+(\d+(?:\.\d+)?)$")

_MIN_TARGET_LEN = 4
_ERROR_PREFIX_LEN = 160
# Below this, compression is effectively an identity transform and any recall number
# it produces is arithmetic, not evidence. Gates must refuse to pass on it.
_MIN_ENGAGED_SAVINGS = 0.01
_SAVINGS_BUCKETS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))

_T0 = Tier0Lossless()
_T1 = Tier1Reversible()


@dataclass
class Tally:
    """Fact counts for one dimension."""

    total: int = 0
    retained: int = 0
    recoverable: int = 0

    @property
    def lost(self) -> int:
        return self.total - self.retained - self.recoverable

    @property
    def visible_recall(self) -> float:
        """Fraction present without an expand call."""
        return self.retained / self.total if self.total else 1.0

    @property
    def recall(self) -> float:
        """Fraction still reachable — the information bound."""
        return (self.retained + self.recoverable) / self.total if self.total else 1.0

    def add(self, other: Tally) -> None:
        self.total += other.total
        self.retained += other.retained
        self.recoverable += other.recoverable

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "retained": self.retained,
            "recoverable": self.recoverable,
            "lost": self.lost,
            "visible_recall": round(self.visible_recall, 4),
            "recall": round(self.recall, 4),
        }


@dataclass
class BlockProbe:
    """Retention outcome for one compressed block."""

    block_id: str
    savings: float  # fraction of characters removed from this block
    dims: dict[str, Tally]
    domain: str = ""  # corpus domain this block came from; "" outside the corpus


@dataclass
class MacroAverage:
    """Per-domain mean, where every domain counts once.

    The fact-weighted (micro) aggregate is dominated by whichever domain happens to
    carry the most probe-able facts. Adding one HTML trajectory took the corpus from
    417 facts to 1083 and moved the reversibility figure from 9.8% to 62.6% — not
    because anything improved, but because HTML pages are dense in `href` URLs and that
    one domain became 61% of the fact count. A headline a single fixture can swing is
    not measuring the compressor.

    So the macro average is the headline: mean the per-domain ratios, not the raw
    counts. Micro is still reported beside it, because the two disagreeing is itself
    information — it says the corpus is unbalanced.
    """

    domains: int = 0
    visible_recall: float = 1.0
    recall: float = 1.0

    @property
    def gap(self) -> float:
        """What reversibility is worth: recoverable-but-not-visible, per domain."""
        return self.recall - self.visible_recall

    def to_dict(self) -> dict[str, float | int]:
        return {
            "domains": self.domains,
            "visible_recall": round(self.visible_recall, 4),
            "recall": round(self.recall, 4),
            "gap": round(self.gap, 4),
        }


@dataclass
class RetentionReport:
    probes: list[BlockProbe] = field(default_factory=list)

    def aggregate(self) -> dict[str, Tally]:
        totals = {name: Tally() for name in DIMENSIONS}
        for p in self.probes:
            for name, tally in p.dims.items():
                totals[name].add(tally)
        return totals

    def overall(self) -> Tally:
        combined = Tally()
        for tally in self.aggregate().values():
            combined.add(tally)
        return combined

    def by_savings_bucket(self) -> dict[str, Tally]:
        buckets: dict[str, Tally] = {}
        for low, high in _SAVINGS_BUCKETS:
            label = f"{low:.0%}-{min(high, 1.0):.0%}"
            tally = Tally()
            for p in self.probes:
                if low <= p.savings < high:
                    for dim in p.dims.values():
                        tally.add(dim)
            buckets[label] = tally
        return buckets

    @property
    def lost_facts(self) -> int:
        return self.overall().lost

    def by_domain(self) -> dict[str, Tally]:
        """Fact tallies per corpus domain, in first-seen order."""
        out: dict[str, Tally] = {}
        for p in self.probes:
            tally = out.setdefault(p.domain or "(uncategorised)", Tally())
            for dim in p.dims.values():
                tally.add(dim)
        return out

    def macro(self) -> MacroAverage:
        """Per-domain mean — the headline, because it cannot be swung by one fixture."""
        per = [t for t in self.by_domain().values() if t.total]
        if not per:
            return MacroAverage()
        return MacroAverage(
            domains=len(per),
            visible_recall=sum(t.visible_recall for t in per) / len(per),
            recall=sum(t.recall for t in per) / len(per),
        )

    def to_dict(self) -> dict[str, object]:
        overall = self.overall()
        return {
            "blocks_probed": len(self.probes),
            # Macro first: it is the headline, and ordering the payload the same way the
            # report reads keeps a consumer from grabbing the swingable number by habit.
            "macro": self.macro().to_dict(),
            "micro": {
                "visible_recall": round(overall.visible_recall, 4),
                "recall": round(overall.recall, 4),
                "gap": round(overall.recall - overall.visible_recall, 4),
                "note": "fact-weighted; moves with corpus composition",
            },
            "overall": overall.to_dict(),
            "by_domain": {k: v.to_dict() for k, v in self.by_domain().items()},
            "by_dimension": {n: t.to_dict() for n, t in self.aggregate().items()},
            "by_savings_bucket": {k: v.to_dict() for k, v in self.by_savings_bucket().items()},
        }


def extract_targets(text: str) -> dict[str, set[str]]:
    """Pull probe-able facts out of an ORIGINAL block, per dimension."""
    targets: dict[str, set[str]] = {name: set() for name in DIMENSIONS}
    targets["numerics"] = {m for m in _NUMERIC_RE.findall(text) if len(m) >= _MIN_TARGET_LEN}
    for pattern in _ARTIFACT_RES:
        targets["artifacts"].update(m for m in pattern.findall(text) if len(m) >= _MIN_TARGET_LEN)
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= _MIN_TARGET_LEN and _ERR_RE.search(stripped):
            targets["errors"].add(stripped[:_ERROR_PREFIX_LEN])
    return targets


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text).strip()


def _in_recovery(value: str, recovery: str) -> bool:
    """Is `value` present in the recovered bytes, at token boundaries?

    A bare substring test inflates recoverable recall on short values: the gold answer
    "12" matches inside "file-12.csv", and "5" matches almost anything. Boundaries are
    checked with lookarounds rather than ``\\b`` so values that begin or end with
    punctuation still anchor correctly. (Audit finding: short dataset answers were
    false-matching against unrelated paths and hashes in other digested documents.)
    """
    if not value or not recovery:
        return False
    if len(value) < _MIN_TARGET_LEN:
        # Very short values ("12", "5") still slip through punctuation boundaries —
        # "12" is a token inside "file-12.csv". Require real whitespace/string edges so
        # only a standalone occurrence counts. Erring toward NOT crediting recovery is
        # the safe direction for a metric whose job is to surface loss.
        return re.search(rf"(?:^|\s){re.escape(value)}(?:\s|$)", recovery) is not None
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", recovery) is not None


def _survives(dimension: str, value: str, haystack: str, normalized: str) -> bool:
    """True if `value` is still present, allowing for legitimate reshaping."""
    if value in haystack:
        return True
    norm_value = _normalize(value)
    if norm_value and norm_value in normalized:
        return True
    if dimension == "errors":
        # A reshaped row drops the JSON key prefix ('"msg": "Error…"' -> a bare
        # cell); the error substance is what has to survive, not the key.
        _, _, remainder = norm_value.partition(" ")
        if len(remainder) >= _MIN_TARGET_LEN and remainder in normalized:
            return True
    if dimension == "numerics":
        # A table separates key from value; count it retained only if BOTH survive,
        # so a stray matching integer elsewhere cannot fake a hit.
        match = _NUMERIC_SPLIT_RE.match(value)
        if match:
            key, number = match.groups()
            norm_key = _normalize(key)
            if norm_key and norm_key in normalized:
                return bool(re.search(rf"\b{re.escape(number)}\b", normalized))
    return False


def probe_block(
    original: Block, compressed_text: str, restore: dict[str, str], *, domain: str = ""
) -> BlockProbe:
    """Classify every fact in `original` as retained / recoverable / lost.

    Recoverability is proven, not assumed: the fact must appear in the restore bytes
    of a handle that this block's own compressed text carries (or of this block's id,
    for a Tier-0 reshape whose marker carries no handle).
    """
    normalized = _normalize(compressed_text)
    keys = _HANDLE_RE.findall(compressed_text)
    if original.id in restore:
        keys.append(original.id)
    recovery = "\n".join(restore[k] for k in keys if k in restore)

    dims: dict[str, Tally] = {}
    for name, values in extract_targets(original.text).items():
        tally = Tally(total=len(values))
        for value in values:
            if _survives(name, value, compressed_text, normalized):
                tally.retained += 1
            elif _in_recovery(value, recovery):
                tally.recoverable += 1
        dims[name] = tally

    savings = 1.0 - (len(compressed_text) / len(original.text)) if original.text else 0.0
    return BlockProbe(original.id, max(savings, 0.0), dims, domain=domain)


def probe_trajectory(entry: CorpusEntry, report: RetentionReport | None = None) -> RetentionReport:
    """Run the real Tier-1 -> Tier-0 pipeline over one trajectory and score retention.

    Mirrors ``compress.strategies.distil``: the cacheable prefix is left alone and the
    VOLATILE tail is digested. Only the tail is probed, because it is the only text
    compression touches.
    """
    report = report if report is not None else RetentionReport()
    for turn in entry.trajectory.turns:
        volatile = [b for b in turn.blocks if b.stability is Stability.VOLATILE]
        if not volatile:
            continue
        t1 = _T1.compress(volatile)
        t0 = _T0.compress(t1.blocks)
        restore = {**t1.restore, **t0.restore}
        by_id = {b.id: b.text for b in t0.blocks}
        for original in volatile:
            compressed_text = by_id.get(original.id, "")
            report.probes.append(
                probe_block(original, compressed_text, restore, domain=entry.domain)
            )
    return report


def run(entries: list[CorpusEntry] | None = None) -> RetentionReport:
    entries = entries if entries is not None else load_corpus()
    report = RetentionReport()
    for entry in entries:
        probe_trajectory(entry, report)
    return report


# ---------------------------------------------------------------------------
# public-benchmark scoring (third-party ground truth)
# ---------------------------------------------------------------------------


def _phrase_survives(value: str, haystack: str, normalized: str) -> bool:
    """Strict survival for a gold answer / gold sentence: the WHOLE phrase must be
    present, verbatim or with punctuation normalized. A partially kept sentence is
    not retention — a model cannot answer from half a fact."""
    if value in haystack:
        return True
    norm = _normalize(value)
    return bool(norm) and norm in normalized


@dataclass
class CaseScore:
    """One benchmark question, scored against its third-party answer key."""

    case_id: str
    savings: float  # token reduction over the retrieved docs
    answer_retained: bool | None  # None => ungraded, see `ungraded`
    answer_recoverable: bool = False
    ungraded: str = ""  # "unanswerable" | "abstractive" | ""
    support_total: int = 0
    support_retained: int = 0
    support_recoverable: int = 0


@dataclass
class DatasetReport:
    dataset: str
    shape: str = "json"
    scores: list[CaseScore] = field(default_factory=list)
    baseline: list[CaseScore] = field(default_factory=list)
    baseline_label: str = "truncation @ matched savings"

    @staticmethod
    def _answer_recall(scores: list[CaseScore], *, with_recovery: bool) -> tuple[float, int]:
        graded = [s for s in scores if s.answer_retained is not None]
        if not graded:
            return 1.0, 0
        hits = sum(
            1 for s in graded if s.answer_retained or (with_recovery and s.answer_recoverable)
        )
        return hits / len(graded), len(graded)

    @staticmethod
    def _support_recall(scores: list[CaseScore], *, with_recovery: bool) -> tuple[float, int]:
        total = sum(s.support_total for s in scores)
        if not total:
            return 1.0, 0
        hits = sum(
            s.support_retained + (s.support_recoverable if with_recovery else 0) for s in scores
        )
        return hits / total, total

    @staticmethod
    def _savings(scores: list[CaseScore]) -> float:
        return sum(s.savings for s in scores) / len(scores) if scores else 0.0

    def ungraded(self, reason: str) -> int:
        return sum(1 for s in self.scores if s.ungraded == reason)

    @property
    def savings(self) -> float:
        return self._savings(self.scores)

    @property
    def engaged(self) -> bool:
        """Did compression actually do anything? If not, a recall of 100% is the
        arithmetic of an identity function and proves nothing about fidelity."""
        return self.savings >= _MIN_ENGAGED_SAVINGS

    def to_dict(self) -> dict[str, object]:
        def block(scores: list[CaseScore]) -> dict[str, object]:
            visible_answer, graded = self._answer_recall(scores, with_recovery=False)
            true_answer, _ = self._answer_recall(scores, with_recovery=True)
            visible_support, support_total = self._support_recall(scores, with_recovery=False)
            true_support, _ = self._support_recall(scores, with_recovery=True)
            return {
                "cases": len(scores),
                "savings": round(self._savings(scores), 4),
                "answer_graded": graded,
                "answer_recall_visible": round(visible_answer, 4),
                "answer_recall": round(true_answer, 4),
                "support_facts": support_total,
                "support_recall_visible": round(visible_support, 4),
                "support_recall": round(true_support, 4),
            }

        return {
            "dataset": self.dataset,
            "shape": self.shape,
            "compression_engaged": self.engaged,
            "ungraded_unanswerable": self.ungraded("unanswerable"),
            "ungraded_abstractive": self.ungraded("abstractive"),
            "distil": block(self.scores),
            "baseline": {"label": self.baseline_label, **block(self.baseline)},
        }


def _payload_blocks(case: Any, shape: str) -> list[Block]:
    """Render retrieved docs the way the agent actually receives them.

    ``json`` (default) is the production shape: a retrieval tool returns an array of
    records, which is exactly the structure distil compresses (columnar, lossless,
    behind a handle). ``prose`` concatenates bare text — kept for comparison, but it
    is not what a RAG agent's tool result looks like, and distil's compressors
    correctly decline to touch short natural-language prose.
    """
    if shape == "prose":
        return [
            Block(id=f"doc{i}", kind=Kind.TOOL_OUTPUT, text=text, stability=Stability.VOLATILE)
            for i, (_title, text) in enumerate(case.docs)
        ]
    payload = json.dumps(
        [{"title": title, "text": text} for title, text in case.docs],
        indent=2,
        ensure_ascii=False,
    )
    return [
        Block(id="retrieval", kind=Kind.TOOL_OUTPUT, text=payload, stability=Stability.VOLATILE)
    ]


def _score_case(case: Any, *, shape: str, truncate_to: float | None = None) -> CaseScore:
    """Compress one benchmark case's retrieved docs and score the answer key.

    `truncate_to` selects the baseline: a character-fraction truncation with NO
    restore table (what every lossy compressor leaves you), instead of distil's
    reversible pipeline.

    A gold answer is only graded when it is present in the UNCOMPRESSED context.
    HotpotQA's comparison questions answer "yes"/"no", which is never a span in the
    passages — grading those would report an abstractive answer format as compression
    loss. They are excluded and counted, not silently scored either way.
    """
    blocks = _payload_blocks(case, shape)
    original = "\n".join(b.text for b in blocks)
    original_norm = _normalize(original)

    if truncate_to is None:
        t1 = _T1.compress(blocks)
        t0 = _T0.compress(t1.blocks)
        compressed, restore = t0.blocks, {**t1.restore, **t0.restore}
    else:
        keep = max(0.0, min(1.0, truncate_to))
        compressed = [b.copy_with(b.text[: int(len(b.text) * keep)]) for b in blocks]
        restore = {}  # lossy: nothing to recover from, by construction

    text = "\n".join(b.text for b in compressed)
    normalized = _normalize(text)
    recovery = "\n".join(restore.values())

    base = sum(_TOK.count(b.text) for b in blocks)
    kept = sum(_TOK.count(b.text) for b in compressed)
    savings = (1.0 - kept / base) if base else 0.0

    answer_retained: bool | None = None
    answer_recoverable = False
    ungraded = ""
    if not (getattr(case, "answerable", True) and case.answer):
        ungraded = "unanswerable"
    elif not _phrase_survives(case.answer, original, original_norm):
        ungraded = "abstractive"
    else:
        answer_retained = _phrase_survives(case.answer, text, normalized)
        if not answer_retained and recovery:
            answer_recoverable = _in_recovery(case.answer, recovery)

    retained = recoverable = 0
    gold = [s for s in case.support if _phrase_survives(s, original, original_norm)]
    for sentence in gold:
        if _phrase_survives(sentence, text, normalized):
            retained += 1
        elif _in_recovery(sentence, recovery):
            recoverable += 1

    return CaseScore(
        case_id=str(getattr(case, "id", "")),
        savings=max(savings, 0.0),
        answer_retained=answer_retained,
        answer_recoverable=answer_recoverable,
        ungraded=ungraded,
        support_total=len(gold),
        support_retained=retained,
        support_recoverable=recoverable,
    )


def score_dataset(cases: list[Any], dataset: str = "", *, shape: str = "json") -> DatasetReport:
    """Score distil against a public answer key, plus a matched-savings baseline.

    The baseline truncates each document to the character fraction that reproduces
    distil's own token savings on that case, so the two are compared at equal cost.
    Both savings figures are reported, so a reader can confirm they actually match.
    """
    if shape not in ("json", "prose"):
        raise ValueError(f"shape must be 'json' or 'prose', got {shape!r}")
    label = dataset or (getattr(cases[0], "dataset", "") if cases else "")
    report = DatasetReport(dataset=label, shape=shape)
    for case in cases:
        score = _score_case(case, shape=shape)
        report.scores.append(score)
        report.baseline.append(_score_case(case, shape=shape, truncate_to=1.0 - score.savings))
    return report


def format_dataset_report(report: DatasetReport) -> str:
    d, b = report.scores, report.baseline
    out = [
        f"public benchmark: {report.dataset}  ({len(d)} cases, third-party ground truth, "
        f"{report.shape} tool-result shape)",
        "",
        f"  {'':<28}{'savings':>9}{'answer':>9}{'support':>9}",
        "-" * 64,
    ]
    for label, scores in (("distil (reversible)", d), (report.baseline_label, b)):
        answer, _ = DatasetReport._answer_recall(scores, with_recovery=True)
        support, _ = DatasetReport._support_recall(scores, with_recovery=True)
        out.append(
            f"  {label:<28}{DatasetReport._savings(scores):>8.1%}{answer:>9.1%}{support:>9.1%}"
        )
    out.append("-" * 64)

    visible_answer, graded = DatasetReport._answer_recall(d, with_recovery=False)
    true_answer, _ = DatasetReport._answer_recall(d, with_recovery=True)
    visible_support, support_total = DatasetReport._support_recall(d, with_recovery=False)
    true_support, _ = DatasetReport._support_recall(d, with_recovery=True)

    out += [
        "",
        f"answer recall   {true_answer:.1%}  ({visible_answer:.1%} visible, "
        f"{graded} gold answers graded)",
    ]
    if support_total:
        out.append(
            f"support recall  {true_support:.1%}  ({visible_support:.1%} visible, "
            f"{support_total} gold sentences)"
        )
    for reason, note in (
        ("unanswerable", "marked unanswerable by the dataset — no answer to retain"),
        ("abstractive", "answer is not a span in the passage (e.g. yes/no) — unprobeable"),
    ):
        count = report.ungraded(reason)
        if count:
            out.append(f"excluded        {count} {note}")

    base_answer, _ = DatasetReport._answer_recall(b, with_recovery=True)
    out.append(
        f"\nat the same savings, a lossy compressor retains {base_answer:.1%} of gold answers "
        f"— distil is {true_answer - base_answer:+.1%}."
    )
    if not report.engaged:
        out.append(
            f"\nWARNING: compression barely engaged ({report.savings:.1%} savings) — recall here "
            "is the arithmetic of a near-identity transform, NOT evidence of fidelity."
        )
    elif graded and visible_answer < 0.5 <= true_answer:
        # Nothing is lost, but the agent pays a round trip for most answers. A user
        # planning capacity needs that number, not just the reassuring one.
        out.append(
            f"\nNOTE: only {visible_answer:.1%} of answers are visible without a tool call — at "
            "this aggressiveness the agent recovers most detail via distil_expand, which is "
            "lossless but costs a round trip."
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# live meter — real traffic, content-free
# ---------------------------------------------------------------------------
#
# The offline probes above grade the corpus and public benchmarks. Neither is YOUR
# traffic. The obvious way to close that — record real sessions to disk and score them
# later — would mean writing prompts and tool output in plaintext, and distil's whole
# privacy posture is that no content ever lands on disk. So the meter scores
# in-process, where the original and compressed text are both already in memory, and
# persists COUNTS ONLY. Nothing recoverable, nothing to leak, no recorder to secure.
#
# It sits in the request path, so it obeys the same three rules as the rest of the
# path: sampled (default off, like shadow mode), bounded (a hard char budget per
# request), and fail-open (an exception here must never cost a user their request).

_LIVE_MAX_CHARS = 65_536  # per request, so one huge tool result cannot stall a proxy


def _tool_result_texts(messages: Any) -> list[str]:
    """Tool-result text from Anthropic (`tool_result` blocks) and OpenAI (`role:tool`)."""
    out: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if msg.get("role") == "tool":
            out.append(content if isinstance(content, str) else _flatten_text(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out.append(_flatten_text(block.get("content")))
    return [t for t in out if t]


def _flatten_text(node: Any) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if node.get("type") == "text":
            return str(node.get("text", ""))
        return "\n".join(_flatten_text(v) for v in node.values())
    if isinstance(node, list):
        return "\n".join(_flatten_text(item) for item in node)
    return ""


def measure_live(original: Any, compressed: Any, store: Any = None) -> dict[str, Tally]:
    """Score one real request's fact retention. Returns counts only — never content.

    `store` is the live RestoreStore; a missing fact counts as recoverable only when a
    handle present in the compressed text expands to bytes that actually contain it —
    the same verified standard the offline probe uses.
    """
    original_text = "\n".join(_tool_result_texts(original))[:_LIVE_MAX_CHARS]
    dims = {name: Tally() for name in DIMENSIONS}
    if not original_text:
        return dims

    compressed_text = _flatten_text(compressed)
    normalized = _normalize(compressed_text)

    recovery = ""
    if store is not None:
        # Only handles this store already holds IN MEMORY. RestoreStore.expand() falls
        # back to a synchronous disk read for unknown handles, and this runs in the
        # request path — a long transcript, or tool output that merely CONTAINS the text
        # "handle=deadbeef", would turn one sampled request into many file reads. The
        # in-memory set is also the only recoverability we can prove for this turn.
        # (Audit finding: synchronous disk I/O in the hot path.)
        known = getattr(store, "handles", None) or frozenset()
        parts: list[str] = []
        for handle in set(_HANDLE_RE.findall(compressed_text)) & set(known):
            try:
                parts.append(store.expand(handle) or "")
            except Exception:  # noqa: BLE001 — a missing handle is not a crash
                continue
        recovery = "\n".join(parts)

    for name, values in extract_targets(original_text).items():
        tally = dims[name]
        tally.total = len(values)
        for value in values:
            if _survives(name, value, compressed_text, normalized):
                tally.retained += 1
            elif _in_recovery(value, recovery):
                tally.recoverable += 1
    return dims


class LiveMeter:
    """Sampled, content-free retention meter for the request path.

    `rate` <= 0 disables it entirely (the default), matching shadow mode: a feature
    that adds latency to real traffic is opt-in, not opt-out.
    """

    def __init__(self, rate: float, *, rng: Any = None, path: Path | None = None) -> None:
        self.rate = max(0.0, min(1.0, rate))
        self._rng = rng or random.Random()
        self._path = path

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def should_sample(self) -> bool:
        return self.enabled and self._rng.random() < self.rate

    def observe(self, original: Any, compressed: Any, store: Any = None) -> None:
        """Score and persist one request if sampled. Never raises."""
        try:
            if not self.should_sample():
                return
            dims = measure_live(original, compressed, store)
            if any(t.total for t in dims.values()):
                self._append(dims)
        except Exception:  # noqa: BLE001 — the meter must never break a request
            pass

    def _append(self, dims: dict[str, Tally]) -> None:
        path = self._path or (_live_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ints only. There is deliberately no field here that could carry content.
        record = {
            "ts": time.time(),
            "dims": {
                name: [tally.total, tally.retained, tally.recoverable]
                for name, tally in dims.items()
            },
        }
        with path.open("a", encoding="utf-8") as fh:
            # Concurrent wrap sessions and proxy workers append here; lock like
            # ledger.py and shadow.py do.
            if _HAVE_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(json.dumps(record) + "\n")
                fh.flush()
            finally:
                if _HAVE_FCNTL:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _live_path() -> Path:
    home = Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))
    return home / "retention.jsonl"


def load_live(path: Path | None = None) -> tuple[dict[str, Tally], int]:
    """Aggregate the live meter's rows. Returns (per-dimension tallies, request count)."""
    target = path or _live_path()
    dims = {name: Tally() for name in DIMENSIONS}
    requests = 0
    if not target.exists():
        return dims, 0
    with target.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                row = record["dims"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            requests += 1
            for name, triple in row.items():
                if name in dims and isinstance(triple, list) and len(triple) == 3:
                    total, retained, recoverable = (int(v) for v in triple)
                    dims[name].add(Tally(total, retained, recoverable))
    return dims, requests


def format_live(dims: dict[str, Tally], requests: int) -> str:
    combined = Tally()
    for tally in dims.values():
        combined.add(tally)
    if not requests or not combined.total:
        return (
            "no live retention samples yet.\n\n"
            "Start the proxy with --retention-rate 0.05 to sample 5% of real requests "
            "(content-free: only counts are stored)."
        )
    out = [
        f"live fact retention  ({requests} sampled requests, real traffic)",
        "",
        f"  {'dimension':<14}{'recall':>7}{'visible':>10}{'lost':>7}{'facts':>8}",
        "-" * 66,
    ]
    for name, tally in dims.items():
        out.append(_row(name, tally))
    out.append("-" * 66)
    out.append(_row("ALL", combined))
    out += [
        "",
        f"recall {combined.recall:.1%} on your own traffic — {combined.lost} facts lost, "
        f"{combined.recoverable} recovered via distil_expand.",
    ]
    return "\n".join(out)


def _bar_char() -> str:
    """The block glyph where the console can encode it, ASCII where it cannot.

    U+2588 is absent from cp1252, so printing it to a stock Windows console raises
    UnicodeEncodeError — which took down the retention gate on the windows-latest CI
    leg while every POSIX leg passed. An em dash survives there (cp1252 has one), so
    the rest of this module's output is safe; only the bar needed degrading.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "█".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return "#"
    return "█"


def _row(label: str, tally: Tally) -> str:
    if not tally.total:
        return f"  {label:<14}       n/a (0 facts)"
    bar = _bar_char() * round(tally.recall * 20)
    return (
        f"  {label:<14}{tally.recall:>7.1%}{tally.visible_recall:>10.1%}"
        f"{tally.lost:>7}{tally.total:>8}  {bar}"
    )


def format_report(report: RetentionReport) -> str:
    overall = report.overall()
    out = [
        f"fact-level retention  ({len(report.probes)} compressed blocks probed)",
        "",
        f"  {'dimension':<14}{'recall':>7}{'visible':>10}{'lost':>7}{'facts':>8}",
        "-" * 66,
    ]
    for name, tally in report.aggregate().items():
        out.append(_row(name, tally))
    out.append("-" * 66)
    out.append(_row("ALL", overall))
    by_domain = report.by_domain()
    if len(by_domain) > 1:
        out += ["", "by domain (each counts once toward the macro average):"]
        for label, tally in by_domain.items():
            if tally.total:
                out.append(_row(label, tally))

    out += ["", "by block savings:"]
    for label, tally in report.by_savings_bucket().items():
        if tally.total:
            out.append(_row(label, tally))

    macro = report.macro()
    out += [
        "",
        f"recall {overall.recall:.1%} — {overall.retained} of {overall.total} facts stay visible, "
        f"{overall.recoverable} more are one distil_expand away, {overall.lost} lost.",
    ]
    if overall.recoverable:
        micro_gap = overall.recall - overall.visible_recall
        if macro.domains > 1:
            # Macro leads. Micro is fact-weighted, so whichever domain happens to carry
            # the most probe-able facts sets it — one HTML fixture moved it 9.8% -> 62.6%
            # without anything about the compressor changing.
            out.append(
                f"reversibility is worth {macro.gap:.1%} recall — the mean across "
                f"{macro.domains} domains, each counted once. A lossy compressor at the "
                "same savings would have dropped those facts for good."
            )
            out.append(
                f"  (fact-weighted, the same figure reads {micro_gap:.1%}; the two diverge "
                "when one domain dominates the fact count, so prefer the macro number.)"
            )
        else:
            out.append(
                f"reversibility is worth {micro_gap:.1%} recall here: a lossy compressor at "
                "the same savings would have dropped those facts for good."
            )
    return "\n".join(out)
