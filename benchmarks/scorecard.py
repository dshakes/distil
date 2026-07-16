"""Referee scorecard — run distil's 5 behavioral invariants against every compressor adapter
and produce a comparative invariant table.

Fairness semantics (per-cell):

  reversibility   — Behavioral + architectural. A compressor PASSES if every compressed block
                    is either byte-identical to the original (no compression applied) OR is
                    recoverable to its exact original bytes via a documented recovery mechanism.
                    Distil stores originals in a local RestoreStore keyed by 8-hex SHA-256
                    handles embedded in the output; this is tested by the full harness.
                    Compressors without a recovery mechanism that change any byte score FAIL —
                    truncation loses bytes observably; that is the test.

  reject-if-bigger — Behavioral. PASS if every output block is no larger than its input plus a
                    64-byte slack (same tolerance as distil.harness). FAIL if any block grows.

  recency-exact   — Behavioral. PASS if the most-recent tool result (the block carrying the
                    sentinel "Zq7Xmarker KEEP EXACT") is present byte-identical in the output.
                    Cases with no such sentinel are skipped (not applicable for that case).

  fail-open       — Behavioral. PASS if the compressor raises no exception on any adversarial
                    input (huge single lines, unicode surrogates, nested JSON, handle-injection,
                    secret-looking strings, malformed/None content, empty tool results).

  content-free    — Architectural. Distil's on-disk telemetry surfaces (ledger, sessions,
                    flywheel, calibration) must hold zero prompt/response/tool text after a
                    recorded run — proved by a unique content marker. Compressors with no
                    on-disk state cannot have a content leak by construction; they score "n/a"
                    (the invariant is inapplicable, not failing — a library with no telemetry
                    cannot be graded on its telemetry's content-safety).

Grading:
  • Distil: graded by distil.harness.run() — the full real-path adversarial harness
    (12 hostile cases × 5 invariants = 60 checks) against the actual proxy path.
  • All other compressors: text-level checks applied to adversarial content extracted from
    the same 12 cases (tool_result text extracted verbatim). This is the honest
    comparison surface: the other adapters expose a list[str]→list[str] interface, so
    invariants are checked at that level rather than at the full messages level.

Run:  python benchmarks/scorecard.py
Writes: benchmarks/scorecard.json  (next to this file)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

# same slack the harness uses for "reject-if-bigger" to tolerate marker overhead on tiny inputs
_SLACK = 64
_KEEP_EXACT = "Zq7Xmarker KEEP EXACT"

CompressorFn = Callable[[list[str]], list[str]]

INVARIANTS = [
    "reversibility",
    "reject-if-bigger",
    "recency-exact",
    "fail-open",
    "content-free",
]


# ---------------------------------------------------------------------------
# Text extraction from harness adversarial cases
# ---------------------------------------------------------------------------


def _extract_texts(messages: list[dict]) -> list[str]:
    """Flatten all tool_result text content from a message list into a plain list of strings.

    This is the extraction surface for text-level compressors: we pull out every
    tool_result string, preserving order. Empty strings are dropped so compressors
    don't see trivial no-ops as their first input.
    """
    out: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            val = block.get("content")
            if isinstance(val, str):
                if val:
                    out.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and item.get("type") == "text":
                        t = item.get("text", "")
                        if t:
                            out.append(t)
    return out


def _load_cases() -> list[tuple[str, list[str]]]:
    """Load the same 12 adversarial cases as harness.py and extract their tool_result texts."""
    from distil.harness import _cases  # private but stable — same package

    return [(name, _extract_texts(msgs)) for name, msgs in _cases()]


# ---------------------------------------------------------------------------
# Text-level invariant checks (for non-distil compressors)
# ---------------------------------------------------------------------------


def _check_reject_if_bigger(
    fn: CompressorFn, cases: list[tuple[str, list[str]]]
) -> tuple[str, str]:
    for name, texts in cases:
        if not texts:
            continue
        try:
            outputs = fn(list(texts))
        except Exception as exc:
            return "FAIL", f"{name}: raised {type(exc).__name__}: {exc}"
        for inp, out in zip(texts, outputs):
            if len(out) > len(inp) + _SLACK:
                return "FAIL", f"{name}: output ({len(out)}) > input ({len(inp)}) + {_SLACK}"
    return "PASS", ""


def _check_recency_exact(fn: CompressorFn, cases: list[tuple[str, list[str]]]) -> tuple[str, str]:
    """The most-recent tool result (the one carrying _KEEP_EXACT) must survive verbatim."""
    for name, texts in cases:
        combined = "\n".join(texts)
        if _KEEP_EXACT not in combined:
            continue  # this case has no marked recent turn — skip
        try:
            outputs = fn(list(texts))
        except Exception as exc:
            return "FAIL", f"{name}: raised {type(exc).__name__}: {exc}"
        if _KEEP_EXACT not in "\n".join(outputs):
            return "FAIL", f"{name}: recency sentinel absent from output"
    return "PASS", ""


def _check_fail_open(fn: CompressorFn, cases: list[tuple[str, list[str]]]) -> tuple[str, str]:
    for name, texts in cases:
        try:
            fn(list(texts))
        except Exception as exc:
            return "FAIL", f"{name}: {type(exc).__name__}: {exc}"
    return "PASS", ""


def _check_reversibility(fn: CompressorFn, cases: list[tuple[str, list[str]]]) -> tuple[str, str]:
    """Behavioral: PASS only if no text is changed (no compression), or there is a documented
    recovery mechanism.  For text-level compressors presented here, any byte change is
    irrecoverable — we test observable behavior, not claims."""
    for name, texts in cases:
        if not texts:
            continue
        try:
            outputs = fn(list(texts))
        except Exception:
            return "FAIL", "raised exception"
        for inp, out in zip(texts, outputs):
            if inp != out:
                return "FAIL", f"{name}: output differs from input; no recovery mechanism"
    return "PASS", ""  # trivially reversible — no compression applied


# ---------------------------------------------------------------------------
# Baseline strategy wrapper → CompressorFn
# ---------------------------------------------------------------------------


def _wrap_strategy(strategy: object, kind_name: str = "TOOL_OUTPUT") -> CompressorFn:
    """Adapt a baselines.py (blocks, turn_index) → blocks strategy to the CompressorFn interface."""
    from distil.trajectory import Block, Kind, Stability

    k = Kind[kind_name]

    def compress(texts: list[str]) -> list[str]:
        blocks = [
            Block(id=f"b{i}", kind=k, text=t, stability=Stability.VOLATILE)
            for i, t in enumerate(texts)
        ]
        out_blocks = strategy(blocks, 0)  # type: ignore[call-arg]
        return [b.text for b in out_blocks]

    return compress


def _build_baseline_adapters() -> list[tuple[str, CompressorFn]]:
    """Build the no-dependency structural baselines, wrapped as CompressorFn callables."""
    from benchmarks.baselines import (  # type: ignore[import]
        keep_last_k_turns,
        recomp_extractive,
        recency_window,
        selective_context,
        truncate_head,
    )

    return [
        ("truncate@500", _wrap_strategy(truncate_head(500))),
        ("recency-w@500", _wrap_strategy(recency_window(500))),
        # keep-last-3-turns targets HISTORY blocks; present it on HISTORY so it exercises compression
        ("keep-last-3", _wrap_strategy(keep_last_k_turns(3), kind_name="HISTORY")),
        ("recomp-extr", _wrap_strategy(recomp_extractive())),
        ("sel-context", _wrap_strategy(selective_context())),
    ]


# ---------------------------------------------------------------------------
# Optional external adapters (headroom, llmlingua)
# ---------------------------------------------------------------------------


def _try_headroom() -> tuple[str, CompressorFn | None, str]:
    try:
        from benchmarks.headroom_adapter import compress as hr_compress  # type: ignore[import]

        # Probe that the underlying package is present by importing it
        import headroom  # noqa: F401

        return "headroom", hr_compress, ""
    except ImportError:
        return "headroom", None, "not installed — skipped (pip install headroom-ai)"


def _try_llmlingua() -> tuple[str, CompressorFn | None, str]:
    try:
        from benchmarks.llmlingua_adapter import compress as ll_compress  # type: ignore[import]
        import llmlingua  # noqa: F401

        return "llmlingua-2", ll_compress, ""
    except ImportError:
        return "llmlingua-2", None, "not installed — skipped (pip install llmlingua)"


# ---------------------------------------------------------------------------
# Distil: graded via the full real-path harness
# ---------------------------------------------------------------------------


def _grade_distil() -> dict[str, str]:
    """Run distil.harness.run() and return per-invariant PASS/FAIL."""
    from distil.harness import run  # type: ignore[import]

    rep = run(verbose=False)
    failed = {f["invariant"] for f in rep["failures"]}
    return {inv: ("FAIL" if inv in failed else "PASS") for inv in INVARIANTS}


# ---------------------------------------------------------------------------
# Grade a text-level compressor
# ---------------------------------------------------------------------------


def _grade_compressor(
    name: str, fn: CompressorFn, cases: list[tuple[str, list[str]]]
) -> dict[str, str]:
    checks: dict[str, tuple[str, str]] = {
        "reversibility": _check_reversibility(fn, cases),
        "reject-if-bigger": _check_reject_if_bigger(fn, cases),
        "recency-exact": _check_recency_exact(fn, cases),
        "fail-open": _check_fail_open(fn, cases),
        "content-free": ("n/a", "no on-disk state — inapplicable"),
    }
    return {k: v[0] for k, v in checks.items()}


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _render_table(columns: list[tuple[str, dict[str, str]]], notes: list[str]) -> str:
    col_names = [c[0] for c in columns]
    col_data = [c[1] for c in columns]

    # column widths: max of header length and widest cell value across all invariants
    widths = [
        max(len(n), max((len(d.get(i, "")) for i in INVARIANTS), default=0))
        for n, d in zip(col_names, col_data)
    ]
    inv_width = max(len(inv) for inv in INVARIANTS)

    sep = "  " + "-" * inv_width + "  " + "  ".join("-" * w for w in widths)
    header = (
        "  " + " " * inv_width + "  " + "  ".join(n.ljust(w) for n, w in zip(col_names, widths))
    )

    lines = [header, sep]
    for inv in INVARIANTS:
        row = (
            "  "
            + inv.ljust(inv_width)
            + "  "
            + "  ".join((d.get(inv, "")).ljust(w) for d, w in zip(col_data, widths))
        )
        lines.append(row)

    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"  note: {note}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("distil invariant scorecard\n")

    cases = _load_cases()

    # --- distil (full harness) ---
    print("grading: distil (real-path harness, 60 checks)…")
    distil_grades = _grade_distil()
    columns: list[tuple[str, dict[str, str]]] = [("distil", distil_grades)]
    notes: list[str] = []

    # --- structural baselines ---
    print("grading: structural baselines…")
    for label, fn in _build_baseline_adapters():
        print(f"  {label}…")
        columns.append((label, _grade_compressor(label, fn, cases)))

    # --- optional external adapters ---
    for factory in (_try_headroom, _try_llmlingua):
        label, fn, note = factory()
        if fn is None:
            notes.append(f"{label}: {note}")
            columns.append((label, {inv: "skip" for inv in INVARIANTS}))
        else:
            print(f"grading: {label}…")
            columns.append((label, _grade_compressor(label, fn, cases)))

    print()
    table = _render_table(columns, notes)
    print(table)

    # --- write JSON ---
    out = {
        "cases": 12,
        "compressors": [{"name": name, "grades": grades} for name, grades in columns],
        "notes": notes,
    }
    out_path = Path(__file__).parent / "scorecard.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Exit non-zero if distil fails any invariant (the thing we actually gate on)
    if any(v == "FAIL" for v in distil_grades.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
