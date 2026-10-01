"""Summarize the model-migration eval flows into one committed artifact.

Every number the docs quote from the model-migration eval is recomputed here from the
per-case rows (`.claude/hillclimb/<flow>/<variant>/results.jsonl`) and the all-turns
savings report, and written to `benchmarks/results/model-migration/summary.json` — the
file `docs/claims.json` points at. Pure arithmetic over recorded rows: no API calls.

    .venv/bin/python benchmarks/model_migration_summary.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from model_migration_eval import cost_usd  # noqa: E402

HC = ROOT / ".claude/hillclimb"
OUT = ROOT / "benchmarks/results/model-migration/summary.json"


def _rows(flow: str, variant: str) -> list[dict]:
    p = HC / flow / variant / "results.jsonl"
    return [
        r
        for r in (json.loads(ln) for ln in p.read_text().splitlines() if ln.strip())
        if r["status"] == "ok"
    ]


def _rate(rows: list[dict], m: str) -> float | None:
    have = [r["grade"][m] for r in rows if m in r["grade"]]
    return round(sum(have) / len(have), 4) if have else None


def _paired(base: list[dict], var: list[dict], m: str, ids: set[str]) -> dict:
    """Mean paired difference (var - base) over shared held-out cases, each case
    averaged over its reps; 95% CI half-width from the case-level differences."""

    def per(rows):
        d: dict[str, list[float]] = {}
        for r in rows:
            if r["prompt_id"] in ids and m in r["grade"]:
                d.setdefault(r["prompt_id"], []).append(r["grade"][m])
        return {k: sum(v) / len(v) for k, v in d.items()}

    b, v = per(base), per(var)
    diffs = [v[k] - b[k] for k in b if k in v]
    mean = st.mean(diffs)
    hw = 1.96 * st.stdev(diffs) / math.sqrt(len(diffs))
    return {
        "delta": round(mean, 4),
        "half_width": round(hw, 4),
        "lower": round(mean - hw, 4),
        "n_cases": len(diffs),
    }


def _variant(flow: str, variant: str, base: str = "baseline", paired_metrics=()) -> dict:
    state = json.loads((HC / flow / "_state.json").read_text())
    test = set(state["test_ids"])
    rows = _rows(flow, variant)
    cfg = json.loads((HC / flow / variant / "config.json").read_text())
    out = {
        "model": cfg["model"],
        "effort": cfg.get("effort"),
        "rows": len(rows),
        "reps": max(r["rep"] for r in rows) + 1,
        "cost_per_case_usd": round(
            sum(cost_usd(r["model"], r["usage"]) for r in rows) / len(rows), 4
        ),
        "output_tokens_per_case": round(st.mean(r["usage"].get("output_tokens", 0) for r in rows)),
        "rates": {
            m: _rate(rows, m)
            for m in (
                "equiv_expand_act",
                "equiv_distil_act",
                "self_consist_act",
                "decided",
                "agree_ref_act",
                "agree_gold_act",
                "trunc_detect",
                "equiv_served_act",
                "equiv_served_expand_act",
            )
            if _rate(rows, m) is not None
        },
    }
    test_rows = [r for r in rows if r["prompt_id"] in test]
    out["test_rates"] = {m: _rate(test_rows, m) for m in out["rates"]}
    if variant != base:
        out["paired_vs_baseline_test"] = {
            m: _paired(_rows(flow, base), rows, m, test) for m in paired_metrics
        }
    return out


def main() -> None:
    real = "decision-equivalence-real"
    summary = {
        "source": "benchmarks/model_migration_eval.py; rows in .claude/hillclimb/<flow>/<variant>/results.jsonl",
        "certifier_migration_real": {
            "cases": "100 τ-bench turns (benchmarks/model_migration_cases_real.json), 60 held-out",
            "variants": {
                v: _variant(
                    real,
                    v,
                    paired_metrics=("equiv_expand_act", "self_consist_act", "equiv_distil_act"),
                )
                for v in ("baseline", "v1", "v2", "v3", "v4")
            },
        },
        "coding_served": {
            "cases": "100 SWE-agent decision points (benchmarks/model_migration_cases_coding.json), 59 held-out",
            "baseline": _variant("compression-coding", "baseline"),
            "all_turns_savings": {
                k: v
                for k, v in json.loads(
                    (HC / "compression-coding/baseline/savings.json").read_text()
                ).items()
                if k != "by_kind"
            },
        },
    }
    v3 = summary["certifier_migration_real"]["variants"]["v3"]
    base = summary["certifier_migration_real"]["variants"]["baseline"]
    summary["certifier_migration_real"]["headline"] = {
        "recommended": "claude-sonnet-5-5 @ effort=low",
        "cost_reduction": round(1 - v3["cost_per_case_usd"] / base["cost_per_case_usd"], 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
