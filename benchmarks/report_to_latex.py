#!/usr/bin/env python3
"""report_to_latex.py — turn a `prove.py --report` JSON into paper-ready LaTeX.

Runs after the headline experiment and writes LaTeX fragments into
``docs/paper/generated/``. The paper (`docs/paper/main.tex`) ``\\input``s them via
``\\IfFileExists`` — so once you run this, the figures, tables, and headline macros
in the PDF reflect your real numbers with **zero hand-copying**. Before you run it,
the paper falls back to clearly-labeled placeholders.

Usage:
  python benchmarks/prove.py ... --report results.json
  python benchmarks/report_to_latex.py results.json
  # then recompile docs/paper/main.tex (Overleaf / latexmk)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.swe_bench_e2e.stats import wilson_ci  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "paper" / "generated"


def _tex(s: str) -> str:
    """Escape LaTeX specials in free text (method names etc.)."""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("@", r"@")
    )


def _pct(x: float) -> str:
    return f"{x * 100:.1f}\\%"


CHECK = r"\checkmark"
CROSS = r"$\times$"
EOL = r" \\"  # LaTeX row terminator (kept out of f-strings: no backslashes allowed there)


def macros(rep: dict) -> str:
    cov = rep.get("coverage") or {}
    sav = cov.get("mean_test_savings")
    c = cov.get("empirical_coverage")
    r = cov.get("mean_realized_risk")
    out = ["% auto-generated headline macros — do not edit"]
    out.append(f"\\renewcommand{{\\HLsavings}}{{{_pct(sav)}}}" if sav is not None else "")
    out.append(f"\\renewcommand{{\\HLcoverage}}{{{_pct(c)}}}" if c is not None else "")
    out.append(f"\\renewcommand{{\\HLrisk}}{{{_pct(r)}}}" if r is not None else "")
    return "\n".join(x for x in out if x) + "\n"


# Methods cited by name in the paper's E5 prose -> a LaTeX-safe macro suffix. Keeping
# this explicit (rather than auto-deriving from method names) means the paper only ever
# materializes macros it actually uses, and the names stay readable.
_E5_CITED = {
    "recency-window@500": "Recency",
    "llmlingua-2": "LLMtwo",
    "longllmlingua": "LongLL",
    "recomp-extractive": "Recomp",
    "lossless": "Lossless",
    "truncate@120": "TruncShort",
    "byte-exact": "ByteExact",
}


def e5_macros(rep: dict, prefix: str = "Shuf") -> str:
    """Emit ``\\renewcommand`` macros for the head-to-head numbers the E5 prose cites,
    namespaced by ``prefix`` (e.g. ``Shuf`` -> ``\\ShufRecencyDC``/``\\ShufRecencySav``).
    Lets the paper compare the original and shuffled-position runs with zero hardcoded
    numbers in the prose. Run once per report with a distinct prefix."""
    rows = {r["method"]: r for r in rep.get("head_to_head") or []}
    cov = rep.get("coverage") or {}
    out = [f"% auto-generated E5 macros (prefix={prefix}) — do not edit"]
    for method, suffix in _E5_CITED.items():
        r = rows.get(method)
        if not r:
            continue
        out.append(f"\\renewcommand{{\\{prefix}{suffix}DC}}{{{_pct(r['decision_change'])}}}")
        out.append(f"\\renewcommand{{\\{prefix}{suffix}Sav}}{{{_pct(r['savings'])}}}")
    if cov.get("empirical_coverage") is not None:
        out.append(f"\\renewcommand{{\\{prefix}Coverage}}{{{_pct(cov['empirical_coverage'])}}}")
    return "\n".join(out) + "\n"


def frontier(rep: dict) -> str:
    rows = rep.get("frontier") or []
    alpha = (rep.get("coverage") or {}).get("alpha", 0.05)
    cert, lossy = [], []
    for r in rows:
        pt = f"({r['savings'] * 100:.2f},{r['decision_change'] * 100:.2f})"
        (cert if r["decision_change"] <= alpha else lossy).append(pt)
    ymax = max([r["decision_change"] * 100 for r in rows] + [alpha * 100 + 5]) + 5
    xmax = max([r["savings"] * 100 for r in rows] + [10]) + 5
    return (
        "% auto-generated frontier (E1)\n"
        "\\begin{tikzpicture}\n\\begin{axis}[width=0.7\\textwidth,height=6cm,"
        "xlabel={token savings (\\%)},ylabel={decision-change (\\%)},"
        f"xmin=0,xmax={xmax:.0f},ymin=-3,ymax={ymax:.0f},"
        "legend style={font=\\scriptsize,at={(0.02,0.98)},anchor=north west}]\n"
        f"\\addplot[only marks,mark=*,distilgreen] coordinates {{{' '.join(cert) or '(0,0)'}}};\n"
        "\\addlegendentry{certified ($\\le\\alpha$)}\n"
        f"\\addplot[only marks,mark=triangle*,distilred] coordinates {{{' '.join(lossy) or '(0,0)'}}};\n"
        "\\addlegendentry{lossy (flips)}\n"
        f"\\addplot[dashed,distilgray] coordinates {{(0,{alpha * 100:.1f}) ({xmax:.0f},{alpha * 100:.1f})}};\n"
        f"\\addlegendentry{{$\\alpha={alpha * 100:.0f}\\%$}}\n"
        "\\end{axis}\n\\end{tikzpicture}\n"
    )


def frontier_ci(rep: dict) -> str:
    """E1 as a table with Wilson 95\\% CIs — the figure plots point estimates only.

    Decision-change is a binomial proportion over the graded turns, so a bare point
    estimate hides how much of the gap between two ladder levels is sampling noise.
    We report the interval over all turns and over the *effective* (non-trivial) turns,
    since a level that leaves most turns byte-identical is scored on far fewer.
    """
    rows = rep.get("frontier") or []
    if not rows:
        return "% no frontier in report\n"
    body = []
    for r in rows:
        n, eff = int(r["n"]), int(r.get("effective_n") or 0)
        lo, hi = wilson_ci(round(r["decision_change"] * n), n)
        elo, ehi = wilson_ci(round(r.get("decision_change_effective", 0) * eff), eff)
        eff_cell = (
            f"{_pct(r.get('decision_change_effective', 0))} [{elo * 100:.1f}--{ehi * 100:.1f}]"
            if eff
            else "---"
        )
        body.append(
            _tex(r["level"])
            + " & "
            + _pct(r["savings"])
            + f" & {n} & "
            + _pct(r["decision_change"])
            + f" [{lo * 100:.1f}--{hi * 100:.1f}]"
            + f" & {eff} & "
            + eff_cell
            + EOL
            + "\n"
        )
    header = (
        "level & savings & $n$ & decision-change (95\\% CI) & $n_{\\text{eff}}$ & "
        "effective (95\\% CI)" + EOL
    )
    return (
        "% auto-generated E1 frontier with Wilson CIs — do not edit\n"
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        f"{header}\n\\midrule\n{''.join(body).rstrip()}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def head_to_head(rep: dict) -> str:
    rows = rep.get("head_to_head") or []
    if not rows:
        return "% no head-to-head in report (run with --baselines)\n"
    body = "\n".join(
        _tex(r["method"])
        + " & "
        + r["kind"]
        + " & "
        + _pct(r["savings"])
        + " & "
        + _pct(r["decision_change"])
        + " & "
        + (CHECK if r["certifies"] else CROSS)
        + EOL
        for r in rows
    )
    header = "method & kind & savings & dec-change & certifies?" + EOL
    return (
        "% auto-generated head-to-head (E5)\n"
        "\\begin{tabular}{@{}llrrc@{}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def coverage(rep: dict) -> str:
    c = rep.get("coverage") or {}
    if not c:
        return "% no coverage in report\n"
    tgt = c.get("target_coverage")
    tgt_s = f"{tgt * 100:.0f}\\%" if tgt else "expected-risk (CRC)"
    return (
        "% auto-generated coverage (E2)\n"
        "\\begin{tabular}{@{}lr@{}}\n\\toprule\n"
        f"method & {c.get('method', 'ltt').upper()} \\\\\n"
        f"$\\alpha$ / $\\delta$ & {c.get('alpha')} / {c.get('delta')} \\\\\n"
        f"splits & {c.get('reps')} \\\\\n"
        f"certified in & {_pct(c.get('certified_frac', 0))} of splits \\\\\n"
        f"empirical coverage $\\Pr(\\text{{realized}}\\le\\alpha)$ & {_pct(c.get('empirical_coverage', 0))} \\\\\n"
        f"target ($1-\\delta$) & {tgt_s} \\\\\n"
        f"mean realized held-out risk & {_pct(c.get('mean_realized_risk', 0))} \\\\\n"
        f"mean certified savings & {_pct(c.get('mean_test_savings', 0))} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )


def task_success(rep: dict) -> str:
    t = rep.get("task_success") or {}
    if not t:
        return "% no task-success in report (need outcome-labeled trajectories)\n"
    note = ""
    if t.get("outcome_evidential") is False:
        note = (
            "% NOTE: outcome is non-evidential (all trajectories share one label, e.g.\n"
            "% swe-hf resolved=True by construction) — read 'retained' as retained\n"
            "% decision-equivalence, NOT a measured task-success rate.\n"
        )
    body = "\n".join(
        _tex(r["level"])
        + " & "
        + _pct(r["savings"])
        + " & "
        + _pct(r["retained_success"])
        + f" [{r['ci_low'] * 100:.0f}--{r['ci_high'] * 100:.0f}]"
        + EOL
        for r in t["levels"]
    )
    header = "level & savings & retained success (95\\% CI)" + EOL
    return (
        f"% auto-generated task-success (E4); baseline={_pct(t.get('baseline_success', 0))}, n={t.get('n')}\n"
        f"{note}"
        "\\begin{tabular}{@{}lrr@{}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def shift(rep: dict) -> str:
    rows = rep.get("shift") or []
    if not rows:
        return "% no distribution-shift in report (need >=2 domains)\n"
    body = "\n".join(
        _tex(r["held_out_domain"])
        + " & "
        + _tex(r.get("certified") or "none")
        + " & "
        + _pct(r.get("realized_risk", 0))
        + " & "
        + _pct(r.get("savings", 0))
        + " & "
        + (CHECK if r.get("held_within_alpha") else CROSS)
        + EOL
        for r in rows
    )
    header = "held-out domain & certified & realized & savings & ok?" + EOL
    return (
        "% auto-generated distribution-shift (E3)\n"
        "\\begin{tabular}{@{}llrrc@{}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def loo(rep: dict) -> str:
    """E3 leave-one-domain-out (`benchmarks/leave_one_domain_out.py`).

    Note this fragment reads a *different* report shape from the rest of this module:
    :func:`shift` renders the per-turn shift rows a ``prove.py`` run would carry in
    ``rep["shift"]`` (empty on every committed run — the τ-bench and SWE corpora were
    graded by different models, so a cross-domain comparison there is confounded),
    while E3 as actually run is the trajectory-level leave-one-repository-out on the E8
    outcomes, where the grader is the deterministic SWE-bench harness. Same experiment
    slot, honest unit.
    """
    rows = rep.get("domains") or []
    if not rows:
        return "% no leave-one-domain-out rows in report\n"
    body = "\n".join(
        _tex(r["domain"])
        + f" & {r['n_calib']} & {r['n_test']} & "
        + _pct(r["divergence"]["certified_bound"])
        + " & "
        + _pct(r["divergence"]["realized_risk"])
        + " & "
        + (CHECK if r["divergence"]["held"] else CROSS)
        + " & "
        + _pct(r["divergence"]["exchangeable_coverage"])
        + f" & {r['divergence']['shift_pvalue']:.2f}"
        + EOL
        for r in rows
    )
    header = (
        r"held-out domain & $n_{\text{cal}}$ & $n_{\text{test}}$ & certified $\beta$ "
        r"& realized & held? & exch.\ cov. & $p$" + EOL
    )
    return (
        "% auto-generated E3 leave-one-domain-out — do not edit\n"
        "\\begin{tabular}{@{}lrrrrcrr@{}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def loo_macros(rep: dict) -> str:
    """Headline macros for the E3 prose, so no number is hand-typed into the paper."""
    s = rep.get("summary") or {}
    if not s:
        return "% no leave-one-domain-out summary in report\n"
    d, h = s["divergence"], s["harm"]
    rows = rep.get("domains") or []
    min_p = min((r["divergence"]["shift_pvalue"] for r in rows), default=1.0)
    out = [
        "% auto-generated E3 macros — do not edit",
        f"\\renewcommand{{\\LooN}}{{{s['n']}}}",
        f"\\renewcommand{{\\LooDomains}}{{{d['domains']}}}",
        f"\\renewcommand{{\\LooCandidate}}{{\\texttt{{{_tex(s['candidate'])}}}}}",
        f"\\renewcommand{{\\LooDelta}}{{{s['delta']}}}",
        f"\\renewcommand{{\\LooPooled}}{{{_pct(d['pooled_risk'])}}}",
        f"\\renewcommand{{\\LooHeld}}{{{d['domains_held']}}}",
        f"\\renewcommand{{\\LooCoverage}}{{{_pct(d['coverage'])}}}",
        f"\\renewcommand{{\\LooExchCoverage}}{{{_pct(d['mean_exchangeable_coverage'])}}}",
        f"\\renewcommand{{\\LooWorstDomain}}{{\\texttt{{{_tex(d['worst_domain'])}}}}}",
        f"\\renewcommand{{\\LooWorstRisk}}{{{_pct(d['worst_realized'])}}}",
        f"\\renewcommand{{\\LooDispP}}{{{d['dispersion_pvalue']:.2f}}}",
        f"\\renewcommand{{\\LooMinP}}{{{min_p:.2f}}}",
        f"\\renewcommand{{\\LooReps}}{{{d['permutation_reps']}}}",
        f"\\renewcommand{{\\LooHarmPooled}}{{{_pct(h['pooled_risk'])}}}",
        f"\\renewcommand{{\\LooHarmCoverage}}{{{_pct(h['coverage'])}}}",
        f"\\renewcommand{{\\LooHarmExchCoverage}}{{{_pct(h['mean_exchangeable_coverage'])}}}",
        f"\\renewcommand{{\\LooHarmDispP}}{{{h['dispersion_pvalue']:.2f}}}",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="results.json from `prove.py --report`")
    ap.add_argument("--out", default=str(OUT), help="output dir for LaTeX fragments")
    ap.add_argument(
        "--suffix",
        default="",
        help="append to each fragment filename (e.g. '_shuffled' → headtohead_shuffled.tex) "
        "so an additive variant run doesn't overwrite the headline fragments",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        choices=[
            "macros",
            "frontier",
            "frontierci",
            "headtohead",
            "coverage",
            "tasksuccess",
            "shift",
            "e5macros",
            "loo",
            "loomacros",
        ],
        help="emit only these fragments (default: all prove.py fragments; e5macros, "
        "loo and loomacros are opt-in because they need a prefix or a different report)",
    )
    ap.add_argument(
        "--macro-prefix",
        default="Shuf",
        help="namespace prefix for e5macros (e.g. 'Shuf' -> \\ShufRecencyDC, 'Orig' -> \\OrigRecencyDC)",
    )
    args = ap.parse_args()

    rep = json.loads(Path(args.report).read_text())
    runner = (rep.get("args") or {}).get("runner")
    if runner == "smoke":
        raise SystemExit(
            "ERROR: this report came from the NON-EVIDENTIAL smoke runner — refusing to\n"
            "       emit paper LaTeX from it. Re-run prove.py with --runner "
            "anthropic/openai/claude-cli on real traces."
        )
    if rep.get("args") and rep["args"].get("samples", 1) < 3:
        print(
            "WARNING: report used --samples < 3; decision-change rate conflates true loss "
            "with grader variance. Use majority-of-3+ for a publishable number.",
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    builders = {
        "macros": macros,
        "frontier": frontier,
        "frontierci": frontier_ci,
        "headtohead": head_to_head,
        "coverage": coverage,
        "tasksuccess": task_success,
        "shift": shift,
        "e5macros": lambda r: e5_macros(r, prefix=args.macro_prefix),
        "loo": loo,
        "loomacros": loo_macros,
    }
    # e5macros needs a prefix and loo/loomacros read a leave_one_domain_out.py report
    # rather than a prove.py one, so the default "all" set excludes all three.
    opt_in = {"e5macros", "loo", "loomacros"}
    selected = args.only or [k for k in builders if k not in opt_in]
    for name in selected:
        (out / f"{name}{args.suffix}.tex").write_text(builders[name](rep))
    print(f"wrote {len(selected)} LaTeX fragment(s) → {out} (suffix={args.suffix!r})")
    print("the paper picks them up automatically (\\IfFileExists). Recompile docs/paper/main.tex.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
