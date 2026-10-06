#!/usr/bin/env python3
"""Render docs/scoreboard.{json,html} from committed benchmark artifacts.

Rows are compressors, run by their real pinned packages through the existing harnesses
(benchmarks/swebench_outcome arms, benchmarks/cost_truth arms; ADR 0024). Every number
is recomputed here from a run's own files, with the harness's own analysis code
(swebench_outcome.report.analyse, cost_truth's analysis.json), so the page cannot say
anything its artifacts do not. An arm with no rows is shown as "pending run", never
estimated. Which runs appear is benchmarks/results/scoreboard-runs.json.

Usage: python3 scripts/build_scoreboard.py [--check]
Checked by tests/test_scoreboard.py (regenerate-and-compare).
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmarks.swebench_outcome.report import (  # noqa: E402
    analyse,
    cache_contamination,
    outcomes,
)

import site_nav  # noqa: E402

INDEX = ROOT / "benchmarks/results/scoreboard-runs.json"
OUT_JSON = ROOT / "docs/scoreboard.json"
OUT_HTML = ROOT / "docs/scoreboard.html"

LABELS = {
    "plain": "plain (no compression)",
    "control": "control (no compression)",
    "distil": "distil",
    "rtk": "RTK",
    "headroom": "Headroom",
    "selective": "Selective Context",
    "provider-cm": "Anthropic context editing",
}
#: Arms each harness can run (benchmarks/swebench_outcome/names.py, cost_truth/arms.py).
SWE_ARMS = ("plain", "distil", "rtk", "selective", "provider-cm", "headroom")
CT_ARMS = ("control", "distil", "rtk", "headroom")
#: The scoreboard's rows: every compressor, in both tables, so a gap is visible.
ROWS = ("plain", "distil", "rtk", "headroom", "selective", "provider-cm")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _cold(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cold-cache re-pricing for a run whose arms read each other's prompt cache: the index's
    `cold_repriced` names a committed results.json and the keys holding the per-arm totals and
    the paired per-task differences vs plain (benchmarks/why_rtk_wins.py's
    lite.cold_cache_repriced and lite.paired_vs_plain)."""
    src = entry.get("cold_repriced")
    if not src:
        return {}, {}
    doc: Any = json.loads((ROOT / src["file"]).read_text(encoding="utf-8"))

    def at(key: str) -> dict[str, Any]:
        node = doc
        for k in key.split("."):
            node = node[k]
        return node

    return at(src["totals"]), at(src["paired"])


def swebench(entry: dict[str, Any]) -> dict[str, Any]:
    d = ROOT / entry["dir"]
    results, grades = _jsonl(d / "results.jsonl"), _jsonl(d / "grades.jsonl")
    contaminated = cache_contamination(results)
    cold, paired = _cold(entry)
    confound: dict[str, Any] = {}
    a = analyse(results, grades)
    oc = outcomes(results, grades)
    graded = [x for x in a["arms"] if oc.get(x)]
    common = set.intersection(*(set(oc[x]) for x in graded)) if graded else set()
    first = results[0] if results else {}
    rows = []
    for arm in ROWS:
        if arm not in SWE_ARMS:
            rows.append({"arm": arm, "label": LABELS[arm], "status": "not in this harness"})
            continue
        s = a["arms"].get(arm)
        if s is None:
            rows.append({"arm": arm, "label": LABELS[arm], "status": "pending run"})
            continue
        mine = [r for r in results if r["arm"] == arm and r["instance_id"] in common]
        cost = sum(r.get("cost_usd", 0.0) for r in mine)
        c = a["comparisons"].get(arm)
        meta = s["meta"]
        pre, n_rows = contaminated.get(arm, (0, 0))
        # The cost columns are the arm's own cold-cache cost. A row that read another arm's
        # cache (report.precached) is not; such an arm shows a cold re-pricing when every one
        # of its tasks was re-priced, and otherwise no dollar figure at all.
        billed = round(cost, 4)
        billed_per = round(cost / s["resolved"], 4) if s["resolved"] else None
        if not pre:
            basis, cost_v, per_v = "cold", billed, billed_per
        elif arm in cold and cold[arm].get("tasks_repriced") == cold[arm]["tasks"] == len(mine):
            basis = "cold re-priced"
            cost_v = round(cold[arm]["usd"], 4)
            per_v = round(cold[arm]["usd"] / s["resolved"], 4) if s["resolved"] else None
            p = paired.get(arm, {})
            if p.get("cold_cache_pairs") == p.get("pairs") == len(mine):
                base = p["plain_mean_usd"]
                confound[arm] = {
                    "per_task_vs_plain_pct_cold": round(
                        p["cold_cache_mean_usd_diff"] / base * 100, 1
                    ),
                    "per_task_vs_plain_ci_pct_cold": [
                        round(x / base * 100, 1) for x in p["cold_cache_ci95"]
                    ],
                    "per_task_vs_plain_pct_billed": round(p["mean_usd_diff"] / base * 100, 1),
                }
        else:
            basis, cost_v, per_v = "confounded", None, None
        rows.append(
            {
                "arm": arm,
                "label": LABELS[arm],
                "status": "measured",
                "version": meta.get("version") or entry.get("versions", {}).get(arm),
                "library": meta.get("library"),
                "n": s["n"],
                "resolved": s["resolved"],
                "rate": round(s["rate"], 6),
                "rate_ci": [round(s["wilson"][0], 6), round(s["wilson"][1], 6)],
                "vs_plain_pts": None if c is None else round(c["diff"] * 100, 2),
                "vs_plain_ci_pts": None
                if c is None
                else [round(c["ci"][0] * 100, 2), round(c["ci"][1] * 100, 2)],
                "verdict": None if c is None else c["verdict"],
                "cost_usd": cost_v,
                "usd_per_solved": per_v,
                "cost_basis": basis,
                "precached_rows": [pre, n_rows],
                **(
                    {"cost_usd_billed": billed, "usd_per_solved_billed": billed_per}
                    if basis != "cold"
                    else {}
                ),
                "tokens_in": sum(
                    sum(
                        (r.get("usage") or {}).get(k, 0)
                        for k in ("input", "cache_write", "cache_read")
                    )
                    for r in mine
                ),
                "tokens_out": sum((r.get("usage") or {}).get("output", 0) for r in mine),
                "steps": sum(r.get("steps", 0) for r in mine),
            }
        )
    return {
        "id": entry["id"],
        "benchmark": entry.get("benchmark", "SWE-bench Lite") + " (official grader)",
        "source": entry["dir"],
        "date": entry["date"],
        "model": first.get("model"),
        "effort": first.get("effort"),
        "n_common": len(common),
        "note": entry.get("note"),
        "cost_note": entry.get("cost_note"),
        "cost_confound": confound,
        "rows": rows,
    }


def cost_truth(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Terminal-Bench (benchmarks/cost_truth). With no analysed run, every arm is pending."""
    res: dict[str, Any] = {}
    if entry is not None:
        res = json.loads((ROOT / entry["dir"] / "analysis.json").read_text(encoding="utf-8"))
    per = res.get("per_arm", {})
    rows = []
    for arm in ROWS:
        name = "control" if arm == "plain" else arm
        if name not in CT_ARMS:
            rows.append({"arm": arm, "label": LABELS[arm], "status": "not in this harness"})
        elif name not in per:
            rows.append({"arm": arm, "label": LABELS[name], "status": "pending run"})
        else:
            p = per[name]
            cmp_ = res.get("comparisons", {}).get(name)
            vs: dict[str, Any] = {}
            if cmp_ is not None:
                lo, hi = cmp_["success_diff_ci"]
                vs = {
                    "vs_plain_pts": round(cmp_["success_diff"] * 100, 1),
                    "vs_plain_ci_pts": [round(lo * 100, 1), round(hi * 100, 1)],
                    # analysis.json's own word ("not shown" for a pilot), never a winner
                    "verdict": f"pilot, {cmp_['verdict']}",
                }
            rows.append(
                {
                    "arm": arm,
                    "label": LABELS[name],
                    "status": "measured",
                    "version": (entry or {}).get("versions", {}).get(name),
                    "n": p["attempts"],
                    "resolved": p["solved"],
                    "rate": round(p["success_rate"], 6),
                    "cost_usd": round(p["cost_usd"], 4),
                    "usd_per_solved": round(p["usd_per_solved"], 4) if p["solved"] else None,
                    **vs,
                }
            )
    return {
        "id": (entry or {}).get("id", "terminal-bench"),
        "benchmark": "Terminal-Bench 2.1"
        + (" pilot" if entry else "")
        + " (cost_truth, neutral meter)",
        "source": (entry or {}).get("dir"),
        "date": (entry or {}).get("date"),
        "model": (entry or {}).get("model"),
        "effort": None,
        "n_common": res.get("tasks"),
        "note": (entry or {}).get("note")
        or "Not run yet: the pilot's canary preflights are committed, no analysed attempts.",
        "rows": rows,
    }


def build() -> dict[str, Any]:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    sections = [swebench(e) for e in index["runs"] if e["kind"] == "swebench_outcome"]
    ct = [e for e in index["runs"] if e["kind"] == "cost_truth"]
    sections += [cost_truth(e) for e in ct] or [cost_truth(None)]
    return {
        "_note": "Generated by scripts/build_scoreboard.py from committed artifacts; do not edit.",
        "sections": sections,
    }


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _row_html(r: dict[str, Any], sid: str = "") -> str:
    name = html.escape(r["label"])
    if r["status"] != "measured":
        return f'<tr><td>{name}</td><td colspan="7"><i>{r["status"]}</i></td></tr>'
    ci = r.get("rate_ci")
    rate = f"{r['resolved']}/{r['n']} · {_pct(r['rate'])}"
    if ci:
        rate += f" ({_pct(ci[0])}–{_pct(ci[1])})"
    vs = "baseline"
    if r.get("vs_plain_pts") is not None:
        lo, hi = r["vs_plain_ci_pts"]
        vs = f"{r['vs_plain_pts']:+.1f} [{lo:+.1f}, {hi:+.1f}] · {r['verdict']}"
    basis = r.get("cost_basis")
    mark = "†" if basis == "cold re-priced" else ""
    if basis == "confounded":
        per = total = f'<i>confounded — <a href="#{sid}-cost-confound">see note</a></i>'
    else:
        per = "—" if r.get("usd_per_solved") is None else f"${r['usd_per_solved']:.4f}{mark}"
        total = f"${r['cost_usd']:.2f}{mark}"
    toks = f"{r['tokens_in']:,} / {r['tokens_out']:,}" if r.get("tokens_in") is not None else "—"
    ver = html.escape(" ".join(x for x in (r.get("library"), r.get("version")) if x) or "—")
    return (
        f"<tr><td>{name}</td><td>{per}</td><td>{rate}</td><td>{vs}</td>"
        f"<td>{total}</td><td>{toks}</td><td>{r.get('steps', '—')}</td>"
        f"<td>{ver}</td></tr>"
    )


def _section_html(s: dict[str, Any]) -> str:
    cfg = " · ".join(
        x
        for x in (
            s["model"] and f"<code>{html.escape(s['model'])}</code>",
            s["effort"] and f"effort {html.escape(s['effort'])}",
            s["n_common"] is not None and f"n = {s['n_common']} tasks graded in every arm",
            s["date"] and f"run {s['date']}",
        )
        if x
    )
    src = (
        f' Source: <a href="https://github.com/dshakes/distil/tree/main/{s["source"]}">'
        f"<code>{s['source']}</code></a>."
        if s["source"]
        else ""
    )
    note = f" {html.escape(s['note'][0].upper() + s['note'][1:])}." if s.get("note") else ""
    measured = [r for r in s["rows"] if r["status"] == "measured"]
    confound = any(r.get("cost_basis", "cold") != "cold" for r in measured)
    per_head = "$ per solved task (cold cache)" if confound else "$ per solved task"
    foot, sid = "", s["id"]
    if confound:
        billed = "; ".join(
            f"{html.escape(r['label'])} ${r['usd_per_solved_billed']:.4f} per solved, "
            f"${r['cost_usd_billed']:.2f} total ({r['precached_rows'][0]}/"
            f"{r['precached_rows'][1]} rows read another arm's cache)"
            for r in measured
            if r.get("cost_basis", "cold") != "cold"
        )
        paired = "".join(
            f" Cold re-priced, {html.escape(LABELS[a])} vs plain per task: "
            f"{c['per_task_vs_plain_pct_cold']:+.1f}% "
            f"[{c['per_task_vs_plain_ci_pct_cold'][0]:+.1f}%, "
            f"{c['per_task_vs_plain_ci_pct_cold'][1]:+.1f}%] (as billed "
            f"{c['per_task_vs_plain_pct_billed']:+.1f}%)."
            for a, c in (s.get("cost_confound") or {}).items()
        )
        foot = (
            f'\n    <p id="{sid}-cost-confound" class="note"><b>Cost confound.</b> '
            f"{html.escape(s.get('cost_note') or '')}{paired} † re-priced as a cold run. "
            f"As billed, not comparable: {billed}.</p>"
        )
    return f"""
    <h2 id="{s["id"]}">{html.escape(s["benchmark"])}</h2>
    <p>{cfg or "Pending"}.{src}{note}</p>
    <div class="table-scroll" tabindex="0" role="region" aria-label="{html.escape(s["benchmark"])} scoreboard">
    <table>
      <thead><tr><th>Compressor</th><th>{per_head}</th><th>Solved (95% CI)</th><th>vs plain, pts [95% CI]</th><th>Total $</th><th>Tokens in / out</th><th>Steps</th><th>Pinned version</th></tr></thead>
      <tbody>
        {chr(10).join("        " + _row_html(r, s["id"]) for r in s["rows"]).lstrip()}
      </tbody>
    </table>
    </div>{foot}"""


_DESC = (
    "Compressors scored on solved tasks and dollars, failed attempts included: distil, RTK, "
    "Headroom, Selective Context and Anthropic context editing, each run by its real pinned "
    "package. Every number is recomputed from a committed artifact; arms not yet run say so."
)


def render_html(data: dict[str, Any]) -> str:
    sections = "\n".join(_section_html(s) for s in data["sections"])
    title = "Scoreboard — Distil"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<link rel="stylesheet" href="site.css"/>
<meta property="og:type" content="website"/>
<meta property="og:site_name" content="Distil"/>
<meta property="og:title" content="{title}"/>
<meta property="og:description" content="{_DESC}"/>
<meta property="og:url" content="https://dshakes.github.io/distil/scoreboard.html"/>
<meta property="og:image" content="https://dshakes.github.io/distil/og.png"/>
<meta property="og:image:width" content="1200"/>
<meta property="og:image:height" content="630"/>
<meta name="twitter:card" content="summary_large_image"/>
<meta name="twitter:title" content="{title}"/>
<meta name="twitter:description" content="{_DESC}"/>
<meta name="twitter:image" content="https://dshakes.github.io/distil/og.png"/>
  <link rel="icon" type="image/svg+xml" href="assets/logo.svg">
<script>(function(){{try{{var t=localStorage.getItem("distil-theme");if(t!=="light"&&t!=="dark")t=window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";document.documentElement.setAttribute("data-theme",t);}}catch(e){{}}}})();</script>
</head>
<body>
<a class="skip-link" href="#content">Skip to main content</a>

<header class="topbar">
  <button class="sidebar-toggle" onclick="toggleSidebar()" aria-label="Toggle navigation" aria-expanded="false" aria-controls="sidebar">☰</button>
  <a href="index.html" class="topbar-logo"><img src="assets/logo.svg" alt="" width="22" height="22" style="border-radius:6px;vertical-align:-6px;margin-right:8px"/>Dist<b>il</b></a>
  <span class="topbar-pill">compression with a quality contract</span>
{site_nav.render_topbar_links("scoreboard.html")}
</header>

<div class="shell">
{site_nav.render_sidebar("scoreboard.html")}

  <main class="content" id="content" tabindex="-1">

    <h1>The <span class="g">scoreboard</span></h1>
    <p class="lead">Every context compressor claims a saving. This page scores them on what a user pays for: dollars per <b>solved</b> task, with every failed attempt's spend counted, and whether tasks still get solved. Each compressor is run by its own real, pinned package through the same harness and the same official grader. distil is one of the rows, not the referee's favourite: the same rules apply to it.</p>
    <p>How it is measured, and why these estimators, is <a href="https://github.com/dshakes/distil/blob/main/docs/adr/0024-distil-as-referee.md">ADR 0024</a>. Paired against plain on the same tasks; the interval on the difference is the harness's Wald interval with an exact McNemar test, at a pre-registered 5-point non-inferiority margin. A row that says <i>pending run</i> has no committed data, and nothing here estimates it. The runs that will fill it, with their costs, are in <a href="https://github.com/dshakes/distil/blob/main/docs/research/referee-runs.md">the run plan</a>. To referee a compressor on your <em>own</em> traffic instead, see <code>distil audit</code> in the <a href="cli.html">CLI reference</a>.</p>
{sections}

    <h2 id="read">How to read it</h2>
    <ul>
      <li><b>$ per solved task</b> is the arm's total spend on the tasks graded in every arm, divided by the tasks it solved. A cheaper arm that solves fewer tasks can cost more per solved task.</li>
      <li><b>Cold cache.</b> Each arm's dollars must be its own. The builder checks every row: a run that started cold writes at least its largest prompt, so <code>cache_write + input &lt; cache_read / (steps &minus; 1)</code> means it read a prompt cache another arm wrote. Such an arm shows a cold re-pricing, or no dollar figure, never the billed one.</li>
      <li><b>vs plain</b> is the paired difference in tasks solved, in points, with its verdict at the 5-point margin. <i>PILOT</i>, <i>INCONCLUSIVE</i> and <i>INFERIOR</i> mean what they say; none of them is a pass.</li>
      <li>Short tasks. On these runs the median task took a handful of steps, so contexts stayed short. A compressor built for long sessions is not exercised much here; the Terminal-Bench table runs longer agent sessions.</li>
      <li>Every figure is regenerated by <code>scripts/build_scoreboard.py</code> from the run directories, and the machine-readable copy is <a href="scoreboard.json">scoreboard.json</a>.</li>
    </ul>

{site_nav.FOOTER}
  </main>
</div>

<script src="site.js" defer></script>
</body>
</html>
"""


def main(argv: list[str]) -> int:
    data = build()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    page = render_html(data)
    if "--check" in argv:
        stale = [
            p.name
            for p, t in ((OUT_JSON, js), (OUT_HTML, page))
            if not p.exists() or p.read_text(encoding="utf-8") != t
        ]
        if stale:
            print(f"stale: {', '.join(stale)} — run python3 scripts/build_scoreboard.py")
            return 1
        return 0
    OUT_JSON.write_text(js, encoding="utf-8")
    OUT_HTML.write_text(page, encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_HTML.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
