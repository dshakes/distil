"""Savings ledger — local-first, privacy-preserving community savings tracking.

Every certified run can append an aggregate record (ids + numbers only, never
context content) to a local JSONL. `summary()` rolls it up so you can see
cumulative tokens and dollars saved across an agent fleet over time.

Community aggregation (a shared leaderboard) is a deliberate OPT-IN: it would
mean network egress of your run metadata, so this module never sends anything.
"""

from __future__ import annotations

import html as _html
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path


def default_path() -> Path:
    """The ledger path, honoring ``DISTIL_HOME`` at call time (configurable
    deployments; isolated tests) — same contract as shadow/learn state."""
    import os

    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "savings.jsonl"


# Back-compat eager constant; runtime paths resolve via default_path().
DEFAULT_PATH = default_path()


@dataclass
class SavingsRecord:
    trajectory_id: str
    model: str
    turns: int
    baseline_dollars: float
    distil_dollars: float
    baseline_input_tokens: int
    distil_input_tokens: int
    tokenizer: str
    ts: float
    session: str = ""  # proxy-process session id — lets the statusline show THIS session

    @property
    def dollars_saved(self) -> float:
        return self.baseline_dollars - self.distil_dollars

    @property
    def tokens_saved(self) -> int:
        return self.baseline_input_tokens - self.distil_input_tokens


def record(
    *,
    trajectory_id: str,
    model: str,
    turns: int,
    baseline_dollars: float,
    distil_dollars: float,
    baseline_input_tokens: int,
    distil_input_tokens: int,
    tokenizer: str = "heuristic",
    session: str = "",
    path: Path | None = None,
) -> SavingsRecord:
    rec = SavingsRecord(
        trajectory_id,
        model,
        turns,
        baseline_dollars,
        distil_dollars,
        baseline_input_tokens,
        distil_input_tokens,
        tokenizer,
        time.time(),
        session,
    )
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
    return rec


@dataclass
class LedgerSummary:
    runs: int
    total_dollars_saved: float
    total_tokens_saved: int
    by_trajectory: dict[str, float]  # id -> dollars saved
    # Absolute totals (not just the delta), so callers can show orig -> compressed.
    total_baseline_tokens: int = 0
    total_distil_tokens: int = 0
    total_baseline_dollars: float = 0.0
    total_distil_dollars: float = 0.0
    # Which tokenizers produced the counts — so callers can caveat heuristic
    # (non-billing-grade) numbers instead of presenting them as exact.
    tokenizers: frozenset[str] = frozenset()


def latest_session(path: Path | None = None) -> tuple[str, float]:
    """(session id, last-activity ts) of the most recently ACTIVE session in the
    ledger, or ("", 0.0). One pass, cheap enough for a status line."""
    path = path or default_path()
    best_id, best_ts = "", 0.0
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            sid = d.get("session") or ""
            ts = d.get("ts", 0.0)
            if sid and ts >= best_ts:
                best_id, best_ts = sid, ts
    except OSError:
        pass
    return best_id, best_ts


def summary(
    path: Path | None = None, *, since: float | None = None, session: str | None = None
) -> LedgerSummary:
    """Roll up the ledger; ``since`` (unix ts) and/or ``session`` restrict the
    window so callers can show a fresh slice (this session) next to lifetime."""
    path = path or default_path()
    if not path.exists():
        return LedgerSummary(0, 0.0, 0, {})
    runs = 0
    dollars = 0.0
    tokens = 0
    base_tok = 0
    dist_tok = 0
    base_usd = 0.0
    dist_usd = 0.0
    by_traj: dict[str, float] = {}
    toks: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if since is not None and d.get("ts", 0.0) < since:
            continue
        if session is not None and (d.get("session") or "") != session:
            continue
        runs += 1
        toks.add(d.get("tokenizer", "heuristic"))
        saved = d["baseline_dollars"] - d["distil_dollars"]
        dollars += saved
        tokens += d["baseline_input_tokens"] - d["distil_input_tokens"]
        base_tok += d["baseline_input_tokens"]
        dist_tok += d["distil_input_tokens"]
        base_usd += d["baseline_dollars"]
        dist_usd += d["distil_dollars"]
        by_traj[d["trajectory_id"]] = by_traj.get(d["trajectory_id"], 0.0) + saved
    return LedgerSummary(
        runs, dollars, tokens, by_traj, base_tok, dist_tok, base_usd, dist_usd, frozenset(toks)
    )


def render_html(
    s: LedgerSummary,
    *,
    change_rate: float | None = None,
    samples: int = 0,
    session: "LedgerSummary | None" = None,
) -> str:
    """Render the ledger as a self-contained dark HTML page — GENUINE savings
    from your own usage (the `live-proxy` source is real proxy traffic).
    ``change_rate``/``samples`` add the decision-equivalence card (shown only
    at >=25 samples — a rate over a handful of samples is noise); ``session``
    adds a this-session card when a live session exists."""
    rows = (
        "".join(
            f'<tr><td>{_html.escape(str(tid))}</td><td class="r">${saved:,.4f}</td></tr>'
            for tid, saved in sorted(s.by_trajectory.items(), key=lambda kv: -kv[1])
        )
        or '<tr><td colspan="2" class="muted">no runs recorded yet</td></tr>'
    )
    live = "live-proxy" in s.by_trajectory
    note = (
        "Includes <b>live-proxy</b> — genuine savings measured on your real traffic."
        if live
        else "Run <code>distil proxy</code> to record genuine savings from real traffic."
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Distil — your savings</title><style>
body{{margin:0;background:#06070b;color:#f2f3f7;font:15px/1.6 Inter,ui-sans-serif,sans-serif;
 -webkit-font-smoothing:antialiased}}
.wrap{{max-width:760px;margin:0 auto;padding:48px 24px}}
h1{{font-size:30px;font-weight:800;letter-spacing:-.02em;margin:0 0 6px}}
.sub{{color:#9aa1b3;margin:0 0 28px}}
.tot{{display:flex;gap:14px;margin:0 0 28px;flex-wrap:wrap}}
.card{{flex:1;min-width:150px;background:linear-gradient(180deg,#12151f,#0b0d15);
 border:1px solid #252c3e;border-radius:14px;padding:20px}}
.card .l{{color:#9aa1b3;font-size:12px}} .card .v{{font-size:30px;font-weight:800;margin-top:4px}}
.g{{background:linear-gradient(135deg,#8b7bff,#5ad1c9);-webkit-background-clip:text;background-clip:text;color:transparent}}
table{{width:100%;border-collapse:collapse;border:1px solid #1b2030;border-radius:12px;overflow:hidden}}
th,td{{padding:11px 14px;border-bottom:1px solid #1b2030;text-align:left}}
th{{color:#5b6177;font-size:11px;text-transform:uppercase;letter-spacing:.07em}}
td.r{{text-align:right;color:#5ad1c9;font-variant-numeric:tabular-nums}} .muted{{color:#5b6177}}
.foot{{color:#5b6177;font-size:12.5px;margin-top:22px}}
</style></head><body><div class="wrap">
<h1>Your <span class="g">savings</span></h1>
<p class="sub">Genuine, local-first — measured from your own runs and proxy traffic. No content leaves your machine.</p>
<div class="tot">
 <div class="card"><div class="l">Tokens saved</div><div class="v">{s.total_tokens_saved:,}</div></div>
 <div class="card"><div class="l">Dollars saved</div><div class="v g">${s.total_dollars_saved:,.4f}</div></div>
 <div class="card"><div class="l">Runs</div><div class="v">{s.runs:,}</div></div>
 {_eq_card(change_rate, samples)}{_session_card(session)}
</div>
<table><thead><tr><th>source</th><th style="text-align:right">$ saved</th></tr></thead><tbody>{rows}</tbody></table>
<p class="foot">{note} Share verifiably across instances with <code>distil federated-leaderboard</code>.</p>
</div></body></html>"""


def _eq_card(change_rate: float | None, samples: int) -> str:
    """Decision-equivalence card — only once there is evidence behind the rate."""
    if change_rate is None or samples < 25:
        return (
            '<div class="card"><div class="l">Decision-equivalence</div>'
            '<div class="v muted" style="font-size:16px">needs 25+ shadow samples<br/>'
            "<code>distil wrap --shadow 0.1 -- &lt;agent&gt;</code></div></div>"
        )
    eq = (1 - change_rate) * 100
    return (
        f'<div class="card"><div class="l">Decision-equivalence</div>'
        f'<div class="v g">{eq:.1f}%</div>'
        f'<div class="l">{samples:,} shadowed requests</div></div>'
    )


def _session_card(session: "LedgerSummary | None") -> str:
    if session is None or not session.runs or not session.total_baseline_tokens:
        return ""
    if session.total_tokens_saved > 0:
        trimmed = (1 - session.total_distil_tokens / session.total_baseline_tokens) * 100
        value, label = f"▼{_human(session.total_tokens_saved)}", f"{trimmed:.0f}% smaller"
    else:
        value, label = "✓ on", "waiting for a large read"
    return (
        f'<div class="card"><div class="l">This session</div>'
        f'<div class="v">{value}</div><div class="l">{label}</div></div>'
    )


def _human(n: float) -> str:
    """Compact human count: 1_234_567 -> '1.2M'."""
    n = float(n)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.1f}{suf}"
    return f"{n:.0f}"


def _bar(frac: float, width: int = 22) -> str:
    """A Unicode progress bar for ``frac`` in [0, 1]."""
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    n = round(frac * width)
    return "█" * n + "░" * (width - n)


def render_dashboard(
    s: LedgerSummary,
    *,
    change_rate: float | None = None,
    samples: int = 0,
    recent: list[int] | None = None,
    subscription: bool = False,
    color: bool = True,
    session: "LedgerSummary | None" = None,
) -> str:
    """A framed, glanceable terminal dashboard of cumulative savings.

    Pure function (no I/O) so it's trivially testable; the live loop in the CLI
    re-renders it on an interval inside the alternate screen. ``change_rate`` is
    the decision-change rate from shadow mode (equivalence is ``1 - change_rate``)."""
    inner = 54  # visible width inside the frame

    def c(code: str, t: str) -> str:
        return f"\033[{code}m{t}\033[0m" if color else t

    def vlen(t: str) -> int:  # visible length, ignoring ANSI colour codes
        return len(re.sub(r"\033\[[0-9;]*m", "", t))

    def row(content: str = "") -> str:
        return "│ " + content + " " * max(0, inner - vlen(content)) + " │"

    top = "╭" + "─" * (inner + 2) + "╮"
    sep = "├" + "─" * (inner + 2) + "┤"
    bot = "╰" + "─" * (inner + 2) + "╯"

    out = [top, row(c("1;38;5;79", "distil") + c("90", "  ·  live savings")), sep]

    if s.runs == 0:
        out.append(row(c("90", "no savings yet — run ") + c("36", "distil wrap -- <agent>")))
        out.append(bot)
        return "\n".join(out)

    if session is not None and session.runs and session.total_baseline_tokens:
        if session.total_tokens_saved > 0:
            st = 1 - session.total_distil_tokens / session.total_baseline_tokens
            live = c("38;5;84", f"▼{_human(session.total_tokens_saved)} · {st * 100:.0f}% smaller")
        else:
            # match the status line — never a broken-looking "▼0 −0%"
            live = c("38;5;84", "✓ on") + c("38;5;80", " · waiting for a large read")
        out.append(row(f"{'this session':<15}{live}"))
    trimmed = (
        0.0 if s.total_baseline_tokens == 0 else 1 - s.total_distil_tokens / s.total_baseline_tokens
    )
    out.append(row(f"{'tokens':<15}{c('36', _bar(trimmed, 18))}  {trimmed * 100:4.1f}% trimmed"))
    out.append(
        row(c("90", f"{'':<15}{_human(s.total_baseline_tokens)} → {_human(s.total_distil_tokens)}"))
    )

    if subscription:
        out.append(row(f"{'cost':<15}" + c("90", "flat-rate subscription — $ notional")))
    else:
        saved = s.total_baseline_dollars - s.total_distil_dollars
        out.append(
            row(
                f"{'cost':<15}${s.total_baseline_dollars:,.2f} → ${s.total_distil_dollars:,.2f}   "
                + c("32", f"(${saved:,.2f} saved)")
            )
        )

    if samples >= 25 and change_rate is not None:
        # 25-sample floor, same as the status line — a rate over a handful is noise.
        eq = 1 - change_rate
        out.append(row(f"{'decision-equiv':<15}{c('35', _bar(eq, 18))}  {eq * 100:4.1f}%"))
        out.append(row(c("90", f"{'':<15}{samples:,} samples")))
    elif samples and change_rate is not None:
        out.append(
            row(
                f"{'decision-equiv':<15}"
                + c("90", f"collecting — {samples} sample{'s' if samples != 1 else ''} (need 25)")
            )
        )
        if recent:
            # Most-recent decisions, newest on the right: ▰ same action, ▱ changed.
            marks = "".join(c("32", "▰") if v else c("31", "▱") for v in recent[-24:])
            out.append(row(f"{'recent':<15}{marks}"))
    else:
        out.append(
            row(f"{'decision-equiv':<15}" + c("90", "— run ") + c("36", "distil wrap --shadow 0.1"))
        )

    out.append(row())
    out.append(row(c("90", f"{s.runs} run{'s' if s.runs != 1 else ''}")))

    if s.by_trajectory:
        top5 = sorted(s.by_trajectory.items(), key=lambda kv: kv[1], reverse=True)[:5]
        mx = max((v for _, v in top5), default=0.0) or 1.0
        out.append(sep)
        for name, val in top5:
            label = (name[:15] + "…") if len(name) > 16 else name
            tail = "" if subscription else f"  ${val:,.2f}"
            out.append(row(f"{label:<17}{c('36', _bar(val / mx, 12))}{tail}"))

    out.append(bot)
    return "\n".join(out)
