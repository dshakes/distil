"""``distil savings`` — the one screen: what you spent, what distil saved, what to fix next.

Composes what already exists rather than re-deriving it:

* **spent** — the provider's own ``usage`` on every booked request the proxy recorded
  (``sessions/*.requests.jsonl``), priced per model through :mod:`distil.pricing` with
  cache reads and writes at their own rates. Ground truth about money.
* **saved** — the savings ledger (``savings.jsonl``), scaled by the learned
  :mod:`distil.calibration` factor so heuristic token counts sit on the billed scale.
* **what to fix next** — the top of :func:`distil.discover.scan`.
* **proof** — one verdict line from :func:`distil.proof_ledger.proof_lines`.

With no ledger at all (distil never wrapped anything) it reads Claude Code's own
transcripts instead — provider ``usage`` fields and tool-result *sizes* only; no
content ever reaches the output — and prints real spend plus a labelled ESTIMATE.
Read-only everywhere: this module never writes a file.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from . import ledger, pricing

#: The realized saving measured on live traffic: saved $ as a share of what the bill
#: would have been without distil. Digest mode, one machine, 13,191 requests.
#: Source: benchmarks/results/2026-09-24/live_savings_decomposition.json
#: (``realized.saved_share_of_counterfactual_bill``). A test pins these to the artifact.
EST_SAVED_SHARE = 0.1024
EST_REQUESTS = 13191
EST_SOURCE = "benchmarks/results/2026-09-24/live_savings_decomposition.json"
#: Tool DEFINITIONS never appear in a transcript, so their share is quoted, not measured.
#: Source: benchmarks/results/2026-09-24/tool_schema_share.json (``tool_share_of_billed_usd``).
TOOL_DEFS_SHARE = 0.3564
TOOL_DEFS_SOURCE = "benchmarks/results/2026-09-24/tool_schema_share.json"

GRAPH_WIDTH = 20
MAX_GRAPH_DAYS = 30  # ponytail: --all on a long history shows the last 30 active days
_PARTIAL = " ▏▎▍▌▋▊▉"


@dataclass
class Tokens:
    uncached: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0


@dataclass
class Day:
    date: str
    spent_usd: float = 0.0
    saved_usd: float = 0.0


@dataclass
class Screen:
    mode: str  # "ledger" | "transcripts" | "empty"
    since: float | None
    days: float
    spent_usd: float = 0.0
    saved_usd: float = 0.0
    saved_tokens: int = 0
    requests: int = 0
    unpriced_requests: int = 0
    tokens: Tokens = field(default_factory=Tokens)
    daily: list[Day] = field(default_factory=list)
    findings: list[dict[str, str]] = field(default_factory=list)
    proof: str | None = None
    notional: bool = False
    calibration_factor: float = 1.0
    #: $/token you actually paid for input in this window (cache reads + writes +
    #: uncached, blended). Saved tokens are priced at it: a token distil removed would
    #: have been billed like the input around it, mostly as a 0.1x cache read.
    blended_input_usd_per_mtok: float | None = None
    #: Transcript mode only: heuristic share of new context that was tool output.
    tool_results_share: float | None = None
    estimate: dict[str, Any] | None = None

    @property
    def net_pct(self) -> float | None:
        """Saved as a share of what you would have paid (spent + saved)."""
        whole = self.spent_usd + self.saved_usd
        return (self.saved_usd / whole * 100.0) if whole > 0 and self.mode == "ledger" else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["net_pct"] = None if self.net_pct is None else round(self.net_pct, 2)
        for k in ("spent_usd", "saved_usd"):
            d[k] = round(d[k], 4)
        for day in d["daily"]:
            day["spent_usd"] = round(day["spent_usd"], 4)
            day["saved_usd"] = round(day["saved_usd"], 4)
        return d


# --------------------------------------------------------------------------- inputs


def parse_since(text: str) -> float:
    """``7d`` / ``24h`` / ``2w`` / bare ``7`` (days) → seconds. ValueError otherwise."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([hdw]?)\s*", text or "")
    if not m:
        raise ValueError(f"cannot read {text!r} as a window — use e.g. 7d, 24h, 2w")
    n = float(m.group(1))
    return n * {"h": 3600, "d": 86400, "w": 7 * 86400, "": 86400}[m.group(2)]


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def request_cost(model: str | None, t: Tokens, write_1h: int = 0) -> float | None:
    """Dollars for one request's billed usage, or None for a model we cannot price.

    ``write_1h`` is the part of ``t.cache_write`` written with a 1-hour TTL (2x input);
    the rest is priced as a 5-minute write (1.25x). Reads are 0.10x."""
    p = pricing.resolve(model)
    if p is None:
        return None
    write_5m = max(0, t.cache_write - write_1h)
    return (
        t.uncached * p.input
        + t.cache_read * p.cache_read
        + write_5m * p.cache_write
        + write_1h * p.cache_write_1h
        + t.output * p.output
    )


def _add(acc: Tokens, t: Tokens) -> None:
    acc.uncached += t.uncached
    acc.cache_read += t.cache_read
    acc.cache_write += t.cache_write
    acc.output += t.output


def _files(root: Path, pattern: str, since: float | None) -> Iterator[Path]:
    try:
        paths = sorted(root.glob(pattern))
    except OSError:
        return
    for p in paths:
        try:
            if since is None or p.stat().st_mtime >= since:
                yield p
        except OSError:
            continue


def _lines(path: Path) -> Iterator[str]:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            yield from fh
    except OSError:
        return


def _daily(buckets: dict[str, Day], since: float | None, now: float) -> list[Day]:
    """Consecutive days (zero days included, so the graph's gaps are real)."""
    if not buckets:
        return []
    stop = _day(since) if since is not None else min(buckets)
    out: list[Day] = []
    d = date.fromtimestamp(now)  # calendar steps: a 25-hour DST day must not repeat a date
    while d.isoformat() >= stop and len(out) < MAX_GRAPH_DAYS:
        out.append(buckets.get(d.isoformat(), Day(d.isoformat())))
        d -= timedelta(days=1)
    return list(reversed(out))


# --------------------------------------------------------------------------- ledger mode


def _ledger_rows(since: float | None) -> list[dict[str, Any]]:
    out = []
    for line in _lines(ledger.default_path()):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and (since is None or float(d.get("ts") or 0) >= since):
            out.append(d)
    return out


def _proof_line() -> str | None:
    try:
        from .proof_ledger import _safe_proof_lines

        lines = _safe_proof_lines()
    except Exception:  # noqa: BLE001 — a verdict is advisory; the accounting still prints
        return None
    return lines[0][1] if lines else None


def _findings(since_days: float | None) -> list[dict[str, str]]:
    try:
        from . import discover

        report = discover.scan(sessions=20, since_days=since_days)
    except Exception:  # noqa: BLE001 — an advisor must never cost the savings screen
        return []
    return [{"title": a.title, "command": a.command} for a in report.actions[:3]]


def ledger_screen(since: float | None, *, now: float | None = None) -> Screen:
    from . import calibration
    from .doctor import subscription_mode

    now = now or time.time()
    days = (now - since) / 86400 if since is not None else 0.0
    s = Screen("ledger", since, days)
    buckets: dict[str, Day] = {}
    in_usd = 0.0
    in_tok = 0

    sessions = ledger.default_path().parent / "sessions"
    for path in _files(sessions, "*.requests.jsonl", since):
        for line in _lines(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict) or not r.get("booked"):
                continue
            ts = float(r.get("ts") or 0)
            if since is not None and ts < since:
                continue
            t = Tokens(
                int(r.get("usage_input_tokens") or 0),
                int(r.get("usage_cache_read") or 0),
                int(r.get("usage_cache_create") or 0),
                int(r.get("usage_output_tokens") or 0),
            )
            s.requests += 1
            _add(s.tokens, t)
            # The proxy records writes without the 1h/5m split, so they price at 5m.
            usd = request_cost(r.get("model"), t)
            if usd is None:
                s.unpriced_requests += 1
                continue
            s.spent_usd += usd
            in_usd += usd - (request_cost(r.get("model"), Tokens(output=t.output)) or 0.0)
            in_tok += t.uncached + t.cache_read + t.cache_write
            buckets.setdefault(_day(ts), Day(_day(ts))).spent_usd += usd

    f, _n = calibration.factor()
    s.calibration_factor = f
    # ponytail: one blended rate for the window; per-request rates if models mix a lot.
    rate = in_usd / in_tok if in_tok else None
    s.blended_input_usd_per_mtok = None if rate is None else round(rate * 1e6, 4)
    for r in _ledger_rows(since):
        try:
            tok = round((int(r["baseline_input_tokens"]) - int(r["distil_input_tokens"])) * f)
            # No billed usage in the window (pre-detail ledger): the ledger's own price,
            # which is the fresh-input list rate — an upper bound, and labelled as one.
            usd = (
                tok * rate
                if rate is not None
                else (float(r["baseline_dollars"]) - float(r["distil_dollars"])) * f
            )
        except (KeyError, TypeError, ValueError):
            continue
        s.saved_usd += usd
        s.saved_tokens += tok
        key = _day(float(r.get("ts") or 0))
        buckets.setdefault(key, Day(key)).saved_usd += usd

    s.daily = _daily(buckets, since, now)
    s.findings = _findings(days or None)
    s.proof = _proof_line()
    s.notional = subscription_mode()
    return s


# --------------------------------------------------------------------------- no-install mode


def claude_projects_root() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))) / "projects"


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict) and "text" in b
        )
    return ""


def _usage_tokens(u: dict[str, Any]) -> tuple[Tokens, int]:
    t = Tokens(
        int(u.get("input_tokens") or 0),
        int(u.get("cache_read_input_tokens") or 0),
        int(u.get("cache_creation_input_tokens") or 0),
        int(u.get("output_tokens") or 0),
    )
    split = u.get("cache_creation")
    w1h = int(split.get("ephemeral_1h_input_tokens") or 0) if isinstance(split, dict) else 0
    return t, min(w1h, t.cache_write)


def transcripts_screen(
    since: float | None, *, root: Path | None = None, now: float | None = None
) -> Screen:
    """Real spend from Claude Code's transcripts. Content is parsed in memory only to
    measure tool-result SIZE; nothing but numbers leaves this function."""
    from .doctor import subscription_mode
    from .tokenizer import HeuristicTokenizer

    now = now or time.time()
    root = root or claude_projects_root()
    days = (now - since) / 86400 if since is not None else 0.0
    s = Screen("transcripts", since, days)
    buckets: dict[str, Day] = {}
    tok = HeuristicTokenizer()
    seen: set[str] = set()  # resumed sessions copy history into a new file: count once
    tool_result_tokens = 0

    for path in _files(root, "**/*.jsonl", since):
        for line in _lines(path):
            if '"usage"' not in line and '"tool_result"' not in line:
                continue  # cheap pre-filter: most lines are neither
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            ts = _epoch(rec.get("timestamp"))
            if since is not None and ts < since:
                continue
            kind = rec.get("type")
            if kind == "assistant" and isinstance(msg.get("usage"), dict):
                # One API response is written as one line PER content block, each carrying
                # the same usage. Keyed on the message id, or the bill counts it N times.
                key = str(msg.get("id") or rec.get("requestId") or rec.get("uuid") or "")
                if not key or key in seen:
                    continue
                seen.add(key)
                t, w1h = _usage_tokens(msg["usage"])
                if not (t.uncached or t.cache_read or t.cache_write or t.output):
                    continue
                s.requests += 1
                _add(s.tokens, t)
                usd = request_cost(msg.get("model"), t, w1h)
                if usd is None:
                    s.unpriced_requests += 1
                    continue
                s.spent_usd += usd
                buckets.setdefault(_day(ts), Day(_day(ts))).spent_usd += usd
            elif kind == "user" and isinstance(msg.get("content"), list):
                key = "u:" + str(rec.get("uuid") or "")
                if key != "u:" and key in seen:
                    continue
                seen.add(key)
                for b in msg["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tool_result_tokens += tok.count(_tool_result_text(b.get("content")))

    fresh = s.tokens.uncached + s.tokens.cache_write
    s.tool_results_share = min(1.0, tool_result_tokens / fresh) if fresh else None
    s.daily = _daily(buckets, since, now)
    s.notional = subscription_mode()
    if s.spent_usd > 0:
        s.estimate = {
            "label": "ESTIMATE",
            "saved_usd": round(s.spent_usd * EST_SAVED_SHARE, 4),
            "saved_share": EST_SAVED_SHARE,
            "basis": (
                f"realized saving in digest mode on one maintainer machine "
                f"({EST_REQUESTS:,} live requests), not your traffic"
            ),
            "source": EST_SOURCE,
            "tool_definitions_share_of_billed_usd": TOOL_DEFS_SHARE,
            "tool_definitions_source": TOOL_DEFS_SOURCE,
        }
    if s.requests == 0:
        s.mode = "empty"
    return s


def _epoch(iso: Any) -> float:
    from .transcripts.claude_code import _epoch as ep

    return ep(iso if isinstance(iso, str) else None)


def build(since: float | None) -> Screen:
    """Ledger mode if distil has ever recorded a run; the transcript reader otherwise."""
    if ledger.summary().runs:
        return ledger_screen(since)
    return transcripts_screen(since)


# --------------------------------------------------------------------------- render


def bar(value: float, scale: float, width: int = GRAPH_WIDTH) -> str:
    """A left-aligned unicode bar, 1/8-cell resolution, fixed ``width``."""
    if scale <= 0 or value <= 0:
        return " " * width
    eighths = max(1, round(value / scale * width * 8))
    full, part = divmod(min(eighths, width * 8), 8)
    return ("█" * full + (_PARTIAL[part] if part else "")).ljust(width)


def _k(n: float) -> str:
    return ledger._human(int(n))


def render(s: Screen) -> str:
    window = "all time" if s.since is None else f"last {round(s.days, 1):g} days"
    out = [f"distil savings  ·  {window}", ""]
    if s.mode == "empty":
        out += [
            "  nothing to show yet — no distil ledger, and no Claude Code transcripts",
            f"  in this window under {claude_projects_root()}.",
            "",
            "  start:  distil setup      then:  distil wrap -- claude",
        ]
        return "\n".join(out)

    note = "  (API list price — you are on a flat-rate plan)" if s.notional else ""
    t = s.tokens
    out.append(
        f"  spent    ${s.spent_usd:,.2f}{note}"
        + ("" if s.requests else "  (no billed usage recorded in this window)")
    )
    if s.requests:
        out.append(
            f"           {s.requests:,} requests · cache reads {_k(t.cache_read)} · "
            f"writes {_k(t.cache_write)} · uncached {_k(t.uncached)} · output {_k(t.output)}"
        )
    if s.unpriced_requests:
        out.append(f"           ({s.unpriced_requests:,} requests on an unpriced model, not in $)")

    if s.mode == "ledger":
        cal = f", calibrated ×{s.calibration_factor:g}" if s.calibration_factor != 1.0 else ""
        out.append(f"  saved    ${s.saved_usd:,.2f}  ({s.saved_tokens:,} tokens{cal})")
        if s.blended_input_usd_per_mtok is not None:
            out.append(
                f"           priced at your blended input rate, "
                f"${s.blended_input_usd_per_mtok:,.2f}/Mtok (cache reads included)"
            )
        elif s.saved_tokens:
            out.append(
                "           priced at the list input rate — an upper bound (no billed usage)"
            )
        if s.net_pct is not None:
            out.append(f"  net      {s.net_pct:.1f}% of what you would have paid")
    else:
        out.append("")
        if s.tool_results_share is not None:
            out.append(
                f"  tool results   ≈{s.tool_results_share * 100:.0f}% of the new context you "
                "sent (your transcripts)"
            )
        out.append(
            f"  tool defs      {TOOL_DEFS_SHARE * 100:.0f}% of billed $ on a measured machine — "
            "not in transcripts"
        )
        if s.estimate:
            e = s.estimate
            out += [
                "",
                f"  ESTIMATE  distil would save ≈${e['saved_usd']:,.2f} "
                f"({e['saved_share'] * 100:.1f}% of this spend)",
                f"            basis: {e['basis']}",
                f"            source: {e['source']}",
            ]
            if s.notional:
                out.append(
                    "            on a subscription distil defaults to lossless-only, "
                    "which saves far less"
                )

    if s.daily:
        scale = max(max(d.spent_usd for d in s.daily), max(d.saved_usd for d in s.daily))
        head = "spent" if s.mode != "ledger" else "spent".ljust(GRAPH_WIDTH + 11) + "saved"
        out += ["", f"  {'day':<7}{head}"]
        for d in s.daily:
            row = f"  {d.date[5:]:<7}{bar(d.spent_usd, scale)} ${d.spent_usd:>8,.2f}"
            if s.mode == "ledger":
                row += f"  {bar(d.saved_usd, scale)} ${d.saved_usd:>7,.2f}"
            out.append(row.rstrip())

    if s.mode == "ledger":
        if s.findings:
            out += ["", "  what to fix next"]
            for i, f in enumerate(s.findings, 1):
                out.append(f"    {i}. {f['title']}")
                out.append(f"       → {f['command']}")
        if s.proof:
            out += ["", f"  proof    {s.proof}"]
        out += ["", "  more: distil stats · distil discover · distil dissect latest · distil cache"]
    else:
        out += [
            "",
            "  what to do next",
            "    1. set up once                    →  distil setup",
            "    2. run your agent through distil  →  distil wrap -- claude",
            "    3. come back for measured numbers →  distil savings",
        ]
        out.append("\n  read-only: only usage fields and tool-result sizes were read; no content.")
    return "\n".join(out)


def to_json(s: Screen) -> str:
    return json.dumps(s.to_dict(), indent=2)
