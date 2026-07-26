"""benchmark — head-to-head comparison of context-compression techniques.

Every compression approach reduces tokens. The honest, load-bearing questions
are: (1) how much does it save *without changing what the agent decides*, and
(2) what does it actually cost once prompt caching is priced in? A method that
shaves 80% of tokens but flips a decision, or that busts the prompt cache and
costs *more*, has not won anything.

This harness runs each technique through the SAME machinery distil holds itself
to — the decision-equivalence + non-inferiority gate (``distil.certify``) and
the cache-aware cost model (``distil.compress.cache_aware.simulate``) — over the
trajectory corpus, so the comparison is apples-to-apples and reproducible
offline. No technique is special-cased; the winner is computed, not assumed.

The built-in baselines are *faithful reference implementations* of the major
technique families used across the tooling landscape — sliding-window / tail
truncation (the most common agent-memory approach), extractive importance
pruning (the LLMLingua / Selective-Context family), abstractive summarisation
(rolling-summary memory), and naive minification. They are implemented in their
best reasonable form, not as strawmen.

To validate against a REAL external tool (e.g. an installed prompt compressor),
register it with ``register_external`` / the CLI ``--external module:function``
seam: it is a ``(list[str]) -> list[str]`` over per-turn block texts, measured
on the identical axes. So "distil leads" is something you can reproduce and
check against the actual package — not take on faith.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .certify.gate import certify
from .certify.stats import tost
from .compress.cache_aware import simulate
from .compress.strategies import Strategy
from .compress.strategies import distil as _distil
from .compress.tier0 import Tier0Lossless
from .corpus import CorpusEntry, load_corpus
from .pricing import Pricing
from .pricing import get as get_pricing
from .replay.ablation import discover
from .replay.runner import AgentRunner
from .tokenizer import DEFAULT, Tokenizer
from .trajectory import Block, Stability

_T0 = Tier0Lossless()

# A technique is built per-corpus-entry (so trajectory-specific methods like
# causal pruning can inspect their trajectory); plain methods ignore the entry.
TechniqueFactory = Callable[[CorpusEntry], Strategy]


@dataclass(frozen=True)
class Technique:
    name: str
    family: str
    reversible: bool  # byte-recoverable (lossless / digest) vs. lossy
    make: TechniqueFactory


# --------------------------------------------------------------------------- #
# Faithful reference implementations of the competing technique families.
# --------------------------------------------------------------------------- #


def _volatile_only(fn: Callable[[str], str]) -> Strategy:
    """Apply a text transform to VOLATILE blocks only (stable prefix untouched),
    with the reject-if-bigger invariant so a method never inflates a block."""

    def strat(blocks: list[Block], turn: int) -> list[Block]:
        out: list[Block] = []
        for b in blocks:
            if b.stability is Stability.VOLATILE:
                t = fn(b.text)
                out.append(b.copy_with(t) if len(t) < len(b.text) else b)
            else:
                out.append(b)
        return out

    return strat


def _minify_all(blocks: list[Block], turn: int) -> list[Block]:
    """Lossless whitespace/JSON minification across ALL blocks — the naive
    'just minify everything' approach. Lossless, but unlike distil it does not
    lift volatile fields out of the prefix, so it cannot stabilise a prefix that
    embeds changing data."""
    return _T0.compress(blocks).blocks


def _truncate_tail(text: str, head: int = 240, tail: int = 360) -> str:
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n…[{len(text) - head - tail} chars truncated]…\n{text[-tail:]}"


def _info(line: str) -> int:
    """Lexical informativeness proxy: count of distinct alphanumeric tokens.
    A stand-in for the self-information / perplexity score that extractive
    compressors (Selective Context, LLMLingua) use to rank tokens."""
    toks = {
        w for w in "".join(c if c.isalnum() else " " for c in line.lower()).split() if len(w) > 2
    }
    return len(toks)


def _extractive(text: str, keep: float = 0.6) -> str:
    """Drop the least-informative lines — extractive importance pruning."""
    lines = text.split("\n")
    if len(lines) <= 3:
        return text
    order = sorted(range(len(lines)), key=lambda i: _info(lines[i]))
    drop = set(order[: int(len(lines) * (1.0 - keep))])
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


def _summarize(text: str, max_lines: int = 6) -> str:
    """Abstractive / rolling-summary stand-in: keep the first and last lines and
    elide the middle. Deterministic so the benchmark is reproducible offline; a
    real LLM summariser would run via --runner anthropic and be measured the
    same way."""
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) <= max_lines:
        return text
    return f"{lines[0]}\n…[{len(lines) - 2} lines summarised]…\n{lines[-1]}"


def _causal_make(entry: CorpusEntry, runner: AgentRunner | None) -> Strategy:
    """distil's lossless pipeline PLUS dropping blocks that ablation proves never
    changed a decision — the full certified-aggressive distil."""
    rep = discover(entry.trajectory, runner=runner)
    prunable = {v.block_id for v in rep.prunable}

    def strat(blocks: list[Block], turn: int) -> list[Block]:
        kept = [b for b in blocks if b.id not in prunable]
        return _distil(kept, turn)

    return strat


def builtin_techniques(runner: AgentRunner | None = None) -> list[Technique]:
    """The faithful baseline families plus distil's own operating points."""
    ignore = lambda fn: lambda _entry: fn  # noqa: E731 — entry-agnostic adapter
    return [
        Technique("baseline (no compression)", "control", True, ignore(lambda b, t: b)),
        Technique("minify-all", "lossless minify", True, ignore(_minify_all)),
        Technique(
            "truncate-tail",
            "sliding-window / truncation",
            False,
            ignore(_volatile_only(_truncate_tail)),
        ),
        Technique(
            "extractive-prune",
            "extractive importance (LLMLingua family)",
            False,
            ignore(_volatile_only(_extractive)),
        ),
        Technique(
            "summarize", "abstractive / rolling summary", False, ignore(_volatile_only(_summarize))
        ),
        Technique("distil-lossless", "distil (cache-aware lossless)", True, ignore(_distil)),
        Technique(
            "distil-causal",
            "distil (cache-aware + causal pruning)",
            False,
            lambda entry: _causal_make(entry, runner),
        ),
        Technique(
            "distil-stream",
            "distil (causal + cross-turn dedup)",
            False,
            lambda entry: _stream_make(entry, runner),
        ),
        Technique(
            "distil-safe",
            "certified-fallback (byte-exact where the digest is unsafe)",
            True,
            lambda entry: _safe_make(entry, runner),
        ),
    ]


def _safe_make(entry: CorpusEntry, runner: AgentRunner | None) -> Strategy:
    """Certified-fallback: most aggressive transform per turn whose decision the
    runner confirms, else byte-exact, else none. 100% decision-equivalent by
    construction under the runner used to build it."""
    from .compress.adaptive import certified_fallback
    from .replay.runner import DeterministicRunner

    return certified_fallback(entry.trajectory, runner or DeterministicRunner())


def _stream_make(entry: CorpusEntry, runner: AgentRunner | None) -> Strategy:
    """The maximal REVERSIBLE Distil: cache-aware lossless (fold + template) PLUS
    cross-turn reversible dedup of recurring tool output the cache can't reach —
    everything recoverable, nothing pruned. Stateful per trajectory; the dedup
    memory auto-resets when a measurement pass restarts."""
    from .compress.dedup import StreamingDedup

    dd = StreamingDedup()

    def strat(blocks: list[Block], turn: int) -> list[Block]:
        compressed = _distil(blocks, turn)  # lossless fold/template, no pruning
        out, _restore = dd.compress(compressed, turn)
        return out

    return strat


def register_external(
    name: str,
    fn: Callable[[list[str]], list[str]],
    *,
    family: str = "external",
    reversible: bool = False,
) -> Technique:
    """Wrap a real external compressor (operating on per-turn block texts) into a
    Technique measured on the identical axes."""

    def make(_entry: CorpusEntry) -> Strategy:
        def strat(blocks: list[Block], turn: int) -> list[Block]:
            texts = fn([b.text for b in blocks])
            if len(texts) != len(blocks):
                raise ValueError(
                    f"external '{name}' returned {len(texts)} texts for {len(blocks)} blocks"
                )
            out: list[Block] = []
            for b, t in zip(blocks, texts):
                out.append(b.copy_with(t) if len(t) < len(b.text) else b)
            return out

        return strat

    return Technique(name, family, reversible, make)


def load_external(spec: str) -> Technique:
    """Load an external technique from a ``module:function[:Display Name]`` spec."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError("external spec must be 'module:function[:name]'")
    mod, func = parts[0], parts[1]
    name = parts[2] if len(parts) > 2 else f"{mod}:{func}"
    fn = getattr(importlib.import_module(mod), func)
    return register_external(name, fn)


# --------------------------------------------------------------------------- #
# Measurement — every technique through the same gate and cost model.
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    name: str
    family: str
    reversible: bool
    token_savings: float  # fraction of tokens removed vs. baseline
    dollar_savings: float  # cache-aware $ vs. baseline (both WITH caching)
    equivalence: float  # decision-equivalence match rate
    certified: bool  # non-inferior (TOST) AND 100% decision-equivalence


@dataclass
class BenchReport:
    results: list[Result] = field(default_factory=list)
    runner: str = "deterministic"
    model: str = ""

    @property
    def certified(self) -> list[Result]:
        return [r for r in self.results if r.certified]

    @property
    def winner(self) -> Result | None:
        """Highest cache-aware $ savings AMONG techniques that pass the gate."""
        passing = [r for r in self.certified if r.name != "baseline (no compression)"]
        return max(passing, key=lambda r: r.dollar_savings) if passing else None

    @property
    def raw_leader(self) -> Result | None:
        """Highest raw token savings regardless of certification — usually a
        lossy method that the gate then disqualifies."""
        cand = [r for r in self.results if r.name != "baseline (no compression)"]
        return max(cand, key=lambda r: r.token_savings) if cand else None


def run_benchmark(
    entries: list[CorpusEntry] | None = None,
    techniques: list[Technique] | None = None,
    *,
    pricing: Pricing | None = None,
    runner: AgentRunner | None = None,
    tok: Tokenizer = DEFAULT,
    margin: float = 0.02,
    alpha: float = 0.05,
) -> BenchReport:
    entries = entries if entries is not None else load_corpus()
    techniques = techniques if techniques is not None else builtin_techniques(runner)
    price = pricing if pricing is not None else get_pricing("claude-opus-4-8")

    # Baseline cost: no compression, WITH caching — the realistic reference, so
    # the comparison isolates each method's COMPRESSION effect on top of caching
    # (and a cache-busting method correctly shows up as costing MORE).
    base_dollars = sum(
        simulate(e.trajectory, price, strategy="none", caching=True, tok=tok).total_dollars
        for e in entries
    )

    report = BenchReport(runner=getattr(runner, "name", "deterministic"), model=price.name)
    for tech in techniques:
        base_tok = comp_tok = 0
        dollars = 0.0
        diffs: list[float] = []
        matches: list[float] = []
        for e in entries:
            strat = tech.make(e)
            rep = certify(e.trajectory, strat, runner=runner, margin=margin, alpha=alpha)
            matches.append(rep.match_rate)
            diffs += [(1.0 if d.matched else 0.0) - 1.0 for d in rep.divergences]
            for turn in e.trajectory.turns:
                base_tok += sum(tok.count(b.text) for b in turn.blocks)
                comp_tok += sum(tok.count(b.text) for b in strat(turn.blocks, turn.index))
            dollars += simulate(
                e.trajectory, price, strategy=strat, caching=True, tok=tok
            ).total_dollars
        token_savings = (1.0 - comp_tok / base_tok) if base_tok else 0.0
        dollar_savings = (1.0 - dollars / base_dollars) if base_dollars else 0.0
        equivalence = sum(matches) / len(matches) if matches else 0.0
        certified = (
            bool(diffs)
            and tost(diffs, margin=margin, alpha=alpha).non_inferior
            and equivalence == 1.0
        )
        report.results.append(
            Result(
                tech.name,
                tech.family,
                tech.reversible,
                token_savings,
                dollar_savings,
                equivalence,
                certified,
            )
        )
    return report


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_report(report: BenchReport) -> str:
    rows = sorted(report.results, key=lambda r: r.dollar_savings, reverse=True)
    out = [
        f"compression benchmark  (model={report.model}, runner={report.runner}, corpus gate)",
        "",
        f"{'technique':<24}{'tok save':>9}{'$ save':>8}{'equiv':>7}{'certified':>11}{'reversible':>12}",
        "-" * 81,
    ]
    for r in rows:
        cert = "✔ PASS" if r.certified else "✘ fails"
        rev = "byte-exact" if r.reversible else "lossy"
        out.append(
            f"{r.name:<24}{r.token_savings * 100:>8.1f}%{r.dollar_savings * 100:>7.1f}%"
            f"{r.equivalence * 100:>6.0f}%{cert:>11}{rev:>12}"
        )
    out.append("-" * 81)
    w, raw = report.winner, report.raw_leader
    if w:
        out.append(
            f"\nLEADER (certified $ savings): {w.name} — {w.dollar_savings * 100:.1f}% cheaper, "
            f"{w.token_savings * 100:.1f}% fewer tokens, {w.equivalence * 100:.0f}% decision-equivalent."
        )
    if raw and (not w or raw.name != w.name) and not raw.certified:
        out.append(
            f"note: {raw.name} removes more raw tokens ({raw.token_savings * 100:.1f}%) but "
            f"FAILS the decision-equivalence gate ({raw.equivalence * 100:.0f}% equiv) — "
            "disqualified. Raw token savings is not savings if it changes the answer."
        )
    out.append(
        "\nEvery technique scored on the same corpus, gate, and cache-aware cost model. "
        "Plug a real tool in with --external module:function to verify."
    )
    return "\n".join(out)


def render_html(report: BenchReport) -> str:
    rows = sorted(report.results, key=lambda r: r.dollar_savings, reverse=True)
    body = ""
    for r in rows:
        cert = (
            '<span style="color:#5ad19a;font-weight:700">✔ certified</span>'
            if r.certified
            else '<span style="color:#ff8a6b">✘ fails gate</span>'
        )
        rev = "byte-exact" if r.reversible else "lossy"
        hl = (
            ' style="background:rgba(139,123,255,.07)"'
            if report.winner and r.name == report.winner.name
            else ""
        )
        body += (
            f'<tr{hl}><td>{r.name}</td><td class="d">{r.family}</td>'
            f'<td class="r">{r.token_savings * 100:.1f}%</td>'
            f'<td class="r">{r.dollar_savings * 100:.1f}%</td>'
            f'<td class="r">{r.equivalence * 100:.0f}%</td><td>{cert}</td><td class="d">{rev}</td></tr>'
        )
    w = report.winner
    verdict = (
        f"Leader on certified savings: <b>{w.name}</b> — {w.dollar_savings * 100:.1f}% cheaper "
        f"at {w.equivalence * 100:.0f}% decision-equivalence."
        if w
        else "No technique certified."
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Distil — compression benchmark</title><style>
body{{margin:0;background:#06070b;color:#f2f3f7;font:15px/1.6 Inter,ui-sans-serif,sans-serif}}
.wrap{{max-width:880px;margin:0 auto;padding:48px 24px}}
h1{{font-size:28px;font-weight:800;letter-spacing:-.02em;margin:0 0 6px}}
.sub{{color:#9aa1b3;margin:0 0 24px}}
.g{{background:linear-gradient(135deg,#8b7bff,#5ad1c9);-webkit-background-clip:text;background-clip:text;color:transparent}}
table{{width:100%;border-collapse:collapse;border:1px solid #1b2030;border-radius:12px;overflow:hidden;font-size:13.5px}}
th,td{{padding:10px 13px;border-bottom:1px solid #1b2030;text-align:left}}
th{{color:#7d8598;font-size:11px;text-transform:uppercase;letter-spacing:.07em}}
td.r{{text-align:right;font-variant-numeric:tabular-nums}} td.d{{color:#9aa1b3}}
.foot{{color:#7d8598;font-size:12.5px;margin-top:20px}}
caption.sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
 clip:rect(0,0,0,0);white-space:nowrap;border:0}}
</style></head><body><div class="wrap">
<h1>Compression <span class="g">benchmark</span></h1>
<p class="sub">model {report.model} · runner {report.runner} · every technique through the same
decision-equivalence + non-inferiority gate and cache-aware cost model.</p>
<table><caption class="sr-only">Compression technique standings</caption>
<thead><tr><th scope="col">technique</th><th scope="col">family</th>
<th scope="col" style="text-align:right">tok save</th>
<th scope="col" style="text-align:right">$ save</th>
<th scope="col" style="text-align:right">equiv</th><th scope="col">verdict</th>
<th scope="col">fidelity</th></tr></thead><tbody>{body}</tbody></table>
<p class="foot">{verdict} Raw token savings that fail the gate are disqualified — savings that
change the answer aren't savings. Reproduce offline: <code>distil benchmark</code>; verify a real
tool: <code>distil benchmark --external module:function</code>.</p>
</div></body></html>"""


def write_raw(report: BenchReport, out_dir: str, stamp: str) -> str:
    import json

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"benchmark-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in report.results:
            f.write(
                json.dumps(
                    {
                        "name": r.name,
                        "family": r.family,
                        "reversible": r.reversible,
                        "token_savings": r.token_savings,
                        "dollar_savings": r.dollar_savings,
                        "equivalence": r.equivalence,
                        "certified": r.certified,
                        "model": report.model,
                        "runner": report.runner,
                    }
                )
                + "\n"
            )
    return str(path)
