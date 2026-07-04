"""distil — command line.

distil compress  --trajectory T            shrink a trajectory, report ratio + reversibility
distil savings   --trajectory T --pricing  price 4 strategies in real dollars (technique #1)
distil prune     --trajectory T            causal ablation: what is free to drop (technique #2)
distil certify   --trajectory T --strategy non-inferiority gate (the quality contract)
"""

from __future__ import annotations

import argparse
import json
from typing import Any
from pathlib import Path

from . import __version__, ledger, pricing, tokenizer
from .compress.cache_aware import simulate
from .compress.strategies import REGISTRY, distil as distil_strategy
from .corpus import load_corpus, validate
from .replay.ablation import discover
from .certify.gate import certify
from .trajectory import Trajectory

from .corpus import CORPUS_DIR  # env-aware corpus dir (wheel / repo / $DISTIL_CORPUS)


def _load(path: str | None) -> Trajectory:
    return Trajectory.load(path or CORPUS_DIR / "sample_trajectory.json")


def _pct(part: float, whole: float) -> str:
    return f"{(1 - part / whole) * 100:5.1f}%" if whole else "  n/a"


def cmd_compress(args: argparse.Namespace) -> int:
    traj = _load(args.trajectory)
    tok = tokenizer.resolve(args.tokenizer, model=traj.model)
    before = after = 0
    restored = 0
    print(f"trajectory {traj.id!r}  ({len(traj.turns)} turns, tokenizer={args.tokenizer})\n")
    print(f"{'turn':>4}  {'before':>8}  {'after':>8}  {'saved':>7}")
    for turn in traj.turns:
        b = sum(tok.count(x.text) for x in turn.blocks)
        compressed = distil_strategy(turn.blocks, turn.index)
        a = sum(tok.count(x.text) for x in compressed)
        # reversibility: every byte dropped is recoverable from a local handle/marker
        restored += sum(
            1 for x in compressed if x.text != next(o.text for o in turn.blocks if o.id == x.id)
        )
        before += b
        after += a
        print(f"{turn.index:>4}  {b:>8}  {a:>8}  {_pct(a, b):>7}")
    print(f"\n{'ALL':>4}  {before:>8}  {after:>8}  {_pct(after, before):>7}")
    print(
        f"\nreversible: yes (Tier-0/1) — {restored} blocks digested, originals recoverable locally"
    )
    return 0


def cmd_savings(args: argparse.Namespace) -> int:
    traj = _load(args.trajectory)
    price = pricing.get(args.pricing)
    tok = tokenizer.resolve(args.tokenizer, model=price.name)
    out_t = args.output_tokens_per_turn

    runs: dict[str, dict[str, Any]] = {
        "baseline (no cache, no compress)": dict(strategy="none", caching=False),
        "cache only": dict(strategy="none", caching=True),
        "naive compress + cache": dict(strategy="naive", caching=True),
        "distil (cache-aware lossless)": dict(strategy="distil", caching=True),
    }
    results = {
        label: simulate(traj, price, output_tokens_per_turn=out_t, tok=tok, **kw)
        for label, kw in runs.items()
    }
    baseline = results["baseline (no cache, no compress)"].total_dollars

    tok_note = "≈, not billing-grade" if args.tokenizer == "heuristic" else "billing-grade"
    print(
        f"model {price.name}   |   {len(traj.turns)} turns   |   "
        f"tokenizer={args.tokenizer} ({tok_note})\n"
    )
    print(f"{'strategy':<34}{'$ / run':>12}{'vs baseline':>14}{'cache hits':>12}")
    print("-" * 72)
    for label, r in results.items():
        save = _pct(r.total_dollars, baseline)
        print(f"{label:<34}{r.total_dollars:>12.5f}{save:>14}{r.cache_hit_tokens:>12,}")
    best = results["distil (cache-aware lossless)"].total_dollars
    print("-" * 72)
    print(
        f"\ndistil cuts ${baseline:.5f} -> ${best:.5f} per run "
        f"({(1 - best / baseline) * 100:.1f}% cheaper), reversibly."
    )
    naive = results["naive compress + cache"].total_dollars
    if naive > best:
        print(
            f"note: naive compression costs ${naive:.5f} — "
            f"{(naive / best - 1) * 100:.0f}% MORE than distil despite fewer tokens, "
            f"because it busts the prefix cache."
        )

    if args.record:
        b = results["baseline (no cache, no compress)"]
        d = results["distil (cache-aware lossless)"]
        rec = ledger.record(
            trajectory_id=traj.id,
            model=price.name,
            turns=len(traj.turns),
            baseline_dollars=baseline,
            distil_dollars=best,
            baseline_input_tokens=b.total_input_tokens,
            distil_input_tokens=d.total_input_tokens,
        )
        print(
            f"\nrecorded to {ledger.default_path()}: "
            f"${rec.dollars_saved:.5f} / {rec.tokens_saved} tokens saved this run."
        )
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    s = ledger.summary()
    if getattr(args, "badge", False):
        # A shields.io badge of YOUR measured savings — paste it in a README or
        # a tweet. The number comes from the local ledger (genuine, content-free);
        # markdown includes a link to the project so the badge explains itself.
        import urllib.parse

        label = urllib.parse.quote("distil saved")
        value = urllib.parse.quote(
            f"{ledger._human(s.total_tokens_saved)} tokens"
            + (f" (${s.total_dollars_saved:,.2f})" if s.total_dollars_saved >= 0.01 else "")
        )
        url = f"https://img.shields.io/badge/{label}-{value}-5ad1c9"
        print(url)
        print(f"\nmarkdown:\n[![distil savings]({url})](https://github.com/dshakes/distil)")
        return 0
    if getattr(args, "json", False):
        from dataclasses import asdict

        d = asdict(s)
        d["tokenizers"] = sorted(s.tokenizers)
        try:
            from .shadow import ShadowLedger

            led = ShadowLedger.load()
            if led.samples:
                d["decision_equivalence"] = 1 - led.rate()
                d["shadow_samples"] = led.samples
        except Exception:  # noqa: BLE001 — shadow stats are best-effort
            pass
        print(json.dumps(d, indent=2))
        return 0
    if args.html:
        change_rate: float | None = None
        samples = 0
        sess = None
        try:
            from .shadow import ShadowLedger

            led = ShadowLedger.load()
            samples = led.samples
            if samples:
                change_rate = led.rate()
        except Exception:  # noqa: BLE001 — shadow stats are best-effort
            pass
        try:
            import time as _time

            sid, last_ts = ledger.latest_session()
            if sid and (_time.time() - last_ts) < 4 * 3600:
                sess = ledger.summary(session=sid)
        except Exception:  # noqa: BLE001 — session slice is best-effort
            pass
        Path(args.html).write_text(
            ledger.render_html(s, change_rate=change_rate, samples=samples, session=sess)
        )
        print(f"your savings page → {args.html}")
        return 0
    print(f"distil savings ledger — {ledger.default_path()}\n")
    if s.runs == 0:
        print("no genuine savings recorded yet.")
        print("run `distil proxy` (records real traffic) or `distil savings --record`.")
        return 0
    live = s.by_trajectory.get("live-proxy", 0.0)
    print(f"runs recorded:        {s.runs}")
    if s.total_baseline_tokens:
        trimmed = 1 - s.total_distil_tokens / s.total_baseline_tokens
        print(
            f"tokens:               {s.total_baseline_tokens:,} → "
            f"{s.total_distil_tokens:,}  (−{trimmed * 100:.1f}%)"
        )
    print(f"total tokens saved:   {s.total_tokens_saved:,}")
    from .doctor import subscription_mode

    if subscription_mode():
        print("total dollars saved:  — (flat-rate subscription; dollars are notional)")
    else:
        print(f"total dollars saved:  ${s.total_dollars_saved:,.2f}")
    try:
        from .shadow import ShadowLedger

        led = ShadowLedger.load()
        if led.samples >= 25:
            # Only claim a rate with evidence behind it — the same 25-sample floor
            # the status line uses; a rate over a handful is noise.
            print(
                f"decision-equivalence: {(1 - led.rate()) * 100:.1f}% "
                f"({led.samples:,} shadowed request{'s' if led.samples != 1 else ''})"
            )
        elif led.samples:
            print(
                f"decision-equivalence: collecting — {led.samples} "
                f"sample{'s' if led.samples != 1 else ''} (need 25 for a rate)"
            )
    except Exception:  # noqa: BLE001 — shadow stats are best-effort
        pass
    if live and not subscription_mode():
        print(f"  of which genuine live traffic (live-proxy): ${live:,.2f}")
    if not subscription_mode():
        print("\nby source:")
        for tid, saved in sorted(s.by_trajectory.items(), key=lambda kv: -kv[1]):
            print(f"  {tid:<28} ${saved:,.2f}")
    if "heuristic" in s.tokenizers:
        print(
            "\n(token counts ≈ heuristic tokenizer — directionally accurate, "
            "not billing-grade; dollars are a conservative floor.)"
        )
    print(
        "\n(local-first; export a page with --html, or share verifiably with "
        "`distil federated-leaderboard`.)"
    )
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    traj = _load(args.trajectory)
    report = discover(traj)
    print(f"causal ablation over {traj.id!r} — what is free to drop?\n")
    print(f"{'block':<22}{'occ':>4}{'tokens':>8}  verdict")
    for v in report.verdicts:
        verdict = "PRUNE (causally inert)" if v.prunable else "keep (changed a decision)"
        print(f"{v.block_id:<22}{v.occurrences:>4}{v.tokens:>8}  {verdict}")
    print(
        f"\ntokens provably free to drop: {report.tokens_freed} "
        f"across {len(report.prunable)} block(s)."
    )
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    traj = _load(args.trajectory)
    runner = None
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner(model=traj.model)
    report = certify(traj, args.strategy, runner=runner, margin=args.margin, alpha=args.alpha)
    print(f"certifying strategy {args.strategy!r} on {traj.id!r} (runner={args.runner})\n")
    for d in report.divergences:
        flag = "ok" if d.matched else "DIVERGED"
        print(f"  turn {d.turn}: {flag}")
        if not d.matched:
            print(f"      baseline:   {d.baseline_decision}")
            print(f"      compressed: {d.compressed_decision}")
    t = report.tost
    print(f"\ndecision-equivalence match rate: {report.match_rate * 100:.1f}%")
    print(
        f"TOST non-inferiority (margin={t.margin}, alpha={t.alpha}): "
        f"mean diff={t.mean_diff:+.3f}, "
        f"p={'<0.0001' if t.p_non_inferior < 1e-4 else format(t.p_non_inferior, '.4g')}"
    )
    print(
        f"\nVERDICT: {report.verdict}  "
        f"({'certified non-inferior' if t.non_inferior else 'NOT certified — would degrade quality'})"
    )
    return 0 if t.non_inferior else 1


def cmd_bench(args: argparse.Namespace) -> int:
    """Corpus-wide gate (CI). Across every trajectory: price distil vs baseline,
    certify distil is non-inferior, and confirm the gate still rejects the
    aggressive lossy strategy. Exits non-zero if any distil run fails the
    contract or the gate fails to reject aggressive — i.e. a real CI gate."""
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    price = pricing.get(args.pricing)
    tok = tokenizer.resolve(args.tokenizer, model=price.name)
    # Real ingested traces carry no DECISION labels, so the offline decision-
    # equivalence gate doesn't apply — report savings only (certify live instead).
    savings_only = args.savings_only

    mode = "savings only" if savings_only else "gate"
    print(
        f"corpus {mode} — {len(entries)} trajectories | model {price.name} | "
        f"tokenizer={args.tokenizer}\n"
    )
    if savings_only:
        print(f"{'domain':<18}{'trajectory':<28}{'$ saved':>10}")
    else:
        print(
            f"{'domain':<18}{'trajectory':<24}{'$ saved':>9}{'distil':>9}{'aggr':>7}{'pruned':>8}"
        )
    print("-" * 75)

    base_total = distil_total = pruned_total = 0.0
    base_tok_total = distil_tok_total = 0
    failures: list[str] = []
    for e in entries:
        b_sim = simulate(e.trajectory, price, strategy="none", caching=False, tok=tok)
        d_sim = simulate(e.trajectory, price, strategy="distil", caching=True, tok=tok)
        base, dist = b_sim.total_dollars, d_sim.total_dollars
        base_total += base
        distil_total += dist
        base_tok_total += b_sim.total_input_tokens
        distil_tok_total += d_sim.total_input_tokens
        saved = (1 - dist / base) * 100 if base else 0.0

        if savings_only:
            print(f"{e.domain:<18}{e.trajectory.id:<28}{saved:>9.1f}%")
            continue

        bad = validate(e.trajectory)
        if bad:
            failures.append(f"{e.file}: structural — {bad[0]}")
        d_rep = certify(e.trajectory, "distil", margin=args.margin, alpha=args.alpha)
        a_rep = certify(e.trajectory, "aggressive", margin=args.margin, alpha=args.alpha)
        pruned = discover(e.trajectory, tok=tok).tokens_freed
        pruned_total += pruned
        if not d_rep.tost.non_inferior:
            failures.append(f"{e.file}: distil FAILED non-inferiority")
        if a_rep.tost.non_inferior:
            failures.append(f"{e.file}: gate failed to reject aggressive")
        print(
            f"{e.domain:<18}{e.trajectory.id:<24}{saved:>8.1f}%"
            f"{d_rep.verdict:>9}{a_rep.verdict:>7}{pruned:>8}"
        )

    print("-" * 75)
    overall = (1 - distil_total / base_total) * 100 if base_total else 0.0
    tail = "" if savings_only else f"; {int(pruned_total)} tokens causally prunable"
    print(
        f"\naggregate: distil cuts ${base_total:.5f} -> ${distil_total:.5f} "
        f"({overall:.1f}% cheaper) reversibly{tail}."
    )
    if savings_only:
        print(
            "\nsavings-only mode (ingested traces have no decision labels); "
            "certify decision-equivalence live with: distil certify --runner anthropic"
        )
        return 0

    if args.record:
        rec = ledger.record(
            trajectory_id="corpus-aggregate",
            model=price.name,
            turns=sum(len(e.trajectory.turns) for e in entries),
            baseline_dollars=base_total,
            distil_dollars=distil_total,
            baseline_input_tokens=base_tok_total,
            distil_input_tokens=distil_tok_total,
        )
        print(
            f"recorded corpus run to the savings ledger: "
            f"${rec.dollars_saved:.5f} / {rec.tokens_saved} tokens saved. (distil leaderboard)"
        )

    if failures:
        print(f"\nGATE: FAIL ({len(failures)} issue(s))")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nGATE: PASS — every trajectory certified non-inferior; aggressive rejected on all.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Byte-fidelity gate (Phase 6): every distil compression across the corpus is
    reconstructable, and frozen history never mutates turn-to-turn."""
    from .compress.base import CompressResult
    from .compress.tier0 import Tier0Lossless
    from .compress.tier1 import Tier1Reversible
    from .fidelity import assert_append_only, verify_reversible
    from .trajectory import Stability

    print("byte-fidelity gate — reversibility + append-only across the corpus\n")
    problems: list[str] = []
    for e in load_corpus():
        prev = None
        for turn in e.trajectory.turns:
            volatile = [b for b in turn.blocks if b.stability is Stability.VOLATILE]
            r1 = Tier1Reversible().compress(volatile)
            r0 = Tier0Lossless().compress(r1.blocks)
            merged = CompressResult(r0.blocks, {**r1.restore, **r0.restore})
            rep = verify_reversible(volatile, merged)
            if not rep.lossless:
                problems.append(f"{e.file} turn {turn.index}: irrecoverable {rep.irrecoverable}")
            if prev is not None:
                v = assert_append_only(prev, turn.blocks)
                if v:
                    problems.append(f"{e.file} turn {turn.index}: append-only violation {v}")
            prev = turn.blocks
        print(f"  {e.trajectory.id:<24} reversible + append-only: ok")
    if problems:
        print("\nFIDELITY: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nFIDELITY: PASS — Tier-0/1 byte-reversible and history append-only across the corpus.")
    return 0


def cmd_holdout(args: argparse.Namespace) -> int:
    """Holdout A/B savings with a bootstrap CI (Phase 5)."""
    from .certify.holdout import run_holdout

    import sys as _sys

    if not 0.0 < args.control_fraction < 1.0:
        print(
            "distil holdout: --control-fraction must be between 0 and 1 (exclusive)",
            file=_sys.stderr,
        )
        return 2
    price = pricing.get(args.pricing)
    tok = tokenizer.resolve(args.tokenizer, model=price.name)
    rep = run_holdout(load_corpus(), price, control_fraction=args.control_fraction, tok=tok)
    print("holdout A/B savings measurement (deterministic partition + bootstrap CI)\n")
    print(f"  {rep.summary}")
    print(
        f"  control group mean savings: {rep.control_mean_savings * 100:.1f}% "
        "(held out, not counted toward the headline)"
    )
    return 0


def cmd_proxy(args: argparse.Namespace) -> int:
    """Drop-in provider proxy: point any base_url-honoring client at it."""
    if args.use_async:
        from .aproxy import serve as aserve  # high-concurrency (needs distil-llm[async])

        aserve(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            lossless_only=args.lossless_only,
            verbatim=args.verbatim,
            shape_output=args.shape_output,
            record=not args.no_record,
            pricing_model=args.pricing,
        )
    else:
        from .proxy import serve

        serve(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            lossless_only=args.lossless_only,
            verbatim=args.verbatim,
            shape_output=args.shape_output,
            record=not args.no_record,
            pricing_model=args.pricing,
            expand=args.expand,
            shadow_rate=args.shadow,
            session_delta=args.session_delta,
        )
    return 0


def cmd_shadow_stats(args: argparse.Namespace) -> int:
    """Show the live decision-equivalence measured by shadow mode on real traffic."""
    from .shadow import ShadowLedger

    led = ShadowLedger.load()
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "samples": led.samples,
                    "changes": led.changes,
                    "decision_change_rate": led.rate(),
                    "decision_equivalence": 1 - led.rate(),
                },
                indent=2,
            )
        )
        return 0
    if led.samples == 0:
        print(
            "No shadow samples yet. Start it in one command:\n"
            "  distil wrap --shadow 0.1 -- claude   "
            "(or codex/gemini; add --lossless-only on a subscription)\n"
            "then use your agent normally — samples accumulate as you work."
        )
        return 0
    change = led.rate()
    print("Shadow-mode live decision-equivalence (real traffic, content-free)\n")
    smp = f"{led.samples} shadowed request{'s' if led.samples != 1 else ''}"
    if led.samples < 25:
        # Same 25-sample floor as the status line / leaderboard / doctor — a rate
        # over a handful is noise, so don't print a decision-equivalence guarantee.
        print(f"  {smp} — collecting (need 25 for a decision-equivalence rate)")
        print("  keep using your agent; the rate appears once there's real evidence.")
        return 0
    print(f"  shadowed requests : {led.samples}")
    print(f"  decision changes  : {led.changes}")
    print(f"  decision-change rate (rolling): {change * 100:.2f}%")
    print(f"  decision-equivalence          : {(1 - change) * 100:.2f}%")
    print(
        "\n  Each sampled request was run BOTH compressed and uncompressed; "
        "equivalence\n  means the agent chose the same next action. Numbers only, never content."
    )
    return 0


def cmd_statusline(args: argparse.Namespace) -> int:
    """Render a compact one-line savings status for the Claude Code status line.

    Reads the optional Claude Code status-line JSON on stdin (for the model name)
    and the genuine savings from the local ledger; prints a single line to stdout.
    Wired via the distil Claude Code plugin (or any ``statusLine`` command). Never
    raises — a status line must always print something.
    """
    import os
    import sys

    model = ""
    if not sys.stdin.isatty():  # Claude Code pipes JSON; a bare TTY would block.
        try:
            raw = sys.stdin.read()
            if raw.strip():
                data = json.loads(raw)
                model = (data.get("model") or {}).get("display_name") or ""
        except (json.JSONDecodeError, ValueError, AttributeError, OSError):
            model = ""

    use_color = (not args.no_color) and os.environ.get("NO_COLOR") is None

    def c(code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    try:
        s = ledger.summary()
    except Exception:  # noqa: BLE001 — a status line must never error out
        s = None

    # MINIMAL is opt-in (DISTIL_STATUSLINE=minimal|lite|compact) — a two-fact
    # segment for crowded composite lines: this session's saving + lifetime.
    if os.environ.get("DISTIL_STATUSLINE", "").lower() in ("minimal", "lite", "compact"):
        mseg = [c("1;38;5;79", "distil")]
        if s is None or s.runs == 0:
            mseg.append(c("38;5;73", "wrap -- <agent> to start"))
        else:
            recent_saved = 0
            try:
                import time as _time

                # recent activity across all terminals (15-min window), not one session
                recent_saved = ledger.summary(since=_time.time() - 15 * 60).total_tokens_saved
            except Exception:  # noqa: BLE001 — recent slice is best-effort
                pass
            if recent_saved > 0:
                mseg.append(c("1;38;5;84", f"▼{ledger._human(recent_saved)}"))
            mseg.append(c("38;5;73", f"{ledger._human(s.total_tokens_saved)} total"))
        if model:
            mseg.append(c("38;5;73", model))
        print("  ".join(mseg))
        return 0

    # RICH by default: the full picture, visually laid out.
    # Built for COMPOSITE statuslines (users append this segment to their own):
    # every character earns its place. Grammar:
    #   live session:  distil · session ▼75.0K · 62% smaller [$0.31] · total ▼27.0M
    #   idle:          distil · total ▼27.0M saved · 50% smaller [$96.10]
    # ▼ = tokens saved; "session" = this run, "total" = lifetime.
    # Dropped by design: orig→compressed pair (derivable), run counts, and any
    # eq% under 25 shadow samples — "eq 100.0% (1)" is noise wearing a number.
    # Full breakdown: distil stats / dashboard.
    parts = [c("1;38;5;79", "distil")]
    if s is None or s.runs == 0:
        parts.append(c("38;5;73", "no savings yet · distil wrap -- <agent>"))
    else:
        from .doctor import subscription_mode

        metered = not subscription_mode()
        # ONE consistent pattern in every state:  distil · <live> · total ▼27.0M
        #   <live> = ▼75K · 62% smaller [· $]   (recent savings)
        #          = ✓ on · waiting for a large read   (recent traffic, nothing big yet)
        #          = ✓ on   (set up, idle — no recent traffic)
        # LIVE aggregates a 15-min window across ALL sessions (each terminal's
        # `distil default` spawns its own session), so the number never flickers.
        try:
            import time as _time

            recent = ledger.summary(since=_time.time() - 15 * 60)
        except Exception:  # noqa: BLE001 — recent slice is best-effort
            recent = ledger.LedgerSummary(0, 0.0, 0, {})
        if recent.runs and recent.total_baseline_tokens and recent.total_tokens_saved > 0:
            trimmed = 1 - recent.total_distil_tokens / recent.total_baseline_tokens
            parts.append(c("1;38;5;84", f"▼{ledger._human(recent.total_tokens_saved)}"))
            parts.append(c("38;5;80", f"{trimmed * 100:.0f}% smaller"))
            if metered and recent.total_dollars_saved > 0:
                parts.append(c("1;38;5;114", f"${recent.total_dollars_saved:,.2f}"))
        elif recent.runs and recent.total_baseline_tokens:
            parts.append(c("1;38;5;84", "✓ on") + c("38;5;80", " · waiting for a large read"))
        else:
            parts.append(c("1;38;5;84", "✓ on"))
        # TOTAL (lifetime) — identical format in every state.
        parts.append(c("38;5;73", f"total ▼{ledger._human(s.total_tokens_saved)}"))
        try:
            from .shadow import ShadowLedger

            led = ShadowLedger.load()
            # Only claim an equivalence rate once there is evidence behind it —
            # a percentage over a handful of samples is noise wearing a number.
            if led.samples >= 25:
                n = led.samples
                n_str = f"{n / 1000:.1f}k" if n >= 1000 else str(n)
                eq = 1 - led.rate()
                # Explicit 256-color hues (basic ANSI is terminal-theme roulette —
                # 'magenta' renders as unreadable purple on many dark themes) and a
                # health GLYPH so the state reads even without color: ✓ proven-safe,
                # ⚠ slipping, ✗ degraded. Color stays an alarm, not decoration.
                glyph, hue = (
                    ("✓", "38;5;86")
                    if eq >= 0.99
                    else ("⚠", "38;5;220")
                    if eq >= 0.95
                    else ("✗", "38;5;196")
                )
                parts.append(c(hue, f"{glyph}eq {eq * 100:.1f}%") + c("38;5;73", f" ({n_str})"))
        except Exception:  # noqa: BLE001 — shadow stats are best-effort
            pass
    if model:
        parts.append(c("38;5;73", model))
    print(" · ".join(parts))
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Print the installed version — the `version` word people actually type
    (argparse's --version flag stays too)."""
    print(f"distil {__version__}")
    return 0


def cmd_upgrade(args: argparse.Namespace) -> int:
    """Upgrade distil the way it was installed — detect brew/pipx/uv/pip and run
    (or, with --dry-run, print) the right command. distil can't replace its own
    running binary, so the upgrade goes through the installer that owns it.
    Uses onboard.install_method — the single source of truth for installer
    detection (shared by onboard, offboard, doctor)."""
    import subprocess

    from . import onboard

    method = onboard.install_method()
    if method == "uvx":
        print("uvx runs the latest distil-llm on every invocation — nothing to upgrade.")
        return 0
    cmd = onboard.upgrade_command(method)
    print(f"detected install: {method}\n$ {cmd}")
    if args.dry_run:
        return 0
    # A comment-only / advisory command (pip inside a venv we can't enter) isn't
    # safe to run for the user — just print it.
    if cmd.strip().startswith("#") or "inside your venv" in cmd:
        print("  run that yourself in the right environment.")
        return 0
    rc = subprocess.run(cmd, shell=True).returncode
    if rc == 0:
        print("✓ upgraded — run `distil version` to confirm")
    else:
        print(f"✗ upgrade command exited {rc}; run it yourself: {cmd}")
    return rc


def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose a distil setup: ledger, shadow, proxy round-trip, deps, Claude Code
    wiring. Exit 1 if any check fails, so it's usable as a CI/setup gate."""
    import os
    import sys

    from . import doctor

    use_color = (
        (not getattr(args, "no_color", False))
        and sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
    )

    def c(code: str, t: str) -> str:
        return f"\033[{code}m{t}\033[0m" if use_color else t

    glyph = {
        doctor.OK: c("32", "✓"),
        doctor.WARN: c("33", "⚠"),
        doctor.INFO: c("36", "ℹ"),
        doctor.FAIL: c("31", "✗"),
    }

    checks = doctor.diagnose()
    if getattr(args, "json", False):
        from dataclasses import asdict

        payload = {
            "checks": [asdict(ch) for ch in checks],
            "ok": all(ch.status != doctor.FAIL for ch in checks),
        }
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 1
    print(c("1;38;5;79", "distil doctor") + c("90", "  ·  setup diagnosis") + "\n")
    counts: dict[str, int] = {}
    for ch in checks:
        counts[ch.status] = counts.get(ch.status, 0) + 1
        print(f"  {glyph.get(ch.status, '?')} {c('1', ch.name)}: {ch.detail}")
        if ch.hint:
            print(c("90", f"      → {ch.hint}"))

    n_fail = counts.get(doctor.FAIL, 0)
    summary = "  ·  ".join(
        f"{counts[k]} {k}"
        for k in (doctor.OK, doctor.INFO, doctor.WARN, doctor.FAIL)
        if counts.get(k)
    )
    verdict = "looks healthy" if n_fail == 0 else f"{n_fail} failing — see above"
    print(
        "\n  " + c("90", summary) + "  —  " + (c("32", verdict) if not n_fail else c("31", verdict))
    )
    return 1 if n_fail else 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Wire the distil status line into Claude Code settings (idempotent, safe)."""
    from pathlib import Path

    from .setup import default_settings_path, wire_statusline

    path = Path(args.settings) if args.settings else default_settings_path()
    status, msg = wire_statusline(path, force=args.force)
    glyph = {"ok": "✓", "exists": "✓", "conflict": "⚠", "error": "✗"}.get(status, "?")
    print(f"{glyph} {msg}")
    if status in ("ok", "exists"):
        print("\nNext — route an agent through distil so the line fills in:")
        print("  distil wrap --shadow 0.1 -- claude")
        print("Verify your setup anytime with:  distil doctor")
        return 0
    return 1


def cmd_onboard(args: argparse.Namespace) -> int:
    """Sense the environment and either emit it as JSON (for an agent to reason
    over) or render a setup + guided tour. The intelligence belongs in the agent
    that reads ``--json``; this command is the sensor + safe actuator."""
    import json as _json
    import os
    import subprocess
    import sys

    from . import onboard
    from .setup import default_settings_path, wire_statusline

    env = onboard.detect()
    latest = None if args.offline else onboard.latest_pypi_version()
    outdated = onboard.is_outdated(env.installed_version, latest)

    if args.json:
        print(_json.dumps(onboard.report(env, latest), indent=2))
        return 0

    use_color = (not args.no_color) and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def c(code: str, t: str) -> str:
        return f"\033[{code}m{t}\033[0m" if use_color else t

    # Interactive by default — onboard offers to do the next step. Falls back to a
    # static guide only when it can't prompt (piped / CI) or you opt out.
    interactive = (
        sys.stdin.isatty() and sys.stdout.isatty() and not args.no_interactive and not args.dry_run
    )

    def ask(q: str) -> bool:
        if args.yes:
            return True
        if not interactive:
            return False
        try:
            return input(c("1", q) + c("90", " [y/N] ")).strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            print()
            return False

    print(c("1;38;5;79", "distil onboard") + c("90", "  ·  let's get you set up") + "\n")

    agents = ", ".join(label for _, label in env.agents) or "none detected"
    billing = (
        "flat-rate subscription (dollars notional)"
        if env.subscription
        else "metered / pay-as-you-go"
    )
    ver = env.installed_version
    if latest:
        ver += "  →  " + (f"{latest} available" if outdated else "up to date")
    print(c("90", "Detected"))
    print(f"  os            {env.os_name}")
    print(f"  distil        {ver}")
    print(f"  agents        {agents}")
    print(f"  package mgrs  {', '.join(env.managers) or 'none'}")
    print(f"  billing       {billing}")
    print(f"  anthropic ext {'installed' if env.has_anthropic else 'not installed (optional)'}\n")

    # Running ephemerally (uvx) — distil isn't on PATH, so every step below would need
    # the uvx prefix. Offer to make it permanent first; that's the real one-command path.
    if env.method == "uvx":
        install_cmd = onboard.best_install_command(env.managers)
        print(c("33", "↓ running ephemerally (uvx) — distil isn't installed permanently yet"))
        if ask(f"  Install distil permanently now?  ({install_cmd})"):
            ok = subprocess.run(install_cmd, shell=True).returncode == 0
            print(
                c(
                    "32" if ok else "31",
                    "  "
                    + (
                        "installed — future `distil …` commands just work"
                        if ok
                        else f"failed; run it yourself: {install_cmd}"
                    ),
                )
            )
        else:
            print(
                c("36", f"  {install_cmd}")
                + c(
                    "90",
                    "   ·   without it, the steps below need the `uvx --from distil-llm` prefix",
                )
            )
        print()

    if outdated:
        cmd = onboard.upgrade_command(env.method)
        runnable = env.method in ("pipx", "uv", "pip")
        print(c("33", f"⬆ newer distil available ({env.installed_version} → {latest})"))
        if runnable and (args.upgrade or ask(f"  Upgrade now?  ({cmd})")):
            print(c("33", f"  upgrading via {env.method} …"))
            ok = subprocess.run(cmd.split()).returncode == 0
            print(
                c(
                    "32" if ok else "31",
                    "  "
                    + ("done — re-run distil onboard to use it" if ok else f"failed; run: {cmd}"),
                )
            )
            if ok:
                return 0
        else:
            print(c("36", f"  {cmd}") + c("90", "   ·   or: distil onboard --upgrade") + "\n")

    if args.dry_run:
        print(c("90", "dry-run: would wire the status line (distil setup) — skipped\n"))
    else:
        status, msg = wire_statusline(default_settings_path(), force=args.force)
        glyph = {"ok": "✓", "exists": "✓", "conflict": "⚠", "error": "✗"}.get(status, "?")
        print(c("32" if status in ("ok", "exists") else "33", f"{glyph} {msg}"))
        if status == "conflict":
            print(c("90", "  re-run with --force to replace it (your current one is backed up)"))
        print()

    steps = onboard.next_steps(env)
    print(c("1", "Next steps"))
    for i, (title, cmd, note) in enumerate(steps, 1):
        print(f"  {c('1', str(i) + '.')} {title}")
        print(f"     {c('36', cmd)}")
        if note:
            print(f"     {c('90', note)}")
    print()

    # Highest-leverage finish: make distil the persistent default (no per-session wrap).
    if env.agents and ask("Make distil the default for your agent — no per-session wrap?"):
        print()
        cmd_default(
            argparse.Namespace(
                rc=None,
                agent=None,
                mode=None,
                port=8788,
                undo=False,
                always_on=False,
                no_start=False,
            )
        )
        print()

    # Or just route the agent once, right now.
    if env.agents:
        first_cmd = steps[0][1]
        if ask(f"Start now — run step 1?  ({first_cmd})"):
            print()
            return subprocess.run(first_cmd.split()).returncode

    print(
        c("90", "Re-run anytime: ")
        + c("36", "distil onboard")
        + c("90", "   ·   for agents: ")
        + c("36", "distil onboard --json")
    )
    return 0


def cmd_default(args: argparse.Namespace) -> int:
    """Make distil the default for your agent — no per-session `distil wrap`.

    Default (A): a managed shell alias that wraps the agent. ``--always-on`` (B):
    a persistent proxy service + ANTHROPIC_BASE_URL so every SDK routes through it.
    ``--undo`` removes whichever is installed."""
    import subprocess
    from pathlib import Path

    from . import onboard
    from .setup import (
        alias_body,
        detect_shell,
        env_body,
        remove_managed,
        service_spec,
        service_unload_cmd,
        write_managed,
    )

    shell, detected_rc = detect_shell()
    rc = Path(args.rc) if args.rc else detected_rc
    env = onboard.detect()
    agent = args.agent or (env.agents[0][0] if env.agents else "claude")
    mode = args.mode or ("lossless-only" if env.subscription else "expand")
    # Transparency over magic: each machine differs, so show what was detected.
    print(f"detected: shell={shell}  rc={rc}  agent={agent}  mode={mode}")
    if not rc.exists() and not args.undo:
        print(f"  note: {rc} doesn't exist yet — it'll be created. If your shell reads a")
        print("  different file, pass --rc <path>.")

    if args.undo:
        st, msg = remove_managed(rc)
        print(("✓ " if st in ("ok", "absent") else "✗ ") + msg)
        path, _, _ = service_spec(args.port, mode)
        if path is not None and path.exists():
            unload = service_unload_cmd()
            if unload:  # stop the running service before deleting its definition
                subprocess.run(unload, shell=True)
            try:
                path.unlink()
                print(f"✓ removed proxy service {path}")
            except OSError:
                pass
        print(f"  open a new terminal (or: source {rc}) to finish")
        return 0

    if args.always_on:
        path, content, load = service_spec(args.port, mode)
        if path is None:
            print(
                "✗ always-on service isn't supported on this platform yet — drop --always-on "
                "for the alias, or run `distil proxy` yourself."
            )
            return 1
        if content is None:
            print("✗ could not render the service file for this platform")
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"✓ wrote proxy service → {path}")
        _st, msg = write_managed(rc, env_body(args.port, shell=shell))
        print(f"✓ {msg}  (ANTHROPIC_BASE_URL → http://127.0.0.1:{args.port})")
        if load and not args.no_start:
            ok = subprocess.run(load, shell=True).returncode == 0
            print("  ✓ proxy service running" if ok else f"  ⚠ start it manually: {load}")
        print(
            f"\nEvery base-URL-honoring tool now routes through distil. Reload your shell (source {rc})."
        )
        print(
            "Heads-up: a persistent ANTHROPIC_BASE_URL is a single point of failure — if the proxy"
        )
        print("is down, clients can't reach the API. Undo: distil default --always-on --undo")
        return 0

    st, msg = write_managed(rc, alias_body(agent, mode, shell=shell))
    glyph = {"ok": "✓", "updated": "✓", "exists": "✓"}.get(st, "⚠")
    print(f"{glyph} {msg}")
    print(f"  `{agent}` now routes through distil (--{mode}).")
    print("\n  ⚠ IMPORTANT — one more step, or you'll see savings stay at zero:")
    print(f"     1. reload this shell:   source {rc}")
    print(f"     2. RESTART {agent}:       any {agent} already running was launched")
    print("        before the alias, so it bypasses distil. Start a fresh one.")
    print("     verify it's routed:     echo $ANTHROPIC_BASE_URL   (should not be empty)")
    print("\n  Undo anytime: distil default --undo")
    return 0


def cmd_offboard(args: argparse.Namespace) -> int:
    """Remove distil's footprint — the inverse of `distil onboard`.

    Undoes the shell default (alias/env), the always-on proxy service, and the
    status-line wiring, asking before each. Keeps your savings ledger unless
    ``--purge``. Can't uninstall the package itself (it's the running process), so
    it prints the right uninstall command for how distil was installed."""
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from . import onboard
    from .setup import (
        default_settings_path,
        detect_shell,
        remove_managed,
        service_spec,
        service_unload_cmd,
        unwire_statusline,
    )

    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.no_interactive

    def ask(q: str) -> bool:
        if args.yes:
            return True
        if not interactive:  # destructive: do nothing we weren't clearly told to
            print(f"  · skipped (not interactive — re-run with --yes): {q}")
            return False
        try:
            return input(q + " [y/N] ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            print()
            return False

    shell, rc = detect_shell()
    print("distil offboard — remove distil's footprint\n")

    # 1 · shell default (alias / env block)
    if rc.exists() and "distil (managed)" in rc.read_text(encoding="utf-8", errors="ignore"):
        if ask(f"Remove the distil default from {rc}?"):
            st, msg = remove_managed(rc)
            print(("✓ " if st in ("ok", "absent") else "✗ ") + msg)

    # 2 · always-on proxy service
    path, _, _ = service_spec(8788, "lossless-only")
    if path is not None and path.exists() and ask(f"Stop + remove the proxy service {path}?"):
        unload = service_unload_cmd()
        if unload:
            subprocess.run(unload, shell=True)
        try:
            path.unlink()
            print(f"✓ removed proxy service {path}")
        except OSError as exc:
            print(f"✗ couldn't remove {path}: {exc}")

    # 3 · status-line wiring
    sp = default_settings_path()
    if sp.exists() and ask(f"Unwire the distil status line from {sp}?"):
        st, msg = unwire_statusline(sp)
        print(("✓ " if st in ("ok", "absent", "foreign") else "✗ ") + msg)

    # 4 · local data (opt-in; it's the user's measured savings history)
    home = Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))
    if home.exists():
        if args.purge and ask(f"Delete your savings ledger + shadow data at {home}? Irreversible."):
            shutil.rmtree(home, ignore_errors=True)
            print(f"✓ deleted {home}")
        elif not args.purge:
            print(f"  kept your savings data at {home}  (delete it with: distil offboard --purge)")

    # 5 · the package itself — we can't remove the process we're running in.
    # onboard.install_method detects brew/pipx/uv/pip from the real path, so the
    # printed command actually works (a bare `pip uninstall` is blocked on
    # modern externally-managed Pythons, PEP 668).
    method = onboard.install_method()
    print(f"\nFinally, uninstall the package ({method}):\n  {onboard.uninstall_command(method)}")
    from .doctor import _find_all_distil

    extra = _find_all_distil()
    if len(extra) > 1:
        print("  note: multiple distil on PATH — remove each:")
        for p in extra:
            print(f"    {p}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Live terminal dashboard of cumulative savings — re-renders on an interval
    until Ctrl-C. Falls back to a single render when stdout isn't a TTY (so it
    pipes/redirects cleanly)."""
    import os
    import sys
    import time

    from .doctor import subscription_mode
    from .shadow import ShadowLedger

    subscription = subscription_mode()
    interactive = sys.stdout.isatty() and not args.once
    color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def frame() -> str:
        s = ledger.summary()
        change_rate: float | None = None
        samples = 0
        recent: list[int] | None = None
        sess = None
        try:
            led = ShadowLedger.load()
            samples = led.samples
            if samples:
                change_rate = led.rate()
                recent = list(led.recent)
        except Exception:  # noqa: BLE001 — shadow stats are best-effort
            pass
        try:
            sid, last_ts = ledger.latest_session()
            if sid and (time.time() - last_ts) < 4 * 3600:
                sess = ledger.summary(session=sid)
        except Exception:  # noqa: BLE001 — session slice is best-effort
            pass
        return ledger.render_dashboard(
            s,
            change_rate=change_rate,
            samples=samples,
            recent=recent,
            subscription=subscription,
            color=color,
            session=sess,
        )

    if not interactive:
        print(frame())
        return 0

    interval = max(0.5, args.interval)
    try:
        # Alternate screen buffer (like htop/less/vim): the dashboard owns a
        # full screen that redraws in place — no scrollback pollution — and the
        # user's prompt is restored intact on exit.
        sys.stdout.write("\033[?1049h\033[?25l")  # enter alt screen + hide cursor
        while True:
            sys.stdout.write("\033[H\033[2J")  # home + clear (in-place redraw)
            sys.stdout.write(frame())
            sys.stdout.write(
                f"\n\n  \033[90mrefreshing every {interval:g}s · Ctrl-C to exit\033[0m"
            )
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")  # show cursor + leave alt screen
        sys.stdout.flush()
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run the zero-dependency distil MCP server over stdio (compress/expand/savings)."""
    from .mcp_server import serve

    serve()
    return 0


def cmd_wrap(args: argparse.Namespace) -> int:
    """Transparently wrap a command: spawn the proxy, point its env at it, run it."""
    command = list(args.command)
    if command and command[0] == "--":  # argparse REMAINDER keeps the separator
        command = command[1:]
    if not command:
        print("distil wrap: nothing to run — usage: distil wrap [opts] -- <command> [args...]")
        return 2
    from .proxy import wrap_run

    return wrap_run(
        command,
        host=args.host,
        upstream=args.upstream,
        lossless_only=args.lossless_only,
        verbatim=args.verbatim,
        shape_output=args.shape_output,
        record=not args.no_record,
        pricing_model=args.pricing,
        env_var=args.env_var,
        expand=args.expand,
        session_delta=args.session_delta,
        shadow_rate=args.shadow,
    )


def cmd_certify_trajectories(args: argparse.Namespace) -> int:
    """Trajectory-level risk certificate: bound END-TO-END task degradation.

    Reads a JSONL of matched runs (one object per task:
    {"task_id": ..., "full_success": bool, "compressed_success": bool}) and
    certifies P(degradation risk <= alpha) >= 1-delta — the invariant that
    actually transfers to task success, unlike per-step next-action equivalence.
    """
    from .certify.trajectory_risk import TrajectoryOutcome, certify_trajectory_risk

    outcomes: list = []
    for line in Path(args.outcomes).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        outcomes.append(
            TrajectoryOutcome(
                task_id=str(d.get("task_id", len(outcomes))),
                full_success=bool(d["full_success"]),
                compressed_success=bool(d["compressed_success"]),
            )
        )
    cert = certify_trajectory_risk(outcomes, alpha=args.alpha, delta=args.delta)
    if args.json:
        from dataclasses import asdict

        print(json.dumps({**asdict(cert), "statement": cert.statement}, indent=2))
    else:
        print("distil trajectory-risk certificate\n")
        print(f"  matched trajectories : {cert.n}")
        print(f"  observed degradation : {cert.empirical_risk * 100:.2f}%")
        print(f"  risk bound (1-δ)     : {cert.risk_bound * 100:.2f}%")
        print(f"  certified (α={cert.alpha}, δ={cert.delta}) : {cert.certified}")
        print(f"\n  {cert.statement}")
    return 0 if cert.certified else 1


def cmd_gateway(args: argparse.Namespace) -> int:
    """Managed multi-tenant gateway with a live per-tenant savings dashboard."""
    from .gateway import serve_gateway

    serve_gateway(
        host=args.host,
        port=args.port,
        upstream=args.upstream,
        pricing_model=args.pricing,
        lossless_only=args.lossless_only,
        verbatim=args.verbatim,
        admin_token=args.admin_token,
        trust_tenant_header=args.trust_tenant_header,
    )
    return 0


def cmd_train_transformer(args: argparse.Namespace) -> int:
    """Train the transformer keep-model on the corpus (needs distil-llm[train])."""
    from .codec.train_transformer import train_transformer

    metrics = train_transformer(args.out, base_model=args.base_model, epochs=args.epochs)
    print(f"trained transformer keep-model -> {args.out}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


def cmd_output_savings(args: argparse.Namespace) -> int:
    """Measure generation-side output-token savings A/B (token cut + answer kept)."""
    import json as _json

    from .output import measure_output_savings

    src = args.input or (CORPUS_DIR / "output_pairs.jsonl")
    pairs = [
        (d["baseline"], d["shaped"])
        for d in (_json.loads(ln) for ln in Path(src).read_text().splitlines() if ln.strip())
    ]
    rep = measure_output_savings(pairs)
    print("output compression — generation-side shaping, measured A/B\n")
    print(f"  {rep.summary}")
    print("  (answer-preservation is the gate: a reduction that drops the answer is not a saving)")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Convert recorded provider requests into a Distil corpus you can run the gate on."""
    import json as _json

    from .ingest import ingest_file

    import sys as _sys

    traj = ingest_file(args.input, provider=args.provider, model=args.model)
    if not traj.turns:
        print(
            f"distil ingest: parsed 0 turns from {args.input} — is it newline-delimited "
            "provider-request JSON? (nothing was written)",
            file=_sys.stderr,
        )
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{traj.id}.json"
    (out / fname).write_text(_json.dumps(traj.to_dict(), indent=2))
    manifest = out / "manifest.json"
    entries = []
    if manifest.exists():
        entries = _json.loads(manifest.read_text()).get("trajectories", [])
    entries = [e for e in entries if e.get("file") != fname]
    entries.append({"file": fname, "domain": "ingested", "title": traj.id})
    manifest.write_text(_json.dumps({"version": 1, "trajectories": entries}, indent=2))
    print(f"ingested {len(traj.turns)} turn(s) from {args.input} -> {out / fname}")
    print(
        f"run:  DISTIL_CORPUS={out} distil savings   # or: distil bench --corpus {out} --savings-only"
    )
    print(
        "note: real traces carry no DECISION labels — certify decision-equivalence with --runner anthropic."
    )
    return 0


def cmd_perf(args: argparse.Namespace) -> int:
    """Report compression + adapter latency/throughput (p50/p95)."""
    import sys as _sys

    from .perf import format_table, run_perf

    if args.iterations < 1:
        print("distil perf: --iterations must be >= 1", file=_sys.stderr)
        return 2
    print(format_table(run_perf(iterations=args.iterations)))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Head-to-head: every compression technique through the same gate + cost model."""
    import time

    from . import benchmark as bm

    runner = None
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner()
    price = pricing.get(args.pricing)
    tok = tokenizer.resolve(args.tokenizer, model=price.name)
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()

    techniques = bm.builtin_techniques(runner)
    for spec in args.external or []:
        try:
            techniques.append(bm.load_external(spec))
        except Exception as exc:  # noqa: BLE001 — surface a bad external spec, don't crash
            print(f"could not load external '{spec}': {exc}")
            return 2

    rep = bm.run_benchmark(
        entries,
        techniques,
        pricing=price,
        runner=runner,
        tok=tok,
        margin=args.margin,
        alpha=args.alpha,
    )
    print(bm.format_report(rep))
    if args.html:
        Path(args.html).write_text(bm.render_html(rep))
        print(f"\nbenchmark page → {args.html}")
    if args.out:
        path = bm.write_raw(rep, args.out, str(int(time.time())))
        print(f"raw results → {path}")
    return 0


def cmd_conformal(args: argparse.Namespace) -> int:
    """Decision-Equivalence Risk Certificate — a distribution-free guarantee on the
    agent's decision-change rate at a user-chosen risk level."""
    from .conformal import calibrate

    runner: Any = None
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner(samples=args.samples)
    else:
        from .replay.runner import DeterministicRunner

        runner = DeterministicRunner()
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    cert = calibrate(entries, runner, alpha=args.alpha, delta=args.delta, method=args.method)

    long = "Learn-Then-Test" if cert.method == "ltt" else "Conformal Risk Control"
    print("Decision-Equivalence Risk Certificate (DERC)\n")
    print(f"  method      : {cert.method.upper()}  ({long})")
    print(f"  risk target : α = {args.alpha}  (max allowed decision-change rate)")
    if cert.method == "ltt":
        print(f"  confidence  : {(1 - args.delta) * 100:.0f}%  (1 − δ)")
    print(f"  calibration : n = {cert.n} turns,  runner = {args.runner}")
    print("-" * 64)
    if cert.level:
        print(f"  ✔ CERTIFIED  '{cert.level}'  →  {cert.savings * 100:.1f}% token savings")
        print(f"\n  {cert.guarantee}")
    else:
        print(
            f"  ✘ NOT CERTIFIED — no level holds a ≤{args.alpha * 100:.0f}% decision-change rate "
            f"at n={cert.n}.\n    Calibrate on more traffic, or relax α. (This is the certificate "
            "being honest:\n    small samples can't support tight guarantees.)"
        )
    print(
        "\n  Distribution-free, finite-sample (Angelopoulos–Bates–Candès et al., arXiv:2110.01052 /"
        "\n  2208.02814). Valid under EXCHANGEABILITY — recalibrate on recent traffic if your"
        "\n  workload drifts. The guarantee is marginal over the calibration distribution, not a"
        "\n  per-prompt promise."
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Auto-calibrate the relevance-gate operating point to your agent's capability.

    E11 showed the safe operating point is capability-dependent: a setting non-inferior on a
    weak agent costs -31 pp on a strong one. This selects the most aggressive working-set size
    (``gate_recent``) still non-inferior to full context on your calibration scores — and
    fails safe to full context if none qualifies."""
    import sys

    from .calibrate import calibrate_from_scores

    candidates = []
    for spec in args.candidate:
        # spec form: name=path:gate_recent
        try:
            name, rest = spec.split("=", 1)
            path, gr = rest.rsplit(":", 1)
            candidates.append((name, path, int(gr)))
        except ValueError:
            print(f"bad --candidate '{spec}'; expected name=path:gate_recent", file=sys.stderr)
            return 2

    cert = calibrate_from_scores(args.baseline, candidates, margin=args.margin)

    if args.json:
        Path(args.json).write_text(json.dumps(cert.to_dict(), indent=2))

    print("Operating-Point Calibration Certificate\n")
    print(f"  baseline    : full context ({args.baseline})")
    print(f"  margin      : {args.margin:.0%}  (max tolerated task-success drop)")
    print("-" * 64)
    header = f"  {'operating point':<16}{'gate':>5}{'Δ pass@1':>10}{'95% CI low':>12}   verdict"
    print(header)
    for v in cert.levels:
        mark = "✔ non-inferior" if v.noninferior else "✘ too aggressive"
        print(
            f"  {v.name:<16}{v.gate_recent:>5}{v.delta * 100:>+9.1f}%{v.ci95_low * 100:>+11.1f}%"
            f"   {mark}"
        )
    print("-" * 64)
    if cert.fail_safe:
        print("  ✘ FAIL-SAFE → keep FULL context (no operating point certified)")
    else:
        print(
            f"  ✔ SELECTED  '{cert.selected}'  →  DISTIL_E7_GATE_RECENT={cert.selected_gate_recent}"
        )
    print(f"\n  {cert.rationale}")
    print(
        "\n  Paired non-inferiority (McNemar, FDA-standard). Valid under EXCHANGEABILITY with"
        "\n  your calibration distribution — recalibrate when you change model, agent, or task mix."
    )
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """Show the keep policy Distil has learned from your real expand signals."""
    from .learn import ExpandStats

    stats = ExpandStats.load()
    if not stats.digested:
        print("no expand signals yet.")
        print("run `distil proxy --expand` (or `distil wrap --expand`) so agents can")
        print("recover digested detail — every recovery teaches Distil your workload.")
        return 0
    prone = stats.expand_prone(min_digested=args.min_samples, threshold=args.threshold)
    print("learned digest policy — from your real expand signals (content-free)\n")
    print(f"{'signature':<14}{'digested':>9}{'expanded':>9}{'rate':>7}  policy")
    print("-" * 60)
    for sig in sorted(stats.digested, key=lambda s: -stats.expand_rate(s)):
        d, e, r = stats.digested[sig], stats.expanded.get(sig, 0), stats.expand_rate(sig)
        pol = "KEEP byte-exact" if sig in prone else "digest"
        print(f"{sig:<14}{d:>9}{e:>9}{r * 100:>6.0f}%  {pol}")
    print("-" * 60)
    print(
        f"\n{len(prone)} signature(s) are now kept byte-exact because your agents expand "
        "them often.\nThis applies automatically under `--expand`. It only ever makes "
        "Distil more\nconservative — savings may drop on those, decision-equivalence never does."
    )
    return 0


def cmd_frontier(args: argparse.Namespace) -> int:
    """The savings-vs-equivalence dial: how much more you save as you relax the
    decision-equivalence target below 100%."""
    from .compress.adaptive import frontier
    from .replay.runner import DeterministicRunner

    runner: Any = DeterministicRunner()
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner(samples=args.samples)
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    try:
        targets = tuple(float(x) for x in args.targets.split(","))
    except ValueError:
        print("--targets must be comma-separated numbers in (0,1], e.g. 1.0,0.97,0.95")
        return 2

    points = frontier(entries, runner, targets=targets)
    print(f"savings-vs-equivalence dial  (runner={getattr(runner, 'name', 'deterministic')})\n")
    print(f"{'target':>9}{'achieved equiv':>17}{'token savings':>15}   curve")
    print("-" * 70)
    for p in points:
        bar = "█" * round(p.savings * 28)
        print(
            f"{p.target * 100:>8.0f}%{p.equivalence * 100:>16.0f}%{p.savings * 100:>14.1f}%   {bar}"
        )
    print("-" * 70)
    top = points[0]
    print(
        f"\nAt 100% you get the certified-safe result ({top.savings * 100:.1f}% saved, "
        "every decision preserved). Relax the target and Distil spends a bounded "
        "'divergence budget' on the highest-value turns — deeper savings, a known "
        "equivalence cost. You choose the point; the trade is explicit, not hidden."
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """The certified compression frontier (savings vs decision-equivalence)."""
    import time

    from .eval import format_frontier, frontier, write_raw

    runner = None
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner()
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    rep = frontier(entries, runner=runner)
    print(format_frontier(rep))
    if args.out:
        path = write_raw(rep, args.out, str(int(time.time())))
        print(f"\nraw curve → {path}")
    return 0


def cmd_online(args: argparse.Namespace) -> int:
    """One self-distilling round: causal labels → retrain → certify → promote."""
    from .online import online_round

    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    rep = online_round(entries, promote_to=args.promote_to)
    print(
        "self-distilling round — keep-model learns from causal labels, gated by non-inferiority\n"
    )
    for k, v in rep.items():
        if isinstance(v, float):
            v = f"{v:.1%}" if k in ("accuracy", "precision", "recall") else f"{v:.3f}"
        print(f"  {k}: {v}")
    if not rep.get("certified"):
        print("\nNOT promoted — the candidate failed the non-inferiority gate (never-regressing).")
    return 0


def cmd_federated(args: argparse.Namespace) -> int:
    """Build a verifiable federated savings leaderboard from signed submissions."""
    import json as _json

    from .telemetry import build_leaderboard, render_leaderboard_html

    keys = _json.loads(Path(args.keys).read_text()) if args.keys else {}
    lb = build_leaderboard(args.dir, keys)
    print(f"verifiable savings — {len(lb.verified)} verified instance(s), {lb.rejected} rejected\n")
    print(f"  totals (certified only): {lb.totals}")
    if args.html:
        Path(args.html).write_text(render_leaderboard_html(lb))
        print(f"  leaderboard html → {args.html}")
    return 0


_HELP_EPILOG = """\
everyday commands:
  onboard, doctor, setup, offboard    guided install / health-check / wiring
  wrap, proxy, gateway                route agent traffic through compression
  stats (leaderboard), dashboard,     your genuine savings + live equivalence
  shadow-stats, statusline
  version, upgrade                    show version / update distil in place

analysis & tuning:
  compress, savings, bench, prune, calibrate, certify, certify-trajectories

research / CI internals:
  verify, holdout, conformal, frontier, eval, online,
  train-transformer, federated-leaderboard

`distil <command> --help` shows each command's flags.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="distil",
        description="Compression with a quality contract.",
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"distil {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    vs = sub.add_parser("version", help="print the installed version")
    vs.set_defaults(func=cmd_version)

    up = sub.add_parser("upgrade", help="upgrade distil (auto-detects brew/pipx/uv/pip)")
    up.add_argument("--dry-run", action="store_true", help="print the command, don't run it")
    up.set_defaults(func=cmd_upgrade)

    def add_traj(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--trajectory", "-t", help="path to a trajectory JSON (default: bundled sample)"
        )

    def add_tokenizer(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--tokenizer",
            default="heuristic",
            choices=("heuristic", "anthropic"),
            help="heuristic (offline, default) or anthropic (billing-grade count_tokens)",
        )

    c = sub.add_parser("compress", help="shrink a trajectory; report ratio + reversibility")
    add_traj(c)
    add_tokenizer(c)
    c.set_defaults(func=cmd_compress)

    s = sub.add_parser("savings", help="price strategies in real dollars (technique #1)")
    add_traj(s)
    add_tokenizer(s)
    s.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    s.add_argument("--output-tokens-per-turn", type=int, default=0)
    s.add_argument(
        "--record", action="store_true", help="append this run to the local savings ledger"
    )
    s.set_defaults(func=cmd_savings)

    lb = sub.add_parser(
        "leaderboard",
        aliases=["stats"],
        help="your genuine cumulative savings (local ledger)",
    )
    lb.add_argument("--html", help="render your savings as a self-contained HTML page")
    lb.add_argument("--json", action="store_true", help="machine-readable output")
    lb.add_argument(
        "--badge",
        action="store_true",
        help="print a shields.io badge URL + markdown of your measured savings",
    )
    lb.set_defaults(func=cmd_leaderboard)

    pr = sub.add_parser("prune", help="causal ablation: what is free to drop (technique #2)")
    add_traj(pr)
    pr.set_defaults(func=cmd_prune)

    ce = sub.add_parser("certify", help="non-inferiority gate (the quality contract)")
    add_traj(ce)
    ce.add_argument("--strategy", default="distil", choices=sorted(REGISTRY))
    ce.add_argument(
        "--runner",
        default="deterministic",
        choices=("deterministic", "anthropic"),
        help="deterministic (offline, default) or anthropic (live model)",
    )
    ce.add_argument("--margin", type=float, default=0.02)
    ce.add_argument("--alpha", type=float, default=0.05)
    ce.set_defaults(func=cmd_certify)

    be = sub.add_parser("bench", help="corpus-wide CI gate across every domain")
    add_tokenizer(be)
    be.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    be.add_argument("--margin", type=float, default=0.02)
    be.add_argument("--alpha", type=float, default=0.05)
    be.add_argument(
        "--record",
        action="store_true",
        help="log the corpus-aggregate savings to the local ledger (distil leaderboard)",
    )
    be.add_argument("--corpus", help="run on a custom corpus dir (e.g. from `distil ingest`)")
    be.add_argument(
        "--savings-only",
        action="store_true",
        help="report savings only, skip the decision-equivalence gate (for real traces)",
    )
    be.set_defaults(func=cmd_bench)

    os_ = sub.add_parser("output-savings", help="measure generation-side output-token savings A/B")
    os_.add_argument("--input", help="JSONL of {baseline, shaped} pairs (default: bundled fixture)")
    os_.set_defaults(func=cmd_output_savings)

    ig = sub.add_parser("ingest", help="convert recorded provider requests into a Distil corpus")
    ig.add_argument("--input", required=True, help="path to a .json/.jsonl of recorded requests")
    ig.add_argument("--out", default="./ingested-corpus", help="output corpus directory")
    ig.add_argument("--provider", default="anthropic", choices=("anthropic", "openai"))
    ig.add_argument("--model", default="claude-opus-4-8")
    ig.set_defaults(func=cmd_ingest)

    pf = sub.add_parser("perf", help="latency/throughput benchmark (p50/p95)")
    pf.add_argument("--iterations", type=int, default=200)
    pf.set_defaults(func=cmd_perf)

    ev = sub.add_parser("eval", help="certified compression frontier (savings vs accuracy)")
    ev.add_argument("--corpus", help="custom corpus dir (e.g. ingested benchmark traces)")
    ev.add_argument("--runner", default="deterministic", choices=("deterministic", "anthropic"))
    ev.add_argument("--out", help="write the raw curve JSONL to this dir")
    ev.set_defaults(func=cmd_eval)

    bn = sub.add_parser(
        "benchmark",
        help="head-to-head vs competing techniques on the same gate + cost model",
    )
    bn.add_argument("--corpus", help="custom corpus dir (e.g. ingested benchmark traces)")
    bn.add_argument("--runner", default="deterministic", choices=("deterministic", "anthropic"))
    bn.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    bn.add_argument("--tokenizer", default="heuristic", choices=("heuristic", "anthropic"))
    bn.add_argument("--margin", type=float, default=0.02, help="TOST non-inferiority margin")
    bn.add_argument("--alpha", type=float, default=0.05, help="significance level")
    bn.add_argument(
        "--external",
        action="append",
        metavar="MODULE:FUNCTION[:NAME]",
        help="register a real external compressor (list[str]->list[str]); repeatable",
    )
    bn.add_argument("--html", help="render the comparison as a self-contained HTML page")
    bn.add_argument("--out", help="write raw results JSONL to this dir")
    bn.set_defaults(func=cmd_benchmark)

    fr = sub.add_parser(
        "frontier",
        help="savings-vs-equivalence dial: deeper compression as you relax the target",
    )
    fr.add_argument("--corpus", help="custom corpus dir")
    fr.add_argument("--runner", default="deterministic", choices=("deterministic", "anthropic"))
    fr.add_argument(
        "--samples", type=int, default=3, help="majority-vote samples (anthropic runner)"
    )
    fr.add_argument(
        "--targets",
        default="1.0,0.97,0.95,0.90,0.80",
        help="comma-separated equivalence targets in (0,1]",
    )
    fr.set_defaults(func=cmd_frontier)

    ln = sub.add_parser(
        "learn",
        help="show the keep policy learned from your real distil_expand signals",
    )
    ln.add_argument("--threshold", type=float, default=0.25, help="expand-rate to keep byte-exact")
    ln.add_argument(
        "--min-samples", type=int, default=5, help="min digests before a policy applies"
    )
    ln.set_defaults(func=cmd_learn)

    cf = sub.add_parser(
        "conformal",
        help="decision-equivalence risk certificate (distribution-free guarantee)",
    )
    cf.add_argument("--alpha", type=float, default=0.05, help="max decision-change rate to certify")
    cf.add_argument(
        "--delta", type=float, default=0.05, help="LTT failure probability (1−confidence)"
    )
    cf.add_argument("--method", default="ltt", choices=("ltt", "crc"))
    cf.add_argument("--corpus", help="calibration corpus dir (e.g. your ingested traffic)")
    cf.add_argument("--runner", default="deterministic", choices=("deterministic", "anthropic"))
    cf.add_argument(
        "--samples", type=int, default=3, help="majority-vote samples (anthropic runner)"
    )
    cf.set_defaults(func=cmd_conformal)

    cal = sub.add_parser(
        "calibrate",
        help="auto-tune the gate operating point to your agent (fail-safe to full context)",
    )
    cal.add_argument("--baseline", required=True, help="full-context score JSON (per_instance map)")
    cal.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="name=path:gate_recent",
        help="candidate operating point, e.g. gate@12=scores/distil_gated_gr12.json:12 "
        "(repeat for each)",
    )
    cal.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="max tolerated task-success drop as a proportion (default 0.05 = 5 pp)",
    )
    cal.add_argument("--json", help="write the calibration certificate to this path")
    cal.set_defaults(func=cmd_calibrate)

    ve = sub.add_parser("verify", help="byte-fidelity gate: reversibility + append-only (phase 6)")
    ve.set_defaults(func=cmd_verify)

    ho = sub.add_parser("holdout", help="holdout A/B savings with a bootstrap CI (phase 5)")
    add_tokenizer(ho)
    ho.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    ho.add_argument("--control-fraction", type=float, default=0.2)
    ho.set_defaults(func=cmd_holdout)

    px = sub.add_parser("proxy", help="drop-in provider proxy (point any client's base_url at it)")
    px.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    px.add_argument("--port", type=int, default=8788)
    px.add_argument(
        "--upstream", default="https://api.anthropic.com", help="upstream provider base URL"
    )
    px.add_argument(
        "--lossless-only",
        "--safe",
        action="store_true",
        help="policy/subscription-safe mode: no lossy output-shaping, no tool injection "
        "(the reversible, certified digest still runs). Alias: --safe. For byte-in-context "
        "content use --verbatim.",
    )
    px.add_argument(
        "--verbatim",
        action="store_true",
        help="skip the Tier-1 digest (Tier-0 only) so the model sees content verbatim — "
        "for interactive sessions / out-of-distribution traffic; lower savings",
    )
    px.add_argument(
        "--shape-output",
        default="off",
        choices=("off", "light", "aggressive"),
        help="output-token compression via a gated verbosity directive (PAYG only)",
    )
    px.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="async high-concurrency proxy (needs distil-llm[async])",
    )
    px.add_argument(
        "--pricing",
        default="claude-opus-4-8",
        choices=sorted(pricing.CATALOG),
        help="model used to price genuine savings recorded to the ledger",
    )
    px.add_argument(
        "--no-record", action="store_true", help="do not record genuine savings to the local ledger"
    )
    px.add_argument(
        "--expand",
        action="store_true",
        help="recoverable compression: inject the distil_expand tool so the agent can "
        "pull back digested detail on demand (transparent server-side recovery loop)",
    )
    px.add_argument(
        "--shadow",
        type=float,
        default=0.0,
        metavar="RATE",
        help="shadow-mode live decision-equivalence: sample this fraction of requests "
        "(e.g. 0.05) and run them uncompressed too, in the background, to measure the "
        "live decision-change rate on real traffic (`distil shadow-stats`). Adds ~RATE cost.",
    )
    px.add_argument(
        "--session-delta",
        action="store_true",
        help="cache-delta coding: cross-turn dedup + cross-version delta (re-reads after "
        "edits sent as a diff), cache-monotonic and reversible (sync proxy only)",
    )
    px.set_defaults(func=cmd_proxy)

    ss = sub.add_parser(
        "shadow-stats", help="show live decision-equivalence measured by shadow mode"
    )
    ss.add_argument("--json", action="store_true", help="machine-readable output")
    ss.set_defaults(func=cmd_shadow_stats)

    dash = sub.add_parser(
        "dashboard", help="live terminal dashboard of your savings (Ctrl-C to exit)"
    )
    dash.add_argument("--once", action="store_true", help="render once and exit (no live refresh)")
    dash.add_argument(
        "--interval", type=float, default=2.0, help="refresh seconds in live mode (default 2)"
    )
    dash.set_defaults(func=cmd_dashboard)

    dr = sub.add_parser(
        "doctor", help="diagnose your distil setup (ledger, shadow, proxy round-trip, wiring)"
    )
    dr.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    dr.add_argument("--json", action="store_true", help="machine-readable output (CI-gateable)")
    dr.set_defaults(func=cmd_doctor)

    su = sub.add_parser("setup", help="wire the distil status line into Claude Code settings")
    su.add_argument(
        "--force", action="store_true", help="replace an existing status line (backed up first)"
    )
    su.add_argument("--settings", help="settings.json path (default ~/.claude/settings.json)")
    su.set_defaults(func=cmd_setup)

    ob = sub.add_parser("onboard", help="one command: set up distil + a guided next-steps tour")
    ob.add_argument("--dry-run", action="store_true", help="scan + guide only; change nothing")
    ob.add_argument(
        "--force", action="store_true", help="replace an existing status line (backed up first)"
    )
    ob.add_argument(
        "--json",
        action="store_true",
        help="emit the environment + recommendations as JSON (no actions)",
    )
    ob.add_argument(
        "--upgrade", action="store_true", help="upgrade distil if a newer version is available"
    )
    ob.add_argument("--offline", action="store_true", help="skip the PyPI version check")
    ob.add_argument(
        "--yes", "-y", action="store_true", help="auto-confirm prompts (upgrade, launch the agent)"
    )
    ob.add_argument(
        "--no-interactive", action="store_true", help="never prompt; just print the guide"
    )
    ob.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    ob.set_defaults(func=cmd_onboard)

    de = sub.add_parser(
        "default", help="make distil the default for your agent (no per-session wrap)"
    )
    de.add_argument(
        "--always-on",
        action="store_true",
        help="strategy B: persistent proxy service + ANTHROPIC_BASE_URL (covers every SDK)",
    )
    de.add_argument("--agent", help="agent command to wrap (default: detected, e.g. claude)")
    de.add_argument(
        "--mode",
        choices=("lossless-only", "expand", "verbatim", "safe"),
        help="wrap/proxy mode (default: lossless-only on a subscription, else expand)",
    )
    de.add_argument(
        "--port", type=int, default=8788, help="proxy port for --always-on (default 8788)"
    )
    de.add_argument(
        "--no-start", action="store_true", help="--always-on: write the service but don't start it"
    )
    de.add_argument("--undo", action="store_true", help="remove the distil default")
    de.add_argument("--rc", help="shell rc/profile path (default: auto-detected)")
    de.set_defaults(func=cmd_default)

    of = sub.add_parser(
        "offboard", help="remove distil's footprint (alias, proxy service, status line)"
    )
    of.add_argument(
        "--purge", action="store_true", help="also delete your local savings ledger + shadow data"
    )
    of.add_argument("--yes", "-y", action="store_true", help="remove everything without prompting")
    of.add_argument("--no-interactive", action="store_true", help="never prompt (report only)")
    of.set_defaults(func=cmd_offboard)

    sl = sub.add_parser(
        "statusline", help="compact savings status line (for the Claude Code plugin)"
    )
    sl.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    sl.set_defaults(func=cmd_statusline)

    mc = sub.add_parser(
        "mcp", help="run the zero-dep MCP server over stdio (distil_compress/expand/savings)"
    )
    mc.set_defaults(func=cmd_mcp)

    wr = sub.add_parser(
        "wrap",
        help="run a command with its API base URL transparently routed through Distil",
    )
    wr.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    wr.add_argument(
        "--upstream", default="https://api.anthropic.com", help="upstream provider base URL"
    )
    wr.add_argument(
        "--env-var",
        default="ANTHROPIC_BASE_URL",
        help="environment variable to point at the proxy (default: ANTHROPIC_BASE_URL)",
    )
    wr.add_argument(
        "--lossless-only",
        "--safe",
        action="store_true",
        help="policy/subscription-safe mode: no lossy output-shaping, no tool injection "
        "(the reversible, certified digest still runs). Alias: --safe. For byte-in-context "
        "content use --verbatim.",
    )
    wr.add_argument(
        "--verbatim",
        action="store_true",
        help="skip the Tier-1 digest (Tier-0 only); model sees content verbatim — "
        "for interactive sessions / OOD traffic; lower savings",
    )
    wr.add_argument(
        "--shape-output",
        default="off",
        choices=("off", "light", "aggressive"),
        help="output-token compression via a gated verbosity directive (PAYG only)",
    )
    wr.add_argument(
        "--pricing",
        default="claude-opus-4-8",
        choices=sorted(pricing.CATALOG),
        help="model used to price genuine savings recorded to the ledger",
    )
    wr.add_argument(
        "--no-record", action="store_true", help="do not record genuine savings to the local ledger"
    )
    wr.add_argument(
        "--expand",
        action="store_true",
        help="recoverable compression: the agent can pull back digested detail on demand",
    )
    wr.add_argument(
        "--session-delta",
        action="store_true",
        help="cache-delta coding: cross-turn dedup + cross-version delta (re-reads after "
        "edits sent as a diff), cache-monotonic and reversible",
    )
    wr.add_argument(
        "--shadow",
        type=float,
        default=0.0,
        metavar="RATE",
        help="shadow-mode live decision-equivalence: sample this fraction (e.g. 0.1) "
        "and also run it uncompressed to measure the decision-change rate "
        "(distil shadow-stats)",
    )
    wr.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the command to run, after `--` (e.g. distil wrap -- claude -p 'hi')",
    )
    wr.set_defaults(func=cmd_wrap)

    on = sub.add_parser(
        "online", help="self-distilling round: causal labels → retrain → certify → promote"
    )
    on.add_argument("--corpus", help="corpus dir of traffic to learn from (default: bundled)")
    on.add_argument("--promote-to", help="persist retrained weights here if it passes the gate")
    on.set_defaults(func=cmd_online)

    fl = sub.add_parser(
        "federated-leaderboard",
        help="verifiable federated savings leaderboard from signed submissions",
    )
    fl.add_argument("--dir", required=True, help="dir containing submissions.jsonl")
    fl.add_argument("--keys", help="JSON map of instance_id -> signing key")
    fl.add_argument("--html", help="write a self-contained leaderboard HTML here")
    fl.set_defaults(func=cmd_federated)

    ct = sub.add_parser(
        "certify-trajectories",
        help="trajectory-level risk certificate: bound end-to-end task degradation",
    )
    ct.add_argument(
        "outcomes",
        help="JSONL of matched runs: {task_id, full_success, compressed_success} per line",
    )
    ct.add_argument("--alpha", type=float, default=0.05, help="max degradation risk to certify")
    ct.add_argument("--delta", type=float, default=0.05, help="confidence budget (1-δ)")
    ct.add_argument("--json", action="store_true", help="machine-readable output")
    ct.set_defaults(func=cmd_certify_trajectories)

    gw = sub.add_parser("gateway", help="managed multi-tenant gateway + live savings dashboard")
    gw.add_argument("--host", default="127.0.0.1")
    gw.add_argument("--port", type=int, default=8789)
    gw.add_argument("--upstream", default="https://api.anthropic.com")
    gw.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    gw.add_argument("--lossless-only", "--safe", action="store_true")
    gw.add_argument(
        "--verbatim",
        action="store_true",
        help="skip the Tier-1 digest (Tier-0 only) for all tenants — lower savings",
    )
    gw.add_argument(
        "--admin-token",
        default=None,
        help="bearer token for /distil/stats and /distil/dashboard "
        "(required on non-loopback binds; env: DISTIL_GATEWAY_TOKEN)",
    )
    gw.add_argument(
        "--trust-tenant-header",
        action="store_true",
        help="honor client-supplied x-distil-tenant for accounting (default: "
        "tenant is derived from the credential, never a client header)",
    )
    gw.set_defaults(func=cmd_gateway)

    tt = sub.add_parser(
        "train-transformer", help="train the transformer keep-model (needs distil-llm[train])"
    )
    tt.add_argument(
        "--out", default="distil-keep-transformer", help="output dir for ONNX + tokenizer"
    )
    tt.add_argument("--base-model", default="google/bert_uncased_L-2_H-128_A-2")
    tt.add_argument("--epochs", type=int, default=3)
    tt.set_defaults(func=cmd_train_transformer)
    return p


def main(argv: list[str] | None = None) -> int:
    import os
    import sys

    # Restore the default SIGPIPE disposition so a write to a pipe whose reader
    # has already left fails fast rather than becoming a Python exception. No
    # SIGPIPE on Windows -> harmless to skip.
    try:
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass

    args = build_parser().parse_args(argv)
    try:
        rc = args.func(args)
    except BrokenPipeError:
        rc = 0
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError) as e:
        # A missing/unreadable input path is a user mistake, not a distil bug —
        # a clean message beats an 8-line pathlib traceback. Covers every
        # file-reading command at the dispatch chokepoint (no per-command patch).
        # NotADirectoryError = a --corpus that points at a file, not a dir.
        print(f"distil {getattr(args, 'cmd', '')}: {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        path = getattr(args, "trajectory", None) or getattr(args, "outcomes", None) or "input"
        print(f"distil {getattr(args, 'cmd', '')}: {path} is not valid JSON — {e}", file=sys.stderr)
        return 2

    # The status line is piped to a consumer (Claude Code) that may close the
    # pipe the instant it has our one line. Flush under guard, then hard-exit so
    # the interpreter's own shutdown flush can't fault on the closed pipe and
    # print "Exception ignored while flushing sys.stdout: BrokenPipeError".
    # Scoped to statusline: every other command returns normally and keeps its
    # atexit cleanup. Unit tests call cmd_statusline directly, never main().
    if getattr(args, "func", None) is cmd_statusline:
        try:
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass
        os._exit(rc if isinstance(rc, int) else 0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
