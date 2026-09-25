"""End-of-session Proof Ledger — compact printout on ``distil wrap`` exit.

Makes distil's rigor visible at exactly the moment users compare tools:
calibrated (not estimated) token/dollar accounting, shadow decision-equivalence
verdicts with honest suppression labeling, and per-session restorability.

Printed on clean exit and Ctrl-C; fail-open (a crash here must never affect
the wrapped command's exit code). Opt-out: DISTIL_NO_LEDGER=1.

The numbers come from the same on-disk sources as the leaderboard and
dashboard — no new estimators, no duplicated math.
"""

from __future__ import annotations

import json
import os
import sys
import time


def _dur(seconds: float) -> str:
    """Human-readable session duration: 47m, 2h3m, 30s."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    h, m = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{h}h{m}m" if m else f"{h}h"


def _session_digest_stats(session_id: str) -> tuple[int, bool]:
    """Count digest handles written this session; check that all are still live.

    Reads the session's ``*.requests.jsonl`` for per-session block handles
    (NOT the global restore store — that would conflate prior-session blocks
    with the current one). Checks each handle file's mtime against the TTL.

    Returns ``(n_digests, all_handles_live)``.
    """
    from . import ledger as _ledger
    from .mcp_server import _RESTORE_TTL_DAYS, _restore_dir

    req_path = _ledger.session_requests_path(session_id)
    if req_path is None or not req_path.exists():
        return 0, True

    session_handles: set[str] = set()
    try:
        for line in req_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                for block in rec.get("blocks") or []:
                    h = (block.get("h") or "").strip()
                    if h:
                        session_handles.add(h)
            except (ValueError, json.JSONDecodeError, AttributeError, TypeError):
                continue
    except OSError:
        return 0, True

    n = len(session_handles)
    if n == 0 or _RESTORE_TTL_DAYS <= 0:
        return n, True

    restore_d = _restore_dir()
    cutoff = time.time() - _RESTORE_TTL_DAYS * 86400
    all_live = True
    for h in session_handles:
        try:
            p = restore_d / h
            if not p.exists() or p.stat().st_mtime < cutoff:
                all_live = False
                break
        except OSError:
            all_live = False
            break

    return n, all_live


def _calib_note() -> str:
    """Calibration annotation for the cost line — honest about uncalibrated state.

    Mirrors the "calibrated to your billed usage (n=N, ±X%)" label that appears
    on the leaderboard. Until MIN_SAMPLES observations exist the factor is exactly
    1.0 (identity), so we call it "estimated" rather than claiming calibration.
    """
    from . import calibration as _calib

    f, n = _calib.factor()
    ci = _calib.relative_ci()
    calibrated = n >= _calib.MIN_SAMPLES and f != 1.0
    if calibrated and ci is not None:
        return f"calibrated to your billed usage (n={n}, ±{ci * 100:.1f}%)"
    if n >= _calib.MIN_SAMPLES:
        return f"calibrated (n={n}, factor≈1.0)"
    return f"estimated · calibrating ({n}/{_calib.MIN_SAMPLES} samples)"


def _shadow_line(since_ts: float, led=None) -> str:  # type: ignore[no-untyped-def]
    """Shadow decision-equivalence line, threshold-suppression-aware.

    Respects ``VERDICT_MIN_AB`` / ``VERDICT_MIN_AA``: when below threshold,
    prints sample counts WITHOUT a verdict so we never claim equivalence over
    statistically thin evidence. Matches the same suppression logic the status
    line and dashboard use.

    ``led`` is the session-scoped ledger when the caller already read the file — see
    :meth:`distil.shadow.ShadowLedger.load_split` for why the exit summary hands one in.
    """
    from .shadow import VERDICT_MIN_AA, VERDICT_MIN_AB, ShadowLedger

    if led is None:
        led = ShadowLedger.load(current_only=True, since_ts=since_ts)
    ab_n = led.samples
    aa_n = led.aa_samples
    changes = led.changes
    if ab_n == 0:
        return "no shadow samples  (enable: distil wrap --shadow-rate 0.1 -- <cmd>)"
    below = ab_n < VERDICT_MIN_AB or aa_n < VERDICT_MIN_AA
    plural = "s" if changes != 1 else ""
    suffix = f"(n={ab_n} A/B, {aa_n} A/A"
    if below:
        suffix += " — below verdict threshold, gathering)"
    else:
        eq = led.equivalence()
        ci = eq.pct_ci
        suffix += f", {eq.estimator}"
        if eq.pct is not None and ci is not None:
            suffix += f" — {eq.pct:.1f}% [{ci[0]:.1f}, {ci[1]:.1f}] 95% CI"
        suffix += ")"
    return f"{changes} decision change{plural}    {suffix}"


# ---------------------------------------------------------------------------
# The four verdict lines. Each one can come back negative — that is the point.
# All of them share shadow's reporting floor: below it they name the shortfall
# and print no number, because a statistic that can print a wrong verdict is
# worse than no line at all.
# ---------------------------------------------------------------------------


def _live_ledger():  # type: ignore[no-untyped-def]  # -> ShadowLedger (lazy import)
    """The shadow ledger, current signature version only — read ONCE per render.

    Three verdicts quote this evidence and they must quote the same evidence, which is the
    correctness argument. The performance one is the same read: the ledger grows one row
    per shadowed request forever, so a line-per-load design re-parsed it three times on
    every wrap exit and every ``distil stats``.
    """
    from .shadow import ShadowLedger

    return ShadowLedger.load(current_only=True)


def _drift_line(diffs: list[int], bound: float | None = None) -> str:
    """Anytime-valid budget alarm — is the certified decision-change budget still intact?

    The certificate (``distil conformal``) is a one-shot statement about a calibration
    corpus. This is the same claim, checked after every sample, with no multiplicity
    penalty: a betting e-process whose capital crossing ``1/delta`` means the live risk
    has exceeded the budget (Ville). See :func:`distil.drift.paired_loss` for the loss.
    It is handed the same bound the risk line prints, so it cannot say "intact" beside
    a bound above the budget, and it says plainly when the guard is holding compression.
    """
    from .drift import LiveDrift

    # Read-only: the e-process is written by the proxies that produce the verdicts
    # (distil.drift.fold). `distil stats` and wrap exit only report it.
    return "decision-change budget: " + LiveDrift.load().line(bound)


def _bound(diffs: list[int]) -> float | None:
    """The (1−δ) upper bound on the live decision-change rate, or None below the floor."""
    from . import conformal
    from .drift import paired_loss
    from .shadow import VERDICT_MIN_AB

    if len(diffs) < VERDICT_MIN_AB:
        return None
    b = 2.0 * conformal.tight_risk_bound([paired_loss(d) for d in diffs], conformal.BUDGET_DELTA)
    return max(0.0, b - 1.0)


def _output_line(led) -> str:  # type: ignore[no-untyped-def]
    """Effect of compression on REPLY length, measured by shadow's paired replays.

    Distinct from ``--shape-output``, which asks the model for shorter replies: this is
    what compression does to reply length on traffic that asked for nothing. The
    direction word is only printed when the interval excludes zero.
    """
    from .shadow import VERDICT_MIN_AB

    cost = led.cost()
    if cost is None or cost.n < VERDICT_MIN_AB:
        n = 0 if cost is None else cost.n
        return f"not enough samples yet ({n}/{VERDICT_MIN_AB})"
    lo, hi = cost.out_delta_ci
    scope = f"n={cost.n}, shadow-measured on all traffic; not --shape-output"
    if lo <= 0.0 <= hi:
        return f"no measurable effect on reply length ({scope})"
    word = "shorter" if cost.out_delta_mean < 0 else "longer"
    return (
        f"the model's replies were {abs(cost.out_delta_mean):.0f} tokens {word} per request "
        f"under compression (95% CI [{lo:+.1f}, {hi:+.1f}], {scope})"
    )


def _risk_line(diffs: list[int], bound: float | None = None) -> str:
    """Distribution-free (1−delta) upper bound on the live decision-change rate.

    ``tight_risk_bound`` on the same affine-mapped paired losses the drift monitor bets
    on; the bound is on ``E[x] = (1 + harm)/2``, so it is mapped back the same way. Wider
    than the bootstrap interval next to it on purpose — this one assumes no distribution
    and holds at finite n, which the percentile bootstrap does not.
    """
    from . import conformal
    from .shadow import VERDICT_MIN_AB

    n = len(diffs)
    if n < VERDICT_MIN_AB:
        return f"not enough samples yet ({n}/{VERDICT_MIN_AB})"
    if bound is None:
        bound = _bound(diffs)
        assert bound is not None  # n is above the floor
    conf = round((1.0 - conformal.BUDGET_DELTA) * 100)
    side = "within" if conformal.within_budget(bound) else "ABOVE"
    return (
        f"decision-change risk ≤ {bound * 100:.1f}% ({conf}% conformal bound, n={n}) — "
        f"{side} the {conformal.budget_pct()} budget"
    )


def _receipts_line() -> str:
    """Hash-chain verdict for the per-request receipts — the one artifact a third party
    can check without trusting us. Verified at exit instead of only in a command nobody
    remembers to run."""
    from . import receipts as _r

    v = _r.verify(full=False)  # the resumed pass: this machine re-checking its own tail
    if v.total == 0:
        return "no receipts recorded"
    if not v.ok:
        return f"chain BROKEN at receipt {v.first_bad_index} of {v.total} — {v.reason}"
    if v.skipped:
        return (
            f"{v.total} receipts, chain intact — but {v.skipped} line(s) in the file are not "
            "receipts (torn write or foreign text); `distil receipts` for details"
        )
    if not v.checked_from:
        return f"{v.total} receipts, chain verified (every hash re-checked)"
    # A checkpointed pass re-hashes only what was appended since this machine's last
    # pass; the prefix is trusted from that checkpoint. Say exactly that, or the line
    # claims a full verification `distil receipts` (the default, full pass) would have to earn.
    return (
        f"{v.total} receipts, chain intact — {v.total - v.checked_from} re-checked since "
        f"this machine's last pass; `distil receipts` re-hashes all"
    )


def proof_lines(led=None) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    """``(label, sentence)`` for every statistical verdict, shared by the wrap exit
    summary, ``distil stats`` and ``distil dissect`` — one implementation, so the three
    surfaces cannot report different verdicts off the same ledger.

    The shadow ledger is read once and handed to the three lines that quote it, and a
    caller that has already read it (lifetime-scoped, current signature) passes it in
    rather than paying for a second parse. Each line is then computed in isolation: a
    statistic that cannot be computed drops its own line and the other three still print,
    because one unreadable state file should not take the whole verdict block down with it.
    """
    if led is None:
        led = _live_ledger()
    diffs = list(led.paired_diffs)
    # One bound, quoted by both the budget and the risk line — they must not disagree.
    try:
        bound = _bound(diffs)
    except Exception:  # noqa: BLE001 — no bound costs the two lines' numbers, not the block
        bound = None
    return [
        (label, text)
        for label, text in (
            ("budget", _guarded(lambda: _drift_line(diffs, bound))),
            ("risk", _guarded(lambda: _risk_line(diffs, bound))),
            ("output", _guarded(lambda: _output_line(led))),
            ("receipts", _guarded(_receipts_line)),
        )
        if text
    ]


def _guarded(fn) -> str:  # type: ignore[no-untyped-def]
    """Run one verdict; an unreadable artifact costs that line, not the block.

    The empty string means "printed nothing", which the caller filters out — deliberately
    NOT a "could not compute" line, because a reader cannot act on that and the accounting
    above it is still true.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001 — a verdict that cannot be computed is not printed
        return ""


def build_ledger_text(session_id: str, start_ts: float) -> str | None:
    """Return the formatted proof ledger block, or None if no proxied requests.

    All data comes from the same on-disk sources the leaderboard uses:
    ``ledger.summary(session=...)`` for token/dollar savings, ``calibration``
    module for the correction factor, ``ShadowLedger.load`` for equivalence
    stats, and the session's ``*.requests.jsonl`` for restorability.
    """
    from . import calibration as _calib
    from . import ledger as _ledger
    from .shadow import ShadowLedger

    sess = _ledger.summary(session=session_id)
    if sess.runs == 0:
        return None  # no proxied requests — stay silent (no noise on bypass)

    # ONE pass for both scopes the block needs: the session line since `start_ts`, the
    # verdicts lifetime-wide. Loading them separately parsed an unbounded file twice.
    shadow_all, shadow_session = ShadowLedger.load_split(start_ts)

    dur = _dur(time.time() - start_ts)
    base_tok = sess.total_baseline_tokens
    dist_tok = sess.total_distil_tokens

    # Apply calibration factor to token counts — same transform as render_html.
    f, _ = _calib.factor()
    cal_base = round(base_tok * f)
    cal_dist = round(dist_tok * f)
    pct = (1.0 - dist_tok / base_tok) * 100.0 if base_tok else 0.0

    base_usd = sess.total_baseline_dollars
    dist_usd = sess.total_distil_dollars

    n_digests, all_live = _session_digest_stats(session_id)
    from .mcp_server import _RESTORE_TTL_DAYS

    ttl = int(_RESTORE_TTL_DAYS)
    digest_label = "digest" if n_digests == 1 else "digests"
    if all_live:
        restore = f"100% recoverable    ({n_digests} {digest_label}, all handles live, TTL {ttl}d)"
    else:
        restore = f"{n_digests} {digest_label}, some handles expired (TTL {ttl}d)"

    lines = [
        f"\n  distil proof ledger — session {dur}",
        f"    tokens   {cal_base:,} → {cal_dist:,}   ({pct:.1f}% smaller)",
        f"    cost     ${base_usd:,.2f} → ${dist_usd:,.2f}        {_calib_note()}",
        f"    shadow   {_shadow_line(start_ts, shadow_session)}",
        f"    restore  {restore}",
        # The verdicts are the reading; the two lines below are what to do about it, so
        # they come after every statistic and before anything that names a command.
        *(f"    {label:<8} {text}" for label, text in _safe_proof_lines(shadow_all)),
    ]
    # At most one extra line, and only when a detector actually fired. The ledger is
    # what a user sees on every exit, so an advisory that prints "0 actions" every
    # time is an ad; one that appears only when there is something to act on is a
    # finding. It comes before the standing next-command line so the summary still
    # ends on the routine action, not a one-off recommendation.
    advice = _discover_line()
    if advice:
        lines.append(f"    next     {advice}  (run: distil discover)")
    lines.append(
        f"    next     distil dissect {session_id}   (or: distil stats for cumulative savings)"
    )
    return "\n".join(lines)


def _discover_line() -> str | None:
    """The missed-savings advisor's one-line teaser, or None.

    Fail-open on its own: `print_proof_ledger` already swallows everything, but a
    cross-session scan reads more files than the rest of this module and must not
    be the reason a session's proof is missing entirely.
    """
    try:
        from .discover import wrap_exit_line

        return wrap_exit_line()
    except Exception:  # noqa: BLE001 — an advisory must never cost the proof ledger
        return None


def _safe_proof_lines(led=None) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    """:func:`proof_lines`, but a broken statistic drops its own line instead of the
    whole ledger. ``print_proof_ledger`` is already fail-open; this keeps the token and
    dollar accounting visible when only the drift state file is unreadable."""
    try:
        return proof_lines(led)
    except Exception:  # noqa: BLE001 — a verdict that cannot be computed is not printed
        return []


def print_proof_ledger(session_id: str, start_ts: float) -> None:
    """Print the proof ledger to stderr. Fail-open; opt-out via DISTIL_NO_LEDGER=1.

    A crash inside this function must never propagate — the ledger is informational
    and the wrapped command's exit code must be preserved regardless.
    """
    if os.environ.get("DISTIL_NO_LEDGER") == "1":
        return
    try:
        text = build_ledger_text(session_id, start_ts)
        if text is not None:
            print(text, file=sys.stderr)
    except Exception:  # noqa: BLE001 — ledger print must never affect exit code
        pass
