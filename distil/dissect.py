"""Post-hoc dissection of one wrap session — everything distil knows about it.

``distil dissect`` answers "what exactly happened to my session?": token and
dollar accounting per model, what was digested (and is still recoverable), what
was *not* optimized and why, plus cache-delta, shadow and expand activity.

Data sources (all local; all content-free except restore-blob *existence*):

- ``savings.jsonl`` rows tagged with the session id — token/dollar accounting.
- ``sessions/<sid>.json`` — wrap manifest (tool, argv, flags, billing).
- ``sessions/<sid>.requests.jsonl`` — per-request detail (token breakdown,
  per-block digest signatures, shadow/expand flags).
- ``sessions/<sid>`` / ``.hb`` / ``.exit`` — liveness breadcrumbs.
- ``restore/<handle>`` existence — is a digested block still recoverable?
- ``shadow.jsonl`` rows inside the session's time window (rows are not
  session-tagged; the join is by time and is labelled as such).

Manifest and request detail exist only for sessions wrapped by this version or
newer; older sessions degrade to the ledger-only view with a note.
"""

from __future__ import annotations

import html as _html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

from .ledger import (
    default_path,
    session_manifest_path,
    session_marker_path,
    session_requests_path,
)

if TYPE_CHECKING:  # pragma: no cover — render-time only
    from .correlate import Correlation


def _state_dir() -> Path:
    return default_path().parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Tolerant JSONL reader: missing file -> [], corrupt lines skipped."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


@dataclass
class SessionOverview:
    """One row of the no-argument session picker."""

    sid: str
    tool: str = ""
    started: float = 0.0
    last_ts: float = 0.0
    requests: int = 0
    baseline_tokens: int = 0
    distil_tokens: int = 0
    status: str = ""  # "live" | "exited" | ""


def list_sessions() -> list[SessionOverview]:
    """Every session distil has heard of: ledger rows ∪ session manifests.

    Newest-last-activity first, so the session you just ran is on top.
    """
    by_sid: dict[str, SessionOverview] = {}
    for rec in _read_jsonl(default_path()):
        sid = rec.get("session")
        if not sid or not isinstance(sid, str):
            continue
        ov = by_sid.setdefault(sid, SessionOverview(sid=sid))
        ts = float(rec.get("ts") or 0.0)
        ov.started = min(ov.started or ts, ts)
        ov.last_ts = max(ov.last_ts, ts)
        ov.requests += int(rec.get("turns") or 0)
        ov.baseline_tokens += int(rec.get("baseline_input_tokens") or 0)
        ov.distil_tokens += int(rec.get("distil_input_tokens") or 0)
    sess_dir = _state_dir() / "sessions"
    try:
        manifests = sorted(sess_dir.glob("s*.json"))
    except OSError:
        manifests = []
    for mp in manifests:
        try:
            man = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = man.get("sid") or mp.stem
        ov = by_sid.setdefault(sid, SessionOverview(sid=sid))
        ov.tool = str(man.get("tool") or "")
        started = float(man.get("started_ts") or 0.0)
        if started:
            ov.started = min(ov.started or started, started)
            ov.last_ts = max(ov.last_ts, started)
    for ov in by_sid.values():
        marker = session_marker_path(ov.sid)
        if marker is not None:
            if marker.with_suffix(".exit").exists():
                ov.status = "exited"
            elif marker.exists():
                ov.status = "live"
    return sorted(by_sid.values(), key=lambda o: o.last_ts, reverse=True)


def resolve_sid(query: str) -> str | None:
    """Resolve ``latest`` or a unique session-id prefix to a full id."""
    sessions = list_sessions()
    if not sessions:
        return None
    if query == "latest":
        return sessions[0].sid
    hits = [o.sid for o in sessions if o.sid == query]
    if hits:
        return hits[0]
    hits = [o.sid for o in sessions if o.sid.startswith(query)]
    return hits[0] if len(hits) == 1 else None


@dataclass
class Dissection:
    """Everything distil knows about one wrap session, joined and totalled."""

    sid: str
    manifest: dict[str, Any] | None
    ledger_rows: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    marker: str | None  # "0" (wrapped, no traffic yet), "1" (traffic seen), None
    heartbeat: str | None
    exit_note: str | None
    shadow_window_rows: int = 0
    shadow_window_agree: int = 0
    #: Output-token accounting from the paired shadow replays in this window: the
    #: mean (B − A) output-token delta and the net dollars after paying for it.
    shadow_out_delta: float | None = None
    shadow_net_usd: float | None = None
    # Derived digest inventory: handle -> {"sig", "tokens", "folds", "recoverable"}
    blocks: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ---- ledger-derived totals (available even for pre-manifest sessions) ----
    @property
    def baseline_tokens(self) -> int:
        return sum(int(r.get("baseline_input_tokens") or 0) for r in self.ledger_rows)

    @property
    def distil_tokens(self) -> int:
        return sum(int(r.get("distil_input_tokens") or 0) for r in self.ledger_rows)

    @property
    def dollars_saved(self) -> float:
        return sum(
            float(r.get("baseline_dollars") or 0.0) - float(r.get("distil_dollars") or 0.0)
            for r in self.ledger_rows
        )

    @property
    def pct_saved(self) -> float:
        b = self.baseline_tokens
        return 100.0 * (b - self.distil_tokens) / b if b else 0.0

    @property
    def started(self) -> float:
        cands = [float(r.get("ts") or 0.0) for r in self.ledger_rows]
        cands += [float(r.get("ts") or 0.0) for r in self.requests]
        if self.manifest:
            cands.append(float(self.manifest.get("started_ts") or 0.0))
        cands = [c for c in cands if c]
        return min(cands) if cands else 0.0

    @property
    def ended(self) -> float:
        cands = [float(r.get("ts") or 0.0) for r in self.ledger_rows]
        cands += [float(r.get("ts") or 0.0) for r in self.requests]
        return max(cands) if cands else 0.0

    @property
    def billing(self) -> str:
        """Manifest billing mode, falling back to this machine's current mode
        (labelled detection, not a session fact) for pre-manifest sessions."""
        if self.manifest and self.manifest.get("billing"):
            return str(self.manifest["billing"])
        try:
            from .doctor import subscription_mode

            return "subscription" if subscription_mode() else "metered"
        except Exception:  # noqa: BLE001 — billing detection is cosmetic
            return "unknown"

    def per_model(self) -> list[tuple[str, int, int, int]]:
        """[(model, booked_requests, baseline_tokens, distil_tokens)] biggest first."""
        agg: dict[str, list[int]] = {}
        for r in self.ledger_rows:
            m = agg.setdefault(str(r.get("model") or "unknown"), [0, 0, 0])
            m[0] += int(r.get("turns") or 0)
            m[1] += int(r.get("baseline_input_tokens") or 0)
            m[2] += int(r.get("distil_input_tokens") or 0)
        return sorted(((k, v[0], v[1], v[2]) for k, v in agg.items()), key=lambda t: -t[2])

    # ---- request-detail-derived (needs sessions/<sid>.requests.jsonl) ----
    @property
    def detail_available(self) -> bool:
        return bool(self.requests)

    @property
    def booked_detail(self) -> list[dict[str, Any]]:
        """Request-detail rows for booked requests only — the same population the
        ledger headline (baseline_tokens/pct_saved) is built from. Summing the
        detail over *all* requests instead double-counts upstream retries, which
        made the mechanism and overhead lines disagree with the savings line."""
        return [r for r in self.requests if r.get("booked")]

    @property
    def delta_tokens_saved(self) -> int:
        return sum(int(r.get("delta_tokens_saved") or 0) for r in self.booked_detail)

    @property
    def overhead_tokens_avg(self) -> int:
        vals = [int(r.get("overhead_tokens") or 0) for r in self.requests]
        return sum(vals) // len(vals) if vals else 0

    @property
    def verbatim_requests(self) -> int:
        return sum(1 for r in self.requests if r.get("mode") in (None, "", "verbatim"))

    @property
    def unbooked_requests(self) -> int:
        return sum(1 for r in self.requests if not r.get("booked"))

    @property
    def shadow_sampled(self) -> int:
        return sum(1 for r in self.requests if r.get("shadow_sampled"))

    @property
    def expand_resolved(self) -> int:
        return sum(1 for r in self.requests if r.get("expanded"))

    def retried_and_recovered(self) -> int:
        """Non-2xx requests immediately followed by a same-size 2xx — i.e. a retry
        that worked.

        Without this a session reads as ~30% failing when nothing was actually lost:
        the SDK retried and succeeded every time. Distinguishing "transient, healed"
        from "fatal" is the whole value of the number — a rate alone caused a real
        misdiagnosis of a live session (a size correlation that looked causal and
        was not).
        """
        rows = sorted(self.requests, key=lambda r: r.get("ts") or 0)

        def size(r: dict[str, Any]) -> int:
            return int(r.get("compressible_tokens") or 0) + int(r.get("overhead_tokens") or 0)

        n = 0
        for i, r in enumerate(rows):
            st = r.get("status")
            if isinstance(st, int) and 200 <= st < 300:
                continue
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            if not nxt:
                continue
            nst = nxt.get("status")
            if isinstance(nst, int) and 200 <= nst < 300 and abs(size(nxt) - size(r)) <= 2000:
                n += 1
        return n

    def error_reasons(self) -> list[tuple[str, int]]:
        """Provider error types for the non-2xx requests, most frequent first.

        Recorded per request as a short enum (`invalid_request_error`,
        `rate_limit_error`, …). Never the human message, which can quote content.
        Empty for sessions captured before 1.50.2, which recorded only the status.
        """
        counts: dict[str, int] = {}
        for r in self.requests:
            et = r.get("error_type")
            if isinstance(et, str) and et:
                counts[et] = counts.get(et, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    @property
    def expand_missed(self) -> int:
        """Handles the agent asked for that the store could not return.

        F3, post-compression access failure: the saving was booked when the block
        was digested, but the original was gone by the time it was wanted (evicted
        past DISTIL_RESTORE_CAP, or aged past the TTL). Fail-open hands the agent a
        placeholder and the request succeeds, so nothing else in the pipeline
        notices — which is exactly why it needs its own counter."""
        return sum(int(r.get("expand_misses") or 0) for r in self.requests)

    # ---- insight metrics (all derived; None/0 when the inputs are absent) ----
    @property
    def tokens_saved_total(self) -> int:
        """Heuristic tokens saved across requests (digest folds + cache-delta).

        Booked-only, so the mechanism split reconciles with the savings headline
        rather than including work on requests the upstream rejected/retried."""
        return sum(int(r.get("tokens_saved") or 0) for r in self.booked_detail)

    @property
    def digest_saved(self) -> int:
        """Mechanism decomposition: the non-delta share of tokens_saved."""
        return max(0, self.tokens_saved_total - self.delta_tokens_saved)

    @property
    def overhead_tokens_total(self) -> int:
        return sum(int(r.get("overhead_tokens") or 0) for r in self.booked_detail)

    @property
    def overhead_share(self) -> float:
        """Fixed tax: system prompt + tool definitions as a share of what was actually
        sent — i.e. measured *after* compression, over the same booked population the
        savings headline uses. Dividing by the pre-compression total understated the
        share (the denominator counted tokens distil had already removed)."""
        comp = sum(int(r.get("compressible_tokens") or 0) for r in self.booked_detail)
        total = self.overhead_tokens_total + max(0, comp - self.tokens_saved_total)
        return 100.0 * self.overhead_tokens_total / total if total else 0.0

    @property
    def eligibility(self) -> list[tuple[str, int, float]]:
        """Why the session compressed as much as it did: ``(reason, tokens, pct)``,
        largest first, summed over every request's per-block census.

        This is the answer to "is a low saving a defect or the design working?" — a
        question the savings number itself cannot settle. ``tool_result_digested``
        is work done; ``assistant_text`` and ``tool_result_recent`` are content the
        policy deliberately protects; ``tool_result_declined`` is the only bucket
        that indicts the compressor. Empty for sessions recorded before the census
        existed, which is why callers must treat [] as "unknown", not as "nothing".
        """
        agg: dict[str, int] = {}
        for r in self.requests:
            for reason, tokens in (r.get("census") or {}).items():
                agg[str(reason)] = agg.get(str(reason), 0) + int(tokens or 0)
        total = sum(agg.values())
        if not total:
            return []
        return sorted(((k, v, 100.0 * v / total) for k, v in agg.items()), key=lambda x: -x[1])

    def _recoverability_note(self) -> str:
        """State recoverability as measured, not as promised.

        The restore store is capped (DISTIL_RESTORE_CAP, LRU), so a long session can
        evict its own earliest folds while it is still running — including the most
        re-used ones. Claiming "everything stays recoverable" while a third of the
        blocks are already gone is the one line in this report a user could catch
        being false, so it reports the real count whenever any block has been evicted."""
        total = len(self.blocks)
        if not total:
            return "Everything summarized stays recoverable on this machine."
        rec = sum(1 for i in self.blocks.values() if i.get("recoverable"))
        if rec == total:
            return "Everything summarized stays recoverable on this machine."
        return (
            f"{rec} of {total} summarized blocks are still recoverable on this machine "
            f"— the rest aged out of the restore store (raise DISTIL_RESTORE_CAP to keep more)."
        )

    def _churn_advice(self) -> str:
        """Recommend session-delta only when the churn is actually being paid for.

        Re-folded tokens look expensive, but on a well-cached agent session the
        provider is already serving them from its prompt cache at the (much cheaper)
        cache-read rate. Recommending cache-delta against a 99%-cached session sends
        the user after a saving that is an order of magnitude smaller than the churn
        number implies — and cache-delta's own value is dedup of content the cache
        already deduplicates."""
        cached = self.cached_input_share
        if cached is not None and cached >= 90.0:
            return (
                f"That said, {cached:.0f}% of this session's billed input was served from "
                "the provider's prompt cache, so this content is already discounted — "
                "cache-delta would recover far less than the number above suggests."
            )
        return (
            "The session-delta cache absorbs exactly this; if it is already on, these "
            "blocks are candidates for the learned codec."
        )

    @property
    def cached_input_share(self) -> float | None:
        """Share of billed input the provider served from its prompt cache.

        Decides whether re-fold churn is actually expensive. A high share means the
        resent content is already billed at the cache-read rate, so the headline churn
        number overstates what any dedup mechanism could recover. None when no usage
        was recorded.

        Older records carry only the aggregate ``usage_cache_tokens`` and neither split
        field. Summing the split fields over those rows gives 0 cached against a nonzero
        input total, so the share reads a confident **0.0%** — "the cache never hit" —
        for a session where it was never measured. Those are opposite diagnoses, so
        require at least one row to carry a split field before reporting anything.
        """
        detail = self.booked_detail
        if not any(
            r.get("usage_cache_read") is not None or r.get("usage_cache_create") is not None
            for r in detail
        ):
            return None
        cached = sum(int(r.get("usage_cache_read") or 0) for r in detail)
        total = sum(
            int(r.get("usage_input_tokens") or 0)
            + int(r.get("usage_cache_read") or 0)
            + int(r.get("usage_cache_create") or 0)
            for r in detail
        )
        return 100.0 * cached / total if total else None

    def protected_share(self, model: str) -> float | None:
        """Share of one model's eligible tokens that policy refuses to digest.

        A per-model 0.0% row is otherwise indistinguishable from a broken gate. When
        a model's traffic is entirely user prompts or assistant text (subagents that
        send one big string and no tools), 0% saved is the protection policy holding,
        and this is what says so. None when the census is absent for that model."""
        agg: dict[str, int] = {}
        for r in self.requests:
            if r.get("model") != model:
                continue
            for reason, tokens in (r.get("census") or {}).items():
                agg[str(reason)] = agg.get(str(reason), 0) + int(tokens or 0)
        total = sum(agg.values())
        if not total:
            return None
        prot = sum(v for k, v in agg.items() if k in _PROTECTED_REASONS)
        return 100.0 * prot / total

    @property
    def churn_tokens(self) -> int:
        """Tokens re-folded after first sight — resent content the client keeps sending."""
        return sum(
            int(i.get("tokens") or 0) * (int(i.get("folds") or 1) - 1) for i in self.blocks.values()
        )

    @property
    def churned_blocks(self) -> int:
        """Blocks folded more than once — the same count churn_tokens sums over."""
        return sum(1 for i in self.blocks.values() if int(i.get("folds") or 0) >= 2)

    @property
    def usage_input_total(self) -> int:
        """Full billed input: uncached + the cached prefix (cache_read/creation).

        ``usage.input_tokens`` alone omits everything served from cache, which on an
        agent session is most of the input — reporting it as "billed in" made output
        look like ~95% of billed traffic when it is well under 1%.
        """
        return sum(
            int(r.get("usage_input_tokens") or 0) + int(r.get("usage_cache_tokens") or 0)
            for r in self.requests
        )

    @property
    def usage_output_total(self) -> int:
        return sum(int(r.get("usage_output_tokens") or 0) for r in self.requests)

    @property
    def usage_requests(self) -> int:
        return sum(1 for r in self.requests if r.get("usage_input_tokens") is not None)

    def calibration(self) -> tuple[int, int] | None:
        """(calibrated_estimate, billed) input tokens over requests that carry usage.

        Estimate = (overhead + compressible-after-savings) x the per-model factor
        distil has already learned from billed usage; billed = the API's own
        usage.input_tokens. This is what turns "we think we saved X" into a
        measured claim (and shows how honest the calibrated count is).
        """
        from .calibration import factor as _factor

        est_f = billed = 0.0
        _fac: dict[str, float] = {}
        for r in self.requests:
            u = r.get("usage_input_tokens")
            if u is None:
                continue
            raw = int(r.get("overhead_tokens") or 0) + max(
                0, int(r.get("compressible_tokens") or 0) - int(r.get("tokens_saved") or 0)
            )
            # The heuristic tokenizer is systematically off per model (opus ~2.3x on this
            # ledger), and distil already learns that ratio from billed usage. Report the
            # calibrated estimate — otherwise every calibrated session cries "off by >50%".
            m = str(r.get("model") or "")
            if m not in _fac:
                _fac[m] = _factor(m)[0]
            est_f += raw * _fac[m]
            # Full billed input = uncached + cached prefix. input_tokens alone omits the cached
            # portion (billed under cache_read/creation), which would make est >> billed and the
            # calibration meaningless on prompt-cached traffic.
            billed += int(u) + int(r.get("usage_cache_tokens") or 0)
        return (round(est_f), round(billed)) if billed else None

    @property
    def headroom_multiplier(self) -> float:
        """How much further the same context budget goes (baseline/sent)."""
        return self.baseline_tokens / self.distil_tokens if self.distil_tokens else 0.0

    @property
    def forced_buffered(self) -> int:
        """Client asked to stream but the expand loop forced full buffering (TTFT tax)."""
        return sum(1 for r in self.requests if r.get("client_stream") and not r.get("stream"))

    def latency_by_path(self) -> list[tuple[str, int, int]]:
        """[(path, requests, avg_ms)] — streamed / buffered (forced) / buffered."""
        groups: dict[str, list[int]] = {}
        for r in self.requests:
            if r.get("duration_ms") is None:
                continue
            if r.get("stream"):
                key = "streamed"
            elif r.get("client_stream"):
                key = "buffered (forced by expand)"
            else:
                key = "buffered"
            groups.setdefault(key, []).append(int(r["duration_ms"]))
        return [(k, len(v), sum(v) // len(v)) for k, v in sorted(groups.items())]

    def expansion_regret(self) -> list[tuple[str, int, int]]:
        """[(sig, expanded_blocks, folded_blocks)] — kinds the agent keeps pulling back."""
        expanded: set[str] = set()
        for r in self.requests:
            expanded.update(h for h in r.get("expanded_handles") or [] if isinstance(h, str))
        by_sig: dict[str, list[int]] = {}
        for h, info in self.blocks.items():
            m = by_sig.setdefault(str(info.get("sig") or "?"), [0, 0])
            m[1] += 1
            if h in expanded:
                m[0] += 1
        return sorted(
            ((s, v[0], v[1]) for s, v in by_sig.items() if v[0]),
            key=lambda t: -t[1],
        )

    def anomalies(self, peers: list[SessionOverview] | None = None) -> list[str]:
        """Things worth your attention — each one is a misconfiguration, a silent
        failure, or upstream weather that plain totals would hide."""
        out: list[str] = []
        flags = (self.manifest or {}).get("flags") or {}
        n = len(self.requests)
        if self.marker == "0" and not self.ledger_rows:
            out.append(
                "wrapped but no traffic went through distil — the agent may be "
                "bypassing the proxy (check ANTHROPIC_BASE_URL overrides)"
            )
        rate = float(flags.get("shadow_rate") or 0.0)
        if rate > 0 and n >= 10 and self.shadow_sampled == 0:
            out.append(
                f"shadow_rate={rate} but 0 of {n} requests were sampled "
                f"(expected ~{rate * n:.0f}) — shadow may be silently failing"
            )
        if self.shadow_sampled > 0 and self.shadow_window_rows == 0:
            out.append(
                f"{self.shadow_sampled} requests were shadow-sampled but no verdicts "
                "were recorded — replays may be failing upstream"
            )
        if flags.get("expand") and self.blocks and n >= 10 and self.expand_resolved == 0:
            if self.forced_buffered == 0 and any(r.get("stream") for r in self.requests):
                out.append(
                    "expand is on and blocks were folded, but every request took the "
                    "streaming pass-through — distil_expand calls could never be "
                    "intercepted (an escaped call surfaces to the agent as "
                    "'no such tool')"
                )
        if self.expand_missed:
            asked = self.expand_resolved + self.expand_missed
            out.append(
                f"{self.expand_missed}/{asked} expand requests could not be recovered "
                "— the saving was booked but the original was gone (raise "
                "DISTIL_RESTORE_CAP or DISTIL_RESTORE_TTL_DAYS)"
            )
        if n >= 5 and self.unbooked_requests / n > 0.2:
            # Name the actual reason. "upstream errors or rate limiting" was a guess
            # dressed as a diagnosis, and it is the difference between "your prompt
            # exceeded the context window" (nothing to fix in distil) and "distil sent
            # a malformed body" (everything to fix) — opposite conclusions from the
            # same line.
            why = self.error_reasons()
            recovered = self.retried_and_recovered()
            detail = ""
            if why:
                detail = " — " + ", ".join(f"{k} x{v}" for k, v in why)
            if recovered:
                # The difference between "your session is broken" and "the SDK did its
                # job". A non-2xx immediately followed by a 2xx of the same size is a
                # retry that worked: no work was lost, and reporting it as a failure
                # sends people hunting a bug that already healed itself.
                detail += (
                    f"\n                   {recovered} of them were retried "
                    "immediately and SUCCEEDED — transient, no work lost"
                )
            out.append(
                f"{self.unbooked_requests}/{n} requests were not booked (non-2xx or "
                f"SDK-retried){detail}"
            )
        if flags.get("lossless_only") and self.billing == "metered":
            out.append(
                "lossless-only mode on metered billing — the digest tier is off; "
                "savings are limited to delta/dedup"
            )
        if peers and self.baseline_tokens > 10_000:
            others = sorted(
                100.0 * (p.baseline_tokens - p.distil_tokens) / p.baseline_tokens
                for p in peers
                if p.sid != self.sid and p.baseline_tokens > 10_000
            )
            if len(others) >= 3:
                median = others[len(others) // 2]
                if self.pct_saved < 0.5 * median:
                    out.append(
                        f"savings ({self.pct_saved:.1f}%) are well below your typical "
                        f"session ({median:.1f}% median) — check the wrap flags"
                    )
        cal = self.calibration()
        if cal is not None:
            est, billed = cal
            if est and (est / billed > 1.5 or est / billed < 0.67):
                out.append(
                    f"token estimate is off by >50% vs billed usage "
                    f"({est:,} est vs {billed:,} billed) — treat % figures as rough"
                )
        return out

    @property
    def system_tokens_avg(self) -> int:
        vals = [int(r.get("system_tokens") or 0) for r in self.requests]
        return sum(vals) // len(vals) if vals else 0

    @property
    def tools_tokens_avg(self) -> int:
        vals = [int(r.get("tools_tokens") or 0) for r in self.requests]
        return sum(vals) // len(vals) if vals else 0

    def system_growth(self) -> tuple[int, int] | None:
        """(first, last) system-prompt size — memory/context injections show up here."""
        vals = [int(r.get("system_tokens") or 0) for r in self.requests if r.get("system_tokens")]
        return (vals[0], vals[-1]) if len(vals) >= 2 else None

    def tool_costs(self) -> list[tuple[str, int, int]]:
        """[(tool_name, tokens_per_request, session_total)] biggest total first.

        A tool definition is resent on every request, so its session cost is
        its size × the requests that carried it — the "trim this" worklist.
        """
        per: dict[str, list[int]] = {}
        for r in self.requests:
            for t in r.get("tools") or []:
                name = str(t.get("name") or "?")
                m = per.setdefault(name, [0, 0])
                m[0] = max(m[0], int(t.get("tokens") or 0))
                m[1] += 1
        return sorted(((name, v[0], v[0] * v[1]) for name, v in per.items()), key=lambda t: -t[2])

    def headlines(self) -> list[tuple[str, str]]:
        """The layman's story of the session: [(headline, detail)] in plain
        language, data-driven, biggest thing first. This is what a reader who
        will never hover a tile should still walk away knowing."""
        out: list[tuple[str, str]] = []
        kind_words = {
            "log": "large logs",
            "prose": "long text output",
            "code": "code listings",
            "error": "error output",
            "traceback": "stack traces",
            "diff": "diffs",
            "columnar": "tabular data",
        }
        if self.baseline_tokens:
            kinds = self.blocks_by_kind()
            mostly = ""
            if kinds:
                word = kind_words.get(kinds[0][0].split(":")[0], "bulky content")
                mostly = f", mostly by summarizing {word}"
            kept = self.baseline_tokens - self.distil_tokens
            out.append(
                (
                    f"distil kept {_human(kept)} of {_human(self.baseline_tokens)} input "
                    f"tokens off the wire ({self.pct_saved:.0f}%)",
                    f"Your session's conversation and tool results would have cost "
                    f"{_human(self.baseline_tokens)} tokens to resend across its requests; "
                    f"{_human(self.distil_tokens)} actually went out{mostly}. "
                    + self._recoverability_note(),
                )
            )
        if self.detail_available and self.overhead_share >= 30:
            n_tools = len(self.tool_costs())
            out.append(
                (
                    f"Your agent's fixed setup is {self.overhead_share:.0f}% of everything sent",
                    f"The system prompt plus {n_tools} tool definitions are resent "
                    "word-for-word on every request, and no compression applies to them. "
                    "Disabling tools/MCP servers you don't use is the single cheapest "
                    "saving available.",
                )
            )
        if self.usage_output_total:
            total_usage = self.usage_input_total + self.usage_output_total
            pct = 100.0 * self.usage_output_total / total_usage if total_usage else 0.0
            shaping = (self.manifest or {}).get("flags", {}).get("shape_output", "off")
            if self.billing == "subscription":
                shaping_note = (
                    "Live replies are never shortened on a subscription — output shaping "
                    "is gated to metered billing, where its effect is measured before "
                    "being trusted."
                )
            elif shaping and shaping != "off":
                shaping_note = f"Output shaping is on ({shaping}) for live replies."
            else:
                shaping_note = (
                    "Live replies can additionally be shortened with --shape-output "
                    "(off for this session)."
                )
            out.append(
                (
                    f"The model wrote {_human(self.usage_output_total)} output tokens "
                    f"({pct:.0f}% of billed traffic)",
                    "Output tokens are the model's own replies. distil trims verbose past "
                    "replies when they re-enter later requests as context (that saving is "
                    f"counted above). {shaping_note}",
                )
            )
        if (
            self.churn_tokens
            and self.tokens_saved_total
            and (self.churn_tokens >= 0.25 * self.tokens_saved_total)
        ):
            out.append(
                (
                    f"{_human(self.churn_tokens)} tokens were resent and re-summarized",
                    "The client kept resending the same content, so distil had to fold it "
                    "again each time. " + self._churn_advice(),
                )
            )
        if self.shadow_window_rows:
            out.append(
                (
                    f"Compression was spot-checked {self.shadow_window_rows} time"
                    f"{'s' if self.shadow_window_rows != 1 else ''} — "
                    f"{self.shadow_window_agree} matched",
                    "Shadow mode re-ran a sample of requests uncompressed in the "
                    "background and compared the answers, so you don't have to take the "
                    "savings on faith.",
                )
            )
        if self.detail_available and self.expand_resolved:
            out.append(
                (
                    f"The model asked for folded detail back {self.expand_resolved} time"
                    f"{'s' if self.expand_resolved != 1 else ''}",
                    "When a summary wasn't enough, the model used its recovery handle and "
                    "distil restored the original content mid-request — nothing was lost, "
                    "it just cost one extra round-trip.",
                )
            )
        return out

    def blocks_by_kind(self) -> list[tuple[str, int, int]]:
        """[(signature, unique_blocks, tokens)] biggest-token first."""
        agg: dict[str, list[int]] = {}
        for info in self.blocks.values():
            m = agg.setdefault(str(info.get("sig") or "?"), [0, 0])
            m[0] += 1
            m[1] += int(info.get("tokens") or 0)
        return sorted(((k, v[0], v[1]) for k, v in agg.items()), key=lambda t: -t[2])

    def top_blocks(self, n: int = 10) -> list[tuple[str, str, int, int, bool]]:
        """[(handle, sig, tokens, folds, recoverable)] biggest first."""
        rows = [
            (
                h,
                str(i.get("sig") or "?"),
                int(i.get("tokens") or 0),
                int(i.get("folds") or 0),
                bool(i.get("recoverable")),
            )
            for h, i in self.blocks.items()
        ]
        return sorted(rows, key=lambda t: -t[2])[:n]


def dissect(sid: str) -> Dissection:
    """Assemble a full Dissection for *sid* from every local source."""
    ledger_rows = [r for r in _read_jsonl(default_path()) if r.get("session") == sid]
    manifest: dict[str, Any] | None = None
    mp = session_manifest_path(sid)
    if mp is not None:
        try:
            loaded = json.loads(mp.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            manifest = None
    rp = session_requests_path(sid)
    requests = _read_jsonl(rp) if rp is not None else []

    marker = heartbeat = exit_note = None
    marker_p = session_marker_path(sid)
    if marker_p is not None:
        for attr, path in (
            ("marker", marker_p),
            ("heartbeat", marker_p.with_suffix(".hb")),
            ("exit_note", marker_p.with_suffix(".exit")),
        ):
            try:
                val = path.read_text(encoding="utf-8").strip()
            except OSError:
                val = None
            if attr == "marker":
                marker = val
            elif attr == "heartbeat":
                heartbeat = val
            else:
                exit_note = val

    d = Dissection(
        sid=sid,
        manifest=manifest,
        ledger_rows=ledger_rows,
        requests=requests,
        marker=marker,
        heartbeat=heartbeat,
        exit_note=exit_note,
    )

    restore_dir = _state_dir() / "restore"
    for rec in requests:
        for blk in rec.get("blocks") or []:
            h = blk.get("h")
            if not isinstance(h, str):
                continue
            info = d.blocks.setdefault(
                h, {"sig": blk.get("sig"), "tokens": int(blk.get("tokens") or 0), "folds": 0}
            )
            info["folds"] += 1
    for h, info in d.blocks.items():
        info["recoverable"] = (restore_dir / h).exists()

    if d.started and d.ended:
        from .shadow import ReplayCost, cost_delta

        costs: list[ReplayCost] = []
        for row in _read_jsonl(_state_dir() / "shadow.jsonl"):
            ts = float(row.get("ts") or 0.0)
            if d.started - 1 <= ts <= d.ended + 300:
                d.shadow_window_rows += 1
                d.shadow_window_agree += 1 if row.get("equivalent") else 0
                if all(k in row for k in ("in_a", "in_b", "out_a", "out_b")):
                    try:
                        costs.append(
                            ReplayCost(
                                model=row.get("model")
                                if isinstance(row.get("model"), str)
                                else None,
                                in_a=int(row["in_a"]),
                                in_b=int(row["in_b"]),
                                out_a=int(row["out_a"]),
                                out_b=int(row["out_b"]),
                            )
                        )
                    except (TypeError, ValueError):
                        pass
        cd = cost_delta(costs)
        if cd is not None:
            d.shadow_out_delta = cd.out_delta_mean
            d.shadow_net_usd = cd.net_usd if cd.priced else None
    return d


# --------------------------------------------------------------------------- render
def _human(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "—"


def render_sessions_text(sessions: list[SessionOverview], *, color: bool = True) -> str:
    """The no-argument picker: one line per known session, newest first."""

    def c(code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if color else s

    if not sessions:
        return (
            "No wrap sessions recorded yet. Start one with:\n"
            "  distil wrap -- claude   (or codex/gemini/any base-url-honoring tool)"
        )
    out = [c("1", "distil sessions — pick one to dissect"), ""]
    hdr = (
        f"{'#':>3}  {'session':<22} {'tool':<10} {'started':<17} {'last':<17} "
        f"{'reqs':>5} {'saved':>7}  status"
    )
    out.append(c("2", hdr))
    for i, o in enumerate(sessions, 1):
        pct = (
            100.0 * (o.baseline_tokens - o.distil_tokens) / o.baseline_tokens
            if o.baseline_tokens
            else 0.0
        )
        out.append(
            f"{i:>3}  {o.sid:<22} {(o.tool or '?'):<10} {_when(o.started):<17} "
            f"{_when(o.last_ts):<17} {o.requests:>5} {pct:>6.1f}%  {o.status}"
        )
    out += [
        "",
        "dissect one:  distil dissect <session>   (a number above, a unique prefix, or `latest`)",
    ]
    return "\n".join(out)


def _flags_line(man: dict[str, Any]) -> str:
    flags = man.get("flags") or {}
    on = [k for k in ("expand", "session_delta", "lossless_only", "verbatim") if flags.get(k)]
    if float(flags.get("shadow_rate") or 0.0) > 0:
        on.append(f"shadow={flags['shadow_rate']}")
    if (flags.get("shape_output") or "off") != "off":
        on.append(f"shape_output={flags['shape_output']}")
    return ", ".join(on) or "defaults"


# Census reason -> what it means to a human reading a report. Kept beside the renderer
# rather than in the adapter: the adapter names mechanisms, this names consequences.
_ELIGIBILITY_LABEL = {
    "assistant_text": "the model's own words (never rewritten)",
    "user_text": "your prompts (lossless only)",
    "tool_result_recent": "freshest tool output (kept byte-exact)",
    "tool_result_digested": "digested",
    "tool_result_html_stripped": "HTML chrome stripped",
    "tool_result_short": "too short to digest",
    "tool_result_verbatim": "verbatim mode",
    "tool_result_learned_keep": "learned keep-byte-exact",
    "tool_result_declined": "digester declined",
}

# Buckets that represent a deliberate protection rather than a missed opportunity.
# Distinguished so a report can say "working as designed" without the reader having to
# know which gate is which.
_PROTECTED_REASONS = frozenset(
    {"assistant_text", "tool_result_recent", "user_text", "tool_result_learned_keep"}
)


def render_text(
    d: Dissection,
    *,
    color: bool = True,
    peers: list[SessionOverview] | None = None,
    corr: "Correlation | None" = None,
) -> str:
    """Terminal report. Sections degrade gracefully when a source is missing."""

    def c(code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if color else s

    man = d.manifest or {}
    subscription = d.billing == "subscription"
    out: list[str] = []
    tool = man.get("tool") or "unknown tool"
    out.append(c("1", f"distil dissect — {d.sid} ({tool})"))

    # Session card
    dur = ""
    if d.started and d.ended and d.ended > d.started:
        dur = f"  ({(d.ended - d.started) / 60:.0f} min)"
    out.append(f"  window   {_when(d.started)} → {_when(d.ended)}{dur}")
    if man:
        out.append(f"  wrap     {' '.join(man.get('argv') or [])}  [{_flags_line(man)}]")
        out.append(f"  distil   v{man.get('distil_version', '?')}, billing: {d.billing}")
    else:
        out.append("  wrap     manifest not recorded (session predates dissect logging)")
    if d.exit_note:
        out.append(f"  exit     {d.exit_note}")
    elif d.marker == "0":
        out.append(
            c("33", "  status   marker=0 — wrapped but NO traffic went through distil (bypass?)")
        )
    elif d.marker == "1":
        out.append(f"  status   live (heartbeat: {d.heartbeat or '—'})")

    warnings = d.anomalies(peers)
    if warnings:
        out.append("")
        out.append(c("33;1", "worth your attention"))
        for w in warnings:
            out.append(c("33", f"  ⚠ {w}"))

    heads = d.headlines()
    if heads:
        out.append("")
        out.append(c("1", "what happened"))
        for title, body in heads:
            out.append(f"  • {title}")
            out.append(c("2", f"    {body}"))

    # Savings (ledger)
    out.append("")
    out.append(c("1", "savings (input tokens, booked 2xx only)"))
    if not d.ledger_rows:
        out.append("  no booked requests in the ledger for this session")
    else:
        note = " (notional — flat-rate plan)" if subscription else ""
        out.append(
            f"  {_human(d.baseline_tokens)} → {_human(d.distil_tokens)}"
            f"  ({d.pct_saved:.1f}% saved, ${d.dollars_saved:.2f}{note})"
        )
        for model, reqs, bt, dt in d.per_model():
            pct = 100.0 * (bt - dt) / bt if bt else 0.0
            # A flat 0.0% reads as a broken gate. When the traffic is all content the
            # policy protects (user prompts, assistant text), say so on the row.
            prot = d.protected_share(model)
            why = ""
            if pct < 1.0 and prot is not None and prot >= 99.0:
                why = "  (all protected content — nothing eligible to digest)"
            out.append(
                f"    {model:<34} {reqs:>4} req  {_human(bt):>8} → {_human(dt):>8}  {pct:>5.1f}%{why}"
            )

    # Request detail
    out.append("")
    out.append(c("1", "request detail"))
    if not d.detail_available:
        out.append(
            "  not recorded — per-request detail needs a wrap from this distil version or newer"
        )
    else:
        n = len(d.requests)
        out.append(
            f"  {n} proxied requests: {n - d.unbooked_requests} booked, {d.unbooked_requests} not booked "
            f"(non-2xx/retry), {d.verbatim_requests} verbatim (nothing worth compressing)"
        )
        saved = d.tokens_saved_total
        if saved:
            dig_pct = 100.0 * d.digest_saved / saved
            out.append(
                f"  savings by mechanism: digest folds {_human(d.digest_saved)} ({dig_pct:.0f}%), "
                f"cache-delta {_human(d.delta_tokens_saved)} ({100 - dig_pct:.0f}%)"
            )
        out.append(
            f"  fixed overhead (not optimizable): system prompt + tool definitions = "
            f"{_human(d.overhead_tokens_total)} tokens, {d.overhead_share:.0f}% of everything sent "
            f"(~{_human(d.overhead_tokens_avg)}/request) — trimming unused tools beats compression here"
        )
        tools = d.tool_costs()
        if tools:
            out.append(
                f"  tool definitions ({len(tools)}): "
                + ", ".join(f"{name} {_human(per)}/req" for name, per, _t in tools[:5])
                + (" …" if len(tools) > 5 else "")
            )
        # Why the savings number is what it is. Printed next to the number itself so a
        # low percentage is read as a diagnosis rather than a malfunction: content the
        # policy protects looks identical to content the compressor failed on, unless
        # the two are named separately.
        elig = d.eligibility
        if elig:
            out.append(
                "  why: "
                + ", ".join(f"{_ELIGIBILITY_LABEL.get(r, r)} {p:.0f}%" for r, _t, p in elig[:5])
            )
            protected = sum(p for r, _t, p in elig if r in _PROTECTED_REASONS)
            declined = sum(p for r, _t, p in elig if r == "tool_result_declined")
            if declined >= 5.0:
                out.append(
                    c(
                        "33",
                        f"    {declined:.0f}% reached the digester and came back unchanged — "
                        f"that share is the compressor's to explain, not policy's",
                    )
                )
            elif protected >= 50.0:
                out.append(
                    f"    {protected:.0f}% is content distil deliberately protects (the model's "
                    f"own words, and your freshest tool output) — low savings here is the "
                    f"design holding, not a fault"
                )
        growth = d.system_growth()
        if growth and growth[1] != growth[0]:
            out.append(
                f"  system prompt: {_human(growth[0])} → {_human(growth[1])} tokens over the session"
            )
        if d.churn_tokens:
            out.append(
                f"  re-fold churn: {_human(d.churn_tokens)} tokens re-digested after first sight "
                f"({d.churned_blocks} block{'s' if d.churned_blocks != 1 else ''} re-folded — "
                "cache-delta/codec candidates)"
            )
        cal = d.calibration()
        if cal is not None:
            est, billed = cal
            out.append(
                f"  billed usage (from API responses): {_human(d.usage_input_total)} in / "
                f"{_human(d.usage_output_total)} out over {d.usage_requests} requests; "
                f"estimate {_human(est)} vs billed {_human(billed)} "
                f"(x{est / billed:.2f})"
                if billed
                else ""
            )
        else:
            out.append("  billed usage: not captured (older records or non-usage responses)")
        # The cache contract (ADR 0008), as the provider scored it. distil promises the
        # prefix it forwards is byte-stable across turns; this is the only number that
        # says whether the provider agreed. A near-zero share on a long session means
        # something rewrote the prefix — the failure is silent otherwise, because the
        # requests all still succeed, they just cost ~2x.
        share = d.cached_input_share
        if share is None:
            out.append("  cache-read share: not captured (no cache fields in these records)")
        else:
            out.append(
                f"  cache-read share: {share:.1f}% of billed input was served from the "
                "provider's prompt cache (at ~0.1x)"
            )
        if d.billing == "subscription" and d.headroom_multiplier > 1:
            out.append(
                f"  flat-rate headroom: the same context budget went ~{d.headroom_multiplier:.1f}x "
                "further than unwrapped"
            )
        lat = d.latency_by_path()
        if lat:
            out.append(
                "  latency: " + ", ".join(f"{k} {n} req @ {ms / 1000:.1f}s avg" for k, n, ms in lat)
            )
            if d.forced_buffered:
                out.append(
                    f"  note: {d.forced_buffered} streamed requests were fully buffered so the "
                    "expand loop could inspect them — that is the --expand time-to-first-token tax"
                )
        if d.blocks:
            out.append("")
            out.append(c("1", "digested blocks (content-free: kind:size, tokens)"))
            for sig, uniq, toks in d.blocks_by_kind():
                out.append(f"    {sig:<12} {uniq:>4} blocks  {_human(toks):>8} tokens")
            recoverable = sum(1 for i in d.blocks.values() if i.get("recoverable"))
            out.append(f"  recoverable now: {recoverable}/{len(d.blocks)} blocks still in restore/")
            out.append("  largest folds:")
            for h, sig, toks, folds, rec in d.top_blocks(5):
                mark = "✓" if rec else "expired"
                out.append(
                    f"    {h}  {sig:<12} {_human(toks):>8} tokens  ×{folds} requests  [{mark}]"
                )

    # Quality loops
    out.append("")
    out.append(c("1", "quality loops"))
    if d.detail_available:
        out.append(
            f"  expand: {d.expand_resolved} requests had distil_expand calls resolved in-proxy"
        )
        if d.expand_missed:
            out.append(f"    unrecoverable: {d.expand_missed} handle(s) asked for and not returned")
        for sig, exp, total in d.expansion_regret():
            out.append(
                f"    regret: {sig} blocks pulled back {exp}/{total} — folding this kind "
                "costs a round-trip more than it saves"
            )
        out.append(f"  shadow: {d.shadow_sampled} requests sampled for decision-equivalence")
    if d.shadow_window_rows:
        out.append(
            f"  shadow verdicts in this session's time window (time-joined, not session-tagged): "
            f"{d.shadow_window_agree}/{d.shadow_window_rows} equivalent"
        )
    if d.shadow_out_delta is not None:
        # The compression paradox: a shorter prompt that buys a longer answer can
        # cost more than it saved, because output is priced several times input.
        line = f"  shadow output tokens (compressed − original): {d.shadow_out_delta:+.1f}/request"
        if d.shadow_net_usd is not None:
            line += f" · net after output ${d.shadow_net_usd:+.4f}"
        out.append(line)
    elif not d.detail_available:
        out.append("  no session-scoped signal recorded for this session")

    if corr is not None:
        out.append("")
        out.append(
            c(
                "1",
                f"conversation correlation ({corr.agent} transcript: {corr.label or 'untitled'})",
            )
        )
        if corr.fold_sources:
            out.append("  largest folds, named:")
            for s in corr.fold_sources[:5]:
                who = s.tool or "unknown tool"
                turn = f' (turn {s.turn}: "{s.turn_text[:40]}…")' if s.turn_text else ""
                out.append(f"    {_human(s.tokens):>8} ×{s.folds}  {who} output{turn}")
        if corr.unnamed_blocks:
            out.append(
                f"  {corr.unnamed_blocks} blocks unattributed (restore blob expired or content transformed)"
            )
        if corr.unused_tools:
            out.append(
                f"  unused tools ({len(corr.unused_tools)}/{corr.tools_defined} defined, "
                f"{_human(corr.unused_tokens_per_request)} tokens/request paid for nothing): "
                + ", ".join(n for n, _t in corr.unused_tools[:8])
                + (" …" if len(corr.unused_tools) > 8 else "")
            )
        for s in corr.refetched[:3]:
            out.append(
                c(
                    "33",
                    f"  ⚠ re-fetched after fold: {s.tool or '?'} output ({_human(s.tokens)} tokens) "
                    f"appeared in {s.refetches} separate results — the digest may have dropped "
                    "something the agent needed",
                )
            )
        if corr.turns:
            out.append("  costliest turns:")
            for t in corr.turns[:3]:
                label = (
                    t.text[:46] + ("…" if len(t.text) > 46 else "") if t.text else "(session start)"
                )
                out.append(
                    f'    turn {t.index}: "{label}" — {t.requests} req, '
                    f"{_human(t.baseline_tokens)} baseline, {_human(t.saved_tokens)} saved"
                )

    out.append("")
    out.append(
        c(
            "2",
            "terms: fold = bulky content replaced by a summary + recovery handle · cache-delta =",
        )
    )
    out.append(
        c("2", "resent content replaced by a reference · verbatim = passed through untouched ·")
    )
    out.append(c("2", "unbooked = upstream failed/retried, not counted as savings"))
    out.append("")
    out.append(
        c(
            "2",
            "sources: savings.jsonl, sessions/<sid>{.json,.requests.jsonl,.hb,.exit}, restore/, shadow.jsonl",
        )
    )
    out.append(
        c(
            "2",
            "retention: session detail follows the sessions/ TTL sweep; restore blobs are pruned separately",
        )
    )
    return "\n".join(out)


def to_json(
    d: Dissection,
    peers: list[SessionOverview] | None = None,
    corr: "Correlation | None" = None,
) -> dict[str, Any]:
    """Machine-readable dissection (same numbers the text/html reports show)."""
    cal = d.calibration()
    correlation = None
    if corr is not None:
        correlation = {
            "agent": corr.agent,
            "label": corr.label,
            "transcript": corr.path,
            "fold_sources": [
                {
                    "handle": s.handle,
                    "sig": s.sig,
                    "tokens": s.tokens,
                    "folds": s.folds,
                    "tool": s.tool,
                    "turn": s.turn,
                    "refetches": s.refetches,
                }
                for s in corr.fold_sources
            ],
            "unnamed_blocks": corr.unnamed_blocks,
            "tools": {
                "defined": corr.tools_defined,
                "invoked": corr.tools_invoked,
                "unused": [{"name": n, "tokens_per_request": t} for n, t in corr.unused_tools],
            },
            "refetched_after_fold": [
                {"tool": s.tool, "tokens": s.tokens, "refetches": s.refetches}
                for s in corr.refetched
            ],
            "turns": [
                {
                    "index": t.index,
                    "text": t.text,
                    "requests": t.requests,
                    "baseline_tokens": t.baseline_tokens,
                    "saved_tokens": t.saved_tokens,
                }
                for t in corr.turns
            ],
        }
    return {
        "correlation": correlation,
        "headlines": [{"headline": h, "detail": b} for h, b in d.headlines()],
        "insights": {
            "mechanism": {
                "digest_tokens_saved": d.digest_saved,
                "delta_tokens_saved": d.delta_tokens_saved,
            },
            "overhead": {
                "tokens_total": d.overhead_tokens_total,
                "share_pct": round(d.overhead_share, 1),
                "system_tokens_avg": d.system_tokens_avg,
                "tools_tokens_avg": d.tools_tokens_avg,
                "system_growth": d.system_growth(),
                "tools": [
                    {"name": n, "tokens_per_request": p, "session_tokens": t}
                    for n, p, t in d.tool_costs()
                ],
            },
            "churn": {"tokens": d.churn_tokens, "blocks": d.churned_blocks},
            # Why the number is what it is. [] means the session predates the census,
            # which is NOT the same as "nothing was eligible" — do not render it as 0.
            "eligibility": [
                {"reason": r, "tokens": t, "share_pct": round(p, 1)} for r, t, p in d.eligibility
            ],
            "usage": {
                "input_tokens": d.usage_input_total,
                "output_tokens": d.usage_output_total,
                "requests_with_usage": d.usage_requests,
                # None (not 0.0) when the records carry no cache fields — "we did not
                # measure this" and "the cache never hit" are opposite diagnoses.
                "cache_read_share_pct": (
                    None if d.cached_input_share is None else round(d.cached_input_share, 1)
                ),
                "calibration": (
                    {"estimated": cal[0], "billed": cal[1]} if cal is not None else None
                ),
            },
            "headroom_multiplier": round(d.headroom_multiplier, 2),
            "latency_by_path": [
                {"path": k, "requests": n, "avg_ms": ms} for k, n, ms in d.latency_by_path()
            ],
            "forced_buffered_requests": d.forced_buffered,
            "expansion_regret": [
                {"sig": s, "expanded": e, "blocks": t} for s, e, t in d.expansion_regret()
            ],
            "anomalies": d.anomalies(peers),
        },
        "session": d.sid,
        "manifest": d.manifest,
        "window": {"started_ts": d.started, "ended_ts": d.ended},
        "savings": {
            "baseline_input_tokens": d.baseline_tokens,
            "distil_input_tokens": d.distil_tokens,
            "pct_saved": round(d.pct_saved, 2),
            "dollars_saved": round(d.dollars_saved, 4),
            "dollars_notional": d.billing == "subscription",
            "per_model": [
                {"model": m, "requests": r, "baseline_tokens": b, "distil_tokens": t}
                for m, r, b, t in d.per_model()
            ],
        },
        "detail_available": d.detail_available,
        "requests": {
            "total": len(d.requests),
            "unbooked": d.unbooked_requests,
            "verbatim": d.verbatim_requests,
            "delta_tokens_saved": d.delta_tokens_saved,
            "overhead_tokens_avg": d.overhead_tokens_avg,
        },
        "blocks": {
            "by_kind": [{"sig": s, "unique": u, "tokens": t} for s, u, t in d.blocks_by_kind()],
            "recoverable": sum(1 for i in d.blocks.values() if i.get("recoverable")),
            "unique": len(d.blocks),
            "top": [
                {"handle": h, "sig": s, "tokens": t, "folds": f, "recoverable": r}
                for h, s, t, f, r in d.top_blocks()
            ],
        },
        "quality": {
            "expand_resolved_requests": d.expand_resolved,
            "shadow_sampled_requests": d.shadow_sampled,
            "shadow_window_rows": d.shadow_window_rows,
            "shadow_window_agree": d.shadow_window_agree,
            "shadow_output_token_delta": d.shadow_out_delta,
            "shadow_net_usd_after_output": d.shadow_net_usd,
        },
        "liveness": {"marker": d.marker, "heartbeat": d.heartbeat, "exit": d.exit_note},
    }


# ---------------------------------------------------------------- svg charts
# Categorical series, fixed order, validated (dataviz six checks) against the
# report's dark card surface #0b0d15: overhead / sent / saved.
_C_OVERHEAD, _C_SENT, _C_SAVED = "#3987e5", "#199e70", "#c98500"
_GRID = "#1b2030"
_INK_MUTED = "#9aa1b3"


def _svg_hbars(
    rows: list[tuple[str, int, list[tuple[str, str, str]]]],
    *,
    color: str = _C_OVERHEAD,
    name: str = "Bar chart",
) -> str:
    """Horizontal bar chart (single measure, one hue): label, thin bar, value.

    Each row is (label, value, tooltip_rows). The whole row — label through
    value — is one oversized hit target for the styled tooltip. ``name`` is
    the chart's accessible name (SC 1.1.1): it is rendered both as the
    ``aria-label`` and as an ``<svg><title>`` so it survives in AT that
    ignores one or the other.
    """
    if not rows:
        return ""
    top = max(v for _, v, _t in rows) or 1
    rh, bar_h, label_w, val_w, width = 26, 12, 220, 64, 640
    plot_w = width - label_w - val_w
    parts = [
        f'<svg viewBox="0 0 {width} {len(rows) * rh + 6}" role="img" '
        f'aria-label="{_html.escape(name, quote=True)}" '
        f'style="width:100%;height:auto;font:12px Inter,ui-sans-serif,sans-serif">'
        f"<title>{_html.escape(name)}</title>"
    ]
    for i, (label, val, tip_rows) in enumerate(rows):
        y = i * rh + 4
        w = max(2, round(plot_w * val / top))
        name = label if len(label) <= 30 else label[:29] + "…"
        parts.append(
            f"<g{_tip_attr(label, tip_rows)}>"
            f'<rect x="0" y="{y - 2}" width="{width}" height="{rh}" fill="transparent"/>'
            f'<text x="{label_w - 8}" y="{y + bar_h - 2}" text-anchor="end" '
            f'fill="{_INK_MUTED}">{_html.escape(name)}</text>'
            f'<rect class="mark" x="{label_w}" y="{y}" width="{w}" height="{bar_h}" '
            f'rx="2" fill="{color}"/>'
            f'<text x="{label_w + w + 6}" y="{y + bar_h - 2}" fill="{_INK_MUTED}">'
            f"{_human(val)}</text></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _svg_stack_timeline(requests: list[dict[str, Any]]) -> str:
    """Per-request composition: overhead + sent content + saved, stacked bars.

    2px surface gaps between bars and between segments; rounded data-end on the
    top segment only; native <title> tooltips carry the per-request numbers.
    """
    if not requests:
        return ""
    comp = []
    for r in requests:
        overhead = int(r.get("overhead_tokens") or 0)
        saved = int(r.get("tokens_saved") or 0)
        sent = max(0, int(r.get("compressible_tokens") or 0) - saved)
        comp.append((overhead, sent, saved, r))
    top = max(o + s + v for o, s, v, _ in comp) or 1
    width, height, pad_l, pad_b = 640, 190, 52, 18
    plot_w, plot_h = width - pad_l - 8, height - pad_b - 8
    n = len(comp)
    step = plot_w / n
    bar_w = max(2, min(28, round(step - 2)))
    total_saved = sum(saved for _o, _s, saved, _r in comp)
    name = (
        f"Per-request token composition for {n} requests: stacked overhead, sent content, "
        f"and tokens saved, {_human(total_saved)} saved in total"
    )
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_html.escape(name, quote=True)}" '
        f'style="width:100%;height:auto;font:11px Inter,ui-sans-serif,sans-serif">'
        f"<title>{_html.escape(name)}</title>"
    ]
    for frac in (0.0, 0.5, 1.0):  # recessive grid, three lines
        y = 8 + plot_h * (1 - frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - 8}" y2="{y:.1f}" '
            f'stroke="{_GRID}" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end" '
            f'fill="{_INK_MUTED}">{_human(top * frac)}</text>'
        )
    for i, (overhead, sent, saved, r) in enumerate(comp):
        x = pad_l + i * step + (step - bar_w) / 2
        tip_rows = [
            (_C_SAVED, "saved by distil", f"{saved:,}"),
            (_C_SENT, "sent content", f"{sent:,}"),
            (_C_OVERHEAD, "overhead", f"{overhead:,}"),
        ]
        billed = r.get("usage_input_tokens")
        if billed is not None:
            billed = int(billed) + int(r.get("usage_cache_tokens") or 0)
            tip_rows.append(("", "billed input", f"{billed:,}"))
        if r.get("duration_ms") is not None:
            tip_rows.append(("", "duration", f"{int(r['duration_ms']) / 1000:.1f}s"))
        y = 8 + plot_h
        segs = [(_C_OVERHEAD, overhead), (_C_SENT, sent), (_C_SAVED, saved)]
        drawn = [(c, v) for c, v in segs if v > 0]
        # The whole column is one hit target: hover anywhere over the request,
        # not just the thin bar, and every series is listed in one tooltip.
        parts.append(
            f"<g{_tip_attr(f'request {i + 1} · ' + str(r.get('model', '?')), tip_rows)}>"
            f'<rect x="{pad_l + i * step:.1f}" y="8" width="{step:.1f}" '
            f'height="{plot_h + pad_b:.1f}" fill="transparent"/>'
        )
        for j, (color, val) in enumerate(drawn):
            h = max(1.0, plot_h * val / top)
            y -= h
            topmost = j == len(drawn) - 1
            # 2px surface gap between segments, shaved off each non-top segment's
            # top edge — bottoms stay anchored (baseline for the first segment).
            gap = 0.0 if topmost else min(2.0, h - 0.5)
            parts.append(
                f'<rect class="mark" x="{x:.1f}" y="{y + gap:.1f}" width="{bar_w}" '
                f'height="{h - gap:.1f}" rx="{2 if topmost else 0}" fill="{color}"/>'
            )
        parts.append("</g>")
    parts.append(
        f'<text x="{pad_l}" y="{height - 4}" fill="{_INK_MUTED}">1</text>'
        f'<text x="{width - 8}" y="{height - 4}" text-anchor="end" '
        f'fill="{_INK_MUTED}">{n}</text>'
        f'<text x="{pad_l + plot_w / 2:.0f}" y="{height - 4}" text-anchor="middle" '
        f'fill="{_INK_MUTED}">request</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _legend() -> str:
    items = (
        (_C_OVERHEAD, "overhead (system + tools)"),
        (_C_SENT, "sent content"),
        (_C_SAVED, "saved by distil"),
    )
    spans = "".join(
        f'<span style="margin-right:16px"><span style="display:inline-block;width:10px;'
        f'height:10px;border-radius:2px;background:{c};margin-right:6px"></span>{t}</span>'
        for c, t in items
    )
    return f'<p class="muted" style="font-size:12.5px">{spans}</p>'


def _timeline_table(requests: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{_html.escape(str(r.get('model', '?')))}</td>"
        f"<td class='r'>{int(r.get('overhead_tokens') or 0):,}</td>"
        f"<td class='r'>{max(0, int(r.get('compressible_tokens') or 0) - int(r.get('tokens_saved') or 0)):,}</td>"
        f"<td class='r'>{int(r.get('tokens_saved') or 0):,}</td>"
        f"<td class='r'>{r.get('usage_input_tokens') if r.get('usage_input_tokens') is not None else '—'}</td>"
        f"<td class='r'>{r.get('duration_ms') if r.get('duration_ms') is not None else '—'}</td></tr>"
        for i, r in enumerate(requests)
    )
    return (
        "<details><summary class='muted'>data table</summary>"
        "<table><caption class='sr-only'>Per-request token timeline</caption>"
        "<tr><th scope='col'>#</th><th scope='col'>model</th><th scope='col'>overhead</th>"
        "<th scope='col'>sent</th>"
        f"<th scope='col'>saved</th><th scope='col'>billed in</th><th scope='col'>ms</th></tr>"
        f"{rows}</table></details>"
    )


def _tool_cost_table(tool_costs: list[tuple[str, int, int]]) -> str:
    """Data-table fallback for the tool-definition-cost chart, mirroring
    :func:`_timeline_table` so the same numbers are available without a chart."""
    rows = "".join(
        f"<tr><td>{_html.escape(name)}</td><td class='r'>{per:,}</td><td class='r'>{tot:,}</td></tr>"
        for name, per, tot in tool_costs
    )
    return (
        "<details><summary class='muted'>data table</summary>"
        "<table><tr><th>tool</th><th>tokens per request</th>"
        f"<th>session total</th></tr>{rows}</table></details>"
    )


def _tip_attr(title: str, rows: list[tuple[str, str, str]] | None = None, body: str = "") -> str:
    """Build the ``data-tip`` attribute for the styled hover layer.

    Payload is JSON — the page's script rebuilds it with textContent (never
    innerHTML), so model/tool/signature names stay data, not markup. ``rows``
    are (series_color, label, value) triples; values render strong, labels
    secondary, keyed by a short stroke of the series color.

    Also wires ``aria-describedby="tip"`` so a screen reader announces the
    same explanation a sighted mouse or keyboard user sees in the floating
    tooltip (the shared ``#tip`` element rendered once per page).
    """
    payload: dict[str, Any] = {"t": title}
    if rows:
        payload["rows"] = [list(r) for r in rows]
    if body:
        payload["body"] = body
    return (
        f' data-tip="{_html.escape(json.dumps(payload, ensure_ascii=False), quote=True)}"'
        ' tabindex="0" aria-describedby="tip"'
    )


def _tile(label: str, value: str, note: str = "", help_text: str = "") -> str:
    """One stat card. ``help_text`` feeds the styled hover/focus tooltip in
    plain English — every distil term on the page must be explainable without
    docs."""
    note_html = f'<div class="n">{note}</div>' if note else ""
    tip = _tip_attr(label, body=help_text) if help_text else ""
    return (
        f'<div class="card tile"{tip}><div class="l">{label}</div>'
        f'<div class="v2">{value}</div>{note_html}</div>'
    )


def render_html(
    d: Dissection,
    peers: list[SessionOverview] | None = None,
    corr: "Correlation | None" = None,
) -> str:
    """Self-contained dark page in the ledger `render_html` style.

    Layout follows the stat-tile pattern: one number per card with a short
    sub-note, anomalies in a bordered callout, and one observation per line —
    never a paragraph of run-together figures.
    """
    man = d.manifest or {}
    subscription = d.billing == "subscription"
    e = _html.escape
    dol_note = " (notional — flat-rate plan)" if subscription else ""

    model_rows = (
        "".join(
            f"<tr><td>{e(m)}</td><td class='r'>{r}</td><td class='r'>{_human(b)}</td>"
            f"<td class='r'>{_human(t)}</td><td class='r'>{100.0 * (b - t) / b if b else 0.0:.1f}%</td></tr>"
            for m, r, b, t in d.per_model()
        )
        or "<tr><td class='muted' colspan='5'>no booked requests</td></tr>"
    )
    kind_rows = "".join(
        f"<tr><td>{e(s)}</td><td class='r'>{u}</td><td class='r'>{_human(t)}</td></tr>"
        for s, u, t in d.blocks_by_kind()
    )
    top_rows = "".join(
        f"<tr><td><code>{e(h)}</code></td><td>{e(s)}</td><td class='r'>{_human(t)}</td>"
        f"<td class='r'>×{f}</td><td>{'recoverable' if r else '<span class=muted>expired</span>'}</td></tr>"
        for h, s, t, f, r in d.top_blocks()
    )

    warn_html = "".join(f"<li>{e(w)}</li>" for w in d.anomalies(peers))
    warn_card = (
        f'<div class="callout"><b>⚠ Worth your attention</b>'
        f'<div class="n">Automatic checks on this session found things that may need '
        f'action.</div><ul class="warn">{warn_html}</ul></div>'
        if warn_html
        else ""
    )

    detail_body = ""
    if d.detail_available:
        saved = d.tokens_saved_total or 1
        tiles = [
            _tile(
                "Requests",
                str(len(d.requests)),
                f"{d.unbooked_requests} unbooked · {d.verbatim_requests} verbatim",
                "API calls the wrapped tool made through distil. Unbooked = the upstream "
                "call failed or was retried, so it is not counted as savings. Verbatim = "
                "the request was passed through untouched because nothing in it was worth "
                "compressing.",
            ),
            _tile(
                "Digest folds",
                _human(d.digest_saved),
                f"{100.0 * d.digest_saved / saved:.0f}% of savings",
                "Tokens distil avoided sending by replacing bulky content (test logs, file "
                "dumps, stack traces) with a short summary plus a recovery handle. The "
                "original bytes stay on this machine and can be restored at any time.",
            ),
            _tile(
                "Cache-delta",
                _human(d.delta_tokens_saved),
                f"{100.0 * d.delta_tokens_saved / saved:.0f}% of savings",
                "Tokens avoided by sending a short reference to content this session "
                "already sent earlier, instead of resending the full text every request.",
            ),
            _tile(
                "Fixed overhead",
                _human(d.overhead_tokens_total),
                f"{d.overhead_share:.0f}% of sent · system {_human(d.system_tokens_avg)}/req "
                f"· tools {_human(d.tools_tokens_avg)}/req",
                "The system prompt and tool definitions are resent word-for-word with "
                "every single request, and distil never alters them. If this share is "
                "large, removing unused tools/MCP servers saves more than compression can.",
            ),
            _tile(
                "Re-fold churn",
                _human(d.churn_tokens),
                f"{d.churned_blocks} block{'s' if d.churned_blocks != 1 else ''} re-folded",
                "The same content arrived again on later requests and had to be folded "
                "again. High churn means smarter caching (cache-delta / the learned codec) "
                "has room to help.",
            ),
        ]
        cal = d.calibration()
        if cal is not None and cal[1]:
            tiles.append(
                _tile(
                    "Billed input",
                    _human(d.usage_input_total),
                    f"{_human(d.usage_output_total)} out · estimate ×{cal[0] / cal[1]:.2f} of billed",
                    "Token counts reported by the API itself in its responses — the ground "
                    "truth. The × figure is how close distil's local estimate came to it "
                    "(×1.00 would be exact).",
                )
            )
        else:
            tiles.append(
                _tile(
                    "Billed input",
                    "—",
                    "usage not captured",
                    "The API reports exact token counts in its responses; none were "
                    "captured for this session (recorded by newer wraps only).",
                )
            )
        if subscription and d.headroom_multiplier > 1:
            tiles.append(
                _tile(
                    "Headroom",
                    f"{d.headroom_multiplier:.1f}×",
                    "flat-rate: context budget stretched",
                    "On a flat-rate subscription there is no per-token bill — the real win "
                    "is that the same rate/context limits go this many times further "
                    "before you hit them.",
                )
            )
        tiles.append(
            _tile(
                "Expand",
                str(d.expand_resolved),
                "requests resolved in-proxy",
                "Times the model asked for a folded block back (it kept a recovery handle) "
                "and distil answered from local storage, invisibly to the session.",
            )
        )
        shadow_note = (
            f"{d.shadow_window_agree}/{d.shadow_window_rows} verdicts equivalent"
            if d.shadow_window_rows
            else "no verdicts in window"
        )
        tiles.append(
            _tile(
                "Shadow",
                str(d.shadow_sampled),
                f"sampled · {shadow_note}",
                "A sample of requests is re-run uncompressed in the background and the "
                "two answers compared — a live check that compression is not changing "
                "the model's decisions.",
            )
        )

        notes: list[str] = []
        growth = d.system_growth()
        if growth and growth[1] != growth[0]:
            notes.append(
                f"System prompt grew {_human(growth[0])} → {_human(growth[1])} tokens over the "
                "session — memory/context injections accumulate there."
            )
        lat = d.latency_by_path()
        if lat:
            notes.append(
                "Latency: " + " · ".join(f"{k} {n} req @ {ms / 1000:.1f}s" for k, n, ms in lat)
            )
        if d.forced_buffered:
            notes.append(
                f"{d.forced_buffered} streamed request"
                f"{'s' if d.forced_buffered != 1 else ''} buffered for the expand loop — "
                "the --expand time-to-first-token tax."
            )
        for s, x, t in d.expansion_regret():
            notes.append(
                f"Expansion regret: {e(s)} blocks pulled back {x}/{t} — folding this kind "
                "costs a round-trip more than it saves."
            )
        notes_html = (
            "<ul class='notes'>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>"
            if notes
            else ""
        )

        tool_costs = d.tool_costs()[:10]
        tool_rows = [
            (
                name,
                per,
                [
                    (_C_OVERHEAD, "tokens per request", f"{per:,}"),
                    ("", "session total (× every request)", f"{tot:,}"),
                ],
            )
            for name, per, tot in tool_costs
        ]
        tools_chart = (
            "<h2>Tool definitions <span class='muted'>(tokens per request)</span></h2>"
            "<p class='desc'>Every enabled tool is re-described to the model on every "
            "request, so each bar is a standing cost you pay per call. Long bars from "
            "tools you rarely use are the cheapest tokens to reclaim — disable them.</p>"
            f"{_svg_hbars(tool_rows, name='Tool definition cost, tokens per request, top 10 tools')}"
            f"{_tool_cost_table(tool_costs)}"
            if tool_rows
            else ""
        )
        kind_rows_chart = d.blocks_by_kind()[:10]
        kind_chart = _svg_hbars(
            [
                (
                    sig,
                    toks,
                    [
                        (_C_SAVED, "tokens folded", f"{toks:,}"),
                        ("", "distinct blocks", str(uniq)),
                    ],
                )
                for sig, uniq, toks in kind_rows_chart
            ],
            color=_C_SAVED,
            name="Tokens folded by block kind, top 10 kinds",
        )
        detail_body = f"""<h2>Request detail</h2>
<p class="desc">Every API request this session made, and where its tokens went — hover any
card for what the term means.</p>
<div class="grid">{"".join(tiles)}</div>
{notes_html}
<h2>Request composition <span class="muted">(tokens per request)</span></h2>
<p class="desc">One bar per request, in order. Blue is the fixed overhead every request must
carry, green is conversation content that was actually sent, yellow is what distil kept off
the wire. Hover a bar for exact numbers.</p>
{_legend()}
{_svg_stack_timeline(d.requests)}
{_timeline_table(d.requests)}
{tools_chart}
<h2>Digested blocks <span class="muted">(what got folded)</span></h2>
<p class="desc">Content distil summarized, grouped by what it looked like. The label is
kind:size — e.g. <code>log:l</code> is a large log, <code>prose:m</code> a medium block of
text. Only these labels are stored, never the content.</p>
{kind_chart}
<table><caption class="sr-only">Folded block kinds</caption>
<tr><th scope="col">kind</th><th scope="col">blocks</th><th scope="col">tokens</th></tr>
{kind_rows}</table>
<h2>Largest folds</h2>
<p class="desc">The biggest single blocks distil summarized. The handle is the short ID the
model can use to ask for the original back; “recoverable” means those original bytes are
still on this machine.</p>
<table><caption class="sr-only">Largest folded blocks</caption>
<tr><th scope="col">handle</th><th scope="col">kind</th><th scope="col">tokens</th>
<th scope="col"{_tip_attr("seen", body="How many requests carried this block.")}>seen</th>
<th scope="col"{_tip_attr("restore", body="Is the original still on disk (restore/)?")}>restore</th></tr>
{top_rows}</table>"""
    else:
        detail_body = (
            "<h2>Request detail</h2><p class='muted'>Not recorded — per-request detail needs a "
            "wrap from this distil version or newer.</p>"
        )
    exit_line = f"<p class='muted'>exit: {e(d.exit_note)}</p>" if d.exit_note else ""
    expand_miss_html = f" · <b>{d.expand_missed} unrecoverable</b>" if d.expand_missed else ""
    corr_html = ""
    if corr is not None:
        fold_rows = "".join(
            f"<tr><td>{e(s.tool or 'unknown')}</td>"
            f"<td class='muted'>{e((s.turn_text or '')[:48])}</td>"
            f"<td class='r'>{_human(s.tokens)}</td><td class='r'>×{s.folds}</td>"
            f"<td class='r'>{s.refetches}</td></tr>"
            for s in corr.fold_sources[:10]
        )
        unused = (
            f"<p><b>{len(corr.unused_tools)}</b> of {corr.tools_defined} defined tools were "
            f"never invoked, costing <b>{_human(corr.unused_tokens_per_request)}</b> tokens on "
            f"every request: <span class='muted'>"
            + e(", ".join(n for n, _t in corr.unused_tools[:10]))
            + ("…" if len(corr.unused_tools) > 10 else "")
            + "</span></p>"
            if corr.unused_tools
            else f"<p>All {corr.tools_defined} defined tools were invoked at least once.</p>"
        )
        refetch = "".join(
            f"<li>{e(s.tool or '?')} output ({_human(s.tokens)} tokens) reappeared in "
            f"{s.refetches} separate tool results — the digest may have dropped something "
            "the agent needed</li>"
            for s in corr.refetched[:3]
        )
        refetch_html = f"<ul class='warn'>{refetch}</ul>" if refetch else ""
        turn_rows = "".join(
            f"<tr><td>{t.index or '—'}</td><td class='muted'>{e(t.text[:56]) or '(session start)'}</td>"
            f"<td class='r'>{t.requests}</td><td class='r'>{_human(t.baseline_tokens)}</td>"
            f"<td class='r'>{_human(t.saved_tokens)}</td></tr>"
            for t in corr.turns[:8]
        )
        corr_html = f"""<h2>Conversation correlation <span class="muted">({e(corr.agent)}:
{e(corr.label or "untitled")})</span></h2>
<p class="desc">Opt-in join with the agent's own transcript (read locally, stored nowhere):
folds named by the tool call that produced them, tools you pay for but never used, and what
each of your prompts cost.</p>
{unused}
{refetch_html}
<h3>Largest folds, named</h3>
<table><caption class="sr-only">Largest folds by source</caption>
<tr><th scope="col">source</th><th scope="col">under turn</th><th scope="col">tokens</th>
<th scope="col">folds</th><th scope="col">results seen in</th></tr>
{fold_rows or "<tr><td class='muted' colspan='5'>no blocks could be attributed</td></tr>"}</table>
<h3>Costliest turns</h3>
<table><caption class="sr-only">Token usage by turn</caption>
<tr><th scope="col">#</th><th scope="col">your prompt</th><th scope="col">req</th>
<th scope="col">baseline</th><th scope="col">saved</th></tr>
{turn_rows or "<tr><td class='muted' colspan='5'>no turns matched</td></tr>"}</table>"""
    heads = d.headlines()
    story = (
        "<h2>What happened</h2>"
        "<p class='desc'>The session in plain language — details and charts below.</p>"
        + "".join(
            f'<div class="story"><b>{e(h)}</b><div class="n">{e(body)}</div></div>'
            for h, body in heads
        )
        if heads
        else ""
    )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="icon" type="image/svg+xml" href="https://dshakes.github.io/distil/assets/logo.svg"/>
<title>Distil — dissect {e(d.sid)}</title><style>
body{{margin:0;background:#06070b;color:#f2f3f7;font:15px/1.6 Inter,ui-sans-serif,sans-serif;
 -webkit-font-smoothing:antialiased}}
.wrap{{max-width:820px;margin:0 auto;padding:48px 24px}}
h1{{font-size:30px;font-weight:800;letter-spacing:-.02em;margin:0 0 6px}}
h2{{font-size:17px;font-weight:700;margin:34px 0 12px}}
h3{{font-size:14px;font-weight:700;margin:20px 0 8px}}
.sub{{color:#9aa1b3;margin:0 0 28px}}
.tot{{display:flex;gap:14px;margin:0 0 14px;flex-wrap:wrap}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}}
.card{{background:linear-gradient(180deg,#12151f,#0b0d15);
 border:1px solid #252c3e;border-radius:14px;padding:20px}}
.tot .card{{flex:1;min-width:150px}}
.card .l{{color:#9aa1b3;font-size:12px}} .card .v{{font-size:30px;font-weight:800;margin-top:4px}}
.card .v2{{font-size:22px;font-weight:700;margin-top:4px}}
.card .n{{color:#7d8598;font-size:12px;margin-top:6px;line-height:1.45}}
.tile{{cursor:help}} .card[data-tip] .l{{border-bottom:1px dotted #3a4257;display:inline-block;
 padding-bottom:1px}}
[data-tip]:focus-visible{{outline:2px solid #8b7bff;outline-offset:2px}}
g[data-tip]:hover .mark,g[data-tip]:focus .mark{{filter:brightness(1.35)}}
#tip{{position:fixed;display:none;background:#161a26;border:1px solid #2c3550;
 border-radius:10px;padding:10px 13px;font-size:12.5px;line-height:1.5;
 z-index:9;max-width:340px;box-shadow:0 8px 24px rgba(0,0,0,.55)}}
#tip .tt{{color:#f2f3f7;font-weight:600;margin-bottom:2px}}
#tip .tb{{color:#9aa1b3}}
#tip .trow{{display:flex;align-items:center;gap:8px;margin:3px 0}}
#tip .tkey{{width:10px;height:3px;border-radius:1.5px;flex:none}}
#tip .trow b{{font-variant-numeric:tabular-nums}}
#tip .tlab{{color:#9aa1b3}}
.desc{{color:#7d8598;font-size:13px;line-height:1.55;margin:-4px 0 14px}}
.story{{background:linear-gradient(180deg,#12151f,#0b0d15);border:1px solid #252c3e;
 border-radius:14px;padding:16px 20px;margin:0 0 10px}}
.story b{{font-size:15px}}
.story .n{{color:#9aa1b3;font-size:13px;line-height:1.55;margin-top:4px}}
.g{{background:linear-gradient(135deg,#8b7bff,#5ad1c9);-webkit-background-clip:text;background-clip:text;color:transparent}}
table{{width:100%;border-collapse:collapse;border:1px solid #1b2030;border-radius:12px;overflow:hidden}}
th,td{{padding:11px 14px;border-bottom:1px solid #1b2030;text-align:left}}
th{{color:#7d8598;font-size:11px;text-transform:uppercase;letter-spacing:.07em}}
td.r{{text-align:right;color:#5ad1c9;font-variant-numeric:tabular-nums}} .muted{{color:#7d8598}}
code{{color:#8b7bff}}
.callout{{border:1px solid #6b5416;background:#14100a;border-radius:12px;
 padding:14px 18px;margin:24px 0}}
.callout b{{color:#e8b34b}}
ul.warn{{margin:8px 0 0;padding-left:20px}} ul.warn li{{color:#e8b34b;margin:6px 0}}
ul.notes{{margin:14px 0 0;padding-left:20px}} ul.notes li{{color:#9aa1b3;margin:8px 0}}
details{{margin:10px 0}} details summary{{cursor:pointer;color:#7d8598}}
.foot{{color:#7d8598;font-size:12.5px;margin-top:26px}}
caption.sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
 clip:rect(0,0,0,0);white-space:nowrap;border:0}}
</style></head><body><div class="wrap">
<h1>Session <span class="g">dissected</span></h1>
<p class="sub">{e(d.sid)} — {e(man.get("tool") or "unknown tool")},
{e(_when(d.started))} → {e(_when(d.ended))} · wrap flags: {e(_flags_line(man)) if man else "unknown"}
· billing: {e(d.billing)}</p>
{warn_card}
<div class="tot">
<div class="card"{_tip_attr("Input tokens", body="Input tokens are everything sent TO the model (your conversation so far, tool outputs, prompts), the part that grows every turn and that distil compresses.")}><div class="l">Input tokens</div><div class="v">{_human(d.baseline_tokens)} → {_human(d.distil_tokens)}</div><div class="n">would have been sent → actually sent</div></div>
<div class="card"{_tip_attr("Saved", body="Share of input tokens distil kept off the wire across the whole session.")}><div class="l">Saved</div><div class="v g">{d.pct_saved:.1f}%</div><div class="n">of input tokens never sent</div></div>
<div class="card"{_tip_attr("Dollars", body="What those saved tokens are worth at API prices. On a flat-rate subscription nothing is billed per token, so this is notional: the real win is headroom.")}><div class="l">Dollars{e(dol_note)}</div><div class="v">${d.dollars_saved:.2f}</div><div class="n">at API prices for this model mix</div></div>
</div>
{story}
<h2>Per model</h2>
<p class="desc">Who talked and what it cost: baseline is what each model <em>would</em> have
received unwrapped; distil is what was actually sent after compression.</p>
<table><caption class="sr-only">Token usage by model</caption>
<tr><th scope="col">model</th><th scope="col">req</th><th scope="col">baseline</th>
<th scope="col">distil</th><th scope="col">saved</th></tr>{model_rows}</table>
{detail_body}
{corr_html}
<h2>Quality loops</h2>
<p class="desc">distil's own safety nets: <b>expand</b> lets the model pull any folded block
back when it needs the detail; <b>shadow</b> re-runs a sample of requests uncompressed to
verify the answers don't change.</p>
<p class="muted">expand: {d.expand_resolved} resolved in-proxy{expand_miss_html} · shadow: {d.shadow_sampled} sampled ·
verdicts near this session (time-joined): {d.shadow_window_agree}/{d.shadow_window_rows} equivalent</p>
{exit_line}
<p class="foot">Local-first: assembled from savings.jsonl, sessions/&lt;sid&gt;*, restore/ and
shadow.jsonl on this machine. Content-free — handles and kind:size signatures only.</p>
</div><div id="tip" role="tooltip"></div><script>
(function () {{
  var tip = document.getElementById("tip");
  function fill(el) {{
    var d; try {{ d = JSON.parse(el.dataset.tip); }} catch (e) {{ return false; }}
    tip.replaceChildren();
    var t = document.createElement("div"); t.className = "tt";
    t.textContent = d.t || ""; tip.append(t);
    if (d.body) {{ var b = document.createElement("div"); b.className = "tb";
      b.textContent = d.body; tip.append(b); }}
    (d.rows || []).forEach(function (r) {{
      var row = document.createElement("div"); row.className = "trow";
      var key = document.createElement("span"); key.className = "tkey";
      if (r[0]) key.style.background = r[0]; else key.style.visibility = "hidden";
      var val = document.createElement("b"); val.textContent = r[2];
      var lab = document.createElement("span"); lab.className = "tlab";
      lab.textContent = r[1];
      row.append(key, val, lab); tip.append(row);
    }});
    tip.style.display = "block";
    return true;
  }}
  function place(x, y) {{
    var r = tip.getBoundingClientRect(), pad = 14;
    var px = x + pad, py = y + pad;
    if (px + r.width > innerWidth - 8) px = x - r.width - pad;
    if (py + r.height > innerHeight - 8) py = y - r.height - pad;
    tip.style.left = Math.max(8, px) + "px"; tip.style.top = Math.max(8, py) + "px";
  }}
  document.addEventListener("mousemove", function (e) {{
    if (e.target === tip || tip.contains(e.target)) return; // hoverable: don't hide over the tip itself
    var el = e.target.closest && e.target.closest("[data-tip]");
    if (el && fill(el)) place(e.clientX, e.clientY);
    else tip.style.display = "none";
  }});
  document.addEventListener("focusin", function (e) {{
    var el = e.target.closest && e.target.closest("[data-tip]");
    if (el && fill(el)) {{ var r = el.getBoundingClientRect(); place(r.right, r.top); }}
  }});
  document.addEventListener("focusout", function () {{ tip.style.display = "none"; }});
  document.addEventListener("keydown", function (e) {{
    if (e.key === "Escape") tip.style.display = "none";
  }});
}})();
</script></body></html>"""


# ------------------------------------------------------------------- portal
def _session_row(o: SessionOverview) -> dict[str, Any]:
    """One session as JSON-serializable data, shared by the initial HTML
    render and the ``/sessions.json`` poll endpoint so both stay in sync."""
    pct = (
        100.0 * (o.baseline_tokens - o.distil_tokens) / o.baseline_tokens
        if o.baseline_tokens
        else 0.0
    )
    return {
        "sid": o.sid,
        "tool": o.tool or "?",
        "started": _when(o.started),
        "last": _when(o.last_ts),
        "requests": o.requests,
        "pct": round(pct, 1),
        "status": o.status,
    }


def sessions_json(sessions: list[SessionOverview]) -> str:
    """JSON payload backing the portal's in-place poll (``/sessions.json``)."""
    return json.dumps([_session_row(o) for o in sessions])


def render_sessions_html(sessions: list[SessionOverview]) -> str:
    """The portal index: the session picker as a clickable page.

    Polls ``/sessions.json`` every 15 s and patches the table body in place
    (keyed by session id) rather than reloading the whole page, so keyboard
    focus and reading position survive an update. A visible Pause button
    lets a user stop the polling entirely.
    """
    e = _html.escape
    rows = (
        "".join(
            f'<tr data-sid="{e(r["sid"])}" onclick="location=\'/session/{e(r["sid"])}\'">'
            f'<td><code><a href="/session/{e(r["sid"])}">{e(r["sid"])}</a></code></td>'
            f"<td>{e(r['tool'])}</td>"
            f"<td>{e(r['started'])}</td><td>{e(r['last'])}</td>"
            f"<td class='r'>{r['requests']}</td>"
            f"<td class='r'>{r['pct']:.1f}%</td>"
            f"<td>{e(r['status'])}</td></tr>"
            for r in (_session_row(o) for o in sessions)
        )
        or "<tr><td class='muted' colspan='7'>no sessions recorded yet — run a wrap first</td></tr>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Distil — sessions</title><style>
body{{margin:0;background:#06070b;color:#f2f3f7;font:15px/1.6 Inter,ui-sans-serif,sans-serif}}
.wrap{{max-width:820px;margin:0 auto;padding:48px 24px}}
h1{{font-size:30px;font-weight:800;letter-spacing:-.02em;margin:0 0 6px}}
.sub{{color:#9aa1b3;margin:0 0 28px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.g{{background:linear-gradient(135deg,#8b7bff,#5ad1c9);-webkit-background-clip:text;background-clip:text;color:transparent}}
table{{width:100%;border-collapse:collapse;border:1px solid #1b2030;border-radius:12px;overflow:hidden}}
th,td{{padding:11px 14px;border-bottom:1px solid #1b2030;text-align:left}}
th{{color:#7d8598;font-size:11px;text-transform:uppercase;letter-spacing:.07em}}
td.r{{text-align:right;color:#5ad1c9;font-variant-numeric:tabular-nums}}
tbody tr{{cursor:pointer}} tbody tr:hover td{{background:#10131d}}
code{{color:#8b7bff}} .muted{{color:#7d8598}}
caption.sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
 clip:rect(0,0,0,0);white-space:nowrap;border:0}}
code a{{color:inherit;text-decoration:underline}}
code a:hover,code a:focus{{color:#5ad1c9}}
code a:focus-visible{{outline:2px solid #8b7bff;outline-offset:2px;border-radius:4px}}
.foot{{color:#7d8598;font-size:12.5px;margin-top:22px}}
.pause-btn{{background:#10131d;border:1px solid #1b2030;color:#e0e2ea;font-size:12.5px;
  padding:5px 12px;border-radius:6px;cursor:pointer}}
.pause-btn:hover{{background:#171a28}}
.pause-btn:focus-visible{{outline:2px solid #8b7bff;outline-offset:2px}}
</style></head><body><div class="wrap">
<h1>Distil <span class="g">sessions</span></h1>
<p class="sub"><span>Pick a session to dissect — newest activity first. Updates automatically.</span>
<button type="button" id="pause-btn" class="pause-btn" aria-pressed="false">Pause</button></p>
<table><caption class="sr-only">Recorded sessions</caption>
<thead><tr><th scope="col">session</th><th scope="col">tool</th><th scope="col">started</th>
<th scope="col">last</th><th scope="col">reqs</th>
<th scope="col">saved</th><th scope="col">status</th></tr></thead>
<tbody id="sessions-body">{rows}</tbody></table>
<p class="foot" id="foot-note">Local-first: served from this machine's ~/.distil only. Reports are
per-session; JSON at /json/&lt;session&gt;. Updates every 15 s.</p>
</div>
<script>
(function () {{
  var POLL_MS = 15000;
  var tbody = document.getElementById("sessions-body");
  var pauseBtn = document.getElementById("pause-btn");
  var note = document.getElementById("foot-note");
  var timer = null, paused = false;
  var FOOT = "Local-first: served from this machine's ~/.distil only. Reports are per-session; " +
    "JSON at /json/<session>.";

  function render(list) {{
    // Index the rows we already have instead of building a selector out of the
    // session id: ids come from $DISTIL_SESSION, so a quote in one would make
    // querySelector throw and kill the poll loop. Null-prototype so an id of
    // "__proto__" is just a key.
    var byId = Object.create(null);
    Array.prototype.forEach.call(tbody.querySelectorAll("tr[data-sid]"), function (tr) {{
      byId[tr.dataset.sid] = tr;
    }});
    var seen = Object.create(null);
    list.forEach(function (r) {{
      seen[r.sid] = true;
      var tr = byId[r.sid];
      if (!tr) {{
        tr = document.createElement("tr");
        tr.dataset.sid = r.sid;
        tr.addEventListener("click", function () {{ location = "/session/" + r.sid; }});
        tr.innerHTML = "<td><code><a></a></code></td><td></td><td></td><td></td>" +
          "<td class='r'></td><td class='r'></td><td></td>";
        tbody.appendChild(tr);
        byId[r.sid] = tr;
      }}
      var c = tr.children;
      var link = c[0].firstChild.firstChild;  // <td> -> <code> -> <a>
      link.textContent = r.sid;
      link.setAttribute("href", "/session/" + encodeURIComponent(r.sid));
      c[1].textContent = r.tool;
      c[2].textContent = r.started;
      c[3].textContent = r.last;
      c[4].textContent = r.requests;
      c[5].textContent = r.pct.toFixed(1) + "%";
      c[6].textContent = r.status;
    }});
    Object.keys(byId).forEach(function (sid) {{
      if (!seen[sid]) byId[sid].remove();
    }});
    var placeholder = tbody.querySelector("tr:not([data-sid])");
    if (list.length && placeholder) placeholder.remove();
    if (!list.length && !tbody.querySelector("tr[data-sid]") && !placeholder) {{
      tbody.innerHTML = "<tr><td class='muted' colspan='7'>no sessions recorded yet — " +
        "run a wrap first</td></tr>";
    }}
  }}

  function poll() {{
    fetch("/sessions.json", {{cache: "no-store"}})
      .then(function (r) {{ return r.json(); }})
      .then(render)
      .catch(function () {{}});
  }}

  function schedule() {{ timer = setInterval(poll, POLL_MS); }}

  pauseBtn.addEventListener("click", function () {{
    paused = !paused;
    pauseBtn.textContent = paused ? "Resume" : "Pause";
    pauseBtn.setAttribute("aria-pressed", paused ? "true" : "false");
    note.textContent = FOOT + (paused ? " Updates paused." : " Updates every 15 s.");
    if (paused) {{
      clearInterval(timer);
    }} else {{
      poll();
      schedule();
    }}
  }});

  schedule();
}})();
</script>
</body></html>"""


def make_server(host: str = "127.0.0.1", port: int = 8790, transcript: str | None = None) -> Any:
    """Build (don't start) the dissect portal server — stdlib only, like the
    gateway. Routes: ``/`` index, ``/session/<sid>`` report, ``/json/<sid>``.

    ``transcript`` makes correlation the default for every report page
    ("auto" or a transcript path — the ``--transcript`` flag passed through);
    ``?t=0`` opts a page out, ``?t=1`` opts in when no default is set.

    Every request re-reads state from disk, so a refresh shows the live
    session as it grows. Binds localhost by default; the data is one user's
    own telemetry, so there is no auth layer — don't bind it wider unless
    that is understood.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Portal(BaseHTTPRequestHandler):
        server_version = "distil-dissect"

        def _send(self, status: int, body: str, ctype: str = "text/html; charset=utf-8") -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 — http.server API
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, render_sessions_html(list_sessions()))
                return
            if path == "/sessions.json":
                self._send(200, sessions_json(list_sessions()), ctype="application/json")
                return
            query = self.path.partition("?")[2]
            for prefix, as_json in (("/session/", False), ("/json/", True)):
                if path.startswith(prefix):
                    sid = resolve_sid(path[len(prefix) :])
                    if sid is None:
                        self._send(
                            404,
                            '<!doctype html><html lang="en"><head><meta charset="utf-8"/>'
                            "<title>Unknown session: distil</title></head>"
                            "<body><h1>Unknown session</h1>"
                            '<p><a href="/">sessions</a></p></body></html>',
                        )
                        return
                    d = dissect(sid)
                    peers = list_sessions()
                    corr = None
                    params = query.split("&")
                    want_corr = "t=1" in params or (transcript is not None and "t=0" not in params)
                    if want_corr:
                        try:
                            from .correlate import correlate
                            from .transcripts import find_transcript

                            man = d.manifest or {}
                            tr = find_transcript(
                                str(man.get("tool") or ""),
                                (d.started, d.ended or d.started),
                                cwd=man.get("cwd"),
                                path=None if transcript in (None, "auto") else transcript,
                            )
                            if tr is not None:
                                corr = correlate(d, tr)
                        except Exception:  # noqa: BLE001 — correlation is best-effort
                            corr = None
                    if as_json:
                        self._send(
                            200,
                            json.dumps(to_json(d, peers, corr), indent=2),
                            ctype="application/json",
                        )
                    else:
                        toggle = (
                            f'<a href="/session/{_html.escape(sid)}?t=0" style="color:#8b7bff">hide correlation</a>'
                            if corr is not None
                            else f'<a href="/session/{_html.escape(sid)}?t=1" style="color:#8b7bff">+ correlate with transcript</a>'
                        )
                        page = render_html(d, peers, corr).replace(
                            "<h1>",
                            f'<p><a href="/" style="color:#8b7bff">← sessions</a> · {toggle}</p><h1>',
                            1,
                        )
                        self._send(200, page)
                    return
            self._send(
                404,
                '<!doctype html><html lang="en"><head><meta charset="utf-8"/>'
                "<title>Not found: distil</title></head>"
                '<body><h1>Not found</h1><p><a href="/">sessions</a></p></body></html>',
            )

        def log_message(self, *args: Any) -> None:  # quiet by design, like the proxy
            pass

    return ThreadingHTTPServer((host, port), _Portal)
