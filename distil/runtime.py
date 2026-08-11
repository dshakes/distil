"""Genuine runtime savings — measure what *using* Distil actually saved.

The corpus numbers are a demonstration; this is the real thing. As live traffic
flows through the proxy/gateway, every compressed request's actual token
reduction is accumulated here and periodically flushed to the local savings
ledger — so `distil leaderboard` reflects *your own usage*, not a synthetic
benchmark. Content is never recorded; only token counts and the priced dollar
estimate. Federate the signed aggregate (`distil/telemetry.py`) and the
community number is genuine, verifiable savings too.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import ledger, pricing


@dataclass
class RuntimeSavings:
    """Per-model savings accumulator.

    ``model`` is only the FALLBACK used when a request carries no model id of
    its own — every request that names a model (the normal case) is accounted
    and priced under that model, so a Claude Code session mixing Opus and
    Haiku calls is never all priced at one rate. Models missing from the
    pricing catalog (e.g. a Gemini upstream) record their genuine token
    savings with dollars=0 rather than being silently priced at Claude rates.
    """

    model: str = "claude-opus-4-8"
    session_id: str = ""  # set in __post_init__; stamps every ledger record
    mode: str = ""  # compression mode (verbatim/lossless-only/digest); stamps rows
    requests: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    ledger_path: Path | None = None
    # model id -> [requests, tokens_before, tokens_after]
    by_model: dict[str, list[int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _last_flush: float = field(default=0.0, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Accept a str path too — the ledger needs a real Path for `.parent`.
        if self.ledger_path is not None and not isinstance(self.ledger_path, Path):
            self.ledger_path = Path(self.ledger_path)
        if not self.session_id:
            import os
            import time

            # DISTIL_SESSION (set by `distil wrap`) is the SAME id the wrapped
            # agent — and the status line it spawns — inherit, so per-session
            # savings correlate across the proxy and the status line. Falls back
            # to one-id-per-proxy-process when wrap didn't set it.
            #
            # EXPORT the fallback, do not just hold it. Every session-scoped path
            # (`session_marker_path`, `session_requests_path`,
            # `session_manifest_path`) resolves the id from os.environ. An
            # always-on proxy runs under launchd/systemd, which does not inherit a
            # shell environment, so the variable is unset: the id minted here went
            # onto ledger rows while every one of those helpers returned None and
            # silently wrote nothing. `distil dissect` then reported blocks=0 on
            # sessions worth over a million tokens — the analysis was not broken,
            # its input was never recorded.
            #
            # setdefault, not assignment: a real `distil wrap` id must always win,
            # or the proxy would rename the session its agent is already using.
            self.session_id = (
                os.environ.get("DISTIL_SESSION") or f"s{int(time.time())}-{os.getpid()}"
            )
            os.environ.setdefault("DISTIL_SESSION", self.session_id)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def dollars_saved(self) -> float:
        # Genuine, conservative floor: tokens saved priced at each model's base
        # input rate (the cache-aware benefit is larger; this is the defensible
        # minimum). Unpriceable models contribute 0 — never guessed rates.
        with self._lock:
            total = 0.0
            for mid, (_n, before, after) in self.by_model.items():
                price = pricing.resolve(mid)
                if price is not None:
                    total += (before - after) * price.input
            return total

    def record(self, before: int, after: int, model: str | None = None) -> None:
        mid = (model or "").strip() or self.model
        with self._lock:
            self.requests += 1
            self.tokens_before += before
            self.tokens_after += after
            cell = self.by_model.setdefault(mid, [0, 0, 0])
            cell[0] += 1
            cell[1] += before
            cell[2] += after

    def maybe_flush(self, every: int = 10, max_age: float = 30.0) -> bool:
        """Flush when *every* requests accumulated OR *max_age* seconds passed
        since the last flush — so the statusline/ledger stays fresh even in
        short or slow sessions instead of sitting on unflushed savings."""
        import time

        now = time.monotonic()
        with self._lock:
            if self.requests == 0:
                return False
            due = self.requests >= every or (now - self._last_flush) >= max_age
        if not due:
            return False
        flushed = self.flush()
        if flushed:
            self._last_flush = now
        return flushed

    def flush(self) -> bool:
        """Persist accumulated genuine savings to the ledger (one record per
        model, priced at that model's rate); reset counters. Returns False
        when there is nothing to flush."""
        with self._lock:
            if self.requests == 0:
                return False
            for mid, (n, before, after) in self.by_model.items():
                if before == 0:
                    continue  # nothing measured — a zero-baseline record is noise
                if before <= after:
                    continue  # nothing saved (e.g. a verbatim/lossless-only window) — a 0-row is noise
                price = pricing.resolve(mid)
                per_tok = price.input if price is not None else 0.0
                ledger.record(
                    trajectory_id="live-proxy",
                    model=mid if price is not None else f"{mid} (unpriced)",
                    turns=n,
                    baseline_dollars=before * per_tok,
                    distil_dollars=after * per_tok,
                    baseline_input_tokens=before,
                    distil_input_tokens=after,
                    session=self.session_id,
                    mode=self.mode,
                    path=self.ledger_path or ledger.default_path(),
                )
            self.requests = 0
            self.tokens_before = 0
            self.tokens_after = 0
            self.by_model.clear()
            return True
