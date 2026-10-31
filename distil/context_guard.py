"""Context guard: keep client auto-compaction working under compression.

When distil compresses a request, the provider reports ``usage.input_tokens``
for the *compressed* context. Clients whose auto-compaction keys off reported
usage (Claude Code) therefore defer compaction — which is the feature — but with
no floor: the raw transcript can grow until even the compressed request exceeds
the real context window, then every turn fails with a 400 "prompt is too long"
and the session enters a compact-every-turn loop.

Three cooperating pieces, all passthrough until the danger zone:

1. ``CONTEXT_WINDOWS`` — default context window per model family.
2. ``learn_from_error`` — parses "prompt is too long: N > M" from upstream 400
   bodies, persisting per-model limits to ``~/.distil/context_guard.json`` so a
   capped account is not re-surprised after a restart.
3. ``nudge_usage`` / ``StreamGuard`` — when the ORIGINAL (pre-compression)
   context or the forwarded (compressed) context crosses ``TRIGGER_FRACTION`` of
   the effective limit, inflates the reported ``input_tokens`` to
   ``REPORT_FRACTION`` of the believed limit, tripping the client's own graceful
   compaction before the 400 wall.

Kill switch: ``DISTIL_CONTEXT_GUARD=0``.
On by default (intelligence is the default; flags are opt-outs only).
Content-free and fail-open: any exception leaves the response unmodified.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

# Fraction of the effective limit at which the nudge arms.
# ponytail: 0.90 leaves ~one normal turn of headroom before the hard wall.
TRIGGER_FRACTION = 0.90
# Client-reported fraction of the believed limit after nudge.
# Claude Code auto-compacts around 92%; 0.95 clears that threshold with margin.
REPORT_FRACTION = 0.95

_PROMPT_TOO_LONG_RE = re.compile(
    r"prompt is too long:\s*(\d+)\s*tokens?\s*>\s*(\d+)\s*maximum", re.IGNORECASE
)

# Sanity band: outside this range "prompt is too long: N > M" describes a quota
# or throttle, not a context window, and learning it would clamp every request.
_MIN_LEARNED = 8_000
_MAX_LEARNED = 10_000_000

# A learned limit expires: entitlements get provisioned, gateways get repointed.
# The cost of expiry is at most one more prompt-too-long error.
_LEARNED_TTL = 3_600.0  # seconds

# Context windows by model prefix; longest prefix match wins.
# ponytail: flat table, extend when new models arrive or limits change.
CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic — 200 k for all current Claude generations;
    # context-1m beta raises to 1 M on supported accounts.
    "claude-fable-5": 200_000,
    "claude-fable": 200_000,
    "claude-opus-5": 200_000,
    "claude-opus-4": 200_000,
    "claude-opus": 200_000,
    "claude-sonnet-5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-sonnet": 200_000,
    "claude-haiku-5": 200_000,
    "claude-haiku-4": 200_000,
    "claude-haiku": 200_000,
    # OpenAI
    "gpt-6": 1_000_000,
    "gpt-5.6": 1_000_000,
    "gpt-5.5": 128_000,
    "gpt-5.4": 128_000,
    "gpt-5.3": 128_000,
    "gpt-5.2": 128_000,
    "gpt-5.1": 128_000,
    "gpt-5": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "o3": 200_000,
    "o4": 200_000,
    # Gemini
    "gemini-3": 1_048_576,
    "gemini-2.5": 1_048_576,
    "gemini-2": 1_000_000,
}

# In-memory cache: model_key -> (limit, learned_at_monotonic)
# model_key is "<model>" or "<model>|1m" for context-1m beta requests.
_learned: dict[str, tuple[int, float]] = {}
_loaded = False


def _state_path() -> Path:
    home = os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))
    return Path(home) / "context_guard.json"


def _load() -> None:
    """Warm the in-memory cache from disk (best-effort, once per process)."""
    global _loaded
    _loaded = True
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        now, wall = time.monotonic(), time.time()
        for key, entry in (data or {}).items():
            if isinstance(entry, dict):
                lim = int(entry.get("limit", 0))
                # Age carries across restarts (monotonic time does not), so a learned
                # limit still expires after _LEARNED_TTL. Entries without a timestamp
                # (older files) are dropped rather than given a fresh hour.
                age = wall - float(entry.get("at", wall - _LEARNED_TTL))
                if _MIN_LEARNED <= lim <= _MAX_LEARNED and 0 <= age < _LEARNED_TTL:
                    _learned[key] = (lim, now - age)
    except (OSError, ValueError, TypeError, KeyError):
        pass


def _save() -> None:
    """Persist in-memory limits to disk (best-effort, fail-open)."""
    try:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        offset = time.time() - time.monotonic()
        payload = {k: {"limit": lim, "at": at + offset} for k, (lim, at) in _learned.items()}
        # Write-then-rename: a concurrent reader sees the old file or the new one, never half.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _ensure_loaded() -> None:
    if not _loaded:
        _load()


def context_guard_enabled() -> bool:
    """True unless ``DISTIL_CONTEXT_GUARD=0``."""
    return os.environ.get("DISTIL_CONTEXT_GUARD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def context_window(model: str | None) -> int | None:
    """Context window for *model* by longest prefix match, or None when unknown."""
    if not model:
        return None
    mid = model.strip()
    for pre in (
        "anthropic/",
        "openai/",
        "azure/",
        "gemini/",
        "google/",
        "vertex_ai/",
        "models/",
    ):
        if mid.startswith(pre):
            mid = mid[len(pre) :]
            break
    if mid.startswith("anthropic."):
        mid = mid[len("anthropic.") :]
    mid = mid.split("@", 1)[0]
    best_key: str | None = None
    best_win: int | None = None
    for prefix, win in CONTEXT_WINDOWS.items():
        if mid.startswith(prefix) and (best_key is None or len(prefix) > len(best_key)):
            best_key, best_win = prefix, win
    return best_win


def _has_1m_beta(beta_header: str | None) -> bool:
    if not beta_header:
        return False
    return any(t.strip().lower().startswith("context-1m") for t in beta_header.split(","))


def _model_key(model: str | None, *, has_1m: bool) -> str:
    return f"{model or ''}{'|1m' if has_1m else ''}"


def _effective_limit(model: str | None, *, has_1m: bool) -> int | None:
    """The real context limit the forwarded request is subject to.

    Starts from the table default (raised to 1 M by the context-1m beta), then
    caps at any limit learned from a real "prompt is too long" error while still
    fresh. Returns None when the model is not in the table and nothing has been
    learned — so the guard stays inert rather than guessing.
    """
    _ensure_loaded()
    base = context_window(model)
    believed = max(base or 0, 1_000_000) if has_1m else base
    if not believed:
        return None
    key = _model_key(model, has_1m=has_1m)
    entry = _learned.get(key)
    if entry is not None:
        lim, learned_at = entry
        if time.monotonic() - learned_at >= _LEARNED_TTL:
            del _learned[key]
        else:
            believed = min(believed, lim)
    return believed


def learn_from_error(
    model: str | None,
    error_body: bytes | str,
    *,
    beta_header: str | None = None,
) -> int | None:
    """Parse a "prompt is too long: N > M" error and store M as the real limit.

    Returns the learned maximum, or None when the body carries no learnable limit.
    Fail-open: any exception returns None without touching the in-memory cache.
    """
    _ensure_loaded()
    try:
        if isinstance(error_body, (bytes, bytearray)):
            text = bytes(error_body).decode("utf-8", errors="replace")
        elif isinstance(error_body, str):
            text = error_body
        else:
            return None
        m = _PROMPT_TOO_LONG_RE.search(text)
        if not m:
            return None
        maximum = int(m.group(2))
        if not _MIN_LEARNED <= maximum <= _MAX_LEARNED:
            return None
        has_1m = _has_1m_beta(beta_header)
        key = _model_key(model, has_1m=has_1m)
        _learned[key] = (maximum, time.monotonic())
        _save()
        return maximum
    except Exception:  # noqa: BLE001 — a diagnostic must never break a request
        return None


def nudged_input_tokens(
    usage: dict[str, Any],
    tokens_saved: int,
    model: str | None,
    *,
    beta_header: str | None = None,
) -> int | None:
    """Compute the inflated ``input_tokens`` value, or None to leave it unchanged.

    *tokens_saved* is the message-level compression delta (before_tok - after_tok)
    that distil achieved; it approximates how much larger the original context was
    than what was forwarded to the provider.
    """
    if not context_guard_enabled():
        return None
    if "input_tokens" not in usage:
        return None  # e.g. OpenAI Chat (prompt_tokens): never inject a field it lacks
    has_1m = _has_1m_beta(beta_header)
    effective = _effective_limit(model, has_1m=has_1m)
    if not effective:
        return None
    base_win = context_window(model)
    # The limit the CLIENT believes it has (uncapped by any learned limit).
    # Used for the target, not the trigger: the nudge reports relative to what
    # the client will compact against, even if we know the real wall is lower.
    believed = max(base_win or 0, 1_000_000) if has_1m else (base_win or effective)
    if not believed:
        return None
    input_tok = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    forwarded_total = input_tok + cache_read + cache_creation
    # Estimate the original (pre-compression) total as forwarded + message savings.
    original_total = forwarded_total + max(0, tokens_saved)
    # Arm when either the original or the forwarded context approaches the real limit.
    if (
        original_total < TRIGGER_FRACTION * effective
        and forwarded_total < TRIGGER_FRACTION * effective
    ):
        return None
    target_total = int(believed * REPORT_FRACTION)
    if forwarded_total >= target_total:
        return None  # already reads full to the client — never deflate
    return max(target_total - cache_read - cache_creation, input_tok)


def nudge_usage(
    usage: dict[str, Any],
    tokens_saved: int,
    model: str | None,
    *,
    beta_header: str | None = None,
) -> bool:
    """Inflate ``usage['input_tokens']`` in place if near the limit. Returns True when modified."""
    new_val = nudged_input_tokens(usage, tokens_saved, model, beta_header=beta_header)
    if new_val is None:
        return False
    usage["input_tokens"] = new_val
    return True


def nudge_body(
    rbody: bytes,
    tokens_saved: int,
    model: str | None,
    *,
    beta_header: str | None = None,
) -> bytes:
    """Non-streaming: parse the response, nudge ``usage.input_tokens`` if needed.

    Handles the Anthropic Messages shape and the OpenAI Responses API (both use
    ``usage.input_tokens``). Returns the original bytes unchanged on any failure
    or when no nudge is needed (fail-open).
    """
    if not context_guard_enabled():
        return rbody
    try:
        resp = json.loads(rbody)
    except (ValueError, TypeError):
        return rbody
    if not isinstance(resp, dict):
        return rbody
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return rbody
    if not nudge_usage(usage, tokens_saved, model, beta_header=beta_header):
        return rbody
    return json.dumps(resp, separators=(",", ":")).encode()


class StreamGuard:
    """SSE byte transformer: inflates usage in client-bound Anthropic stream events.

    Feed every chunk headed to the client through :meth:`feed`; call
    :meth:`flush` when the stream ends. Inert (zero per-chunk overhead) until
    the first ``message_start`` event is encountered.

    Rewrites both ``message_start`` (the initial usage snapshot) and the final
    ``message_delta`` (which carries cumulative input usage on current Anthropic
    releases and overrides the start values in Claude Code's parser).
    """

    def __init__(
        self,
        tokens_saved: int,
        model: str | None,
        *,
        beta_header: str | None = None,
    ) -> None:
        self._tokens_saved = tokens_saved
        self._model = model
        self._beta_header = beta_header
        self._buf = bytearray()
        self._done = not context_guard_enabled()
        self._seen_start = False
        self._target_total: int | None = None
        self._start_cache_read = 0
        self._start_cache_creation = 0

    def __call__(self, chunk: bytes) -> bytes:
        """Alias for :meth:`feed` so the guard can be passed as a ``chunk_transform`` callable."""
        return self.feed(chunk)

    def feed(self, chunk: bytes) -> bytes:
        if self._done:
            return chunk
        self._buf.extend(chunk)
        out = bytearray()
        while not self._done:
            end = _event_end(self._buf)
            if end == -1:
                # Give up after a large unframed stretch to avoid unbounded buffering.
                if len(self._buf) > 256 * 1024:
                    out += self._finish()
                break
            event = bytes(self._buf[:end])
            del self._buf[:end]
            out += self._process(event)
        if self._done:
            # Went inert mid-chunk: the rest of this chunk follows in order, now.
            out += self._finish()
        return bytes(out)

    def flush(self) -> bytes:
        """Return any buffered bytes and go inert."""
        return self._finish()

    def _finish(self) -> bytes:
        self._done = True
        rem = bytes(self._buf)
        self._buf = bytearray()
        return rem

    def _process(self, event: bytes) -> bytes:
        etype = _event_type(event)
        if etype is None or etype == b"ping":
            return event  # ping / keepalive / comment — not part of the message protocol
        if not self._seen_start:
            self._seen_start = True
            if etype != b"message_start":
                self._done = True
                return event
            try:
                return self._rewrite_start(event)
            except Exception:  # noqa: BLE001 — guard must not break a stream
                self._done = True
                return event
        if etype != b"message_delta":
            return event  # pass content deltas through unchanged
        self._done = True
        if self._target_total is None:
            return event
        try:
            return self._rewrite_delta(event)
        except Exception:  # noqa: BLE001
            return event

    def _rewrite_start(self, event: bytes) -> bytes:
        lines = event.split(b"\n")
        for i, line in enumerate(lines):
            if not line.startswith(b"data:"):
                continue
            payload = json.loads(line[5:].strip())
            if payload.get("type") != "message_start":
                return event
            usage = (payload.get("message") or {}).get("usage")
            if not isinstance(usage, dict):
                return event
            new_val = nudged_input_tokens(
                usage,
                self._tokens_saved,
                self._model,
                beta_header=self._beta_header,
            )
            if new_val is None:
                self._done = True
                return event
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
            self._target_total = new_val + cache_read + cache_creation
            self._start_cache_read = cache_read
            self._start_cache_creation = cache_creation
            usage["input_tokens"] = new_val
            eol = b"\r" if line.endswith(b"\r") else b""
            lines[i] = b"data: " + json.dumps(payload, separators=(",", ":")).encode() + eol
            return b"\n".join(lines)
        return event

    def _rewrite_delta(self, event: bytes) -> bytes:
        assert self._target_total is not None  # guarded: only called when _target_total is set
        lines = event.split(b"\n")
        for i, line in enumerate(lines):
            if not line.startswith(b"data:"):
                continue
            payload = json.loads(line[5:].strip())
            if payload.get("type") != "message_delta":
                return event
            usage = payload.get("usage")
            # A delta without cumulative input usage cannot override message_start — leave it.
            if not isinstance(usage, dict) or "input_tokens" not in usage:
                return event
            cache_read = int(usage.get("cache_read_input_tokens", self._start_cache_read) or 0)
            cache_creation = int(
                usage.get("cache_creation_input_tokens", self._start_cache_creation) or 0
            )
            new_input = self._target_total - cache_read - cache_creation
            if int(usage.get("input_tokens") or 0) >= new_input:
                return event  # never deflate
            usage["input_tokens"] = new_input
            eol = b"\r" if line.endswith(b"\r") else b""
            lines[i] = b"data: " + json.dumps(payload, separators=(",", ":")).encode() + eol
            return b"\n".join(lines)
        return event


def _event_end(buf: bytearray) -> int:
    """End offset (exclusive) of the first complete SSE event in *buf*, or -1.

    SSE allows LF or CRLF line endings, so an event ends at the earliest blank
    line in either form.
    """
    ends = [i + len(sep) for sep in (b"\n\n", b"\r\n\r\n") if (i := buf.find(sep)) != -1]
    return min(ends) if ends else -1


def _event_type(event: bytes) -> bytes | None:
    """The type of one SSE event frame, or None for non-data frames (comments, pings)."""
    saw_data = False
    for line in event.split(b"\n"):
        if line.startswith(b"event:"):
            return line[6:].strip()
        if line.startswith(b"data:"):
            saw_data = True
    if not saw_data:
        return None
    # No explicit ``event:`` line — read the type from inside the payload.
    for line in event.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        try:
            payload = json.loads(line[5:].strip())
        except Exception:  # noqa: BLE001
            return b""
        val = payload.get("type") if isinstance(payload, dict) else None
        return val.encode() if isinstance(val, str) else b""
    return b""


def reset_learned_limits() -> None:
    """Test hook: clear all in-memory learned limits and reset the load flag."""
    global _loaded
    _learned.clear()
    _loaded = False
