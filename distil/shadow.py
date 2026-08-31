"""Shadow-mode live decision-equivalence — continuous, on real traffic.

The certificate (``distil conformal``) proves decision-equivalence *offline*, on a
calibration corpus. Shadow mode closes the loop *online*: it samples a fraction of
live requests, runs the decision BOTH on the compressed and the uncompressed
context, compares the agent's chosen action, and records a content-free
equivalence signal. You get a rolling, live decision-change rate on your own
production traffic — the thing periodic re-certification can only approximate.

Design constraints (this is in the request path):
  * **Never blocks the user.** The shadow (second, uncompressed) call runs in a
    background thread; the client gets the compressed response immediately.
  * **Sampled.** Only ``rate`` of requests are shadowed, so the cost overhead is
    ``rate`` (e.g. 5%), not 2x.
  * **Content-free.** The ledger stores only a decision *signature* and an
    ``equivalent`` bool — never prompt or response content (same privacy posture
    as the savings ledger / telemetry).

The "decision" is the agent's next action: the first ``tool_use`` block (Anthropic),
``tool_call`` (OpenAI), or ``functionCall`` (Gemini). Two responses are decision-
equivalent iff that action matches — exactly the ``{action, target}`` fingerprint
the certificate uses.

Streaming-aware: real agent sessions (Claude Code, Codex, the Gemini CLI) stream
their responses (SSE), so the decision must be reconstructed from the stream.
:func:`decision_signature_from_body` reads a non-streaming JSON body directly and
reconstructs a streamed (SSE / chunk-array) one via :func:`_decision_from_chunks`,
yielding the same signature either way.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time

try:
    import fcntl  # POSIX advisory locking; absent on Windows

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - windows
    _HAVE_FCNTL = False
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _state_dir() -> Path:
    import os

    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


# Decision-signature algorithm version. Bump on ANY change to how a signature is
# computed OR how the compared sample is generated — both change what a row means,
# so old and new rows are NEVER compared (a methodology fix must not silently
# invalidate, or be averaged into, old evidence).
# v1: tool name + full input, only Python-code strings AST-normalized.
# v2: + formatting-whitespace canonicalized on all strings (and non-Python code),
#     so `ls -la` == `ls  -la` and re-serialization jitter isn't a "decision change".
# v3: both A/A and A/B sides re-issued at temperature 0 (see force_deterministic).
#     v2 compared the live served response (produced at the agent's hot sampling
#     temperature) against a hot replay, so A/A self-agreement read ~38% — pure
#     sampling noise, not compression harm. v3 collapses that noise floor toward
#     ~100%, making A/B a trustworthy compression signal. v2 rows are discarded.
SIG_VERSION = 3

# A verdict (✓/⚠/✗) is only rendered once evidence is robust — a percentage over a
# handful of samples is noise wearing a number, and the noise-adjusted rate divides
# by the A/A self-agreement, which is itself unstable at small n. Below these, the
# status line shows a neutral warming state instead of a colored verdict.
VERDICT_MIN_AB = 50  # A/B (compressed-vs-original) samples
VERDICT_MIN_AA = 30  # A/A (self-agreement noise-baseline) samples


def _canon(obj: Any) -> str:
    """A short, stable hash of a JSON-able object — content-free in the ledger."""
    try:
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        blob = str(obj)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _current_version() -> str:
    """The running distil version, for stamping ledger rows. Lazy-imported to
    avoid a shadow<->hotswap import cycle; never raises (a stamp must not break
    recording) — falls back to "?" if the version can't be determined."""
    try:
        from .hotswap import installed_version

        return installed_version() or "?"
    except Exception:  # noqa: BLE001 — best-effort provenance only
        return "?"


def _canon_ws(s: str) -> str:
    """Collapse formatting whitespace: strip ends, runs of whitespace → one space.

    Kills wording/serialization jitter (`ls -la` vs `ls  -la`, pretty-printed vs
    minified JSON, trailing newlines) WITHOUT merging genuinely different tokens —
    different content still differs. This is the v2 fix for the ~27% A/A self-noise
    that made identical replayed requests read as "decision changed".
    """
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Edit-equivalence — AST-normalize code-bearing decision inputs
# ---------------------------------------------------------------------------
#
# For coding agents the decision IS the edit. Two responses that make the agent
# write the *same code* with trivially different whitespace/comments must count as
# decision-equivalent, not as a spurious change — otherwise the live signal
# over-reports drift and the certificate under-claims safe savings. We normalize
# any code-shaped string value inside a tool input through Python's AST (stdlib,
# model-free), so semantically identical edits hash equal while real logic changes
# still differ. Non-code strings and non-Python pass through untouched.

import ast as _ast  # noqa: E402

_CODE_HINTS = ("def ", "class ", "import ", "return ", "self.", " = ", "):")


def _looks_like_code(s: str) -> bool:
    if len(s) < 8:
        return False
    return "\n" in s or any(h in s for h in _CODE_HINTS)


def _normalize_code(s: str) -> str:
    try:
        return "py:" + _ast.dump(_ast.parse(s))
    except (SyntaxError, ValueError):
        return _canon_ws(s)  # not Python — at least strip formatting jitter (v2)


def _normalize_decision(value: Any) -> Any:
    """Recursively normalize decision inputs: AST-normalize code-shaped strings,
    and canonicalize formatting whitespace on plain strings (v2) so identical
    replayed decisions don't read as changed just from wording/serialization."""
    if isinstance(value, str):
        return _normalize_code(value) if _looks_like_code(value) else _canon_ws(value)
    if isinstance(value, dict):
        return {k: _normalize_decision(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_decision(v) for v in value]
    return value


def _sig_anthropic(name: Any, input_obj: Any) -> str:
    return "tool:" + _canon({"name": name, "input": _normalize_decision(input_obj)})


def _sig_openai(name: Any, arguments: Any) -> str:
    norm: Any = arguments
    if isinstance(arguments, str):
        try:
            norm = _normalize_decision(json.loads(arguments))
        except (ValueError, TypeError):
            norm = _normalize_code(arguments) if _looks_like_code(arguments) else arguments
    else:
        norm = _normalize_decision(arguments)
    return "tool:" + _canon({"name": name, "arguments": norm})


def _sig_gemini(name: Any, args: Any) -> str:
    return "tool:" + _canon({"name": name, "args": _normalize_decision(args)})


def decision_signature(resp_json: Any) -> str:
    """A content-free signature of the agent's chosen next action.

    ``tool:<hash>`` when the model called a tool (the decision that matters for an
    agent), ``text`` when it answered without acting, ``none`` when no decision
    could be read. Two responses are decision-equivalent iff their signatures match.
    """
    if not isinstance(resp_json, dict):
        return "none"

    # Anthropic Messages API
    content = resp_json.get("content")
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                return _sig_anthropic(b.get("name"), b.get("input"))
        return "text"  # answered without calling a tool

    # OpenAI Chat Completions
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list) and tcs and isinstance(tcs[0], dict):
            fn = tcs[0].get("function") or {}
            return _sig_openai(fn.get("name"), fn.get("arguments"))
        return "text"

    # Gemini generateContent
    candidates = resp_json.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        content = candidates[0].get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict) and isinstance(p.get("functionCall"), dict):
                    fc = p["functionCall"]
                    return _sig_gemini(fc.get("name"), fc.get("args"))
        return "text"  # responded without calling a function

    return "none"


def _decision_from_chunks(chunks: list[Any]) -> str:
    """Reconstruct the decision signature from a sequence of *streaming* chunks.

    Handles all three providers' streaming shapes by accumulating the first tool
    call across chunks, so the signature matches the non-streaming
    :func:`decision_signature` form exactly:

    * Anthropic SSE — ``content_block_start`` (tool_use name) + ``input_json_delta``
      fragments accumulated into the input object.
    * OpenAI SSE — ``choices[].delta.tool_calls[].function`` name + concatenated
      ``arguments`` string.
    * Gemini ``streamGenerateContent`` — ``candidates[].content.parts[].functionCall``.
    """
    a_name = None
    a_buf = ""
    a_tool = False
    a_text = False
    o_name = None
    o_args = ""
    o_tool = False
    o_text = False
    g_call = None
    g_text = False

    for ch in chunks:
        if not isinstance(ch, dict):
            continue

        # Anthropic streaming events
        ctype = ch.get("type")
        if ctype == "content_block_start":
            cb = ch.get("content_block") or {}
            if cb.get("type") == "tool_use" and not a_tool:
                a_tool = True
                a_name = cb.get("name")
                if isinstance(cb.get("input"), dict) and cb["input"]:
                    a_buf = json.dumps(cb["input"])
            elif cb.get("type") == "text":
                a_text = True
        elif ctype == "content_block_delta":
            delta = ch.get("delta") or {}
            if delta.get("type") == "input_json_delta" and a_tool:
                a_buf += delta.get("partial_json") or ""
            elif delta.get("type") == "text_delta":
                a_text = True

        # OpenAI streaming deltas
        choices = ch.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta") or {}
            tcs = delta.get("tool_calls")
            if isinstance(tcs, list) and tcs and isinstance(tcs[0], dict):
                o_tool = True
                fn = tcs[0].get("function") or {}
                if fn.get("name"):
                    o_name = o_name or fn["name"]
                if fn.get("arguments"):
                    o_args += fn["arguments"]
            elif delta.get("content"):
                o_text = True

        # Gemini streaming chunks
        cands = ch.get("candidates")
        if isinstance(cands, list) and cands and isinstance(cands[0], dict):
            content = cands[0].get("content") or {}
            for p in content.get("parts") or []:
                if isinstance(p, dict):
                    if isinstance(p.get("functionCall"), dict) and g_call is None:
                        g_call = p["functionCall"]
                    elif isinstance(p.get("text"), str):
                        g_text = True

    if a_tool:
        try:
            inp = json.loads(a_buf) if a_buf.strip() else {}
        except (ValueError, TypeError):
            inp = {}
        return _sig_anthropic(a_name, inp)
    if o_tool:
        return _sig_openai(o_name, o_args)
    if g_call is not None:
        return _sig_gemini(g_call.get("name"), g_call.get("args"))
    if a_text or o_text or g_text:
        return "text"
    return "none"


def _sse_payloads(text: str) -> list[Any]:
    """Extract the JSON ``data:`` payloads from an SSE stream (skipping ``[DONE]``)."""
    out: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            out.append(json.loads(payload))
        except (ValueError, TypeError):
            continue
    return out


def decision_signature_from_body(raw: Any) -> str:
    """Decision signature for a raw response body — JSON, SSE stream, or chunk array.

    This is what makes shadow-mode work on **streaming** sessions (Claude Code,
    Codex, Gemini CLI all stream): a non-streaming JSON body is read directly; an
    SSE stream or a JSON array of chunks is reconstructed via
    :func:`_decision_from_chunks`. Returns the same ``tool:``/``text``/``none``
    signature as :func:`decision_signature`, so streamed and non-streamed responses
    compare correctly.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if not isinstance(raw, str):
        return decision_signature(raw)
    raw = raw.strip()
    if not raw:
        return "none"
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return _decision_from_chunks(_sse_payloads(raw))
    if isinstance(obj, dict):
        return decision_signature(obj)
    if isinstance(obj, list):
        return _decision_from_chunks(obj)
    return "none"


def force_deterministic(raw: bytes | None) -> bytes | None:
    """Rewrite a chat-completion *request* body to sample deterministically.

    The shadow gate exists to measure whether *compression* changes the agent's
    next decision — not whether the model happened to sample differently. A hot
    (temperature > 0) request answers the same prompt differently on repeat, which
    is why the v2 A/A self-agreement baseline read ~38% under live temps: pure
    sampling noise. Replaying both the served and replay sides at ``temperature 0``
    collapses that noise floor toward ~100%, so any remaining A/B disagreement is
    genuine compression drift.

    Only ``temperature`` is set — deliberately not ``top_p``/``seed``. Anthropic
    (Claude Code's own upstream) rejects unknown params and warns on temperature
    +top_p together; a 400 there would zero out exactly the samples we most need.
    On models that still expose it, temperature 0 gives greedy decoding and the
    decision signature (tool + normalized args) is coarse enough that it agrees ~100%.
    EXCEPTIONS where pinning 0 is forbidden (400) and the request is replayed as-sent
    — the A/A baseline then absorbs the residual sampling noise (see the body comment):
    extended ``thinking`` (requires temperature unset/1), and Opus 4.7+ (removed the
    temperature/top_p/top_k knobs entirely). We therefore only pin an EXISTING
    temperature and never inject one.

    Returns ``None`` when the body isn't a JSON object (not a chat request → skip
    the sample rather than compare it under hot sampling).
    """
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    # Pinning temperature 0 for a greedy replay must be applied carefully — done
    # unconditionally it 400'd ~every sampled request (observed: 295/323 replay_failed,
    # last_fail_reason 400), because two API constraints reject it:
    #   1. Extended thinking requires temperature unset/1 (Anthropic 400s on anything
    #      else), and Claude Code runs thinking by DEFAULT.
    #   2. Opus 4.7+ REMOVED temperature/top_p/top_k entirely — any value 400s — so the
    #      client already omits temperature on current models (e.g. claude-opus-4-8).
    # Injecting temperature where the client didn't send it, or overriding it under
    # thinking, is exactly what breaks. Rule: only pin an EXISTING temperature, and
    # never when thinking is on; otherwise replay as-sent (already API-valid). A model
    # that still accepts temperature is the only one that carries the field, so this
    # keeps greedy determinism where it's allowed and falls back to the A/A-noise-
    # adjusted comparison (aa_agreement / adjusted_rate) everywhere else.
    thinking = obj.get("thinking")
    # Thinking "on" in ANY form — the legacy ``enabled`` or the Opus 4.7+ ``adaptive``
    # (verified live: adaptive also 400s on `temperature != 1`), or any future type.
    # Only an explicit ``disabled`` / absent counts as off.
    thinking_on = isinstance(thinking, dict) and thinking.get("type") not in (None, "disabled")
    if "temperature" in obj and not thinking_on:
        obj["temperature"] = 0
    _strip_thinking_blocks(obj)
    return json.dumps(obj).encode("utf-8")


def _strip_thinking_blocks(obj: dict[str, Any]) -> None:
    """Remove prior-turn ``thinking``/``redacted_thinking`` blocks from a replay body.

    A thinking block is cryptographically signed and bound to the request that
    produced it. Replaying that conversation as a NEW request makes the provider
    re-validate the signature, which fails:

        400 messages.1.content.0: Invalid `signature` in `thinking` block

    Verified live, including the part that rules out the obvious workaround: the
    signature is validated even when the replay itself sets ``thinking`` off, so
    disabling thinking on the replay does not help.

    Claude Code runs extended thinking by DEFAULT, so most turns of a real session
    carry these blocks — which is why shadow could only ever sample the minority of
    turns that had none. That did not merely lower the sample count, it BIASED it:
    every published rate was computed over whichever turns happened to be
    thinking-free. Measured before this fix: 594 of 1,571 replays failed, ~94% of
    them 400s, and the A/A control stayed below its own n=30 reporting floor.

    Dropping the blocks is sound for the question shadow asks. The comparison is
    "did compression change the agent's NEXT action", and prior thinking is
    regenerated by the model rather than read back as evidence; both the served and
    replay sides are stripped identically, so neither side is advantaged. An
    assistant turn left with no content at all is dropped, since the API rejects an
    empty content array.
    """
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return
    kept: list[Any] = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            kept.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, list):
            kept.append(m)
            continue
        blocks = [
            b
            for b in content
            if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))
        ]
        if len(blocks) == len(content):
            kept.append(m)
        elif blocks:
            kept.append({**m, "content": blocks})
        # else: the turn was thinking-only — drop it rather than send empty content.
    obj["messages"] = kept


class ShadowSampler:
    """Probabilistic sampling: each request is shadowed independently with
    probability ``rate`` (in (0,1]; rate<=0 disables shadowing). Pass a seeded
    ``rng`` for deterministic tests. Independent draws avoid the phase-locking a
    fixed 1-in-N stride can hit against periodic traffic."""

    def __init__(self, rate: float, *, rng: random.Random | None = None) -> None:
        self.rate = max(0.0, min(1.0, rate))
        self._rng = rng or random

    def should_sample(self) -> bool:
        if self.rate <= 0:
            return False
        return self._rng.random() < self.rate


@dataclass
class ShadowLedger:
    """Rolling, content-free live decision-equivalence stats.

    Two sample kinds:
      * ``ab`` — compressed vs uncompressed (the classic shadow compare).
      * ``aa`` — the SAME compressed request replayed: the model's
        self-agreement on identical input. Raw A/B disagreement conflates
        compression harm with sampling nondeterminism (a Bash command worded
        two ways is a "changed decision" with zero compression involvement);
        the A/A baseline is what makes the A/B number interpretable.
    """

    window: int = 1000
    samples: int = 0  # A/B only — the meaning is unchanged from earlier versions
    changes: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=1000))
    aa_samples: int = 0
    aa_changes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        equivalent: bool,
        *,
        kind: str = "ab",
        evidence: dict[str, str] | None = None,
        path: Path | None = None,
    ) -> None:
        with self._lock:
            if kind == "aa":
                self.aa_samples += 1
                if not equivalent:
                    self.aa_changes += 1
            else:
                self.samples += 1
                if not equivalent:
                    self.changes += 1
                self.recent.append(1 if equivalent else 0)
        self._append(equivalent, kind, evidence, path)

    def rate(self) -> float:
        """Live A/B decision-CHANGE rate over the rolling window (0.0 = fully equivalent)."""
        with self._lock:
            if not self.recent:
                return 0.0
            return 1.0 - (sum(self.recent) / len(self.recent))

    def aa_agreement(self) -> float | None:
        """Model self-agreement on identical requests, or None below n=10
        (a baseline built on a handful of coin flips is worse than none)."""
        with self._lock:
            if self.aa_samples < 10:
                return None
            return 1.0 - (self.aa_changes / self.aa_samples)

    def adjusted_rate(self) -> float:
        """A/B change rate with the model's own nondeterminism factored out:
        agreement is judged relative to how often the model agrees with
        ITSELF on the identical request.

        WARNING: falls back to the RAW rate when no A/A baseline exists yet
        (aa_agreement() is None). In that state the return value is NOT
        noise-adjusted and conflates sampling nondeterminism with real harm.
        Any caller that renders a verdict (a health glyph, an "adjusted_*"
        field, a pass/fail) MUST gate on `aa_agreement() is not None` first —
        see cmd_statusline. This method stays total (returns a float) on
        purpose so best-effort display callers don't have to None-check."""
        base = self.aa_agreement()
        if base is None or base <= 0.0:
            return self.rate()
        return max(0.0, 1.0 - min(1.0, (1.0 - self.rate()) / base))

    def _append(
        self,
        equivalent: bool,
        kind: str,
        evidence: dict[str, str] | None,
        path: Path | None,
    ) -> None:
        try:
            p = path or (_state_dir() / "shadow.jsonl")
            p.parent.mkdir(parents=True, exist_ok=True)
            rec: dict[str, Any] = {
                "equivalent": bool(equivalent),
                "ts": time.time(),
                "kind": kind,
                "sig": SIG_VERSION,  # scope verdicts to the current algorithm
                "v": _current_version(),  # attribute the row to a build
            }
            if evidence:
                rec.update(evidence)  # content-free: signatures are _canon hashes
            with p.open("a", encoding="utf-8") as f:
                # Concurrent wrap sessions append here (shadow is on by default);
                # lock like ledger.py does — rc4 rows carry two signatures and can
                # exceed the pipe-atomicity size a bare append silently relies on.
                if _HAVE_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                finally:
                    if _HAVE_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass  # telemetry must never break the request path

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        current_only: bool = False,
        since_ts: float | None = None,
    ) -> ShadowLedger:
        """Read the shadow ledger.

        With ``current_only``, count ONLY rows written under the current
        ``SIG_VERSION`` — old-algorithm signatures are not comparable and must not
        drag a live verdict (a wording-jitter fix bumps the version, see ADR). Rows
        without a ``sig`` (pre-v2/legacy) are excluded when scoped. ``since_ts``
        drops rows older than a rolling window. Unscoped (default) reads everything,
        for auditing and backward compatibility with the certificate path.
        """
        led = cls()
        try:
            p = path or (_state_dir() / "shadow.jsonl")
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if current_only and rec.get("sig") != SIG_VERSION:
                    continue  # v1/legacy row — not comparable to current signatures
                if since_ts is not None and float(rec.get("ts", 0.0)) < since_ts:
                    continue
                eq = bool(rec.get("equivalent", True))
                if rec.get("kind", "ab") == "aa":
                    led.aa_samples += 1
                    if not eq:
                        led.aa_changes += 1
                else:  # pre-rc4 rows carry no kind — they were all A/B
                    led.samples += 1
                    if not eq:
                        led.changes += 1
                    led.recent.append(1 if eq else 0)
        except OSError:
            pass
        return led


class ShadowCounters:
    """Content-free sampling diagnostics — counts only, never prompt/response content.

    note_seen() / note_sampled() are safe to call from the request path (in-memory
    only, no I/O).  flush_with() is called from the background shadow thread and
    drains pending in-memory increments to disk via flock-guarded JSON, matching the
    ShadowLedger persistence style.
    """

    _FILENAME = "shadow_counters.json"

    #: Resolved once per process: `_current_version()` imports hotswap lazily, and
    #: this runs on every flush from the background shadow thread.
    _VERSION_CACHE: str | None = None

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (_state_dir() / self._FILENAME)
        self._lock = threading.Lock()
        self._pending: dict[str, int] = {}

    # ---- request-path helpers (in-memory only, no I/O) -------------------------

    def note_seen(self) -> None:
        """Increment requests_seen. In-memory only — safe for the hot request path."""
        with self._lock:
            self._pending["requests_seen"] = self._pending.get("requests_seen", 0) + 1

    def note_sampled(self) -> None:
        """Increment sampled. In-memory only — safe for the hot request path."""
        with self._lock:
            self._pending["sampled"] = self._pending.get("sampled", 0) + 1

    # ---- background-thread helper -----------------------------------------------

    def flush_with(
        self,
        *,
        replay_attempted: bool = False,
        replay_failed: bool = False,
        fail_reason: str = "",
        sig_none_skipped: bool = False,
        recorded: bool = False,
    ) -> None:
        """Drain pending in-memory counters + append outcome flags, then persist.

        Always called from the background shadow thread — never from the request path.
        Swallows all exceptions so counter writes never surface to callers.
        """
        try:
            with self._lock:
                deltas: dict[str, int] = dict(self._pending)
                self._pending.clear()
            if replay_attempted:
                deltas["replay_attempted"] = deltas.get("replay_attempted", 0) + 1
            if replay_failed:
                deltas["replay_failed"] = deltas.get("replay_failed", 0) + 1
            if sig_none_skipped:
                deltas["signature_none_skipped"] = deltas.get("signature_none_skipped", 0) + 1
            if recorded:
                deltas["recorded"] = deltas.get("recorded", 0) + 1
            self._write(deltas, fail_reason if replay_failed else "")
        except Exception:  # noqa: BLE001 — counter writes must never surface
            pass

    def _write(self, deltas: dict[str, int], fail_reason: str) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a+") as f:
                if _HAVE_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    try:
                        data: dict[str, Any] = json.load(f)
                    except (json.JSONDecodeError, ValueError):
                        data = {}
                    for k, v in deltas.items():
                        data[k] = data.get(k, 0) + v
                    # Per-version buckets, mirroring what ShadowLedger already does
                    # for its rows. Without them these counters accumulate for the
                    # life of the install, so failures from an ALREADY-FIXED bug stay
                    # in the displayed rate forever: the temperature-0 replay bug
                    # (fixed in 8c744df) left 295/323 failures behind, which held the
                    # lifetime rate at 42% while the real rate since the fix was 5.3%.
                    #
                    # Two costs, and the second is the one that matters: a fixed bug
                    # makes the sampler look broken, and the next REAL regression has
                    # to clear that stale noise floor before anyone can see it.
                    #
                    # Deliberately NOT resetting on a version change — that would lose
                    # the trend and make the opposite mistake, hiding a rate that
                    # degrades across releases. Lifetime totals stay for continuity.
                    by_version = data.get("by_version")
                    if not isinstance(by_version, dict):
                        by_version = {}
                    bucket = by_version.get(_counter_version())
                    if not isinstance(bucket, dict):
                        bucket = {}
                    for k, v in deltas.items():
                        bucket[k] = int(bucket.get(k, 0)) + v
                    if fail_reason:
                        # A histogram, not a single value. `last_fail_reason` keeps
                        # only the most recent, so a run mixing 400s, 429s and
                        # exceptions reported whichever landed last — which turned
                        # diagnosing the above into archaeology.
                        reasons = bucket.get("fail_reasons")
                        if not isinstance(reasons, dict):
                            reasons = {}
                        reasons[fail_reason] = int(reasons.get(fail_reason, 0)) + 1
                        bucket["fail_reasons"] = reasons
                    by_version[_counter_version()] = bucket
                    data["by_version"] = by_version
                    if fail_reason:
                        data["last_fail_reason"] = fail_reason
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f)
                finally:
                    if _HAVE_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path | None = None) -> dict[str, Any]:
        """Read the persisted counter JSON. Returns {} if missing or unreadable."""
        try:
            p = path or (_state_dir() / cls._FILENAME)
            if not p.exists():
                return {}
            with open(p) as f:
                if _HAVE_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                except (json.JSONDecodeError, ValueError):
                    return {}
                finally:
                    if _HAVE_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            return {}


def _counter_version() -> str:
    """The version bucket key for the shadow counters (memoised)."""
    if ShadowCounters._VERSION_CACHE is None:
        ShadowCounters._VERSION_CACHE = _current_version()
    return ShadowCounters._VERSION_CACHE


def compare_decisions(compressed_resp: Any, original_resp: Any) -> bool:
    """True iff the agent made the same decision with and without compression."""
    return decision_signature(compressed_resp) == decision_signature(original_resp)
