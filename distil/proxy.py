"""HTTP proxy that applies distil compression to LLM API requests.

Drop-in for any client that honours a ``base_url`` parameter — Anthropic SDK,
OpenAI SDK, LiteLLM, LangChain, etc. Point the client at the proxy and every
``/v1/messages``, ``/v1/chat/completions``, or ``/v1/responses`` request will
have its ``messages`` array compressed before being forwarded to the real
upstream. All other paths and methods are forwarded unchanged.

Usage
-----
::

    from distil.proxy import serve
    serve(host="127.0.0.1", port=8788, upstream="https://api.anthropic.com")

Or as a module::

    python -m distil.proxy
"""

from __future__ import annotations

import json
import os
import ssl
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ._log import log
from .adapters.anthropic import compress_messages
from .adapters.gemini import compress_generate_request
from .adapters.gemini import count_tokens
from .adapters.gemini import is_gemini_path
from .httpguard import parse_content_length, safe_forward_path, strip_query
from .otel import request_span, set_result_attrs
from .tokenizer import DEFAULT as _tokenizer

# ---------------------------------------------------------------------------
# Paths that carry a ``messages`` payload worth compressing
# ---------------------------------------------------------------------------

_COMPRESSIBLE_PATHS = frozenset({"/v1/messages", "/v1/chat/completions", "/v1/responses"})

# Hop-by-hop headers must never be forwarded; they are connection-specific.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailers",
        "upgrade",
    }
)

# A distil digest stub embeds an 8-hex content handle ("<< +N lines, handle=1a2b3c4d >>",
# columnar/delta variants). RestoreStore persists to disk, so a stub can outlive the
# request that created it and be expanded turns later.
_HANDLE_STUB_RE = re.compile(r"handle=[0-9a-fA-F]{6,}")


def _has_recoverable_stub(body: dict) -> bool:
    """True if the outgoing conversation still carries any distil digest handle.

    Checks ``messages`` (Anthropic/OpenAI Chat), ``contents`` (Gemini), and
    ``input`` (OpenAI Responses API) so cross-turn handle detection works for
    all request shapes.
    """
    try:
        msgs = body.get("messages") or body.get("contents") or body.get("input") or []
        blob = json.dumps(msgs)
    except (TypeError, ValueError):
        return False
    return _HANDLE_STUB_RE.search(blob) is not None


def _expand_should_intercept(expand: bool, store: object, body: dict) -> bool:
    """Whether the expand tool must be injected AND the response buffered to run the
    expand loop. True whenever expand mode is on and the outgoing conversation carries
    ANY recoverable handle — one created THIS request, or one that persisted from an
    earlier turn. Keying on ``store.handles`` alone (this request only) let a *streamed*
    turn that digested nothing new but referenced an older stub emit a ``distil_expand``
    tool_use with no tool injected and no expand loop, so the call escaped to the client
    as "No such tool available" (#25). Cheap case (new handles this request) short-circuits
    before the message scan.
    ponytail: buffering whenever a stub is in context costs streaming TTFT on long expand
    sessions; that is the price of never leaking an unresolvable tool call. Stream-intercept
    of the tool_use frame would recover TTFT if it ever matters."""
    if not expand:
        return False
    if getattr(store, "handles", None):
        return True
    return _has_recoverable_stub(body)


# Upstream socket timeout (seconds). Generous — LLM generations run minutes —
# but finite, so a wedged upstream can never pin a worker thread forever.
_UPSTREAM_TIMEOUT = float(os.environ.get("DISTIL_UPSTREAM_TIMEOUT", "600"))

# Client socket timeout (seconds) — the same bargain as the upstream one above,
# for the other end of the pipe. Without it (StreamRequestHandler.timeout is None)
# a client that opens a connection and then stops reading, or stops sending
# mid-request, pins its handler thread forever: the write blocks on a full socket
# buffer that will never drain. Generous, because HTTP/1.1 keep-alive means an
# idle agent between turns is sitting in exactly this read — but finite, so a
# stalled client leaks a thread for minutes instead of for the process's life.
_CLIENT_TIMEOUT = float(os.environ.get("DISTIL_CLIENT_TIMEOUT", "600"))


def _is_timeout(exc: urllib.error.URLError) -> bool:
    return isinstance(exc.reason, (socket.timeout, TimeoutError))


class QuietHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that doesn't traceback-spam on client disconnects.

    Agents (Claude Code especially) reset/abandon connections constantly —
    cancelled streams, retries, statusline polls. Those surface here as
    ConnectionResetError/BrokenPipeError in the handler thread; they are
    routine, not bugs. Everything else still gets the stdlib traceback.
    """

    def handle_error(self, request, client_address):  # noqa: ANN001 - stdlib signature
        import sys

        exc = sys.exc_info()[1]
        # socket.timeout (== TimeoutError on 3.10+) joins the list: with a client
        # socket timeout set, a stalled peer surfaces here on write the same way a
        # vanished one does. Upstream timeouts never reach this — they are caught
        # in the handler and answered 504.
        if isinstance(
            exc,
            (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, socket.timeout),
        ):
            return
        super().handle_error(request, client_address)


def _listen(host: str, port: int, handler: type) -> tuple[QuietHTTPServer, bool]:
    """The proxy's server, preferring a socket the supervisor already holds.

    Returns ``(server, activated)``. When *activated* is True the listening
    socket belongs to launchd/systemd, so connections queue in the kernel
    backlog across a crash-and-restart instead of being refused — see
    :mod:`distil.activation`. When it is False we bound the socket ourselves,
    which is the historical behaviour and works identically until we die.
    """
    from .activation import inherited_listener

    inherited = inherited_listener()
    if inherited is None:
        return QuietHTTPServer((host, port), handler), False
    # bind_and_activate=False so TCPServer does not bind a second socket to a
    # port the supervisor is already holding — that would raise EADDRINUSE and
    # turn a fault-tolerance feature into a startup failure.
    server = QuietHTTPServer((host, port), handler, bind_and_activate=False)
    server.socket.close()
    server.socket = inherited
    server.server_address = inherited.getsockname()
    # server_bind() normally sets these and we skipped it; BaseHTTPRequestHandler
    # reads server_name for the Host/Server headers it generates.
    server.server_name = socket.getfqdn(host)
    server.server_port = port
    return server, True


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Relay 3xx instead of following: auto-following would re-send the
    client's credentials to whatever host the upstream redirect names."""

    def redirect_request(self, *a, **k):  # noqa: ANN002, ANN003
        return None


#: CA-bundle variables, in the order the ecosystem already resolves them. distil
#: honours the names a corporate environment has ALREADY set for curl/requests
#: rather than inventing a distil-specific one nobody has exported.
_CA_BUNDLE_VARS = ("DISTIL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE")


def ca_bundle_path() -> str | None:
    """The CA bundle to trust for upstream TLS, or None for the system default.

    Behind an SSL-inspecting corporate proxy (Zscaler, Netskope, a company MITM
    box) every outbound TLS connection is re-signed by an internal CA. Python
    does not read the macOS keychain or the Windows store for this, so distil's
    upstream call fails certificate verification while curl and the browser —
    which do read them, or have been pointed at the corporate bundle — work
    fine. The user sees "upstream connection failed" and concludes distil is
    broken, which from where they are sitting is indistinguishable from true.

    Returns the first variable that names a file that actually exists. A path
    that does not exist is ignored rather than passed to OpenSSL, which would
    raise at context-construction time and take down the proxy at startup for a
    stale export in someone's shell profile.
    """
    for var in _CA_BUNDLE_VARS:
        val = (os.environ.get(var) or "").strip()
        if val and os.path.isfile(val):
            return val
    return None


def tls_failure_hint(reason: object) -> str | None:
    """Turn a certificate-verification failure into the sentence that fixes it.

    Raw, this arrives as ``[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
    failed: unable to get local issuer certificate``, which reads like distil is
    broken. It usually means one thing: an SSL-inspecting corporate proxy is
    re-signing traffic with a CA that Python does not trust. The user's browser
    and curl work, so distil looks uniquely at fault.

    Returns None for every other connection error — a hint about certificates on
    a DNS failure or a refused connection is worse than silence, because it sends
    the reader somewhere the problem is not.
    """
    text = str(reason)
    if "CERTIFICATE_VERIFY_FAILED" not in text and "SSLCertVerificationError" not in text:
        return None
    current = ca_bundle_path()
    if current is not None:
        return (
            f"TLS verification failed against the CA bundle at {current} "
            f"(from {_which_ca_var()}). That file exists but does not contain the "
            "issuer your network re-signs with — check it is the full corporate "
            "root chain, not a single leaf certificate."
        )
    return (
        "This is what an SSL-inspecting corporate proxy looks like: traffic is "
        "re-signed by an internal CA that Python does not trust, so curl and your "
        "browser work while distil does not. Point distil at your organisation's "
        "CA bundle, e.g. export REQUESTS_CA_BUNDLE=/path/to/corp-ca.pem (or "
        "SSL_CERT_FILE / CURL_CA_BUNDLE / DISTIL_CA_BUNDLE), then restart the proxy."
    )


def _which_ca_var() -> str:
    """Which variable supplied the active bundle — named so the user edits the
    right one when several are exported and only the first takes effect."""
    for var in _CA_BUNDLE_VARS:
        val = (os.environ.get(var) or "").strip()
        if val and os.path.isfile(val):
            return var
    return "(none)"


def _build_opener() -> urllib.request.OpenerDirector:
    """The upstream opener, with a corporate CA bundle applied when present.

    Verification is never disabled. There is no distil flag to turn it off, and
    that is deliberate: "it works with verification off" is how a proxy holding
    every prompt on the machine ends up trusting anything on the wire.
    """
    bundle = ca_bundle_path()
    if bundle is None:
        return urllib.request.build_opener(_NoRedirect)
    ctx = ssl.create_default_context(cafile=bundle)
    return urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=ctx))


_OPENER = _build_opener()


class _ErrStream:
    """Adapt a urllib error (or a synthetic status) to the streamexpand response
    interface — ``.status`` / ``.headers.items()`` / ``.read1(n)`` — so the streaming
    expand sender never raises and a non-2xx first response relays cleanly."""

    def __init__(self, status: int, headers: Any, body: bytes) -> None:
        self.status = status
        self.headers = headers  # http.client.HTTPMessage or dict — both expose .items()
        self._buf = body
        self._i = 0

    def read1(self, n: int) -> bytes:
        out = self._buf[self._i : self._i + n]
        self._i += len(out)
        return out


# ---------------------------------------------------------------------------
# Token-saving estimator
# ---------------------------------------------------------------------------


def _image_tokens(block: dict[str, Any]) -> int:
    """Billed token cost of a vision block, which is NOT its base64 length.

    Providers price an image by pixel area, so counting the base64 payload as
    text (or, as this function used to, not counting it at all) makes the
    baseline wrong for any agent that screenshots. Left uncounted, an elided
    duplicate scored as *negative* savings: zero on the before side, the
    replacement stub's tokens on the after side.
    """
    from .compress.vision import block_tokens

    # Shared with the eligibility census, which is compared against this baseline by an
    # exhaustiveness test — so the rule lives in one place rather than two.
    return block_tokens(block)


def _count_messages(msgs: list[dict[str, Any]]) -> int:
    """Heuristic token count of an Anthropic/OpenAI messages list."""
    total = 0
    for msg in msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _tokenizer.count(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "image":
                    total += _image_tokens(block)
                    continue
                for key in ("text", "content"):
                    val = block.get(key)
                    if isinstance(val, str):
                        total += _tokenizer.count(val)
                    elif isinstance(val, list):
                        for sub in val:
                            if isinstance(sub, dict):
                                if sub.get("type") == "image":
                                    total += _image_tokens(sub)
                                    continue
                                sv = sub.get("text", "")
                                if isinstance(sv, str):
                                    total += _tokenizer.count(sv)
    return total


def _adapter_census() -> dict[str, int] | None:
    """The eligibility census left by this thread's most recent compression pass.

    Read here rather than threaded through the call sites because all three relays
    (JSON, streaming, expand-splice) compress on the handler's own thread, and every
    entry point opens a fresh census — so the value is this request's by construction.
    """
    try:
        from .adapters.anthropic import take_census

        return take_census()
    except Exception:  # noqa: BLE001 — a diagnostic must never break a request
        return None


def _tokens_saved(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int:
    """Rough estimate of tokens saved via the default heuristic tokeniser."""
    return max(0, _count_messages(before) - _count_messages(after))


def _model_from_path(path: str) -> str | None:
    """Extract the model id from a Gemini-style URL (``.../models/<id>:action``)."""
    marker = "/models/"
    idx = path.find(marker)
    if idx < 0:
        return None
    tail = path[idx + len(marker) :]
    return tail.split(":", 1)[0].split("/", 1)[0] or None


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


_VERSION_CHECK_TTL = 30.0  # seconds between on-disk version re-checks (throttled, cheap)


def _warn_if_version_skew(state: dict[str, Any]) -> None:
    """Warn ONCE if distil was upgraded on disk while this long-lived proxy keeps
    running its old in-memory code — a running interpreter can't reload itself, so
    an in-place ``pip install -U`` leaves the proxy stale until it is restarted.

    Throttled by ``_VERSION_CHECK_TTL`` so it costs ~nothing per request. ``state``
    carries ``{"running": <version at start>, "checked": ts, "warned": bool}``.
    """
    import sys
    import time

    if state.get("warned"):
        return
    now = time.monotonic()
    if now - state.get("checked", 0.0) < _VERSION_CHECK_TTL:
        return
    state["checked"] = now
    try:
        from importlib.metadata import version as _pkg_version

        installed = _pkg_version("distil-llm")
    except Exception:  # noqa: BLE001 — a version check must never affect a request
        return
    running = state.get("running")
    if running and installed != running:
        state["warned"] = True
        print(
            f"distil: upgraded on disk to {installed}; this proxy still runs {running} "
            "— restart wrap to pick up the new version.",
            file=sys.stderr,
        )


def _mark_session_traffic() -> None:
    """Flip this wrap session's traffic marker to "1" — agent traffic reaches
    the proxy. Only acts when the marker exists (i.e. wrap_run created it), so
    a standalone `distil proxy` that happens to inherit DISTIL_SESSION never
    fabricates one."""
    from .ledger import session_marker_path

    mp = session_marker_path()
    try:
        if mp is not None and mp.exists():
            mp.write_text("1", encoding="utf-8")
    except OSError:
        pass


def build_handler(
    upstream: str,
    *,
    lossless_only: bool = False,
    verbatim: bool = False,
    shape_output: str = "off",
    savings: Any = None,
    flush_every: int = 10,
    expand: bool = False,
    shadow_rate: float = 0.0,
    retention_rate: float = 0.0,
    session_delta: bool = False,
) -> type[BaseHTTPRequestHandler]:
    """Return a ``BaseHTTPRequestHandler`` subclass configured for *upstream*.

    Parameters
    ----------
    upstream:
        Base URL of the real LLM API, e.g. ``"https://api.anthropic.com"``.
        Must not have a trailing slash.
    lossless_only:
        When *True* only Tier-0 lossless transforms are applied.
    shape_output:
        Output-compression level (``"off"``/``"light"``/``"aggressive"``). When
        not ``"off"`` and lossy compression is permitted, a verbosity-control
        ``role:"system"`` directive is appended so the model emits fewer tokens.
    """

    _upstream = upstream.rstrip("/")
    # A human-readable mode label, echoed on every compressed response as
    # x-distil-mode so a user seeing ▼0 can tell *why*: verbatim disables the
    # reversible digest (savings come only from lossless whitespace/JSON), so
    # ▼0 there is the mode, not a bug.
    _mode_label = "verbatim" if verbatim else ("lossless-only" if lossless_only else "digest")
    # Stamp the recorder so every savings row records the mode it was produced under
    # — answers "why was ▼ low?" from the ledger instead of by inference.
    if savings is not None:
        savings.mode = _mode_label

    # lossless-only is a hard safety boundary: with no injected expand tool the
    # agent can never recover a Tier-1 digest stub, so a stub there is irreversibly
    # lossy. Force Tier-0-only (verbatim) whenever lossless_only is set. The label
    # above stays distinct so x-distil-mode still reports which of the two it is.
    from .policy import AuthMode, may_compress_lossy

    # Route the lossy-allowed decision through policy as the single source of truth:
    # subscription / OAuth sessions are lossless-only (a tightening boundary a project
    # can never loosen). This forces Tier-0-only (verbatim) and gates output shaping.
    _auth_mode = AuthMode.SUBSCRIPTION if lossless_only else AuthMode.PAYG
    _lossy_ok = may_compress_lossy(_auth_mode)
    # lossless-only forces verbatim because an *unrecoverable* Tier-1 stub is
    # irreversibly lossy in-context. But `--expand` injects distil_expand, so every
    # stub IS recoverable — the very condition the verbatim-force guards against no
    # longer holds. So an explicit `--expand` lifts the force even on a subscription:
    # the user opted in, and nothing is irreversibly lost. The default (no --expand)
    # stays lossless-only. Output shaping stays gated on `_lossy_ok` (PAYG-only) — it
    # rewrites the *response*, which expand does not make recoverable. See issue #28.
    #
    # Recoverable by default: wherever lossy Tier-1 digest WILL run (any metered/PAYG
    # session that didn't force verbatim), turn the expand loop ON so every stub the
    # digest leaves is recoverable via distil_expand. streamexpand keeps this fully
    # streaming (it speculatively relays and only intercepts an actual expand call), so
    # there is no TTFT reason to leave a metered session emitting irreversible stubs —
    # the exact harm the subscription force-verbatim prevents, now closed on PAYG too.
    # An explicit --verbatim (verbatim=True) still wins; subscription stays lossless-only.
    if _lossy_ok and not verbatim:
        expand = True
    verbatim = verbatim or not (_lossy_ok or expand)
    # Disclose injection on EVERY route into it on a flat-rate session, not just
    # one. This used to be gated on `lossless_only and expand`, which is only the
    # `--lossless-only --expand` spelling. The other way in — `distil default
    # --mode expand`, which is what the docs tell a subscription user to run —
    # leaves lossless_only False, so the notice never fired and the user opted
    # into tool injection with no notice at all. Injecting into a first-party
    # session is precisely what onboard.py promises the safe default does not do,
    # so whoever leaves that path is owed a sentence saying so.
    #
    # Keyed on real billing, not on the flag: the flag says which spelling was
    # used, `subscription_mode()` says whether this is a flat-rate session.
    if expand:
        from .doctor import subscription_mode as _sub_mode

        if lossless_only or _sub_mode():  # startup-only; not on the request path
            import sys as _sys

            print(
                "distil: recoverable digest ON for a flat-rate session (you opted in "
                "with --expand). distil_expand is injected, so nothing is irreversibly "
                "lost — but the request IS modified, which the subscription-safe "
                "default never does. Drop --expand to go back to lossless-only.",
                file=_sys.stderr,
            )

    # Learning flywheel state (loaded once when expand is on): the learned
    # keep-byte-exact policy + the accumulating expand stats. See distil.learn.
    _learn_stats = None
    _expand_keep = None
    if expand:
        from . import query_flywheel
        from .learn import ExpandStats, keep_predicate

        _learn_stats = ExpandStats.load()
        _expand_keep = keep_predicate(_learn_stats)
        # Phase-2 dark collection: pair each digested block's dropped-line query-features with
        # whether it was later expanded (content-free, sampled). Only under --expand, where
        # handles are recoverable and expands produce the labels a retrain learns from.
        query_flywheel.enable()

    # Outcome-guided policy (always on — never-regressing by construction):
    # content classes whose digestion co-occurred with END-TO-END task
    # regressions are kept byte-exact. See distil.compress.guideline.
    from .compress.guideline import OutcomeStats

    _outcome_keep = OutcomeStats.load().keep_predicate()

    # Eager-load the other request-path module that handlers import lazily, so a
    # proxy upgraded in place never loads a post-upgrade .py mid-serve against the
    # already-running interpreter (version skew). guideline is warmed just above;
    # streamrelay is the remaining per-request import. Warmed here at server setup
    # the per-request `from .streamrelay import ...` is a module-cache hit, and CLI
    # cold start stays cheap because this is not run at `import distil` time.
    from .streamrelay import stream_upstream as _stream_upstream  # noqa: F401

    def _learn_keep(text: str) -> bool:
        return _outcome_keep(text) or (_expand_keep is not None and _expand_keep(text))

    # Shadow-mode live decision-equivalence: sample a fraction of requests, run the
    # decision uncompressed too (in the background), and record whether it matched.
    _shadow_sampler = None
    _shadow_ledger = None
    _shadow_counters = None
    _shadow_threads: list[threading.Thread] = []
    _shadow_threads_lock = threading.Lock()
    from . import __version__ as _running_version

    # Version-skew guard: a long-lived wrap proxy keeps running the code it started
    # with, even after an in-place upgrade. Stamp the running version; the request
    # path re-checks the installed version (throttled) and warns once on drift.
    _version_state: dict[str, Any] = {"running": _running_version}
    if shadow_rate and shadow_rate > 0:
        from .shadow import ShadowCounters, ShadowLedger, ShadowSampler

        _shadow_sampler = ShadowSampler(shadow_rate)
        _shadow_ledger = ShadowLedger()
        _shadow_counters = ShadowCounters()

    # Live fact-retention meter: same posture as shadow (sampled, off by default), but
    # cheaper — it needs no second upstream call, only an in-process scan, and it
    # persists counts alone.
    _retention_meter = None
    if retention_rate and retention_rate > 0:
        from .retention import LiveMeter

        _retention_meter = LiveMeter(retention_rate)

    # First-POST latch for the session traffic marker: only the 0→1 transition
    # matters, so after one write the check is a single list lookup.
    _traffic_seen = [False]

    class _DistilHandler(BaseHTTPRequestHandler):
        # HTTP/1.1 so streamed responses can use chunked transfer framing
        # (every non-streaming response still carries an exact Content-Length).
        protocol_version = "HTTP/1.1"

        # StreamRequestHandler.setup() applies this to the accepted socket, so
        # no read or write to a client can block forever. Read at class-creation
        # time (build_handler runs per server), so tests can dial it down.
        timeout = _CLIENT_TIMEOUT

        # ----------------------------------------------------------------
        # Silence request logs — quiet by design
        # ----------------------------------------------------------------

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
            pass

        # ----------------------------------------------------------------
        # HTTP verb dispatch
        # ----------------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            if not _traffic_seen[0]:
                _traffic_seen[0] = True
                _mark_session_traffic()
            _warn_if_version_skew(_version_state)
            p = strip_query(self.path)
            if p in _COMPRESSIBLE_PATHS or is_gemini_path(p):
                # Content-free integration-surface counter (census schema 3).
                try:
                    from . import surfaces as _surfaces

                    _surfaces.bump(p)
                except Exception:  # noqa: BLE001 — counting must never affect a request
                    pass
                self._handle_compressible()
            else:
                self._passthrough()

        def do_GET(self) -> None:  # noqa: N802
            if strip_query(self.path) == "/distil/health":
                self._respond_health()
                return
            self._passthrough()

        def _respond_health(self) -> None:
            # Liveness probe for load balancers/k8s: answers locally, never
            # touches the (billed) upstream and needs no auth.
            payload = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_PUT(self) -> None:  # noqa: N802
            self._passthrough()

        def do_DELETE(self) -> None:  # noqa: N802
            self._passthrough()

        def do_PATCH(self) -> None:  # noqa: N802
            self._passthrough()

        def do_HEAD(self) -> None:  # noqa: N802
            self._passthrough()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._passthrough()

        # ----------------------------------------------------------------
        # Helpers
        # ----------------------------------------------------------------

        def _read_body(self) -> bytes | None:
            """Read the request body; on a malformed/oversized/chunked request,
            send the error response itself and return None (caller just returns)."""
            if not self.headers.get("Content-Length") and "chunked" in (
                self.headers.get("Transfer-Encoding") or ""
            ):
                # A chunked body would otherwise be read as empty and silently
                # dropped — fail loudly instead (LLM SDKs always send a length).
                self._reject(411, "chunked request bodies are not supported; send Content-Length")
                return None
            length = parse_content_length(self.headers.get("Content-Length"))
            if length is None:
                self._reject(413, "request body too large or malformed Content-Length")
                return None
            return self.rfile.read(length) if length else b""

        def _reject(self, code: int, message: str) -> None:
            body = json.dumps({"error": message}).encode()
            self._relay(code, {"Content-Type": "application/json"}, body)

        def _client_headers(self, *, identity: bool = False) -> dict[str, str]:
            """Client headers with hop-by-hop stripped (Content-Length excluded
            so we can recompute it after compression). ``identity=True`` also
            drops Accept-Encoding: on compressible paths a gzip upstream body
            would silently defeat the expand loop and shadow decision parsing
            (both read the response), so those requests ask for identity."""
            out = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
            if identity:
                out = {k: v for k, v in out.items() if k.lower() != "accept-encoding"}
            return out

        def _relay(
            self,
            status: int,
            resp_headers: dict[str, str],
            resp_body: bytes,
            extras: dict[str, str] | None = None,
        ) -> None:
            """Write *status*, *resp_headers*, optional *extras*, and *resp_body* to caller."""
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() not in _HOP_BY_HOP:
                    self.send_header(k, v)
            if extras:
                for k, v in extras.items():
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def _post_upstream(
            self,
            path: str,
            body: bytes,
            headers: dict[str, str],
        ) -> tuple[int, dict[str, str], bytes]:
            """POST *body* to upstream *path*. Returns (status, headers, body)."""
            if safe_forward_path(path) is None:
                return (
                    400,
                    {"Content-Type": "application/json"},
                    b'{"error":"invalid request path"}',
                )
            url = _upstream + path
            req = urllib.request.Request(
                url,
                data=body,
                headers={**headers, "Content-Length": str(len(body))},
                method="POST",
            )
            # Module-cache hit: streamrelay is warmed at server setup (see above).
            from .streamrelay import capture_ratelimit

            try:
                with _OPENER.open(req, timeout=_UPSTREAM_TIMEOUT) as resp:
                    rbody = resp.read()
                    rhdrs = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
                    self._distil_ratelimit = capture_ratelimit(resp.headers)
                    return resp.status, rhdrs, rbody
            except urllib.error.HTTPError as exc:
                rbody = exc.read() if exc.fp else b'{"error":"upstream error"}'
                rhdrs = {k: v for k, v in exc.headers.items() if k.lower() not in _HOP_BY_HOP}
                self._distil_ratelimit = capture_ratelimit(exc.headers)
                return exc.code, rhdrs, rbody
            except urllib.error.URLError as exc:
                status = 504 if _is_timeout(exc) else 502
                payload: dict[str, Any] = {
                    "error": "upstream connection failed",
                    "detail": str(exc.reason)[:200],
                }
                hint = tls_failure_hint(exc.reason)
                if hint:
                    payload["hint"] = hint
                rbody = json.dumps(payload).encode()
                return status, {"Content-Type": "application/json"}, rbody
            except TimeoutError:
                rbody = b'{"error":"upstream timed out"}'
                return 504, {"Content-Type": "application/json"}, rbody

        # ----------------------------------------------------------------
        # Compression path
        # ----------------------------------------------------------------

        def _handle_compressible(self) -> None:
            if safe_forward_path(self.path) is None:
                self._reject(400, "invalid request path")
                return
            raw = self._read_body()
            if raw is None:
                return  # _read_body already sent the error response
            headers = self._client_headers(identity=True)
            extras: dict[str, str] = {}
            store: Any = None  # RestoreStore once messages are compressed (for expand)
            before_tok: int | None = None  # set only if a messages/gemini branch below runs
            after_tok: int | None = None
            # Savings are booked only after a confirmed 2xx (P0-1): (before, after, model).
            _pending_savings: tuple[int, int, str | None] | None = None

            try:
                body: dict[str, Any] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON — forward as-is, no extras.
                status, rhdrs, rbody = self._post_upstream(self.path, raw, headers)
                self._relay(status, rhdrs, rbody)
                return

            # Path-based dispatch: route to the right adapter per endpoint.
            # /v1/messages    → Anthropic adapter (compress_messages, below)
            # /v1/chat/completions → OpenAI Chat Completions adapter
            # /v1/responses   → OpenAI Responses API adapter (handled here first)
            # Gemini paths    → Gemini adapter (contents branch, below)
            _path = strip_query(self.path)

            if _path == "/v1/responses" and "input" in body and isinstance(body["input"], list):
                # OpenAI Responses API: compress ``function_call_output`` items
                # (Tier-1 reversible digest) and user ``message`` items (Tier-0).
                # Expand-tool injection for Responses API is not yet wired —
                # the tool schema differs from the assistant-turn inject used in
                # the messages path; see distil.adapters.openai module docstring.
                from .adapters.openai import compress_responses_input, count_responses_tokens

                _orig_input: list[dict[str, Any]] = body["input"]
                before_tok = count_responses_tokens(_orig_input)
                try:
                    _compressed_input, store = compress_responses_input(
                        _orig_input, verbatim=verbatim, keep=_learn_keep
                    )
                except Exception:  # noqa: BLE001 — compression must never break a request
                    log.debug(
                        "compress_responses_input failed; forwarding uncompressed", exc_info=True
                    )
                    _compressed_input, store = _orig_input, None
                after_tok = count_responses_tokens(_compressed_input)
                saved = max(0, before_tok - after_tok)
                body = {**body, "input": _compressed_input}
                extras = {
                    "x-distil-compressed": "1",
                    "x-distil-tokens-saved": str(saved),
                    "x-distil-mode": _mode_label,
                    "x-distil-compressible-tokens": str(before_tok),
                }
                if savings is not None:
                    _pending_savings = (before_tok, after_tok, body.get("model"))
                # Recoverable compression: inject distil_expand so the model can pull
                # back any digested block by handle — same gating as the messages path.
                if _expand_should_intercept(expand, store, body):
                    from .expand import inject_expand_tool_responses

                    body = inject_expand_tool_responses(body)
                # Output shaping: append verbosity directive to top-level ``instructions``.
                if shape_output != "off" and _lossy_ok:
                    from .output import shape_request

                    body = shape_request(body, level=shape_output, allow=True, shape="responses")
                    extras["x-distil-output-shaping"] = shape_output

            elif "messages" in body and isinstance(body["messages"], list):
                original: list[dict[str, Any]] = body["messages"]
                # Cache-delta coding (opt-in): cross-turn dedup + cross-version delta,
                # applied to the ORIGINALS before compression so re-reads match across
                # turns. Cache-monotonic (suffix-only) and reversible.
                pre = original
                _dstats = None
                _dstore = None
                if session_delta:
                    try:
                        from .cachedelta import delta_encode, get_session, session_key

                        _sess = get_session(session_key(original))
                        pre, _dstore, _dstats = delta_encode(original, session=_sess)
                    except Exception:  # noqa: BLE001 — never break a request
                        log.debug("cache-delta encode failed", exc_info=True)
                        pre, _dstore, _dstats = original, None, None
                # Dispatch to the right compressor: OpenAI Chat Completions needs
                # a dedicated adapter (role:"tool" list content is Tier-1; the
                # Anthropic adapter applies Tier-0 to generic list text items).
                # /v1/messages stays on the Anthropic adapter.
                if _path == "/v1/chat/completions":
                    from .adapters.openai import compress_chat_completions

                    _compress_fn = compress_chat_completions
                else:
                    _compress_fn = compress_messages
                try:
                    compressed, store = _compress_fn(pre, verbatim=verbatim, keep=_learn_keep)
                except Exception:  # noqa: BLE001 — compression must never break a request
                    log.debug("compress_messages failed; forwarding uncompressed", exc_info=True)
                    compressed, store = pre, None
                # Merge cache-delta references into the store so distil_expand recovers them.
                if _dstore is not None and store is not None:
                    for _h in _dstore.handles:
                        try:
                            store._record(_h, _dstore.expand(_h))
                        except Exception:  # noqa: BLE001
                            pass
                # Learning: tally what we digested, by content-free signature.
                if _learn_stats is not None and getattr(store, "handles", None):
                    from .learn import signature

                    for h in store.handles:
                        try:
                            _learn_stats.record_digest(signature(store.expand(h)))
                        except Exception:  # noqa: BLE001 — learning never breaks a request
                            log.debug("learning tally failed", exc_info=True)
                # Live retention meter: sampled, content-free (counts only), fail-open.
                if _retention_meter is not None and _retention_meter.enabled:
                    _retention_meter.observe(original, compressed, store)
                before_tok = _count_messages(original)
                after_tok = _count_messages(compressed)
                saved = max(0, before_tok - after_tok)
                body = {**body, "messages": compressed}
                extras = {
                    "x-distil-compressed": "1",
                    "x-distil-tokens-saved": str(saved),
                    "x-distil-mode": _mode_label,
                    # Bytes in the compressible zone (user/tool content distil is
                    # allowed to touch) — when this is ~0, a ▼0 is "nothing large
                    # to compress this turn", not a failure. System prompt, tool
                    # definitions, images and assistant text are never counted.
                    "x-distil-compressible-tokens": str(_count_messages(original)),
                }
                if _dstats is not None:
                    extras["x-distil-cache-refs"] = str(_dstats.exact_refs + _dstats.delta_refs)
                    extras["x-distil-cache-delta"] = str(_dstats.delta_refs)
                    extras["x-distil-cache-tokens-saved"] = str(_dstats.tokens_saved)
                    # Cache-prefix observability: how many leading messages were
                    # byte-stable vs the previous turn (the prompt-cache-read region).
                    # Stateful, content-free — the verifiable benefit of a prefix-freeze
                    # router, without the lossy rewrite (distil is cache-monotonic).
                    extras["x-distil-cache-prefix-msgs"] = str(_dstats.prefix_msgs)
                # Recoverable compression: if anything was digested, offer the model
                # the distil_expand tool so it can pull back detail on demand.
                if _expand_should_intercept(expand, store, body):
                    from .expand import inject_expand_tool

                    body = inject_expand_tool(body)
                # Accumulate GENUINE savings from real traffic into the ledger,
                # priced per the model THIS request names (agents mix models).
                if savings is not None:
                    _pending_savings = (before_tok, after_tok, body.get("model"))
                # Output compression: gated by lossless_only (only on PAYG-style).
                if shape_output != "off" and _lossy_ok:
                    from .output import shape_request

                    _shape = "anthropic" if _path == "/v1/messages" else "openai"
                    body = shape_request(body, level=shape_output, allow=True, shape=_shape)
                    extras["x-distil-output-shaping"] = shape_output

            elif "contents" in body and isinstance(body["contents"], list):
                # Gemini generateContent shape. Content compression + output shaping.
                before_tok = count_tokens(body)
                try:
                    body, store = compress_generate_request(
                        body, verbatim=verbatim, keep=_learn_keep
                    )
                except Exception:  # noqa: BLE001 — compression must never break a request
                    log.debug("gemini compression failed; forwarding uncompressed", exc_info=True)
                    store = None
                if _learn_stats is not None and getattr(store, "handles", None):
                    from .learn import signature

                    for h in store.handles:
                        try:
                            _learn_stats.record_digest(signature(store.expand(h)))
                        except Exception:  # noqa: BLE001 — learning never breaks a request
                            log.debug("learning tally failed", exc_info=True)
                after_tok = count_tokens(body)
                saved = max(0, before_tok - after_tok)
                extras = {
                    "x-distil-compressed": "1",
                    "x-distil-tokens-saved": str(saved),
                    "x-distil-mode": _mode_label,
                    "x-distil-compressible-tokens": str(before_tok),
                }
                if savings is not None:
                    # Gemini requests carry the model in the URL path, not the body.
                    _pending_savings = (before_tok, after_tok, _model_from_path(self.path))
                # Output shaping: inject systemInstruction directive (PAYG only).
                if shape_output != "off" and _lossy_ok:
                    from .output import shape_request

                    body = shape_request(body, level=shape_output, allow=True, shape="gemini")
                    extras["x-distil-output-shaping"] = shape_output
                # Expand-tool injection (Gemini): offer distil_expand under functionDeclarations
                # so the model can recover any digested block by handle.  Same PAYG/--expand
                # gating as the messages path — _expand_should_intercept checks store.handles
                # (this request) and contents stubs (cross-turn persistence).
                if _expand_should_intercept(expand, store, body):
                    from .expand import inject_expand_tool_gemini

                    body = inject_expand_tool_gemini(body)

            new_raw = json.dumps(body).encode()
            _span_model = body.get("model") or _model_from_path(self.path) or "unknown"

            # Decide shadow sampling BEFORE relaying so the marker header can be
            # sent on the streaming path too (headers go out before the body).
            shadow_sampled = _shadow_sampler is not None and _shadow_sampler.should_sample()
            if _shadow_sampler is not None and _shadow_counters is not None:
                _shadow_counters.note_seen()
            if shadow_sampled and _shadow_counters is not None:
                _shadow_counters.note_sampled()
            if shadow_sampled:
                extras["x-distil-shadow"] = "sampled"

            # Streaming: relay upstream bytes as they arrive so time-to-first-token is
            # preserved. Recoverable-digest requests used to fall back to the buffered
            # expand loop (losing TTFT); streamexpand now speculatively streams and
            # intercepts a distil_expand call mid-stream, splicing the re-query so the
            # agent keeps streaming AND the digest stays recoverable.
            want_stream = bool(body.get("stream")) or ":streamGenerateContent" in self.path
            t_req = time.monotonic()  # upstream + relay latency (compression excluded)
            if want_stream and _expand_should_intercept(expand, store, body):
                from .streamexpand import stream_with_expand

                def _send_stream(_b: dict[str, Any]) -> Any:
                    _rb = json.dumps(_b).encode()
                    _req = urllib.request.Request(
                        _upstream + self.path,
                        data=_rb,
                        headers={**headers, "Content-Length": str(len(_rb))},
                        method="POST",
                    )
                    try:
                        return _OPENER.open(_req, timeout=_UPSTREAM_TIMEOUT)
                    except urllib.error.HTTPError as exc:
                        return _ErrStream(exc.code, exc.headers, exc.read() if exc.fp else b"")
                    except (urllib.error.URLError, TimeoutError) as exc:
                        _st = 504 if isinstance(exc, TimeoutError) or _is_timeout(exc) else 502
                        return _ErrStream(
                            _st,
                            {"Content-Type": "application/json"},
                            b'{"error":"upstream connection failed"}',
                        )

                _usage_x: dict[str, int] = {}
                with request_span(_span_model, self.path) as _span:
                    status_x = stream_with_expand(
                        self,
                        _send_stream,
                        body,
                        store,
                        hop_by_hop=_HOP_BY_HOP,
                        extras=extras,
                        usage_sink=_usage_x,
                    )
                    set_result_attrs(
                        _span,
                        original_tokens=before_tok,
                        compressed_tokens=after_tok,
                        compression_ratio=(
                            after_tok / before_tok if before_tok and after_tok is not None else None
                        ),
                        compressed="x-distil-compressed" in extras,
                        shadow_sampled=shadow_sampled,
                    )
                if _learn_stats is not None:
                    _learn_stats.save()
                if savings is not None and _pending_savings is not None and 200 <= status_x < 300:
                    _bt, _at, _m = _pending_savings
                    savings.record(_bt, _at, model=_m)
                    savings.maybe_flush(every=flush_every)
                self._emit_detail(
                    extras=extras,
                    store=store,
                    body=body if isinstance(body, dict) else None,
                    model=_span_model,
                    stream=True,
                    client_stream=want_stream,
                    status=status_x,
                    booked=(
                        savings is not None
                        and _pending_savings is not None
                        and 200 <= status_x < 300
                    ),
                    duration_ms=int((time.monotonic() - t_req) * 1000),
                    usage=_usage_x,
                )
                if shadow_sampled:
                    self._spawn_shadow(raw, headers, new_raw)
                return
            if want_stream:
                from .streamrelay import stream_upstream

                _usage_s: dict[str, int] = {}
                with request_span(_span_model, self.path) as _span:
                    status_s, _rbody_opt = stream_upstream(
                        self,
                        _upstream + self.path,
                        new_raw,
                        headers,
                        timeout=_UPSTREAM_TIMEOUT,
                        hop_by_hop=_HOP_BY_HOP,
                        extras=extras,
                        want_body=False,  # v3 shadow re-issues its own temp-0 calls; no need to buffer
                        usage_sink=_usage_s,
                    )
                    set_result_attrs(
                        _span,
                        original_tokens=before_tok,
                        compressed_tokens=after_tok,
                        compression_ratio=(
                            after_tok / before_tok if before_tok and after_tok is not None else None
                        ),
                        compressed="x-distil-compressed" in extras,
                        shadow_sampled=shadow_sampled,
                    )
                if _learn_stats is not None:
                    _learn_stats.save()
                # Book savings only after a fully-relayed 2xx (P0-1).
                if savings is not None and _pending_savings is not None and 200 <= status_s < 300:
                    _bt, _at, _m = _pending_savings
                    savings.record(_bt, _at, model=_m)
                    savings.maybe_flush(every=flush_every)
                self._emit_detail(
                    extras=extras,
                    store=store,
                    body=body if isinstance(body, dict) else None,
                    model=_span_model,
                    stream=True,
                    client_stream=want_stream,
                    status=status_s,
                    booked=(
                        savings is not None
                        and _pending_savings is not None
                        and 200 <= status_s < 300
                    ),
                    duration_ms=int((time.monotonic() - t_req) * 1000),
                    usage=_usage_s,
                )
                if shadow_sampled:
                    self._spawn_shadow(raw, headers, new_raw)
                return

            with request_span(_span_model, self.path) as _span:
                status, rhdrs, rbody = self._post_upstream(self.path, new_raw, headers)
                set_result_attrs(
                    _span,
                    original_tokens=before_tok,
                    compressed_tokens=after_tok,
                    compression_ratio=(
                        after_tok / before_tok if before_tok and after_tok is not None else None
                    ),
                    compressed="x-distil-compressed" in extras,
                    shadow_sampled=shadow_sampled,
                )

            # Transparent expand loop: resolve any distil_expand tool calls against
            # the local store and re-query, invisibly, before returning to the agent.
            # Dispatches to the Gemini loop (contents/functionCall shape) or the
            # Anthropic/OpenAI loop (messages/tool_use shape) based on body type.
            _expanded_handles: list[str] = []
            if _expand_should_intercept(expand, store, body):
                try:
                    resp_json = json.loads(rbody)
                except (ValueError, TypeError):
                    resp_json = None
                if isinstance(resp_json, dict):
                    from .expand import (
                        record_signal,
                        run_expand_loop,
                        run_expand_loop_gemini,
                        run_expand_loop_responses,
                    )

                    def _post(b: dict[str, Any]) -> dict[str, Any]:
                        _s, _h, rb = self._post_upstream(self.path, json.dumps(b).encode(), headers)
                        return json.loads(rb)

                    def _on_signal(handle: str, original: str) -> None:
                        _expanded_handles.append(handle)
                        record_signal(handle, original)  # content-free expand log
                        if _learn_stats is not None:  # learn the expanded signature
                            from .learn import signature

                            _learn_stats.record_expand(signature(original))

                    if "contents" in body and isinstance(body.get("contents"), list):
                        final = run_expand_loop_gemini(
                            body, resp_json, store, _post, on_signal=_on_signal
                        )
                    elif "input" in body and isinstance(body.get("input"), list):
                        final = run_expand_loop_responses(
                            body, resp_json, store, _post, on_signal=_on_signal
                        )
                    else:
                        final = run_expand_loop(body, resp_json, store, _post, on_signal=_on_signal)
                    if final is not resp_json:
                        rbody = json.dumps(final).encode()
                        extras["x-distil-expanded"] = "1"
            if _learn_stats is not None:  # persist the learned policy (atomic)
                _learn_stats.save()

            # Shadow-mode: on a sampled request, re-run the decision UNCOMPRESSED in
            # the background and record whether it matched — a live decision-change
            # signal on real traffic. Never blocks the client's response.
            if shadow_sampled:
                self._spawn_shadow(raw, headers, new_raw)
            _usage_b: dict[str, int] = {}
            try:
                from .streamrelay import scan_usage

                _usage_b = scan_usage(rbody[:16384] + b"\n" + rbody[-16384:])
            except Exception:  # noqa: BLE001 — usage capture is bookkeeping only
                pass
            # Per-request detail written synchronously before the relay: this guarantees
            # a record for every request (deterministic, none lost on abrupt shutdown) —
            # the property dissect relies on. The write is a bounded ~5-15ms of local disk
            # I/O and is fail-open. (A future async-write could shave that latency without
            # losing the guarantee; deliberately not doing it under a synchronous contract.)
            self._emit_detail(
                extras=extras,
                store=store,
                body=body if isinstance(body, dict) else None,
                model=_span_model,
                stream=False,
                client_stream=want_stream,
                status=status,
                booked=(
                    savings is not None and _pending_savings is not None and 200 <= status < 300
                ),
                duration_ms=int((time.monotonic() - t_req) * 1000),
                usage=_usage_b,
                expanded_handles=_expanded_handles,
            )
            # Book savings only after a confirmed 2xx (P0-1): failed or SDK-retried
            # upstream calls must not be counted as savings.
            if savings is not None and _pending_savings is not None and 200 <= status < 300:
                _bt, _at, _m = _pending_savings
                savings.record(_bt, _at, model=_m)
                savings.maybe_flush(every=flush_every)
            self._relay(status, rhdrs, rbody, extras=extras)

        def _emit_detail(
            self,
            *,
            extras: dict[str, str],
            store: Any,
            body: dict[str, Any] | None,
            model: str,
            stream: bool,
            client_stream: bool,
            status: int,
            booked: bool,
            duration_ms: int | None = None,
            usage: dict[str, int] | None = None,
            expanded_handles: list[str] | None = None,
        ) -> None:
            """Append one content-free per-request record to the wrap session's
            ``sessions/<sid>.requests.jsonl`` (read by ``distil dissect``).
            Records token accounting, per-block digest signatures (handle + kind
            + size only — never content), and shadow/expand flags. Best-effort:
            any failure is a debug log — bookkeeping must never break a request."""
            # Per-request receipt — the exportable, hash-chained artifact. Emitted for
            # every 2xx the proxy serves: not only inside a `wrap` session, and not only
            # when a savings ledger happens to be attached (`booked`). A compliance
            # record that disappears depending on how you launched the proxy, or on
            # whether accounting is on, is not a compliance record. Own try/except so it
            # cannot be skipped by — or skip — the diagnostic record below.
            if 200 <= int(status or 0) < 300:
                try:
                    import hashlib as _hashlib

                    from . import receipts as _receipts

                    _ts = time.time()
                    _handles = sorted(getattr(store, "handles", None) or ())
                    # "restorable" is measured, not assumed: every handle this request
                    # issued must resolve right now, or the receipt says it did not.
                    _restorable = True
                    for _h in _handles:
                        try:
                            if store.expand(_h) is None:
                                _restorable = False
                                break
                        except Exception:  # noqa: BLE001 — unresolvable counts as not restorable
                            _restorable = False
                            break
                    _mode = extras.get("x-distil-mode", "verbatim")
                    _saved = int(extras.get("x-distil-tokens-saved", 0) or 0)
                    _orig = int(extras.get("x-distil-compressible-tokens", 0) or 0)
                    _receipts.append(
                        _receipts.Receipt(
                            ts=_ts,
                            request_id=_hashlib.sha256(
                                f"{_ts}:{model}:{_saved}:{','.join(_handles)}".encode()
                            ).hexdigest()[:16],
                            session=str(os.environ.get("DISTIL_SESSION") or ""),
                            model=str(model or "unknown"),
                            mode=_mode,
                            tokens_original=_orig,
                            tokens_compressed=max(0, _orig - _saved),
                            # Tier-0 (verbatim/lossless) round-trips byte-exact. digest is
                            # recoverable-on-demand via a handle — a weaker claim, so it is
                            # not reported as reversible.
                            reversible=_mode in ("verbatim", "lossless"),
                            handles=list(_handles),
                            restorable=_restorable,
                            certificate=str(extras.get("x-distil-certificate", "")),
                        )
                    )
                except Exception:  # noqa: BLE001 — a receipt must never break a request
                    log.debug("receipt emit failed", exc_info=True)

            try:
                from . import ledger

                if ledger.session_requests_path() is None:
                    return  # not a wrap session; nothing to attribute the request to
                from .learn import signature

                blocks: list[dict[str, Any]] = []
                for h in sorted(getattr(store, "handles", None) or ()):
                    try:
                        text = store.expand(h)
                        blocks.append(
                            {"h": h, "sig": signature(text), "tokens": _tokenizer.count(text)}
                        )
                    except Exception:  # noqa: BLE001 — one bad handle must not drop the record
                        continue
                system_tok = tools_tok = 0
                tool_costs: list[dict[str, Any]] = []
                if isinstance(body, dict):
                    sys_val = body.get("system")
                    if sys_val:
                        system_tok = _tokenizer.count(
                            sys_val if isinstance(sys_val, str) else json.dumps(sys_val)
                        )
                    for tool in body.get("tools") or []:
                        try:
                            n = _tokenizer.count(json.dumps(tool))
                            tools_tok += n
                            name = tool.get("name") if isinstance(tool, dict) else None
                            tool_costs.append({"name": str(name or "?"), "tokens": n})
                        except Exception:  # noqa: BLE001 — one odd tool must not drop the rest
                            continue
                    tool_costs.sort(key=lambda t: -t["tokens"])
                overhead = system_tok + tools_tok
                # Full billed input = uncached + cached prefix. Prompt caching bills the cached
                # prefix under cache_read/cache_creation, NOT input_tokens — so the true input
                # the heuristic estimates is the sum. Comparing est to input_tokens alone (as the
                # old dissect.calibration did) is apples-to-oranges on cached traffic.
                _u = usage or {}
                _cache = int(_u.get("cache_read_input_tokens", 0) or 0) + int(
                    _u.get("cache_creation_input_tokens", 0) or 0
                )
                _billed_full = int(_u.get("input_tokens") or 0) + _cache
                # A+C: feed the (heuristic estimate, full billed) pairing into the token calibrator
                # so reported counts converge to the real tokenizer. Content-free, fail-open.
                if _billed_full > 0:
                    _est = overhead + max(
                        0,
                        int(extras.get("x-distil-compressible-tokens", 0) or 0)
                        - int(extras.get("x-distil-tokens-saved", 0) or 0),
                    )
                    from . import calibration

                    calibration.record(str(model or "unknown"), _est, _billed_full)
                _prefix_hash, _prefix_bytes = "", 0
                if isinstance(body, dict):
                    try:
                        from . import prefix as _prefix

                        _stable = {k: body[k] for k in _prefix.STABLE_KEYS if k in body}
                        _rep = _prefix.analyse(None, _stable)
                        _prefix_hash, _prefix_bytes = _rep.stable_hash, _rep.stable_bytes
                    except Exception:  # noqa: BLE001 — a diagnostic must not drop the record
                        pass
                rec = {
                    "ts": time.time(),
                    "model": model,
                    "stream": stream,
                    "client_stream": client_stream,
                    "status": status,
                    "booked": booked,
                    "duration_ms": duration_ms,
                    "usage_input_tokens": (usage or {}).get("input_tokens"),
                    "usage_output_tokens": (usage or {}).get("output_tokens"),
                    "usage_cache_tokens": _cache or None,
                    # Split, because the sum cannot tell a working cache from a thrashing
                    # one: a write is a 25% surcharge, a read a ~90% discount, and a prefix
                    # that drifts every turn writes forever and never reads — which looks
                    # identical to a healthy cache once the two are added together.
                    "usage_cache_read": int(_u.get("cache_read_input_tokens", 0) or 0) or None,
                    "usage_cache_create": int(_u.get("cache_creation_input_tokens", 0) or 0)
                    or None,
                    # Content-free fingerprint of the stable prefix we actually sent. Lets
                    # `distil cache` say WHERE a prefix broke between two turns instead of
                    # only that the provider re-billed it.
                    "prefix_hash": _prefix_hash,
                    "prefix_bytes": _prefix_bytes,
                    "expanded_handles": expanded_handles or [],
                    "mode": extras.get("x-distil-mode", "verbatim"),
                    "compressible_tokens": int(extras.get("x-distil-compressible-tokens", 0) or 0),
                    "tokens_saved": int(extras.get("x-distil-tokens-saved", 0) or 0),
                    "overhead_tokens": overhead,
                    "system_tokens": system_tok,
                    "tools_tokens": tools_tok,
                    "tools": tool_costs[:24],  # names + token counts only; content-free
                    "delta_refs": int(extras.get("x-distil-cache-refs", 0) or 0),
                    "delta_tokens_saved": int(extras.get("x-distil-cache-tokens-saved", 0) or 0),
                    "prefix_msgs": int(extras.get("x-distil-cache-prefix-msgs", 0) or 0),
                    # Provider-reported quota state (counters/timestamps only). Billed
                    # tokens say what was sent; this says what it cost the plan's budget.
                    "ratelimit": getattr(self, "_distil_ratelimit", None),
                    # Why this request compressed as much (or as little) as it did:
                    # tokens bucketed by the gate that claimed each block. Without it,
                    # a low saving is only explicable by inference from outside — and
                    # inference cannot tell "mostly assistant prose" (working as designed)
                    # from "the digester declined" (a defect) from "mostly recent"
                    # (transient). Content-free; see adapters.anthropic.take_census.
                    "census": _adapter_census(),
                    "shadow_sampled": extras.get("x-distil-shadow") == "sampled",
                    "expanded": extras.get("x-distil-expanded") == "1",
                    "output_shaping": extras.get("x-distil-output-shaping", ""),
                    "blocks": blocks,
                }
                ledger.append_session_request(rec)

            except Exception:  # noqa: BLE001 — bookkeeping must never break a request
                log.debug("request-detail record failed", exc_info=True)

        def _spawn_shadow(
            self,
            orig_raw: bytes,
            headers: dict[str, str],
            compressed_raw: bytes,
        ) -> None:
            """Re-run a sampled request in the background and record whether the
            agent's decision matched. Never blocks the client's response.

            v3: both sides are re-issued at temperature 0 (see force_deterministic),
            never reusing the live served response. Two sample kinds (see
            ShadowLedger): most replays are A/B (compressed vs original — did
            compression change the decision?); a third are A/A (the SAME compressed
            request twice — a provider-honesty probe that must read ~100% at temp 0).
            v2 compared hot samples, so A/A read ~38% noise and buried the A/B signal."""
            import hashlib
            import random as _random

            is_aa = (
                _random.random() < 1 / 3
            )  # ponytail: fixed 1/3 split; make configurable if the baseline needs tuning

            def _shadow_compare() -> None:
                _attempted = False
                _failed = False
                _fail_reason = ""
                _skipped = False
                _written = False
                try:
                    from .shadow import decision_signature_from_body, force_deterministic

                    # Re-issue BOTH sides at temperature 0 — never reuse the live
                    # served response (produced at the agent's hot temperature).
                    # A non-JSON body can't be made deterministic; skip it rather
                    # than fall back to a hot comparison that re-poisons the baseline.
                    served_det = force_deterministic(compressed_raw)
                    replay_det = force_deterministic(compressed_raw if is_aa else orig_raw)
                    if served_det is None or replay_det is None:
                        return  # not deterministic; flush pending seen/sampled via finally
                    _attempted = True
                    _s1, _h1, served_rbody = self._post_upstream(self.path, served_det, headers)
                    _s2, _h2, replay_rbody = self._post_upstream(self.path, replay_det, headers)
                    if not (200 <= _s1 < 300 and 200 <= _s2 < 300):
                        _failed = True
                        _fail_reason = str(_s2 if not (200 <= _s2 < 300) else _s1)
                    # decision_signature_from_body handles both JSON and streamed
                    # (SSE / chunk-array) bodies, so this works for Claude Code /
                    # Codex / Gemini sessions, which stream their responses.
                    comp_sig = decision_signature_from_body(served_rbody)
                    replay_sig = decision_signature_from_body(replay_rbody)
                    # "none" means no decision could be extracted (transient upstream
                    # error or empty/unparseable body). Recording it as agreement or
                    # change would inflate the decision-equivalence rate on noise.
                    if _shadow_ledger is not None and "none" not in (comp_sig, replay_sig):
                        _shadow_ledger.record(
                            comp_sig == replay_sig,
                            kind="aa" if is_aa else "ab",
                            # Evidence for diagnosing divergences: which request
                            # (digest) produced which pair of decisions. All three
                            # values are content-free hashes/signatures.
                            evidence={
                                "digest": hashlib.sha256(orig_raw).hexdigest()[:16],
                                "sig_served": comp_sig,
                                "sig_replay": replay_sig,
                            },
                        )
                        _written = True
                    elif "none" in (comp_sig, replay_sig):
                        _skipped = True
                except Exception:  # noqa: BLE001 — shadow must never affect the request
                    log.debug("shadow compare failed", exc_info=True)
                    if _attempted and not _written:
                        _failed, _fail_reason = True, "exception"
                finally:
                    if _shadow_counters is not None:
                        _shadow_counters.flush_with(
                            replay_attempted=_attempted,
                            replay_failed=_failed,
                            fail_reason=_fail_reason,
                            sig_none_skipped=_skipped,
                            recorded=_written,
                        )

            # Track the thread so teardown can drain it: a daemon thread is
            # killed on process exit, which loses the sample on a quick run
            # (e.g. `claude -p`) or right after the last turn. Prune finished
            # ones first so the list stays bounded on long sessions.
            _t = threading.Thread(target=_shadow_compare, daemon=True)
            with _shadow_threads_lock:
                # Prune finished threads and append under one lock — concurrent
                # sampled requests otherwise race here and drop a thread, which
                # _drain_shadow would then miss on shutdown.
                _shadow_threads[:] = [t for t in _shadow_threads if t.is_alive()]
                _shadow_threads.append(_t)
            _t.start()

        # ----------------------------------------------------------------
        # Transparent passthrough (unchanged body, any verb)
        # ----------------------------------------------------------------

        def _passthrough(self) -> None:
            if safe_forward_path(self.path) is None:
                self._reject(400, "invalid request path")
                return
            raw = self._read_body()
            if raw is None:
                return  # _read_body already sent the error response
            from .streamrelay import stream_upstream

            stream_upstream(
                self,
                _upstream + self.path,
                raw or None,
                self._client_headers(),
                method=self.command,
                timeout=_UPSTREAM_TIMEOUT,
                hop_by_hop=_HOP_BY_HOP,
            )

    _DistilHandler.shadow_threads = _shadow_threads  # type: ignore[attr-defined]  # drained on shutdown
    _DistilHandler.shadow_lock = _shadow_threads_lock  # type: ignore[attr-defined]
    return _DistilHandler


def _signal_breadcrumb(name: str) -> None:
    """Append a wrap-level signal record to the session's ``.exit`` file.

    A process-group kill (terminal tab close = SIGHUP, plain ``kill`` = SIGTERM)
    takes the wrap down WITH the child, so the child-exit breadcrumb never gets
    written — this line is the only post-mortem trace of what happened."""
    try:
        from .ledger import session_marker_path

        mp = session_marker_path()
        if mp is not None and mp.parent.is_dir():
            with open(mp.with_name(mp.name + ".exit"), "a", encoding="utf-8") as f:
                f.write(f"wrap received {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except OSError:
        pass


def _start_heartbeat_timer(interval: float = 60.0) -> threading.Event:
    """While the proxy serves, tick the near-real-time community heartbeat so a
    long-lived interactive session pulses too (not only at exit). The heartbeat
    is internally throttled (≤1/5min) and sends only when tokens grew, so this
    loop is a cheap no-op most of the time. Daemon thread; stops on the event.
    Fail-open: a heartbeat problem must never touch the serving path."""
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval):
            try:
                from . import census as _census

                _census.maybe_heartbeat()
            except Exception:  # noqa: BLE001 — heartbeat is best-effort
                pass

    threading.Thread(target=_loop, daemon=True, name="distil-heartbeat").start()
    return stop


def _install_sigterm_flush(proc_holder: list | None = None) -> None:
    """Turn SIGTERM/SIGHUP into KeyboardInterrupt so the caller's ``finally``
    block (savings flush, shadow drain) runs on a plain ``kill`` or a closed
    terminal tab instead of dropping up to a flush-window of recorded savings.
    Writes a signal breadcrumb first (the group kill may not leave time for
    more), then forwards to a wrapped child if one is registered."""
    import signal

    def _on_term(signum: int, frame: object) -> None:  # noqa: ARG001
        try:
            _signal_breadcrumb(signal.Signals(signum).name)
        except Exception:  # noqa: BLE001 — dying; breadcrumb is best-effort
            pass
        if proc_holder:
            try:
                proc_holder[0].terminate()
            except Exception:  # noqa: BLE001 — best-effort child shutdown
                pass
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue  # Windows has no SIGHUP
        try:
            signal.signal(sig, _on_term)
        except ValueError:
            pass  # not the main thread (embedded use) — finally-blocks still cover Ctrl+C


def _drain_shadow(handler: type, budget: float = 6.0) -> None:
    """Let in-flight shadow comparisons finish recording before the proxy exits.

    Shadow runs each sampled decision uncompressed in a background thread; without
    draining, a quick run (or the last turn before shutdown) loses the sample.
    Bounded by ``budget`` seconds total so a hung upstream can't block teardown."""
    import time

    lock = getattr(handler, "shadow_lock", None)
    src = getattr(handler, "shadow_threads", []) or []
    if lock is not None:
        with lock:  # consistent snapshot vs a concurrent spawn prune+append
            threads = [t for t in src if t.is_alive()]
    else:
        threads = [t for t in src if t.is_alive()]
    if not threads:
        return
    deadline = time.monotonic() + budget
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------


def serve(
    host: str = "127.0.0.1",
    port: int = 8788,
    upstream: str = "https://api.anthropic.com",
    *,
    lossless_only: bool = False,
    verbatim: bool = False,
    shape_output: str = "off",
    record: bool = True,
    pricing_model: str = "claude-opus-4-8",
    expand: bool = False,
    shadow_rate: float = 0.0,
    retention_rate: float = 0.0,
    session_delta: bool = False,
) -> None:
    """Run a blocking :class:`ThreadingHTTPServer` proxy.

    Parameters
    ----------
    host:       Interface to bind on.
    port:       Port to listen on.
    upstream:   Real LLM API base URL (no trailing slash).
    lossless_only:
        Policy mode: no lossy output-shaping and no tool injection. The reversible
        Tier-1 digest still runs (it is the lossless, certified strategy).
    verbatim:
        When *True*, skip the Tier-1 digest entirely (Tier-0 only) so the model
        sees content verbatim — for interactive sessions / out-of-distribution
        traffic. Lower savings, byte-in-context fidelity.
    shape_output:
        Output-compression level: ``"off"``/``"light"``/``"aggressive"``.
    record:
        When *True* (default), accumulate GENUINE per-request token savings from
        real traffic into the local ledger (`distil leaderboard`). Numbers only,
        never content.
    pricing_model:
        Model id used to price the genuine dollar savings.
    """
    savings = None
    if record:
        from .runtime import RuntimeSavings

        savings = RuntimeSavings(model=pricing_model)
    handler = build_handler(
        upstream,
        lossless_only=lossless_only,
        verbatim=verbatim,
        shape_output=shape_output,
        savings=savings,
        expand=expand,
        shadow_rate=shadow_rate,
        retention_rate=retention_rate,
        session_delta=session_delta,
    )
    server, activated = _listen(host, port, handler)
    print(f"distil proxy listening on http://{host}:{port}")
    if activated:
        # Worth saying out loud: it is the difference between "a crash is a
        # blip" and "a crash is every session on this machine failing".
        print("  → socket-activated: the listener survives a restart of this process")
    print(f"  → upstream: {upstream}")
    if shadow_rate and shadow_rate > 0:
        print(
            f"  → shadow-mode live decision-equivalence: sampling {shadow_rate * 100:.0f}% "
            "(distil shadow-stats)"
        )
    if retention_rate and retention_rate > 0:
        print(
            f"  → live fact-retention meter: sampling {retention_rate * 100:.0f}% "
            "(content-free; distil retention --live)"
        )
    if expand:
        print(
            "  → recoverable compression: distil_expand tool active (agent recovers detail on demand)"
        )
    if shape_output != "off":
        if lossless_only:
            print(
                "  ⚠ --shape-output requested but SUPPRESSED: lossless-only never modifies "
                "the response. No shaping will happen. Drop --lossless-only to enable it."
            )
        else:
            print(f"  → output shaping: {shape_output}")
    if savings is not None:
        print("  → recording genuine savings → distil leaderboard")
    _install_sigterm_flush()
    _hb_stop = _start_heartbeat_timer()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _hb_stop.set()
        _drain_shadow(handler)
        if savings is not None:
            savings.flush()  # persist remaining genuine savings on shutdown
        try:
            from . import surfaces as _surfaces

            _surfaces.flush()  # persist integration-surface counts (census schema 3)
        except Exception:  # noqa: BLE001 — counters must never affect shutdown
            pass
        server.server_close()


def wrap_run(
    command: list[str],
    *,
    host: str = "127.0.0.1",
    upstream: str = "https://api.anthropic.com",
    lossless_only: bool = False,
    verbatim: bool = False,
    shape_output: str = "off",
    record: bool = True,
    pricing_model: str = "claude-opus-4-8",
    env_var: str = "ANTHROPIC_BASE_URL",
    expand: bool = False,
    session_delta: bool = False,
    shadow_rate: float = 0.0,
    retention_rate: float = 0.0,
) -> int:
    """Run *command* with its API base URL transparently pointed at a Distil proxy.

    Starts the proxy on an ephemeral local port in a background thread, injects
    ``env_var`` (default ``ANTHROPIC_BASE_URL``) into the child's environment so
    any base-url-honoring SDK routes through compression with no code change,
    runs the command to completion, then tears the proxy down — flushing genuine
    savings to the local ledger. Returns the child process's exit code.
    """
    import subprocess
    import sys
    import time

    # Capture session start time before any setup so proof-ledger duration is accurate.
    _start_ts = time.time()

    # One stable id for THIS wrap invocation, exported so BOTH the in-process
    # proxy (which tags every ledger record) and the wrapped agent — plus the
    # status line the agent spawns — see the same value and can attribute
    # savings to this exact session. Each terminal's wrap gets its own.
    os.environ.setdefault("DISTIL_SESSION", f"s{int(_start_ts)}-{os.getpid()}")

    # Statusline bypass detection: marker starts at "0" (wrapped, no request has
    # reached the proxy yet); the first proxied POST flips it to "1". One file
    # per session, single writer — no locking needed.
    from .ledger import session_marker_path

    marker = session_marker_path()
    if marker is not None:
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            now = time.time()
            for old in marker.parent.iterdir():  # opportunistic 7-day TTL sweep
                if now - old.stat().st_mtime > 7 * 86400:
                    old.unlink()
            marker.write_text("0", encoding="utf-8")
            # A nested wrap can inherit the sid — don't let a previous life's
            # exit breadcrumb masquerade as this session's.
            marker.with_name(marker.name + ".exit").unlink(missing_ok=True)
        except OSError:
            pass  # marker is best-effort; never block the wrap over it

    # Session manifest: what this wrap *is* (tool, argv, flags, billing) — the
    # header `distil dissect` reads. Best-effort, like the marker above.
    try:
        from . import __version__ as _ver
        from . import ledger as _ledger

        try:
            from .doctor import subscription_mode

            _billing = "subscription" if subscription_mode() else "metered"
        except Exception:  # noqa: BLE001 — billing detection is cosmetic
            _billing = "unknown"
        _ledger.write_session_manifest(
            {
                "sid": os.environ["DISTIL_SESSION"],
                "tool": os.path.basename(command[0]) if command else "",
                # command[:1] only — never persist full argv: a user may pass a
                # credential as a flag (--api-key sk-...) and the manifest is on disk.
                # The "tool" field already carries basename(command[0]).
                "argv": command[:1],
                "cwd": os.getcwd(),
                "started_ts": time.time(),
                "distil_version": _ver,
                "billing": _billing,
                "flags": {
                    "upstream": upstream,
                    "env_var": env_var,
                    "lossless_only": lossless_only,
                    "verbatim": verbatim,
                    "shape_output": shape_output,
                    "expand": expand,
                    "session_delta": session_delta,
                    "shadow_rate": shadow_rate,
                    "retention_rate": retention_rate,
                },
            }
        )
    except Exception:  # noqa: BLE001 — manifest is best-effort; never block the wrap
        log.debug("session manifest write failed", exc_info=True)

    # Hot-swap (POSIX default): the proxy runs as a supervised subprocess on a
    # listener FD the wrap owns, so a `pipx upgrade` mid-session swaps in a
    # fresh worker (new code, same port) without touching the agent. Windows
    # and DISTIL_HOT_SWAP=0 keep the historical in-thread proxy; a supervisor
    # start failure falls back to it too — the feature can never cost a session.
    supervisor = None
    if os.name == "posix" and os.environ.get("DISTIL_HOT_SWAP", "1") != "0":
        from .hotswap import ProxySupervisor, WorkerConfig

        try:
            supervisor = ProxySupervisor(
                WorkerConfig(
                    upstream=upstream,
                    lossless_only=lossless_only,
                    verbatim=verbatim,
                    shape_output=shape_output,
                    record=record,
                    pricing_model=pricing_model,
                    expand=expand,
                    session_delta=session_delta,
                    shadow_rate=shadow_rate,
                    retention_rate=retention_rate,
                ),
                host=host,
            )
            supervisor.start()
        except Exception:  # noqa: BLE001 — fall back rather than lose the session
            log.warning("hot-swap supervisor failed; using in-thread proxy", exc_info=True)
            supervisor = None

    savings = None
    handler = None
    server = None
    if supervisor is not None:
        base = f"http://{host}:{supervisor.port}"
    else:
        if record:
            from .runtime import RuntimeSavings

            savings = RuntimeSavings(model=pricing_model)
        handler = build_handler(
            upstream,
            lossless_only=lossless_only,
            verbatim=verbatim,
            shape_output=shape_output,
            savings=savings,
            expand=expand,
            session_delta=session_delta,
            shadow_rate=shadow_rate,
            retention_rate=retention_rate,
        )
        server = QuietHTTPServer((host, 0), handler)  # port 0 → OS picks a free port
        base = f"http://{host}:{server.server_address[1]}"

    if server is not None:

        def _serve_resilient() -> None:
            # Self-heal: if serve_forever ever dies, the wrapped agent would get
            # connection-refused for the rest of the session with no signal. Log
            # loudly and re-enter the accept loop; the socket stays bound.
            # (The hot-swap path has the same contract: the supervisor respawns
            # a worker that dies underneath the session.)
            import sys as _sys

            while True:
                try:
                    server.serve_forever()
                    return  # clean shutdown()
                except Exception:  # noqa: BLE001 — keep the session alive
                    log.warning("wrap proxy accept loop crashed; restarting", exc_info=True)
                    print("distil: proxy accept loop crashed — restarting", file=_sys.stderr)

        threading.Thread(target=_serve_resilient, daemon=True).start()

    child_env = dict(os.environ)
    child_env[env_var] = base
    print(f"distil wrap → proxy {base} (upstream {upstream})")
    print(f"  → {env_var}={base}")
    if lossless_only:
        print("  → lossless-only (no shaping / no tool injection)")
    if verbatim:
        print("  → verbatim (Tier-0 only, no digest)")
    if record:  # savings recorder lives in-process or in the worker, same meaning
        print("  → recording genuine savings → distil leaderboard")
    if supervisor is not None:
        print(
            f"  → hot-swap: upgrades apply live (worker v{supervisor.worker_version}, "
            "kill -USR1 to force)"
        )
    if shadow_rate and shadow_rate > 0:
        print(
            f"  → shadow-mode live decision-equivalence: sampling "
            f"{shadow_rate * 100:.0f}% (distil shadow-stats)"
        )

    # Save the controlling terminal's mode before handing the tty to the child: an
    # agent that dies in raw mode (TUI, readline, password prompt) would otherwise
    # leave the user's shell wedged (no echo / no line editing). POSIX-only — the
    # import is guarded so this is a no-op on Windows.
    _saved_tty: tuple[int, Any] | None = None
    try:
        import termios

        if sys.stdin.isatty():
            _tty_fd = sys.stdin.fileno()
            _saved_tty = (_tty_fd, termios.tcgetattr(_tty_fd))
    except Exception:  # noqa: BLE001 — never fail wrap over terminal bookkeeping
        _saved_tty = None

    code = 0
    proc_holder: list = []
    _install_sigterm_flush(proc_holder)
    # Ctrl+C belongs to the child. The terminal delivers SIGINT to the whole
    # foreground group, and agents like Claude Code use the first press to
    # cancel the turn, not exit — a KeyboardInterrupt raised in the parent at
    # ANY point (catching it only around proc.wait() loses the race when a
    # rapid second press lands inside the except clause) tears the proxy down
    # under a live agent: dead port on its next API call, session killed.
    # Ignore SIGINT here instead. A Python-level handler (unlike SIG_IGN) is
    # reset to default across exec, so the child still receives its Ctrl+C
    # and decides its own fate. SIGTERM keeps terminate+flush+exit semantics.
    import signal

    try:
        signal.signal(signal.SIGINT, lambda *_: None)
    except ValueError:
        pass  # not the main thread (embedded use) — finally-block still covers teardown
    # main() set SIGPIPE to SIG_DFL (right for CLI filters). The in-thread proxy
    # server runs in THIS process; a write to a client socket that hung up must
    # raise a catchable BrokenPipeError, not kill the whole wrap with exit=-13.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (ValueError, AttributeError):
        pass  # not main thread, or no SIGPIPE (Windows) — harmless to skip
    if supervisor is not None:
        try:
            # Manual hot-swap: `kill -USR1 <wrap pid>` — handler only sets an
            # event; the supervisor's watch thread does the actual work.
            signal.signal(signal.SIGUSR1, lambda *_: supervisor.request_handover())
        except (ValueError, AttributeError):
            pass  # not the main thread, or platform without SIGUSR1
    try:
        # Reserve the slot before Popen so a SIGTERM in the spawn window still
        # finds the child: the handler no-ops on the None placeholder, then the
        # single-statement store binds the real proc as tightly as possible.
        proc_holder.append(None)
        proc_holder[0] = proc = subprocess.Popen(command, env=child_env)
        code = proc.wait()
    except FileNotFoundError:
        print(f"distil wrap: command not found: {command[0]}", file=sys.stderr)
        code = 127
    except KeyboardInterrupt:
        code = 130  # SIGTERM, translated by _install_sigterm_flush (child already terminated)
    finally:
        if supervisor is not None:
            # Worker owns the flushes: its SIGTERM drain finishes in-flight
            # requests, drains shadow, and flushes savings before exiting.
            supervisor.shutdown()
        else:
            assert server is not None and handler is not None  # in-thread mode
            server.shutdown()
            _drain_shadow(handler)
            if savings is not None:
                savings.flush()  # SIGTERM lands here too — no savings are ever dropped
            server.server_close()
        if _saved_tty is not None:
            # Restore the terminal even if the child died mid-raw-mode. TCSADRAIN
            # waits for pending output to flush first.
            try:
                import termios

                termios.tcsetattr(_saved_tty[0], termios.TCSADRAIN, _saved_tty[1])
                # tcsetattr can't undo xterm private modes a crashed TUI leaves
                # on: mouse reporting (shows as "65;76;9M" junk on click),
                # bracketed paste, the alternate screen, a hidden cursor. Reset
                # them explicitly — all idempotent on a clean exit.
                os.write(
                    _saved_tty[0],
                    b"\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l"  # mouse off
                    b"\x1b[?2004l"  # bracketed paste off
                    b"\x1b[?1049l"  # leave alternate screen
                    b"\x1b[?25h",  # cursor visible
                )
            except Exception:  # noqa: BLE001 — best-effort; child may have closed the tty
                pass

    # Post-mortem breadcrumb: how the child ended. A silent agent quit (e.g. a
    # runtime OOM abort) is undiagnosable after the fact — the wrap is the only
    # witness to the exit status. scripts/soak-report.sh surfaces this file.
    try:
        mp = session_marker_path()
        if mp is not None and mp.parent.is_dir():
            if code < 0:
                import signal as _signal

                try:
                    desc = f"signal {_signal.Signals(-code).name}"
                except ValueError:
                    desc = f"signal {-code}"
            else:
                desc = f"exit code {code}"
            # Append — a signal breadcrumb may already be in the file, and both
            # lines together tell the story (e.g. SIGTERM → child exit 143).
            from .hotswap import memory_evidence

            # Memory context rides along: on the 2026-07-07 soak day agents
            # died under swap exhaustion and bare exit codes couldn't say why.
            with open(mp.with_name(mp.name + ".exit"), "a", encoding="utf-8") as f:
                f.write(
                    f"child {desc} at {time.strftime('%Y-%m-%d %H:%M:%S')} | {memory_evidence()}\n"
                )
    except OSError:
        pass
    # Proof Ledger — printed on clean exit and Ctrl-C; fail-open (must not alter exit code).
    try:
        from .proof_ledger import print_proof_ledger as _print_proof_ledger

        _print_proof_ledger(os.environ.get("DISTIL_SESSION", ""), _start_ts)
    except Exception:  # noqa: BLE001 — ledger print must never affect exit code
        pass
    # Persist any pending integration-surface counts before the census reads them.
    try:
        from . import surfaces as _surfaces

        _surfaces.flush()
    except Exception:  # noqa: BLE001 — counters must never affect exit code
        pass
    # Adoption census — opt-in only, ≤1/day, content-free; fail-open like the
    # ledger print (see distil/census.py and TELEMETRY.md).
    try:
        from . import census as _census

        # Ask for consent BEFORE the ping, so a first-time yes takes effect in the
        # same session the user granted it — otherwise their first census would be
        # a day late, and the number they just agreed to share is the number on
        # screen. Self-gating: no-ops unless never asked, a TTY, and something was
        # actually saved. `distil onboard` was the only consent surface, which
        # meant anyone who installed and went straight to `wrap` was never asked.
        _census.maybe_ask_consent()
        _census.maybe_ping()
        _census.maybe_heartbeat()  # near-real-time community pulse (≤1/5min, only-if-grew)
    except Exception:  # noqa: BLE001 — census must never affect exit code
        pass
    return code


if __name__ == "__main__":
    serve()
