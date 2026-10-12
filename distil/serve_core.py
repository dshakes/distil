"""One compress-or-forward path for the threaded servers (ADR 0023).

``distil/proxy.py`` (the local ``wrap`` proxy) and ``distil/gateway.py`` (the
multi-tenant gateway) used to carry their own copy of "parse the body, pick the
adapter, compress, fail open, count, inject distil_expand, replay the prefix". The
copies drifted: cold-point recompression, cache-delta coding and the learned keep
policy existed in the proxy only, the gateway's token counter missed images and
thinking blocks, and the gateway re-encoded every body so prefix replay modelled
bytes it never sent. Both now call :func:`compress_or_forward`, and the response-side
expand plumbing they share (buffer-for-expand, loop dispatch, SSE re-emission) lives
here too.

Everything per-caller stays a parameter: *scope* is the account hash on the proxy and
the tenant on the gateway, so cold-point lineages, expanded-handle exclusions and
cache-delta sessions are partitioned exactly where the caller draws its trust
boundary. :class:`TenantHandles` is the gateway's restore boundary: a tenant's
``distil_expand`` resolves only handles the gateway issued to that tenant.

``aproxy.py`` is verbatim-only by design (it runs no expand loop) and is not routed
through here.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from . import coldpoint as _coldpoint
from ._log import log
from .adapters.anthropic import compress_messages
from .adapters.gemini import compress_generate_request
from .adapters.gemini import count_tokens as _gemini_count
from .httpguard import is_chat_completions_path, is_responses_path, strip_query

# Body fields that are only legal on a streaming request. Dropped together when an
# intercepted request is forced to buffered mode — a leftover stream_options with no
# stream:true is a 400 from OpenAI, not a warning.
_STREAM_ONLY_FIELDS = frozenset({"stream", "stream_options"})


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


def _serialize_if_changed(raw: bytes, body: dict[str, Any]) -> bytes:
    """Return the ORIGINAL bytes when the body is unchanged; re-serialize only if not.

    The provider's prompt cache matches on exact bytes, and ``json.dumps`` is not a
    byte-faithful round-trip of what arrived: key order survives, but separators
    (``", "`` vs ``","``) and non-ASCII escaping (``\\uXXXX`` vs raw UTF-8) do not.
    Re-encoding an *unmodified* body therefore rewrites the cached prefix and turns
    cheap cache reads into expensive cache writes — while saving nothing, because
    nothing was compressed. That is the worst possible trade, and it is exactly what
    lossless-only mode did on a subscription: 0% savings at measured 1.56x baseline
    cache-creation tokens (2.52x on short sessions).

    Comparing the parsed body against a re-parse of the original is O(body) and runs
    once per request — far cheaper than re-billing the prefix. When a transform did
    change something we re-serialize compactly and accept the byte drift, because
    then the bytes genuinely differ anyway.
    """
    try:
        if json.loads(raw) == body:
            return raw
    except (ValueError, TypeError):
        pass  # unparseable original — fall through and serialize what we have
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()


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


def _unstream_path(target: str) -> str:
    """The non-streaming twin of a request target.

    OpenAI names the streaming mode in the body (``stream: true``), but Gemini names
    it in the URL (``:streamGenerateContent`` plus ``alt=sse``). Dropping ``stream``
    from a Gemini body would leave it streaming anyway, so the path has to change too.
    """
    path, _, query = target.partition("?")
    path = path.replace(":streamGenerateContent", ":generateContent")
    if query:
        query = "&".join(p for p in query.split("&") if p != "alt=sse")
    return f"{path}?{query}" if query else path


def _is_timeout(exc: urllib.error.URLError) -> bool:
    return isinstance(exc.reason, (socket.timeout, TimeoutError))


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


def open_stream(
    opener: urllib.request.OpenerDirector,
    url: str,
    data: bytes,
    headers: dict[str, str],
    timeout: float,
) -> Any:
    """POST *data* and return the open response, or an :class:`_ErrStream` for an
    HTTP/connection failure — the shape ``streamexpand.stream_with_expand`` sends on."""
    req = urllib.request.Request(
        url, data=data, headers={**headers, "Content-Length": str(len(data))}, method="POST"
    )
    try:
        return opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return _ErrStream(exc.code, exc.headers, exc.read() if exc.fp else b"")
    except (urllib.error.URLError, TimeoutError) as exc:
        st = 504 if isinstance(exc, TimeoutError) or _is_timeout(exc) else 502
        return _ErrStream(
            st, {"Content-Type": "application/json"}, b'{"error":"upstream connection failed"}'
        )


@dataclass
class Outbound:
    """What :func:`compress_or_forward` decided to send.

    ``pending`` is ``(before, after, model)`` when this request has real savings to
    book (a shape was compressed AND its counts are real); callers book it only after
    a confirmed 2xx. ``cold_key`` is the cold-point lineage the request planned
    against: the caller MUST ``coldpoint.end(cold_key)`` once the response is fully
    relayed (a stream or an expand re-query refreshes the cache later than the
    forward did).
    """

    body: dict[str, Any]
    store: Any = None
    extras: dict[str, str] = field(default_factory=dict)
    before_tok: int | None = None
    after_tok: int | None = None
    pending: tuple[int, int, str | None] | None = None
    cold_key: str | None = None


def _model_from_path(path: str) -> str | None:
    """Extract the model id from a Gemini-style URL (``.../models/<id>:action``)."""
    marker = "/models/"
    idx = path.find(marker)
    if idx < 0:
        return None
    tail = path[idx + len(marker) :]
    return tail.split(":", 1)[0].split("/", 1)[0] or None


def _cold_plan(
    scope: str,
    body: dict[str, Any],
    items: list[Any],
    candidates: Callable[..., frozenset[str]],
    keep: Any,
    held: bool,
    *,
    openai: bool = True,
) -> tuple[Any, str | None, dict[str, Any]]:
    """Plan this request's cold-point eviction: ``(plan, lineage key, compressor kwargs)``,
    or ``(None, None, {})`` when planning failed (never break a request for a saving)."""
    try:
        from . import prefixreplay as _prep

        # Account/tenant scope, not the credential: an OAuth bearer refreshes mid-session
        # and must not fork the lineage (ADR 0014).
        ck = scope + _prep.lineage_key(body, items)
        cold_plan = _coldpoint.plan(
            ck,
            body,
            items,
            lambda: candidates(items, keep=keep, exclude_handles=_coldpoint.expanded(scope)),
            # A drift-guard hold decides nothing new: the evicted set keeps applying
            # (byte-stable prefix), no fresh eviction.
            held=held,
            ttl=_coldpoint.openai_request_ttl(body) if openai else None,
        )
    except Exception:  # noqa: BLE001 — never break a request for a saving
        log.debug("cold-point plan failed; compressing as usual", exc_info=True)
        return None, None, {}
    return cold_plan, ck, ({"evict": cold_plan.evict} if cold_plan.evict else {})


def _cold_extras(extras: dict[str, str], cold_plan: Any) -> None:
    """Why this turn did (or did not) evict, and how many ids the lineage carries evicted.
    The tokens are already inside tokens-saved."""
    if cold_plan is not None:
        extras["x-distil-cold"] = cold_plan.reason
        extras["x-distil-cold-evicted"] = str(len(cold_plan.evict))


def compress_or_forward(
    body: dict[str, Any],
    path: str,
    *,
    count: Callable[[list[dict[str, Any]]], int],
    verbatim: bool,
    expand: bool,
    mode: str,
    keep: Callable[[str], bool] | None = None,
    session_delta: bool = False,
    cold: bool = False,
    scope: str = "",
    held: bool = False,
    shape_output: str = "off",
    replay_scope: str | None = None,
    on_compressed: Callable[[str, Any, Any, Any], None] | None = None,
    admit: Callable[[int], bool] | None = None,
) -> Outbound | None:
    """Compress one parsed request body, or forward it as received if anything fails.

    Dispatches by shape: OpenAI Responses (``input``), Anthropic Messages / OpenAI Chat
    Completions (``messages``, adapter chosen by *path*), Gemini (``contents``); any other
    body comes back untouched with no extras. Every compressor and every planner is
    fail-open — an exception forwards the uncompressed items and the request proceeds.

    *count* is the messages token counter (passed in so the caller's module-level
    counter is the one used). *scope* partitions per-identity state: cold-point lineages
    and expanded-handle exclusions (``coldpoint``) and cache-delta sessions
    (``cachedelta``). *cold* enables cold-point recompression on Anthropic Messages and
    both OpenAI shapes (Gemini documents no cache lifetime to wait out). *replay_scope* enables forwarded-bytes prefix replay (ADR 0011) under that scope;
    ``None`` turns it off. *on_compressed(kind, original, compressed, store)* runs right
    after the messages/contents compressors (learning, retention metering). *admit* is
    called with the pre-compression token count before anything stateful happens;
    returning False aborts with ``None`` (the caller has already answered the client).
    """
    extras: dict[str, str] = {}
    store: Any = None
    before_tok: int | None = None
    after_tok: int | None = None
    pending: tuple[int, int, str | None] | None = None
    replay_key: str | None = None
    replay_orig: list[Any] | None = None
    cold_key: str | None = None
    cold_plan: Any = None  # ADR 0014 plan, when cold-point runs on this shape
    cold_kw: dict[str, Any] = {}
    _path = strip_query(path)

    try:
        if is_responses_path(_path) and isinstance(body.get("input"), list):
            # OpenAI Responses API: compress ``function_call_output`` items (Tier-1
            # reversible digest) and user ``message`` items (Tier-0).
            from .adapters.openai import compress_responses_input, count_responses_tokens

            orig_input: list[dict[str, Any]] = body["input"]
            replay_key, replay_orig = "input", orig_input
            before_tok = count_responses_tokens(orig_input)
            if admit is not None and not admit(before_tok):
                return None
            if cold:
                from .adapters.openai import cold_candidates_responses

                cold_plan, cold_key, cold_kw = _cold_plan(
                    scope, body, orig_input, cold_candidates_responses, keep, held
                )
            try:
                new_input, store = compress_responses_input(
                    orig_input, verbatim=verbatim, keep=keep, **cold_kw
                )
            except Exception:  # noqa: BLE001 — compression must never break a request
                log.debug("compress_responses_input failed; forwarding uncompressed", exc_info=True)
                new_input, store = orig_input, None
            after_tok = count_responses_tokens(new_input)
            body = {**body, "input": new_input}
            extras = {
                "x-distil-compressed": "1",
                "x-distil-tokens-saved": str(max(0, before_tok - after_tok)),
                "x-distil-mode": mode,
                "x-distil-compressible-tokens": str(before_tok),
            }
            pending = (before_tok, after_tok, body.get("model"))
            _cold_extras(extras, cold_plan)
            # Recoverable compression: inject distil_expand so the model can pull back
            # any digested block by handle. Session-sticky (see the messages branch).
            if expand:
                from .expand import inject_expand_tool_responses

                body = inject_expand_tool_responses(body)
            if shape_output != "off":
                from .output import shape_request

                body = shape_request(body, level=shape_output, allow=True, shape="responses")
                extras["x-distil-output-shaping"] = shape_output

        elif "messages" in body and isinstance(body["messages"], list):
            original: list[dict[str, Any]] = body["messages"]
            replay_key, replay_orig = "messages", original
            # Counted BEFORE anything stateful so a quota rejection (admit) leaves the
            # cold-point lineage and the delta session exactly as they were. Accounting
            # is bookkeeping: a counter that raises serves the request without it.
            try:
                before_tok = count(original)
            except Exception:  # noqa: BLE001 — a counter must never break a request
                log.debug("token accounting failed; serving without it", exc_info=True)
                before_tok = None
            if admit is not None and not admit(before_tok or 0):
                return None
            pre = original
            dstats: Any = None
            dstore: Any = None
            # Decided ONCE, above both users. The keep-list below and the compressor
            # dispatch further down must agree on which shape this body is, or the
            # exemption is computed by the wrong adapter and comes back empty — which on
            # an Azure Chat Completions path is exactly the guarantee-voiding bug this
            # block exists to fix, reintroduced by a second, narrower path test.
            is_chat = is_chat_completions_path(_path)
            # OpenAI Chat Completions needs its own adapter (role:"tool" list content is
            # Tier-1; the Anthropic adapter applies Tier-0 to generic list text items).
            if is_chat:
                from .adapters.openai import compress_chat_completions

                compress_fn: Callable[..., Any] = compress_chat_completions
            else:
                compress_fn = compress_messages
            # Cold-point recompression (ADR 0014). Fail-open: any error plans nothing and
            # the request compresses exactly as before. Chat's TTL comes from OpenAI's
            # documented retention bounds (coldpoint.openai_request_ttl).
            if cold:
                if is_chat:
                    from .adapters.openai import cold_candidates_chat as _cands
                else:
                    from .adapters.anthropic import cold_candidates as _cands
                cold_plan, cold_key, cold_kw = _cold_plan(
                    scope, body, original, _cands, keep, held, openai=is_chat
                )
            if session_delta:
                # Cache-delta coding: cross-turn dedup + cross-version delta, applied to
                # the ORIGINALS before compression so re-reads match across turns.
                # Cache-monotonic (suffix-only) and reversible. Runs BEFORE compression
                # and skips the exact-quote blocks (they stay delta bases): running it
                # after would delta against digest stubs and rewrite cached blocks, and
                # running it over them silently voided the 1.49.0 exact-quote guarantee.
                try:
                    from .cachedelta import delta_encode, get_session, session_key

                    sess = get_session(scope + session_key(original))
                    if is_chat:
                        from .adapters.openai import exact_quote_tool_call_ids as _exact_ids
                    else:
                        from .adapters.anthropic import exact_quote_tool_use_ids as _exact_ids
                    keep_ids = frozenset(_exact_ids(original))
                    if is_chat and not verbatim:
                        # Same rule, Chat shape (no cold-point on this path, so no evict).
                        from .adapters.openai import refetch_tool_call_ids

                        keep_ids |= refetch_tool_call_ids(original, keep=keep)
                    elif not verbatim:
                        # A re-fetch of folded content goes out verbatim (ADR 0025). Delta
                        # coding it first would turn it into a reference to the folded
                        # copy, the stub the agent re-ran the command to get past.
                        from .adapters.anthropic import refetch_tool_use_ids

                        keep_ids |= refetch_tool_use_ids(original, keep=keep, **cold_kw)
                    pre, dstore, dstats = delta_encode(original, session=sess, keep_ids=keep_ids)
                except Exception:  # noqa: BLE001 — never break a request
                    log.debug("cache-delta encode failed", exc_info=True)
                    pre, dstore, dstats = original, None, None
            try:
                compressed, store = compress_fn(pre, verbatim=verbatim, keep=keep, **cold_kw)
            except Exception:  # noqa: BLE001 — compression must never break a request
                log.debug("compress_messages failed; forwarding uncompressed", exc_info=True)
                compressed, store = pre, None
            # Merge cache-delta references into the store so distil_expand recovers them.
            if dstore is not None and store is not None:
                for h in dstore.handles:
                    try:
                        store._record(h, dstore.expand(h))
                    except Exception:  # noqa: BLE001
                        pass
            if on_compressed is not None:
                on_compressed("messages", original, compressed, store)
            try:
                after_tok = count(compressed) if before_tok is not None else None
            except Exception:  # noqa: BLE001 — a counter must never break a request
                log.debug("token accounting failed; serving without it", exc_info=True)
                after_tok = None
            if after_tok is None:
                before_tok = None
            saved = (
                max(0, before_tok - after_tok)
                if before_tok is not None and after_tok is not None
                else 0
            )
            body = {**body, "messages": compressed}
            extras = {
                "x-distil-compressed": "1",
                "x-distil-tokens-saved": str(saved),
                "x-distil-mode": mode,
                # Tokens in the compressible zone (user/tool content distil may touch) —
                # when this is ~0, a ▼0 is "nothing large to compress this turn".
                "x-distil-compressible-tokens": str(before_tok if before_tok is not None else 0),
            }
            if dstats is not None:
                extras["x-distil-cache-refs"] = str(dstats.exact_refs + dstats.delta_refs)
                extras["x-distil-cache-delta"] = str(dstats.delta_refs)
                extras["x-distil-cache-tokens-saved"] = str(dstats.tokens_saved)
                # How many leading messages were byte-stable vs the previous turn.
                extras["x-distil-cache-prefix-msgs"] = str(dstats.prefix_msgs)
            _cold_extras(extras, cold_plan)
            # Injected on EVERY request while expand is on, not only when a handle
            # exists: Anthropic caches the tools array at the very front of the prefix,
            # so a tools list that gains an entry mid-session invalidates the whole
            # cached entry. Two schemas share this key: Chat Completions wants the
            # OpenAI function shape, or the whole request 400s at the provider.
            if expand:
                if is_chat:
                    from .expand import inject_expand_tool_chat

                    body = inject_expand_tool_chat(body)
                else:
                    from .expand import inject_expand_tool

                    body = inject_expand_tool(body)
            # Only when the counts are real: a request whose accounting failed goes
            # unbooked rather than entering the ledger with a fabricated zero.
            if before_tok is not None and after_tok is not None:
                pending = (before_tok, after_tok, body.get("model"))
            if shape_output != "off":
                from .output import shape_request

                shape = "anthropic" if _path == "/v1/messages" else "openai"
                body = shape_request(body, level=shape_output, allow=True, shape=shape)
                extras["x-distil-output-shaping"] = shape_output

        elif "contents" in body and isinstance(body["contents"], list):
            # Gemini generateContent shape. Content compression + output shaping.
            replay_key, replay_orig = "contents", body["contents"]
            before_tok = _gemini_count(body)
            if admit is not None and not admit(before_tok):
                return None
            try:
                body, store = compress_generate_request(body, verbatim=verbatim, keep=keep)
            except Exception:  # noqa: BLE001 — compression must never break a request
                log.debug("gemini compression failed; forwarding uncompressed", exc_info=True)
                store = None
            if on_compressed is not None:
                on_compressed("contents", None, None, store)
            after_tok = _gemini_count(body)
            extras = {
                "x-distil-compressed": "1",
                "x-distil-tokens-saved": str(max(0, before_tok - after_tok)),
                "x-distil-mode": mode,
                "x-distil-compressible-tokens": str(before_tok),
            }
            # Gemini requests carry the model in the URL path, not the body.
            pending = (before_tok, after_tok, _model_from_path(path))
            if shape_output != "off":
                from .output import shape_request

                body = shape_request(body, level=shape_output, allow=True, shape="gemini")
                extras["x-distil-output-shaping"] = shape_output
            if expand:
                from .expand import inject_expand_tool_gemini

                body = inject_expand_tool_gemini(body)

        # Forwarded-bytes prefix replay (ADR 0011). Runs last, on the final body, so it
        # is the one thing between distil's decisions and the wire. Fail-open inside.
        if replay_scope is not None and replay_orig is not None and replay_key is not None:
            from . import prefixreplay as _prep

            body = _prep.apply(body, replay_key, replay_orig, scope=replay_scope, extras=extras)
    except BaseException:
        # A planned lineage is in flight until end(); a raise after plan() would leave it
        # in flight forever (never cold again). The caller never sees the key, so close it.
        if cold_key is not None:
            _coldpoint.end(cold_key)
        raise

    return Outbound(body, store, extras, before_tok, after_tok, pending, cold_key)


def buffer_for_expand(
    body: dict[str, Any], raw: bytes, path: str
) -> tuple[str, dict[str, Any], bytes, str] | None:
    """Turn a streamed non-Anthropic request that needs the expand loop into a buffered
    one: ``(sse_shape, body, new_raw, forward_path)``, or ``None`` to give up the expand.

    The streaming splice only speaks Anthropic Messages SSE; OpenAI and Gemini streams
    take the buffered loop and get the answer back re-encoded as SSE
    (:func:`sse_body`). ``n > 1`` gives up instead: the SSE re-encoder renders
    ``choices[0]`` only, and silently dropping n-1 completions is worse than an
    unanswered expand call the client can see.
    """
    if isinstance(body.get("n"), int) and body["n"] > 1:
        return None
    if isinstance(body.get("contents"), list):
        # The client's own stream format: SSE frames with alt=sse, else a JSON array.
        shape = "gemini" if "alt=sse" in path else "gemini-array"
    elif isinstance(body.get("input"), list):
        shape = "responses"
    else:
        shape = "chat"
    # stream_options is only legal alongside stream:true — OpenAI 400s on it otherwise.
    body = {k: v for k, v in body.items() if k not in _STREAM_ONLY_FIELDS}
    # Same serializer as the plain path: the encoding prefixreplay._wire models.
    return shape, body, _serialize_if_changed(raw, body), _unstream_path(path)


def run_expand(
    body: dict[str, Any],
    resp_json: dict[str, Any],
    store: Any,
    post: Callable[[dict[str, Any]], dict[str, Any]],
    path: str,
    on_signal: Callable[[str, str], None] | None,
) -> dict[str, Any]:
    """Resolve ``distil_expand`` calls in *resp_json* with the loop for this body's shape
    (Gemini ``contents``, Responses ``input``, Chat Completions, Anthropic Messages)."""
    from .expand import (
        run_expand_loop,
        run_expand_loop_chat,
        run_expand_loop_gemini,
        run_expand_loop_responses,
    )

    if "contents" in body and isinstance(body.get("contents"), list):
        return run_expand_loop_gemini(body, resp_json, store, post, on_signal=on_signal)
    if "input" in body and isinstance(body.get("input"), list):
        return run_expand_loop_responses(body, resp_json, store, post, on_signal=on_signal)
    if is_chat_completions_path(strip_query(path)):
        # Chat Completions shares the ``messages`` key with Anthropic but not the
        # tool-call shape: the answer goes back as a role:"tool" message.
        return run_expand_loop_chat(body, resp_json, store, post, on_signal=on_signal)
    return run_expand_loop(body, resp_json, store, post, on_signal=on_signal)


def sse_body(
    shape: str, rhdrs: dict[str, str], rbody: bytes
) -> tuple[dict[str, str], bytes] | None:
    """A buffered 2xx answer re-encoded as the SSE stream the client asked for, or
    ``None`` when the body is not a JSON object (relay it untouched: an error the SDK
    can read beats a well-formed stream carrying nothing)."""
    try:
        final = json.loads(rbody)
    except (ValueError, TypeError):
        return None
    if not isinstance(final, dict):
        return None
    from .streamexpand import sse_from_response

    # Drop the upstream's own content-type rather than adding a second one.
    hdrs = {k: v for k, v in rhdrs.items() if k.lower() != "content-type"}
    hdrs["Content-Type"] = "application/json" if shape == "gemini-array" else "text/event-stream"
    return hdrs, sse_from_response(shape, final)


# ---------------------------------------------------------------------------
# Gateway restore boundary
# ---------------------------------------------------------------------------


class TenantHandles:
    """Which restore handles the gateway issued to which tenant.

    The restore store is content-addressed and shared on disk (``RestoreStore``'s
    fallback), so a bare store would answer ANY 8-hex handle a model asks for — a
    tenant could write ``handle=…`` into its own history and read another tenant's
    originals back through ``distil_expand``. Every handle a request's store holds is
    granted to that request's tenant; :meth:`view` resolves only granted handles.

    ponytail: in-memory and LRU-bounded, so a gateway restart (or a grant aged out
    under heavy load) turns an older stub's expand into the miss placeholder — the
    fail-safe direction. Persist grants per hashed tenant if that ever bites.
    """

    MAX = 65536

    def __init__(self) -> None:
        self._grants: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._lock = threading.Lock()

    def grant(self, tenant: str, handles: Any) -> None:
        with self._lock:
            for h in handles or ():
                self._grants[(tenant, h)] = None
                self._grants.move_to_end((tenant, h))
            while len(self._grants) > self.MAX:
                self._grants.popitem(last=False)

    def owns(self, tenant: str, handle: str) -> bool:
        with self._lock:
            return (tenant, handle) in self._grants

    def view(self, tenant: str, store: Any) -> _TenantStore:
        """*store* (this request's, possibly None) as *tenant* may see it; grants its
        handles first, so everything this request issued resolves."""
        self.grant(tenant, getattr(store, "handles", None))
        return _TenantStore(self, tenant, store)


class _TenantStore:
    """A RestoreStore restricted to one tenant's grants. ``handles`` is the request's
    own (what intercept and accounting key on); ``expand`` raises KeyError for any
    handle not granted to this tenant, which every expand loop turns into a miss."""

    def __init__(self, owner: TenantHandles, tenant: str, store: Any) -> None:
        self._owner, self._tenant, self._store = owner, tenant, store

    @property
    def handles(self) -> frozenset[str]:
        return frozenset(getattr(self._store, "handles", None) or ())

    def expand(self, handle: str) -> str:
        if not self._owner.owns(self._tenant, handle):
            raise KeyError(handle)
        if self._store is None:
            from .adapters.anthropic import RestoreStore

            self._store = RestoreStore()
        return self._store.expand(handle)
