"""Contract tests for the wrap-time base-URL each agent preset exports.

This is the SDK-convention half of the routing contract that
tests/test_upstream_contracts.py already pins the env-var-name half of.

The defect this file exists to catch: a preset exports a base URL for its
agent's HTTP client, but that client's own convention for turning a base URL
into a request path doesn't match what the export assumed — the export lands
on a path ``distil.httpguard.is_compressible_path`` (or, for Gemini,
``distil.adapters.gemini.is_gemini_path``) does not recognise. ``wrap``
reports success and the session runs uncompressed; against the REAL upstream
(not this file's fake one) the same mismatch is a 404. Commit 89526f0 fixed
exactly this for Kilo Code (its baseURL needed ``/v1`` appended, which the
``kilo`` entry in ``AGENT_ENV_TEMPLATES`` now supplies) — the ``kilo-anthropic``
/``kilo-openai`` parametrized cases below are the regression guard for that
fix, and the same proof extends to every other preset that also needs a
template (aider/opencode/qwen; and grok/kimi/openhands, added this session —
see the doc comment on ``AGENT_ENV_TEMPLATES`` in onboard.py) plus the two
bare-URL presets whose SDK convention is independently verified.

Each case's assertion has two halves: (a) the path the client's convention
produces is one ``is_compressible_path``/``is_gemini_path`` recognises, and
(b) a genuinely-compressible tool-result body sent down that path comes back
with ``x-distil-tokens-saved`` > 0 — proof the compression branch actually
ran, not just that the request passed through unchanged (a preset could
"pass" (a) by landing on a compressible-shaped path that never gets a
compressible body in practice; (b) rules that out).

Base-URL -> request-path conventions used below (the "one small table" the
task asked for), each cited to where it was checked:

* Anthropic's official SDK (used by Claude Code) — a bare base_url with no
  path gets ``/v1/messages`` appended BY THE SDK. Verified live 2026-09-24:
  ``anthropic.Anthropic(base_url="http://host:port").messages.create(...)``
  against a real HTTP server capture. Doc:
  https://github.com/anthropics/anthropic-sdk-python#usage
* ``@ai-sdk/anthropic`` / ``@ai-sdk/openai`` (Vercel AI SDK), which Kilo Code
  forks its provider layer from — the configured ``baseURL`` is used
  LITERALLY and only the leaf (``/messages`` or ``/chat/completions``) is
  appended, so the caller must already include ``/v1``. Independently
  re-verified this session by installing the real packages
  (``@ai-sdk/anthropic@4.0.63``, ``@ai-sdk/openai@4.0.75``) into a scratch
  npm project and reading the bundled source directly (no live API call):
  ``normalizeBaseURL()`` in ``@ai-sdk/anthropic`` only auto-inserts ``/v1``
  for the exact literal string ``"https://api.anthropic.com"``, any other
  baseURL (including ours) is used as-is, and the request URL is
  ``${baseURL}/messages``; ``@ai-sdk/openai``'s ``createOpenAI()`` resolves
  ``options.baseURL ?? env.OPENAI_BASE_URL ?? "https://api.openai.com/v1"``
  with no auto-insert case at all.
* The official OpenAI SDK (Python AND Node), and LiteLLM's
  ``api_base=`` — all three use an explicitly-set base_url LITERALLY and
  append only ``/chat/completions``, never inserting ``/v1`` themselves.
  Verified live 2026-09-24 against real installs of ``openai`` (both
  languages) and ``litellm``. Docs:
  https://github.com/openai/openai-python#configuring-the-http-client ,
  https://docs.litellm.ai/docs/completion/input#input-params-1
* The OpenAI Responses API, which ``@ai-sdk/openai`` calling a model
  directly (``openai(modelId)``, no ``.chat``/``.responses`` suffix) now
  defaults to in v4.0.75 (``createLanguageModel`` -> ``createResponsesModel``)
  — the same literal-baseURL convention as above, with leaf ``/responses``
  instead of ``/chat/completions``. opencode's own provider layer
  (``anomalyco/opencode``, ``packages/llm/src/providers/openai.ts``)
  independently defaults ``model = responses`` too, so opencode's OpenAI
  preset is modelled on this convention below, not the chat one.
* Gemini CLI's ``generateContent`` route — ``/v1beta/models/{model}:generat
  eContent`` on a bare host, matching ``distil.adapters.gemini.is_gemini_path``
  (which this file also exercises as the ground truth for the shape rather
  than re-deriving it). Doc:
  https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md

codex/goose/copilot are deliberately NOT covered here: none has an
AGENT_ENV_TEMPLATES entry, and unlike the presets above I have no
independent, live-verified citation for what their own client does with a
bare base_url — asserting one would be encoding a guess as a passing test.
codex in particular turned out deeper than a leaf/``/v1`` question: codex-rs
is a native Rust client (not the OpenAI SDK the original preset comment
assumed), its ``ModelProviderInfo.base_url`` is populated only from the TOML
``openai_base_url`` config key, and no env-var-to-config-field mapping for it
was found in ``codex-rs/config`` — so setting ``OPENAI_BASE_URL`` may not
reach codex's request path *at all*, independent of ``/v1``. Left unfixed and
documented in ``onboard.py``'s ``AGENT_META["codex"]`` note rather than
guessed at here.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import pytest

from distil.adapters.gemini import is_gemini_path
from distil.httpguard import is_compressible_path
from distil.onboard import AGENT_ENV_TEMPLATES
from distil.proxy import _render_base_template, build_handler

# ---------------------------------------------------------------------------
# Fake upstream that records the path each request actually landed on
# ---------------------------------------------------------------------------


def _make_inspect_handler() -> type[BaseHTTPRequestHandler]:
    """A fresh handler class per test, so ``received`` isn't shared/leaked."""
    received: list[str] = []

    class _InspectHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
            pass

        def _record(self) -> None:
            received.append(self.path)
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = _record  # noqa: N815

    _InspectHandler.received = received  # type: ignore[attr-defined]
    return _InspectHandler


@pytest.fixture()
def servers() -> Any:
    """Yield (proxy_port, received_paths); shut both servers down after."""
    handler_cls = _make_inspect_handler()
    upstream_server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    upstream_port = upstream_server.server_address[1]
    upstream_thread = threading.Thread(target=upstream_server.serve_forever, daemon=True)
    upstream_thread.start()

    upstream_url = f"http://127.0.0.1:{upstream_port}"
    proxy_handler_cls = build_handler(upstream_url)
    proxy_server = ThreadingHTTPServer(("127.0.0.1", 0), proxy_handler_cls)
    proxy_port = proxy_server.server_address[1]
    proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    proxy_thread.start()

    yield proxy_port, handler_cls.received  # type: ignore[attr-defined]

    proxy_server.shutdown()
    upstream_server.shutdown()


def _post(port: int, path: str, payload: dict[str, Any]) -> urllib.request.Request:
    body = json.dumps(payload).encode()
    return urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        method="POST",
    )


# ---------------------------------------------------------------------------
# SDK conventions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SdkConvention:
    """How one client SDK turns its configured base_url into a request path.

    ``leaf`` is appended verbatim. ``auto_insert`` is the path segment
    (``"v1"``/``"v1beta"``) the SDK inserts itself even over a bare
    base_url, or ``None`` when the SDK uses the base_url LITERALLY — in
    which case whatever's exported must already carry that segment, which is
    exactly what AGENT_ENV_TEMPLATES is for.
    """

    leaf: str
    auto_insert: str | None
    doc_url: str


ANTHROPIC_SDK = SdkConvention(
    leaf="/messages",
    auto_insert="v1",
    doc_url="https://github.com/anthropics/anthropic-sdk-python#usage",
)
AI_SDK_LITERAL = SdkConvention(
    leaf="/messages",
    auto_insert=None,
    doc_url="https://github.com/Kilo-Org/kilocode/blob/main/packages/kilo-docs/pages/"
    "contributing/architecture/cli-runtime.md",
)
OPENAI_SDK_LITERAL = SdkConvention(
    leaf="/chat/completions",
    auto_insert=None,
    doc_url="https://github.com/openai/openai-python#configuring-the-http-client",
)
OPENAI_RESPONSES_LITERAL = SdkConvention(
    leaf="/responses",
    auto_insert=None,
    doc_url="https://opencode.ai/docs/providers/",
)
GEMINI_SDK = SdkConvention(
    leaf="/models/gemini-2.0-flash:generateContent",
    auto_insert="v1beta",
    doc_url="https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md",
)


def _client_path(base_url: str, conv: SdkConvention) -> str:
    """The path *conv*'s SDK would request, given it was configured with *base_url*."""
    path = urlsplit(base_url).path.rstrip("/")
    if conv.auto_insert and not path.endswith(f"/{conv.auto_insert}"):
        path = f"{path}/{conv.auto_insert}"
    return f"{path}{conv.leaf}"


# A tool output long enough to clear every adapter's Tier-1 digest threshold
# (>= 6 lines). Two trailing turns push it outside each adapter's
# RECENCY_KEEP_TURNS=2 carve-out so it actually digests instead of staying
# verbatim as "too recent" — see tests/test_proxy.py and tests/test_gemini.py
# for the same pattern.
_BIG_OUTPUT = "\n".join(f"row {i}: value_{i} status=ok detail=lorem ipsum dolor" for i in range(20))

_ANTHROPIC_BODY = {
    "model": "claude-opus-4-5",
    "max_tokens": 8,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t0", "content": _BIG_OUTPUT}],
        },
        {"role": "user", "content": "next 1"},
        {"role": "user", "content": "next 2"},
    ],
}
_OPENAI_BODY = {
    "model": "gpt-5",
    "messages": [
        {"role": "tool", "tool_call_id": "c0", "content": _BIG_OUTPUT},
        {"role": "user", "content": "next 1"},
        {"role": "user", "content": "next 2"},
    ],
}
_OPENAI_RESPONSES_BODY = {
    "model": "gpt-5",
    "input": [
        {"type": "function_call_output", "call_id": "call_0", "output": _BIG_OUTPUT},
        {"type": "function_call_output", "call_id": "call_1", "output": _BIG_OUTPUT},
        {"type": "function_call_output", "call_id": "call_2", "output": _BIG_OUTPUT},
    ],
}
_GEMINI_BODY = {
    "contents": [
        {"role": "user", "parts": [{"text": "round 0"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "fetch", "args": {"id": 0}}}]},
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "fetch", "response": {"output": _BIG_OUTPUT}}}],
        },
        {"role": "user", "parts": [{"text": "round 1"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "fetch", "args": {"id": 1}}}]},
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "fetch", "response": {"output": _BIG_OUTPUT}}}],
        },
        {"role": "user", "parts": [{"text": "round 2"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "fetch", "args": {"id": 2}}}]},
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "fetch", "response": {"output": _BIG_OUTPUT}}}],
        },
    ]
}


# ---------------------------------------------------------------------------
# Per-preset exported base URL, mirroring wrap_run's own resolution
# (distil/proxy.py: ``env_value = _render_base_template(...) if template else base``)
# ---------------------------------------------------------------------------


def _vibe_client_base(proxy_base: str) -> str:
    exported = _render_base_template(AGENT_ENV_TEMPLATES["vibe"], proxy_base)
    return json.loads(exported)[0]["api_base"]


def _kilo_client_base(proxy_base: str, provider: str) -> str:
    exported = _render_base_template(AGENT_ENV_TEMPLATES["kilo"], proxy_base)
    return json.loads(exported)["provider"][provider]["options"]["baseURL"]


def _templated_client_base(cmd: str, proxy_base: str) -> str:
    return _render_base_template(AGENT_ENV_TEMPLATES[cmd], proxy_base)


CASES = [
    # (id, provider, convention, base_of(proxy_base) -> client's configured base_url)
    ("claude", "anthropic", ANTHROPIC_SDK, lambda b: b),
    ("gemini", "gemini", GEMINI_SDK, lambda b: b),
    ("vibe", "openai", OPENAI_SDK_LITERAL, _vibe_client_base),
    ("kilo-anthropic", "anthropic", AI_SDK_LITERAL, lambda b: _kilo_client_base(b, "anthropic")),
    ("kilo-openai", "openai", OPENAI_SDK_LITERAL, lambda b: _kilo_client_base(b, "openai")),
    ("aider", "openai", OPENAI_SDK_LITERAL, lambda b: _templated_client_base("aider", b)),
    (
        "opencode",
        "openai_responses",
        OPENAI_RESPONSES_LITERAL,
        lambda b: _templated_client_base("opencode", b),
    ),
    ("qwen", "openai", OPENAI_SDK_LITERAL, lambda b: _templated_client_base("qwen", b)),
    ("grok", "openai", OPENAI_SDK_LITERAL, lambda b: _templated_client_base("grok", b)),
    ("kimi", "openai", OPENAI_SDK_LITERAL, lambda b: _templated_client_base("kimi", b)),
    ("openhands", "openai", OPENAI_SDK_LITERAL, lambda b: _templated_client_base("openhands", b)),
]

_BODY_BY_PROVIDER = {
    "anthropic": _ANTHROPIC_BODY,
    "openai": _OPENAI_BODY,
    "openai_responses": _OPENAI_RESPONSES_BODY,
    "gemini": _GEMINI_BODY,
}


@pytest.mark.parametrize(
    ("preset_id", "provider", "convention", "base_of"),
    CASES,
    ids=[c[0] for c in CASES],
)
def test_preset_reaches_a_compressible_path(
    preset_id: str,
    provider: str,
    convention: SdkConvention,
    base_of: Any,
    servers: Any,
) -> None:
    proxy_port, received = servers
    proxy_base = f"http://127.0.0.1:{proxy_port}"
    client_base = base_of(proxy_base)
    client_path = _client_path(client_base, convention)

    # (a) the path the preset's client would request is one the proxy treats
    # as compressible (or, for Gemini, routable to its own adapter) — not a
    # path that silently falls through to unrecognised passthrough.
    checker = is_gemini_path if provider == "gemini" else is_compressible_path
    assert checker(client_path), (
        f"{preset_id}: {client_path!r} is not a path distil's proxy compresses "
        f"(convention doc: {convention.doc_url})"
    )

    req = _post(proxy_port, client_path, _BODY_BY_PROVIDER[provider])
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"{preset_id}: proxy returned {resp.status} for {client_path!r}"
        tokens_saved = resp.headers.get("x-distil-tokens-saved")
        compressed_header = resp.headers.get("x-distil-compressed")

    # (b) the upstream received that EXACT path — no silent rewrite/drop between
    # the proxy accepting the request and forwarding it on.
    assert received[-1] == client_path, (
        f"{preset_id}: upstream received {received[-1]!r}, expected {client_path!r}"
    )

    # (c) the compression branch actually RAN on this path, not just passthrough:
    # every body above carries a genuinely-compressible tool result, so a real
    # digest must have fired and the response headers must say so.
    assert compressed_header == "1", (
        f"{preset_id}: x-distil-compressed missing or wrong ({compressed_header!r}) — "
        f"the request landed on {client_path!r} but never reached the compression branch"
    )
    assert tokens_saved is not None and int(tokens_saved) > 0, (
        f"{preset_id}: x-distil-tokens-saved was {tokens_saved!r} — the compressible "
        "tool result in this case's body should have produced a real digest"
    )


# ---------------------------------------------------------------------------
# /v1/v1: current (unfixed) behaviour, documented rather than patched over.
# ---------------------------------------------------------------------------
#
# httpguard's `_CHAT_RE`/`_RESPONSES_RE` are anchored (`^...$`), and
# `distil.proxy._post_upstream` forwards `_upstream + self.path` unchanged. A
# client whose base_url already carries `/v1` and appends its OWN `/v1` leaf on
# top (misconfiguration, not any preset distil ships today) lands on
# `/v1/v1/chat/completions`, which neither regex matches. This is deliberate:
# loosening the allowlist to also match a doubled prefix would make a
# genuinely malformed path look "compressible" and forward it to a URL that
# 404s at the real upstream anyway. This test pins that the proxy still
# forwards it as an uncompressed passthrough rather than silently dropping it
# or (worse) matching it.


def test_double_v1_prefix_is_not_compressed_but_still_forwarded(servers: Any) -> None:
    proxy_port, received = servers
    doubled_path = "/v1/v1/chat/completions"

    assert not is_compressible_path(doubled_path), (
        "a doubled /v1/v1 prefix must stay outside the allowlist — matching it "
        "would forward a malformed path as if it were a real endpoint"
    )

    req = _post(proxy_port, doubled_path, _OPENAI_BODY)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, (
            f"passthrough should still reach the fake upstream: {resp.status}"
        )
        compressed_header = resp.headers.get("x-distil-compressed")

    assert received[-1] == doubled_path, (
        f"the proxy must forward the path unchanged, not rewrite it: got {received[-1]!r}"
    )
    assert compressed_header != "1", (
        "a path the allowlist rejects must never report as compressed — that "
        "would be a false savings claim on a request that never got a real digest"
    )
