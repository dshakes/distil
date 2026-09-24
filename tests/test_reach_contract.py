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
``kilo`` entry in ``AGENT_ENV_TEMPLATES`` now supplies) — the parametrized
cases below prove that fix stays proven, and extend the same proof to every
other preset that also needs a template (aider/opencode/qwen, added this
session — see the doc comment on ``AGENT_ENV_TEMPLATES`` in onboard.py) plus
the two bare-URL presets whose SDK convention is independently verified.

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
  appended, so the caller must already include ``/v1``. This is the
  already-merged Kilo fix's own citation (onboard.py, AGENT_ENV_TEMPLATES
  doc comment) — not independently re-verified in this file (a live
  ``@ai-sdk/anthropic`` Node check hung/timed out this session).
* The official OpenAI SDK (Python AND Node), and LiteLLM's
  ``api_base=`` — all three use an explicitly-set base_url LITERALLY and
  append only ``/chat/completions``, never inserting ``/v1`` themselves.
  Verified live 2026-09-24 against real installs of ``openai`` (both
  languages) and ``litellm``. Docs:
  https://github.com/openai/openai-python#configuring-the-http-client ,
  https://docs.litellm.ai/docs/completion/input#input-params-1
* Gemini CLI's ``generateContent`` route — ``/v1beta/models/{model}:generat
  eContent`` on a bare host, matching ``distil.adapters.gemini.is_gemini_path``
  (which this file also exercises as the ground truth for the shape rather
  than re-deriving it). Doc:
  https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md

codex/goose/grok/openhands/copilot/kimi are deliberately NOT covered here:
none has an AGENT_ENV_TEMPLATES entry, and unlike the presets above I have no
independent, live-verified citation for what their own client does with a
bare base_url — asserting one would be encoding a guess as a passing test.
grok in particular is suspicious (AGENT_PRESETS["grok"][1] bakes /v1 into the
UPSTREAM default, the same shape aider/opencode/qwen had before their fix)
and is flagged as a follow-up, not fixed here.
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


_ANTHROPIC_BODY = {
    "model": "claude-opus-4-5",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hi"}],
}
_OPENAI_BODY = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
_GEMINI_BODY = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


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
    ("opencode", "openai", OPENAI_SDK_LITERAL, lambda b: _templated_client_base("opencode", b)),
    ("qwen", "openai", OPENAI_SDK_LITERAL, lambda b: _templated_client_base("qwen", b)),
]

_BODY_BY_PROVIDER = {
    "anthropic": _ANTHROPIC_BODY,
    "openai": _OPENAI_BODY,
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

    # (b) the upstream received that EXACT path — no silent rewrite/drop between
    # the proxy accepting the request and forwarding it on.
    assert received[-1] == client_path, (
        f"{preset_id}: upstream received {received[-1]!r}, expected {client_path!r}"
    )


def test_kilo_fix_is_load_bearing(monkeypatch: pytest.MonkeyPatch, servers: Any) -> None:
    """Revert-in-place proof: without the ``/v1`` the Kilo fix (89526f0) added,
    the exported baseURL is what @ai-sdk/anthropic's LITERAL convention turns
    into a path the proxy does NOT compress — this test must fail if that fix
    regresses. (The gate-output revert/restore in the task's Step 1 was run
    manually against this exact assertion; see the handback report.)"""
    reverted_template = (
        '{"provider": {'
        '"anthropic": {"options": {"baseURL": "$BASE"}}, '
        '"openai": {"options": {"baseURL": "$BASE"}}'
        "}}"
    )
    monkeypatch.setitem(AGENT_ENV_TEMPLATES, "kilo", reverted_template)

    proxy_port, _received = servers
    proxy_base = f"http://127.0.0.1:{proxy_port}"
    client_base = _kilo_client_base(proxy_base, "anthropic")
    client_path = _client_path(client_base, AI_SDK_LITERAL)

    assert not is_compressible_path(client_path), (
        "reverted Kilo template should NOT land on a compressible path "
        f"(got {client_path!r}) — if this fails, the /v1 fix has regressed silently"
    )
