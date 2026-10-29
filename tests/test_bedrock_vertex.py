"""Bedrock InvokeModel / Vertex AI (Anthropic + Gemini) routes through the proxy.

SigV4 (Bedrock) signs Host + body hash, so it cannot pass a proxy hop: refused with a
pointer to Bedrock API keys (bearer, body unsigned). Vertex OAuth bearers sign nothing.
Converse has a different body shape and is forwarded untouched.
"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from distil.httpguard import is_compressible_path, safe_forward_path
from distil.proxy import QuietHTTPServer, build_handler
from distil.serve_core import _model_from_path

LOGS = (
    "get_logs()\n" + "\n".join(f"info: verbose log line {i}" for i in range(60)) + "\nDECISION: act"
)
ANTHROPIC = {
    "messages": [
        {"role": "user", "content": "go"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": LOGS}],
        },
        {"role": "user", "content": "next"},
        {"role": "user", "content": "next"},
        {"role": "user", "content": "next"},
    ],
    "max_tokens": 64,
}
BEDROCK = {**ANTHROPIC, "anthropic_version": "bedrock-2023-05-31"}
VERTEX = {**ANTHROPIC, "anthropic_version": "vertex-2023-10-16"}
GEMINI = {
    "contents": [
        {"role": "user", "parts": [{"text": "go"}]},
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "get_logs", "response": {"output": LOGS}}}],
        },
        {"role": "user", "parts": [{"text": "next"}]},
        {"role": "user", "parts": [{"text": "next"}]},
        {"role": "user", "parts": [{"text": "next"}]},
    ]
}

BR = "/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/invoke"
VX = "/v1/projects/p/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
VG = "/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-pro:generateContent"


@pytest.fixture
def stack():
    seen: list[dict] = []

    class Up(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.append({"path": self.path, "raw": raw, "auth": self.headers.get("Authorization")})
            out = b"\x00\x01binary-eventstream\xff" if self.path.endswith("-stream") else b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):  # noqa: ANN002
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    servers = [up]

    def start(**kw):
        px = QuietHTTPServer(
            ("127.0.0.1", 0), build_handler(f"http://127.0.0.1:{up.server_address[1]}", **kw)
        )
        threading.Thread(target=px.serve_forever, daemon=True).start()
        servers.append(px)
        return px.server_address[1]

    def post(port, path, body, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        c.request(
            "POST", path, body=raw, headers={"Content-Type": "application/json", **(headers or {})}
        )
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read(), raw

    yield start, post, seen
    for s in servers:
        s.shutdown()


@pytest.mark.parametrize(
    ("path", "body"),
    [(BR, BEDROCK), (VX, VERTEX), (VG, GEMINI)],
    ids=["bedrock", "vertex-claude", "vertex-gemini"],
)
def test_route_is_compressed_and_bearer_passes(stack, path, body):
    start, post, seen = stack
    st, hdrs, _, raw = post(start(), path, body, {"Authorization": "Bearer tok"})
    h = {k.lower(): v for k, v in hdrs.items()}
    assert st == 200 and int(h["x-distil-tokens-saved"]) > 0
    assert seen[0]["path"] == path and seen[0]["auth"] == "Bearer tok"  # '@' and %3A intact
    sent = json.loads(seen[0]["raw"])
    assert len(seen[0]["raw"]) < len(raw)
    assert ("anthropic_version" in sent) == ("anthropic_version" in body)  # version kept


def test_bedrock_sigv4_refused_without_touching_upstream(stack):
    start, post, seen = stack
    for p in (BR, "/model/m/converse"):
        st, _, out, _ = post(
            start(), p, BEDROCK, {"Authorization": "AWS4-HMAC-SHA256 Credential=x"}
        )
        assert st == 400 and b"AWS_BEARER_TOKEN_BEDROCK" in out
    assert seen == []


def test_converse_and_non_anthropic_invoke_are_forwarded_untouched(stack):
    start, post, seen = stack
    port = start()
    nova = {"messages": ANTHROPIC["messages"][:1] + [{"role": "user", "content": [{"text": LOGS}]}]}
    for path, body in (
        ("/model/m/converse", ANTHROPIC),
        ("/model/amazon.nova-pro-v1%3A0/invoke", nova),
    ):
        _, _, _, raw = post(port, path, body, {"Authorization": "Bearer t"})
        assert seen[-1]["raw"] == raw


def test_eventstream_is_verbatim_with_no_expand_tool_and_relayed_raw(stack):
    start, post, seen = stack
    st, _, out, _ = post(
        start(expand=True), BR + "-with-response-stream", BEDROCK, {"Authorization": "Bearer t"}
    )
    assert st == 200 and out == b"\x00\x01binary-eventstream\xff"
    assert "distil_expand" not in seen[0]["raw"].decode()


def test_paths_and_model_extraction():
    assert _model_from_path(BR) == "anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert _model_from_path(VX) == "claude-sonnet-4-5@20250929"
    assert _model_from_path(VG) == "gemini-2.5-pro"
    assert (
        is_compressible_path(BR)
        and is_compressible_path(VX)
        and not is_compressible_path("/model/m/converse")
    )
    assert safe_forward_path(VX) == VX
    assert (
        safe_forward_path("/x@evil.com/y") is None
        and safe_forward_path("/v1/models/a@b/c@d") is None
    )


def test_bedrock_inference_profile_prices_like_base_model():
    from distil import pricing

    base = pricing.resolve("claude-haiku-4-5")
    assert base is not None
    for mid in (
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    ):
        assert pricing.resolve(mid) == base
