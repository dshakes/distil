"""Codex opens the Responses API as a WebSocket; distil is HTTP-only.

codex-cli 0.142.5 (verified locally against a stub): OPENAI_BASE_URL is ignored and the
built-in provider sends ``GET /v1/responses`` + ``Upgrade: websocket`` first. `wrap` must
force HTTP via a custom provider, and the proxy must refuse (not silently relay) an upgrade.
"""

from __future__ import annotations

import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from distil import config_wrap
from distil.proxy import QuietHTTPServer, build_handler


def test_codex_preset_forces_http_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with config_wrap.CONFIG_PRESETS["codex"].apply(
        "https://api.openai.com", "http://127.0.0.1:9"
    ) as argv:
        assert argv[:2] == ["-c", 'model_provider="distil"']
        prov = argv[3]
        assert 'base_url="http://127.0.0.1:9/v1"' in prov
        assert "supports_websockets=false" in prov and 'wire_api="responses"' in prov
        assert 'env_key="OPENAI_API_KEY"' in prov
        assert "sk-test" not in " ".join(argv)  # never embeds the credential


def test_codex_preset_without_key_uses_codex_login(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with config_wrap.CONFIG_PRESETS["codex"].apply("https://api.openai.com", "http://x") as argv:
        assert "requires_openai_auth=true" in argv[3] and "env_key" not in argv[3]


def test_proxy_refuses_websocket_upgrade_without_touching_upstream():
    hits: list[str] = []

    class Up(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):  # noqa: ANN002
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    px = QuietHTTPServer(
        ("127.0.0.1", 0), build_handler(f"http://127.0.0.1:{up.server_address[1]}")
    )
    threading.Thread(target=px.serve_forever, daemon=True).start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", px.server_address[1], timeout=5)
        c.request(
            "GET",
            "/v1/responses",
            headers={
                "Connection": "Upgrade",
                "Upgrade": "websocket",
                "OpenAI-Beta": "responses_websockets=2026-02-06",
            },
        )
        r = c.getresponse()
        assert r.status == 426 and b"websocket_unsupported" in r.read()
        assert hits == []
        # a plain GET still passes through
        c.request("GET", "/v1/models")
        assert c.getresponse().status == 200 and hits == ["/v1/models"]
    finally:
        px.shutdown()
        up.shutdown()
