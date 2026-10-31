"""Codex opens the Responses API as a WebSocket; distil is HTTP-only.

codex-cli 0.142.5 (verified locally against a stub): OPENAI_BASE_URL is ignored and the
built-in provider sends ``GET /v1/responses`` + ``Upgrade: websocket`` first. `wrap` must
force HTTP via a custom provider, and the proxy must refuse (not silently relay) an upgrade.
"""

from __future__ import annotations

import http.client
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from distil import config_wrap, proxy
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


def test_codex_preset_without_key_uses_codex_login(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
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


def _fake_codex(tmp_path):
    script = tmp_path / "codex"
    out = str(tmp_path / "argv")
    body = f"import sys, pathlib\npathlib.Path({out!r}).write_text('\\0'.join(sys.argv[1:]))\n"
    script.write_text(f"#!{sys.executable}\n{body}")
    script.chmod(0o755)
    return str(script)


@pytest.mark.skipif(sys.platform == "win32", reason="shebang fake codex is POSIX-only")
@pytest.mark.parametrize(
    "tail", [["exec", "--", "prompt"], ["prompt"], ["exec", "-m", "x", "prompt"]]
)
def test_wrap_puts_codex_overrides_before_user_args(tmp_path, monkeypatch, tail):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    code = proxy.wrap_run(
        [_fake_codex(tmp_path), *tail],
        upstream="http://127.0.0.1:9",
        record=False,
        env_var="OPENAI_BASE_URL",
        config_ctx=config_wrap.CONFIG_PRESETS["codex"].apply,
    )
    argv = (tmp_path / "argv").read_text().split("\0")
    assert code == 0 and argv[:1] == ["-c"] and argv[4:] == tail
    assert "supports_websockets=false" in argv[3]


def test_codex_chatgpt_login_is_not_injected_and_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text('{"auth_mode": "chatgpt", "tokens": {}}')
    with config_wrap.CONFIG_PRESETS["codex"].apply("https://api.openai.com", "http://x") as argv:
        assert argv == []
    assert "BYPASSES distil" in capsys.readouterr().out
    (tmp_path / "auth.json").write_text('{"auth_mode": "apikey"}')
    with config_wrap.CONFIG_PRESETS["codex"].apply("https://api.openai.com", "http://x") as argv:
        assert argv[0] == "-c"
