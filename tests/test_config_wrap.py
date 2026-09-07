"""Config-file wrap targets (distil/config_wrap.py).

Covers: family inference, the three injection strategies (flag / overlay /
patch) each round-tripping a real backup/restore, refusing to invent a
credential, crash recovery via ``restore_stale_backups``, and one end-to-end
proof (a fake ``cn`` binary that actually reads the generated ``--config``
file and reaches a real upstream through the proxy) — the same "a request
provably arrived" bar tests/test_wrap_presets.py holds every env-var preset
to.
"""

from __future__ import annotations

import json
import sys


from distil import config_wrap


# ---------------------------------------------------------------------------
# _family
# ---------------------------------------------------------------------------


def test_family_infers_from_upstream_host():
    assert config_wrap._family("https://api.anthropic.com") == "anthropic"
    assert config_wrap._family("https://api.openai.com") == "openai"
    assert config_wrap._family("http://127.0.0.1:9999") == "openai"  # unknown → openai default


# ---------------------------------------------------------------------------
# Continue (`cn`) — flag strategy: nothing on disk but a throwaway temp file.
# ---------------------------------------------------------------------------


def test_continue_apply_writes_temp_config_and_cleans_up(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with config_wrap._continue_apply("https://api.anthropic.com", "http://127.0.0.1:1234") as argv:
        assert argv[0] == "--config"
        tmp_path = argv[1]
        text = open(tmp_path, encoding="utf-8").read()
        assert "apiBase: http://127.0.0.1:1234" in text
        assert "apiKey: sk-ant-test" in text
        assert "provider: anthropic" in text
    assert not __import__("os").path.exists(tmp_path), "temp config outlived the wrap"


def test_continue_apply_skips_without_a_credential(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with config_wrap._continue_apply("https://api.anthropic.com", "http://127.0.0.1:1234") as argv:
        assert argv == []
    assert "never inventing a credential" in capsys.readouterr().out


def test_a_config_flag_preset_actually_carries_a_request_to_the_provider(tmp_path, monkeypatch):
    """Same end-to-end bar as the env-var presets: a fake `cn` that reads the
    generated --config file's apiBase and makes a real request through it."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from distil import proxy

    seen: list[str] = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.append(self.path)
            body = json.dumps(
                {"id": "m", "content": [{"type": "text", "text": "ok"}], "model": "m"}
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # noqa: ANN002
            pass

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    # upstream below is a local http:// test server, not api.anthropic.com, so
    # _family() falls back to "openai" — set the key that family actually reads.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    up_url = f"http://127.0.0.1:{server.server_address[1]}"

    # Stands in for `cn --config <path>`: reads the generated YAML's apiBase
    # line with plain string parsing (no yaml dependency needed here either —
    # the fixture only needs to prove the value distil wrote is the value the
    # child actually used).
    child = (
        "import json, sys, urllib.request\n"
        "cfg = open(sys.argv[sys.argv.index('--config') + 1]).read()\n"
        "base = next(l for l in cfg.splitlines() if 'apiBase:' in l).split('apiBase:')[1].strip()\n"
        "body = json.dumps({'model':'m','max_tokens':4,"
        "'messages':[{'role':'user','content':'hi'}]}).encode()\n"
        "req = urllib.request.Request(base + '/v1/messages', data=body,\n"
        "    headers={'content-type':'application/json'})\n"
        "urllib.request.urlopen(req, timeout=10).read()\n"
        "sys.exit(0)\n"
    )
    try:
        code = proxy.wrap_run(
            [sys.executable, "-c", child],
            upstream=up_url,
            env_var="ANTHROPIC_BASE_URL",  # unused by the fake `cn`; config_ctx does the routing
            record=False,
            config_ctx=config_wrap.CONFIG_PRESETS["cn"].apply,
        )
    finally:
        server.shutdown()

    assert code == 0, "the fake cn could not reach the proxy through the generated --config"
    assert seen, "the request never arrived upstream — config injection routed nothing"


# ---------------------------------------------------------------------------
# Factory Droid (`droid`) — overlay strategy: settings.local.json.
# ---------------------------------------------------------------------------


def test_droid_apply_creates_and_deletes_overlay_when_absent(tmp_path, monkeypatch):
    path = tmp_path / "settings.local.json"
    monkeypatch.setattr(config_wrap, "_factory_settings_path", lambda: path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")

    with config_wrap._droid_apply("https://api.openai.com", "http://127.0.0.1:1234") as argv:
        assert argv == []
        doc = json.loads(path.read_text())
        [entry] = doc["customModels"]
        assert entry["baseUrl"] == "http://127.0.0.1:1234"
        assert entry["apiKey"] == "sk-oai-test"
        assert entry["provider"] == "generic-chat-completion-api"

    assert not path.exists(), "an overlay we created must not survive the wrap"
    assert not config_wrap._created_marker(path).exists()
    assert not config_wrap._backup_path(path).exists()


def test_droid_apply_merges_and_restores_when_present(tmp_path, monkeypatch):
    path = tmp_path / "settings.local.json"
    original = json.dumps(
        {"customModels": [{"model": "their-own", "baseUrl": "https://elsewhere"}]}
    )
    path.write_text(original)
    monkeypatch.setattr(config_wrap, "_factory_settings_path", lambda: path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")

    with config_wrap._droid_apply("https://api.openai.com", "http://127.0.0.1:1234"):
        doc = json.loads(path.read_text())
        ids = {m["model"] for m in doc["customModels"]}
        assert ids == {"their-own", "distil"}, "must ADD, not replace, the user's existing models"

    assert path.read_text() == original, "restore must be byte-for-byte, not a re-serialization"
    assert not config_wrap._backup_path(path).exists()


def test_droid_apply_skips_anthropic_family(tmp_path, monkeypatch, capsys):
    path = tmp_path / "settings.local.json"
    monkeypatch.setattr(config_wrap, "_factory_settings_path", lambda: path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")

    with config_wrap._droid_apply("https://api.anthropic.com", "http://127.0.0.1:1234") as argv:
        assert argv == []
    assert not path.exists()
    assert "only the OpenAI-compatible provider type" in capsys.readouterr().out


def test_droid_apply_skips_without_a_credential(tmp_path, monkeypatch):
    path = tmp_path / "settings.local.json"
    monkeypatch.setattr(config_wrap, "_factory_settings_path", lambda: path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with config_wrap._droid_apply("https://api.openai.com", "http://127.0.0.1:1234"):
        pass
    assert not path.exists()


# ---------------------------------------------------------------------------
# Oh My Pi (`omp`) — patch strategy: models.yml, no flag, no overlay layer.
# ---------------------------------------------------------------------------


def test_omp_patch_creates_providers_section_when_file_absent():
    text = config_wrap._omp_patch(None, "  distil:\n    baseUrl: X\n")
    assert text.startswith("providers:\n  distil:\n")


def test_omp_patch_inserts_after_existing_providers_key():
    original = "providers:\n  spark:\n    baseUrl: http://elsewhere\n"
    fenced = "  distil:\n    baseUrl: X\n"
    text = config_wrap._omp_patch(original, fenced)
    assert (
        text == "providers:\n  distil:\n    baseUrl: X\n  spark:\n    baseUrl: http://elsewhere\n"
    )


def test_omp_patch_appends_providers_section_when_key_missing():
    original = "modelRoles:\n  default: spark/x\n"
    fenced = "  distil:\n    baseUrl: X\n"
    text = config_wrap._omp_patch(original, fenced)
    assert text == "modelRoles:\n  default: spark/x\nproviders:\n  distil:\n    baseUrl: X\n"


def test_omp_apply_restores_exact_bytes_when_file_existed(tmp_path, monkeypatch):
    path = tmp_path / "models.yml"
    original = "providers:\n  spark:\n    baseUrl: http://elsewhere\n"
    path.write_text(original)
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    with config_wrap._omp_apply("https://api.anthropic.com", "http://127.0.0.1:1234"):
        text = path.read_text()
        assert "distil:" in text and "spark:" in text
        assert config_wrap._MARKER_BEGIN in text

    assert path.read_text() == original
    assert not config_wrap._backup_path(path).exists()


def test_omp_apply_deletes_when_file_absent(tmp_path, monkeypatch):
    path = tmp_path / "models.yml"
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")

    with config_wrap._omp_apply("https://api.openai.com", "http://127.0.0.1:1234"):
        assert path.exists()
        assert "api: openai-completions" in path.read_text()

    assert not path.exists()
    assert not config_wrap._created_marker(path).exists()


def test_omp_apply_skips_without_a_credential(tmp_path, monkeypatch):
    path = tmp_path / "models.yml"
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with config_wrap._omp_apply("https://api.anthropic.com", "http://127.0.0.1:1234"):
        pass
    assert not path.exists()


# ---------------------------------------------------------------------------
# Crush (`crush`) — patch strategy: the legacy crush.json, no override layer.
# ---------------------------------------------------------------------------


def test_crush_apply_creates_and_deletes_when_absent(tmp_path, monkeypatch):
    path = tmp_path / "crush.json"
    monkeypatch.setattr(config_wrap, "_crush_config_path", lambda: path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    with config_wrap._crush_apply("https://api.anthropic.com", "http://127.0.0.1:1234") as argv:
        assert argv == []
        doc = json.loads(path.read_text())
        entry = doc["providers"]["distil"]
        assert entry["base_url"] == "http://127.0.0.1:1234"
        assert entry["api_key"] == "sk-ant-test"
        assert entry["type"] == "anthropic"
        assert "models" not in entry, "no models array — discover_models defaults to true"

    assert not path.exists(), "a crush.json we created must not survive the wrap"
    assert not config_wrap._created_marker(path).exists()
    assert not config_wrap._backup_path(path).exists()


def test_crush_apply_merges_and_restores_when_present(tmp_path, monkeypatch):
    path = tmp_path / "crush.json"
    original = json.dumps({"providers": {"openai": {"id": "openai", "api_key": "$OPENAI_API_KEY"}}})
    path.write_text(original)
    monkeypatch.setattr(config_wrap, "_crush_config_path", lambda: path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")

    with config_wrap._crush_apply("https://api.openai.com", "http://127.0.0.1:1234"):
        doc = json.loads(path.read_text())
        assert set(doc["providers"]) == {"openai", "distil"}, "must ADD, not replace"
        assert doc["providers"]["distil"]["type"] == "openai"

    assert path.read_text() == original, "restore must be byte-for-byte, not a re-serialization"
    assert not config_wrap._backup_path(path).exists()


def test_crush_apply_skips_without_a_credential(tmp_path, monkeypatch):
    path = tmp_path / "crush.json"
    monkeypatch.setattr(config_wrap, "_crush_config_path", lambda: path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with config_wrap._crush_apply("https://api.anthropic.com", "http://127.0.0.1:1234") as argv:
        assert argv == []
    assert not path.exists()


def test_crush_apply_treats_unparseable_json_as_empty(tmp_path, monkeypatch):
    path = tmp_path / "crush.json"
    path.write_text("not json")
    monkeypatch.setattr(config_wrap, "_crush_config_path", lambda: path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    with config_wrap._crush_apply("https://api.anthropic.com", "http://127.0.0.1:1234"):
        doc = json.loads(path.read_text())
        assert list(doc["providers"]) == ["distil"]

    assert path.read_text() == "not json", "original garbage bytes are still restored exactly"


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def test_restore_stale_backups_restores_a_leftover_backup(tmp_path, monkeypatch):
    path = tmp_path / "models.yml"
    path.write_text(
        "providers:\n  distil:\n    baseUrl: X\n"
    )  # what a crash mid-session left behind
    config_wrap._backup_path(path).write_text("providers:\n  spark: {}\n")  # the real original
    monkeypatch.setattr(config_wrap, "CONFIG_PRESETS", {"omp": config_wrap.CONFIG_PRESETS["omp"]})
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)

    config_wrap.restore_stale_backups()

    assert path.read_text() == "providers:\n  spark: {}\n"
    assert not config_wrap._backup_path(path).exists()


def test_restore_stale_backups_deletes_a_leftover_created_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.local.json"
    path.write_text('{"customModels": [{"model": "distil"}]}')  # a crash left this behind
    config_wrap._created_marker(path).touch()
    monkeypatch.setattr(
        config_wrap, "CONFIG_PRESETS", {"droid": config_wrap.CONFIG_PRESETS["droid"]}
    )
    monkeypatch.setattr(config_wrap, "_factory_settings_path", lambda: path)

    config_wrap.restore_stale_backups()

    assert not path.exists()
    assert not config_wrap._created_marker(path).exists()


def test_restore_stale_backups_is_a_silent_no_op_when_nothing_is_stale(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: tmp_path / "models.yml")
    monkeypatch.setattr(
        config_wrap, "_factory_settings_path", lambda: tmp_path / "settings.local.json"
    )
    config_wrap.restore_stale_backups()
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# cmd_wrap wiring: a config-file command resolves a preset and passes it through.
# ---------------------------------------------------------------------------


def test_cmd_wrap_resolves_config_preset_for_droid(monkeypatch, capsys):
    from distil.cli import cmd_wrap
    from tests.test_wrap_presets import _ns

    captured: dict = {}

    def fake(cmd, *, config_ctx=None, **kw):
        captured["config_ctx"] = config_ctx
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr("distil.proxy.wrap_run", fake)
    monkeypatch.setattr("distil.config_wrap.restore_stale_backups", lambda: None)
    rc = cmd_wrap(_ns(command=["droid"]))
    assert rc == 0
    assert captured["config_ctx"] is config_wrap.CONFIG_PRESETS["droid"].apply
    assert "Factory Droid" in capsys.readouterr().out


def test_cmd_wrap_leaves_config_ctx_none_for_a_normal_preset(monkeypatch):
    from distil.cli import cmd_wrap
    from tests.test_wrap_presets import _ns

    captured: dict = {}

    def fake(cmd, *, config_ctx=None, **kw):
        captured["config_ctx"] = config_ctx
        return 0

    monkeypatch.setattr("distil.proxy.wrap_run", fake)
    monkeypatch.setattr("distil.config_wrap.restore_stale_backups", lambda: None)
    rc = cmd_wrap(_ns(command=["claude"]))
    assert rc == 0
    assert captured["config_ctx"] is None
