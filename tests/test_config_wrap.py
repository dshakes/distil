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
import os
import stat
import subprocess
import sys
import threading
import time
from typing import NamedTuple

import pytest

from distil import config_wrap


# ---------------------------------------------------------------------------
# _family
# ---------------------------------------------------------------------------


def test_family_infers_from_upstream_host():
    assert config_wrap._family("https://api.anthropic.com") == "anthropic"
    assert config_wrap._family("https://api.openai.com") == "openai"
    assert config_wrap._family("http://127.0.0.1:9999") == "openai"  # unknown → openai default


# ---------------------------------------------------------------------------
# _pid_is_alive — the Windows branch only actually runs on Windows CI, so
# fake os.name + ctypes.windll here to exercise its real logic everywhere
# else. Regression coverage for a real bug: a non-null OpenProcess handle
# does NOT mean alive (Windows can still hand one back for an exited pid) —
# only GetExitCodeProcess() == STILL_ACTIVE does.
# ---------------------------------------------------------------------------


def test_pid_is_alive_windows_branch_checks_exit_code_not_just_the_handle(monkeypatch):
    import ctypes
    import types

    monkeypatch.setattr(config_wrap.os, "name", "nt")
    state = {"handle": 1234, "exit_code": 259, "last_error": 0}  # STILL_ACTIVE = 259

    def fake_open_process(access, inherit, pid):
        return state["handle"]

    def fake_get_exit_code_process(handle, out_ptr):
        out_ptr._obj.value = state["exit_code"]
        return 1  # non-zero == WinAPI success

    fake_kernel32 = types.SimpleNamespace(
        OpenProcess=fake_open_process,
        GetExitCodeProcess=fake_get_exit_code_process,
        CloseHandle=lambda h: 1,
        GetLastError=lambda: state["last_error"],
    )
    monkeypatch.setattr(
        ctypes, "windll", types.SimpleNamespace(kernel32=fake_kernel32), raising=False
    )

    # A handle whose exit code is STILL_ACTIVE → alive.
    assert config_wrap._pid_is_alive(999) is True

    # The bug this guards against: same non-null handle, but the process has
    # actually exited.
    state["exit_code"] = 0
    assert config_wrap._pid_is_alive(999) is False

    # OpenProcess itself fails because the pid never existed → confirmed dead.
    state["handle"] = 0
    state["last_error"] = 87  # ERROR_INVALID_PARAMETER
    assert config_wrap._pid_is_alive(999) is False

    # OpenProcess fails for any other reason (e.g. access denied) → fail-safe alive.
    state["last_error"] = 5  # ERROR_ACCESS_DENIED
    assert config_wrap._pid_is_alive(999) is True


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


def test_omp_patch_replaces_its_own_block_instead_of_stacking_a_second():
    """Patching an already-patched models.yml must REPLACE distil's fenced
    block, not append another one: two blocks mean duplicate `distil:` keys
    under `providers:`, and one block too many for a single restore to undo."""
    original = "providers:\n  spark:\n    baseUrl: http://elsewhere\n"
    once = config_wrap._omp_patch(
        original, config_wrap._omp_fenced_block("anthropic", "http://a", "k")
    )
    twice = config_wrap._omp_patch(
        once, config_wrap._omp_fenced_block("anthropic", "http://b", "k")
    )

    assert twice.count(config_wrap._MARKER_BEGIN) == 1
    assert twice.count(config_wrap._MARKER_END) == 1
    assert "baseUrl: http://b" in twice and "baseUrl: http://a" not in twice
    assert "spark:" in twice, "the user's own provider must survive the re-patch"


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
# 0600 + atomic writes — every write that can carry a credential.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
@pytest.mark.parametrize(
    ("preset_name", "path_attr", "upstream", "key_var"),
    [
        ("droid", "_factory_settings_path", "https://api.openai.com", "OPENAI_API_KEY"),
        ("omp", "_omp_models_path", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
        ("crush", "_crush_config_path", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    ],
)
def test_apply_creates_config_file_owner_only(
    tmp_path, monkeypatch, preset_name, path_attr, upstream, key_var
):
    path = tmp_path / "cfg"
    monkeypatch.setattr(config_wrap, path_attr, lambda: path)
    monkeypatch.setenv(key_var, "sk-test")

    with config_wrap.CONFIG_PRESETS[preset_name].apply(upstream, "http://127.0.0.1:1234"):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"{preset_name} wrote {path} with a credential at mode {oct(mode)}"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_apply_backup_of_a_preexisting_file_is_also_owner_only(tmp_path, monkeypatch):
    path = tmp_path / "models.yml"
    path.write_text("providers:\n  spark:\n    baseUrl: http://elsewhere\n")
    path.chmod(0o644)  # the user's own file, at the ordinary umask default
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    with config_wrap._omp_apply("https://api.anthropic.com", "http://127.0.0.1:1234"):
        backup = config_wrap._backup_path(path)
        mode = stat.S_IMODE(backup.stat().st_mode)
        assert mode == 0o600, "the backup is a copy of the user's config and may carry its own keys"


def test_atomic_write_secure_leaves_original_untouched_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "target"
    path.write_bytes(b"original bytes")

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(config_wrap.os, "replace", _boom)
    with pytest.raises(OSError):
        config_wrap._atomic_write_secure(path, b"new bytes")

    assert path.read_bytes() == b"original bytes", "a failed replace must not touch the target"
    leftovers = list(tmp_path.iterdir())
    assert leftovers == [path], f"a temp file survived the failed write: {leftovers}"


def test_atomic_write_secure_round_trips_new_content(tmp_path):
    path = tmp_path / "target"
    config_wrap._atomic_write_secure(path, b"hello")
    assert path.read_bytes() == b"hello"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_secure_concurrent_writers_never_produce_a_mixed_file(tmp_path):
    """Regression: a STATIC temp name (``<name>.distil-tmp``) let two
    concurrent writers open/truncate the SAME temp file, so one's
    ``os.replace`` could publish the other's half-written bytes, or a stale
    temp could survive. A unique temp name per call (``tempfile.mkstemp``)
    makes that structurally impossible — the final file must be byte-exact
    to exactly one writer's payload, never a mix, and no temp survives
    either writer."""
    path = tmp_path / "target"
    payload_a = b"A" * 500_000
    payload_b = b"B" * 500_000
    start = threading.Barrier(2)
    errors: list[BaseException] = []

    def _write(data: bytes) -> None:
        start.wait()
        try:
            for _ in range(10):
                config_wrap._atomic_write_secure(path, data)
        except BaseException as exc:  # noqa: BLE001 — a collision must fail the test, not just warn
            errors.append(exc)

    t1 = threading.Thread(target=_write, args=(payload_a,))
    t2 = threading.Thread(target=_write, args=(payload_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"concurrent writers collided: {errors!r}"
    final = path.read_bytes()
    assert final in (payload_a, payload_b), (
        "the file must be one writer's whole payload, never a mix"
    )
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == [], f"a temp file survived concurrent writes: {leftovers}"


def test_config_survives_the_wrapped_child_being_killed(tmp_path, monkeypatch):
    """The wrap process itself keeps running when the CHILD dies by signal —
    proxy.wrap_run's finally block (which __exit__s the config context
    manager) must still run, restoring the config exactly as apply/restore
    do on a clean exit. Runs on Windows too, not skipped: SIGKILL doesn't
    exist there, and os.kill(pid, SIGTERM) maps to TerminateProcess(handle,
    SIGTERM) — a positive exit code, not POSIX's negative-signal convention
    — so the assertion branches on platform instead of assuming one."""
    import signal

    from distil import proxy

    path = tmp_path / "models.yml"
    original = "providers:\n  spark:\n    baseUrl: http://elsewhere\n"
    path.write_text(original)
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    code = proxy.wrap_run(
        [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
        record=False,
        config_ctx=config_wrap.CONFIG_PRESETS["omp"].apply,
    )
    if sys.platform == "win32":
        assert code != 0, "a terminated child must not report a clean exit"
    else:
        assert code == -signal.SIGTERM
    assert path.read_text() == original, "child crash left the injected provider block behind"
    assert not config_wrap._backup_path(path).exists()


# ---------------------------------------------------------------------------
# Idempotence — every strategy that touches a real config file. `cn` is absent
# on purpose: it renders a fresh self-contained temp config per invocation and
# never re-patches anything, so there is nothing to be idempotent about.
# ---------------------------------------------------------------------------


class _PatchTarget(NamedTuple):
    path_attr: str  # the module attribute naming the real config path
    filename: str
    upstream: str
    key_var: str
    original: str
    theirs: str  # a substring of the user's own config that must survive
    ours: str  # the substring that marks exactly ONE distil entry


_PATCH_TARGETS = {
    "droid": _PatchTarget(
        "_factory_settings_path",
        "settings.local.json",
        "https://api.openai.com",
        "OPENAI_API_KEY",
        '{"customModels": [{"model": "their-own"}]}',
        "their-own",
        '"model": "distil"',
    ),
    "omp": _PatchTarget(
        "_omp_models_path",
        "models.yml",
        "https://api.anthropic.com",
        "ANTHROPIC_API_KEY",
        "providers:\n  spark:\n    baseUrl: http://elsewhere\n",
        "spark:",
        config_wrap._MARKER_BEGIN,
    ),
    "crush": _PatchTarget(
        "_crush_config_path",
        "crush.json",
        "https://api.anthropic.com",
        "ANTHROPIC_API_KEY",
        '{"providers": {"spark": {"id": "spark"}}}',
        '"spark"',
        '"id": "distil"',
    ),
}


@pytest.mark.parametrize("tool", sorted(_PATCH_TARGETS))
def test_patching_an_already_patched_config_never_appends_a_second_entry(
    tool, tmp_path, monkeypatch
):
    """A config can legitimately arrive already carrying a `distil` entry — a
    concurrent sibling session put it there, or a crash left one behind — and
    a second patch must REPLACE it. Appending a second entry survives the
    restore (only one can be undone) and, for the YAML splice, writes a
    duplicate key. droid drops-and-re-appends its customModels entry and crush
    assigns a dict key, so both were already safe; omp appended."""
    target = _PATCH_TARGETS[tool]
    path = tmp_path / target.filename
    path.write_text(target.original)
    monkeypatch.setattr(config_wrap, target.path_attr, lambda: path)
    monkeypatch.setenv(target.key_var, "sk-test")
    preset = config_wrap.CONFIG_PRESETS[tool]

    with preset.apply(target.upstream, "http://a"):
        with preset.apply(target.upstream, "http://b"):  # patches over our own patch
            text = path.read_text()
            assert text.count(target.ours) == 1, f"a second distil entry was appended:\n{text}"
            assert target.theirs in text, "the user's own entry must survive the re-patch"
        assert path.read_text().count(target.ours) == 1

    assert path.read_text() == target.original, "the true original must come back byte-for-byte"
    assert not config_wrap._backup_path(path).exists()
    assert not config_wrap._registry_dir(path).exists()


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """A pid that provably isn't in use: spawned, exited, and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_crash_between_the_claim_and_the_backup_does_not_strand_the_injection(
    tmp_path, monkeypatch
):
    """The blocking defect. "Do I create the backup?" used to be answered by
    "was the session registry empty?", and the backup was written after the
    claim. A SIGKILL in that window left a registry holding one dead entry, so
    the NEXT `distil wrap` computed is_first=False, wrote no backup, and
    patched the real config anyway — the injected block was then permanent,
    with nothing on disk able to undo it. That crash is reproduced exactly
    here: a dead pid registered against the path, no backup, no sentinel."""
    path = tmp_path / "models.yml"
    original = "providers:\n  spark:\n    baseUrl: http://elsewhere\n"
    path.write_text(original)
    registry_dir = config_wrap._registry_dir(path)
    registry_dir.mkdir()
    (registry_dir / f"{_dead_pid()}.1").touch()  # claimed, then killed before backing up
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    with config_wrap._omp_apply("https://api.anthropic.com", "http://127.0.0.1:1234"):
        assert config_wrap._MARKER_BEGIN in path.read_text(), "nothing was injected at all"
        assert config_wrap._backup_path(path).read_text() == original, (
            "a dead session's claim must not stand in for a backup that was never written"
        )

    assert path.read_text() == original, "the injected block outlived the session"
    assert not config_wrap._backup_path(path).exists()
    assert not registry_dir.exists()


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


def test_restore_stale_backups_reclaims_a_registrant_from_a_dead_pid(tmp_path, monkeypatch):
    """A sentinel isn't proof of a crash by itself — but one whose recorded
    owner is verifiably dead is exactly the crash-leftover case restore_stale
    _backups exists for."""
    path = tmp_path / "models.yml"
    path.write_text("providers:\n  distil:\n    baseUrl: X\n")  # what the dead session left
    config_wrap._backup_path(path).write_text("providers:\n  spark: {}\n")  # the real original
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()  # reaped: this pid is now provably not in use
    registry_dir = config_wrap._registry_dir(path)
    registry_dir.mkdir()
    (registry_dir / f"{proc.pid}.1").touch()
    monkeypatch.setattr(config_wrap, "CONFIG_PRESETS", {"omp": config_wrap.CONFIG_PRESETS["omp"]})
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)

    config_wrap.restore_stale_backups()

    assert path.read_text() == "providers:\n  spark: {}\n"
    assert not config_wrap._backup_path(path).exists()
    assert not registry_dir.exists()


def test_restore_stale_backups_clears_a_registry_a_crash_left_without_a_backup(
    tmp_path, monkeypatch, capsys
):
    """Every registrant dead and NO backup or sentinel beside the config: the
    backup is written before the patch, so nothing was ever patched and there
    is nothing to restore — but the registry directory is a dead session's
    bookkeeping and must not be left behind claiming the path. The config
    itself is not touched, and nothing is reported (nothing was repaired)."""
    path = tmp_path / "models.yml"
    original = "providers:\n  spark: {}\n"
    path.write_text(original)
    registry_dir = config_wrap._registry_dir(path)
    registry_dir.mkdir()
    (registry_dir / f"{_dead_pid()}.1").touch()
    monkeypatch.setattr(config_wrap, "CONFIG_PRESETS", {"omp": config_wrap.CONFIG_PRESETS["omp"]})
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)

    config_wrap.restore_stale_backups()

    assert not registry_dir.exists(), "a dead session's registry kept the path claimed forever"
    assert path.read_text() == original, "there was nothing to restore — leave the config alone"
    assert capsys.readouterr().out == "", "nothing was repaired, so nothing to report"

    # ...and the same sweep must still leave a LIVE registrant's claim alone.
    registry_dir.mkdir()
    mine = registry_dir / f"{os.getpid()}.1"
    mine.touch()
    config_wrap.restore_stale_backups()
    assert mine.exists(), "a live session's claim was swept"


def test_restore_stale_backups_does_not_reclaim_a_live_pid(tmp_path, monkeypatch):
    """The exact scenario the fix targets: a second, unrelated `distil wrap`
    invocation must not restore a config a LIVE sibling session still owns,
    even though a backup/sentinel is sitting right there."""
    path = tmp_path / "models.yml"
    injected = "providers:\n  distil:\n    baseUrl: X\n"
    path.write_text(injected)
    config_wrap._backup_path(path).write_text("providers:\n  spark: {}\n")
    registry_dir = config_wrap._registry_dir(path)
    registry_dir.mkdir()
    my_entry = registry_dir / f"{os.getpid()}.1"  # this test process — definitely alive
    my_entry.touch()
    monkeypatch.setattr(config_wrap, "CONFIG_PRESETS", {"omp": config_wrap.CONFIG_PRESETS["omp"]})
    monkeypatch.setattr(config_wrap, "_omp_models_path", lambda: path)

    config_wrap.restore_stale_backups()

    assert path.read_text() == injected, "a live session's config must not be touched"
    assert config_wrap._backup_path(path).exists()
    assert my_entry.exists()


def test_overlapping_sessions_on_the_same_config_restore_to_the_true_original(
    tmp_path, monkeypatch
):
    """Two `distil wrap` sessions patching the SAME config concurrently must
    both end up restoring the true pre-either-session bytes once BOTH have
    exited, regardless of which one happens to exit first. Before per-session
    ownership, the second session's own "existed=True" snapshot already
    included the first session's injected entry, so it silently overwrote
    the shared backup — an early exit then restored an intermediate,
    already-modified state instead of the real original."""
    path = tmp_path / "crush.json"
    true_original = '{"providers": {"spark": {"id": "spark"}}}'
    monkeypatch.setattr(config_wrap, "_crush_config_path", lambda: path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    for exit_order in ("a_first", "b_first"):
        path.write_text(true_original)  # reset between the two sub-cases

        gen_a = config_wrap.CONFIG_PRESETS["crush"].apply("https://api.anthropic.com", "http://a")
        gen_b = config_wrap.CONFIG_PRESETS["crush"].apply("https://api.anthropic.com", "http://b")
        gen_a.__enter__()
        gen_b.__enter__()
        assert json.loads(path.read_text())["providers"].get("distil") is not None

        first, second = (gen_a, gen_b) if exit_order == "a_first" else (gen_b, gen_a)
        first.__exit__(None, None, None)
        # The non-last exit must NOT touch the shared file — the surviving
        # sibling session still depends on it staying injected.
        assert json.loads(path.read_text())["providers"].get("distil") is not None, exit_order

        second.__exit__(None, None, None)
        assert path.read_text() == true_original, f"exit order {exit_order} lost the true original"
        assert not config_wrap._backup_path(path).exists()
        assert not config_wrap._registry_dir(path).exists()


def test_session_lock_serializes_a_release_against_a_concurrent_claim(tmp_path):
    """Regression for the exact race `_session_lock` exists to close: A is the
    last live session and has already decided it must restore; B claims fresh
    before A's restore actually lands, and that restore then silently clobbers
    B's brand-new config. A is parked in exactly that window on purpose while
    B races to claim.

    Both threads are JOINED before anything is asserted, and what's asserted
    is the property rather than a fixed event ordering measured at a fixed
    instant: an earlier version pinned `order` while B was still legitimately
    blocked, which on Windows recorded only A's two events and failed — the
    lock doing its job, scored as a bug (msvcrt's LK_LOCK retries on a ~1s
    cadence, so B lands about a second later, not microseconds). Remove the
    lock and both assertions below still fail: B writes its config inside A's
    window, and A's restore overwrites it."""
    path = tmp_path / "crush.json"
    path.write_text("true original")
    config_wrap._backup_path(path).write_text("true original")
    registry_dir = config_wrap._registry_dir(path)
    registry_dir.mkdir()
    (registry_dir / "111.1").touch()  # A is the sole, about-to-release registrant

    order: list[str] = []
    a_in_the_restore_window = threading.Event()
    let_a_continue = threading.Event()
    errors: list[BaseException] = []

    def _release_a() -> None:
        # The caller-side shape of every preset's `finally`: release, then (if
        # last) restore — with the pause in the window between the two.
        try:
            with config_wrap._session_lock(path):
                last = config_wrap._release_session(registry_dir, "111.1")
                a_in_the_restore_window.set()
                let_a_continue.wait(timeout=30)
                if last:
                    path.write_text(config_wrap._backup_path(path).read_text())
                    order.append("a_restored")
        except BaseException as exc:  # noqa: BLE001 — a thread failure must fail the test
            errors.append(exc)

    def _claim_b() -> None:
        try:
            assert a_in_the_restore_window.wait(timeout=30), "A never reached the race window"
            with config_wrap._session_lock(path):
                order.append("b_claimed")
                config_wrap._claim_session(path)
                path.write_text("b's fresh config")
        except BaseException as exc:  # noqa: BLE001 — a thread failure must fail the test
            errors.append(exc)

    t_a = threading.Thread(target=_release_a)
    t_b = threading.Thread(target=_claim_b)
    t_a.start()
    t_b.start()
    assert a_in_the_restore_window.wait(timeout=30), "A never reached the race window"
    time.sleep(0.05)  # give B a real chance to try (and correctly fail) to jump the lock
    let_a_continue.set()
    t_a.join(timeout=60)
    t_b.join(timeout=60)

    assert not t_a.is_alive() and not t_b.is_alive(), "a thread never finished — lock deadlock?"
    assert not errors, f"a thread raised: {errors!r}"
    assert order == ["a_restored", "b_claimed"], (
        "B's claim must not straddle A's release-and-restore — it must fully follow it"
    )
    assert path.read_text() == "b's fresh config", "A's restore must not clobber B's fresh config"


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


def test_cmd_wrap_gives_droid_its_own_default_upstream_and_it_actually_writes(
    tmp_path, monkeypatch
):
    """A bare `distil wrap -- droid` used to inherit cmd_wrap's hardcoded
    Anthropic default upstream — which _droid_apply silently no-ops against,
    since it only fires for an OpenAI-shaped upstream — so the one-word
    preset injected nothing. Exercises the FULL cmd_wrap path, not a direct
    _droid_apply call, and proves the config actually gets written."""
    from distil.cli import cmd_wrap
    from tests.test_wrap_presets import _ns

    captured: dict = {}

    def fake(cmd, *, config_ctx=None, upstream=None, **kw):
        captured["config_ctx"] = config_ctx
        captured["upstream"] = upstream
        return 0

    monkeypatch.setattr("distil.proxy.wrap_run", fake)
    monkeypatch.setattr("distil.config_wrap.restore_stale_backups", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    path = tmp_path / "settings.local.json"
    monkeypatch.setattr(config_wrap, "_factory_settings_path", lambda: path)

    rc = cmd_wrap(_ns(command=["droid"]))
    assert rc == 0
    assert captured["upstream"] == "https://api.openai.com"

    with captured["config_ctx"](captured["upstream"], "http://127.0.0.1:9"):
        assert path.exists(), "droid's own default upstream must let _droid_apply actually write"
        doc = json.loads(path.read_text())
        assert any(m.get("model") == "distil" for m in doc["customModels"])
    assert not path.exists()  # restored (created-from-nothing) on exit


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
