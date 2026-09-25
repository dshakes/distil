"""distil mcp install: rewrite a client's MCP config through the proxy, and undo it exactly."""

from __future__ import annotations

import argparse
import json
import stat
import sys

import pytest

from distil import cli
from distil.mcpproxy import install as mi

CONFIG = {
    "theme": "dark",
    "mcpServers": {
        "fs": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/w"],
            "env": {"K": "v"},
        },
        "remote": {"url": "https://mcp.example.com/sse"},
        "git": {"command": "uvx", "args": ["mcp-server-git"]},
    },
}


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.setattr(mi, "distil_command", lambda: "/opt/bin/distil")


def _write(path, data, mode=0o644):
    path.write_text(json.dumps(data, indent=4) + "\n")
    path.chmod(mode)
    return path.read_bytes()


def test_install_wraps_stdio_servers_only_and_undo_is_byte_exact(tmp_path, launcher):
    cfg = tmp_path / "mcp.json"
    original = _write(cfg, CONFIG)
    status, msg = mi.install("custom", path=cfg, level="L2")
    assert status == "ok" and "fs, git" in msg
    data = json.loads(cfg.read_text())
    fs = data["mcpServers"]["fs"]
    assert fs["command"] == "/opt/bin/distil"
    assert fs["args"] == [
        "mcp",
        "wrap",
        "--name",
        "fs",
        "--level",
        "L2",
        "--",
        "npx",
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/w",
    ]
    assert fs["env"] == {"K": "v"}  # untouched
    assert data["mcpServers"]["remote"] == CONFIG["mcpServers"]["remote"]  # never touched
    assert data["theme"] == "dark"
    if sys.platform != "win32":
        assert stat.S_IMODE(cfg.stat().st_mode) == 0o600
    backup = cfg.with_name(cfg.name + mi.BACKUP_SUFFIX)
    assert backup.read_bytes() == original
    # idempotent: nothing left to wrap
    assert mi.install("custom", path=cfg)[0] == "exists"
    status, _ = mi.uninstall("custom", path=cfg)
    assert status == "restored"
    assert cfg.read_bytes() == original  # byte-exact, including the 4-space indent
    if sys.platform != "win32":
        assert stat.S_IMODE(cfg.stat().st_mode) == 0o644  # and the original mode
    assert not backup.exists()
    assert mi.uninstall("custom", path=cfg)[0] == "absent"


def test_undo_after_user_edits_keeps_their_edits(tmp_path, launcher):
    cfg = tmp_path / "mcp.json"
    _write(cfg, CONFIG)
    mi.install("custom", path=cfg, results=False)
    data = json.loads(cfg.read_text())
    assert "--no-results" in data["mcpServers"]["git"]["args"]
    data["mcpServers"]["new"] = {"command": "added-later"}
    cfg.write_text(json.dumps(data))
    status, msg = mi.uninstall("custom", path=cfg)
    assert status == "unwrapped" and "backup is kept" in msg
    after = json.loads(cfg.read_text())
    assert after["mcpServers"]["new"] == {"command": "added-later"}
    assert after["mcpServers"]["fs"] == CONFIG["mcpServers"]["fs"]
    assert after["mcpServers"]["git"] == CONFIG["mcpServers"]["git"]


def test_dry_run_writes_nothing(tmp_path, launcher):
    cfg = tmp_path / "mcp.json"
    original = _write(cfg, CONFIG)
    status, msg = mi.install("custom", path=cfg, dry_run=True)
    assert status == "dry-run" and '"mcp"' in msg
    assert cfg.read_bytes() == original
    assert not cfg.with_name(cfg.name + mi.BACKUP_SUFFIX).exists()


def test_refusals(tmp_path, launcher):
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{ not json")
    with pytest.raises(mi.InstallError):
        mi.install("custom", path=cfg)
    assert cfg.read_text() == "{ not json"
    cfg.write_text("[]")
    with pytest.raises(mi.InstallError):
        mi.install("custom", path=cfg)
    with pytest.raises(mi.InstallError):
        mi.install("custom")
    with pytest.raises(mi.InstallError):
        mi.install("vim", path=cfg)
    with pytest.raises(mi.InstallError):
        mi.install("custom", path=cfg, level="L9")
    assert mi.install("custom", path=tmp_path / "missing.json")[0] == "absent"
    assert mi.uninstall("custom", path=tmp_path / "missing.json")[0] == "absent"


def test_second_install_keeps_the_first_backup(tmp_path, launcher):
    cfg = tmp_path / "mcp.json"
    original = _write(cfg, {"mcpServers": {"a": {"command": "x"}}})
    mi.install("custom", path=cfg)
    data = json.loads(cfg.read_text())
    data["mcpServers"]["b"] = {"command": "y"}
    cfg.write_text(json.dumps(data))
    assert mi.install("custom", path=cfg)[0] == "ok"
    assert cfg.with_name(cfg.name + mi.BACKUP_SUFFIX).read_bytes() == original
    assert set(mi._load_records()[str(cfg)]["servers"]) == {"a", "b"}


def test_opencode_command_arrays(tmp_path, launcher):
    cfg = tmp_path / "opencode.json"
    doc = {
        "mcp": {
            "fs": {"type": "local", "command": ["npx", "-y", "pkg"]},
            "r": {"type": "remote", "url": "https://x"},
        }
    }
    original = _write(cfg, doc)
    assert mi.install("opencode", path=cfg)[0] == "ok"
    fs = json.loads(cfg.read_text())["mcp"]["fs"]
    assert fs["command"][:3] == ["/opt/bin/distil", "mcp", "wrap"] and fs["command"][-3:] == [
        "npx",
        "-y",
        "pkg",
    ]
    assert mi.install("opencode", path=cfg)[0] == "exists"
    cfg.write_text(cfg.read_text() + " ")  # edited since: surgical undo
    assert mi.uninstall("opencode", path=cfg)[0] == "unwrapped"
    assert json.loads(cfg.read_text()) == json.loads(original)


CODEX = """# my codex config
model = "gpt-5.2"

[mcp_servers.fs]
command = "npx"
args = [
  "-y",   # comment inside the array
  "@modelcontextprotocol/server-filesystem",
]
startup_timeout_sec = 20

[mcp_servers.fs.env]
TOKEN = "abc"

[mcp_servers."git"]
command = 'uvx'
args = ["mcp-server-git"]

[profiles.x]
model = "o3"
"""


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")
def test_codex_toml_patch_round_trips_and_undoes(tmp_path, launcher):
    import tomllib

    cfg = tmp_path / "config.toml"
    cfg.write_text(CODEX)
    assert mi.install("codex", path=cfg, level="L1")[0] == "ok"
    text = cfg.read_text()
    assert "# my codex config" in text and "startup_timeout_sec = 20" in text
    doc = tomllib.loads(text)
    fs = doc["mcp_servers"]["fs"]
    assert fs["command"] == "/opt/bin/distil"
    assert fs["args"][-3:] == ["npx", "-y", "@modelcontextprotocol/server-filesystem"]
    assert fs["env"] == {"TOKEN": "abc"} and doc["profiles"] == {"x": {"model": "o3"}}
    assert doc["mcp_servers"]["git"]["args"][-2:] == ["uvx", "mcp-server-git"]
    assert mi.install("codex", path=cfg)[0] == "exists"
    assert mi.uninstall("codex", path=cfg)[0] == "restored"
    assert cfg.read_text() == CODEX
    # surgical path for TOML too
    mi.install("codex", path=cfg)
    cfg.write_text(cfg.read_text() + "\n[extra]\nk = 1\n")
    assert mi.uninstall("codex", path=cfg)[0] == "unwrapped"
    after = tomllib.loads(cfg.read_text())
    assert after["mcp_servers"]["fs"]["command"] == "npx" and after["extra"] == {"k": 1}


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")
def test_codex_refuses_what_it_cannot_patch_safely(tmp_path, launcher):
    cfg = tmp_path / "config.toml"
    inline = 'mcp_servers = { fs = { command = "npx", args = ["x"] } }\n'
    cfg.write_text(inline)
    with pytest.raises(mi.InstallError, match="plain table header"):
        mi.install("codex", path=cfg)
    assert cfg.read_text() == inline
    cfg.write_text("not = [valid")
    with pytest.raises(mi.InstallError):
        mi.install("codex", path=cfg)
    cfg.write_text('model = "x"\n')
    assert mi.install("codex", path=cfg)[0] == "exists"


def test_codex_needs_tomllib(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "tomllib", None)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[mcp_servers.a]\ncommand = 'x'\n")
    with pytest.raises(mi.InstallError, match="3.11"):
        mi.install("codex", path=cfg)


def test_default_client_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(mi.Path, "home", lambda: tmp_path)
    assert mi.CLIENTS["cursor"].path() == tmp_path / ".cursor" / "mcp.json"
    assert mi.CLIENTS["codex"].path() == tmp_path / ".codex" / "config.toml"
    assert mi.CLIENTS["claude-desktop"].path().name == "claude_desktop_config.json"
    for plat in ("darwin", "linux"):
        monkeypatch.setattr(mi.sys, "platform", plat)
        assert mi._claude_desktop_path().name == "claude_desktop_config.json"


def test_wrap_and_unwrap_argv_are_inverse():
    argv = mi.wrap_argv("fs", ["npx", "-y", "pkg"], "L0", True)
    assert mi._is_wrapped(argv) and mi.unwrap_argv(argv) == ["npx", "-y", "pkg"]
    assert not mi._is_wrapped(["-y", "pkg"])


def test_cli_install_and_undo(tmp_path, launcher, capsys):
    cfg = tmp_path / "mcp.json"
    original = _write(cfg, CONFIG)
    ns = argparse.Namespace(
        mcp_cmd="install",
        client="custom",
        path=str(cfg),
        undo=False,
        dry_run=False,
        level="L0",
        no_results=False,
    )
    assert cli.cmd_mcp(ns) == 0
    assert "restart the client" in capsys.readouterr().out
    ns.undo = True
    assert cli.cmd_mcp(ns) == 0
    assert cfg.read_bytes() == original
    cfg.write_text("{")
    ns.undo = False
    assert cli.cmd_mcp(ns) == 1
