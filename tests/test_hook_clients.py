"""Post-tool hooks beyond Claude Code: Cursor, Gemini CLI, Codex CLI.

Each client's envelope is pinned exactly as its own docs publish it (see
``distil.hook.CLIENTS`` for the source + date). A near-miss field name is a silent
no-op in production, so the shape is the test, not a detail of it.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from distil import hook
from distil.hook import compress_text, run

#: Distinct lines, so Tier-0 cannot collapse them and the Tier-1 digest is what wins.
VARIED = (
    "\n".join(
        f"2026-09-25 INFO worker-{i % 7} processed batch {i * 7} in {i % 13}ms" for i in range(400)
    )
    + "\n=== 12 passed in 3.2s ==="
)
REPEATED = "ERROR connection refused\n" * 400


def _handle(text: str) -> str:
    import re

    m = re.search(r"`distil expand ([0-9a-f]{8})`", text)
    assert m, text[-200:]
    return m.group(1)


class TestDigest:
    def test_varied_log_is_digested_and_recoverable(self):
        out = compress_text(VARIED)
        assert out is not None and len(out) < len(VARIED) / 3
        assert "12 passed" in out  # the verdict line is kept verbatim
        from distil.mcp_server import load_restore

        assert load_restore(_handle(out)) == VARIED

    def test_no_stub_when_the_original_cannot_be_persisted(self, monkeypatch):
        monkeypatch.setattr(hook, "_persist", lambda h, t: False)
        out = compress_text(VARIED)
        assert out is None or "distil expand" not in out

    def test_json_stays_lossless(self):
        payload = json.dumps([{"n": i, "name": f"row {i}"} for i in range(300)], indent=2)
        out = compress_text(payload)
        assert out is not None and json.loads(out) == json.loads(payload)

    def test_small_or_non_text_declined(self):
        assert compress_text("tiny") is None
        assert compress_text(None) is None  # type: ignore[arg-type]


class TestExactQuote:
    @pytest.mark.parametrize(
        "name,inp",
        [
            ("Read", None),
            ("mcp__fs__read_file", None),
            ("mcp_fs_read_file", None),
            ("MCP:read_file", None),
            ("mcp__distil__distil_expand", None),
            ("Bash", {"command": "cat app.py"}),
            ("run_shell_command", {"command": "cd /r && sed -n '1,80p' app.py"}),
        ],
    )
    def test_exempt(self, name, inp):
        assert hook.is_exact_quote(name, inp)

    @pytest.mark.parametrize(
        "name,inp",
        [("Bash", {"command": "pytest -q"}), ("mcp__db__query", {}), ("Bash", None)],
    )
    def test_not_exempt(self, name, inp):
        assert not hook.is_exact_quote(name, inp)

    def test_claude_bash_cat_is_never_touched(self):
        ev = {
            "tool_name": "Bash",
            "tool_input": {"command": "cat big.log"},
            "tool_response": {"stdout": VARIED, "stderr": "", "interrupted": False},
        }
        assert run(json.dumps(ev)) == "{}"


class TestCursor:
    def _ev(self, name="MCP:query", payload=None):
        payload = payload or {"content": [{"type": "text", "text": VARIED}]}
        return json.dumps(
            {"tool_name": name, "tool_input": "{}", "tool_output": json.dumps(payload)}
        )

    def test_mcp_envelope(self):
        out = json.loads(run(self._ev(), "cursor"))
        assert list(out) == ["updated_mcp_tool_output"]
        text = out["updated_mcp_tool_output"]["content"][0]["text"]
        assert len(text) < len(VARIED)

    @pytest.mark.parametrize("name", ["Shell", "Read", "Write", "Grep", "Delete", "Task"])
    def test_builtins_never_touched(self, name):
        assert run(self._ev(name), "cursor") == "{}"

    def test_error_and_garbage_declined(self):
        err = {"isError": True, "content": [{"type": "text", "text": VARIED}]}
        assert run(self._ev(payload=err), "cursor") == "{}"
        bad = json.dumps({"tool_name": "MCP:q", "tool_output": "{not json"})
        assert run(bad, "cursor") == "{}"
        # A bare-string MCP result has no object to put in the object-typed field.
        s = json.dumps({"tool_name": "MCP:q", "tool_output": json.dumps(VARIED)})
        assert run(s, "cursor") == "{}"

    def test_exact_quote_mcp_read_declined(self):
        assert run(self._ev("MCP:read_file"), "cursor") == "{}"


class TestGemini:
    def _ev(self, name="run_shell_command", **resp):
        resp = {"llmContent": VARIED, "returnDisplay": "x", **resp}
        return json.dumps(
            {"tool_name": name, "tool_input": {"command": "make test"}, "tool_response": resp}
        )

    def test_deny_reason_envelope(self):
        out = json.loads(run(self._ev(), "gemini"))
        assert out["decision"] == "deny"
        assert out["reason"].startswith(hook._BLOCK_NOTE)
        assert len(out["reason"]) < len(VARIED)

    def test_mcp_tool(self):
        assert json.loads(run(self._ev("mcp_db_query"), "gemini"))["decision"] == "deny"

    @pytest.mark.parametrize(
        "ev",
        [
            {"error": {"message": "boom"}},
            {"llmContent": [{"text": "parts are not handled"}]},
            {"llmContent": "tiny"},
        ],
        ids=["error", "parts", "tiny"],
    )
    def test_declines(self, ev):
        assert run(self._ev(**ev), "gemini") == "{}"

    def test_other_tools_declined(self):
        assert run(self._ev("read_file"), "gemini") == "{}"
        assert run(self._ev("web_fetch"), "gemini") == "{}"


class TestCodex:
    def test_bash_string(self):
        ev = {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_response": VARIED}
        out = json.loads(run(json.dumps(ev), "codex"))
        assert out["decision"] == "block" and len(out["reason"]) < len(VARIED)
        assert "updatedMCPToolOutput" not in json.dumps(out)  # documented as unsupported

    def test_mcp_all_text(self):
        resp = {"content": [{"type": "text", "text": VARIED}]}
        ev = {"tool_name": "mcp__db__q", "tool_response": resp}
        assert json.loads(run(json.dumps(ev), "codex"))["decision"] == "block"

    @pytest.mark.parametrize(
        "name,resp",
        [
            ("mcp__db__q", {"isError": True, "content": [{"type": "text", "text": VARIED}]}),
            ("mcp__db__q", {"content": [{"type": "image", "data": "x"}]}),
            ("mcp__db__q", {"content": []}),
            ("mcp__db__q", {"content": [{"type": "text", "text": 3}]}),
            ("Bash", {"stdout": VARIED}),
            ("apply_patch", VARIED),
        ],
        ids=["error", "image", "empty", "non-str", "bash-dict", "apply-patch"],
    )
    def test_declines(self, name, resp):
        assert run(json.dumps({"tool_name": name, "tool_response": resp}), "codex") == "{}"


class TestMain:
    def test_client_flag_and_unknown_client(self, capsys, monkeypatch):
        import io

        ev = json.dumps({"tool_name": "Bash", "tool_response": VARIED})
        monkeypatch.setattr("sys.stdin", io.StringIO(ev))
        assert hook.main(["--client", "codex"]) == 0
        assert json.loads(capsys.readouterr().out)["decision"] == "block"
        monkeypatch.setattr("sys.stdin", io.StringIO(ev))
        assert hook.main(["--client", "nope"]) == 0
        assert capsys.readouterr().out == "{}"
        assert hook.main(["--client"]) == 0
        assert capsys.readouterr().out == "{}"

    def test_non_object_event(self):
        assert run("[]", "gemini") == "{}"

    def test_selftest_writes_no_receipts(self):
        assert hook.main(["--selftest"]) == 0
        assert not hook._receipt_path().exists()


# --------------------------------------------------------------------------- installer

CLIENT_KEYS = ["claude", "cursor", "gemini", "codex"]


def _entries(client: str) -> list:
    data = json.loads(hook.config_path(client).read_text())
    return data["hooks"][hook.CLIENTS[client].event]


class TestInstaller:
    @pytest.mark.parametrize("client", CLIENT_KEYS)
    def test_install_uninstall_round_trip_on_fresh_home(self, client):
        path = hook.config_path(client)
        assert hook.install_hook(client) == 0
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert sum(hook._ours(e) for e in _entries(client)) == 1
        assert hook.hook_status(client)[0]
        assert hook.install_hook(client) == 0  # idempotent
        assert sum(hook._ours(e) for e in _entries(client)) == 1
        assert hook.uninstall_hook(client) == 0
        assert not path.exists(), "distil created the file, so undo removes it"
        assert hook._read_owned() == {}

    @pytest.mark.parametrize("client", CLIENT_KEYS)
    def test_foreign_hooks_and_keys_survive(self, client):
        path = hook.config_path(client)
        path.parent.mkdir(parents=True)
        event = hook.CLIENTS[client].event
        theirs = {"command": "their-hook", "matcher": "Write"}
        original = {"model": "x", "hooks": {event: [theirs], "Other": [{"command": "o"}]}}
        path.write_text(json.dumps(original))
        os.chmod(path, 0o640)
        assert hook.install_hook(client) == 0
        assert stat.S_IMODE(path.stat().st_mode) == 0o640, "mode kept"
        got = json.loads(path.read_text())
        assert got["model"] == "x" and got["hooks"]["Other"] == [{"command": "o"}]
        assert theirs in got["hooks"][event]
        assert hook.uninstall_hook(client) == 0
        assert json.loads(path.read_text()) == original

    def test_empty_event_list_distil_did_not_create_is_kept(self):
        path = hook.config_path("gemini")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"hooks": {"AfterTool": []}}))
        hook.install_hook("gemini")
        hook.uninstall_hook("gemini")
        assert json.loads(path.read_text()) == {"hooks": {"AfterTool": []}}

    def test_event_distil_created_is_removed_but_user_file_kept(self):
        path = hook.config_path("codex")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"description": "mine"}))
        hook.install_hook("codex")
        hook.uninstall_hook("codex")
        assert json.loads(path.read_text()) == {"description": "mine"}

    def test_cursor_new_file_is_versioned(self):
        hook.install_hook("cursor")
        data = json.loads(hook.config_path("cursor").read_text())
        assert data["version"] == 1
        assert data["hooks"]["postToolUse"][0]["matcher"] == "MCP:"
        assert "--client cursor" in data["hooks"]["postToolUse"][0]["command"]

    def test_codex_home_honoured(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "cx"))
        assert hook.install_hook("codex") == 0
        assert (tmp_path / "cx" / "hooks.json").is_file()
        assert "/hooks" in capsys.readouterr().out  # the trust step is spelled out

    def test_symlinked_config_is_written_through(self, tmp_path):
        real = tmp_path / "dotfiles" / "settings.json"
        real.parent.mkdir()
        real.write_text("{}")
        link = hook.config_path("gemini")
        link.parent.mkdir(parents=True)
        link.symlink_to(real)
        hook.install_hook("gemini")
        assert link.is_symlink()
        assert "AfterTool" in json.loads(real.read_text())["hooks"]

    @pytest.mark.parametrize(
        "content", ["{ nope", json.dumps({"hooks": []}), json.dumps({"hooks": {"AfterTool": {}}})]
    )
    def test_refuses_unreadable_or_odd_shapes(self, content, capsys):
        path = hook.config_path("gemini")
        path.parent.mkdir(parents=True)
        path.write_text(content)
        assert hook.install_hook("gemini") == 1
        assert path.read_text() == content

    def test_failed_write_records_no_ownership(self, monkeypatch, capsys):
        from distil import setup as _setup

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(_setup, "_write_settings", boom)
        assert hook.install_hook("cursor") == 1
        assert hook._read_owned() == {}
        assert not hook.config_path("cursor").exists()

    def test_ownership_record_failure_still_installs(self, monkeypatch, capsys):
        monkeypatch.setattr(hook, "_write_owned", lambda d: (_ for _ in ()).throw(OSError("ro")))
        assert hook.install_hook("claude") == 0
        assert "ownership record was not updated" in capsys.readouterr().out

    def test_uninstall_edge_cases(self, capsys):
        assert hook.uninstall_hook("cursor") == 0  # no file
        path = hook.config_path("cursor")
        path.parent.mkdir(parents=True)
        path.write_text("{ nope")
        assert hook.uninstall_hook("cursor") == 1
        path.write_text(json.dumps({"hooks": {"postToolUse": [{"command": "x"}]}}))
        assert hook.uninstall_hook("cursor") == 0
        path.write_text(json.dumps({"version": 1}))
        assert hook.uninstall_hook("cursor") == 0
        assert "no distil hook found" in capsys.readouterr().out

    def test_uninstall_write_failure(self, monkeypatch, capsys):
        from distil import setup as _setup

        path = hook.config_path("gemini")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"x": 1}))
        hook.install_hook("gemini")
        monkeypatch.setattr(_setup, "_write_settings", lambda *a: (_ for _ in ()).throw(OSError()))
        assert hook.uninstall_hook("gemini") == 1

    def test_uninstall_without_ownership_record_keeps_containers(self):
        hook.install_hook("codex")
        hook._owned_path().unlink()
        hook.uninstall_hook("codex")
        assert json.loads(hook.config_path("codex").read_text()) == {"hooks": {"PostToolUse": []}}

    def test_detected_and_status(self, capsys):
        assert hook.detected_clients() == ["claude"]
        (Path.home() / ".gemini").mkdir()
        assert hook.detected_clients() == ["claude", "gemini"]
        hook.install_hook("gemini")
        assert hook.print_status() == 0
        out = capsys.readouterr().out
        assert "✓ installed" in out and "Windsurf" in out and "unsupported" in out

    def test_config_path_unknown(self):
        with pytest.raises(KeyError):
            hook.config_path("windsurf")


class TestCli:
    def test_setup_hooks_installs_detected_clients(self, capsys):
        from distil.cli import main

        (Path.home() / ".cursor").mkdir()
        assert main(["setup", "--hooks"]) == 0
        assert hook.hook_status("claude")[0] and hook.hook_status("cursor")[0]
        assert not hook.hook_status("gemini")[0]

    def test_hook_install_all_status_uninstall(self, capsys):
        from distil.cli import main

        assert main(["hook", "install", "--client", "all"]) == 0
        assert all(hook.hook_status(k)[0] for k in CLIENT_KEYS)
        assert main(["hook", "status"]) == 0
        assert main(["hook", "uninstall", "--client", "all"]) == 0
        assert not any(hook.hook_status(k)[0] for k in CLIENT_KEYS)
        assert main(["hook", "--install"]) == 0
        assert main(["hook", "--uninstall"]) == 0

    def test_expand_command(self, capsys):
        from distil.cli import main

        h = _handle(compress_text(VARIED) or "")
        capsys.readouterr()
        assert main(["expand", h]) == 0
        assert capsys.readouterr().out == VARIED
        assert main(["expand", "zzzz"]) == 1

    def test_front_door_stays_four(self):
        from distil.cli import FRONT_DOOR, build_parser, front_help

        assert FRONT_DOOR == ("setup", "wrap", "savings", "doctor")
        short = front_help(build_parser())
        assert "hook" not in short.split("More:")[0] and "expand" not in short


def test_end_to_end_big_output_through_the_hook_process(tmp_path):
    """A real subprocess, exactly as a client runs it: big output in, smaller out,
    and the handle in the stub recovers the original byte-exact via `distil expand`."""
    env = {**os.environ, "HOME": str(tmp_path), "DISTIL_HOME": str(tmp_path / "dh")}
    big = VARIED + "\n" + "\n".join(f"row {i}: value={i * i}" for i in range(2000))
    ev = {"tool_name": "run_shell_command", "tool_input": {"command": "make"}}
    ev["tool_response"] = {"llmContent": big}
    proc = subprocess.run(
        [sys.executable, "-m", "distil.hook", "--client", "gemini"],
        input=json.dumps(ev),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    reason = json.loads(proc.stdout)["reason"]
    assert len(reason) < len(big) / 5
    h = _handle(reason)
    back = subprocess.run(
        [
            sys.executable,
            "-c",
            "from distil.cli import main; raise SystemExit(main())",
            "expand",
            h,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert back.stdout == big
