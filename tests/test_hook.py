"""The PostToolUse hook: lossless, fail-safe, and shaped exactly like the real thing.

A schema near-miss on this path fails *silently* — Claude Code ignores the
replacement and uses the original, with no error anywhere. So these tests pin the
wire shape as tightly as the behaviour: a green suite that doesn't check the envelope
would pass while the feature quietly does nothing in production.
"""

from __future__ import annotations

import json

import pytest

from distil.hook import compress_tool_output, run

BIG = "ERROR connection refused\n" * 400
# Exactly what live Claude Code sends, including `noOutputExpected`, which is absent
# from the published four-field example. Verified against a real session 2026-08-16.
LIVE_BASH = {
    "stdout": BIG,
    "stderr": "",
    "interrupted": False,
    "isImage": False,
    "noOutputExpected": False,
}


def _bash(**over):
    return {**LIVE_BASH, **over}


class TestLossless:
    def test_shrinks_repeated_lines(self):
        out = compress_tool_output("Bash", _bash())
        assert out is not None
        assert len(out["stdout"]) < len(BIG)

    def test_preserves_unknown_keys(self):
        """Unknown keys must survive: rebuilding from the documented four fields
        would produce a shape live Claude Code rejects — silently."""
        out = compress_tool_output("Bash", _bash())
        assert set(out) == set(LIVE_BASH)
        assert out["noOutputExpected"] is False

    def test_json_minification_is_reversible_in_meaning(self):
        payload = json.dumps({"items": [{"n": i, "ok": True} for i in range(300)]}, indent=2)
        out = compress_tool_output("Bash", _bash(stdout=payload))
        assert out is not None
        # Lossless: the compressed form must parse back to the identical object.
        assert json.loads(out["stdout"]) == json.loads(payload)

    def test_never_grows_output(self):
        """Reject-if-bigger: compression must never cost more than it saves."""
        for text in ("x" * 3000, "a b c\n" * 500, json.dumps({"k": "v" * 3000})):
            out = compress_tool_output("Bash", _bash(stdout=text))
            if out is not None:
                assert len(out["stdout"]) <= len(text)


class TestFailSafe:
    @pytest.mark.parametrize(
        "over",
        [
            {"stderr": "boom"},  # error text is load-bearing
            {"interrupted": True},  # partial output
            {"isImage": True},  # not text
            {"stdout": "tiny"},  # below the size floor
        ],
    )
    def test_refuses_risky_shapes(self, over):
        assert compress_tool_output("Bash", _bash(**over)) is None

    @pytest.mark.parametrize("tool", ["Read", "Grep", "Glob", "Write", "Edit", "WebFetch"])
    def test_refuses_undocumented_tools(self, tool):
        """Shapes we haven't verified live are declined, not guessed at."""
        assert compress_tool_output(tool, {"content": BIG}) is None

    @pytest.mark.parametrize(
        "payload",
        ["not json", "{}", '{"tool_name": "Bash"}', '{"tool_response": {}}', "", "null", "[]"],
    )
    def test_malformed_input_is_a_no_op(self, payload):
        assert run(payload) == "{}"

    def test_exception_never_escapes(self):
        # A non-dict where a dict is expected must degrade, not raise.
        assert run(json.dumps({"tool_name": "Bash", "tool_response": "a string"})) == "{}"


class TestWireShape:
    """The envelope is load-bearing: a wrong field name is a silent no-op forever."""

    def test_exact_envelope(self):
        emitted = json.loads(
            json.dumps(
                json.loads(run(json.dumps({"tool_name": "Bash", "tool_response": LIVE_BASH})))
            )
        )
        assert list(emitted) == ["hookSpecificOutput"]
        hso = emitted["hookSpecificOutput"]
        assert hso["hookEventName"] == "PostToolUse"
        assert "updatedToolOutput" in hso
        # `toolResult` is the name a plausible-but-wrong reading produces; it appears
        # nowhere in the official docs and would be ignored at runtime.
        assert "toolResult" not in hso

    def test_replacement_matches_input_schema(self):
        hso = json.loads(run(json.dumps({"tool_name": "Bash", "tool_response": LIVE_BASH})))[
            "hookSpecificOutput"
        ]
        assert set(hso["updatedToolOutput"]) == set(LIVE_BASH)

    def test_stdout_is_parseable_json_always(self):
        """Claude Code parses stdout; emitting anything else corrupts the turn."""
        for payload in ("garbage", json.dumps({"tool_name": "Bash", "tool_response": LIVE_BASH})):
            json.loads(run(payload))


class TestMCP:
    def test_compresses_text_blocks(self):
        out = compress_tool_output("mcp__srv__tool", {"content": [{"type": "text", "text": BIG}]})
        assert out is not None
        assert len(out["content"][0]["text"]) < len(BIG)

    def test_refuses_error_results(self):
        assert (
            compress_tool_output(
                "mcp__srv__tool", {"isError": True, "content": [{"type": "text", "text": BIG}]}
            )
            is None
        )

    def test_passes_non_text_blocks_through(self):
        img = {"type": "image", "data": "z" * 5000}
        out = compress_tool_output(
            "mcp__srv__tool", {"content": [{"type": "text", "text": BIG}, img]}
        )
        assert out is not None
        assert out["content"][1] == img


class TestHostileInputs:
    """The adversarial-real-path invariants, applied to the hook."""

    # `ids=` matters here, not just for readability: without it pytest builds the test
    # ID from the parameter VALUE, and these values are thousands of characters long.
    # pytest exports the current ID in PYTEST_CURRENT_TEST, and Windows caps an
    # environment variable at 32,767 characters — so the unnamed version failed the
    # windows-latest gate with "the environment variable is longer than 32767
    # characters" before a single assertion ran.
    @pytest.mark.parametrize(
        "text",
        [
            "\x00\x01\x02" * 1000,
            "😀🎉" * 2000,
            "日本語のログ\n" * 500,
            "a" * 100_000,
            '{"unclosed": [1,2,3' + "x" * 3000,
            "\n" * 5000,
            "\t \r\n" * 2000,
        ],
        ids=[
            "control-bytes",
            "emoji",
            "cjk",
            "100kb-single-run",
            "truncated-json",
            "newline-flood",
            "whitespace-mix",
        ],
    )
    def test_survives_hostile_content(self, text):
        out = compress_tool_output("Bash", _bash(stdout=text))
        if out is not None:
            assert isinstance(out["stdout"], str)
            assert len(out["stdout"]) <= len(text)
            json.dumps(out)  # must stay serialisable


class TestEntryPoints:
    """`main` and `--selftest` are shipped surfaces: a user runs them to verify an
    install, so a break here reads as "distil is broken" even when compression works.
    """

    def test_selftest_passes(self, capsys):
        from distil.hook import main

        assert main(["--selftest"]) == 0
        assert "checks passed" in capsys.readouterr().out

    def test_main_reads_stdin_and_writes_stdout(self, capsys, monkeypatch):
        import io

        from distil.hook import main

        payload = json.dumps({"tool_name": "Bash", "tool_response": LIVE_BASH})
        monkeypatch.setattr("sys.stdin", io.StringIO(payload))
        assert main([]) == 0
        emitted = json.loads(capsys.readouterr().out)
        assert emitted["hookSpecificOutput"]["hookEventName"] == "PostToolUse"

    def test_mcp_plain_string_output(self):
        """Some MCP servers return a bare string rather than content blocks."""
        out = compress_tool_output("mcp__srv__tool", BIG)
        assert isinstance(out, str) and len(out) < len(BIG)

    def test_mcp_small_string_declined(self):
        assert compress_tool_output("mcp__srv__tool", "short") is None

    def test_mcp_non_list_content_declined(self):
        assert compress_tool_output("mcp__srv__tool", {"content": "not-a-list"}) is None


class TestInstaller:
    """Writing settings.json is the step where a mistake costs someone else's config.

    A hand-written hook entry is also where a silent typo costs every byte of savings
    with no error, which is why distil writes it rather than documenting it.
    """

    def _settings(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        return tmp_path / "settings.json"

    def test_creates_settings_when_absent(self, tmp_path, monkeypatch, capsys):
        from distil.hook import install_hook

        p = self._settings(tmp_path, monkeypatch)
        assert install_hook() == 0
        entries = json.loads(p.read_text())["hooks"]["PostToolUse"]
        assert len(entries) == 1
        assert "distil.hook" in entries[0]["hooks"][0]["command"]

    def test_preserves_foreign_hooks_and_other_keys(self, tmp_path, monkeypatch):
        from distil.hook import install_hook

        p = self._settings(tmp_path, monkeypatch)
        p.write_text(
            json.dumps(
                {
                    "model": "opus",
                    "hooks": {
                        "PostToolUse": [
                            {
                                "matcher": "Write",
                                "hooks": [{"type": "command", "command": "someone-elses-hook"}],
                            }
                        ]
                    },
                }
            )
        )
        install_hook()
        got = json.loads(p.read_text())
        assert got["model"] == "opus"
        cmds = [e["hooks"][0]["command"] for e in got["hooks"]["PostToolUse"]]
        assert "someone-elses-hook" in cmds
        assert any("distil.hook" in c for c in cmds)

    def test_install_is_idempotent(self, tmp_path, monkeypatch):
        from distil.hook import install_hook

        p = self._settings(tmp_path, monkeypatch)
        install_hook()
        install_hook()
        install_hook()
        entries = json.loads(p.read_text())["hooks"]["PostToolUse"]
        assert sum("distil.hook" in e["hooks"][0]["command"] for e in entries) == 1

    def test_uninstall_removes_only_ours(self, tmp_path, monkeypatch):
        from distil.hook import install_hook, uninstall_hook

        p = self._settings(tmp_path, monkeypatch)
        p.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PostToolUse": [
                            {
                                "matcher": "Write",
                                "hooks": [{"type": "command", "command": "someone-elses-hook"}],
                            }
                        ]
                    }
                }
            )
        )
        install_hook()
        assert uninstall_hook() == 0
        entries = json.loads(p.read_text())["hooks"]["PostToolUse"]
        assert len(entries) == 1
        assert entries[0]["hooks"][0]["command"] == "someone-elses-hook"

    def test_uninstall_when_nothing_installed(self, tmp_path, monkeypatch, capsys):
        from distil.hook import uninstall_hook

        self._settings(tmp_path, monkeypatch)
        assert uninstall_hook() == 0
        assert "nothing to remove" in capsys.readouterr().out

    def test_refuses_to_clobber_unreadable_settings(self, tmp_path, monkeypatch, capsys):
        """A corrupt settings.json is the user's data — report, don't overwrite."""
        from distil.hook import install_hook

        p = self._settings(tmp_path, monkeypatch)
        p.write_text("{ this is not json")
        assert install_hook() == 1
        assert "refusing to overwrite" in capsys.readouterr().out
        assert p.read_text() == "{ this is not json"

    def test_refuses_unexpected_hook_shape(self, tmp_path, monkeypatch):
        from distil.hook import install_hook

        p = self._settings(tmp_path, monkeypatch)
        p.write_text(json.dumps({"hooks": {"PostToolUse": "not-a-list"}}))
        assert install_hook() == 1

    def test_matcher_covers_bash_and_mcp(self, tmp_path, monkeypatch):
        from distil.hook import install_hook

        p = self._settings(tmp_path, monkeypatch)
        install_hook()
        entry = json.loads(p.read_text())["hooks"]["PostToolUse"][0]
        assert "Bash" in entry["matcher"] and "mcp__" in entry["matcher"]


def test_no_test_id_can_break_windows():
    """No parametrized ID may approach Windows' environment-variable ceiling.

    pytest exports the running test's ID in PYTEST_CURRENT_TEST. Windows caps an
    environment variable at 32,767 characters, so a parametrize over large values
    without `ids=` aborts the whole windows-latest gate in setup — before any
    assertion runs, with an error that names no test. That is exactly how
    test_survives_hostile_content broke CI, so this guards the class rather than the
    instance.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only", "--no-header"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    longest = max((line for line in out.stdout.splitlines() if "::" in line), key=len, default="")
    # 8k leaves generous headroom under the 32,767 hard limit while still catching the
    # "someone parametrized over a 4KB blob" mistake.
    assert len(longest) < 8192, (
        f"test ID is {len(longest)} chars — Windows caps env vars at 32,767 and pytest "
        f"puts the ID in PYTEST_CURRENT_TEST. Add ids=[...] to that parametrize.\n"
        f"{longest[:200]}..."
    )
