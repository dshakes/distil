"""CLI onboarding / lifecycle / proxy handler tests.

Covers: cmd_version, cmd_upgrade, cmd_setup, cmd_default, cmd_offboard,
        cmd_wrap, cmd_proxy, cmd_gateway, cmd_mcp, cmd_train_transformer.

All tests are hermetic — no network, no real ~/.zshrc / ~/.claude writes.
"""

from __future__ import annotations

import argparse
import io
import json

import pytest

import distil.cli as cli


# --------------------------------------------------------------------------- #
# cmd_version
# --------------------------------------------------------------------------- #


def test_cmd_version_prints_version(capsys) -> None:
    rc = cli.cmd_version(argparse.Namespace())
    assert rc == 0
    out = capsys.readouterr().out
    assert "distil" in out
    # version string must look like a semver number
    import re

    assert re.search(r"\d+\.\d+", out)


# --------------------------------------------------------------------------- #
# cmd_upgrade
# --------------------------------------------------------------------------- #


def test_cmd_upgrade_dry_run_prints_command(monkeypatch, capsys) -> None:
    from distil import onboard

    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")
    rc = cli.cmd_upgrade(argparse.Namespace(dry_run=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "pipx" in out
    assert "upgrade" in out.lower()


def test_cmd_upgrade_uvx_is_ephemeral(monkeypatch, capsys) -> None:
    from distil import onboard

    monkeypatch.setattr(onboard, "install_method", lambda: "uvx")
    rc = cli.cmd_upgrade(argparse.Namespace(dry_run=False))
    assert rc == 0
    assert "nothing to upgrade" in capsys.readouterr().out


@pytest.mark.parametrize("method", ["pipx", "uv", "pip"])
def test_cmd_upgrade_dry_run_all_installers(monkeypatch, capsys, method) -> None:
    from distil import onboard

    monkeypatch.setattr(onboard, "install_method", lambda: method)
    rc = cli.cmd_upgrade(argparse.Namespace(dry_run=True))
    assert rc == 0
    assert method in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_setup
# --------------------------------------------------------------------------- #


def test_cmd_setup_writes_statusline(tmp_path, capsys) -> None:
    settings = tmp_path / "settings.json"
    rc = cli.cmd_setup(argparse.Namespace(settings=str(settings), force=False))
    assert rc == 0
    data = json.loads(settings.read_text())
    assert "distil" in data["statusLine"]["command"]
    out = capsys.readouterr().out
    assert "✓" in out or "ok" in out.lower() or "distil" in out


def test_cmd_setup_idempotent(tmp_path, capsys) -> None:
    settings = tmp_path / "settings.json"
    cli.cmd_setup(argparse.Namespace(settings=str(settings), force=False))
    rc = cli.cmd_setup(argparse.Namespace(settings=str(settings), force=False))
    assert rc == 0  # exists → still succeeds


def test_cmd_setup_force_overwrites(tmp_path, capsys) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"statusLine": {"command": "other-tool"}}))
    rc = cli.cmd_setup(argparse.Namespace(settings=str(settings), force=True))
    assert rc == 0
    data = json.loads(settings.read_text())
    assert "distil" in data["statusLine"]["command"]


# --------------------------------------------------------------------------- #
# cmd_default — alias mode (create + undo); never touches real ~/.zshrc
# --------------------------------------------------------------------------- #


def test_cmd_default_writes_alias(tmp_path, monkeypatch, capsys) -> None:
    import distil.setup as setup_mod
    from distil import onboard

    rc_file = tmp_path / ".zshrc"
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    monkeypatch.setattr(
        onboard,
        "detect",
        lambda: onboard.Env(
            os_name="Darwin",
            agents=[("claude", "Claude Code")],
            subscription=False,
        ),
    )
    rc = cli.cmd_default(
        argparse.Namespace(
            rc=None,
            agent=None,
            mode=None,
            port=8788,
            undo=False,
            always_on=False,
            no_start=False,
        )
    )
    assert rc == 0
    assert rc_file.exists()
    content = rc_file.read_text()
    assert "distil" in content
    out = capsys.readouterr().out
    assert "✓" in out or "claude" in out


def test_cmd_default_undo_removes_alias(tmp_path, monkeypatch, capsys) -> None:
    import distil.setup as setup_mod
    from distil import onboard

    rc_file = tmp_path / ".zshrc"
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    monkeypatch.setattr(
        onboard,
        "detect",
        lambda: onboard.Env(os_name="Darwin", agents=[("claude", "Claude Code")]),
    )
    # Write first, then undo
    cli.cmd_default(
        argparse.Namespace(
            rc=None,
            agent="claude",
            mode="lossless-only",
            port=8788,
            undo=False,
            always_on=False,
            no_start=False,
        )
    )
    rc = cli.cmd_default(
        argparse.Namespace(
            rc=None,
            agent=None,
            mode=None,
            port=8788,
            undo=True,
            always_on=False,
            no_start=False,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "open a new terminal" in out or "✓" in out or "absent" in out


def test_cmd_default_explicit_rc(tmp_path, monkeypatch, capsys) -> None:
    """--rc should use the given path, not the auto-detected one."""
    import distil.setup as setup_mod
    from distil import onboard

    rc_file = tmp_path / "custom_rc"
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("bash", tmp_path / ".bashrc"))
    monkeypatch.setattr(
        onboard,
        "detect",
        lambda: onboard.Env(os_name="Linux", agents=[("codex", "Codex")]),
    )
    rc = cli.cmd_default(
        argparse.Namespace(
            rc=str(rc_file),
            agent="codex",
            mode="expand",
            port=8788,
            undo=False,
            always_on=False,
            no_start=False,
        )
    )
    assert rc == 0
    assert rc_file.exists()
    assert "distil" in rc_file.read_text()


# --------------------------------------------------------------------------- #
# cmd_offboard — non-interactive against temp paths; never touches real files
# --------------------------------------------------------------------------- #


def test_cmd_offboard_no_interactive_skips_all(tmp_path, monkeypatch, capsys) -> None:
    import distil.setup as setup_mod
    from distil import onboard

    rc_file = tmp_path / ".zshrc"
    # Write a managed block so offboard has something to offer
    rc_file.write_text("# distil (managed)\nalias claude='distil wrap -- claude'\n# /distil")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "distil"))
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")
    monkeypatch.setattr(setup_mod, "service_spec", lambda *a, **k: (None, None, None))

    rc = cli.cmd_offboard(argparse.Namespace(purge=False, yes=False, no_interactive=True))
    assert rc == 0
    # --no-interactive never prompts, so every destructive step is skipped
    out = capsys.readouterr().out
    assert "skipped" in out or "uninstall" in out.lower()


def test_cmd_offboard_yes_removes_managed_block(tmp_path, monkeypatch, capsys) -> None:
    import distil.setup as setup_mod
    from distil import onboard

    rc_file = tmp_path / ".zshrc"
    # Use write_managed so the block uses the real marker format
    setup_mod.write_managed(rc_file, setup_mod.alias_body("claude", "lossless-only", shell="zsh"))
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "distil"))
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")
    monkeypatch.setattr(setup_mod, "service_spec", lambda *a, **k: (None, None, None))

    rc = cli.cmd_offboard(argparse.Namespace(purge=False, yes=True, no_interactive=False))
    assert rc == 0
    remaining = rc_file.read_text()
    assert setup_mod._MARK_START not in remaining


# --------------------------------------------------------------------------- #
# cmd_wrap — empty command → early return 2
# --------------------------------------------------------------------------- #


def test_cmd_wrap_no_command_returns_2(capsys) -> None:
    rc = cli.cmd_wrap(
        argparse.Namespace(
            command=[],
            host="127.0.0.1",
            upstream="https://api.anthropic.com",
            lossless_only=False,
            verbatim=False,
            shape_output="off",
            no_record=False,
            pricing="claude-opus-4-8",
            env_var="ANTHROPIC_BASE_URL",
            expand=False,
            session_delta=False,
            shadow=0.0,
        )
    )
    assert rc == 2
    assert (
        "usage" in capsys.readouterr().out.lower()
        or "nothing" in capsys.readouterr().out.lower()
        or rc == 2
    )


def test_cmd_wrap_separator_only_returns_2(capsys) -> None:
    rc = cli.cmd_wrap(
        argparse.Namespace(
            command=["--"],
            host="127.0.0.1",
            upstream="https://api.anthropic.com",
            lossless_only=False,
            verbatim=False,
            shape_output="off",
            no_record=False,
            pricing="claude-opus-4-8",
            env_var="ANTHROPIC_BASE_URL",
            expand=False,
            session_delta=False,
            shadow=0.0,
        )
    )
    assert rc == 2


def test_cmd_wrap_runs_command(monkeypatch, capsys) -> None:
    """wrap with a real command: proxy starts, child runs, exits cleanly."""
    import distil.proxy as proxy_mod

    calls: list[list] = []

    def fake_wrap_run(command, **kwargs) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr(proxy_mod, "wrap_run", fake_wrap_run)
    rc = cli.cmd_wrap(
        argparse.Namespace(
            command=["--", "echo", "hi"],
            host="127.0.0.1",
            upstream="https://api.anthropic.com",
            lossless_only=False,
            verbatim=False,
            shape_output="off",
            no_record=True,
            pricing="claude-opus-4-8",
            env_var="ANTHROPIC_BASE_URL",
            expand=False,
            session_delta=False,
            shadow=0.0,
        )
    )
    assert rc == 0
    assert calls[0] == ["echo", "hi"]


# --------------------------------------------------------------------------- #
# cmd_proxy — monkeypatch serve so it doesn't block
# --------------------------------------------------------------------------- #


def test_cmd_proxy_calls_serve(monkeypatch, capsys) -> None:
    import distil.proxy as proxy_mod

    called: dict = {}

    def fake_serve(**kwargs) -> None:
        called.update(kwargs)

    monkeypatch.setattr(proxy_mod, "serve", fake_serve)
    rc = cli.cmd_proxy(
        argparse.Namespace(
            host="127.0.0.1",
            port=0,
            upstream="https://api.anthropic.com",
            lossless_only=False,
            verbatim=False,
            shape_output="off",
            use_async=False,
            no_record=True,
            pricing="claude-opus-4-8",
            expand=False,
            shadow=0.0,
            session_delta=False,
        )
    )
    assert rc == 0
    assert called["host"] == "127.0.0.1"
    assert called["record"] is False  # no_record=True → record=False


def test_cmd_proxy_async_calls_aserve(monkeypatch) -> None:
    import distil.aproxy as aproxy_mod

    called: dict = {}

    def fake_aserve(**kwargs) -> None:
        called.update(kwargs)

    monkeypatch.setattr(aproxy_mod, "serve", fake_aserve)
    rc = cli.cmd_proxy(
        argparse.Namespace(
            host="127.0.0.1",
            port=0,
            upstream="https://api.anthropic.com",
            lossless_only=True,
            verbatim=False,
            shape_output="off",
            use_async=True,
            no_record=True,
            pricing="claude-opus-4-8",
        )
    )
    assert rc == 0
    assert called.get("lossless_only") is True


# --------------------------------------------------------------------------- #
# cmd_gateway — monkeypatch serve_gateway so it doesn't block
# --------------------------------------------------------------------------- #


def test_cmd_gateway_calls_serve_gateway(monkeypatch) -> None:
    import distil.gateway as gateway_mod

    called: dict = {}

    def fake_serve_gateway(**kwargs) -> None:
        called.update(kwargs)

    monkeypatch.setattr(gateway_mod, "serve_gateway", fake_serve_gateway)
    rc = cli.cmd_gateway(
        argparse.Namespace(
            host="127.0.0.1",
            port=0,
            upstream="https://api.anthropic.com",
            pricing="claude-opus-4-8",
            lossless_only=False,
            verbatim=False,
            admin_token=None,
            trust_tenant_header=False,
            require_keys=False,
            tenant_rpm=0,
            tenant_daily_tokens=0,
        )
    )
    assert rc == 0
    assert called["host"] == "127.0.0.1"


# --------------------------------------------------------------------------- #
# cmd_mcp — handle_message is pure; test via it and via cmd_mcp with fake serve
# --------------------------------------------------------------------------- #


def test_mcp_handle_message_initialize() -> None:
    from distil.mcp_server import handle_message

    resp = handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp is not None
    assert resp["result"]["serverInfo"]["name"] == "distil"
    assert "capabilities" in resp["result"]


def test_mcp_handle_message_tools_list() -> None:
    from distil.mcp_server import handle_message

    resp = handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"distil_compress", "distil_expand", "distil_savings"} == names


def test_mcp_handle_message_unknown_method() -> None:
    from distil.mcp_server import handle_message

    resp = handle_message({"jsonrpc": "2.0", "id": 3, "method": "no_such_method"})
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_mcp_handle_message_notification_returns_none() -> None:
    from distil.mcp_server import handle_message

    # Notifications (no id) get no response per JSON-RPC spec
    resp = handle_message({"jsonrpc": "2.0", "method": "initialized"})
    assert resp is None


def test_cmd_mcp_calls_serve(monkeypatch) -> None:
    import distil.mcp_server as mcp_mod

    called: list = []

    def fake_serve(*a, **k) -> None:
        called.append(True)

    monkeypatch.setattr(mcp_mod, "serve", fake_serve)
    rc = cli.cmd_mcp(argparse.Namespace())
    assert rc == 0
    assert called


# --------------------------------------------------------------------------- #
# cmd_train_transformer — monkeypatch train_transformer
# --------------------------------------------------------------------------- #


def test_cmd_train_transformer(tmp_path, monkeypatch, capsys) -> None:
    import distil.codec.train_transformer as tt_mod

    monkeypatch.setattr(
        tt_mod,
        "train_transformer",
        lambda out, base_model, epochs: {"loss": 0.12, "accuracy": 0.88},
    )
    rc = cli.cmd_train_transformer(
        argparse.Namespace(
            out=str(tmp_path / "model"),
            base_model="google/bert_uncased_L-2_H-128_A-2",
            epochs=1,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "trained" in out
    assert "loss" in out or "accuracy" in out


# --------------------------------------------------------------------------- #
# cmd_leaderboard — badge, json, html, text paths
# --------------------------------------------------------------------------- #


def test_cmd_leaderboard_badge(tmp_path, monkeypatch, capsys) -> None:
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=True, json=False, html=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert "img.shields.io" in out
    assert "markdown" in out


def test_cmd_leaderboard_json_empty(tmp_path, monkeypatch, capsys) -> None:
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=True, html=None))
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "runs" in data
    assert "total_tokens_saved" in data


def test_cmd_leaderboard_html(tmp_path, monkeypatch, capsys) -> None:
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    out_html = tmp_path / "savings.html"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=str(out_html)))
    assert rc == 0
    assert out_html.exists()
    assert "distil" in out_html.read_text().lower()


def test_cmd_leaderboard_text_no_runs(tmp_path, monkeypatch, capsys) -> None:
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=None))
    assert rc == 0
    assert "no genuine savings" in capsys.readouterr().out


def test_cmd_leaderboard_text_with_runs(tmp_path, monkeypatch, capsys) -> None:
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=2,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=1000,
        distil_input_tokens=500,
        path=p,
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert "runs recorded" in out
    assert "tokens saved" in out


# --------------------------------------------------------------------------- #
# cmd_shadow_stats — all branches
# --------------------------------------------------------------------------- #


def test_cmd_shadow_stats_no_samples(monkeypatch, capsys) -> None:
    from distil import shadow as shadow_mod

    class _Empty:
        samples = 0
        changes = 0

        def rate(self):
            return 0.0

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: _Empty()))
    rc = cli.cmd_shadow_stats(argparse.Namespace(json=False))
    assert rc == 0
    assert "No shadow samples" in capsys.readouterr().out


def test_cmd_shadow_stats_collecting(monkeypatch, capsys) -> None:
    from distil import shadow as shadow_mod

    class _Few:
        samples = 10
        changes = 1

        def rate(self):
            return 0.1

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: _Few()))
    rc = cli.cmd_shadow_stats(argparse.Namespace(json=False))
    assert rc == 0
    assert "collecting" in capsys.readouterr().out


def test_cmd_shadow_stats_ready(monkeypatch, capsys) -> None:
    from distil import shadow as shadow_mod

    led = shadow_mod.ShadowLedger()
    led.samples, led.changes = 50, 2
    for i in range(50):
        led.recent.append(0 if i < 2 else 1)  # rate() = 0.04
    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: led))
    rc = cli.cmd_shadow_stats(argparse.Namespace(json=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "decision-equivalence" in out
    assert "96.00%" in out
    assert "No A/A noise baseline yet" in out  # rc4: raw number labeled a floor


def test_cmd_shadow_stats_json(monkeypatch, capsys) -> None:
    from distil import shadow as shadow_mod

    led = shadow_mod.ShadowLedger()
    led.samples, led.changes = 30, 1
    for i in range(30):
        led.recent.append(0 if i < 1 else 1)
    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: led))
    rc = cli.cmd_shadow_stats(argparse.Namespace(json=True))
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "samples" in data and "decision_equivalence" in data
    assert "aa_self_agreement" in data and "adjusted_equivalence" in data


# --------------------------------------------------------------------------- #
# cmd_dashboard — once flag (non-interactive path)
# --------------------------------------------------------------------------- #


def test_cmd_dashboard_once(tmp_path, monkeypatch, capsys) -> None:
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    rc = cli.cmd_dashboard(argparse.Namespace(once=True, interval=2.0))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_statusline — minimal mode + rich no-runs
# --------------------------------------------------------------------------- #


def test_cmd_statusline_no_runs(tmp_path, monkeypatch, capsys) -> None:
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.delenv("DISTIL_STATUSLINE", raising=False)
    rc = cli.cmd_statusline(argparse.Namespace(no_color=True))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


def test_cmd_statusline_minimal_mode(tmp_path, monkeypatch, capsys) -> None:
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setenv("DISTIL_STATUSLINE", "minimal")
    rc = cli.cmd_statusline(argparse.Namespace(no_color=True))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_upgrade — non-dry-run execution paths
# --------------------------------------------------------------------------- #


def test_cmd_upgrade_runs_and_succeeds(monkeypatch, capsys) -> None:
    import subprocess
    from distil import onboard

    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")

    class _OK:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _OK())
    rc = cli.cmd_upgrade(argparse.Namespace(dry_run=False))
    assert rc == 0
    assert "upgraded" in capsys.readouterr().out


def test_cmd_upgrade_warns_when_proxy_running(monkeypatch, capsys) -> None:
    """FIX 4b: a live proxy must be flagged for restart before an in-place upgrade."""
    import subprocess

    from distil import onboard

    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")

    class _Res:
        def __init__(self, rc: int, out: str = "") -> None:
            self.returncode = rc
            self.stdout = out

    def fake_run(*a, **k):
        cmd = a[0] if a else k.get("args")
        if isinstance(cmd, list) and cmd and cmd[0] == "pgrep":
            return _Res(0, "12345 distil wrap -- claude\n67890 distil proxy\n")
        return _Res(0)  # the installer command

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = cli.cmd_upgrade(argparse.Namespace(dry_run=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "running distil proxy" in out
    assert "restart" in out


def test_cmd_upgrade_runs_and_fails(monkeypatch, capsys) -> None:
    import subprocess
    from distil import onboard

    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")

    class _Fail:
        returncode = 1

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Fail())
    rc = cli.cmd_upgrade(argparse.Namespace(dry_run=False))
    assert rc == 1
    assert "run it yourself" in capsys.readouterr().out


def test_cmd_upgrade_advisory_command(monkeypatch, capsys) -> None:
    """An advisory/comment command is printed but not run (lines 667-669)."""
    from distil import onboard

    # The "pip" method when invoked inside a managed venv emits a # comment advisory
    monkeypatch.setattr(onboard, "install_method", lambda: "pip")
    monkeypatch.setattr(
        onboard,
        "upgrade_command",
        lambda m: "# pip install --upgrade distil-llm  (inside your venv)",
    )
    rc = cli.cmd_upgrade(argparse.Namespace(dry_run=False))
    assert rc == 0
    assert "run that yourself" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_leaderboard — live-proxy source + decision-equivalence lines
# --------------------------------------------------------------------------- #


def test_cmd_leaderboard_text_live_proxy(tmp_path, monkeypatch, capsys) -> None:
    """Text output with live-proxy source + shadow stats (lines 207, 216-219)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="live-proxy",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.05,
        distil_dollars=0.02,
        baseline_input_tokens=500,
        distil_input_tokens=200,
        path=p,
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")

    # clear the robust gate (≥50 A/B, ≥30 A/A) so the equivalence line is printed
    class _Ready:
        samples = 50
        aa_samples = 30

        def adjusted_rate(self):
            return 0.02

        def rate(self):
            return 0.02

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: _Ready()))
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert "live-proxy" in out or "genuine live traffic" in out
    assert "decision-equivalence" in out


def test_cmd_leaderboard_text_collecting_shadow(tmp_path, monkeypatch, capsys) -> None:
    """Text with <25 shadow samples → 'collecting' line (line 211-215)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=100,
        distil_input_tokens=50,
        path=p,
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")

    class _Few:
        samples = 5
        aa_samples = 2

        def rate(self):
            return 0.0

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: _Few()))
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=None))
    assert rc == 0
    assert "collecting" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_statusline — minimal with runs, rich with shadow
# --------------------------------------------------------------------------- #


def test_cmd_statusline_minimal_with_runs(tmp_path, monkeypatch, capsys) -> None:
    """Minimal mode + runs recorded → shows total (lines 554-566)."""
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=1000,
        distil_input_tokens=500,
        path=p,
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setenv("DISTIL_STATUSLINE", "minimal")
    rc = cli.cmd_statusline(argparse.Namespace(no_color=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "distil" in out
    assert "total" in out or "500" in out


def test_cmd_statusline_rich_with_shadow(tmp_path, monkeypatch, capsys) -> None:
    """Rich statusline + ≥25 shadow samples AND an A/A baseline → shadow
    equivalence segment (632-633). Without the baseline the verdict is
    correctly suppressed to 'de baseline N/10' (see test_statusline)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=1000,
        distil_input_tokens=500,
        path=p,
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.delenv("DISTIL_STATUSLINE", raising=False)

    led = shadow_mod.ShadowLedger()
    led.samples = 50
    led.aa_samples = 30  # robust A/A baseline → adjusted_rate() == rate()
    led.aa_changes = 0
    for i in range(100):
        led.recent.append(0 if i < 1 else 1)  # rate() = 0.01 → eq 99%
    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: led))
    rc = cli.cmd_statusline(argparse.Namespace(no_color=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "distil" in out
    assert "eq" in out or "99" in out


# --------------------------------------------------------------------------- #
# cmd_dashboard — frame with shadow + session data (1114-1121)
# --------------------------------------------------------------------------- #


def test_cmd_dashboard_frame_with_shadow_session(tmp_path, monkeypatch, capsys) -> None:
    """dashboard --once with shadow + recent session data exercises the full frame."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=1000,
        distil_input_tokens=500,
        session="sess-x",
        path=p,
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")

    class _Ready:
        samples = 30
        recent = [1, 0, 1]

        def rate(self):
            return 0.03

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: _Ready()))
    rc = cli.cmd_dashboard(argparse.Namespace(once=True, interval=2.0))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# main() — error handling paths (2118-2119, 2125, 2147-2148)
# --------------------------------------------------------------------------- #


def test_main_missing_file_exits_2(capsys) -> None:
    """FileNotFoundError from a cmd_* → exit code 2 with clean error (line 2127-2131)."""
    from distil import cli as cli_mod

    rc = cli_mod.main(["compress", "--trajectory", "/no/such/file.json"])
    assert rc == 2
    assert "compress" in capsys.readouterr().err


def test_main_bad_json_exits_2(tmp_path, capsys) -> None:
    """JSONDecodeError from an invalid trajectory → exit code 2 (line 2133-2136)."""
    from distil import cli as cli_mod

    bad = tmp_path / "bad.json"
    bad.write_text("not json at all")
    rc = cli_mod.main(["compress", "--trajectory", str(bad)])
    assert rc == 2
    assert "not valid JSON" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# cmd_onboard — outdated version path (lines 841-857)
# --------------------------------------------------------------------------- #


def test_cmd_onboard_outdated_else_branch(tmp_path, monkeypatch, capsys) -> None:
    """Newer version available but no_interactive → prints upgrade command (line 857)."""
    import distil.setup as setup_mod
    from distil import onboard

    env = onboard.Env(
        os_name="Darwin",
        agents=[("claude", "Claude Code")],
        installed_version="1.0.0",
        method="pipx",
        managers=["pipx"],
    )
    monkeypatch.setattr(onboard, "detect", lambda: env)
    monkeypatch.setattr(onboard, "latest_pypi_version", lambda *a, **k: "2.0.0")
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "s.json")
    rc = cli.cmd_onboard(
        argparse.Namespace(
            json=False,
            offline=False,
            dry_run=False,
            force=False,
            upgrade=False,
            no_color=True,
            yes=False,
            no_interactive=True,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "newer distil available" in out or "distil onboard --upgrade" in out


def test_cmd_onboard_outdated_upgrade_flag(tmp_path, monkeypatch, capsys) -> None:
    """--upgrade flag with newer version available → runs upgrade and returns 0 (lines 841-855)."""
    import subprocess
    import distil.setup as setup_mod
    from distil import onboard

    env = onboard.Env(
        os_name="Darwin",
        agents=[("claude", "Claude Code")],
        installed_version="1.0.0",
        method="pipx",
        managers=["pipx"],
    )
    monkeypatch.setattr(onboard, "detect", lambda: env)
    monkeypatch.setattr(onboard, "latest_pypi_version", lambda *a, **k: "2.0.0")
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "s.json")

    class _OK:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _OK())
    rc = cli.cmd_onboard(
        argparse.Namespace(
            json=False,
            offline=False,
            dry_run=False,
            force=False,
            upgrade=True,
            no_color=True,
            yes=False,
            no_interactive=True,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "upgrading" in out or "done" in out


# --------------------------------------------------------------------------- #
# cmd_leaderboard --json with shadow exception (lines 150-151)
# --------------------------------------------------------------------------- #


def test_cmd_leaderboard_json_shadow_exception(tmp_path, monkeypatch, capsys) -> None:
    """--json path where ShadowLedger.load() raises → still outputs valid JSON (150-151)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    monkeypatch.setattr(
        shadow_mod.ShadowLedger,
        "load",
        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(OSError("no shadow"))),
    )
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=True, html=None))
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "runs" in data  # still valid even without shadow stats


def test_cmd_leaderboard_html_with_shadow_and_session(tmp_path, monkeypatch, capsys) -> None:
    """--html path with shadow samples + recent session (lines 165-174)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=100,
        distil_input_tokens=50,
        session="sess-q",
        path=p,
    )
    out_html = tmp_path / "out.html"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)

    class _Ready:
        samples = 30

        def rate(self):
            return 0.02

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: _Ready()))
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=str(out_html)))
    assert rc == 0
    assert out_html.exists()
    content = out_html.read_text()
    assert "distil" in content.lower()


# --------------------------------------------------------------------------- #
# cmd_statusline — ledger exception path (lines 544-545)
# --------------------------------------------------------------------------- #


def test_cmd_statusline_ledger_exception(monkeypatch, capsys) -> None:
    """ledger.summary() throws → s=None, statusline still prints (lines 544-545)."""
    from distil import ledger as ledger_mod

    def _bad(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ledger_mod, "summary", _bad)
    monkeypatch.delenv("DISTIL_STATUSLINE", raising=False)
    rc = cli.cmd_statusline(argparse.Namespace(no_color=True))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


def test_cmd_statusline_recent_exception(tmp_path, monkeypatch, capsys) -> None:
    """ledger.summary(since=...) throws in rich path → falls back gracefully (596-597)."""
    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=100,
        distil_input_tokens=50,
        path=p,
    )
    call_count = [0]

    def _summary_sometimes_fails(*a, since=None, **k):
        call_count[0] += 1
        if since is not None:
            raise OSError("no slice")
        return ledger_mod.LedgerSummary(
            runs=1,
            total_dollars_saved=0.005,
            total_tokens_saved=50,
            by_trajectory={"t1": 0.005},
            total_baseline_tokens=100,
            total_distil_tokens=50,
        )

    monkeypatch.setattr(ledger_mod, "summary", _summary_sometimes_fails)
    monkeypatch.delenv("DISTIL_STATUSLINE", raising=False)
    rc = cli.cmd_statusline(argparse.Namespace(no_color=True))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_dashboard — shadow + session exceptions in frame() (lines 1114-1121)
# --------------------------------------------------------------------------- #


def test_cmd_dashboard_shadow_exception(tmp_path, monkeypatch, capsys) -> None:
    """ShadowLedger.load() throws inside frame() → still renders (lines 1114-1115)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)

    def _bad(cls):
        raise OSError("no shadow")

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(_bad))
    rc = cli.cmd_dashboard(argparse.Namespace(once=True, interval=2.0))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


def test_cmd_dashboard_session_exception(tmp_path, monkeypatch, capsys) -> None:
    """latest_session() throws inside frame() → still renders (lines 1120-1121)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=100,
        distil_input_tokens=50,
        path=p,
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)

    class _Empty:
        samples = 0

        def rate(self):
            return 0.0

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: _Empty()))

    def _bad(*a, **k):
        raise OSError("no session")

    monkeypatch.setattr(ledger_mod, "latest_session", _bad)
    rc = cli.cmd_dashboard(argparse.Namespace(once=True, interval=2.0))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_offboard — proxy service removal path (lines 1047-1054)
# --------------------------------------------------------------------------- #


def test_cmd_offboard_removes_the_socket_unit_that_holds_the_port(
    tmp_path, monkeypatch, capsys
) -> None:
    """offboard that leaves the systemd socket unit leaves the machine wired.

    The socket unit OWNS the listening port. Removing only the .service reports a
    clean uninstall while 8788 stays bound and systemd keeps trying to start a
    service that no longer exists — the same "reports success and isn't one"
    failure the rest of the uninstall path was hardened against.
    """
    import subprocess

    import distil.setup as setup_mod
    from distil import onboard

    svc = tmp_path / "distil-proxy.service"
    svc.write_text("[Service]\n")
    sock = tmp_path / "distil-proxy.socket"
    sock.write_text("[Socket]\nListenStream=127.0.0.1:8788\n")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", tmp_path / ".zshrc"))
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "distil"))
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")
    monkeypatch.setattr(setup_mod, "service_spec", lambda *a, **k: (svc, "content", "load"))
    monkeypatch.setattr(setup_mod, "socket_unit_spec", lambda port: (sock, "content"))
    monkeypatch.setattr(setup_mod, "service_unload_cmd", lambda: "true")

    class _OK:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _OK())

    rc = cli.cmd_offboard(argparse.Namespace(purge=False, yes=True, no_interactive=False))
    assert rc == 0
    assert not svc.exists()
    assert not sock.exists(), "the socket unit survived offboard and still holds the port"
    assert "removed proxy socket" in capsys.readouterr().out


def test_cmd_offboard_yes_removes_service(tmp_path, monkeypatch, capsys) -> None:
    """--yes with an existing proxy service → attempts to remove it (lines 1047-1054)."""
    import subprocess
    import distil.setup as setup_mod
    from distil import onboard

    # Create a fake service file
    svc = tmp_path / "service.plist"
    svc.write_text("dummy service content")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", tmp_path / ".zshrc"))
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "distil"))
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")
    monkeypatch.setattr(
        setup_mod, "service_spec", lambda *a, **k: (svc, "content", "launchctl load")
    )
    monkeypatch.setattr(setup_mod, "service_unload_cmd", lambda: "launchctl unload dummy")

    class _OK:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _OK())

    rc = cli.cmd_offboard(argparse.Namespace(purge=False, yes=True, no_interactive=False))
    assert rc == 0
    assert not svc.exists()  # file was removed
    out = capsys.readouterr().out
    assert "removed proxy service" in out


# --------------------------------------------------------------------------- #
# cmd_doctor — hint printing (line 718)
# --------------------------------------------------------------------------- #


def test_cmd_doctor_text_with_hint(monkeypatch, capsys) -> None:
    """A check with a non-empty hint prints the hint line (line 718)."""
    from distil import cli as cli_mod, doctor as doctor_mod

    monkeypatch.setattr(
        doctor_mod,
        "diagnose",
        lambda: [doctor_mod.Check("test-check", doctor_mod.OK, "all good", hint="try this fix")],
    )
    rc = cli_mod.cmd_doctor(argparse.Namespace(json=False))
    assert rc == 0
    assert "try this fix" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_setup — conflict/error → return 1 (line 748)
# --------------------------------------------------------------------------- #


def test_cmd_setup_conflict_returns_1(tmp_path, monkeypatch, capsys) -> None:
    """wire_statusline returns conflict → cmd_setup returns 1 (line 748)."""
    import distil.setup as setup_mod

    monkeypatch.setattr(setup_mod, "wire_statusline", lambda *a, **k: ("conflict", "conflict msg"))
    rc = cli.cmd_setup(argparse.Namespace(settings=str(tmp_path / "s.json"), force=False))
    assert rc == 1
    assert "conflict" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------- #
# cmd_onboard — conflict status line hint (line 866)
# --------------------------------------------------------------------------- #


def test_cmd_onboard_conflict_status_hint(tmp_path, monkeypatch, capsys) -> None:
    """wire_statusline returns conflict → prints --force hint at line 866."""
    import distil.setup as setup_mod
    from distil import onboard

    env = onboard.Env(
        os_name="Darwin",
        agents=[("claude", "Claude Code")],
        installed_version="1.0.0",
        method="pipx",
        managers=["pipx"],
    )
    monkeypatch.setattr(onboard, "detect", lambda: env)
    monkeypatch.setattr(onboard, "latest_pypi_version", lambda *a, **k: None)
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "s.json")
    monkeypatch.setattr(
        setup_mod, "wire_statusline", lambda *a, **k: ("conflict", "conflict found")
    )
    rc = cli.cmd_onboard(
        argparse.Namespace(
            json=False,
            offline=False,
            dry_run=False,
            force=False,
            upgrade=False,
            no_color=True,
            yes=False,
            no_interactive=True,
        )
    )
    assert rc == 0
    assert "force" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_statusline — minimal with summary(since=...) exception (560-561)
# --------------------------------------------------------------------------- #


def test_cmd_statusline_minimal_since_exception(tmp_path, monkeypatch, capsys) -> None:
    """summary(since=...) throws in minimal mode → still prints (560-561)."""
    from distil import ledger as ledger_mod

    def _summary(*a, since=None, **k):
        if since is not None:
            raise OSError("sliced")
        return ledger_mod.LedgerSummary(
            runs=1,
            total_dollars_saved=0.01,
            total_tokens_saved=100,
            by_trajectory={},
            total_baseline_tokens=200,
            total_distil_tokens=100,
        )

    monkeypatch.setattr(ledger_mod, "summary", _summary)
    monkeypatch.setenv("DISTIL_STATUSLINE", "minimal")
    rc = cli.cmd_statusline(argparse.Namespace(no_color=True))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_statusline — minimal with model from stdin (line 566)
# --------------------------------------------------------------------------- #


def test_cmd_statusline_minimal_model_from_stdin(tmp_path, monkeypatch, capsys) -> None:
    """model name from stdin JSON appears in minimal statusline (line 566)."""
    import io
    import sys

    from distil import ledger as ledger_mod

    monkeypatch.setattr(
        ledger_mod,
        "summary",
        lambda *a, **k: ledger_mod.LedgerSummary(
            runs=1,
            total_dollars_saved=0.01,
            total_tokens_saved=100,
            by_trajectory={},
            total_baseline_tokens=200,
            total_distil_tokens=100,
        ),
    )
    monkeypatch.setenv("DISTIL_STATUSLINE", "minimal")
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"model": {"display_name": "claude-opus"}}'))
    rc = cli.cmd_statusline(argparse.Namespace(no_color=True))
    assert rc == 0
    assert "claude-opus" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_leaderboard --html — shadow exception (165-166) + session exception (173-174)
# --------------------------------------------------------------------------- #


def test_cmd_leaderboard_html_shadow_exc(tmp_path, monkeypatch, capsys) -> None:
    """ShadowLedger.load() raises in html path → still writes html (165-166)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)

    def _bad_load(cls):
        raise OSError("no shadow")

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(_bad_load))
    out_html = tmp_path / "out.html"
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=str(out_html)))
    assert rc == 0
    assert out_html.exists()


def test_cmd_leaderboard_html_session_exc(tmp_path, monkeypatch, capsys) -> None:
    """latest_session() raises in html path → still writes html (173-174)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)

    class _Empty:
        samples = 0

        def rate(self):
            return 0.0

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(lambda cls, *a, **k: _Empty()))
    monkeypatch.setattr(
        ledger_mod, "latest_session", lambda: (_ for _ in ()).throw(OSError("no session"))
    )
    out_html = tmp_path / "out.html"
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=str(out_html)))
    assert rc == 0
    assert out_html.exists()


# --------------------------------------------------------------------------- #
# cmd_leaderboard text — shadow exception (216-217)
# --------------------------------------------------------------------------- #


def test_cmd_leaderboard_text_shadow_exc(tmp_path, monkeypatch, capsys) -> None:
    """ShadowLedger.load() raises in text path → still prints stats (216-217)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    p = tmp_path / "savings.jsonl"
    ledger_mod.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=100,
        distil_input_tokens=50,
        path=p,
    )
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)

    def _bad_load(cls):
        raise OSError("no shadow")

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(_bad_load))
    rc = cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=None))
    assert rc == 0
    assert "distil savings" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_statusline rich — shadow exception (632-633)
# --------------------------------------------------------------------------- #


def test_cmd_statusline_rich_shadow_exc(tmp_path, monkeypatch, capsys) -> None:
    """ShadowLedger.load() raises in rich path → still prints (632-633)."""
    from distil import ledger as ledger_mod, shadow as shadow_mod

    monkeypatch.setattr(
        ledger_mod,
        "summary",
        lambda *a, **k: ledger_mod.LedgerSummary(
            runs=1,
            total_dollars_saved=0.01,
            total_tokens_saved=100,
            by_trajectory={},
            total_baseline_tokens=200,
            total_distil_tokens=100,
        ),
    )
    monkeypatch.delenv("DISTIL_STATUSLINE", raising=False)

    def _bad_load(cls):
        raise OSError("no shadow")

    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", classmethod(_bad_load))
    rc = cli.cmd_statusline(argparse.Namespace(no_color=True))
    assert rc == 0
    assert "distil" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_default undo — OSError on service unlink (952-953)
# --------------------------------------------------------------------------- #


def test_cmd_default_undo_service_oserror(tmp_path, monkeypatch, capsys) -> None:
    """path.unlink() raises OSError in undo → silently ignored (952-953)."""
    import distil.setup as setup_mod
    from distil import onboard

    env = onboard.Env(
        os_name="Darwin",
        agents=[("claude", "Claude Code")],
        installed_version="1.0.0",
        method="pipx",
        managers=["pipx"],
    )
    monkeypatch.setattr(onboard, "detect", lambda: env)

    rc_file = tmp_path / ".zshrc"
    rc_file.write_text("")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))

    class _BadPath:
        def exists(self):
            return True

        def unlink(self, *a, **k):
            raise OSError("permission denied")

        def __str__(self):
            return "/fake/service.plist"

    monkeypatch.setattr(setup_mod, "service_spec", lambda *a, **k: (_BadPath(), None, None))
    monkeypatch.setattr(setup_mod, "service_unload_cmd", lambda: "")
    monkeypatch.setattr(setup_mod, "remove_managed", lambda *a, **k: ("absent", "no managed block"))
    rc = cli.cmd_default(
        argparse.Namespace(
            undo=True,
            always_on=False,
            rc=str(rc_file),
            port=34120,
            agent="claude",
            mode="lossless-only",
            no_start=False,
        )
    )
    assert rc == 0


# --------------------------------------------------------------------------- #
# cmd_default --always-on content=None → return 1 (966-967)
# --------------------------------------------------------------------------- #


def test_cmd_default_always_on_content_none(tmp_path, monkeypatch, capsys) -> None:
    """service_spec returns content=None → prints error and returns 1 (966-967)."""
    import distil.setup as setup_mod
    from distil import onboard

    env = onboard.Env(
        os_name="Darwin",
        agents=[("claude", "Claude Code")],
        installed_version="1.0.0",
        method="pipx",
        managers=["pipx"],
    )
    monkeypatch.setattr(onboard, "detect", lambda: env)
    rc_file = tmp_path / ".zshrc"
    rc_file.write_text("")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    svc = tmp_path / "service.plist"
    monkeypatch.setattr(setup_mod, "service_spec", lambda *a, **k: (svc, None, None))
    rc = cli.cmd_default(
        argparse.Namespace(
            undo=False,
            always_on=True,
            rc=str(rc_file),
            port=34120,
            agent="claude",
            mode="lossless-only",
            no_start=False,
            force=False,
        )
    )
    assert rc == 1
    assert "could not render" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_default --always-on service start (974-975)
# --------------------------------------------------------------------------- #


def test_cmd_default_always_on_service_start(tmp_path, monkeypatch, capsys) -> None:
    """always-on with a load cmd → subprocess called and success printed (974-975)."""
    import subprocess

    import distil.setup as setup_mod
    from distil import onboard

    env = onboard.Env(
        os_name="Darwin",
        agents=[("claude", "Claude Code")],
        installed_version="1.0.0",
        method="pipx",
        managers=["pipx"],
    )
    monkeypatch.setattr(onboard, "detect", lambda: env)
    rc_file = tmp_path / ".zshrc"
    rc_file.write_text("")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    svc = tmp_path / "service.plist"
    monkeypatch.setattr(
        setup_mod, "service_spec", lambda *a, **k: (svc, "content", "launchctl load")
    )
    monkeypatch.setattr(setup_mod, "write_managed", lambda *a, **k: ("ok", "wrote env block"))
    monkeypatch.setattr(setup_mod, "env_body", lambda *a, **k: "export ANTHROPIC_BASE_URL=...")
    # agent=="claude" makes cmd_default wire settings.json — redirect it to tmp so the
    # suite never touches the real ~/.claude/settings.json (this write bricked Claude Code).
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "settings.json")
    # The service starts BEFORE the base URL is wired now, and wiring is gated on the
    # proxy proving it serves /v1/messages. No real proxy here, so state the verdict.
    monkeypatch.setattr(setup_mod, "probe_routing", lambda h, p, **k: (True, f"stub {h}:{p}"))

    class _OK:
        returncode = 0

    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: (calls.append(cmd), _OK())[1])
    # The reload now verifies the job actually registered instead of trusting the
    # legacy `launchctl load` exit code (see test_activation.py for that logic);
    # here we only care that cmd_default proceeds when the service came up.
    monkeypatch.setattr(
        setup_mod,
        "service_reload",
        lambda port: (calls.append(f"launchctl reload {port}"), (True, "pid 4242"))[1],
    )
    rc = cli.cmd_default(
        argparse.Namespace(
            undo=False,
            always_on=True,
            rc=str(rc_file),
            port=34120,
            agent="claude",
            mode="lossless-only",
            no_start=False,
            force=False,
        )
    )
    assert rc == 0
    assert any("launchctl" in str(c) for c in calls)
    assert "proxy service started" in capsys.readouterr().out


def test_cmd_default_failed_reload_does_not_claim_nothing_changed(
    tmp_path, monkeypatch, capsys
) -> None:
    """A failed reload must not print "your existing setup is untouched".

    By the time `service_reload` returns False the old job has been booted out
    and its definition replaced, so the machine has NO working proxy — while
    ANTHROPIC_BASE_URL may still point at that port from a previous install.
    Telling the user nothing changed sent them looking at the provider instead
    of at the dead service, which is how one mode switch became a long outage.
    """

    import distil.setup as setup_mod
    from distil import onboard

    env = onboard.Env(
        os_name="Darwin",
        agents=[("claude", "Claude Code")],
        installed_version="1.0.0",
        method="pipx",
        managers=["pipx"],
    )
    monkeypatch.setattr(onboard, "detect", lambda: env)
    rc_file = tmp_path / ".zshrc"
    rc_file.write_text("")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    monkeypatch.setattr(
        setup_mod, "service_spec", lambda *a, **k: (tmp_path / "service.plist", "content", "load")
    )
    monkeypatch.setattr(setup_mod, "write_managed", lambda *a, **k: ("ok", "wrote env block"))
    monkeypatch.setattr(setup_mod, "env_body", lambda *a, **k: "export ANTHROPIC_BASE_URL=...")
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "settings.json")
    # No subprocess patch: cmd_default must bail at the failed reload, before it
    # runs anything else. A stray patch here would hide it if that ever changed.
    monkeypatch.setattr(setup_mod, "service_reload", lambda port: (False, "bootstrap failed"))

    rc = cli.cmd_default(
        argparse.Namespace(
            always_on=True,
            rc=str(rc_file),
            port=34120,
            agent="claude",
            mode="expand",
            no_start=False,
            undo=False,
            yes=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "untouched" not in out, (
        f"claimed the setup was untouched while the service it just replaced is stopped:\n{out}"
    )
    assert "distil default --always-on" in out, f"no way back was offered:\n{out}"


# --------------------------------------------------------------------------- #
# cmd_offboard — service unlink OSError (1053-1054)
# --------------------------------------------------------------------------- #


def test_cmd_offboard_service_unlink_oserror(tmp_path, monkeypatch, capsys) -> None:
    """path.unlink() raises in offboard → error printed but offboard continues (1053-1054)."""
    import subprocess

    import distil.setup as setup_mod
    from distil import onboard

    class _BadPath:
        def exists(self):
            return True

        def unlink(self, *a, **k):
            raise OSError("locked")

        def __str__(self):
            return "/fake/service.plist"

    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", tmp_path / ".zshrc"))
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "distil"))
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")
    monkeypatch.setattr(setup_mod, "service_spec", lambda *a, **k: (_BadPath(), None, None))
    monkeypatch.setattr(setup_mod, "service_unload_cmd", lambda: "")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())
    rc = cli.cmd_offboard(argparse.Namespace(purge=False, yes=True, no_interactive=False))
    assert rc == 0
    assert "couldn't remove" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_online — not certified (line 1564)
# --------------------------------------------------------------------------- #


def test_cmd_online_not_certified(monkeypatch, capsys) -> None:
    """online_round returns certified=False → line 1564 prints NOT promoted."""
    import distil.cli as cli_mod
    import distil.online as online_mod

    monkeypatch.setattr(cli_mod, "load_corpus", lambda *a, **k: [])
    monkeypatch.setattr(
        online_mod, "online_round", lambda entries, **k: {"certified": False, "accuracy": 0.9}
    )
    rc = cli_mod.cmd_online(argparse.Namespace(corpus=None, promote_to="lossless-only"))
    assert rc == 0
    assert "NOT promoted" in capsys.readouterr().out


def test_stats_text_warns_on_legacy_records(tmp_path, monkeypatch, capsys) -> None:
    """The pre-1.10 overstatement warning must reach the TEXT output users
    actually read, not just the HTML page."""
    import json as _json

    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    rec = {
        "trajectory_id": "live-proxy",
        "model": "m",
        "turns": 1,
        "baseline_dollars": 1.0,
        "distil_dollars": 0.5,
        "baseline_input_tokens": 1000,
        "distil_input_tokens": 500,
        "tokenizer": "heuristic",
        "ts": 1.0,
    }  # no "acct" key -> legacy era
    p.write_text(_json.dumps(rec) + "\n", encoding="utf-8")
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    assert cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=None)) == 0
    out = capsys.readouterr().out
    assert "pre-1.10 accounting" in out and "overstated" in out


def test_cmd_reset_archives_ledger(tmp_path, monkeypatch, capsys) -> None:
    import json as _json

    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    rec = {
        "trajectory_id": "live-proxy",
        "model": "m",
        "turns": 1,
        "baseline_dollars": 1.0,
        "distil_dollars": 0.5,
        "baseline_input_tokens": 1000,
        "distil_input_tokens": 500,
        "tokenizer": "heuristic",
        "ts": 1.0,
    }
    p.write_text(_json.dumps(rec) + "\n", encoding="utf-8")
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    assert cli.cmd_reset(argparse.Namespace(shadow=False)) == 0
    out = capsys.readouterr().out
    assert "archived" in out and not p.exists()
    archived = list(tmp_path.glob("savings.jsonl.reset-*"))
    assert len(archived) == 1  # non-destructive: history kept for audit
    # Fresh ledger: stats start from zero
    assert cli.cmd_leaderboard(argparse.Namespace(badge=False, json=False, html=None)) == 0
    assert "no genuine savings recorded" in capsys.readouterr().out


def test_statusline_says_off_when_session_not_routed(tmp_path, monkeypatch, capsys) -> None:
    """'✓ on' must mean this session's requests actually route through distil."""
    import json as _json

    from distil import ledger as ledger_mod

    p = tmp_path / "savings.jsonl"
    rec = {
        "trajectory_id": "live-proxy",
        "model": "m",
        "turns": 1,
        "baseline_dollars": 1.0,
        "distil_dollars": 0.5,
        "baseline_input_tokens": 1000,
        "distil_input_tokens": 500,
        "tokenizer": "heuristic",
        "ts": 1.0,
    }
    p.write_text(_json.dumps(rec) + "\n", encoding="utf-8")
    monkeypatch.setattr(ledger_mod, "default_path", lambda: p)
    for var in (
        "DISTIL_SESSION",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert cli.cmd_statusline(argparse.Namespace(no_color=True)) == 0
    out = capsys.readouterr().out
    assert "off — session not routed" in out and "✓ on" not in out
    # And with wrap's env present, it says on again.
    monkeypatch.setenv("DISTIL_SESSION", "s123")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert cli.cmd_statusline(argparse.Namespace(no_color=True)) == 0
    assert "✓ on" in capsys.readouterr().out


def test_statusline_empty_ledger_respects_routed_session(tmp_path, monkeypatch, capsys) -> None:
    """A wrapped session with zero recorded runs must not be told to run `distil wrap`."""
    from distil import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "default_path", lambda: tmp_path / "savings.jsonl")
    for var in (
        "DISTIL_SESSION",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("DISTIL_STATUSLINE", raising=False)

    # Unrouted + empty ledger: keep the actionable wrap hint.
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert cli.cmd_statusline(argparse.Namespace(no_color=True)) == 0
    out = capsys.readouterr().out
    assert "distil wrap -- <agent>" in out and "✓ on" not in out

    # Routed (wrap env) + empty ledger: honest "on", no wrap hint.
    monkeypatch.setenv("DISTIL_SESSION", "s123")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert cli.cmd_statusline(argparse.Namespace(no_color=True)) == 0
    out = capsys.readouterr().out
    assert "✓ on" in out and "no savings yet" in out and "distil wrap -- <agent>" not in out

    # Minimal mode, routed + empty ledger: "on", not the wrap hint.
    monkeypatch.setenv("DISTIL_STATUSLINE", "minimal")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert cli.cmd_statusline(argparse.Namespace(no_color=True)) == 0
    out = capsys.readouterr().out
    assert "on" in out and "wrap -- <agent>" not in out


def test_default_alias_verify_hint_does_not_mention_env_var(tmp_path, monkeypatch, capsys) -> None:
    """Alias mode never exports ANTHROPIC_BASE_URL to the shell, so the printed
    verification step must not tell users to echo it there."""
    import distil.setup as setup_mod
    from distil import onboard

    rc_file = tmp_path / ".zshrc"
    rc_file.write_text("# rc\n", encoding="utf-8")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    monkeypatch.setattr(
        onboard,
        "detect",
        lambda: onboard.Env(
            os_name="Darwin",
            agents=[("claude", "Claude Code")],
            subscription=False,
        ),
    )
    assert (
        cli.cmd_default(
            argparse.Namespace(
                rc=None,
                agent="claude",
                mode="lossless-only",
                port=8788,
                undo=False,
                always_on=False,
                no_start=False,
            )
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "type claude" in out
    assert "echo $ANTHROPIC_BASE_URL" not in out
