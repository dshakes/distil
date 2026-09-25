"""The hook follows the proxy's billing policy: subscription → lossless-only unless
the user opted in with --digest; metered → digest by default; an entry written
before tiers existed keeps the lossless-only behaviour it was installed with."""

from __future__ import annotations

import io
import json

import pytest

from distil import hook
from distil.cli import main as cli

VARIED = "\n".join(
    f"2026-09-25 INFO worker-{i % 7} processed batch {i * 7} in {i % 13}ms" for i in range(400)
)
EVENT = json.dumps(
    {
        "tool_name": "Bash",
        "tool_input": {"command": "make test"},
        "tool_response": {"stdout": VARIED, "stderr": "", "interrupted": False},
    }
)


@pytest.fixture
def subscription(monkeypatch):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")


@pytest.fixture
def metered(monkeypatch):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")


def _digested(out: str) -> bool:
    return "distil expand" in out


def _run_main(argv, monkeypatch, capsys) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(EVENT))
    assert hook.main(argv) == 0
    return capsys.readouterr().out


def test_subscription_auto_is_lossless_only(subscription, monkeypatch, capsys):
    assert not _digested(_run_main(["--tier", "auto"], monkeypatch, capsys))
    assert hook.tier_decision("auto") == (
        False,
        "lossless-only: subscription login detected (opt in with --digest)",
    )


def test_subscription_opt_in_digests(subscription, monkeypatch, capsys):
    assert _digested(_run_main(["--tier", "digest"], monkeypatch, capsys))


def test_metered_auto_digests(metered, monkeypatch, capsys):
    assert _digested(_run_main(["--tier", "auto"], monkeypatch, capsys))
    assert hook.tier_decision("auto")[0]


def test_oauth_login_is_detected_by_the_proxys_own_detector(monkeypatch, tmp_path):
    """No second detector: a Claude login in ~/.claude.json is what subscription_mode reads."""
    monkeypatch.delenv("DISTIL_SUBSCRIPTION", raising=False)
    assert hook.tier_decision("auto")[0]  # sandbox HOME: no login -> metered
    (tmp_path / ".claude.json").write_text(json.dumps({"oauthAccount": {"x": 1}}))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert not hook.tier_decision("auto")[0]


@pytest.mark.parametrize("billing", ["0", "1"])
def test_pre_tier_entry_keeps_lossless_only_on_upgrade(billing, monkeypatch, capsys):
    """`python -m distil.hook` with no --tier is what every earlier install wrote."""
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", billing)
    assert not _digested(_run_main([], monkeypatch, capsys))
    assert not hook.tier_decision(None)[0]


def test_unknown_tier_is_a_no_op(metered, monkeypatch, capsys):
    assert _run_main(["--tier", "bogus"], monkeypatch, capsys) == "{}"


def test_run_restores_the_tier_after_a_call(metered):
    hook.run(EVENT, "claude", "lossless")
    assert hook._TIER == "auto"
    assert hook.compress_text(VARIED) is not None and "distil expand" in hook.compress_text(VARIED)


# ------------------------------------------------------------------ install / status


def test_upgrade_path_legacy_entry_then_opt_in(subscription, capsys):
    path = hook.config_path("claude")
    path.parent.mkdir(parents=True)
    legacy = {
        "matcher": "Bash|mcp__.*",
        "hooks": [{"type": "command", "command": "py -m distil.hook"}],
    }
    path.write_text(json.dumps({"hooks": {"PostToolUse": [legacy]}}))
    assert hook.installed_tier("claude") == "lossless"
    assert cli(["hook", "status"]) == 0
    assert "installed before hook tiers" in capsys.readouterr().out

    assert cli(["hook", "install", "--digest"]) == 0
    assert "opted in with --digest" in capsys.readouterr().out
    (entry,) = json.loads(path.read_text())["hooks"]["PostToolUse"]
    assert entry["hooks"][0]["command"].endswith("-m distil.hook --tier digest")
    rec = hook._read_owned()[str(path.resolve())]
    assert rec["tier"] == "digest" and rec["digest_opt_in_ts"] > 0

    assert cli(["hook", "install"]) == 0  # re-install without the flag: back to policy
    assert hook.installed_tier("claude") == "auto"
    assert "digest_opt_in_ts" not in hook._read_owned()[str(path.resolve())]
    assert "subscription login detected" in capsys.readouterr().out


def test_setup_hooks_digest_flag(metered, capsys):
    (hook.config_path("gemini").parent).mkdir()
    assert cli(["setup", "--hooks", "--digest"]) == 0
    assert hook.installed_tier("claude") == "digest"
    assert hook.installed_tier("gemini") == "digest"
    assert hook.installed_tier("cursor") is None


def test_status_on_metered_says_digest(metered, capsys):
    hook.install_hook("codex")
    assert hook.print_status() == 0
    assert "digest: metered API key" in capsys.readouterr().out
