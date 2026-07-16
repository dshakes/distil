"""Pins the UNDOCUMENTED upstream contracts distil's agent interception rests on.

distil wraps coding agents by injecting a base-url env var and, for subscription
sessions, detecting the agent's OAuth login from its local config. Neither surface
is a documented, stable API — an agent update can silently change them (this is an
open item on the GA risk register). These tests exist so that when WE change our
side of the contract it's a visible, reviewed diff; the runtime half of the tripwire
is `cmd_wrap`'s zero-traffic warning (a preset session that ends with no proxied
requests points at the agent no longer honoring the env var).

If one of these fails after an intentional change, update BOTH the assumption here
and the places that consume it (doctor.subscription_mode, onboard.AGENT_PRESETS).
"""

from __future__ import annotations

from pathlib import Path


def test_claude_oauth_detection_contract(monkeypatch, tmp_path):
    """Contract: a Claude Pro/Max login is detectable as the literal key string
    ``"oauthAccount"`` inside ``~/.claude.json`` — a content-free presence check
    (never parsed, no token read). If Claude Code moves or renames its session
    file, subscription auto-detection silently degrades to PAYG semantics, which
    would apply lossy tiers to a ToS-sensitive session — hence this pin."""
    from distil import doctor

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DISTIL_SUBSCRIPTION", raising=False)

    # no file → not a subscription session
    assert doctor.subscription_mode() is False

    # the documented-by-observation shape: top-level "oauthAccount" key
    (tmp_path / ".claude.json").write_text('{"oauthAccount": {"x": 1}}', encoding="utf-8")
    assert doctor.subscription_mode() is True

    # a metered key means real dollars, regardless of the OAuth file
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert doctor.subscription_mode() is False


def test_agent_preset_env_var_contract():
    """Contract: each wrapped agent honors exactly this env var for its base URL.
    These names are the agents' SDK conventions, not guaranteed APIs — a rename
    upstream is invisible to unit tests (the runtime zero-traffic warning is the
    live tripwire); THIS test pins our side so a change is deliberate."""
    from distil.onboard import AGENT_PRESETS

    assert AGENT_PRESETS["claude"][0] == "ANTHROPIC_BASE_URL"
    assert AGENT_PRESETS["codex"][0] == "OPENAI_BASE_URL"
    assert AGENT_PRESETS["gemini"][0] == "GOOGLE_GEMINI_BASE_URL"
    assert AGENT_PRESETS["aider"][0] == "OPENAI_BASE_URL"
    # cursor-agent deliberately absent: its env contract is not publicly documented
    assert "cursor-agent" not in AGENT_PRESETS
