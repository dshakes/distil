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

    # OAuth login WINS over a stray metered key: Claude Code authenticates that
    # traffic with the OAuth token, not the key, so it stays subscription-safe
    # (lossless-only) — the compression mode can't flip to PAYG/digest just because a
    # key happens to be in the env (the reported flip). This is what the docstring's
    # ToS-safety rationale above actually requires.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert doctor.subscription_mode() is True
    # …but an explicit override still forces metered for a genuinely-PAYG OAuth user.
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
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
    # aider goes through LiteLLM, which reads the OLDER name. OPENAI_BASE_URL is not
    # a variable aider consults, so the preset used to report success and route
    # nothing (aider.chat/docs/llms/openai-compat.html).
    assert AGENT_PRESETS["aider"][0] == "OPENAI_API_BASE"
    # GROK_BASE_URL does not exist; the documented override is this
    # (docs.x.ai/build/settings). Same silent no-op as aider's.
    assert AGENT_PRESETS["grok"][0] == "GROK_MODELS_BASE_URL"
    assert AGENT_PRESETS["openhands"][0] == "LLM_BASE_URL"
    assert AGENT_PRESETS["copilot"][0] == "COPILOT_PROVIDER_BASE_URL"
    assert AGENT_PRESETS["copilot"][3]["COPILOT_PROVIDER_TYPE"] == "anthropic"
    assert AGENT_PRESETS["kimi"][0] == "KIMI_BASE_URL"
    # cursor-agent deliberately absent: its env contract is not publicly documented
    assert "cursor-agent" not in AGENT_PRESETS
    # IDE extensions (still, sans copilot — GitHub Copilot CLI has a documented
    # BYOK contract and is a real preset now) have no argv to wrap and no
    # published env contract. Their supported path is the always-on proxy plus
    # the editor's own base-URL setting (docs/IDE-AGENTS.md). A preset here
    # would route nothing and report success.
    for ide in ("cursor", "cline", "continue", "windsurf"):
        assert ide not in AGENT_PRESETS, f"{ide} has no published env contract to honour"


def test_grok_upstream_carries_its_own_version_suffix():
    """Grok's default endpoint is https://api.x.ai/v1 — the /v1 is part of the base
    URL, not appended by the SDK the way the OpenAI client does it. Dropping it here
    would send every request to a 404 that looks like a distil bug."""
    from distil.onboard import AGENT_PRESETS

    assert AGENT_PRESETS["grok"][1].endswith("/v1")
