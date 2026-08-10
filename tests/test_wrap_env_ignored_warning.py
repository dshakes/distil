"""Some agents ignore the environment unless told not to — say so before the session.

Setting a base-URL variable an agent will ignore is worse than doing nothing: the
wrap prints its usual success, the agent talks straight to the provider, and the
only symptom is savings that stay at zero. The user has no way to tell that apart
from "compression isn't working".

distil's rule is that "wired" means "a request provably reached the proxy".
`_warn_if_env_ignored` is what keeps a preset from quietly violating it.
"""

from __future__ import annotations

from distil.cli import _ENV_REQUIRES_FLAG, _warn_if_env_ignored


def test_openhands_without_the_flag_is_warned(capsys) -> None:
    _warn_if_env_ignored("openhands", ["openhands", "--headless"])
    err = capsys.readouterr().err
    assert "--override-with-envs" in err, "the user was not told which flag they need"
    assert "route NOTHING" in err, "the consequence has to be stated, not implied"


def test_openhands_with_the_flag_is_silent(capsys) -> None:
    _warn_if_env_ignored("openhands", ["openhands", "--override-with-envs", "--headless"])
    assert capsys.readouterr().err == "", "warning fired on a correctly-invoked wrap"


def test_agents_that_honour_the_environment_are_not_nagged(capsys) -> None:
    """A warning on every wrap is noise, and noise is what gets the real one ignored."""
    for agent in ("claude", "codex", "gemini", "grok"):
        _warn_if_env_ignored(agent, [agent])
    assert capsys.readouterr().err == ""


def test_every_flagged_agent_is_a_real_preset() -> None:
    """A flag requirement for an agent distil cannot wrap is dead configuration."""
    from distil.onboard import AGENT_PRESETS

    unknown = set(_ENV_REQUIRES_FLAG) - set(AGENT_PRESETS)
    assert not unknown, f"flag requirements for non-presets: {sorted(unknown)}"
