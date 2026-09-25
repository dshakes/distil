"""The conftest HOME/USERPROFILE sandbox holds: no in-process resolver of a user's agent
config can point at the real home during a test, whatever the test forgot to set."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from distil import config_wrap
from distil import setup as setup_mod

# Resolved at import (collection), before any fixture redirects HOME/USERPROFILE.
_REAL_HOME = Path(os.path.expanduser("~")).resolve()


def _resolvers() -> dict[str, Path]:
    return {
        "claude settings": setup_mod.default_settings_path(),
        "factory": config_wrap._factory_settings_path(),
        "omp": config_wrap._omp_models_path(),
        "crush": config_wrap._crush_config_path(),
        "cline": config_wrap._cline_providers_path(["cline"]),
    }


def _in_sandbox(p: Path, base: Path) -> bool:
    """Under pytest's basetemp. Not "outside the real home": on Windows the temp dir
    (AppData\\Local\\Temp) is itself under the real profile."""
    try:
        p.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def test_no_agent_config_resolves_under_the_real_home(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    base = tmp_path_factory.getbasetemp()
    for name, p in _resolvers().items():
        assert _in_sandbox(p, base), f"{name} resolves outside the test sandbox: {p}"
    assert Path.home().resolve() != _REAL_HOME


def test_a_test_that_sets_only_home_still_cannot_reach_the_real_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The Windows failure mode: HOME set, USERPROFILE forgotten. USERPROFILE is already
    the sandbox, so Path.home() on Windows lands in tmp, never the runner's profile."""
    monkeypatch.setenv("HOME", str(tmp_path))
    base = tmp_path_factory.getbasetemp()
    assert _in_sandbox(Path(os.environ["USERPROFILE"]), base)
    assert _in_sandbox(setup_mod.default_settings_path(), base)


@pytest.mark.real_home
def test_the_opt_out_sees_the_genuine_home() -> None:
    """Read-only: resolution tests can ask for the real home. Nothing is written."""
    assert setup_mod.default_settings_path().parent.parent.resolve() == _REAL_HOME
