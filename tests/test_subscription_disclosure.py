"""Both routes into tool injection on a flat-rate session must disclose it.

`onboard.py` promises flat-rate users the safe default does "no tool injection".
Leaving that path is the user's call and it is allowed — but it must never be
silent, because the promise is what they are deciding against.

The notice used to be gated on `lossless_only and expand`, which is only the
`--lossless-only --expand` spelling. The route the docs actually hand a
subscription user, `distil default --mode expand`, leaves `lossless_only` False
— so the one path most people take was the one that said nothing. See ADR 0006.
"""

from __future__ import annotations

import pytest

from distil import proxy


@pytest.fixture
def _sub(monkeypatch):
    """A flat-rate machine, whatever the caller's real ~/.claude.json says."""
    monkeypatch.setattr("distil.doctor.subscription_mode", lambda: True)


def _make(**kw) -> str:
    """Build the app and return whatever it wrote to stderr at startup."""
    import io
    import sys

    err = io.StringIO()
    real = sys.stderr
    sys.stderr = err
    try:
        proxy.build_handler("https://api.anthropic.com", **kw)
    finally:
        sys.stderr = real
    return err.getvalue()


def test_the_documented_route_discloses_injection(_sub) -> None:
    """`distil default --mode expand` runs `distil proxy --expand`: lossless_only
    is False. This is the path that was silent."""
    out = _make(expand=True, lossless_only=False)
    assert "distil_expand is injected" in out or "injected" in out, (
        f"opted the user into tool injection with no notice:\n{out!r}"
    )
    assert "--expand" in out, "the notice must name the flag that caused it"


def test_the_explicit_spelling_still_discloses(_sub) -> None:
    out = _make(expand=True, lossless_only=True)
    assert "injected" in out


def test_the_safe_default_says_nothing_about_injection(_sub) -> None:
    """No opt-in, no injection, so no notice — a warning on the safe path is
    noise, and noise is what gets the real one tuned out."""
    out = _make(expand=False, lossless_only=True)
    assert "injected" not in out


def test_a_metered_session_is_not_told_about_a_flat_rate_boundary(monkeypatch) -> None:
    """PAYG turns expand on by default and has no such promise to weigh, so the
    notice would be pure noise there."""
    monkeypatch.setattr("distil.doctor.subscription_mode", lambda: False)
    out = _make(expand=True, lossless_only=False)
    assert "injected" not in out
