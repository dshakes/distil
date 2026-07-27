"""Consent asked at wrap exit — every guard, enumerated.

This is the code that decides whether a user is enrolled in telemetry. The
failure that matters is not "the prompt didn't show" — it is enrolling someone
who did not say yes, or recording a decision they did not make. So the guards
are tested individually rather than sampled, and the ones that could silently
enrol are tested hardest.
"""

from __future__ import annotations

import io

import pytest

from distil import census


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal."""

    def __init__(self, tty: bool = True) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:  # noqa: D102
        return self._tty


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("DISTIL_NO_TELEMETRY", raising=False)
    # Default: a session that actually saved something, so the "nothing to
    # share" guard does not silently satisfy every test below.
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 5_000)


def _answer(monkeypatch, text: str) -> None:
    monkeypatch.setattr("builtins.input", lambda *a, **k: text)


def _raises(monkeypatch, exc) -> None:
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr("builtins.input", boom)


def _tripwire(monkeypatch) -> list:
    """Record whether the prompt was reached, WITHOUT raising.

    A raising tripwire is useless here: maybe_ask_consent ends in a broad
    `except Exception` (it must — it runs on the teardown path and may not alter
    a user's exit code), so an AssertionError from input() is swallowed and the
    test passes even with the guard deleted. Verified: removing the TTY check
    left the raising version green. Guards are therefore asserted on OBSERVABLE
    effects — the prompt never printed, and nobody was enrolled.
    """
    calls: list = []

    def spy(*a, **k):
        calls.append(a)
        return "y"  # the dangerous answer, so a leak enrols and is caught

    monkeypatch.setattr("builtins.input", spy)
    return calls


# ---------------------------------------------------------------------------
# The happy paths
# ---------------------------------------------------------------------------


def test_yes_opts_in(monkeypatch):
    _answer(monkeypatch, "y")
    out = FakeTTY()
    assert census.maybe_ask_consent(out=out, inp=FakeTTY()) is True
    assert census.consent() == "on"
    assert "census on" in out.getvalue()


def test_no_opts_out_and_is_remembered(monkeypatch):
    _answer(monkeypatch, "n")
    assert census.maybe_ask_consent(out=FakeTTY(), inp=FakeTTY()) is False
    assert census.consent() == "off"
    # Asked once: a decision ends it permanently.
    calls = _tripwire(monkeypatch)
    out = FakeTTY()
    assert census.maybe_ask_consent(out=out, inp=FakeTTY()) is None
    assert not calls, "re-asked after a recorded decision"
    assert out.getvalue() == ""
    assert census.consent() == "off", "a recorded decline was overwritten"


def test_bare_enter_declines(monkeypatch):
    """The prompt is [y/N] — the default must be no, not yes."""
    _answer(monkeypatch, "")
    assert census.maybe_ask_consent(out=FakeTTY(), inp=FakeTTY()) is False
    assert census.consent() == "off"


@pytest.mark.parametrize("word", ["yes", "Y", "YES", " y "])
def test_affirmative_spellings(monkeypatch, word):
    _answer(monkeypatch, word)
    assert census.maybe_ask_consent(out=FakeTTY(), inp=FakeTTY()) is True


@pytest.mark.parametrize("word", ["nope", "later", "sure", "1", "true"])
def test_anything_not_clearly_yes_is_a_decline(monkeypatch, word):
    """Ambiguity must never resolve to consent."""
    _answer(monkeypatch, word)
    assert census.maybe_ask_consent(out=FakeTTY(), inp=FakeTTY()) is False
    assert census.consent() == "off"


# ---------------------------------------------------------------------------
# The guards that prevent silent enrolment
# ---------------------------------------------------------------------------


def test_never_prompts_when_not_a_tty(monkeypatch):
    """Pipes, CI and headless runs must never prompt, and therefore never enrol.
    input() is a tripwire here: reaching it at all is the failure."""
    calls = _tripwire(monkeypatch)
    for out, inp in ((FakeTTY(True), FakeTTY(False)), (FakeTTY(False), FakeTTY(True))):
        assert census.maybe_ask_consent(out=out, inp=inp) is None
        assert out.getvalue() == "", "printed a consent prompt to a non-terminal"
    assert not calls, "reached the prompt in a non-interactive session"
    assert census.consent() is None


@pytest.mark.parametrize("var", ["DO_NOT_TRACK", "DISTIL_NO_TELEMETRY"])
def test_kill_switches_prevent_the_question(monkeypatch, var):
    """Someone who set DO_NOT_TRACK has already answered."""
    monkeypatch.setenv(var, "1")
    calls = _tripwire(monkeypatch)
    out = FakeTTY()
    assert census.maybe_ask_consent(out=out, inp=FakeTTY()) is None
    assert not calls, f"prompted despite {var}"
    assert out.getvalue() == ""
    assert census.consent() is None


def test_does_not_ask_when_nothing_was_saved(monkeypatch):
    """Asking a user who got no value to share their numbers is a worse question
    and a worse experience than not asking."""
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 0)
    calls = _tripwire(monkeypatch)
    out = FakeTTY()
    assert census.maybe_ask_consent(out=out, inp=FakeTTY()) is None
    assert not calls, "prompted with nothing to share"
    assert out.getvalue() == ""
    assert census.consent() is None


def test_interrupt_is_not_an_answer(monkeypatch):
    """Ctrl-C means 'not now', not 'no'. Recording a decline here would burn the
    one chance to ask on a keystroke that meant nothing of the sort."""
    for exc in (KeyboardInterrupt, EOFError):
        _raises(monkeypatch, exc)
        assert census.maybe_ask_consent(out=FakeTTY(), inp=FakeTTY()) is None
        assert census.consent() is None, "an interrupt was recorded as a decision"


def test_unanswered_prompts_stop_after_a_cap(monkeypatch):
    """A prompt nobody answers must not become a prompt nobody escapes."""
    _raises(monkeypatch, KeyboardInterrupt)
    for _ in range(census.MAX_UNANSWERED_ASKS):
        assert census.maybe_ask_consent(out=FakeTTY(), inp=FakeTTY()) is None
    # Past the cap it must go quiet.
    calls = _tripwire(monkeypatch)
    out = FakeTTY()
    assert census.maybe_ask_consent(out=out, inp=FakeTTY()) is None
    assert not calls, "kept nagging past the cap"
    assert out.getvalue() == ""
    assert census.consent() is None, "the cap must not imply a decision"


def test_never_raises_on_the_teardown_path(monkeypatch):
    """This runs at wrap exit. An exception here would change a user's exit code
    over telemetry."""

    def explode():
        raise RuntimeError("ledger on fire")

    monkeypatch.setattr(census, "_current_saved_tokens", explode)
    assert census.maybe_ask_consent(out=FakeTTY(), inp=FakeTTY()) is None


def test_a_yes_takes_effect_in_the_same_session(monkeypatch):
    """Consent is asked before the ping, so the number the user just agreed to
    share is the number they were looking at — not one a day later."""
    _answer(monkeypatch, "y")
    assert census.maybe_ask_consent(out=FakeTTY(), inp=FakeTTY()) is True
    assert census.enabled() is True, "opting in did not take effect immediately"
