"""`make gate` must run every gate CI runs.

`make gate` exists so that "green locally" means "green on push". It stops being
worth anything the moment CI runs a gate the Makefile does not — a local pass then
carries a promise it cannot keep, and the failure shows up after the push instead of
before it. That had already happened once: CI ran `distil validate`, `make gate` did
not, so an adversarial-invariant regression was un-catchable locally.

This compares the two by the distil subcommand each invokes, not by step name or
flags, because flags legitimately differ (CI passes `--allow-unavailable` for a third
party's outage; a developer running it by hand should not have to).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_INVOKE = re.compile(r"uv run distil ([a-z-]+)")

# Gates that are deliberately CI-only, with the reason. Anything not listed here must
# appear in both — an unexplained absence is the bug this test exists to catch.
_CI_ONLY = {
    # Builds and installs the wheel; `make build` covers it and it is slow by design.
    "packaging": "covered by `make build`",
}


def _distil_commands(text: str, *, after: str = "") -> set[str]:
    if after:
        text = text.split(after, 1)[-1]
    return set(_INVOKE.findall(text))


def _make_gate_commands() -> set[str]:
    """Resolve `gate:`'s prerequisites to the distil subcommand each target runs."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    gate_line = next(ln for ln in makefile.splitlines() if ln.startswith("gate:"))
    targets = gate_line.split(":", 1)[1].split("##")[0].split()

    found: set[str] = set()
    for target in targets:
        body = re.search(rf"^{re.escape(target)}:.*?\n((?:[\t ].*\n|\n)*)", makefile, re.M)
        assert body, f"`make gate` depends on `{target}`, which has no target in the Makefile"
        found |= _distil_commands(body.group(1))
    return found


def test_make_gate_runs_every_gate_ci_runs() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    # Start after the test step so setup/lint jobs earlier in the file are not counted.
    ci_cmds = _distil_commands(ci, after="- name: Tests") - set(_CI_ONLY)
    missing = ci_cmds - _make_gate_commands()
    assert not missing, (
        f"CI runs `distil {', '.join(sorted(missing))}` but `make gate` does not. "
        "A local gate weaker than CI promises a green push it cannot deliver — add the "
        "target to `gate:`, or list it in _CI_ONLY with the reason."
    )


def test_every_gate_target_exists_and_is_phony() -> None:
    """A gate prerequisite with no target fails the whole gate with a Make error, which
    reads like a broken build rather than a missing check."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    phony = next(ln for ln in makefile.splitlines() if ln.startswith(".PHONY:"))
    gate_line = next(ln for ln in makefile.splitlines() if ln.startswith("gate:"))
    for target in gate_line.split(":", 1)[1].split("##")[0].split():
        assert re.search(rf"^{re.escape(target)}:", makefile, re.M), f"no target `{target}`"
        assert target in phony.split(), f"`{target}` missing from .PHONY (a stray file shadows it)"
