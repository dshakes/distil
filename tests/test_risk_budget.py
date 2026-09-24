"""One risk budget: every surface that says "safe" reads ``conformal.BUDGET_ALPHA``.

The failure this pins is the one the audit found: the certificate, the live drift
alarm and the proof ledger's risk line each carried their own threshold, so one ledger
could print "intact" beside a bound above budget. The test that matters is the
counterfactual — move the ONE constant and every verdict must move with it. A surface
that kept a private copy would stay put and fail here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest

from distil import conformal, drift
from distil.shadow import SIG_VERSION

# 10% sustained compression-attributable harm, spread evenly (an anytime-valid alarm
# rightly trips on a front-loaded run): above a 5% budget, below a 30% one.
_DIFFS = ([-1] + [0] * 9) * 60


def _write(home: Path, diffs: list[int]) -> None:
    with (home / "shadow.jsonl").open("w", encoding="utf-8") as f:
        for d in diffs:
            eq, aa = {-1: (False, True), 0: (True, True), 1: (True, False)}[d]
            row = {"equivalent": eq, "aa_equal": aa, "ts": time.time(), "kind": "paired"}
            f.write(json.dumps({**row, "sig": SIG_VERSION, "mode": "digest"}) + "\n")


def _verdicts(home: Path) -> dict[str, object]:
    """Every budget-reading surface, computed from scratch against the same evidence."""
    from distil.conformal import calibrate
    from distil.corpus import load_corpus
    from distil.proof_ledger import proof_lines
    from distil.replay.runner import DeterministicRunner

    (home / "drift.json").unlink(missing_ok=True)
    guard = drift.DriftGuard.start(watch=False)  # bootstraps the e-process from the ledger
    lines = dict(proof_lines())
    return {
        "certificate": calibrate(load_corpus(), DeterministicRunner()).level is not None,
        "drift": "BREACHED" in lines["budget"],
        "risk": "within" in lines["risk"],
        "guard": guard.engaged,
    }


def test_one_constant_moves_every_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    _write(tmp_path, _DIFFS)

    monkeypatch.setattr(conformal, "BUDGET_ALPHA", 0.01)
    tight = _verdicts(tmp_path)
    monkeypatch.setattr(conformal, "BUDGET_ALPHA", 0.30)
    loose = _verdicts(tmp_path)

    assert tight == {"certificate": False, "drift": True, "risk": False, "guard": True}
    assert loose == {"certificate": True, "drift": False, "risk": True, "guard": False}


def test_the_shipped_budget_is_the_published_one():
    """5% at 95% is what the site and every certificate state. Changing it is a public
    claim change, not a refactor — this is the tripwire that makes that visible."""
    assert conformal.BUDGET_ALPHA == 0.05
    assert conformal.BUDGET_DELTA == 0.05
    assert conformal.CERT_MARGIN == 0.02


def test_the_gate_margin_is_never_looser_than_the_budget():
    """The TOST gate and the budget measure the same thing (paired decision change).
    A gate margin above the budget would certify points the alarm exists to stop."""
    assert 0.0 < conformal.CERT_MARGIN <= conformal.BUDGET_ALPHA


def _default(parser: argparse.ArgumentParser, cmd: str, dest: str) -> object:
    sub = parser._subparsers._group_actions[0].choices[cmd]  # type: ignore[union-attr]
    return next(a.default for a in sub._actions if a.dest == dest)


@pytest.mark.parametrize(
    ("cmd", "dest", "name"),
    [
        ("conformal", "alpha", "BUDGET_ALPHA"),
        ("conformal", "delta", "BUDGET_DELTA"),
        ("certify-trajectories", "alpha", "BUDGET_ALPHA"),
        ("certify-trajectories", "delta", "BUDGET_DELTA"),
        ("calibrate", "margin", "BUDGET_ALPHA"),
        ("certify", "margin", "CERT_MARGIN"),
        ("certify", "alpha", "BUDGET_DELTA"),
        ("bench", "margin", "CERT_MARGIN"),
        ("benchmark", "margin", "CERT_MARGIN"),
    ],
)
def test_every_cli_default_reads_the_budget_at_parse_time(monkeypatch, cmd, dest, name):
    from distil.cli import build_parser

    monkeypatch.setattr(conformal, name, 0.0123)
    assert _default(build_parser(), cmd, dest) == 0.0123


def test_the_library_gate_defaults_are_the_budget():
    import inspect

    from distil.calibrate import calibrate_operating_point
    from distil.certify.gate import certify, certify_pooled
    from distil.certify.trajectory_risk import certify_trajectory_risk

    for fn in (certify, certify_pooled):
        ps = inspect.signature(fn).parameters
        assert ps["margin"].default == conformal.CERT_MARGIN
        assert ps["alpha"].default == conformal.BUDGET_DELTA
    ps = inspect.signature(certify_trajectory_risk).parameters
    assert ps["alpha"].default == conformal.BUDGET_ALPHA
    assert ps["delta"].default == conformal.BUDGET_DELTA
    margin = inspect.signature(calibrate_operating_point).parameters["margin"]
    assert margin.default == conformal.BUDGET_ALPHA
