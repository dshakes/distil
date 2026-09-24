"""The four statistical verdicts in the wrap exit summary, ``distil stats`` and dissect.

What these guard is the one failure that matters for a line printed on every exit: a
verdict computed over evidence too thin to support it. Every assertion here is either
"below the floor it prints no number" or "the number it prints is the one the estimator
actually supports".
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from distil.conformal import tight_risk_bound
from distil.conformal import BUDGET_ALPHA, BUDGET_DELTA
from distil.drift import paired_loss
from distil.shadow import SIG_VERSION, VERDICT_MIN_AA, VERDICT_MIN_AB, bootstrap_ci

ARTIFACT = Path(__file__).resolve().parent.parent / "benchmarks/results/shadow-live-2026-09-15.json"


def _write_paired(
    home: Path, diffs: list[int], *, out_a: int | None = None, out_b: int = 0
) -> None:
    """Write paired shadow rows whose per-request difference is exactly ``diffs``."""
    p = home / "shadow.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for d in diffs:
            # d = 1{A==B} - 1{A==A'}: (eq, aa_eq) = (1,1)->0, (0,1)->-1, (1,0)->+1
            eq, aa_eq = {(-1): (False, True), 0: (True, True), 1: (True, False)}[d]
            row = {
                "equivalent": eq,
                "aa_equal": aa_eq,
                "ts": time.time(),
                "kind": "paired",
                "sig": SIG_VERSION,
                "mode": "digest",
            }
            if out_a is not None:
                row.update({"in_a": 1000, "in_b": 500, "out_a": out_a, "out_b": out_b})
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Conformal bound
# ---------------------------------------------------------------------------


def test_conformal_coverage_meets_nominal():
    """1000 runs per cell: the bound must cover the true harm at least 95% of the time."""
    for harm, u, n in ((0.05, 0.26, 100), (0.05, 0.26, 400), (0.15, 0.30, 200)):
        covered = 0
        for tr in range(1000):
            rng = random.Random(0xC0FFEE ^ (tr * 7919))
            diffs = []
            for _ in range(n):
                r = rng.random()
                diffs.append(-1 if r < u else (1 if r < u + (u - harm) else 0))
            bound = 2 * tight_risk_bound([paired_loss(d) for d in diffs], BUDGET_DELTA) - 1
            covered += bound >= harm
        assert covered / 1000 >= 0.95, f"harm={harm} n={n}: coverage {covered / 1000:.3f}"


def test_bound_is_never_tighter_than_the_paired_interval_on_the_real_sample():
    """A distribution-free bound that undercuts the bootstrap interval it sits next to
    is a bug, not a better number. Checked against the maintainer's committed artifact."""
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    n = art["n_ab_samples"]
    ci_upper_change = -art["paired_difference_ci95"][0]  # harm side of the interval
    total_d = round(art["paired_difference"] * n)
    # The joint is not published (the artifact is content-free), so check every
    # dispersion consistent with it — the bound must clear the interval at all of them.
    for u in (0.05, 0.15, 0.26, 0.40):
        neg = round(u * n)
        pos = neg + total_d
        if pos < 0 or neg + pos > n:
            continue
        diffs = [-1] * neg + [1] * pos + [0] * (n - neg - pos)
        assert sum(diffs) == total_d
        bound = 2 * tight_risk_bound([paired_loss(d) for d in diffs], BUDGET_DELTA) - 1
        assert bound >= ci_upper_change, (
            f"u={u}: bound {bound:.4f} < CI upper {ci_upper_change:.4f}"
        )


def test_artifact_is_internally_consistent():
    """The committed artifact is what the docs and ``claims.json`` cite, and the raw
    ``shadow.jsonl`` it was summarised from is on one machine and never committed. So the
    check that can run forever is arithmetic: every published figure has to follow from
    the counts published beside it, and from the constants this repo actually ships.

    Every assertion here reads only repo files. A hand-edited number in this artifact —
    the realistic failure, since it is written by a human running one command — moves at
    least one of these identities.
    """
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    n_ab, n_aa = art["n_ab_samples"], art["n_aa_samples"]

    # 1. The per-mode counts are the sample, not a separate tally of it.
    modes = art["by_mode"].values()
    assert sum(m["ab_n"] for m in modes) == n_ab
    assert sum(m["aa_n"] for m in modes) == n_aa
    assert 100 * sum(m["ab_eq"] for m in modes) / n_ab == pytest.approx(
        art["raw_agreement_pct"], abs=0.05
    )
    assert 100 * sum(m["aa_eq"] for m in modes) / n_aa == pytest.approx(
        art["self_agreement_pct"], abs=0.05
    )

    # 2. Equivalence is the paired difference restated, not an independent number.
    assert 100 * (1 + art["paired_difference"]) == pytest.approx(
        art["decision_equivalence_pct"], abs=0.05
    )
    for d, pct in zip(art["paired_difference_ci95"], art["decision_equivalence_ci95"]):
        assert 100 * (1 + d) == pytest.approx(pct, abs=0.05)

    # 3. Every interval brackets its own point estimate, low end first.
    for point, ci in (
        ("raw_agreement_pct", "raw_agreement_ci95"),
        ("self_agreement_pct", "self_agreement_ci95"),
        ("paired_difference", "paired_difference_ci95"),
        ("decision_equivalence_pct", "decision_equivalence_ci95"),
        ("output_token_delta_per_request", "output_token_delta_ci95"),
    ):
        lo, hi = art[ci]
        assert lo <= art[point] <= hi, ci

    # 4. The artifact clears the floor this code ships, and names the estimator #165 left.
    assert art["reporting_floor"] == {"ab_min": VERDICT_MIN_AB, "aa_min": VERDICT_MIN_AA}
    assert n_ab >= VERDICT_MIN_AB and n_aa >= VERDICT_MIN_AA
    assert art["estimator"] == "paired" and art["sig_version"] == SIG_VERSION

    # 5. The two verdicts the prose states must be the ones the numbers support: harm
    #    inside the certified budget, and an output effect whose interval excludes zero.
    assert -art["paired_difference_ci95"][0] <= BUDGET_ALPHA, art["verdict"]
    assert art["output_token_delta_ci95"][1] < 0


# ---------------------------------------------------------------------------
# The printed lines
# ---------------------------------------------------------------------------


def test_every_line_withholds_its_number_below_the_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    _write_paired(tmp_path, [0] * (VERDICT_MIN_AB - 1), out_a=100, out_b=50)
    from distil.proof_ledger import proof_lines

    lines = dict(proof_lines())
    for label in ("budget", "risk", "output"):
        assert "not enough samples yet" in lines[label], label
        assert "%" not in lines[label]


def test_risk_line_reports_the_bound_it_computed(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    diffs = [-1] * 10 + [0] * 190
    _write_paired(tmp_path, diffs)
    from distil.proof_ledger import proof_lines

    lines = dict(proof_lines())
    expected = 2 * tight_risk_bound([paired_loss(d) for d in diffs], BUDGET_DELTA) - 1
    assert f"≤ {expected * 100:.1f}%" in lines["risk"]
    assert "95% conformal bound, n=200" in lines["risk"]


def test_budget_line_flips_to_breached_on_sustained_harm(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    # Enough evidence for the bound to clear the budget: "intact" is earned, not assumed.
    _write_paired(tmp_path, [0] * 200 + [1] * 20)
    from distil.proof_ledger import proof_lines

    assert "intact (e-value" in dict(proof_lines())["budget"]
    _write_paired(tmp_path, [-1] * 300)
    line = dict(proof_lines())["budget"]
    assert "BREACHED at sample" in line
    assert "held at lossless-only" in line  # and it says the proxy acted on it


def test_budget_line_never_says_intact_beside_a_bound_above_budget(tmp_path, monkeypatch):
    """The regression this commit fixes: 60 neutral samples have not tripped the alarm,
    but their bound is far above the budget — the old line printed "intact" next to it."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    _write_paired(tmp_path, [0] * 60)
    from distil.proof_ledger import proof_lines

    lines = dict(proof_lines())
    assert "ABOVE the 5% budget" in lines["risk"]
    assert "intact" not in lines["budget"]
    assert "unproven" in lines["budget"]


def test_budget_line_follows_a_truncated_shadow_file(tmp_path, monkeypatch):
    """End to end through the real reporting path: the drift state indexes into the
    shadow ledger, so archiving shadow.jsonl outside `distil reset --shadow` must not
    leave every surface quoting an n the file can no longer support."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil.proof_ledger import proof_lines

    _write_paired(tmp_path, [-1] * 200)
    assert "BREACHED at sample" in dict(proof_lines())["budget"]

    (tmp_path / "shadow.jsonl").unlink()  # archived by hand, state file left behind
    _write_paired(tmp_path, [0] * 60)
    line = dict(proof_lines())["budget"]
    assert "n=60" in line
    assert "BREACHED at sample" not in line  # the e-process verdict follows the new stream
    assert "restarted: the shadow stream was replaced" in line
    # ...but the guard's hold does not: hand-archiving the evidence is not the reset.
    assert "BREACHED earlier" in line and "held at lossless-only" in line


def test_output_line_names_a_direction_only_when_the_ci_excludes_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    # Every replay: 200 output tokens on A, 100 on B — an unambiguous shortening.
    _write_paired(tmp_path, [0] * 60, out_a=200, out_b=100)
    from distil.proof_ledger import proof_lines

    line = dict(proof_lines())["output"]
    assert "the model's replies were 100 tokens shorter per request under compression" in line
    assert "not --shape-output" in line
    assert "95% CI" in line


def test_output_line_refuses_a_direction_when_the_ci_straddles_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    p = tmp_path / "shadow.jsonl"
    with p.open("a", encoding="utf-8") as f:
        for i in range(60):
            f.write(
                json.dumps(
                    {
                        "equivalent": True,
                        "aa_equal": True,
                        "ts": time.time(),
                        "kind": "paired",
                        "sig": SIG_VERSION,
                        "in_a": 1000,
                        "in_b": 500,
                        "out_a": 100,
                        "out_b": 100 + (20 if i % 2 else -20),
                    }
                )
                + "\n"
            )
    from distil.proof_ledger import proof_lines

    line = dict(proof_lines())["output"]
    assert line.startswith("no measurable effect on reply length (n=60,")
    assert bootstrap_ci([20, -20] * 30)[0] < 0 < bootstrap_ci([20, -20] * 30)[1]


def test_receipt_line_reports_verified_then_broken(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import receipts as _r
    from distil.proof_ledger import proof_lines

    for i in range(3):
        _r.append(
            _r.Receipt(
                ts=time.time(),
                request_id=f"r{i}",
                session="s1",
                model="claude-opus-4-8",
                mode="digest",
                tokens_original=100,
                tokens_compressed=50,
                reversible=False,
            )
        )
    assert dict(proof_lines())["receipts"] == "3 receipts, chain verified (every hash re-checked)"

    path = _r.receipts_path()
    rows = path.read_text(encoding="utf-8").splitlines()
    bad = json.loads(rows[1])
    bad["tokens_compressed"] = 1  # one edited field
    rows[1] = json.dumps(bad, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert dict(proof_lines())["receipts"].startswith("chain BROKEN at receipt 1 of 3")


def test_ledger_survives_a_broken_verdict(tmp_path, monkeypatch):
    """A statistic that cannot be computed drops its own line, not the accounting."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    import distil.proof_ledger as pl

    monkeypatch.setattr(pl, "proof_lines", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert pl._safe_proof_lines() == []


def test_one_broken_verdict_drops_only_its_own_line(tmp_path, monkeypatch):
    """The docstring has always claimed this; now it is true. A corrupt artifact behind
    one statistic must not cost the other three, which read different files."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    _write_paired(tmp_path, [0] * 60, out_a=200, out_b=100)
    import distil.proof_ledger as pl

    def _boom(_diffs):
        raise RuntimeError("unreadable drift state")

    monkeypatch.setattr(pl, "_drift_line", _boom)
    labels = dict(pl.proof_lines())
    assert "budget" not in labels, "a verdict that cannot be computed must not print"
    assert "risk" in labels and "output" in labels and "receipts" in labels
    assert "95% conformal bound" in labels["risk"]


def test_incremental_receipt_verification_matches_a_full_pass(tmp_path, monkeypatch):
    """The cached prefix may only make the answer cheaper, never different."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import receipts as _r

    for i in range(5):
        _r.append(_r.Receipt(1.0 + i, f"r{i}", "s", "m", "digest", 10, 5, False))
    first = _r.verify(full=False)  # cold: full pass, writes the checkpoint
    assert first.ok and first.total == 5 and first.checked_from == 0

    _r.append(_r.Receipt(9.0, "r5", "s", "m", "digest", 10, 5, False))
    warm = _r.verify(full=False)  # resumed: only the new receipt re-hashed
    assert warm.ok and warm.total == 6 and warm.checked_from == 5
    assert _r.verify(full=True) == _r.Verdict(6, True, -1, "", 0)


def test_a_stale_checkpoint_cannot_print_broken_over_a_healthy_chain(tmp_path, monkeypatch):
    """Archive the chain and start a new one: the resume point now names bytes that are
    gone, or bytes that belong to a different chain. Either way the answer is the truth
    about the file on disk, reached by re-verifying in full."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import receipts as _r

    for i in range(6):
        _r.append(_r.Receipt(1.0 + i, f"r{i}", "s", "m", "digest", 10, 5, False))
    assert _r.verify().ok
    _r.receipts_path().unlink()  # archived by hand; the checkpoint survives
    for i in range(3):
        _r.append(_r.Receipt(50.0 + i, f"n{i}", "s", "m", "digest", 10, 5, False))

    v = _r.verify()
    assert v.ok, v.statement
    assert v.total == 3, "the verdict must describe the file that exists"


def test_a_tampered_prefix_is_still_caught_when_the_checkpoint_is_trusted(tmp_path, monkeypatch):
    """The resume point is validated by re-reading the receipt it claims to have
    verified. Editing that receipt invalidates the cache and forces the full pass."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import receipts as _r

    for i in range(4):
        _r.append(_r.Receipt(1.0 + i, f"r{i}", "s", "m", "digest", 10, 5, False))
    assert _r.verify().ok

    path = _r.receipts_path()
    rows = path.read_text(encoding="utf-8").splitlines()
    bad = json.loads(rows[3])
    bad["tokens_compressed"] = 1
    rows[3] = json.dumps(bad, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    v = _r.verify()
    assert not v.ok and v.first_bad_index == 3, v.statement
    assert v.total == 4


def test_the_resumed_pass_names_its_own_boundary(tmp_path, monkeypatch):
    """An edit inside the already-verified prefix is NOT found by a resumed pass, and IS
    found by --full. That is the documented contract, so it is tested rather than left as
    a hole that looks closed. The same trust boundary the chain always had: whoever can
    rewrite the receipts can rewrite the checkpoint beside them."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import receipts as _r

    for i in range(6):
        _r.append(_r.Receipt(1.0 + i, f"r{i}", "s", "m", "digest", 10, 5, False))
    assert _r.verify(full=False).ok  # checkpoint now covers all six

    path = _r.receipts_path()
    rows = path.read_text(encoding="utf-8").splitlines()
    bad = json.loads(rows[1])  # deep in the prefix, NOT the receipt the resume validates
    bad["tokens_compressed"] = 1
    rows[1] = json.dumps(bad, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert _r.verify(full=False).ok, "the resumed pass does not re-hash the prefix — by design"
    full = _r.verify(full=True)
    assert not full.ok and full.first_bad_index == 1, full.statement
    assert "`distil receipts` re-hashes all" in _r.verify(full=False).statement, (
        "the fast path must name what it skipped"
    )


def test_a_corrupt_checkpoint_falls_back_to_the_full_pass(tmp_path, monkeypatch):
    """The resume point is a file on disk that nothing promises is well-formed — a torn
    write, a half-truncated JSON object, a hand edit. It is read on every wrap exit, so
    garbage there must cost a re-scan, never a verdict and never a traceback."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import receipts as _r

    for i in range(4):
        _r.append(_r.Receipt(1.0 + i, f"r{i}", "s", "m", "digest", 10, 5, False))
    assert _r.verify().ok

    for junk in ("{not json", "[]", '{"count": "many", "head": null}', ""):
        _r._checkpoint_path().write_text(junk, encoding="utf-8")
        v = _r.verify()
        assert v.ok and v.total == 4, f"{junk!r} broke the verdict: {v.statement}"
        assert v.checked_from == 0, "garbage must not be trusted as a resume point"
