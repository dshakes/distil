"""CLI command handler tests: certify / conformal / verify / calibrate / holdout /
certify-trajectories / bench / eval / frontier / benchmark / perf / output-savings.

Each test exercises *behavior*: real output substrings, correct exit codes, and
branching paths (PASS vs FAIL, certified vs not, error vs success).  Nothing
here tests the framework or the mocks — only distil's own decision-making.

Run:
    uv run --with pytest python -m pytest tests/test_cli_certify.py -q
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from distil.cli import (
    cmd_bench,
    cmd_benchmark,
    cmd_calibrate,
    cmd_certify,
    cmd_certify_trajectories,
    cmd_conformal,
    cmd_eval,
    cmd_frontier,
    cmd_holdout,
    cmd_output_savings,
    cmd_perf,
    cmd_verify,
    main,
)


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# ---------------------------------------------------------------------------
# certify
# ---------------------------------------------------------------------------


class TestCmdCertify:
    def _ns(self, strategy: str = "distil") -> argparse.Namespace:
        return _ns(
            trajectory=None,
            strategy=strategy,
            runner="deterministic",
            margin=0.02,
            alpha=0.05,
        )

    def test_distil_strategy_passes(self, capsys):
        rc = cmd_certify(self._ns("distil"))
        out = capsys.readouterr().out
        assert rc == 0
        assert "PASS" in out

    def test_aggressive_strategy_fails(self, capsys):
        rc = cmd_certify(self._ns("aggressive"))
        out = capsys.readouterr().out
        assert rc == 1
        assert "FAIL" in out

    def test_pass_reports_100_pct_match_rate(self, capsys):
        cmd_certify(self._ns("distil"))
        out = capsys.readouterr().out
        assert "100.0%" in out

    def test_fail_reports_zero_match_rate(self, capsys):
        cmd_certify(self._ns("aggressive"))
        out = capsys.readouterr().out
        assert "0.0%" in out

    def test_output_includes_tost_line(self, capsys):
        cmd_certify(self._ns("distil"))
        out = capsys.readouterr().out
        assert "TOST non-inferiority" in out


# ---------------------------------------------------------------------------
# conformal
# ---------------------------------------------------------------------------


class TestCmdConformal:
    def _ns(self, alpha: float = 0.30, delta: float = 0.10) -> argparse.Namespace:
        return _ns(
            corpus=None,
            runner="deterministic",
            alpha=alpha,
            delta=delta,
            method="ltt",
            samples=3,
        )

    def test_certifies_at_generous_alpha(self, capsys):
        rc = cmd_conformal(self._ns(alpha=0.30))
        out = capsys.readouterr().out
        assert rc == 0
        assert "CERTIFIED" in out

    def test_not_certified_at_very_tight_alpha(self, capsys):
        # small corpus (~28 turns) can't support a 1% guarantee
        rc = cmd_conformal(self._ns(alpha=0.01))
        out = capsys.readouterr().out
        assert rc == 0
        assert "NOT CERTIFIED" in out

    def test_output_shows_method_and_risk_target(self, capsys):
        cmd_conformal(self._ns())
        out = capsys.readouterr().out
        assert "LTT" in out
        assert "α = 0.3" in out

    def test_certified_output_reports_savings(self, capsys):
        cmd_conformal(self._ns(alpha=0.30))
        out = capsys.readouterr().out
        assert "token savings" in out


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


class TestCmdVerify:
    def test_corpus_passes_fidelity_gate(self, capsys):
        rc = cmd_verify(_ns())
        out = capsys.readouterr().out
        assert rc == 0
        assert "FIDELITY: PASS" in out

    def test_output_mentions_reversible(self, capsys):
        cmd_verify(_ns())
        out = capsys.readouterr().out
        assert "byte-reversible" in out

    def test_output_mentions_append_only(self, capsys):
        cmd_verify(_ns())
        out = capsys.readouterr().out
        assert "append-only" in out


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


class TestCmdCalibrate:
    def _scores(self, tmp_path: Path, data: dict, name: str = "scores.json") -> str:
        p = tmp_path / name
        p.write_text(json.dumps(data))
        return str(p)

    def test_fail_safe_when_candidate_too_lossy(self, tmp_path, capsys):
        # ~33% loss rate — clearly non-inferior to baseline (margin=5%)
        base = {f"t{i}": True for i in range(40)}
        cand = {f"t{i}": (i % 3 != 0) for i in range(40)}
        ns = _ns(
            baseline=self._scores(tmp_path, base, "base.json"),
            candidate=[f"gate@6={self._scores(tmp_path, cand, 'cand.json')}:6"],
            margin=0.05,
            json=None,
        )
        rc = cmd_calibrate(ns)
        out = capsys.readouterr().out
        assert rc == 0
        assert "FAIL-SAFE" in out

    def test_selects_when_candidate_is_non_inferior(self, tmp_path, capsys):
        # identical base and candidate — must be selected
        base = {f"t{i}": True for i in range(40)}
        ns = _ns(
            baseline=self._scores(tmp_path, base, "base.json"),
            candidate=[f"gate@12={self._scores(tmp_path, base, 'cand.json')}:12"],
            margin=0.05,
            json=None,
        )
        rc = cmd_calibrate(ns)
        out = capsys.readouterr().out
        assert rc == 0
        assert "SELECTED" in out
        assert "gate@12" in out

    def test_bad_candidate_spec_exits_2(self, tmp_path, capsys):
        base = {f"t{i}": True for i in range(10)}
        ns = _ns(
            baseline=self._scores(tmp_path, base),
            candidate=["no-equals-sign"],
            margin=0.05,
            json=None,
        )
        rc = cmd_calibrate(ns)
        assert rc == 2

    def test_writes_json_cert(self, tmp_path, capsys):
        base = {f"t{i}": True for i in range(40)}
        out_path = str(tmp_path / "cert.json")
        ns = _ns(
            baseline=self._scores(tmp_path, base, "base.json"),
            candidate=[f"gate@12={self._scores(tmp_path, base, 'cand.json')}:12"],
            margin=0.05,
            json=out_path,
        )
        cmd_calibrate(ns)
        data = json.loads(Path(out_path).read_text())
        assert "selected" in data


# ---------------------------------------------------------------------------
# holdout
# ---------------------------------------------------------------------------


class TestCmdHoldout:
    def _ns(self, control_fraction: float = 0.2) -> argparse.Namespace:
        return _ns(
            corpus=None,
            tokenizer="heuristic",
            pricing="claude-opus-4-8",
            control_fraction=control_fraction,
        )

    def test_normal_run_exits_0(self, capsys):
        rc = cmd_holdout(self._ns())
        out = capsys.readouterr().out
        assert rc == 0
        assert "holdout" in out.lower()

    def test_output_includes_bootstrap_ci(self, capsys):
        cmd_holdout(self._ns())
        out = capsys.readouterr().out
        assert "CI" in out

    def test_control_fraction_zero_exits_2(self, capsys):
        rc = cmd_holdout(self._ns(control_fraction=0.0))
        assert rc == 2

    def test_control_fraction_one_exits_2(self, capsys):
        rc = cmd_holdout(self._ns(control_fraction=1.0))
        assert rc == 2

    def test_control_fraction_over_one_exits_2(self, capsys):
        rc = cmd_holdout(self._ns(control_fraction=1.5))
        assert rc == 2


# ---------------------------------------------------------------------------
# certify-trajectories
# ---------------------------------------------------------------------------


class TestCmdCertifyTrajectories:
    def _write(self, tmp_path: Path, outcomes: list[dict]) -> str:
        p = tmp_path / "outcomes.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in outcomes))
        return str(p)

    def test_certified_with_100_clean_tasks(self, tmp_path, capsys):
        # 100 tasks, 0 degradation → certifies at α=0.10, δ=0.10
        data = [
            {"task_id": f"t{i}", "full_success": True, "compressed_success": True}
            for i in range(100)
        ]
        rc = cmd_certify_trajectories(
            _ns(outcomes=self._write(tmp_path, data), alpha=0.10, delta=0.10, json=False)
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "True" in out  # certified: True
        assert "certified" in out.lower()

    def test_not_certified_below_min_n(self, tmp_path, capsys):
        # n=4 < min_n=20 — too few samples for any guarantee
        data = [
            {"task_id": f"t{i}", "full_success": True, "compressed_success": True} for i in range(4)
        ]
        rc = cmd_certify_trajectories(
            _ns(outcomes=self._write(tmp_path, data), alpha=0.10, delta=0.10, json=False)
        )
        out = capsys.readouterr().out
        assert rc == 1
        assert "NOT CERTIFIED" in out

    def test_empirical_risk_math(self, tmp_path, capsys):
        # 1 degradation (full=True, compressed=False) out of 4 tasks → 25% risk
        data = [
            {"task_id": "t0", "full_success": True, "compressed_success": False},
            {"task_id": "t1", "full_success": True, "compressed_success": True},
            {"task_id": "t2", "full_success": False, "compressed_success": False},
            {"task_id": "t3", "full_success": True, "compressed_success": True},
        ]
        cmd_certify_trajectories(
            _ns(outcomes=self._write(tmp_path, data), alpha=0.10, delta=0.10, json=False)
        )
        out = capsys.readouterr().out
        assert "25.00%" in out

    def test_json_output_has_required_keys(self, tmp_path, capsys):
        data = [
            {"task_id": f"t{i}", "full_success": True, "compressed_success": True}
            for i in range(100)
        ]
        cmd_certify_trajectories(
            _ns(outcomes=self._write(tmp_path, data), alpha=0.10, delta=0.10, json=True)
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["certified"] is True
        assert "empirical_risk" in payload
        assert "risk_bound" in payload
        assert "n" in payload

    def test_missing_file_exits_2(self, tmp_path):
        missing = str(tmp_path / "no-such-file.jsonl")
        rc = main(["certify-trajectories", missing])
        assert rc == 2

    def test_malformed_json_exits_2(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not valid json}\n")
        rc = main(["certify-trajectories", str(bad)])
        assert rc == 2

    def test_blank_lines_in_jsonl_are_skipped(self, tmp_path, capsys):
        # JSONL with blank lines between entries — parser must skip them (line 1206)
        data = [
            {"task_id": f"t{i}", "full_success": True, "compressed_success": True}
            for i in range(100)
        ]
        p = tmp_path / "outcomes.jsonl"
        p.write_text("\n\n".join(json.dumps(d) for d in data) + "\n\n")
        rc = cmd_certify_trajectories(_ns(outcomes=str(p), alpha=0.10, delta=0.10, json=False))
        out = capsys.readouterr().out
        assert rc == 0
        assert "100" in out  # all 100 trajectories parsed despite blank lines


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


class TestCmdBench:
    def _ns(self, savings_only: bool = False) -> argparse.Namespace:
        return _ns(
            corpus=None,
            tokenizer="heuristic",
            pricing="claude-opus-4-8",
            margin=0.02,
            alpha=0.05,
            record=False,
            savings_only=savings_only,
        )

    def test_gate_passes_on_bundled_corpus(self, capsys):
        rc = cmd_bench(self._ns())
        out = capsys.readouterr().out
        assert rc == 0
        assert "GATE: PASS" in out

    def test_distil_certified_on_every_trajectory(self, capsys):
        cmd_bench(self._ns())
        out = capsys.readouterr().out
        # 7 corpus entries, each should have a PASS in the distil column
        assert out.count("PASS") >= 7

    def test_aggressive_rejected_on_every_trajectory(self, capsys):
        cmd_bench(self._ns())
        out = capsys.readouterr().out
        # 7 corpus entries, each should have a FAIL in the aggressive column
        assert out.count("FAIL") >= 7

    def test_savings_only_skips_gate(self, capsys):
        rc = cmd_bench(self._ns(savings_only=True))
        out = capsys.readouterr().out
        assert rc == 0
        assert "GATE: PASS" not in out

    def test_verdict_retention_check_runs(self, capsys):
        rc = cmd_bench(self._ns())
        out = capsys.readouterr().out
        assert rc == 0
        assert "verdict-retention: PASS" in out

    def test_verdict_retention_check_can_fail(self, capsys, monkeypatch):
        # the gate must be real: a digest that drops verdict lines flips it red
        import distil.cli as cli_mod

        def lossy_digest(text, *a, **kw):
            kept = [ln for ln in text.splitlines() if "passed" not in ln]
            return "\n".join(kept), True

        monkeypatch.setattr(cli_mod, "_tier1_digest", lossy_digest)
        rc = cmd_bench(self._ns())
        out = capsys.readouterr().out
        assert rc == 1
        assert "verdict-retention (green)" in out
        assert "GATE: FAIL" in out

    def test_aggregate_savings_positive(self, capsys):
        cmd_bench(self._ns())
        out = capsys.readouterr().out
        assert "aggregate" in out.lower()
        assert "cheaper" in out

    def test_record_appends_to_ledger(self, tmp_path, capsys, monkeypatch):
        # Isolate from the real ledger by redirecting DISTIL_HOME to tmp_path.
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        ns = _ns(
            corpus=None,
            tokenizer="heuristic",
            pricing="claude-opus-4-8",
            margin=0.02,
            alpha=0.05,
            record=True,
            savings_only=False,
        )
        rc = cmd_bench(ns)
        out = capsys.readouterr().out
        assert rc == 0
        assert "recorded" in out.lower()


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


class TestCmdEval:
    def _ns(self, out_dir: str | None = None) -> argparse.Namespace:
        return _ns(corpus=None, runner="deterministic", out=out_dir)

    def test_exits_0_and_shows_frontier_header(self, capsys):
        rc = cmd_eval(self._ns())
        out = capsys.readouterr().out
        assert rc == 0
        assert "certified compression frontier" in out

    def test_distil_certified_in_output(self, capsys):
        cmd_eval(self._ns())
        out = capsys.readouterr().out
        assert "distil" in out
        assert "PASS" in out

    def test_uncertified_truncation_shown(self, capsys):
        cmd_eval(self._ns())
        out = capsys.readouterr().out
        # the aggressive truncation levels are present and marked as failing
        assert "truncate" in out

    def test_writes_raw_jsonl(self, tmp_path, capsys):
        rc = cmd_eval(self._ns(str(tmp_path)))
        assert rc == 0
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".jsonl"


# ---------------------------------------------------------------------------
# frontier
# ---------------------------------------------------------------------------


class TestCmdFrontier:
    def _ns(self, targets: str = "1.0,0.97,0.95") -> argparse.Namespace:
        return _ns(corpus=None, runner="deterministic", samples=3, targets=targets)

    def test_exits_0_and_shows_dial_header(self, capsys):
        rc = cmd_frontier(self._ns())
        out = capsys.readouterr().out
        assert rc == 0
        assert "savings-vs-equivalence dial" in out

    def test_three_targets_produce_three_rows(self, capsys):
        cmd_frontier(self._ns("1.0,0.97,0.95"))
        out = capsys.readouterr().out
        # data rows have leading whitespace and exactly the pattern "  N%  " (target column)
        rows = [
            ln
            for ln in out.splitlines()
            if ln.strip().endswith("%")
            or (
                ln.strip()
                and "%" in ln
                and "---" not in ln
                and "target" not in ln.lower()
                and "At 100%" not in ln
                and "saved" not in ln
            )
        ]
        # at minimum 3 rows (one per target)
        assert len(rows) >= 3

    def test_bad_targets_exits_2(self, capsys):
        rc = cmd_frontier(self._ns("not,valid,targets"))
        assert rc == 2

    def test_output_has_summary_line(self, capsys):
        cmd_frontier(self._ns())
        out = capsys.readouterr().out
        assert "At 100%" in out


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


class TestCmdBenchmark:
    def _ns(self, **kw) -> argparse.Namespace:
        defaults = dict(
            corpus=None,
            runner="deterministic",
            pricing="claude-opus-4-8",
            tokenizer="heuristic",
            margin=0.02,
            alpha=0.05,
            external=None,
            html=None,
            out=None,
        )
        defaults.update(kw)
        return _ns(**defaults)

    def test_exits_0_and_shows_leader(self, capsys):
        rc = cmd_benchmark(self._ns())
        out = capsys.readouterr().out
        assert rc == 0
        assert "LEADER" in out

    def test_distil_leads_on_certified_savings(self, capsys):
        cmd_benchmark(self._ns())
        out = capsys.readouterr().out
        # the winner is always a distil variant
        leader_line = next(ln for ln in out.splitlines() if "LEADER" in ln)
        assert "distil" in leader_line.lower()

    def test_bad_external_spec_exits_2(self, capsys):
        rc = cmd_benchmark(self._ns(external=["no-colon-here"]))
        assert rc == 2

    def test_writes_html(self, tmp_path, capsys):
        html_path = str(tmp_path / "bench.html")
        rc = cmd_benchmark(self._ns(html=html_path))
        assert rc == 0
        assert Path(html_path).exists()

    def test_writes_raw_results_jsonl(self, tmp_path, capsys):
        rc = cmd_benchmark(self._ns(out=str(tmp_path)))
        assert rc == 0
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".jsonl"


# ---------------------------------------------------------------------------
# perf
# ---------------------------------------------------------------------------


class TestCmdPerf:
    def test_normal_run_exits_0(self, capsys):
        rc = cmd_perf(_ns(iterations=5))
        out = capsys.readouterr().out
        assert rc == 0
        assert "p50" in out

    def test_output_includes_ops_per_sec(self, capsys):
        cmd_perf(_ns(iterations=5))
        out = capsys.readouterr().out
        assert "ops/sec" in out

    def test_zero_iterations_exits_2(self, capsys):
        rc = cmd_perf(_ns(iterations=0))
        assert rc == 2

    def test_negative_iterations_exits_2(self, capsys):
        rc = cmd_perf(_ns(iterations=-1))
        assert rc == 2


# ---------------------------------------------------------------------------
# output-savings
# ---------------------------------------------------------------------------


class TestCmdOutputSavings:
    def test_default_fixture_exits_0(self, capsys):
        rc = cmd_output_savings(_ns(input=None))
        out = capsys.readouterr().out
        assert rc == 0
        assert "output compression" in out

    def test_output_mentions_answer_preservation(self, capsys):
        cmd_output_savings(_ns(input=None))
        out = capsys.readouterr().out
        assert "answer-preservation" in out

    def test_custom_valid_jsonl(self, tmp_path, capsys):
        pairs = [
            {
                "baseline": "The quick brown fox jumps over the lazy dog, very quickly.",
                "shaped": "Fox jumped.",
            },
            {
                "baseline": "This is a longer sentence with more tokens inside it.",
                "shaped": "Shorter sentence.",
            },
        ]
        p = tmp_path / "pairs.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in pairs))
        rc = cmd_output_savings(_ns(input=str(p)))
        out = capsys.readouterr().out
        assert rc == 0
        assert "output compression" in out


class TestCmdCertifyPooled:
    """-t <dir> pools every trajectory into ONE TOST (the live-gate unit)."""

    def _ns(self, trajectory: str) -> argparse.Namespace:
        return _ns(
            trajectory=trajectory,
            strategy="distil",
            runner="deterministic",
            margin=0.02,
            alpha=0.05,
        )

    def test_directory_pools_all_trajectories(self, capsys):
        rc = cmd_certify(self._ns("corpus"))
        out = capsys.readouterr().out
        assert rc == 0
        assert "trajectories" in out and "turns pooled" in out
        # divergence lines are trajectory-qualified in pooled mode
        assert "coding-bugfix turn" in out
        assert "PASS" in out

    def test_empty_directory_is_a_clean_error(self, tmp_path, capsys):
        rc = cmd_certify(self._ns(str(tmp_path)))
        out = capsys.readouterr().out
        assert rc == 2
        assert "no trajectory JSON files" in out
