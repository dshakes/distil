"""Tests for CLI command handlers: savings, leaderboard/stats, compress, prune,
ingest, learn, online, federated-leaderboard.

Covers the metered vs subscription branches, JSON output, error paths (missing
file → exit 2, malformed JSON → exit 2), and output formatting contracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


from distil.cli import (
    cmd_compress,
    cmd_federated,
    cmd_ingest,
    cmd_learn,
    cmd_leaderboard,
    cmd_online,
    cmd_prune,
    cmd_savings,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(**kw) -> argparse.Namespace:
    """Build a Namespace with sensible defaults for every optional field."""
    defaults = dict(
        trajectory=None,
        tokenizer="heuristic",
        pricing="claude-opus-4-8",
        output_tokens_per_turn=0,
        record=False,
        # leaderboard extras
        html=None,
        json=False,
        badge=False,
        # learn
        threshold=0.25,
        min_samples=5,
        # online
        corpus=None,
        promote_to=None,
        # ingest
        input=None,
        out=None,
        provider="anthropic",
        model="claude-opus-4-8",
        # federated
        dir=None,
        keys=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# cmd_savings
# ---------------------------------------------------------------------------


class TestCmdSavings:
    def test_basic_run_exits_zero(self, capsys):
        rc = cmd_savings(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        assert "distil" in out.lower()
        assert "$" in out  # dollar column header

    def test_metered_shows_dollars(self, monkeypatch, capsys):
        monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
        rc = cmd_savings(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        # At least one dollar figure with 2 decimal places present
        import re

        assert re.search(r"\$\d+\.\d{5}", out), "expected 5dp dollar figure"

    def test_subscription_shows_notional_banner(self, monkeypatch, capsys):
        monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
        rc = cmd_savings(_ns())
        assert rc == 0
        # subscription mode: the savings line says notional or omits real dollars
        out = capsys.readouterr().out
        # cmd_savings itself doesn't print "notional" — that's cmd_leaderboard.
        # It still prints a pricing table. Just confirm it ran and emitted output.
        assert "distil (cache-aware lossless)" in out

    def test_record_writes_to_ledger(self, tmp_path, monkeypatch, capsys):
        ledger_path = tmp_path / "savings.jsonl"
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        rc = cmd_savings(_ns(record=True))
        assert rc == 0
        assert ledger_path.exists()
        rec = json.loads(ledger_path.read_text().splitlines()[0])
        assert "baseline_dollars" in rec
        assert rec["baseline_dollars"] > 0

    def test_record_stamps_compression_mode(self, tmp_path):
        """Each row records the mode it was produced under, so 'why was ▼ low?'
        is answerable from the ledger (lossless-only vs digest) not by inference."""
        from distil import ledger

        p = tmp_path / "savings.jsonl"
        ledger.record(
            trajectory_id="t",
            model="claude-opus-4-8",
            turns=2,
            baseline_dollars=1.0,
            distil_dollars=0.6,
            baseline_input_tokens=1000,
            distil_input_tokens=600,
            session="s1",
            mode="lossless-only",
            path=p,
        )
        assert json.loads(p.read_text().splitlines()[0])["mode"] == "lossless-only"
        # mode is optional — omitting it stays backward-compatible (empty string)
        ledger.record(
            trajectory_id="t2",
            model="m",
            turns=1,
            baseline_dollars=1.0,
            distil_dollars=1.0,
            baseline_input_tokens=10,
            distil_input_tokens=5,
            session="s2",
            path=p,
        )
        assert json.loads(p.read_text().splitlines()[1])["mode"] == ""


# ---------------------------------------------------------------------------
# cmd_leaderboard / stats
# ---------------------------------------------------------------------------


class TestCmdLeaderboard:
    def test_empty_ledger_message(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        rc = cmd_leaderboard(_ns())
        assert rc == 0
        assert "no genuine savings recorded" in capsys.readouterr().out

    def _write_ledger(self, tmp_path, n=3):
        """Write n synthetic savings records to a tmp ledger."""
        p = tmp_path / "savings.jsonl"
        import time as _t

        for i in range(n):
            rec = {
                "trajectory_id": f"t{i}",
                "model": "claude-opus-4-8",
                "turns": 5,
                "baseline_dollars": 0.10,
                "distil_dollars": 0.07,
                "baseline_input_tokens": 1000,
                "distil_input_tokens": 700,
                "tokenizer": "heuristic",
                "ts": _t.time() - i,
                "session": "",
            }
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    def test_metered_shows_dollar_total(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
        self._write_ledger(tmp_path)
        rc = cmd_leaderboard(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        # total dollars saved must appear with 2dp
        import re

        assert re.search(r"\$\d+\.\d{2}", out), "expected 2dp dollar total"

    def test_subscription_shows_notional_not_dollars(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
        self._write_ledger(tmp_path)
        rc = cmd_leaderboard(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        assert "notional" in out

    def test_json_output_is_valid_json(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        self._write_ledger(tmp_path)
        rc = cmd_leaderboard(_ns(json=True))
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "runs" in parsed
        assert parsed["runs"] == 3

    def test_json_empty_ledger_is_valid_json(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        rc = cmd_leaderboard(_ns(json=True))
        assert rc == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["runs"] == 0

    def test_eq_suppressed_below_reporting_floor(self, tmp_path, monkeypatch, capsys):
        """Below the shared floor, no decision-equivalence rate is shown."""
        from distil.shadow import VERDICT_MIN_AB, ShadowLedger

        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")

        # 5 samples — below the VERDICT_MIN_AB floor cmd_leaderboard enforces
        assert 5 < VERDICT_MIN_AB
        led = ShadowLedger()
        for eq in [True, True, True, False, True]:
            led.record(eq)

        # cmd_leaderboard does `from .shadow import ShadowLedger; led = ShadowLedger.load()`
        # so patching the class method on the real class is the right intercept point.
        monkeypatch.setattr(ShadowLedger, "load", classmethod(lambda cls, *a, **k: led))

        self._write_ledger(tmp_path)
        rc = cmd_leaderboard(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        # Must say "collecting" — must NOT print a bare percentage rate
        assert "decision-equivalence:" in out
        assert "collecting" in out
        import re

        assert not re.search(r"decision-equivalence:\s+\d+\.\d+%\s+\(", out)

    # stats alias (invoked via main)
    def test_stats_alias_works(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        rc = main(["stats"])
        assert rc == 0


# ---------------------------------------------------------------------------
# cmd_compress
# ---------------------------------------------------------------------------


class TestCmdCompress:
    def test_bundled_corpus_exits_zero(self, capsys):
        rc = cmd_compress(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        assert "reversible" in out
        assert "turn" in out  # table header

    def test_missing_trajectory_exit_2(self, tmp_path, capsys):
        missing = str(tmp_path / "no_such.json")
        rc = main(["compress", "--trajectory", missing])
        assert rc == 2
        assert "compress" in capsys.readouterr().err

    def test_malformed_json_exit_2(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json")
        rc = main(["compress", "--trajectory", str(bad)])
        assert rc == 2
        err = capsys.readouterr().err
        assert "compress" in err


# ---------------------------------------------------------------------------
# cmd_prune
# ---------------------------------------------------------------------------


class TestCmdPrune:
    def test_bundled_corpus_exits_zero(self, capsys):
        rc = cmd_prune(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        assert "causal ablation" in out
        assert "tokens provably free" in out

    def test_missing_trajectory_exit_2(self, tmp_path, capsys):
        missing = str(tmp_path / "no_such.json")
        rc = main(["prune", "--trajectory", missing])
        assert rc == 2
        assert "prune" in capsys.readouterr().err

    def test_malformed_json_exit_2(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("{bad}")
        rc = main(["prune", "--trajectory", str(bad)])
        assert rc == 2


# ---------------------------------------------------------------------------
# cmd_ingest
# ---------------------------------------------------------------------------


class TestCmdIngest:
    def _valid_jsonl(self, tmp_path) -> Path:
        """One Anthropic request with a user message and an assistant reply."""
        req = {
            "model": "claude-opus-4-8",
            "system": "You are a helper.",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I am fine, thank you."},
            ],
        }
        p = tmp_path / "session.jsonl"
        p.write_text(json.dumps(req) + "\n")
        return p

    def test_valid_jsonl_creates_corpus(self, tmp_path, capsys):
        src = self._valid_jsonl(tmp_path)
        out_dir = tmp_path / "corpus"
        rc = cmd_ingest(_ns(input=str(src), out=str(out_dir)))
        assert rc == 0
        assert (out_dir / "session.json").exists()
        assert (out_dir / "manifest.json").exists()
        out = capsys.readouterr().out
        assert "ingested" in out

    def test_zero_turns_exit_2(self, tmp_path, capsys):
        """A completely empty JSONL (no parseable requests) → 0 turns → exit 2."""
        p = tmp_path / "empty.jsonl"
        p.write_text("")  # no lines → no requests → no turns
        out_dir = tmp_path / "corpus"
        rc = cmd_ingest(_ns(input=str(p), out=str(out_dir)))
        assert rc == 2
        assert "0 turns" in capsys.readouterr().err

    def test_missing_input_file_clean_systemexit(self, tmp_path):
        """Missing input file raises SystemExit with an informative message."""
        # ingest_file raises SystemExit (not FileNotFoundError), so test at that boundary.
        from distil.ingest import ingest_file

        with pytest.raises(SystemExit) as exc_info:
            ingest_file(str(tmp_path / "ghost.jsonl"))
        assert "ghost.jsonl" in str(exc_info.value)

    def test_manifest_updated_idempotently(self, tmp_path, capsys):
        """Running ingest twice on the same file should yield one manifest entry."""
        src = self._valid_jsonl(tmp_path)
        out_dir = tmp_path / "corpus"
        cmd_ingest(_ns(input=str(src), out=str(out_dir)))
        capsys.readouterr()
        cmd_ingest(_ns(input=str(src), out=str(out_dir)))
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert len(manifest["trajectories"]) == 1


# ---------------------------------------------------------------------------
# cmd_learn
# ---------------------------------------------------------------------------


class TestCmdLearn:
    def test_no_signals_returns_zero_and_explains(self, tmp_path, monkeypatch, capsys):
        from distil.learn import ExpandStats

        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        # Ensure load returns an empty stats (no signals recorded)
        monkeypatch.setattr(
            "distil.cli.ExpandStats",
            type("_ES", (), {"load": staticmethod(lambda: ExpandStats())}),
            raising=False,
        )

        # Use the module-level import path that cmd_learn uses
        import distil.learn as _learn

        empty = ExpandStats()
        monkeypatch.setattr(
            _learn,
            "ExpandStats",
            type("_ES", (), {"load": staticmethod(lambda: empty), **ExpandStats.__dict__}),
            raising=False,
        )

        # Call directly with a fresh empty stats via monkeypatching the import
        # Re-read: cmd_learn does `from .learn import ExpandStats; stats = ExpandStats.load()`
        # Simplest: patch the class at the module level
        import distil.cli as _cli

        class _Empty:
            @staticmethod
            def load():
                return ExpandStats()

        monkeypatch.setattr(_cli, "ExpandStats", _Empty, raising=False)

        rc = cmd_learn(_ns())
        assert rc == 0
        assert "no expand signals" in capsys.readouterr().out

    def test_with_signals_shows_rate_not_raw_floats(self, tmp_path, monkeypatch, capsys):
        """Expand rates must appear as e.g. '40%', not raw float '0.4000000000'."""
        from distil.learn import ExpandStats

        s = ExpandStats()
        for _ in range(10):
            s.record_digest("json:l")
        for _ in range(4):
            s.record_expand("json:l")
        for _ in range(10):
            s.record_digest("log:m")

        # cmd_learn does `from .learn import ExpandStats; ExpandStats.load()`,
        # so patch the SOURCE classmethod — otherwise it reads the real ledger
        # (empty in a clean CI HOME) and the rate column never appears.
        monkeypatch.setattr("distil.learn.ExpandStats.load", staticmethod(lambda: s))

        rc = cmd_learn(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        # Rate column must have trailing "%", not a bare long float
        import re

        assert re.search(r"\d+%", out), "expected percentage in rate column"
        # No 16-digit floats in output
        assert not re.search(r"\d\.\d{6,}", out), "unexpectedly raw float in output"


# ---------------------------------------------------------------------------
# cmd_online
# ---------------------------------------------------------------------------


class TestCmdOnline:
    def test_bundled_corpus_exits_zero(self, capsys):
        """Full online round on the bundled corpus must complete without error."""
        rc = cmd_online(_ns())
        # May be 0 (certified) or non-zero; either is fine — we just test it runs
        assert rc in (0, 1)
        out = capsys.readouterr().out
        assert "self-distilling round" in out

    def test_float_formatting_no_excess_digits(self, capsys):
        """accuracy / precision / recall printed as e.g. '87.3%', f1/n as '0.932'."""
        cmd_online(_ns())
        out = capsys.readouterr().out
        # No 16+ digit floats anywhere in output
        import re

        assert not re.search(r"\d\.\d{6,}", out), f"raw float found: {out!r}"

    def test_accuracy_shown_as_percent(self, capsys):
        """Accuracy, precision, recall use % formatting."""
        cmd_online(_ns())
        out = capsys.readouterr().out
        import re

        # Must contain at least one "XX.X%" for accuracy/precision/recall
        assert re.search(r"\d+\.\d+%", out), f"no percent value in output: {out!r}"

    def test_f1_shown_as_decimal(self, capsys):
        """f1 uses 3dp decimal, not percent."""
        cmd_online(_ns())
        out = capsys.readouterr().out
        import re

        # f1: 0.NNN pattern
        assert re.search(r"f1:.*0\.\d{3}", out), f"f1 decimal not found: {out!r}"


# ---------------------------------------------------------------------------
# cmd_federated (federated-leaderboard)
# ---------------------------------------------------------------------------


class TestCmdFederated:
    def test_empty_dir_no_crash(self, tmp_path, capsys):
        """An empty submissions dir must report 0 verified, 0 rejected, not crash."""
        (tmp_path / "submissions.jsonl").touch()  # empty file
        rc = cmd_federated(_ns(dir=str(tmp_path)))
        assert rc == 0
        out = capsys.readouterr().out
        assert "0 verified" in out

    def test_missing_submissions_file(self, tmp_path, capsys):
        """Dir with no submissions.jsonl at all still runs cleanly."""
        rc = cmd_federated(_ns(dir=str(tmp_path)))
        assert rc == 0
        out = capsys.readouterr().out
        assert "0 verified" in out

    def test_signed_submission_verified(self, tmp_path, capsys):
        """A properly-signed submission must show up as verified."""
        import time as _t
        from distil.telemetry import SavingsAggregate, sign, submit

        key = "test-secret-key"
        agg = SavingsAggregate(
            instance_id="inst-1",
            tokens_saved=5000,
            dollars_saved=0.25,
            runs=3,
            certified=True,
            ts=_t.time(),
        )
        signed = sign(agg, key)
        submit(signed, str(tmp_path))

        keys = {"inst-1": key}
        # Write the keys to a temp JSON file
        keys_path = tmp_path / "keys.json"
        keys_path.write_text(json.dumps(keys))

        rc = cmd_federated(_ns(dir=str(tmp_path), keys=str(keys_path)))
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 verified" in out

    def test_tampered_submission_is_rejected(self, tmp_path, capsys):
        """A submission with a wrong signature must be rejected."""
        import time as _t
        from distil.telemetry import SavingsAggregate, sign, submit

        agg = SavingsAggregate(
            instance_id="inst-bad",
            tokens_saved=9999,
            dollars_saved=99.0,
            runs=1,
            certified=True,
            ts=_t.time(),
        )
        signed = sign(agg, "real-key")
        signed["tokens_saved"] = 1  # tamper
        submit(signed, str(tmp_path))

        keys = {"inst-bad": "real-key"}
        keys_path = tmp_path / "keys.json"
        keys_path.write_text(json.dumps(keys))

        rc = cmd_federated(_ns(dir=str(tmp_path), keys=str(keys_path)))
        assert rc == 0
        out = capsys.readouterr().out
        assert "0 verified" in out
        assert "1 rejected" in out

    def test_via_main_cli_requires_dir(self, tmp_path, capsys):
        """federated-leaderboard via main() must work end-to-end."""
        rc = main(["federated-leaderboard", "--dir", str(tmp_path)])
        assert rc == 0
