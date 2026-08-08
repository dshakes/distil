"""`distil stats` must not let a lifetime figure be read as the current rate.

This was found by validating the published adoption numbers against this repo's own
ledger. Both were arithmetically correct and both were misleading in the same way:

    lifetime   −20.4%   over 11,500 runs
    last 7d     −0.4%   over  1,979 runs

Two orders of magnitude apart, and only the lifetime number was shown. The cause is
not a bug — a subscription session defaults to lossless-only, so no Tier-1 digest is
left unrecoverable — but a reader looking at `−20.4%` concludes distil is compressing
their traffic by a fifth, and it is not.

A cumulative is history. Printing it alone, next to nothing, is how a true number
tells a false story.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time


def _write_ledger(home, rows: list[tuple[float, int, int]]) -> None:
    """rows: (ts, baseline_tokens, distil_tokens)."""
    path = home / "savings.jsonl"
    with path.open("w") as f:
        for ts, base, dist in rows:
            f.write(
                json.dumps(
                    {
                        "ts": ts,
                        "acct": "2",
                        "model": "claude-opus-4-8",
                        "tokenizer": "heuristic",
                        "session": "s-test",
                        "trajectory_id": "t1",
                        "turns": 3,
                        "baseline_input_tokens": base,
                        "distil_input_tokens": dist,
                        "baseline_dollars": 0.0,
                        "distil_dollars": 0.0,
                    }
                )
                + "\n"
            )


def _stats(home) -> str:
    env = dict(os.environ, DISTIL_HOME=str(home))
    out = subprocess.run(
        [sys.executable, "-m", "distil.cli", "stats"], capture_output=True, text=True, env=env
    )
    return out.stdout


class TestRecentWindowIsShownWhenItDisagrees:
    def test_a_collapse_in_compression_is_surfaced(self, tmp_path) -> None:
        now = time.time()
        old = [(now - 40 * 86400, 1_000_000, 400_000)] * 20  # −60% long ago
        recent = [(now - 2 * 86400, 1_000_000, 996_000)] * 20  # −0.4% now
        _write_ledger(tmp_path, old + recent)
        out = _stats(tmp_path)
        assert "last 7 days" in out, "a divergent recent window must be shown"
        assert "history" in out, "the lifetime figure must be labelled as history"

    def test_the_remedy_is_named_when_compression_is_near_zero(self, tmp_path) -> None:
        now = time.time()
        _write_ledger(
            tmp_path,
            [(now - 40 * 86400, 1_000_000, 400_000)] * 20
            + [(now - 2 * 86400, 1_000_000, 999_000)] * 20,
        )
        out = _stats(tmp_path)
        assert "--expand" in out, "a user at ~0% needs to be told what restores compression"
        assert "lossless-only" in out, "and why it is happening"

    def test_a_steady_ledger_stays_quiet(self, tmp_path) -> None:
        """The line must not fire when recent and lifetime agree, or it is noise
        that gets tuned out exactly when it matters."""
        now = time.time()
        rows = [(now - 40 * 86400, 1_000_000, 400_000)] * 20
        rows += [(now - 2 * 86400, 1_000_000, 400_000)] * 20
        _write_ledger(tmp_path, rows)
        out = _stats(tmp_path)
        assert "last 7 days" not in out

    def test_an_empty_recent_window_does_not_crash_or_claim(self, tmp_path) -> None:
        now = time.time()
        _write_ledger(tmp_path, [(now - 40 * 86400, 1_000_000, 400_000)] * 5)
        out = _stats(tmp_path)
        assert "last 7 days" not in out
        assert "runs recorded" in out, "stats must still render"


class TestOutputSurvivesALegacyConsole:
    """`distil stats` crashed on a Windows console instead of degrading.

    The savings line contains `→`, and `errors="strict"` on a cp1252 stream turns
    that into a UnicodeEncodeError mid-render: exit 1, a traceback, and output cut
    off after `runs recorded`. It only triggered once a ledger had baseline tokens,
    and no test had ever written one — so the line never executed off UTF-8 and CI
    was green on Windows for the whole life of the bug.

    A reporting tool must never fail on the report.
    """

    def _run(self, home, encoding: str) -> subprocess.CompletedProcess[str]:
        now = time.time()
        _write_ledger(
            home,
            [(now - 40 * 86400, 1_000_000, 400_000)] * 20
            + [(now - 2 * 86400, 1_000_000, 996_000)] * 20,
        )
        env = dict(os.environ, DISTIL_HOME=str(home), PYTHONIOENCODING=encoding)
        return subprocess.run(
            [sys.executable, "-m", "distil.cli", "stats"],
            capture_output=True,
            text=True,
            env=env,
            # Decode with the SAME encoding the child was told to write. Without this
            # the parent falls back to its own locale — cp1252 on a Windows runner —
            # so a UTF-8 child's `—` arrives as `â€”` and the glyph assertion fails on
            # a report that rendered perfectly. That is the test mis-reading the
            # output, not the CLI mis-writing it.
            encoding=encoding,
            errors="replace",
        )

    def test_cp1252_console_renders_the_whole_report(self, tmp_path) -> None:
        r = self._run(tmp_path, "cp1252")
        assert r.returncode == 0, f"stats crashed on a legacy console:\n{r.stderr[-600:]}"
        assert "total tokens saved" in r.stdout, "render stopped early"
        assert "last 7 days" in r.stdout

    def test_pure_ascii_console_renders_the_whole_report(self, tmp_path) -> None:
        r = self._run(tmp_path, "ascii")
        assert r.returncode == 0, f"stats crashed on an ascii console:\n{r.stderr[-600:]}"
        assert "total tokens saved" in r.stdout

    def test_utf8_is_unaffected(self, tmp_path) -> None:
        r = self._run(tmp_path, "utf-8")
        assert r.returncode == 0
        assert "→" in r.stdout, "utf-8 must still get the real glyphs"


class TestTheSubscriptionDefaultStatesItsCost:
    """The safe default silently costs the entire product's value.

    A subscription session runs lossless-only so no digest is left unrecoverable —
    correct. But it is Tier-0 only, which measured ~0.4% on this repo's own ledger
    against 30-60% for the recoverable digest. The old notice named `--expand` without
    saying what not using it costs, and never mentioned the PERSISTENT opt-in, so the
    remedy had to be remembered on every invocation.

    A default that quietly turns the product off has to say so.
    """

    def test_the_notice_quantifies_the_cost_and_names_the_persistent_fix(
        self, monkeypatch, capsys
    ) -> None:
        """Asserted on what the user actually sees, not on the function's source.

        Matching source text let the message be rewritten freely as long as the
        old phrases survived somewhere in it — including inside a comment, which
        is where "What this costs" ended up. What has to hold is a property of
        the OUTPUT: it names the cost and the permanent fix.
        """
        import argparse

        from distil import cli

        monkeypatch.setattr("distil.doctor.subscription_mode", lambda: True)
        args = argparse.Namespace(lossless_only=False, verbatim=False, expand=False)
        cli._apply_subscription_safe_default(args)

        notice = capsys.readouterr().err
        assert args.lossless_only is True, "the safe default must still be applied"
        assert "0-2%" in notice, "the notice must quantify what the default costs"
        assert "distil default --mode expand" in notice, "and name the persistent opt-in"
        assert "--expand" in notice, "and the per-run escape"

    def test_the_notice_stays_short_enough_to_be_read(self, monkeypatch, capsys) -> None:
        """It prints on EVERY bare wrap, so length is a correctness property.

        The original was seven lines. A wall of text printed on every invocation
        is skipped, which cost the two facts that matter — savings are ~0, and one
        command fixes it — the readership the notice exists for.
        """
        import argparse

        from distil import cli

        monkeypatch.setattr("distil.doctor.subscription_mode", lambda: True)
        cli._apply_subscription_safe_default(
            argparse.Namespace(lossless_only=False, verbatim=False, expand=False)
        )
        lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
        assert len(lines) <= 3, f"the notice grew back to {len(lines)} lines:\n" + "\n".join(lines)

    def test_the_persistent_mode_is_a_real_choice(self) -> None:
        """`distil default --mode expand` must actually be accepted, or the notice
        sends users to a command that errors."""
        from distil.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["default", "--mode", "expand"])
        assert args.mode == "expand"


class TestNegativeSavingsAreNotMisreported:
    """A window can come out LARGER than baseline, and the report said `−-10.0%`.

    A garbled number is worse than none — the reader cannot tell whether it means
    -10% or +10%, which are opposite outcomes. Worse, the advice branch treated any
    trim below 5% as "lossless-only, add --expand", which is exactly wrong when the
    problem is net expansion: `--expand` would add more, not less.
    """

    def _expansion(self, home) -> str:
        now = time.time()
        _write_ledger(
            home,
            [(now - 40 * 86400, 1_000_000, 400_000)] * 20
            + [(now - 2 * 86400, 1_000_000, 1_100_000)] * 20,
        )
        return _stats(home)

    def test_the_sign_is_rendered_correctly(self, tmp_path) -> None:
        out = self._expansion(tmp_path)
        assert "−-" not in out, "a doubled sign makes the number unreadable"
        assert "LARGER" in out, "expansion must be named as such"

    def test_expand_is_not_recommended_for_expansion(self, tmp_path) -> None:
        out = self._expansion(tmp_path)
        assert "--expand" not in out, "--expand would add more overhead, not less"
        assert "overhead" in out, "the actual cause must be named"

    def test_a_normal_saving_window_still_reads_as_a_saving(self, tmp_path) -> None:
        now = time.time()
        _write_ledger(
            tmp_path,
            [(now - 40 * 86400, 1_000_000, 400_000)] * 20
            + [(now - 2 * 86400, 1_000_000, 996_000)] * 20,
        )
        out = _stats(tmp_path)
        assert "LARGER" not in out and "--expand" in out
