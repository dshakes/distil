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
