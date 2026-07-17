"""Per-content-type keep policy tests.

Covers classify() kind detection, must_keep() per kind, summarize_dropped(),
and digest() integration with TRACEBACK frame preservation and DIFF hunk headers.
"""

from __future__ import annotations

from distil.compress.keep_policy import (
    ContentKind,
    classify,
    must_keep,
    summarize_dropped,
)
from distil.compress.tier1 import digest


# ---- classify ----------------------------------------------------------------


def test_classify_log_vitest():
    text = "\n".join(
        [
            "RUN v1.6.0",
            "✓ auth.test.ts (3 tests) 5ms",
            " Test Files  188 passed (188)",
            "      Tests  1955 passed (1955)",
            " Duration  3.21s",
        ]
    )
    assert classify(text) is ContentKind.LOG


def test_classify_traceback():
    tb = 'Traceback (most recent call last):\n  File "app.py", line 7, in main\nValueError: boom'
    assert classify(tb) is ContentKind.TRACEBACK


def test_classify_diff():
    patch = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,4 @@\n context"
    assert classify(patch) is ContentKind.DIFF


def test_classify_generic():
    assert classify("ordinary prose\nno signals here") is ContentKind.GENERIC


def test_classify_log_wins_over_traceback():
    # A failing test log has both a traceback AND a verdict — LOG wins so the
    # summary is pinned and the frames are caught via the generic error net.
    text = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 7, in main\n'
        "ValueError: boom\n"
        "  Tests  3 failed | 1952 passed (1955)"
    )
    assert classify(text) is ContentKind.LOG


# ---- must_keep — GENERIC -----------------------------------------------------


def test_generic_keeps_decision_and_errors():
    assert must_keep("DECISION: do X", ContentKind.GENERIC)
    assert must_keep("ERROR: boom", ContentKind.GENERIC)
    assert must_keep("warn: low disk", ContentKind.GENERIC)
    assert not must_keep("ordinary prose line", ContentKind.GENERIC)
    assert not must_keep("      Tests  1955 passed (1955)", ContentKind.GENERIC)


# ---- must_keep — LOG ---------------------------------------------------------


def test_log_policy_keeps_verdict_lines():
    for line in [
        "      Tests  1955 passed (1955)",  # vitest
        " Test Files  188 passed (188)",  # vitest files
        "Tests:       1955 passed, 1955 total",  # jest
        "===== 1955 passed in 12.34s =====",  # pytest === summary
        "test result: ok. 42 passed; 0 failed; 0 ignored",  # cargo
        "  1955 passing",  # mocha
        "ok  \tgithub.com/x/y\t0.02s",  # go ok
        "PASS",  # go bare PASS
        "--- FAIL: TestFoo (0.00s)",  # go subtest
        "BUILD SUCCESSFUL in 3s",  # gradle/maven
        "process exited with exit code 1",  # exit status
    ]:
        assert must_keep(line, ContentKind.LOG), f"verdict dropped: {line!r}"


def test_log_policy_does_not_over_keep_per_test_lines():
    # Individual passing test lines should be droppable (no error word, no verdict).
    assert not must_keep(" ✓ src/a.test.ts (3 tests) 5ms", ContentKind.LOG)
    assert not must_keep("  Duration  3.21s", ContentKind.LOG)


# ---- must_keep — TRACEBACK ---------------------------------------------------


def test_traceback_policy_keeps_frames():
    assert must_keep('  File "a.py", line 3, in f', ContentKind.TRACEBACK)
    assert must_keep("Traceback (most recent call last):", ContentKind.TRACEBACK)
    assert not must_keep("some plain middle line", ContentKind.TRACEBACK)
    assert not must_keep("    x = foo()", ContentKind.TRACEBACK)


# ---- must_keep — DIFF --------------------------------------------------------


def test_diff_policy_keeps_hunk_and_file_headers():
    assert must_keep("@@ -1,3 +1,4 @@", ContentKind.DIFF)
    assert must_keep("diff --git a/x b/x", ContentKind.DIFF)
    assert must_keep("--- a/x.py", ContentKind.DIFF)
    assert must_keep("+++ b/x.py", ContentKind.DIFF)
    assert not must_keep(" unchanged context line", ContentKind.DIFF)
    assert not must_keep("+added line", ContentKind.DIFF)
    assert not must_keep("-removed line", ContentKind.DIFF)


# ---- summarize_dropped -------------------------------------------------------


def test_summarize_dropped_counts_error_and_warn():
    lines = ["ERROR boom", "ERROR again", "WARN careful", "ordinary line", "another ordinary"]
    s = summarize_dropped(lines)
    assert "2 error" in s
    assert "1 warn" in s


def test_summarize_dropped_empty_when_no_flagged():
    # A fold with only mundane lines needs no annotation.
    assert summarize_dropped(["plain line", "another plain"]) == ""


# ---- digest integration — TRACEBACK frames survive ---------------------------


def test_traceback_frames_survive_digest():
    lines = ["start of output", ""]
    lines += [f"filler line {i}" for i in range(20)]
    lines += [
        "Traceback (most recent call last):",
        '  File "app.py", line 42, in run',
        '  File "lib.py", line 7, in call',
        "RuntimeError: connection refused",
    ]
    lines += [f"more filler {i}" for i in range(20)]
    lines += ["end"]
    text = "\n".join(lines)

    d, changed = digest(text)
    assert changed
    assert 'File "app.py", line 42, in run' in d, "stack frame dropped"
    assert 'File "lib.py", line 7, in call' in d, "stack frame dropped"
    assert "RuntimeError: connection refused" in d


# ---- digest integration — DIFF hunk headers survive -------------------------


def test_diff_hunk_headers_survive_digest():
    lines = ["diff --git a/foo.py b/foo.py", "--- a/foo.py", "+++ b/foo.py"]
    lines += ["@@ -1,3 +1,4 @@"]
    lines += [" context line"] * 5
    lines += ["+added line"]
    lines += [" context line"] * 5
    lines += ["@@ -10,3 +11,4 @@"]
    lines += [" context line"] * 5
    lines += ["-removed line"]
    lines += [" context line"] * 20
    lines += ["-- end of patch"]
    text = "\n".join(lines)

    d, changed = digest(text)
    assert changed
    assert "diff --git a/foo.py b/foo.py" in d
    assert "@@ -1,3 +1,4 @@" in d, "first hunk header dropped"
    assert "@@ -10,3 +11,4 @@" in d, "second hunk header dropped"


# ---- digest integration — LOG verdict survives (regression) -----------------


def test_log_verdict_survives_digest():
    lines = ["RUN v1.6.0 /repo", ""]
    lines += [f"ERROR [Crypto] Decryption failed attempt {i}" for i in range(20)]
    lines += [f"WARN  [Auth] token near expiry {i}" for i in range(20)]
    lines += [""]
    lines += [" Test Files  188 passed (188)", "      Tests  1955 passed (1955)"]
    text = "\n".join(lines)

    d, changed = digest(text)
    assert changed
    assert "1955 passed" in d, "verdict dropped"
    assert "188 passed" in d, "file verdict dropped"


def test_url_lines_survive_digestion_every_kind():
    """URL-bearing lines are load-bearing for every content kind — live opus
    grading caught a digested report URL turning fetchurl into a websearch
    detour (EVALUATION.md 2.1)."""
    from distil.compress.keep_policy import ContentKind, must_keep
    from distil.compress.tier1 import digest

    line = "full report: https://www.ncei.noaa.gov/access/monitoring/monthly-report/global/202313"
    for kind in ContentKind:
        assert must_keep(line, kind), kind

    filler = [f"row {i}: ordinary unremarkable output line" for i in range(20)]
    text = "\n".join(filler[:10] + [line] + filler[10:])
    out, changed = digest(text)
    assert changed
    assert "https://www.ncei.noaa.gov" in out  # the URL survived the fold


def test_url_spam_is_still_bounded_by_shape_dedup():
    """A log spamming same-shape URLs must not defeat compression — must_keep
    routes URLs through tier1's per-shape repeat cap."""
    from distil.compress.tier1 import digest

    spam = "\n".join(
        f"GET https://health.internal/check/{i:04d} 200 OK in 3ms" for i in range(60)
    )
    out, changed = digest(spam)
    assert changed
    assert len(out) < len(spam) / 2  # spam still folds
