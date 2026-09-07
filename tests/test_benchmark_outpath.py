"""A default benchmark run must never write a committed paper artifact.

E3 (``benchmarks/leave_one_domain_out.py``) defaulted its ``--out`` to
``docs/paper/results/leave_one_domain_out.json``, so a reduced smoke run
(``--control-reps 5``) overwrote the paper's source with a number nobody meant to
publish. These tests hold the guard for every script that can reach a tracked path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.outpath import ROOT, SCRATCH, add_out_args, resolve_out, scratch_for

# Every benchmark whose output path can reach a committed artifact:
# (module, tracked path, argparse dest).
GUARDED = [
    ("benchmarks.leave_one_domain_out", "docs/paper/results/leave_one_domain_out.json", "out"),
    (
        "benchmarks.trajectory_certificate",
        "docs/paper/results/swe_e2e_longhorizon/trajectory_certificate.json",
        "out",
    ),
    (
        "benchmarks.trajectory_bound",
        "docs/paper/results/swe_e2e_longhorizon/trajectory_bound.json",
        "out",
    ),
    (
        "benchmarks.skeleton_certificate",
        "docs/paper/results/swe_e2e_longhorizon/skeleton_certificate.json",
        "out",
    ),
    ("benchmarks.swe_bench_e2e.sample", "docs/paper/results/swe_e2e/sample.json", "out"),
    (
        "benchmarks.swe_bench_e2e.aggregate",
        "docs/paper/results/swe_bench_verified_e2e.json",
        "out",
    ),
    (
        "benchmarks.swe_bench_e2e.preload_images",
        "docs/paper/results/swe_e2e/preload_report.json",
        "report",
    ),
    # a DIRECTORY of committed predictions, guarded the same way
    ("benchmarks.swe_bench_e2e.run_agent", "docs/paper/results/swe_e2e", "out_dir"),
]


def _tracked_files() -> set[Path]:
    out = subprocess.run(
        ["git", "ls-files", "docs/paper/results"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {ROOT / line for line in out.stdout.splitlines() if line}


def _parser(tracked: Path, dest: str = "out") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_out_args(ap, tracked, flag="--" + dest.replace("_", "-"))
    return ap


@pytest.mark.parametrize(("module", "rel", "dest"), GUARDED, ids=[m for m, _, _ in GUARDED])
def test_default_out_is_not_a_tracked_path(module: str, rel: str, dest: str) -> None:
    tracked = ROOT / rel
    out = resolve_out(_parser(tracked, dest).parse_args([]), tracked, dest=dest)
    assert out != tracked
    assert SCRATCH in out.parents, f"{module} default must land in scratch, got {out}"
    assert not out.is_relative_to(ROOT / "docs")


def test_every_benchmark_that_can_reach_a_tracked_path_is_guarded() -> None:
    """A new benchmark that defaults into docs/paper/results/ must land in this list.

    Greps for the literal so the next script to reach for a tracked default trips
    this test rather than a reviewer noticing the diff a week later. A module that
    only *reads* those artifacts (or writes to generated/) is exempt by name.
    """
    reads_only = {
        # emits LaTeX macros into docs/paper/generated/, never into results/
        "benchmarks/e14_macros.py",
        # --out is required, so it has no default to get wrong
        "benchmarks/swe_bench_e2e/score.py",
    }
    hits = subprocess.run(
        ["git", "grep", "-l", "docs/paper/results", "--", "benchmarks/*.py", "benchmarks/**/*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    writers = {
        h
        for h in hits
        if h not in reads_only and ".write_text(" in (ROOT / h).read_text(encoding="utf-8")
    }
    guarded = {h for h in writers if "add_out_args" in (ROOT / h).read_text(encoding="utf-8")}
    assert writers == guarded, f"unguarded tracked-results writers: {sorted(writers - guarded)}"
    assert len(guarded) == len(GUARDED)


def test_scratch_directory_is_git_ignored() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(scratch_for(Path("x.json")))],
        cwd=ROOT,
        check=False,
    )
    assert ignored.returncode == 0, "benchmarks/results/scratch/ must be git-ignored"


def test_write_tracked_selects_the_committed_artifact() -> None:
    tracked = ROOT / GUARDED[0][1]
    assert resolve_out(_parser(tracked).parse_args(["--write-tracked"]), tracked) == tracked


def test_explicit_out_wins_over_everything(tmp_path: Path) -> None:
    tracked = ROOT / GUARDED[0][1]
    mine = tmp_path / "nested" / "mine.json"
    args = _parser(tracked).parse_args(["--out", str(mine), "--write-tracked"])
    out = resolve_out(args, tracked)
    assert out == mine
    assert out.parent.is_dir(), "resolve_out must create the directory it hands back"
    # typing the tracked path out in full is consent, and still works
    assert resolve_out(_parser(tracked).parse_args(["--out", str(tracked)]), tracked) == tracked


def test_e3_smoke_run_leaves_every_tracked_artifact_untouched() -> None:
    """The regression itself: the reduced run that clobbered the paper's E3 source."""
    before = {p: p.read_bytes() for p in _tracked_files() if p.is_file()}
    assert before, "expected committed paper artifacts to guard"

    proc = subprocess.run(
        [sys.executable, "benchmarks/leave_one_domain_out.py", "--control-reps", "5"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert str(SCRATCH.relative_to(ROOT)) in proc.stdout.replace(str(ROOT) + "/", "")
    assert {p: p.read_bytes() for p in before} == before
