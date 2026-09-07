"""Where a benchmark writes its results — scratch by default, tracked on request.

Several E-series scripts default their ``--out`` to a file that is *committed*
under ``docs/paper/results/``. That default is a loaded gun: a smoke run with
reduced parameters (``--control-reps 5`` on E3) silently overwrote the paper's
source artifact with a number nobody intended to publish, and it was only caught
because someone read the diff.

So the default now lands in ``benchmarks/results/scratch/`` (git-ignored), and
overwriting a paper artifact takes a deliberate act: ``--write-tracked``, or an
explicit ``--out <that path>``. Both leave a trace in the shell history.

Usage in a script::

    from benchmarks.outpath import add_out_args, resolve_out

    TRACKED = ROOT / "docs/paper/results/leave_one_domain_out.json"
    add_out_args(ap, TRACKED)
    ...
    out = resolve_out(args, TRACKED)
    out.write_text(...)
    print(f"-> {out}")
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "benchmarks" / "results" / "scratch"


def scratch_for(tracked: Path) -> Path:
    """The throwaway twin of a tracked artifact: same filename, scratch directory."""
    return SCRATCH / tracked.name


def add_out_args(ap: argparse.ArgumentParser, tracked: Path, *, flag: str = "--out") -> None:
    """Add *flag* (defaulting to scratch) and ``--write-tracked`` to *ap*.

    *flag* exists only because one script spells it ``--out-dir``; changing that
    name would break every recorded E7 invocation for no gain.
    """
    ap.add_argument(
        flag,
        type=Path,
        default=None,
        help=f"output path (default: {scratch_for(tracked).relative_to(ROOT)}, not tracked)",
    )
    ap.add_argument(
        "--write-tracked",
        action="store_true",
        help=f"publish to the committed artifact {tracked.relative_to(ROOT)} — "
        "only for a full, real run",
    )


def resolve_out(args: argparse.Namespace, tracked: Path, *, dest: str = "out") -> Path:
    """Pick the output path and make sure its parent directory exists.

    An explicit path wins (including an explicit path *at* the tracked file —
    typing it out in full is consent). ``--write-tracked`` selects the tracked
    artifact. Otherwise: scratch.
    """
    given = getattr(args, dest, None)
    out = given or (tracked if getattr(args, "write_tracked", False) else scratch_for(tracked))
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
