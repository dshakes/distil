"""Cross-platform advisory locking, one API for every ``fcntl.flock`` call site.

``fcntl`` is POSIX-only; before this module, every writer in this codebase
guarded ``import fcntl`` behind ``ImportError`` and, on Windows, silently
dropped the lock — the gateway key store and audit log (and the savings
ledger, shadow counters, retention meter, and adoption surfaces file) could
all interleave writes from concurrent workers there. ``msvcrt.locking`` is
Windows' equivalent, but it locks a byte range of an *already-sized* file,
which is fiddly against a data file that starts empty or gets truncated and
rewritten. Locking a small sidecar ``<path>.lock`` file next to the real data
file sidesteps that: the sidecar always holds exactly one byte, and the real
file is never touched by the locking itself. Same trick ``mcp_server.py``
already used for its store; this just makes it available everywhere else.

Platform dispatch happens at call time (not at import time) so a test can
monkeypatch ``sys.platform`` and stub ``msvcrt`` in ``sys.modules`` to exercise
the Windows branch on a POSIX CI runner.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import BinaryIO, Iterator


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _lock(fh: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)


def _unlock(fh: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock scoped to ``path`` for the block.

    Best-effort and fail-open, matching every call site's existing contract:
    a lock that can't be taken (missing dir, odd filesystem, permissions, an
    unimportable locking module) degrades to no lock rather than raising.
    """
    lp = _lock_path(path)
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lp, "a+b")  # noqa: SIM115 — closed in the finally below
    except OSError:
        yield
        return
    try:
        try:
            # msvcrt.locking needs the byte range it locks to already exist.
            if os.fstat(fh.fileno()).st_size == 0:
                fh.write(b"\0")
                fh.flush()
            _lock(fh)
        except (OSError, ImportError):
            yield
            return
        try:
            yield
        finally:
            with contextlib.suppress(OSError, ImportError):
                _unlock(fh)
    finally:
        fh.close()
