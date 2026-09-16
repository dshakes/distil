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

``replace_retrying`` is the other half of the same story. Locking is not the
only file primitive Windows spells differently: ``os.replace`` onto a path
another process is replacing or reading fails there, transiently, where POSIX
``rename`` simply succeeds. Every atomic writer in this codebase ends with that
one call, so the retry belongs here rather than in each of them.

Platform dispatch happens at call time (not at import time) so a test can
monkeypatch ``sys.platform`` and stub ``msvcrt`` in ``sys.modules`` to exercise
the Windows branch on a POSIX CI runner.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path
from typing import BinaryIO, Iterator


# Transient Windows failures of a replace onto a contended path: ERROR_ACCESS_DENIED (5)
# when another writer is mid-replace, ERROR_SHARING_VIOLATION (32) when a reader holds an
# open handle without FILE_SHARE_DELETE. Neither means "you may not write here".
_WIN_REPLACE_RETRY = frozenset({5, 32})
_WIN_REPLACE_TRIES = 10
_WIN_REPLACE_DELAY = 0.005  # ~45ms of waiting before the last attempt gives up


def replace_retrying(src: Path, dst: Path) -> None:
    """``os.replace(src, dst)``, with a bounded retry on Windows only.

    Every atomic write in this codebase ends the same way: write a temp file beside the
    target, fsync, then swap it in with one ``os.replace``. POSIX ``rename`` is atomic
    against a concurrent rename and against readers, so there is nothing to retry and that
    branch is byte-identical to the bare call. Windows' ``MoveFileEx`` has to open the
    destination, so two writers racing on one target — two overlapping ``distil wrap``
    sessions, two gateway workers persisting counters — fail each other's replace with
    ``ERROR_ACCESS_DENIED`` or ``ERROR_SHARING_VIOLATION``.

    The error name invites the wrong diagnosis. The condition is contention, it clears in
    microseconds, and by the time this is called the bytes are already written and
    fsync'd: aborting throws away completed work over a collision that resolves itself if
    you wait. A ``winerror`` outside the pair is NOT retried — it raises on the first
    attempt, because retrying a genuine permission failure only makes an accurate answer
    slower. One that never clears exhausts the budget and raises the last one, which is
    the same failure a single attempt gives, ~45ms later.

    It lives here, beside ``locked``, because this is the module that owns the
    cross-platform file primitives and all three atomic writers already route through it
    or sit next to one. Platform dispatch is at call time, like the rest of this module,
    so a test can monkeypatch ``sys.platform`` to drive the Windows branch on POSIX.

    ponytail: fixed delay, no backoff. Two writers is the observed case and the first
    retry wins it. Make it exponential if a target ever has more than a handful.
    """
    if sys.platform != "win32":
        os.replace(src, dst)
        return
    for attempt in range(_WIN_REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:  # PermissionError is an OSError; winerror picks the class
            if (
                getattr(exc, "winerror", None) not in _WIN_REPLACE_RETRY
                or attempt == _WIN_REPLACE_TRIES - 1
            ):
                raise
            time.sleep(_WIN_REPLACE_DELAY)


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
