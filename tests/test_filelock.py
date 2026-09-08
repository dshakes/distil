"""Cross-platform advisory locking (distil._filelock) — the module every
fcntl call site in the codebase now routes through.

Covers: real POSIX mutual exclusion, the simulated Windows (msvcrt) path
(so the win32 branch gets exercised on whatever CI runner executes this),
fail-open when the underlying lock call errors, and that the sidecar lock
file never pollutes the real data file it protects.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from distil import _filelock


def test_posix_two_threads_serialize_through_the_lock(tmp_path):
    """A second locker must not enter the critical section until the first
    releases — the whole point of taking a lock at all."""
    if sys.platform == "win32":
        pytest.skip("this test drives the real fcntl branch, POSIX-only")

    path = tmp_path / "data.json"
    order: list[str] = []
    a_inside = threading.Event()
    a_may_release = threading.Event()

    def writer_a():
        with _filelock.locked(path):
            order.append("a-start")
            a_inside.set()
            a_may_release.wait(5)
            order.append("a-end")

    def writer_b():
        a_inside.wait(5)
        with _filelock.locked(path):
            order.append("b-start")

    ta, tb = threading.Thread(target=writer_a), threading.Thread(target=writer_b)
    ta.start()
    assert a_inside.wait(5)
    tb.start()
    a_may_release.set()
    ta.join(5)
    tb.join(5)

    # B could only start after A finished — never interleaved.
    assert order == ["a-start", "a-end", "b-start"]


def _install_fake_msvcrt(monkeypatch) -> threading.Lock:
    """A minimal stand-in for the stdlib msvcrt module, backed by a real
    threading.Lock so it enforces the same mutual exclusion a real Windows
    byte-range lock would, letting the win32 branch run on any platform."""
    real_lock = threading.Lock()
    held_by: dict[int, bool] = {}

    def locking(fd: int, mode: int, nbytes: int) -> None:
        if mode == 1:  # LK_LOCK
            real_lock.acquire()
            held_by[fd] = True
        elif mode == 0:  # LK_UNLCK
            if held_by.pop(fd, False):
                real_lock.release()

    fake = types.SimpleNamespace(LK_LOCK=1, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(sys, "platform", "win32")
    return real_lock


def test_windows_branch_serializes_two_threads(tmp_path, monkeypatch):
    """Simulated win32: _filelock must dispatch to msvcrt.locking and still
    serialize concurrent lockers, not just silently no-op."""
    _install_fake_msvcrt(monkeypatch)
    path = tmp_path / "data.json"
    order: list[str] = []
    a_inside = threading.Event()
    a_may_release = threading.Event()

    def writer_a():
        with _filelock.locked(path):
            order.append("a-start")
            a_inside.set()
            a_may_release.wait(5)
            order.append("a-end")

    def writer_b():
        a_inside.wait(5)
        with _filelock.locked(path):
            order.append("b-start")

    ta, tb = threading.Thread(target=writer_a), threading.Thread(target=writer_b)
    ta.start()
    assert a_inside.wait(5)
    tb.start()
    a_may_release.set()
    ta.join(5)
    tb.join(5)

    assert order == ["a-start", "a-end", "b-start"]


def test_windows_branch_reuses_the_sidecar_across_calls(tmp_path, monkeypatch):
    """A second, later `locked()` call must still succeed (lock file already
    holds its reserved byte from the first call)."""
    _install_fake_msvcrt(monkeypatch)
    path = tmp_path / "data.json"
    with _filelock.locked(path):
        pass
    with _filelock.locked(path):  # must not raise / hang
        pass


def test_lock_failure_degrades_to_no_lock(tmp_path, monkeypatch):
    """A lock call that raises must never block the caller — every site using
    this module is fail-open by contract."""
    monkeypatch.setattr(_filelock, "_lock", lambda fh: (_ for _ in ()).throw(OSError("nope")))
    path = tmp_path / "data.json"
    ran = False
    with _filelock.locked(path):
        ran = True
    assert ran


def test_sidecar_never_touches_the_real_file(tmp_path):
    """The lock lives in a `.lock` sidecar; the protected file's own bytes
    must be exactly what the caller writes, never the lock's reserved byte."""
    path = tmp_path / "data.json"
    with _filelock.locked(path):
        path.write_text('{"ok": true}', encoding="utf-8")
    assert path.read_text(encoding="utf-8") == '{"ok": true}'
    assert (tmp_path / "data.json.lock").exists()
