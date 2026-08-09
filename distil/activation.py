"""Socket activation: let the supervisor own the listening socket.

The outage this exists to prevent
--------------------------------
``distil default --always-on`` pins ``ANTHROPIC_BASE_URL`` at a loopback port in
``~/.claude/settings.json``. That file is durable and outranks the environment,
so from then on *every* API request the machine makes goes through one local
process. If that process is not listening for even a second — a crash, a
memory-pressure kill, a restart after an upgrade — the client does not retry and
does not fall back. It gets ``ECONNREFUSED`` immediately and reports
``API Error: Unable to connect to API (ConnectionRefused)``, naming the provider
rather than us. Observed in the field repeatedly: sessions dying several times a
day on a machine whose proxy was, at every point anyone thought to look, up.

``KeepAlive`` does not fix this. It restarts the process *after* it dies, and
every connection attempted in the gap is refused. The gap is the bug.

The fix is that the process should never have owned the socket. When the
supervisor (launchd, systemd) creates the listening socket and hands the
descriptor down, the socket survives the worker: the kernel accepts and queues
connections onto its backlog for as long as the supervisor holds it, whether or
not any worker is alive to call ``accept()``. A crash becomes latency — the
client's connection sits in the backlog until the restarted worker drains it —
instead of a hard refusal. ``ECONNREFUSED`` stops being reachable at all.

This module answers one question: "did my supervisor hand me a listening
socket?" It returns ``None`` when run outside a supervisor (the ordinary
``distil proxy`` in a terminal), so the caller binds for itself as before.
"""

from __future__ import annotations

import os
import socket

__all__ = ["inherited_listener"]

# launchd's key inside the plist's `Sockets` dict, and systemd's first passed fd.
_LAUNCHD_SOCKET_NAME = b"Listeners"
_SD_LISTEN_FDS_START = 3


def _close_extra_fds(fds) -> None:
    """Close descriptors we were handed but will not serve on.

    Left open, a second listener accepts nothing: the kernel keeps queueing
    connections onto a backlog no one drains, so clients hang rather than fail.
    A hang is worse than a refusal — nothing surfaces an error to act on.
    """
    for fd in fds:
        try:
            os.close(int(fd))
        except OSError:  # pragma: no cover - already closed / not ours
            pass


def _from_systemd() -> socket.socket | None:
    """The socket from a systemd ``.socket`` unit, via the LISTEN_FDS protocol."""
    # LISTEN_PID must be present AND ours. Treating a missing value as "probably
    # for me" is how a stale LISTEN_FDS=1 inherited from an unrelated parent gets
    # us to adopt whatever happens to be on fd 3 — which for a proxy means serving
    # traffic on someone else's socket. sd_listen_fds requires the match; so do we.
    if os.environ.get("LISTEN_PID") != str(os.getpid()):
        return None
    try:
        count = int(os.environ.get("LISTEN_FDS", "0"))
    except ValueError:
        return None
    if count < 1:
        return None
    # A dual-stack socket unit (ListenStream twice, e.g. IPv4 + IPv6) passes down
    # MORE than one descriptor. We serve on the first; the rest must be closed
    # rather than left open. Leaking them would be the lesser problem — the real
    # one is that connections arriving on an unaccepted listener sit in the kernel
    # backlog forever, so a client hangs instead of failing, which is worse than a
    # refusal because nothing times out quickly enough to look like an error.
    _close_extra_fds(range(_SD_LISTEN_FDS_START + 1, _SD_LISTEN_FDS_START + count))
    return socket.socket(fileno=_SD_LISTEN_FDS_START)


def _from_launchd() -> socket.socket | None:
    """The socket from a launchd ``Sockets`` entry, via ``launch_activate_socket``."""
    import ctypes
    import ctypes.util

    lib = ctypes.util.find_library("System")
    if not lib:
        return None
    libc = ctypes.CDLL(lib, use_errno=True)
    fn = getattr(libc, "launch_activate_socket", None)
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    fn.restype = ctypes.c_int
    fds = ctypes.POINTER(ctypes.c_int)()
    count = ctypes.c_size_t(0)
    # Non-zero is the ordinary case, not an error: ESRV_NOT_MANAGED (the plain
    # `distil proxy` in a terminal) and ESRV_NO_SOCKET (a plist without the
    # `Sockets` key, i.e. any machine wired by an older distil) both land here.
    if fn(_LAUNCHD_SOCKET_NAME, ctypes.byref(fds), ctypes.byref(count)) != 0:
        return None
    if count.value < 1:
        return None
    sock = socket.socket(fileno=fds[0])
    # Same dual-stack case as systemd above: launchd hands down one descriptor per
    # ListenStream under the name. Serve the first, close the rest.
    if count.value > 1:
        _close_extra_fds(fds[i] for i in range(1, count.value))
    # The array is malloc'd by libSystem and owned by us; the descriptors inside
    # it are now owned by `sock` (or closed above), so only the array is freed.
    # argtypes is set explicitly: ctypes' default conversion happens to pass a
    # 64-bit pointer correctly today, but "happens to" is not a contract.
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free(fds)
    return sock


def inherited_listener() -> socket.socket | None:
    """A listening socket handed down by the supervisor, or ``None``.

    ``None`` means "nobody handed me one" — run unsupervised, or supervised by a
    unit that does not declare a socket. The caller binds its own socket in that
    case, which is the pre-existing behaviour and always works; it simply does
    not survive a restart of this process.
    """
    for source in (_from_systemd, _from_launchd):
        try:
            sock = source()
        except (OSError, AttributeError, ValueError):
            # A supervisor we cannot talk to is not a reason to fail to start.
            # Falling back to binding ourselves is strictly better than exiting.
            continue
        if sock is not None:
            return sock
    return None
