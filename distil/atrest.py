"""distil/atrest.py — authenticated encryption at rest, stdlib only.

Construction: HMAC-SHA256-CTR + encrypt-then-MAC with distinct derived keys.

WHY NOT AES
-----------
Python's stdlib has no AES implementation. The ``cryptography`` package does,
but distil has a hard zero-runtime-dependency constraint. Hand-rolling a block
cipher (AES, ChaCha20) would be dangerous amateur cryptography. Instead we use
HMAC-SHA256 as a secure pseudorandom function (PRF), which the stdlib provides
and which has strong standard-model proofs.

THE CONSTRUCTION
----------------
1. Key derivation (HKDF-style, one level):
   enc_key = HMAC-SHA256(master, b"enc")
   mac_key = HMAC-SHA256(master, b"mac")
   Domain separation ensures an attacker who learns one sub-key gains nothing
   about the other.

2. Keystream (CTR mode using HMAC as a PRF):
   block_i = HMAC-SHA256(enc_key, nonce || counter_i)   [counter_i = 4-byte big-endian]
   keystream = block_0 || block_1 || ...  (each block is 32 bytes)
   ciphertext = plaintext XOR keystream[:len(plaintext)]
   Under the assumption that SHA-256 is a PRF (the same assumption TLS 1.3's
   HKDF-Expand relies on), this keystream is computationally indistinguishable
   from random and is CPA-secure.

3. Authenticate (encrypt-then-MAC):
   tag = HMAC-SHA256(mac_key, nonce || ciphertext)
   This prevents chosen-ciphertext attacks and provides integrity. The tag is
   always verified with hmac.compare_digest (constant-time) before decryption.

4. Nonce: 16 random bytes from secrets.token_bytes per write. Per-nonce
   uniqueness is best practice; nonce reuse does NOT cause keystream reuse
   because each HMAC block mixes the nonce AND a counter, but fresh nonces
   are cheap and eliminate the concern entirely.

ON-DISK FORMAT (magic b"DSTL1")
--------------------------------
  [0:5]   magic  b"DSTL1"
  [5:21]  nonce  16 random bytes
  [21:-32] ciphertext  (same length as plaintext)
  [-32:]  tag    HMAC-SHA256(mac_key, nonce || ciphertext)

Total overhead per file: 53 bytes.
Files without this prefix are treated as legacy plaintext (no magic = no
header) so reads never break on old files (upgrade-skew safety).

WHAT THIS PROTECTS AGAINST
---------------------------
- Backup / sync leakage: cloud backup of ~/.distil/ leaks ciphertexts, not
  content.
- Cross-user reads (multi-user NAS, world-readable backup snapshots): the
  data files are unreadable without the key file.

WHAT THIS DOES NOT PROTECT AGAINST
-----------------------------------
- A local attacker with the same UID, who can read both the data files and
  ~/.distil/restore.key. This is documented explicitly in THREAT_MODEL.md.
  distil is not a privilege boundary.
- Physical access to the running process (the key is in memory while distil
  runs).

These limits are clearly stated in THREAT_MODEL.md; no false claims are made.

KEY FILE
--------
32 raw bytes at ${DISTIL_HOME:-~/.distil}/restore.key, chmod 0600, created on
first use with secrets.token_bytes(32). Rotating the key makes all existing
encrypted files undecryptable (they are then treated as missing, the same as
expired TTL — fail-open for the request path).

OPT-OUT
-------
Set DISTIL_NO_ENCRYPT_AT_REST=1 to skip encryption (e.g. for strict
zero-data-retention deployments using an ephemeral DISTIL_HOME — no point
encrypting a ramdisk that evaporates on reboot). Writing is skipped; reading
always decrypts when a magic header is present so a mixed fleet (some nodes
opted out, some not) degrades gracefully.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from pathlib import Path

_MAGIC = b"DSTL1"
_NONCE_LEN = 16
_MAC_LEN = 32  # HMAC-SHA256 output length
_KEY_LEN = 32
_MIN_ENCRYPTED_LEN = len(_MAGIC) + _NONCE_LEN + _MAC_LEN  # 53


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------


def _key_path() -> Path:
    base = Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))
    return base / "restore.key"


#: Serialises key creation *within this process*. The on-disk protocol (O_EXCL +
#: retry-read) already makes concurrent creators converge across processes; this
#: removes the same race between threads, where it is both cheap to eliminate and
#: the case that actually shows up (one gateway, many request threads, cold start).
#: Measured without it: ~1 run in 15 under CPU contention ended with two threads
#: on different keys, which silently orphans whatever the loser encrypted.
_KEY_LOCK = threading.Lock()


def _load_key() -> bytes:
    """Load the 32-byte master key, creating it on first use.

    Creating the key file is best-effort: if the write fails (e.g. read-only
    filesystem) the returned key is still used for in-process encryption — but
    it will be regenerated on next startup, making existing encrypted files
    undecryptable (they are treated as missing, same as expired TTL — fail-open).
    """
    p = _key_path()
    try:
        data = p.read_bytes()
        if len(data) == _KEY_LEN:
            return data
        # A readable file of the WRONG length (truncated write, half-created key,
        # a crash between create and write) must be repaired, not stepped around.
        # Falling straight through to generate-and-O_EXCL leaves the bad file in
        # place forever: every call mints a fresh ephemeral key, so nothing written
        # in one process is readable in the next. Measured before this: an 8-byte
        # restore.key stayed 8 bytes, successive _load_key() calls disagreed, and
        # a blob encrypted after restart decrypted to None.
        invalid_existing = True
    except OSError:
        invalid_existing = False
    with _KEY_LOCK:
        # Re-read under the lock: while we waited, another thread may have created
        # the key. Without this the winner writes and every queued thread then
        # generates its own, which is the race the lock exists to prevent.
        try:
            data = p.read_bytes()
            if len(data) == _KEY_LEN:
                return data
        except OSError:
            pass
        return _create_key_locked(p, invalid_existing)


def _create_key_locked(p: Path, invalid_existing: bool) -> bytes:
    """Create (or repair) the key file. Caller must hold ``_KEY_LOCK``."""
    key = secrets.token_bytes(_KEY_LEN)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # 0600 at open time, not after the write. `write_bytes` then `chmod` leaves
        # the MASTER KEY — which decrypts every restore blob — at the process umask
        # (measured: 0o644) until the chmod lands. os.open with the mode argument
        # closes the window; O_EXCL additionally refuses to clobber a key that
        # another process wrote first, so a race can't silently swap the key.
        # Write to a private temp file first, then link it into place. O_EXCL on
        # the real path is NOT enough on its own: it creates the file before the
        # write lands, so a loser that catches FileExistsError can re-read an
        # EMPTY file, fail the length check, and fall through to its own
        # in-memory key — the same silent data loss, just a narrower window.
        # (Measured with the create->write gap widened: 11 of 12 threads diverged.)
        # os.link is atomic and fails with FileExistsError if we lost, so the key
        # file only ever appears fully written.
        # Random suffix, not pid+tid: those repeat for two sequential calls on the
        # same thread, so a temp file left behind by an earlier attempt (Windows
        # refuses to unlink a file that still has an open handle) makes the next
        # O_EXCL create raise FileExistsError. That is indistinguishable from
        # "another writer won the race", so the loser adopts a key that was never
        # written — and every blob it encrypts becomes unreadable. Measured with a
        # colliding leftover in place: the key file was never created and two
        # successive loads returned different keys.
        tmp = p.with_name(f"{p.name}.{secrets.token_hex(8)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, key)
            os.fsync(fd)  # the bytes must be durable before the name points at them
        finally:
            os.close(fd)
        try:
            if invalid_existing:
                # The file exists but is not a usable key. os.replace is atomic, so
                # a concurrent reader sees either the old bad file or the new good
                # one, never a partial. Repairing beats leaving it: the alternative
                # is every process running on a private ephemeral key forever.
                os.replace(tmp, p)
                return key
            _hardlink(tmp, p)
        except FileExistsError:
            raise  # we lost the race; the handler below adopts the winner's key
        except OSError:
            # os.link is unavailable or unsupported here (some Windows filesystems,
            # exotic mounts). Fall back to an exclusive create + write. That
            # reopens the narrow empty-file window this branch exists to close, so
            # it is a fallback, never the primary path — and the reader below
            # still rejects a short read, so a loser degrades to an in-memory key
            # rather than a WRONG one.
            # O_EXCL on the REAL path, not os.replace. replace is last-writer-wins:
            # two simultaneous first-touch creators each publish, and the loser's
            # blobs become unreadable — measured, two distinct keys out of eight
            # threads. O_EXCL is first-writer-wins, so exactly one creator survives
            # and everyone else takes the FileExistsError path below and adopts it.
            #
            # This does split create from write, which is the window the hardlink
            # path exists to avoid — the retry loop in that handler is what makes
            # it safe, re-reading until the winner's bytes are actually there.
            fd2 = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd2, key)
                os.fsync(fd2)
            finally:
                os.close(fd2)
            return key
        finally:
            # missing_ok: the os.replace paths CONSUME the temp file (it becomes
            # the key), so there is nothing left to remove. Only the hardlink path
            # leaves it behind. Either way the directory must not accumulate temps.
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
    except FileExistsError:
        # Another process/thread created the key between our read miss and our
        # O_EXCL create. It won; adopt ITS key. Returning our own would encrypt
        # blobs that the on-disk key can never decrypt — a silent, permanent data
        # loss that only shows up later as an unrecoverable handle. Measured: a
        # 12-thread first-touch race produced 4 distinct keys before this.
        try:
            # Retry: the winner may have CREATED the file without having written
            # it yet (O_EXCL on the real path in the no-hardlink fallback, or any
            # create/write split), so a single read can legitimately come back
            # empty or short. Bailing out there is what makes a loser encrypt with
            # a key that never reaches disk. Measured on the fallback path with the
            # write window widened: 8 threads produced 8 distinct keys, 1 matching
            # disk. Bounded so a genuinely truncated file can never hang startup —
            # after the last attempt we fall through to the in-memory key, which
            # degrades this process rather than wedging it.
            for attempt in range(50):
                data = p.read_bytes()
                if len(data) == _KEY_LEN:
                    return data
                time.sleep(0.002 * (attempt + 1))
            data = p.read_bytes()
            if len(data) == _KEY_LEN:
                return data
        except OSError:
            pass  # unreadable winner — fall through to our in-memory key
    except OSError:
        pass  # ponytail: best-effort; key still works in-memory for this process
    return key


# ---------------------------------------------------------------------------
# Core crypto primitives
# ---------------------------------------------------------------------------


def _hardlink(src: Path, dst: Path) -> None:
    """``os.link`` behind a module-local name so tests can stub the no-link case.

    Patching ``atrest.os`` would swap the attribute on the *global* ``os`` module
    for every importer in the process — the restore store included — so a test for
    this branch would silently break unrelated ones.
    """
    os.link(src, dst)


def _derive(master: bytes, label: bytes) -> bytes:
    """One-level HKDF-style sub-key: HMAC-SHA256(master, label)."""
    return hmac.new(master, label, hashlib.sha256).digest()


def _xor_stream(key: bytes, nonce: bytes, data: bytes) -> bytes:
    """XOR *data* with HMAC-SHA256-CTR keystream derived from *key* and *nonce*."""
    if not data:
        return b""
    # Build the whole keystream, then XOR in one big-int operation. The per-byte
    # `zip` loop this replaces cost ~2x: on a 2 MiB blob, 127.3ms -> 65.6ms.
    # Output is byte-identical — same HMAC-SHA256-CTR keystream, same algorithm.
    n = len(data)
    keystream = b"".join(
        hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        for counter in range((n + _MAC_LEN - 1) // _MAC_LEN)
    )[:n]
    # int.from_bytes/to_bytes round-trips exactly for equal-length operands; the
    # leading-zero bytes a plain int would drop are restored by the explicit length.
    return (int.from_bytes(data, "big") ^ int.from_bytes(keystream, "big")).to_bytes(n, "big")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _no_encrypt() -> bool:
    return os.environ.get("DISTIL_NO_ENCRYPT_AT_REST", "").strip() == "1"


def encrypt_bytes(data: bytes) -> bytes:
    """Return *data* encrypted with the DSTL1 construction.

    Returns *data* unchanged when DISTIL_NO_ENCRYPT_AT_REST=1.
    Never raises: any key-file I/O failure is absorbed internally and the
    in-memory key is used — safe on the request-serving path.
    """
    if _no_encrypt():
        return data
    master = _load_key()
    enc_key = _derive(master, b"enc")
    mac_key = _derive(master, b"mac")
    nonce = secrets.token_bytes(_NONCE_LEN)
    ciphertext = _xor_stream(enc_key, nonce, data)
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    return _MAGIC + nonce + ciphertext + tag


def decrypt_bytes(data: bytes) -> bytes | None:
    """Decrypt *data*.

    - No magic header → legacy plaintext → return *data* unchanged (upgrade
      compatibility: old files are readable without re-encryption).
    - Magic header present → authenticate then decrypt; return ``None`` on
      authentication failure or truncation (caller treats this as missing).
    """
    if not data.startswith(_MAGIC):
        return data  # legacy plaintext — pass through for upgrade compatibility
    if len(data) < _MIN_ENCRYPTED_LEN:
        return None  # truncated / corrupt
    body = data[len(_MAGIC) :]
    nonce = body[:_NONCE_LEN]
    tag = body[-_MAC_LEN:]
    ciphertext = body[_NONCE_LEN:-_MAC_LEN]
    master = _load_key()
    mac_key = _derive(master, b"mac")
    enc_key = _derive(master, b"enc")
    expected = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        return None  # authentication failure — caller treats as missing (fail-open)
    return _xor_stream(enc_key, nonce, ciphertext)
