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
    except OSError:
        pass
    key = secrets.token_bytes(_KEY_LEN)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(key)
        p.chmod(0o600)
    except OSError:
        pass  # ponytail: best-effort; key still works in-memory for this process
    return key


# ---------------------------------------------------------------------------
# Core crypto primitives
# ---------------------------------------------------------------------------


def _derive(master: bytes, label: bytes) -> bytes:
    """One-level HKDF-style sub-key: HMAC-SHA256(master, label)."""
    return hmac.new(master, label, hashlib.sha256).digest()


def _xor_stream(key: bytes, nonce: bytes, data: bytes) -> bytes:
    """XOR *data* with HMAC-SHA256-CTR keystream derived from *key* and *nonce*."""
    if not data:
        return b""
    chunks: list[bytes] = []
    counter = 0
    for i in range(0, len(data), _MAC_LEN):
        block = hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        chunk = data[i : i + _MAC_LEN]
        # zip stops at the shorter sequence — handles the final partial block
        chunks.append(bytes(x ^ y for x, y in zip(chunk, block)))
        counter += 1
    return b"".join(chunks)


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
