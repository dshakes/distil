"""Tests for distil.atrest — authenticated encryption at rest.

Covers: round-trip, tamper detection, legacy plaintext passthrough, key
auto-creation with 0600 permissions, opt-out, nonce uniqueness across writes,
and integration with mcp_server persistence (mcp_store.json + restore/ files).
"""

from __future__ import annotations

import json
import stat

import sys

import pytest

import distil.atrest as atrest
from distil import mcp_server as mcp
from distil.adapters.anthropic import RestoreStore


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    # Ensure encryption is on (default) for every test unless the test overrides.
    monkeypatch.delenv("DISTIL_NO_ENCRYPT_AT_REST", raising=False)


# ---------------------------------------------------------------------------
# atrest unit tests
# ---------------------------------------------------------------------------


def test_round_trip_encrypt_decrypt():
    plaintext = b"hello world, this is some agent tool output content"
    blob = atrest.encrypt_bytes(plaintext)
    assert blob != plaintext
    assert blob.startswith(atrest._MAGIC)
    result = atrest.decrypt_bytes(blob)
    assert result == plaintext


def test_empty_bytes_round_trip():
    blob = atrest.encrypt_bytes(b"")
    assert blob.startswith(atrest._MAGIC)
    assert atrest.decrypt_bytes(blob) == b""


def test_tamper_body_causes_auth_failure():
    blob = atrest.encrypt_bytes(b"sensitive content")
    # Flip a byte in the ciphertext region (after magic+nonce, before tag).
    tampered = bytearray(blob)
    mid = atrest._MIN_ENCRYPTED_LEN  # just inside ciphertext
    tampered[mid // 2] ^= 0xFF
    assert atrest.decrypt_bytes(bytes(tampered)) is None


def test_tamper_tag_causes_auth_failure():
    blob = atrest.encrypt_bytes(b"sensitive content")
    tampered = bytearray(blob)
    tampered[-1] ^= 0x01  # flip last tag byte
    assert atrest.decrypt_bytes(bytes(tampered)) is None


def test_truncated_blob_causes_auth_failure():
    blob = atrest.encrypt_bytes(b"data")
    # Truncate to just the magic header — below minimum valid length.
    assert atrest.decrypt_bytes(blob[:6]) is None


def test_legacy_plaintext_passes_through():
    # A file written before 1.20.0 has no magic header; decrypt_bytes must
    # return it unchanged so callers can still parse it (upgrade compatibility).
    legacy = b'{"key": "value"}'
    result = atrest.decrypt_bytes(legacy)
    assert result == legacy


def test_nonce_uniqueness_across_writes():
    data = b"same plaintext repeated"
    blob1 = atrest.encrypt_bytes(data)
    blob2 = atrest.encrypt_bytes(data)
    # Different nonces → different ciphertexts.
    assert blob1 != blob2
    # Both decrypt correctly.
    assert atrest.decrypt_bytes(blob1) == data
    assert atrest.decrypt_bytes(blob2) == data


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod 0600 is a no-op on Windows (mode reads back 0o666); the owner-only guarantee is POSIX-only",
)
def test_key_file_created_with_0600(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    atrest._load_key()
    key_file = tmp_path / "restore.key"
    assert key_file.exists()
    assert key_file.stat().st_size == 32
    mode = stat.S_IMODE(key_file.stat().st_mode)
    assert mode == 0o600


def test_key_file_is_stable_across_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    k1 = atrest._load_key()
    k2 = atrest._load_key()
    assert k1 == k2


def test_opt_out_produces_plaintext(monkeypatch):
    monkeypatch.setenv("DISTIL_NO_ENCRYPT_AT_REST", "1")
    data = b"unencrypted"
    result = atrest.encrypt_bytes(data)
    assert result == data
    assert not result.startswith(atrest._MAGIC)


def test_opt_out_decrypt_passes_through_plaintext(monkeypatch):
    monkeypatch.setenv("DISTIL_NO_ENCRYPT_AT_REST", "1")
    data = b'{"handle": "abc"}'
    assert atrest.decrypt_bytes(data) == data


def test_wrong_key_causes_auth_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    blob = atrest.encrypt_bytes(b"secret")
    # Overwrite the key file with a different key.
    (tmp_path / "restore.key").write_bytes(b"\x00" * 32)
    assert atrest.decrypt_bytes(blob) is None


# ---------------------------------------------------------------------------
# Integration: mcp_store.json encryption
# ---------------------------------------------------------------------------


def test_mcp_store_file_is_encrypted_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    mcp._store_add("abcd1234", "original text content")
    raw = (tmp_path / "mcp_store.json").read_bytes()
    assert raw.startswith(atrest._MAGIC), "store file should have DSTL1 header"
    # The raw bytes must not contain the plaintext handle verbatim.
    assert b"abcd1234" not in raw


def test_mcp_store_round_trip_through_encrypt(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    mcp._store_add("abcd1234", "original text content")
    loaded = mcp._load_store()
    assert loaded.get("abcd1234") == "original text content"


def test_mcp_store_legacy_plaintext_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    # Write an old-format plaintext store.
    store_file = tmp_path / "mcp_store.json"
    store_file.write_text(json.dumps({"deadbeef": "legacy content"}))
    loaded = mcp._load_store()
    assert loaded.get("deadbeef") == "legacy content"


def test_mcp_store_tamper_treated_as_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    mcp._store_add("abcd1234", "content")
    store_file = tmp_path / "mcp_store.json"
    raw = bytearray(store_file.read_bytes())
    raw[-1] ^= 0xFF  # tamper the auth tag
    store_file.write_bytes(bytes(raw))
    assert mcp._load_store() == {}


# ---------------------------------------------------------------------------
# Integration: restore/ directory encryption
# ---------------------------------------------------------------------------


def test_restore_file_is_encrypted_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    mcp.record_restore("abcd1234", "tool output: sensitive data")
    restore_file = mcp._restore_dir() / "abcd1234"
    raw = restore_file.read_bytes()
    assert raw.startswith(atrest._MAGIC)
    assert b"sensitive" not in raw


def test_restore_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    mcp.record_restore("abcd1234", "tool output: sensitive data")
    assert mcp.load_restore("abcd1234") == "tool output: sensitive data"


def test_restore_legacy_plaintext_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    restore_dir = mcp._restore_dir()
    restore_dir.mkdir(parents=True, exist_ok=True)
    (restore_dir / "abcd1234").write_text("legacy plain content", encoding="utf-8")
    assert mcp.load_restore("abcd1234") == "legacy plain content"


def test_restore_legacy_file_upgraded_on_rewrite(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    restore_dir = mcp._restore_dir()
    restore_dir.mkdir(parents=True, exist_ok=True)
    p = restore_dir / "abcd1234"
    p.write_text("legacy content", encoding="utf-8")
    # record_restore with same content should rewrite (upgrade to encrypted).
    mcp.record_restore("abcd1234", "legacy content")
    raw = p.read_bytes()
    assert raw.startswith(atrest._MAGIC), "file should be upgraded to DSTL1 format"


def test_restore_tamper_treated_as_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    mcp.record_restore("abcd1234", "content")
    p = mcp._restore_dir() / "abcd1234"
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 0xFF
    p.write_bytes(bytes(raw))
    assert mcp.load_restore("abcd1234") is None


def test_restore_store_survives_restart_encrypted(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    RestoreStore()._record("deadbeef", "original text")
    # Simulate restart: fresh RestoreStore instance.
    result = RestoreStore().expand("deadbeef")
    assert result == "original text"


def test_restore_collision_guard_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    mcp.record_restore("abcd1234", "first content")
    # A different original claiming the same handle must be rejected.
    mcp.record_restore("abcd1234", "different content")
    assert mcp.load_restore("abcd1234") == "first content"


def test_opt_out_writes_plaintext_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_NO_ENCRYPT_AT_REST", "1")
    mcp.record_restore("abcd1234", "plain content")
    p = mcp._restore_dir() / "abcd1234"
    raw = p.read_bytes()
    assert not raw.startswith(atrest._MAGIC)
    assert mcp.load_restore("abcd1234") == "plain content"


def test_restore_cap_of_zero_does_not_wipe_the_store(tmp_path, monkeypatch):
    """`[:-0]` is the whole list — an unguarded cap of 0 evicts everything.

    0 must mean "no count cap", not "delete every blob", or disabling the cap would
    silently destroy every digest original and break expand for the whole session."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(mcp, "_RESTORE_CAP", 0)
    for i in range(5):
        mcp.record_restore(f"cafe{i:04d}", f"original {i}")
    assert len(list((tmp_path / "restore").iterdir())) == 5
    assert mcp.load_restore("cafe0000") == "original 0"


def test_restore_cap_evicts_oldest_beyond_the_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(mcp, "_RESTORE_CAP", 3)
    for i in range(6):
        mcp.record_restore(f"beef{i:04d}", f"original {i}")
    assert len(list((tmp_path / "restore").iterdir())) == 3
    assert mcp.load_restore("beef0005") == "original 5"


def test_concurrent_first_touch_keeps_one_key(tmp_path, monkeypatch) -> None:
    """Threads racing to create the master key must all end up on the SAME key.

    The loser of the O_EXCL create used to keep the key it had generated in memory
    and encrypt with it, while the winner's key was the one persisted. Every blob
    written by a loser then became permanently unrecoverable — silent data loss
    that only surfaces later as a dead restore handle. Measured before the fix:
    15 of 16 concurrently-written blobs unrecoverable.
    """
    import importlib
    import secrets as real_secrets
    import threading
    import time

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import atrest

    importlib.reload(atrest)

    class _SlowSecrets:
        """Widen the read-miss -> create window so the race is deterministic."""

        def token_bytes(self, n: int) -> bytes:
            time.sleep(0.05)
            return real_secrets.token_bytes(n)

    monkeypatch.setattr(atrest, "secrets", _SlowSecrets())

    blobs: list[tuple[int, bytes]] = []
    threads = [
        threading.Thread(
            target=lambda i=i: blobs.append((i, atrest.encrypt_bytes(b"payload-%d" % i)))
        )
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Fresh module state: only the key that actually landed on disk is available.
    importlib.reload(atrest)
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    for i, blob in blobs:
        assert atrest.decrypt_bytes(blob) == b"payload-%d" % i, (
            f"blob {i} was encrypted with a key that never reached disk — "
            "the first-touch key race is back"
        )


def test_key_race_falls_back_when_the_winners_key_is_unreadable(tmp_path, monkeypatch) -> None:
    """If the winner's key file exists but cannot be read, keep working in memory.

    The loser of the create race normally re-reads the winner's key. When that read
    fails too (permissions, a truncated file), encryption must still work for this
    process rather than raising — the same fail-open contract as the rest of the
    at-rest path.
    """
    import importlib

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import atrest

    importlib.reload(atrest)

    key_path = atrest._key_path()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(b"short")  # exists, but wrong length -> not a usable key

    def _boom(*_args, **_kwargs):
        raise OSError("unreadable")

    monkeypatch.setattr(type(key_path), "read_bytes", _boom)

    key = atrest._load_key()
    assert len(key) == atrest._KEY_LEN, "must still return a usable in-memory key"


def test_unwritable_key_directory_still_yields_a_usable_key(tmp_path, monkeypatch) -> None:
    """A key that cannot be persisted must not break encryption for this process.

    Read-only or full filesystems happen; distil degrades to an in-memory key
    (restore blobs simply do not survive the process) rather than refusing to run.
    """
    import importlib

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import atrest

    importlib.reload(atrest)

    def _boom(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(atrest.os, "open", _boom)

    key = atrest._load_key()
    assert len(key) == atrest._KEY_LEN
