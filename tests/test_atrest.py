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
