"""RestoreStore persistence — digest handles must survive a proxy restart and be
expandable cross-process (mcp_server), the fix for lossless-mode handle orphaning."""

import pytest

from distil import mcp_server
from distil.adapters.anthropic import RestoreStore


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))


def test_expand_survives_restart():
    RestoreStore()._record("deadbeef", "original text")
    # fresh instance = restarted proxy process
    assert RestoreStore().expand("deadbeef") == "original text"


def test_mcp_expand_sees_proxy_handles():
    RestoreStore()._record("cafef00d", "proxy original")
    assert mcp_server._tool_expand({"handle": "cafef00d"}) == "proxy original"


def test_missing_handle_still_raises():
    with pytest.raises(KeyError):
        RestoreStore().expand("00000000")


def test_load_restore_rejects_traversal(tmp_path):
    (tmp_path / "secret").write_text("leak")
    assert mcp_server.load_restore("../secret") is None


def test_restore_cap_prunes_oldest(monkeypatch):
    monkeypatch.setattr(mcp_server, "_RESTORE_CAP", 2)
    # The sweep is amortized over _SWEEP_EVERY records (it is O(files) in stat() calls);
    # this asserts what it does when it runs, so it runs on every record here. The
    # bounded overshoot in between is asserted in tests/test_mcp_server.py.
    monkeypatch.setattr(mcp_server, "_SWEEP_EVERY", 1)
    store = RestoreStore()
    for h in ("aaaaaaa1", "aaaaaaa2", "aaaaaaa3"):
        store._record(h, h)
    # ponytail: same-second mtimes can tie, so assert cap + newest survivor only
    assert len(list(mcp_server._restore_dir().iterdir())) == 2
    assert mcp_server.load_restore("aaaaaaa3") == "aaaaaaa3"
