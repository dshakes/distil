"""Zero-dependency MCP server — JSON-RPC handling, compress/expand round-trip."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path
import time

import pytest

from distil import mcp_server as mcp

BIG = "\n".join(f"line {i}: some content value_{i}" for i in range(40))


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))


def test_initialize_echoes_protocol_and_serverinfo():
    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2025-06-18"
    assert resp["result"]["serverInfo"]["name"] == "distil"
    assert "tools" in resp["result"]["capabilities"]


def test_tools_list_has_three_tools():
    resp = mcp.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {"distil_compress", "distil_expand", "distil_savings"}


def test_compress_then_expand_round_trip():
    c = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "distil_compress", "arguments": {"text": BIG}},
        }
    )
    assert c["result"]["isError"] is False
    payload = json.loads(c["result"]["content"][0]["text"])
    assert payload["handle"] and payload["tokens_saved"] > 0
    assert len(payload["compressed"]) < len(BIG)

    e = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "distil_expand", "arguments": {"handle": payload["handle"]}},
        }
    )
    assert e["result"]["content"][0]["text"] == BIG  # byte-exact recovery


def test_expand_unknown_handle_is_error():
    e = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "distil_expand", "arguments": {"handle": "deadbeef"}},
        }
    )
    assert e["result"]["isError"] is True


def test_unknown_tool_is_jsonrpc_error():
    r = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert r["error"]["code"] == -32602


def test_notification_returns_none():
    assert mcp.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_is_error():
    r = mcp.handle_message({"jsonrpc": "2.0", "id": 7, "method": "bogus"})
    assert r["error"]["code"] == -32601


def test_serve_loop_over_stdio():
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n")
    stdout = io.StringIO()
    mcp.serve(stdin=stdin, stdout=stdout)
    out_lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    # initialize + tools/list answered; the notification produced no line.
    assert [o["id"] for o in out_lines] == [1, 2]


def test_concurrent_compress_calls_do_not_drop_handles():
    """Two racing distil_compress calls must both survive in the store —
    the unlocked load/load/save/save interleaving used to drop one."""
    import threading

    from distil.mcp_server import _load_store, _tool_compress

    texts = [
        "\n".join(f"alpha line {i} of stream A with some padding text" for i in range(30)),
        "\n".join(f"beta line {i} of stream B with some padding text" for i in range(30)),
    ]
    results: list[str] = []
    barrier = threading.Barrier(2)

    def go(t: str) -> None:
        barrier.wait()
        results.append(_tool_compress({"text": t}))

    threads = [threading.Thread(target=go, args=(t,)) for t in texts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    handles = [json.loads(r)["handle"] for r in results]
    assert all(handles)
    store = _load_store()
    for h in handles:
        assert h in store


def test_record_restore_expires_by_age(tmp_path, monkeypatch):
    """Restore originals older than the TTL are pruned even under the count cap."""
    import os
    import time as _time

    import distil.mcp_server as m

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_RESTORE_TTL_DAYS", 14.0)
    # The sweep is amortized over _SWEEP_EVERY records; this asserts what it does when it
    # runs, so it runs on every record here.
    monkeypatch.setattr(m, "_SWEEP_EVERY", 1)
    m.record_restore("aaaaaaaa", "old content")
    old_file = m._restore_dir() / "aaaaaaaa"
    ancient = _time.time() - 15 * 86400
    os.utime(old_file, (ancient, ancient))
    m.record_restore("bbbbbbbb", "new content")
    assert not old_file.exists()
    assert (m._restore_dir() / "bbbbbbbb").exists()


# ---------------------------------------------------------------------------
# Owner-only at creation, not by a chmod afterwards
# ---------------------------------------------------------------------------


@pytest.fixture()
def _no_chmod(monkeypatch):
    """Neuter Path.chmod and pin the umask.

    The invariant under test is that the mode is applied by os.open AT CREATION.
    A post-write chmod reaches 0600 too — a plain mode assertion passes either
    way — so the only way to test the window is to take the chmod away: what is
    left is whatever the file was created with.
    """
    monkeypatch.setattr(Path, "chmod", lambda *a, **k: None)
    old = os.umask(0o022)
    yield
    os.umask(old)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX modes only; Windows reads back 0o666 whatever we ask for",
)
def test_restore_blob_is_created_owner_only(tmp_path, _no_chmod):
    """The blob holds one agent's tool output — and under
    DISTIL_NO_ENCRYPT_AT_REST it holds it as plaintext, which is exactly the
    documented configuration where the 0644 window was observable."""
    mcp.record_restore("a" * 8, "some captured tool output")
    blob = tmp_path / "restore" / ("a" * 8)
    assert blob.exists()
    assert stat.S_IMODE(blob.stat().st_mode) == 0o600


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX modes only; Windows reads back 0o666 whatever we ask for",
)
def test_handle_store_is_created_owner_only(tmp_path, _no_chmod):
    mcp._save_store({"b" * 8: "some captured tool output"})
    assert stat.S_IMODE(mcp._store_path().stat().st_mode) == 0o600


def test_a_disk_handle_collision_declines_the_stub(tmp_path, monkeypatch):
    """An 8-hex handle that already maps to DIFFERENT bytes on disk is a collision one
    restart away from resolving to the other block's content. The in-memory map would
    hide it — the running proxy answers correctly and only a later process is wrong — so
    the disk's refusal has to reach the caller and the stub has to be declined. Keeping
    the block verbatim is always safe.
    """
    from distil.adapters.anthropic import RestoreStore

    import distil.mcp_server as m

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    assert m.record_restore("abcd1234", "the first block") is True
    # A fresh store, as a restarted or second proxy would have: nothing in memory to
    # catch the collision, so only the disk can.
    assert RestoreStore()._record("abcd1234", "a different block") is False
    assert m.load_restore("abcd1234") == "the first block", "the first writer was clobbered"


def test_the_store_sweep_is_amortized_rather_than_run_per_handle(tmp_path, monkeypatch):
    """The sweep is O(files) in `stat()` calls and used to run twice per recorded handle,
    so at the 5,000-file cap one handle cost up to 10,000 stats — and the re-read delta
    records several handles a turn. Bounded overshoot is the trade; the cap still holds
    the moment a sweep runs."""
    import distil.mcp_server as m

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_RESTORE_CAP", 2)
    monkeypatch.setattr(m, "_SWEEP_EVERY", 8)
    monkeypatch.setattr(m, "_since_sweep", 0)
    # The trigger has two inputs now — records AND elapsed time — so both have to be
    # pinned or this reads whatever the previous test left behind. Unset, `_last_sweep`
    # is 0.0 ("never swept"), which fires the time trigger on the first record and
    # silently retunes the count this test is about.
    monkeypatch.setattr(m, "_last_sweep", time.time())

    d = m._restore_dir()
    for i in range(7):
        m.record_restore(f"0000{i:04x}", f"original {i}")
    assert len(list(d.iterdir())) == 7, "the store was swept on a record that was not due"

    m.record_restore("00000007", "original 7")
    assert len(list(d.iterdir())) == m._RESTORE_CAP, "the due sweep did not enforce the cap"
    assert m.load_restore("00000007") == "original 7", "the newest handle was evicted"

    # ...and the overshoot in between is bounded by the interval, not unbounded.
    for i in range(8, 40):
        m.record_restore(f"0000{i:04x}", f"original {i}")
    assert len(list(d.iterdir())) <= m._RESTORE_CAP + m._SWEEP_EVERY - 1


def test_an_expired_blob_is_not_served_even_when_no_sweep_is_due(tmp_path, monkeypatch):
    """The TTL is a retention boundary for content that can hold secrets, so it has to
    hold between sweeps. Amortizing the sweep to one run per `_SWEEP_EVERY` records left a
    quiet store serving expired originals for as long as no 64th handle arrived — the read
    path never checked, and nothing else was going to."""
    import os

    import distil.mcp_server as m

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_RESTORE_TTL_DAYS", 14.0)
    monkeypatch.setattr(m, "_SWEEP_EVERY", 10_000)  # no count-triggered sweep, ever
    monkeypatch.setattr(m, "_last_sweep", time.time())  # ...and none on elapsed time

    m.record_restore("aaaaaaaa", "content past its retention date")
    blob = m._restore_dir() / "aaaaaaaa"
    ancient = time.time() - 15 * 86400
    os.utime(blob, (ancient, ancient))

    assert m.load_restore("aaaaaaaa") is None, "an expired original was still expandable"
    assert not blob.exists(), "the expired blob was served-as-absent but left on disk"


def test_a_blob_inside_the_ttl_is_still_served(tmp_path, monkeypatch):
    """The control: the read-side check must expire, not merely delete."""
    import distil.mcp_server as m

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_RESTORE_TTL_DAYS", 14.0)
    m.record_restore("bbbbbbbb", "still in date")
    assert m.load_restore("bbbbbbbb") == "still in date"


def test_the_sweep_fires_on_elapsed_time_with_no_new_records(tmp_path, monkeypatch):
    """Bulk expiry cannot depend on traffic that may never come. One record after the
    interval has passed sweeps, where the count trigger alone would not have."""
    import os

    import distil.mcp_server as m

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_RESTORE_TTL_DAYS", 14.0)
    monkeypatch.setattr(m, "_SWEEP_EVERY", 10_000)  # the count trigger can never fire
    monkeypatch.setattr(m, "_last_sweep", time.time())

    m.record_restore("cccccccc", "old")
    stale = m._restore_dir() / "cccccccc"
    ancient = time.time() - 15 * 86400
    os.utime(stale, (ancient, ancient))
    assert stale.exists(), "nothing should have swept yet"

    # A twenty-fourth of the TTL has gone by; the next record is due regardless of count.
    monkeypatch.setattr(m, "_last_sweep", time.time() - 14 * 3600 - 1)
    m.record_restore("dddddddd", "new")

    assert not stale.exists(), "the sweep did not fire on elapsed time"
    assert m.load_restore("dddddddd") == "new"


def test_the_disk_collision_check_uses_an_exclusive_create(tmp_path, monkeypatch):
    """`p.exists()` then write is check-then-act ACROSS PROCESSES, which is the only case
    this guard is for: two proxies folding the same block could both see "absent", and on
    a real collision the second would clobber the first. The create must be exclusive, so
    the loser finds the file already there and compares rather than overwriting."""
    import distil.mcp_server as m

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    opened: list[str] = []
    real_open = open

    def spy(file, mode="r", *a, **kw):
        opened.append(mode)
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr("builtins.open", spy)
    assert m.record_restore("abcd1234", "the first block") is True
    assert "xb" in opened, "the first write was not an exclusive create"

    # The second writer loses the race it never saw: same handle, different bytes.
    assert m.record_restore("abcd1234", "a different block") is False
    assert m.load_restore("abcd1234") == "the first block", "the first writer was clobbered"


def test_an_exclusive_create_still_refreshes_an_identical_blob(tmp_path, monkeypatch):
    """Re-recording the same bytes must not read as a collision — it is how a still-used
    handle keeps its mtime ahead of the TTL, and how a legacy plaintext blob is upgraded
    to the encrypted format."""
    import os

    import distil.mcp_server as m

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_RESTORE_TTL_DAYS", 14.0)
    assert m.record_restore("eeeeeeee", "same bytes") is True
    blob = m._restore_dir() / "eeeeeeee"
    old = time.time() - 13 * 86400
    os.utime(blob, (old, old))

    assert m.record_restore("eeeeeeee", "same bytes") is True, "a refresh read as a collision"
    assert blob.stat().st_mtime > old, "the mtime was not refreshed, so the TTL will expire it"
    assert m.load_restore("eeeeeeee") == "same bytes"


def test_an_expired_blob_does_not_block_a_new_handle(tmp_path, monkeypatch):
    """A blob past its retention date is gone as far as every reader is concerned, so it
    must not keep refusing a new stub for the same 8-hex handle forever."""
    import os

    import distil.mcp_server as m

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(m, "_RESTORE_TTL_DAYS", 14.0)
    monkeypatch.setattr(m, "_SWEEP_EVERY", 10_000)
    monkeypatch.setattr(m, "_last_sweep", time.time())
    m.record_restore("ffffffff", "long-expired content")
    blob = m._restore_dir() / "ffffffff"
    ancient = time.time() - 15 * 86400
    os.utime(blob, (ancient, ancient))

    assert m.record_restore("ffffffff", "a different block") is True
    assert m.load_restore("ffffffff") == "a different block"
