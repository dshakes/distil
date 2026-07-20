"""Integration-surface counters: shape mapping, surface env, persistence,
cross-process merge, allowlist filtering, fail-open."""

from __future__ import annotations

import json

import pytest

from distil import surfaces


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_SURFACE", raising=False)
    # Reset module state between tests (module-level accumulator).
    surfaces._pending["surfaces"] = {}
    surfaces._pending["shapes"] = {}
    surfaces._since_flush = 0


def test_shape_mapping():
    assert surfaces.shape_for_path("/v1/messages") == "anthropic"
    assert surfaces.shape_for_path("/v1/chat/completions") == "openai-chat"
    assert surfaces.shape_for_path("/v1/responses") == "openai-responses"
    assert surfaces.shape_for_path("/v1beta/models/gemini-pro:generateContent") == "gemini"


def test_surface_from_env(monkeypatch):
    assert surfaces.current_surface() == "proxy"  # unset → proxy
    monkeypatch.setenv("DISTIL_SURFACE", "wrap")
    assert surfaces.current_surface() == "wrap"
    monkeypatch.setenv("DISTIL_SURFACE", "bogus")
    assert surfaces.current_surface() == "proxy"  # non-allowlisted → proxy


def test_bump_snapshot_and_flush(monkeypatch, tmp_path):
    monkeypatch.setenv("DISTIL_SURFACE", "wrap")
    for _ in range(3):
        surfaces.bump("/v1/messages")
    surfaces.bump("/v1/chat/completions")
    snap = surfaces.snapshot()  # pending-only, not yet flushed
    assert snap["shapes"] == {"anthropic": 3, "openai-chat": 1}
    assert snap["surfaces"] == {"wrap": 4}
    surfaces.flush()
    disk = json.loads(surfaces.store_path().read_text())
    assert disk["shapes"]["anthropic"] == 3
    # Snapshot after flush is identical (disk now, pending empty).
    assert surfaces.snapshot()["surfaces"] == {"wrap": 4}


def test_flush_merges_with_other_process_counts(tmp_path):
    """Two processes flushing must ADD, not clobber."""
    surfaces.store_path().parent.mkdir(parents=True, exist_ok=True)
    surfaces.store_path().write_text(
        json.dumps({"shapes": {"anthropic": 10}, "surfaces": {"gateway": 5}})
    )
    surfaces.bump("/v1/messages")
    surfaces.flush()
    disk = json.loads(surfaces.store_path().read_text())
    assert disk["shapes"]["anthropic"] == 11
    assert disk["surfaces"] == {"gateway": 5, "proxy": 1}


def test_auto_flush_every_n(monkeypatch):
    for _ in range(surfaces.FLUSH_EVERY):
        surfaces.bump("/v1/messages")
    disk = json.loads(surfaces.store_path().read_text())
    assert disk["shapes"]["anthropic"] == surfaces.FLUSH_EVERY


def test_snapshot_filters_garbage_on_disk():
    surfaces.store_path().parent.mkdir(parents=True, exist_ok=True)
    surfaces.store_path().write_text(
        json.dumps(
            {
                "shapes": {"anthropic": 2, "evil/../shape": 9, "gemini": -1},
                "surfaces": {"wrap": "NaN"},
            }
        )
    )
    snap = surfaces.snapshot()
    assert snap["shapes"] == {"anthropic": 2}  # non-allowlisted + negative dropped
    assert snap["surfaces"] == {}  # non-numeric dropped


def test_corrupt_store_is_fail_open():
    surfaces.store_path().parent.mkdir(parents=True, exist_ok=True)
    surfaces.store_path().write_text("{corrupt")
    surfaces.bump("/v1/messages")
    surfaces.flush()  # must not raise; rewrites cleanly
    assert surfaces.snapshot()["shapes"] == {"anthropic": 1}
