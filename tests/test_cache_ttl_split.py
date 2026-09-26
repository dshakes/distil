"""1-hour vs 5-minute cache writes: recorded by the proxy, priced 2x vs 1.25x.

Claude Code asks for 1-hour caching, so pricing every write at the 5-minute rate
understated its spend. Rows written before the split existed must price exactly as
they always did.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from distil import dissect as dz
from distil import pricing
from distil.proxy import _cache_ttl
from distil.savings_screen import Tokens, ledger_screen, request_cost, row_tokens
from distil.streamrelay import scan_usage

MODEL = "claude-sonnet-4-6"
USAGE = (
    b'{"usage":{"input_tokens":10,"cache_read_input_tokens":0,'
    b'"cache_creation_input_tokens":1000,'
    b'"cache_creation":{"ephemeral_5m_input_tokens":400,"ephemeral_1h_input_tokens":600}}}'
)


def test_scan_usage_reads_the_ttl_split():
    u = scan_usage(b"event: message_start\ndata: " + USAGE)
    assert u["ephemeral_1h_input_tokens"] == 600
    assert u["ephemeral_5m_input_tokens"] == 400
    assert "ephemeral_1h_input_tokens" not in scan_usage(b'{"usage":{"input_tokens":1}}')


@pytest.mark.parametrize(
    "usage,want",
    [
        ({"ephemeral_1h_input_tokens": 7}, 7),
        ({"cache_creation": {"ephemeral_1h_input_tokens": 8}}, 8),
        ({"cache_creation": {"ephemeral_1h_input_tokens": None}}, 0),
        ({"cache_creation": "odd"}, None),
        ({}, None),
    ],
)
def test_cache_ttl_flat_nested_absent(usage, want):
    assert _cache_ttl(usage, "ephemeral_1h_input_tokens") == want


def test_row_tokens_is_backward_compatible():
    old = {"usage_input_tokens": 1, "usage_cache_create": 100, "usage_output_tokens": 2}
    assert row_tokens(old) == (Tokens(1, 0, 100, 2), 0)
    new = {**old, "usage_cache_create_1h": 60, "usage_cache_create_5m": 40}
    assert row_tokens(new)[1] == 60
    # A split larger than the total can never price more than the writes that happened.
    assert row_tokens({**old, "usage_cache_create_1h": 10**6})[1] == 100


def _seed(home: Path, rows: list[dict]) -> None:
    sess = home / "sessions"
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "s1.requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_ledger_screen_prices_1h_writes_at_2x(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    base = {
        "ts": time.time(),
        "booked": True,
        "model": MODEL,
        "usage_input_tokens": 0,
        "usage_cache_read": 0,
        "usage_cache_create": 1_000_000,
        "usage_output_tokens": 0,
    }
    _seed(tmp_path, [base])
    old = ledger_screen(None).spent_usd
    _seed(tmp_path, [{**base, "usage_cache_create_1h": 1_000_000, "usage_cache_create_5m": 0}])
    new = ledger_screen(None).spent_usd
    p = pricing.resolve(MODEL)
    assert p is not None
    assert old == pytest.approx(1_000_000 * p.cache_write)  # pre-split rows: unchanged
    assert new == pytest.approx(1_000_000 * p.cache_write_1h)
    assert new / old == pytest.approx(2.0 / 1.25)


def test_request_cost_split():
    t = Tokens(cache_write=100)
    p = pricing.resolve(MODEL)
    assert p is not None
    assert request_cost(MODEL, t, 100) == pytest.approx(100 * p.cache_write_1h)


def _dissection(tmp_path, monkeypatch, rows: list[dict]) -> dz.Dissection:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    _seed(tmp_path, rows)
    return dz.dissect("s1", shadow=False)


def test_dissect_cache_write_usd(tmp_path, monkeypatch):
    row = {
        "ts": 1.0,
        "booked": True,
        "model": MODEL,
        "usage_input_tokens": 5,
        "usage_cache_read": 0,
        "usage_cache_create": 1000,
        "usage_cache_create_1h": 600,
        "usage_cache_create_5m": 400,
    }
    unpriced = {**row, "model": "no-such-model"}
    d = _dissection(tmp_path, monkeypatch, [row, {**row, "usage_cache_create": None}, unpriced])
    p = pricing.resolve(MODEL)
    assert p is not None
    five, one = d.cache_write_usd or (0.0, 0.0)
    assert five == pytest.approx(400 * p.cache_write)
    assert one == pytest.approx(600 * p.cache_write_1h)
    assert "1-hour 2x rate" in dz.render_text(d, color=False)
    js = dz.to_json(d)
    assert js["insights"]["usage"]["cache_write_usd"]["1h"] == pytest.approx(
        round(600 * p.cache_write_1h, 6)
    )


def test_dissect_cache_write_usd_none_without_writes(tmp_path, monkeypatch):
    d = _dissection(tmp_path, monkeypatch, [{"ts": 1.0, "booked": True, "model": MODEL}])
    assert d.cache_write_usd is None
