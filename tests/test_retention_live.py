"""Live retention meter — content-free, sampled, fail-open.

The meter runs in the request path and is the one part of the retention work that
touches real user content, so these tests are about the guarantees rather than the
arithmetic: a unique marker planted in tool output must NEVER reach disk, an
exception must never propagate, and sampling must actually gate the work.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from distil.adapters.anthropic import compress_messages
from distil.retention import LiveMeter, format_live, load_live, measure_live

MARKER = "Zq9LiveMarker"


def _tool_result(text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}],
    }


def _big_log() -> str:
    """A realistic log: mostly benign lines, a couple of errors.

    Deliberately not all-errors — distil's keep policy pins error lines verbatim, so a
    log of nothing but errors is never digested and would test nothing.
    """
    lines = [
        f"2026-07-29 10:00:{i % 60:02d} INFO worker-{i} latency_ms={i * 7} "
        f"path=/var/log/app/{MARKER}-{i}.log id={i} ok"
        for i in range(300)
    ]
    lines.insert(150, f"2026-07-29 10:02:00 ERROR connection refused to db-primary {MARKER}")
    return "\n".join(lines)


def test_measures_real_compression_path() -> None:
    """Drive the actual adapter, not a stub: facts must be accounted for."""
    original = [_tool_result(_big_log())]
    compressed, store = compress_messages(original)
    dims = measure_live(original, compressed, store)
    total = sum(t.total for t in dims.values())
    accounted = sum(t.retained + t.recoverable + t.lost for t in dims.values())
    assert total > 0
    assert accounted == total


def _older_turns() -> list[dict[str, Any]]:
    """The recency rule keeps the last RECENCY_KEEP_TURNS tool results verbatim, so a
    log only gets digested once newer turns exist. That is the shape to test against."""
    return [_tool_result(_big_log())] + [
        _tool_result(f"later tool result {i} latency_ms={i}") for i in range(4)
    ]


def test_digested_facts_are_recoverable_via_the_store() -> None:
    """Facts behind a handle must resolve through the live RestoreStore."""
    original = _older_turns()
    compressed, store = compress_messages(original)
    dims = measure_live(original, compressed, store)
    assert sum(t.recoverable for t in dims.values()) > 0
    # Without the store, those same facts have no proven recovery path.
    blind = measure_live(original, compressed, None)
    assert sum(t.recoverable for t in blind.values()) == 0
    assert sum(t.lost for t in blind.values()) > sum(t.lost for t in dims.values())


def test_no_content_reaches_disk(tmp_path: Path) -> None:
    """The load-bearing privacy invariant: counts only, never text."""
    path = tmp_path / "retention.jsonl"
    original = [_tool_result(_big_log())]
    compressed, store = compress_messages(original)
    LiveMeter(1.0, path=path).observe(original, compressed, store)

    raw = path.read_text(encoding="utf-8")
    assert MARKER not in raw
    assert "ERROR" not in raw and "/var/log" not in raw and "latency_ms" not in raw
    record = json.loads(raw.strip().splitlines()[0])
    assert set(record) == {"ts", "dims"}
    for triple in record["dims"].values():
        assert len(triple) == 3
        assert all(isinstance(v, int) for v in triple)


def test_rate_zero_is_fully_disabled(tmp_path: Path) -> None:
    path = tmp_path / "retention.jsonl"
    meter = LiveMeter(0.0, path=path)
    assert not meter.enabled and not meter.should_sample()
    meter.observe([_tool_result(_big_log())], [], None)
    assert not path.exists()


def test_sampling_gates_writes(tmp_path: Path) -> None:
    path = tmp_path / "retention.jsonl"
    original = [_tool_result(_big_log())]
    compressed, store = compress_messages(original)
    meter = LiveMeter(0.5, rng=random.Random(1234), path=path)
    for _ in range(40):
        meter.observe(original, compressed, store)
    written = len(path.read_text(encoding="utf-8").strip().splitlines())
    assert 5 < written < 35  # sampled, not all-or-nothing


def test_never_raises_on_hostile_input(tmp_path: Path) -> None:
    """Fail-open: the meter must not be able to cost a user their request."""
    meter = LiveMeter(1.0, path=tmp_path / "r.jsonl")
    for junk in (None, [None], [{"role": "tool"}], [{"content": {"a": {"b": 1}}}], "nonsense"):
        meter.observe(junk, junk, None)  # must not raise


def test_broken_store_is_survived(tmp_path: Path) -> None:
    class _Exploding:
        def expand(self, handle: str) -> str:
            raise RuntimeError("store is gone")

    original = _older_turns()  # so digest handles actually exist to be expanded
    compressed, _ = compress_messages(original)
    dims = measure_live(original, compressed, _Exploding())
    assert sum(t.total for t in dims.values()) > 0  # scored anyway
    assert sum(t.recoverable for t in dims.values()) == 0  # unprovable => not credited


def test_unwritable_path_does_not_raise(tmp_path: Path) -> None:
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory", encoding="utf-8")
    meter = LiveMeter(1.0, path=blocked / "nested" / "retention.jsonl")
    original = [_tool_result(_big_log())]
    compressed, store = compress_messages(original)
    meter.observe(original, compressed, store)  # swallowed


def test_scan_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single enormous tool result must not be scanned without limit."""
    import distil.retention as r

    monkeypatch.setattr(r, "_LIVE_MAX_CHARS", 500)
    huge = [_tool_result("path=/a/b/c.log latency_ms=5 " * 20000)]
    compressed, store = compress_messages(huge)
    dims = measure_live(huge, compressed, store)
    # Bounded input => bounded target count, not a target per repetition.
    assert sum(t.total for t in dims.values()) < 50


def test_openai_tool_role_is_read() -> None:
    original = [{"role": "tool", "content": f"latency_ms=42 path=/var/log/{MARKER}.log"}]
    dims = measure_live(original, original, None)
    assert dims["numerics"].total > 0
    assert dims["artifacts"].total > 0


def test_anthropic_text_block_content_is_read() -> None:
    """tool_result content arrives as a list of {"type":"text"} blocks, not a bare str."""
    original = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "latency_ms=99 path=/srv/app/x.log"}],
                }
            ],
        }
    ]
    dims = measure_live(original, original, None)
    assert dims["numerics"].total > 0 and dims["artifacts"].total > 0


def test_unknown_handle_in_text_is_skipped() -> None:
    """A marker naming a handle the store cannot expand must not raise or credit."""

    class _Empty:
        def expand(self, handle: str) -> str | None:
            return None

    original = [_tool_result("latency_ms=42 path=/var/log/app/x.log")]
    compressed = [_tool_result("<< +9 lines, handle=deadbeef >>")]
    dims = measure_live(original, compressed, _Empty())
    assert sum(t.recoverable for t in dims.values()) == 0
    assert sum(t.lost for t in dims.values()) > 0


def test_empty_traffic_scores_nothing() -> None:
    dims = measure_live([{"role": "user", "content": "just a question"}], [], None)
    assert all(t.total == 0 for t in dims.values())


def test_aggregation_and_render(tmp_path: Path) -> None:
    path = tmp_path / "retention.jsonl"
    original = [_tool_result(_big_log())]
    compressed, store = compress_messages(original)
    meter = LiveMeter(1.0, path=path)
    meter.observe(original, compressed, store)
    meter.observe(original, compressed, store)

    dims, requests = load_live(path)
    assert requests == 2
    combined = sum(t.total for t in dims.values())
    assert combined > 0
    rendered = format_live(dims, requests)
    assert "2 sampled requests" in rendered
    assert "recall" in rendered


def test_malformed_rows_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "retention.jsonl"
    path.write_text(
        "\n".join(
            [
                "not json",
                "",  # blank line mid-file (a flush boundary) must be skipped
                json.dumps({"no_dims": 1}),
                json.dumps({"ts": 1, "dims": {"numerics": [3, 2, 1]}}),
                json.dumps({"ts": 2, "dims": {"numerics": "bad"}}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    dims, requests = load_live(path)
    assert requests == 2  # the two well-formed rows
    assert dims["numerics"].total == 3


def test_missing_file_reads_empty(tmp_path: Path) -> None:
    dims, requests = load_live(tmp_path / "absent.jsonl")
    assert requests == 0 and all(t.total == 0 for t in dims.values())
    assert "no live retention samples yet" in format_live(dims, requests)


def test_cli_wires_the_flags_through() -> None:
    """The call sites use getattr defaults so programmatic Namespaces keep working —
    which would also hide a broken flag. Pin the real argparse path instead."""
    from distil.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["wrap", "claude"]).retention == 0.05  # on by default: free
    assert parser.parse_args(["wrap", "--retention", "0", "claude"]).retention == 0.0
    assert parser.parse_args(["proxy"]).retention == 0.0  # explicit opt-in for a raw proxy
    assert parser.parse_args(["proxy", "--retention", "0.1"]).retention == 0.1
    assert parser.parse_args(["retention", "--live"]).live is True
    assert parser.parse_args(["retention", "--dataset", "hotpotqa", "-n", "5"]).n == 5


def test_proxy_accepts_retention_rate() -> None:
    """build_handler must take the kwarg the CLI passes it."""
    from distil.proxy import build_handler

    assert build_handler(upstream="http://127.0.0.1:1", retention_rate=0.5) is not None


def test_home_default_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    original = [_tool_result(_big_log())]
    compressed, store = compress_messages(original)
    LiveMeter(1.0).observe(original, compressed, store)
    assert (tmp_path / "retention.jsonl").exists()
