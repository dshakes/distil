"""First-sight tightening of the Tier-1 digest: a dropped run no longer than the
marker that would replace it is shown inline. Marker format is unchanged — every
marker still names the block's handle."""

from __future__ import annotations

import re

import pytest

from distil import query_flywheel
from distil.adapters.anthropic import RestoreStore, _compress_tool_result_text
from distil.compress.tier1 import _handle, digest

_MARKER = re.compile(r"^<< \+(\d+) lines, handle=([0-9a-f]{8}) >>$")


def _markers(out: str) -> list[tuple[int, str]]:
    return [(int(m[1]), m[2]) for ln in out.splitlines() if (m := _MARKER.match(ln))]


def _long(i: int) -> str:
    return f"row {i:03d} " + "payload " * 12


def _three_runs() -> str:
    lines = ["h0", "h1", "h2"] + [_long(i) for i in range(10)]
    lines += ["ERROR one"] + [_long(i) for i in range(10, 20)]
    lines += ["ERROR two"] + [_long(i) for i in range(20, 30)] + ["tail"]
    return "\n".join(lines)


def test_every_marker_names_the_handle() -> None:
    text = _three_runs()
    out, changed = digest(text)
    assert changed
    assert _markers(out) == [(10, _handle(text))] * 3
    assert not re.search(r"<< \+\d+ lines >>", out)  # no handle-less form


def test_inline_threshold_is_the_real_marker_length() -> None:
    marker_len = len(f"<< +1 lines, handle={'0' * 8} >>")
    # Longer than a handle-less marker would be, no longer than the real one: inlined.
    fits = "x" * (marker_len - 1)  # + its newline == marker_len
    assert len(fits) + 1 > len("<< +1 lines >>")
    lines = ["h0", "h1", "h2", "ERROR a", fits, "ERROR b"]
    lines += [_long(i) for i in range(10)] + ["tail"]
    text = "\n".join(lines)
    out, _ = digest(text)
    assert f"ERROR a\n{fits}\nERROR b" in out
    assert _markers(out) == [(10, _handle(text))]
    # One char longer than the real marker: folded.
    lines[4] = "x" * marker_len
    text = "\n".join(lines)
    out, _ = digest(text)
    assert _markers(out) == [(1, _handle(text)), (10, _handle(text))]


def test_all_gaps_inlined_means_unchanged() -> None:
    _, changed = digest("\n".join(["h0", "h1", "h2", "a", "b", "tail"]))
    assert not changed


def test_inlined_lines_are_not_reported_to_the_flywheel(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[set[int]] = []
    monkeypatch.setattr(
        query_flywheel, "maybe_record", lambda h, i, lines, k, dropped: seen.append(dropped)
    )
    lines = ["h0", "h1", "h2", "ERROR a", "x", "ERROR b"]
    lines += [_long(i) for i in range(10)] + ["tail"]
    digest("\n".join(lines), intent=frozenset({"zzqqunmatched"}))
    assert seen
    assert 4 not in seen[0]  # shown inline, so not dropped
    assert set(range(6, 16)) <= seen[0]


def test_later_marker_content_recovers_through_the_adapter() -> None:
    text = _three_runs()
    store = RestoreStore()
    out = _compress_tool_result_text(text, store)
    assert out == _compress_tool_result_text(text, RestoreStore())  # deterministic
    marks = _markers(out)
    assert len(marks) == 3
    assert _long(25) not in out  # folded behind the LAST marker...
    assert store.expand(marks[-1][1]) == text  # ...and recovered through its own handle
