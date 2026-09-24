"""First-sight tightening of the Tier-1 digest.

(a) a dropped run shorter than its marker is shown inline instead;
(b) only the first marker in a block names the handle.
Both must keep the digest byte-deterministic and the original recoverable.
"""

from __future__ import annotations

import re

from distil.adapters.anthropic import RestoreStore, _compress_tool_result_text
from distil.compress.tier1 import _handle, digest

_MARKER = re.compile(r"^<< \+(\d+) lines(?:, handle=([0-9a-f]{8}))? >>$")


def _markers(out: str) -> list[tuple[int, str | None]]:
    return [(int(m[1]), m[2]) for ln in out.splitlines() if (m := _MARKER.match(ln))]


def _long(i: int) -> str:
    return f"row {i:03d} " + "payload " * 12


def test_only_first_marker_names_the_handle() -> None:
    # Two ERROR lines split the body into three dropped runs.
    lines = ["h0", "h1", "h2"] + [_long(i) for i in range(10)]
    lines += ["ERROR one"] + [_long(i) for i in range(10, 20)]
    lines += ["ERROR two"] + [_long(i) for i in range(20, 30)] + ["tail"]
    text = "\n".join(lines)
    out, changed = digest(text)
    assert changed
    marks = _markers(out)
    assert len(marks) == 3
    assert marks[0][1] == _handle(text)
    assert [h for _, h in marks[1:]] == [None, None]
    assert sum(n for n, _ in marks) == 30  # every dropped line is still counted


def test_gap_cheaper_than_its_marker_is_shown_inline() -> None:
    # A one-char line between two pinned ERROR lines costs less than "<< +1 lines >>".
    lines = ["h0", "h1", "h2", "ERROR a", "x", "ERROR b"]
    lines += [_long(i) for i in range(10)] + ["tail"]
    text = "\n".join(lines)
    out, changed = digest(text)
    assert changed
    assert "ERROR a\nx\nERROR b" in out
    assert _markers(out) == [(10, _handle(text))]  # the handle moves to the first REAL marker


def test_all_gaps_cheap_means_unchanged() -> None:
    text = "\n".join(["h0", "h1", "h2", "a", "b", "tail"])  # 1-char gaps only
    out, changed = digest(text)
    assert not changed


def test_deterministic_and_recoverable_through_the_adapter() -> None:
    lines = ["h0", "h1", "h2"] + [_long(i) for i in range(10)]
    lines += ["ERROR mid"] + [_long(i) for i in range(10, 20)] + ["tail"]
    text = "\n".join(lines)
    store = RestoreStore()
    a = _compress_tool_result_text(text, store)
    b = _compress_tool_result_text(text, RestoreStore())
    assert a == b  # same input -> same bytes
    assert len(a) < len(text)
    (handle,) = {h for _, h in _markers(a) if h}
    assert store.expand(handle) == text
