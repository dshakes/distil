"""Scratch module used to prove the SDLC auto-fix loop closes end to end.

Throwaway branch. Round 2 re-introduces a defect so the Reviewer flags BLOCKING
again — this time with SDLC_TRIGGER_BOTS deleted, so the authorize gate must
recognise the App by its own identity rather than by a hand-set allowlist.
"""

from __future__ import annotations


def tail_window(values: list[int], size: int) -> list[int]:
    """Return the last `size` values (fewer if `values` is shorter)."""
    if size <= 0:
        return []
    return values[max(0, len(values) - size) :]


def mean_of_tail(values: list[int], size: int) -> float:
    """Mean of the last `size` values."""
    w = tail_window(values, size)
    if not w:
        return 0.0
    return sum(w) / len(w)


def rolling_max(values: list[int], size: int) -> list[int]:
    """Max over each sliding window of width `size`."""
    # BUG: range end is off by one so the final window is never emitted, and a
    # size larger than the input silently yields [] instead of one full-span max.
    return [max(values[i : i + size]) for i in range(len(values) - size)]
