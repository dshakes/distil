"""Scratch module used to prove the SDLC auto-fix loop closes end to end.

Throwaway branch; delete afterwards. It must never reach main.
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
    """Max over each sliding window of width `size`.

    Returns one entry per window; if `size` is at least as long as `values`,
    the single full-span max is returned. Non-positive `size` or empty
    `values` yields an empty list.
    """
    if size <= 0 or not values:
        return []
    if size >= len(values):
        return [max(values)]
    return [max(values[i : i + size]) for i in range(len(values) - size + 1)]
