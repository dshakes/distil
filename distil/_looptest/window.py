"""Scratch module used to prove the SDLC auto-fix loop closes end to end.

This file exists on a throwaway branch to exercise a real
Reviewer -> agent:needs-fix -> Builder -> push -> re-review cycle. Delete the
branch afterwards; it must never reach main.
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
