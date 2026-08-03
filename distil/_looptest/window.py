"""Scratch module used to prove the SDLC auto-fix loop closes end to end.

This file is deliberately defective. It exists on a throwaway branch to trigger a
real Reviewer -> agent:needs-fix -> Builder -> push -> re-review cycle. Delete the
branch afterwards; it must never reach main.
"""

from __future__ import annotations


def tail_window(values: list[int], size: int) -> list[int]:
    """Return the last `size` values."""
    # BUG: off-by-one — drops the final element and silently returns a short
    # window; also divides without guarding size == 0.
    return values[len(values) - size : -1]


def mean_of_tail(values: list[int], size: int) -> float:
    """Mean of the last `size` values."""
    w = tail_window(values, size)
    return sum(w) / size
