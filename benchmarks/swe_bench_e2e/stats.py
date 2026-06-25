#!/usr/bin/env python3
"""Statistics helpers for the SWE-bench end-to-end eval (Phase 5 / E7).

Pass@1 is a binomial proportion; we report the **Wilson score** 95% confidence
interval rather than the normal approximation because n is small (30-50) and the
proportion can sit near 0 or 1, where the Wald interval misbehaves.
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054  # standard normal quantile for a two-sided 95% interval


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns ``(low, high)`` in [0,1]."""
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def pass_at_1(resolved: int, n: int) -> dict[str, float | int]:
    lo, hi = wilson_ci(resolved, n)
    return {
        "resolved": resolved,
        "n": n,
        "pass_at_1": round(resolved / n, 4) if n else 0.0,
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
    }
