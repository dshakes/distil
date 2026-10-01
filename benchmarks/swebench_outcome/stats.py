"""Wilson CI, exact McNemar, paired-difference CI, non-inferiority sample size (stdlib only)."""

from __future__ import annotations

import math
from statistics import NormalDist


def _z(alpha: float = 0.05) -> float:
    return NormalDist().inv_cdf(1 - alpha / 2)


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = _z(alpha), k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p on discordant pairs: b = plain-only solved, c = distil-only solved."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_diff(b: int, c: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """distil - plain resolve-rate difference (c - b)/n with a paired Wald CI.

    ponytail: Wald, not Tango/exact; fine at n>=100, revisit for tiny pilots (report says so).
    """
    if n == 0:
        return (0.0, -1.0, 1.0)
    d = (c - b) / n
    se = math.sqrt(max((b + c) - (c - b) ** 2 / n, 0.0)) / n
    z = _z(alpha)
    return (d, d - z * se, d + z * se)


def sample_size_noninferiority(
    margin: float, p_discordant: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Pairs needed so the (1-alpha) lower bound clears -margin w.p. `power` when the true
    difference is 0. Var(d) ~= p_discordant / n (paired binary)."""
    nd = NormalDist()
    z = nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)
    return math.ceil(z * z * p_discordant / (margin * margin))
