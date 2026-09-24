"""The bootstrap and the closed form it hands off to must agree at the threshold.

``bootstrap_ci`` switches estimator above ``BOOTSTRAP_MAX_N``. That is a performance
decision about a published statistic, so it needs a test that fails if the two ever stop
describing the same interval — otherwise the width of a reported CI would depend on which
side of an arbitrary constant the sample happened to land.
"""

from __future__ import annotations

import math
import random
import statistics

from distil.shadow import BOOTSTRAP_MAX_N, bootstrap_ci


def _normal(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    half = 1.959963984540054 * statistics.stdev(xs) / math.sqrt(n)
    return (sum(xs) / n - half, sum(xs) / n + half)


def test_the_two_estimators_agree_where_they_meet():
    """Across symmetric, skewed and three-valued samples, at the handoff point."""
    n = BOOTSTRAP_MAX_N
    for trial in range(6):
        rng = random.Random(4242 + trial)
        for name, draw in (
            ("gauss", lambda: rng.gauss(-68.0, 120.0)),
            ("skewed", lambda: rng.expovariate(1 / 50) - 120),
            ("three-valued", lambda: float(rng.choice([-1, 0, 1]))),
        ):
            xs = [draw() for _ in range(n)]
            lo_b, hi_b = bootstrap_ci(xs)  # n == MAX: still the bootstrap
            lo_n, hi_n = _normal(xs)
            width = hi_b - lo_b
            worst = max(abs(lo_b - lo_n), abs(hi_b - hi_n)) / width
            assert worst < 0.15, f"{name} trial {trial}: {worst:.3f} of interval width apart"


def test_published_sample_sizes_still_use_the_bootstrap():
    """Every number this project has published sits at n in the hundreds. The handoff
    must not reach them, or a committed result would move without anyone touching it."""
    assert BOOTSTRAP_MAX_N > 1000
    rng = random.Random(9)
    xs = [float(rng.choice([-1, 0, 1])) for _ in range(398)]  # the live sample's n
    assert bootstrap_ci(xs) == bootstrap_ci(xs), "the bootstrap must stay seeded"
    # identical below the threshold, whatever the threshold is set to
    assert bootstrap_ci(xs) != _normal(xs)


def test_a_constant_sample_has_a_zero_width_interval_on_both_sides():
    """The closed form divides by the spread; a sample with none must not raise."""
    lo, hi = bootstrap_ci([7.0] * (BOOTSTRAP_MAX_N + 1))
    assert lo == hi == 7.0
    lo, hi = bootstrap_ci([7.0] * 10)
    assert lo == hi == 7.0
