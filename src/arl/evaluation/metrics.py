"""Metric math: Wilson score intervals for rates and percentiles for latency.
No metric is ever invented: every aggregate is computed from recorded checks
and traces, and carries its n."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Z95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion (95% by default).

    Returns None when total == 0 (an interval over nothing would be a
    fabricated number)."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100]). None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo))


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def intervals_overlap(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool | None:
    if a is None or b is None:
        return None
    return not (a[1] < b[0] or b[1] < a[0])
