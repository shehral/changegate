"""Interval estimates for battery rates and agreement: Wilson, Newcombe, Bonett-Price, kappa.

- Wilson score interval for one proportion (Wilson 1927).
- Newcombe's hybrid score interval for a difference of two proportions from independent
  samples (Newcombe 1998, Statistics in Medicine 17:873, method 10).
- Bonett and Price's adjusted Wald interval for a difference of two proportions on paired
  data, where the same items were scored on both models (Bonett and Price 2012, Journal of
  Educational and Behavioral Statistics 37:479). Chosen over Newcombe's paired hybrid score
  interval, which in simulation covered about 90% at a nominal 95% when flips run one way
  and many items fail on both sides (n 50 to 78), the regression case this gate looks for.
- Cohen's kappa for two binary labelings of the same responses.

Stdlib only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist


def _z(level: float) -> float:
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + level / 2)


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Return the Wilson score interval for k successes in n trials.

    Args:
        k: successes (here, failed items), 0 <= k <= n.
        n: trials, at least 1.
        level: two-sided confidence level.

    Returns:
        (lower, upper), within [0, 1]; the lower bound is 0 when k is 0 and the upper is 1
        when k is n.

    Raises:
        ValueError: if n < 1, k is outside [0, n], or level is outside (0, 1).

    Does not establish: coverage when items are not independent draws, for example when a
    battery holds many samples of the same question.
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    z = _z(level)
    p, z2 = k / n, z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi


def newcombe_unpaired(
    k1: int, n1: int, k2: int, n2: int, level: float = 0.95
) -> tuple[float, float]:
    """Return Newcombe's hybrid score interval for p1 - p2 from two independent samples.

    Args:
        k1: successes in the first sample.
        n1: size of the first sample.
        k2: successes in the second sample.
        n2: size of the second sample.
        level: two-sided confidence level.

    Returns:
        (lower, upper) for p1 - p2, within [-1, 1].

    Raises:
        ValueError: as ``wilson_interval`` does, for either sample.

    Does not establish: coverage for paired data; use ``bonett_price_paired`` when the same
    items were scored on both sides.
    """
    l1, u1 = wilson_interval(k1, n1, level)
    l2, u2 = wilson_interval(k2, n2, level)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return max(-1.0, lo), min(1.0, hi)


def bonett_price_paired(
    both: int, first_only: int, second_only: int, neither: int, level: float = 0.95
) -> tuple[float, float]:
    """Return the Bonett-Price adjusted Wald interval for p1 - p2 on paired binary outcomes.

    The four counts are the cells of the 2x2 table of the same items scored twice: success
    on both sides, on the first only, on the second only, and on neither, so that
    p1 - p2 = (first_only - second_only) / n. One is added to each discordant cell and two
    to n before the Wald interval is formed.

    Args:
        both: items with a success on both sides.
        first_only: items with a success on the first side only.
        second_only: items with a success on the second side only.
        neither: items with a success on neither side.
        level: two-sided confidence level.

    Returns:
        (lower, upper) for p1 - p2, within [-1, 1]. The interval is centred on the adjusted
        difference, which sits slightly closer to 0 than the observed one.

    Raises:
        ValueError: if a count is negative, all are zero, or level is outside (0, 1).

    Does not establish: coverage when items are not exchangeable draws from one population;
    the interval is about these items' pass rates, not a deployment's.
    """
    cells = (both, first_only, second_only, neither)
    if min(cells) < 0 or sum(cells) == 0:
        raise ValueError(f"need non-negative counts with a positive total, got {cells}")
    z = _z(level)
    m = sum(cells) + 2
    p12, p21 = (first_only + 1) / m, (second_only + 1) / m
    d = p12 - p21
    half = z * math.sqrt(max(0.0, (p12 + p21 - d * d) / m))
    return max(-1.0, d - half), min(1.0, d + half)


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Return Cohen's kappa between two binary labelings of the same items.

    Args:
        a: the first labeling (for example, the judge's pass/fail).
        b: the second labeling, same length and order (for example, a human's).

    Returns:
        kappa in [-1, 1], or None when chance agreement is 1 (both labelings use one and the
        same class throughout), where kappa is undefined.

    Raises:
        ValueError: if the labelings are empty or differ in length.

    Does not establish: that either labeling is correct; kappa measures agreement only, and
    labels from one person are themselves noisy.
    """
    if not a or len(a) != len(b):
        raise ValueError("need two non-empty labelings of equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)
