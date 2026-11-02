"""Interval and quantile arithmetic for the Drift Report, standard library only."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

CLOPPER_PEARSON = "clopper-pearson"
QUANTILE_METHOD = "linear interpolation between order statistics (Hyndman-Fan type 7)"
_BISECTIONS = 100


class Proportion(BaseModel):
    """k of n, the share k/n and its exact two-sided Clopper-Pearson interval.

    With n = 0 the share and the interval are None.

    Does not establish: coverage beyond independent draws at one common rate. A census
    stratum is sampled without replacement from a finite list; the interval ignores the
    finite-population correction, which makes it wider than needed, not narrower.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(ge=0)
    n: int = Field(ge=0)
    share: float | None
    lo: float | None
    hi: float | None
    level: float = Field(gt=0, lt=1)
    method: str = CLOPPER_PEARSON

    @model_validator(mode="after")
    def _consistent(self) -> Proportion:
        """Reject k > n and a share or interval that does not match k and n.

        Does not establish: that k and n were counted correctly.
        """
        if self.k > self.n:
            raise ValueError("k must not exceed n")
        if (self.n == 0) != (self.share is None):
            raise ValueError("share is None exactly when n is 0")
        return self

    @classmethod
    def of(cls, k: int, n: int, level: float = 0.95) -> Proportion:
        """Build the proportion k/n with its Clopper-Pearson interval.

        Args:
            k: the count of events.
            n: the number of trials.
            level: two-sided confidence level.

        Returns:
            The proportion; share and interval are None when n is 0.

        Does not establish: anything about units outside the n counted.
        """
        if n == 0:
            return cls(k=k, n=n, share=None, lo=None, hi=None, level=level)
        lo, hi = clopper_pearson(k, n, level)
        return cls(k=k, n=n, share=k / n, lo=lo, hi=hi, level=level)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Return the exact two-sided binomial interval for k events in n trials.

    The bounds solve P(X >= k | lo) = a/2 and P(X <= k | hi) = a/2 with a = 1 - level, by
    bisection on binomial tail sums computed in log space. With k = 0 the lower bound is 0 and
    the upper bound is 1 - (a/2)^(1/n) (for n = 50 at 95%, about 0.071, close to the rule of
    three's 3/n).

    Args:
        k: events, 0 <= k <= n.
        n: trials, n >= 1.
        level: two-sided confidence level, strictly between 0 and 1.

    Returns:
        (lo, hi), each accurate to well below 1e-12.

    Raises:
        ValueError: if the counts or the level are out of range.

    Does not establish: that the trials were independent or shared one rate.
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1")
    tail = (1 - level) / 2
    lo = 0.0 if k == 0 else _bisect(lambda p: _upper_tail(k, n, p) < tail)
    hi = 1.0 if k == n else _bisect(lambda p: _lower_tail(k, n, p) > tail)
    return lo, hi


def quantile(values: Sequence[float], q: float) -> float:
    """Return the q-quantile of the values by linear interpolation (Hyndman-Fan type 7).

    Args:
        values: finite numbers, in any order.
        q: the quantile, 0 <= q <= 1.

    Returns:
        The interpolated quantile; with one value, that value.

    Raises:
        ValueError: if there are no values, a value is not finite, or q is out of range.

    Does not establish: a population quantile; with fewer than about 1/(1-q) values an upper
    quantile is an interpolation between the top order statistics, not an estimate of a tail.
    """
    if not values:
        raise ValueError("quantile of no values")
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")
    xs = sorted(values)
    if not all(math.isfinite(x) for x in xs):
        raise ValueError("values must be finite")
    h = (len(xs) - 1) * q
    i = math.floor(h)
    if i + 1 >= len(xs):
        return xs[-1]
    return xs[i] + (h - i) * (xs[i + 1] - xs[i])


def _log_pmf(i: int, n: int, p: float) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * math.log(p)
        + (n - i) * math.log1p(-p)
    )


def _lower_tail(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), 0 < p < 1."""
    return math.fsum(math.exp(_log_pmf(i, n, p)) for i in range(k + 1))


def _upper_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), 0 < p < 1."""
    return math.fsum(math.exp(_log_pmf(i, n, p)) for i in range(k, n + 1))


def _bisect(below: Callable[[float], bool]) -> float:
    """Find the p in (0, 1) where ``below(p)`` switches from True to False."""
    lo, hi = 0.0, 1.0
    for _ in range(_BISECTIONS):
        mid = (lo + hi) / 2
        if mid in (lo, hi):  # float resolution reached; never evaluate at 0 or 1
            break
        if below(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
