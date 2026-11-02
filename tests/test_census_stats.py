import math

import pytest

from changegate.census.stats import (
    Proportion,
    _lower_tail,
    _upper_tail,
    clopper_pearson,
    quantile,
)


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [
        (1, 10, 0.0025, 0.4450),  # exact intervals as tabulated, 4 decimals
        (5, 50, 0.0333, 0.2181),
        (10, 20, 0.2720, 0.7280),
    ],
)
def test_clopper_pearson_matches_published_values(k, n, lo, hi):
    got = clopper_pearson(k, n)
    assert got == pytest.approx((lo, hi), abs=1e-4)


def test_zero_and_all_events_use_the_closed_forms():
    assert clopper_pearson(0, 50) == pytest.approx((0.0, 1 - 0.025 ** (1 / 50)), abs=1e-12)
    assert clopper_pearson(50, 50) == pytest.approx((0.025 ** (1 / 50), 1.0), abs=1e-12)


@pytest.mark.parametrize(("k", "n"), [(3, 1000), (7, 30), (1, 2)])
def test_bounds_solve_the_tail_equations(k, n):
    lo, hi = clopper_pearson(k, n, level=0.9)
    assert _upper_tail(k, n, lo) == pytest.approx(0.05, abs=1e-9)
    assert _lower_tail(k, n, hi) == pytest.approx(0.05, abs=1e-9)


@pytest.mark.parametrize(
    ("k", "n", "level"), [(-1, 5, 0.95), (6, 5, 0.95), (0, 0, 0.95), (1, 5, 1.0)]
)
def test_bad_arguments_raise(k, n, level):
    with pytest.raises(ValueError):
        clopper_pearson(k, n, level)


def test_proportion_of_zero_trials_has_no_share():
    p = Proportion.of(0, 0)
    assert (p.share, p.lo, p.hi) == (None, None, None)


def test_proportion_rejects_inconsistent_fields():
    with pytest.raises(ValueError):
        Proportion(k=3, n=2, share=1.5, lo=0, hi=1, level=0.95)
    with pytest.raises(ValueError):
        Proportion(k=0, n=0, share=0.0, lo=0, hi=1, level=0.95)


def test_proportion_of_carries_the_interval():
    p = Proportion.of(5, 50)
    assert p.share == 0.1
    assert (p.lo, p.hi) == clopper_pearson(5, 50)
    assert p.method == "clopper-pearson"


def test_quantile_is_type_7():
    xs = [3.0, 1.0, 2.0, 4.0]
    assert quantile(xs, 0.5) == 2.5
    assert quantile(xs, 0.0) == 1.0
    assert quantile(xs, 1.0) == 4.0
    assert quantile(list(range(1, 11)), 0.95) == pytest.approx(9.55)
    assert quantile([7.0], 0.95) == 7.0


@pytest.mark.parametrize(("values", "q"), [([], 0.5), ([1.0], 1.5), ([1.0, math.nan], 0.5)])
def test_quantile_rejects_bad_input(values, q):
    with pytest.raises(ValueError):
        quantile(values, q)
