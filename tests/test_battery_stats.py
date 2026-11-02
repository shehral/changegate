"""Wilson, Newcombe and kappa reference values were computed with statsmodels 0.14
(proportion_confint(method="wilson"), confint_proportions_2indep(method="newcomb")) and
scikit-learn (cohen_kappa_score). Bonett-Price values follow its closed form."""

import pytest

from changegate.battery.stats import (
    bonett_price_paired,
    cohen_kappa,
    newcombe_unpaired,
    wilson_interval,
)


@pytest.mark.parametrize(
    "k,n,level,expected",
    [
        (0, 10, 0.95, (0.0, 0.27753279986288926)),
        (3, 10, 0.95, (0.10779126740630104, 0.6032218525388546)),
        (10, 10, 0.95, (0.7224672001371106, 1.0)),
        (7, 200, 0.95, (0.01705554200856719, 0.07047061152229073)),
        (1, 1, 0.95, (0.2065493143772374, 1.0)),
        (50, 100, 0.95, (0.4038315303659956, 0.5961684696340044)),
        (7, 200, 0.90, (0.019085848319896202, 0.06332701108904515)),
    ],
)
def test_wilson_matches_reference(k, n, level, expected):
    assert wilson_interval(k, n, level) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("k,n,level", [(0, 0, 0.95), (3, 2, 0.95), (-1, 5, 0.95), (1, 5, 1.0)])
def test_wilson_rejects_bad_input(k, n, level):
    with pytest.raises(ValueError):
        wilson_interval(k, n, level)


@pytest.mark.parametrize(
    "args,expected",
    [
        ((12, 100, 5, 100), (-0.009458882754380615, 0.15314235948872298)),
        ((0, 50, 0, 50), (-0.07134759913335874, 0.07134759913335874)),
        ((30, 60, 10, 60), (0.16612339366654427, 0.4763380328851414)),
    ],
)
def test_newcombe_unpaired_matches_reference(args, expected):
    assert newcombe_unpaired(*args) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "cells,level,expected",
    [
        # hand check: p12 = 8/102, p21 = 2/102, se = sqrt((10/102 - (6/102)^2) / 102)
        ((5, 7, 1, 87), 0.95, (-0.0008587473953335367, 0.11850580621886295)),
        ((5, 7, 1, 87), 0.90, (0.008736583812142089, 0.10891047501138731)),
        ((0, 40, 0, 60), 0.95, (0.2935881490492711, 0.49072557644092496)),
        ((3, 0, 9, 8), 0.95, (-0.6500964390885804, -0.16808537909323767)),
    ],
)
def test_bonett_price_values(cells, level, expected):
    assert bonett_price_paired(*cells, level=level) == pytest.approx(expected, abs=1e-12)


def test_bonett_price_uses_only_discordant_cells_and_n():
    assert bonett_price_paired(5, 7, 1, 87) == bonett_price_paired(0, 7, 1, 92)


def test_bonett_price_is_antisymmetric_and_centred_without_flips():
    lo, hi = bonett_price_paired(5, 7, 1, 87)
    assert bonett_price_paired(5, 1, 7, 87) == pytest.approx((-hi, -lo))
    lo, hi = bonett_price_paired(40, 0, 0, 60)
    assert lo == pytest.approx(-hi) and hi > 0.02  # no flips still leaves width


def test_bonett_price_contains_the_observed_difference():
    for cells in [(5, 7, 1, 87), (0, 0, 0, 100), (0, 30, 0, 0), (40, 0, 0, 0), (3, 0, 9, 8)]:
        d = (cells[1] - cells[2]) / sum(cells)
        lo, hi = bonett_price_paired(*cells)
        assert -1 <= lo <= d <= hi <= 1


@pytest.mark.parametrize("cells", [(0, 0, 0, 0), (-1, 2, 3, 4)])
def test_bonett_price_rejects_bad_counts(cells):
    with pytest.raises(ValueError):
        bonett_price_paired(*cells)


def test_kappa_matches_reference():
    a = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    b = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0]
    assert cohen_kappa([bool(x) for x in a], [bool(x) for x in b]) == pytest.approx(0.4)


def test_kappa_edge_cases():
    assert cohen_kappa([True, False], [True, False]) == pytest.approx(1.0)
    assert cohen_kappa([True, True], [True, True]) is None
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])
