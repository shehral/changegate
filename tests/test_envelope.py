import math

import pytest

from changegate.checks import Verdict
from changegate.config import CheckConfig, Envelope
from changegate.envelope import (
    MIN_LIBRARY_FOR_FAIL,
    MIN_LIBRARY_FOR_WARN,
    CalibrationError,
    aggregate,
    calibrate,
)


def library(n, start=0):
    return {f"change-{i:03d}": float(start + i) for i in range(n)}


def test_higher_is_worse_uses_p95_and_p99_from_100_changes():
    cal = calibrate("adapter-diff", "org/base@rev", library(100), "higher_is_worse")
    # type-7 interpolation on 0..99: p95 sits between 94 and 95, p99 between 98 and 99
    assert cal.warn == pytest.approx(94.05)
    assert cal.fail == pytest.approx(98.01)
    assert (cal.warn_percentile, cal.fail_percentile) == (95, 99)
    assert cal.library_warn_rate == pytest.approx(0.05)
    assert cal.library_fail_rate == pytest.approx(0.01)
    assert cal.n == 100


def test_lower_is_worse_uses_the_low_tail():
    cal = calibrate("drift.refusal", "org/base@rev", library(100), "lower_is_worse")
    assert cal.warn == pytest.approx(4.95)
    assert cal.fail == pytest.approx(0.99)
    assert (cal.warn_percentile, cal.fail_percentile) == (5, 1)
    assert cal.library_warn_rate == pytest.approx(0.05)
    assert cal.library_fail_rate == pytest.approx(0.01)


def test_under_30_changes_is_refused():
    with pytest.raises(CalibrationError, match="at least 30"):
        calibrate("adapter-diff", "b", library(MIN_LIBRARY_FOR_WARN - 1), "higher_is_worse")


def test_under_100_changes_needs_the_users_fail_line():
    with pytest.raises(CalibrationError, match="fail_fallback"):
        calibrate("adapter-diff", "b", library(50), "higher_is_worse")
    cal = calibrate("adapter-diff", "b", library(50), "higher_is_worse", fail_fallback=60.0)
    assert cal.fail == 60.0
    assert cal.fail_percentile is None
    assert cal.library_fail_rate == 0.0


def test_fallback_is_ignored_from_100_changes():
    cal = calibrate(
        "adapter-diff", "b", library(MIN_LIBRARY_FOR_FAIL), "higher_is_worse", fail_fallback=1e6
    )
    assert cal.fail_percentile == 99
    assert cal.fail == pytest.approx(98.01)


def test_fallback_inside_the_warn_line_is_refused():
    with pytest.raises(CalibrationError, match="fail must be at or beyond warn"):
        calibrate("adapter-diff", "b", library(50), "higher_is_worse", fail_fallback=1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_scores_are_refused(bad):
    scores = library(40)
    scores["change-000"] = bad
    with pytest.raises(CalibrationError, match="finite"):
        calibrate("adapter-diff", "b", scores, "higher_is_worse", fail_fallback=100.0)


def test_non_finite_fallback_is_refused():
    with pytest.raises(CalibrationError):
        calibrate("adapter-diff", "b", library(40), "higher_is_worse", fail_fallback=math.inf)


def test_ties_show_up_in_the_library_rates():
    scores = dict.fromkeys(library(100), 0.0)
    cal = calibrate("battery.over-refusal", "b", scores, "higher_is_worse")
    assert cal.warn == cal.fail == 0.0
    assert cal.library_warn_rate == 1.0  # every benign change reaches a WARN line set at a tie


def test_library_digest_ignores_order_and_tracks_content():
    scores = library(40)
    reordered = dict(reversed(list(scores.items())))
    a = calibrate("x", "b", scores, "higher_is_worse", fail_fallback=100.0)
    b = calibrate("x", "b", reordered, "higher_is_worse", fail_fallback=100.0)
    assert a.library_sha256 == b.library_sha256
    scores["change-000"] = 0.5
    c = calibrate("x", "b", scores, "higher_is_worse", fail_fallback=100.0)
    assert c.library_sha256 != a.library_sha256
    other_base = calibrate("x", "b2", library(40), "higher_is_worse", fail_fallback=100.0)
    assert other_base.library_sha256 != a.library_sha256


def test_as_envelope_goes_into_a_check_config():
    cal = calibrate("adapter-diff", "b", library(100), "higher_is_worse")
    env = cal.as_envelope()
    assert env == Envelope(warn=cal.warn, fail=cal.fail, direction="higher_is_worse")
    assert CheckConfig(envelope=env).envelope.source == "user-declared"


def test_calibration_round_trips_through_json():
    cal = calibrate("adapter-diff", "b", library(60), "higher_is_worse", fail_fallback=80.0)
    assert type(cal).model_validate_json(cal.model_dump_json()) == cal


P, W, F, N = Verdict.PASS, Verdict.WARN, Verdict.FAIL, Verdict.NO_VERDICT


@pytest.mark.parametrize(
    "gated,expected",
    [
        ({}, N),
        ({"a": P}, P),
        ({"a": P, "b": W}, W),
        ({"a": W, "b": F}, F),
        ({"a": F, "b": N}, F),
        ({"a": W, "b": N}, N),
        ({"a": P, "b": N}, N),
    ],
)
def test_aggregate_precedence(gated, expected):
    assert aggregate(gated)[0] is expected


def test_aggregate_names_every_check_by_verdict():
    verdict, reason = aggregate({"a": P, "b": F, "c": N, "d": W, "e": F})
    assert verdict is F
    assert reason == "FAIL on: b, e; no verdict on: c; WARN on: d; PASS on: a"
    assert aggregate({})[1] == "no check with a declared envelope ran"
