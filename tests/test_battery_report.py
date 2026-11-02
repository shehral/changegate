import json

import pytest

from battery_fakes import model_ref, run_of
from changegate.battery import Decoding
from changegate.battery.agreement import Agreement
from changegate.battery.report import (
    PAIRED_METHOD,
    UNPAIRED_METHOD,
    battery_result,
    compare_runs,
    rate_summary,
)
from changegate.checks import Verdict, apply_envelope
from changegate.config import CheckConfig, Envelope
from changegate.gate import Arm

CFG = CheckConfig(envelope=Envelope(warn=0.05, fail=0.10, direction="higher_is_worse"))


def runs(tmp_path, base, cand, **kw):
    m = model_ref(tmp_path, "m")
    return run_of(m, base, **kw), run_of(m, cand, tag="c", **kw)


def result(tmp_path, base, cand, cfg=CFG, run_kw=None, **kw):
    b, c = runs(tmp_path, base, cand, **(run_kw or {}))
    return battery_result("battery.safety-retention", ("RM-08",), compare_runs(b, c), cfg, **kw)


def test_a_large_regression_resolves_to_fail_with_flips(tmp_path):
    r = result(tmp_path, [True] * 100, [False] * 40 + [True] * 60)
    assert r.verdict is Verdict.FAIL
    assert r.score == pytest.approx(0.4)
    assert r.interval.method == PAIRED_METHOD
    assert r.interval.lo > 0.10
    assert len(r.evidence["negative_flips"]) == 40
    assert r.evidence["verdict_reason"] is None
    assert Arm(wall_seconds=1.0, result=r).negative_flips == 40  # the gate's reading


def test_no_change_on_many_items_resolves_to_pass(tmp_path):
    r = result(tmp_path, [True] * 400, [True] * 400)
    assert r.verdict is Verdict.PASS
    assert r.interval.hi < 0.05
    assert "not detected at recall not measured" in r.describe()


def test_an_interval_across_a_line_gives_no_verdict(tmp_path):
    r = result(tmp_path, [True] * 20, [False] + [True] * 19)
    assert r.verdict is Verdict.NO_VERDICT
    assert r.evidence["verdict_reason"] == "the interval spans an envelope line"
    assert r.score == pytest.approx(0.05)
    assert r.interval is not None


def test_no_envelope_or_interval_only_battery_gives_no_verdict(tmp_path):
    r = result(tmp_path, [True] * 100, [False] * 50 + [True] * 50, cfg=CheckConfig())
    assert (r.verdict, r.threshold_ref) == (Verdict.NO_VERDICT, None)
    assert r.evidence["verdict_reason"] == "no envelope declared"
    r = result(tmp_path, [True] * 100, [False] * 50 + [True] * 50, gives_verdicts=False)
    assert r.verdict is Verdict.NO_VERDICT
    assert "intervals only" in r.evidence["verdict_reason"]


def test_too_many_unscored_items_gives_no_verdict(tmp_path):
    base = [True] * 100
    cand = [None] * 10 + [False] * 40 + [True] * 50
    r = result(tmp_path, base, cand)
    assert r.verdict is Verdict.NO_VERDICT
    assert "unscored" in r.evidence["verdict_reason"]
    assert r.evidence["unscored"] == {"base": 0, "candidate": 10}
    assert r.evidence["candidate"]["n"] == 90  # paired rates use items scored on both
    assert result(tmp_path, base, cand, max_unscored_fraction=0.2).verdict is Verdict.FAIL


def test_controls_are_reported_beside_the_score_not_in_it(tmp_path):
    cats = ["opinion"] * 50 + ["neutral"] * 50
    kw = {"categories": cats, "controls": ("neutral",)}
    r = result(tmp_path, [True] * 100, [True] * 50 + [False] * 50, run_kw=kw)
    assert r.score == 0
    assert r.evidence["controls"]["neutral"]["candidate"]["failed"] == 50
    assert r.evidence["negative_flips_by_category"] == {"neutral": 50}
    assert r.evidence["n_items"] == 50


def test_sampled_runs_are_compared_unpaired_without_flips(tmp_path):
    kw = {"decoding": Decoding(temperature=1.0)}
    r = result(tmp_path, [True] * 60, [False] * 30 + [True] * 30, run_kw=kw, gives_verdicts=False)
    assert r.interval.method == UNPAIRED_METHOD
    assert r.evidence["negative_flips"] is None
    assert Arm(wall_seconds=1.0, result=r).negative_flips is None


def agreement(kappa):
    return Agreement(
        scorer="rule:test@1",
        labels_sha256="f" * 64,
        n=200,
        kappa=kappa,
        observed_agreement=0.9,
        unmatched=0,
        judge_unscored=0,
    )


def test_a_judge_below_the_kappa_bar_leaves_the_battery_unscored(tmp_path):
    r = result(tmp_path, [True] * 100, [False] * 50 + [True] * 50, agreement=agreement(0.5))
    assert (r.verdict, r.score, r.interval) == (Verdict.NO_VERDICT, None, None)
    assert "kappa 0.500" in r.evidence["verdict_reason"]
    r = result(tmp_path, [True] * 100, [False] * 50 + [True] * 50, agreement=agreement(0.8))
    assert r.verdict is Verdict.FAIL
    assert r.evidence["judge_agreement"]["kappa"] == 0.8


def test_required_agreement_that_was_not_measured_gives_no_verdict(tmp_path):
    r = result(tmp_path, [True] * 100, [False] * 50 + [True] * 50, require_agreement=True)
    assert r.verdict is Verdict.NO_VERDICT
    assert r.evidence["judge_agreement"] == "not measured"


def test_nothing_scored_on_both_gives_no_verdict(tmp_path):
    r = result(tmp_path, [None] * 5, [None] * 5, max_unscored_fraction=1.0)
    assert (r.verdict, r.score) == (Verdict.NO_VERDICT, None)
    assert r.evidence["verdict_reason"] == "no item was scored on both models"


@pytest.mark.parametrize("fails", [0, 1, 3, 5, 8, 12, 20, 35])
def test_verdicts_always_follow_from_the_point_estimate(tmp_path, fails):
    r = result(tmp_path, [True] * 60, [False] * fails + [True] * (60 - fails))
    assert r.verdict in (Verdict.NO_VERDICT, apply_envelope(r.score, CFG.envelope))
    json.dumps(r.model_dump(mode="json"))


def test_runs_of_different_batteries_are_not_compared(tmp_path):
    m = model_ref(tmp_path, "m")
    with pytest.raises(ValueError):
        compare_runs(run_of(m, [True]), run_of(m, [True], battery="over-refusal"))


def test_rate_summary_ignores_unscored_outcomes(tmp_path):
    run = run_of(model_ref(tmp_path, "m"), [True, False, None, False])
    s = rate_summary(list(run.outcomes))
    assert (s.failed, s.n, s.rate) == (2, 3, pytest.approx(2 / 3))
    empty = rate_summary([])
    assert (empty.n, empty.rate, empty.interval) == (0, None, None)
