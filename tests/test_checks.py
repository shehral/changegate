import math

import pytest

from changegate.artifacts import ArtifactRef
from changegate.checks import Check, CheckResult, Verdict, apply_envelope
from helpers import HIGHER, LOWER, result


@pytest.mark.parametrize(
    "score,envelope,expected",
    [
        (1.0, HIGHER, Verdict.PASS),
        (2.0, HIGHER, Verdict.WARN),
        (2.9, HIGHER, Verdict.WARN),
        (3.0, HIGHER, Verdict.FAIL),
        (0.99, LOWER, Verdict.PASS),
        (0.95, LOWER, Verdict.WARN),
        (0.90, LOWER, Verdict.FAIL),
        (None, HIGHER, Verdict.NO_VERDICT),
        (math.nan, HIGHER, Verdict.NO_VERDICT),
        (5.0, None, Verdict.NO_VERDICT),
        (3, HIGHER, Verdict.FAIL),
        (1, HIGHER, Verdict.PASS),
    ],
)
def test_apply_envelope(score, envelope, expected):
    assert apply_envelope(score, envelope) is expected


def test_verdict_must_follow_from_envelope():
    with pytest.raises(ValueError):
        result(5.0, HIGHER, verdict=Verdict.PASS)
    with pytest.raises(ValueError):
        result(1.0, None, verdict=Verdict.PASS)


def test_no_verdict_is_always_allowed():
    assert result(5.0, HIGHER, verdict=Verdict.NO_VERDICT).verdict is Verdict.NO_VERDICT


def test_rm_ids_are_validated():
    with pytest.raises(ValueError):
        CheckResult(
            check="x", rm_ids=("RM8",), score=None, threshold_ref=None, verdict=Verdict.NO_VERDICT
        )


def test_describe_uses_claims_wording():
    assert "not detected at recall not measured" in result(1.0, HIGHER).describe()
    declared = result(1.0, HIGHER, recall=0.85).describe()
    assert "not detected at recall 0.85" in declared
    assert "not measured in this version" in declared  # no recall has provenance yet
    assert "FAIL against the user-declared envelope" in result(3.5, HIGHER).describe()
    assert "not detected" not in result(2.5, HIGHER).describe()
    assert "measurement only" in result(1.0).describe()


def test_protocol_is_structural(tmp_path):
    class Stub:
        name = "stub"
        rm_ids = ("RM-13",)
        tier = "B"

        def run(self, base: ArtifactRef, candidate: ArtifactRef, cfg) -> CheckResult:
            return result(None)

    assert isinstance(Stub(), Check)
