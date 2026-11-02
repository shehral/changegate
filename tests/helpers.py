"""Small builders shared by tests."""

from changegate.checks import CheckResult, ThresholdRef, Verdict, apply_envelope
from changegate.config import Envelope

DIGEST = "d" * 64


def result(score, envelope=None, recall=None, verdict=None, check="adapter-diff"):
    ref = ThresholdRef(check=check, envelope=envelope, config_sha256=DIGEST) if envelope else None
    return CheckResult(
        check=check,
        rm_ids=("RM-08",),
        score=score,
        threshold_ref=ref,
        verdict=verdict or apply_envelope(score, envelope),
        recall=recall,
        evidence={"layers": [0.1, 0.2]},
        cost={"wall_seconds": 1.0},
    )


HIGHER = Envelope(warn=2.0, fail=3.0, direction="higher_is_worse")
LOWER = Envelope(warn=0.95, fail=0.90, direction="lower_is_worse")
__all__ = ["DIGEST", "HIGHER", "LOWER", "Verdict", "result"]
