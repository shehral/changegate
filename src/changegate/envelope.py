"""The change envelope's arithmetic: proposed WARN and FAIL lines, and the overall verdict.

In this version every envelope in gate.yaml is user-declared. ``calibrate`` proposes WARN and
FAIL lines from a library of known-benign changes on one base; the user adopts them by writing
them into gate.yaml, where they become user-declared. ``aggregate`` turns the verdicts of the
checks that have an envelope into the gate's overall verdict.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from changegate.checks import Verdict, apply_envelope
from changegate.config import Direction, Envelope
from changegate.digests import sha256_json
from changegate.errors import ChangegateError

WARN_PERCENTILE = 95
FAIL_PERCENTILE = 99
MIN_LIBRARY_FOR_WARN = 30
MIN_LIBRARY_FOR_FAIL = 100

# Overall precedence: a FAIL anywhere wins; a gated check with no verdict outranks WARN, so an
# incomplete run never exits 0 on the strength of the checks that did finish.
_PRECEDENCE = (
    (Verdict.FAIL, "FAIL"),
    (Verdict.NO_VERDICT, "no verdict"),
    (Verdict.WARN, "WARN"),
    (Verdict.PASS, "PASS"),
)


class CalibrationError(ChangegateError):
    """The benign library cannot support the requested WARN or FAIL line.

    Does not establish: anything about the changes in the library.
    """


class Calibration(BaseModel):
    """WARN and FAIL lines proposed from one check's scores on known-benign changes to one base.

    ``warn`` sits at the benign library's 95th percentile when higher scores are worse (the 5th
    when lower is worse); ``fail`` at the 99th (or 1st) once the library holds at least 100
    changes, and otherwise is the user's own fallback, marked by ``fail_percentile`` None.
    Percentiles use linear interpolation between order statistics (Hyndman-Fan type 7).
    ``library_warn_rate`` and ``library_fail_rate`` are the shares of library scores at or
    beyond each line under the gate's own rule; ties can push them above the nominal 5% / 1%.

    Does not establish: the WARN or FAIL rate on benign changes outside the library, that the
    library's changes are benign, that public derivatives resemble private fine-tunes, or an
    overall false-WARN rate across several checks (with k checks it can reach k times 5%).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str = Field(min_length=1)
    base_ref: str = Field(min_length=1)
    n: int = Field(ge=MIN_LIBRARY_FOR_WARN)
    library_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    direction: Direction
    warn: float
    warn_percentile: float
    fail: float
    fail_percentile: float | None
    library_warn_rate: float = Field(ge=0, le=1)
    library_fail_rate: float = Field(ge=0, le=1)

    def as_envelope(self) -> Envelope:
        """Return the envelope to declare in gate.yaml for this check.

        Returns:
            An Envelope with this calibration's lines. The schema labels it user-declared:
            in this version a user adopts calibrated lines by declaring them.

        Does not establish: provenance in results.json; the calibration's digest and size
        travel only with this object.
        """
        return Envelope(warn=self.warn, fail=self.fail, direction=self.direction)


def calibrate(
    check: str,
    base_ref: str,
    benign_scores: Mapping[str, float],
    direction: Direction,
    fail_fallback: float | None = None,
) -> Calibration:
    """Propose WARN and FAIL lines for one check on one base from known-benign changes.

    Args:
        check: the gate.yaml check key the scores come from.
        base_ref: the base every change in the library derives from (repo id with revision).
        benign_scores: change id -> that change's score on the check, at least 30 of them.
        direction: which way is worse for this check.
        fail_fallback: the user's FAIL line, required when the library holds fewer than 100
            changes and ignored from 100 on, where the 99th percentile is used.

    Returns:
        The proposed lines with the library's size, digest and in-library WARN / FAIL rates.

    Raises:
        CalibrationError: under 30 changes, a non-finite score, no fallback under 100
            changes, or a fallback that sits inside the WARN line.

    Does not establish: that the proposed lines suit any change outside this library; see
    Calibration.
    """
    n = len(benign_scores)
    if n < MIN_LIBRARY_FOR_WARN:
        raise CalibrationError(
            f"{check}: {n} benign changes; a WARN line needs at least {MIN_LIBRARY_FOR_WARN}"
        )
    values = [float(v) for v in benign_scores.values()]
    if not all(math.isfinite(v) for v in values):
        raise CalibrationError(f"{check}: every benign score must be a finite number")
    cuts = statistics.quantiles(values, n=100, method="inclusive")  # cuts[k - 1] = k-th pct
    high = direction == "higher_is_worse"
    warn_pct = WARN_PERCENTILE if high else 100 - WARN_PERCENTILE
    fail_pct: int | None
    if n >= MIN_LIBRARY_FOR_FAIL:
        fail_pct = FAIL_PERCENTILE if high else 100 - FAIL_PERCENTILE
        fail = cuts[fail_pct - 1]
    elif fail_fallback is None:
        raise CalibrationError(
            f"{check}: {n} benign changes; a FAIL line from the library needs at least "
            f"{MIN_LIBRARY_FOR_FAIL}, so pass your own fail_fallback"
        )
    else:
        fail, fail_pct = float(fail_fallback), None
    try:
        envelope = Envelope(warn=cuts[warn_pct - 1], fail=fail, direction=direction)
    except ValidationError as exc:
        raise CalibrationError(f"{check}: {exc}") from exc
    verdicts = [apply_envelope(v, envelope) for v in values]
    return Calibration(
        check=check,
        base_ref=base_ref,
        n=n,
        library_sha256=sha256_json(
            {"check": check, "base_ref": base_ref, "scores": sorted(benign_scores.items())}
        ),
        direction=direction,
        warn=envelope.warn,
        warn_percentile=warn_pct,
        fail=envelope.fail,
        fail_percentile=fail_pct,
        library_warn_rate=sum(v is not Verdict.PASS for v in verdicts) / n,
        library_fail_rate=sum(v is Verdict.FAIL for v in verdicts) / n,
    )


def aggregate(gated: Mapping[str, Verdict]) -> tuple[Verdict, str]:
    """Combine the verdicts of the checks that have a declared envelope into one.

    Precedence: any FAIL gives FAIL; otherwise any gated check without a verdict (it declined,
    its score was undefined, or it raised) gives NO_VERDICT; otherwise any WARN gives WARN;
    otherwise PASS. No gated check at all gives NO_VERDICT.

    Args:
        gated: check key -> verdict, for checks with an envelope only, in config order. A
            check that raised is passed as NO_VERDICT. Checks without an envelope are
            informational and are left out by the caller.

    Returns:
        The overall verdict and a reason naming the checks behind each verdict present.

    Does not establish: anything about checks that were not run or have no envelope; a PASS
    covers only the checks named in the reason.
    """
    if not gated:
        return Verdict.NO_VERDICT, "no check with a declared envelope ran"
    present = [
        (verdict, label, [name for name, v in gated.items() if v is verdict])
        for verdict, label in _PRECEDENCE
    ]
    present = [p for p in present if p[2]]
    reason = "; ".join(f"{label} on: {', '.join(names)}" for _, label, names in present)
    return present[0][0], reason
