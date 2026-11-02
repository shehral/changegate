"""The Check interface every gate check implements, its result type, and envelope verdicts.

Compute tiers, by what a check reads:

- **A**: the adapter file only (CPU, minutes).
- **B**: full weights, streamed, no forward pass (CPU).
- **C**: activations from forward passes of base and candidate (one GPU).
- **D**: generations from a served endpoint, scored by a pinned instrument (GPU to serve).

Under ``change_type: base_upgrade`` only tier D results compare like with like.
"""

from __future__ import annotations

import math
import operator
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from changegate.artifacts import ArtifactRef
from changegate.config import CheckConfig, Envelope

Tier = Literal["A", "B", "C", "D"]
RmId = Annotated[str, StringConstraints(pattern=r"^RM-\d{2}$")]


class Verdict(StrEnum):
    """PASS / WARN / FAIL against the declared envelope, or NO_VERDICT.

    NO_VERDICT means the check ran and its score is reported as a measurement only: no
    envelope was declared, the score is undefined, or the check is informational.

    Does not establish: anything outside the envelope. PASS means the score stayed inside
    the thresholds the user declared for this check.
    """

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    NO_VERDICT = "NO_VERDICT"


class Interval(BaseModel):
    """A confidence interval on a score.

    Does not establish: coverage under conditions other than the stated method's assumptions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lo: float
    hi: float
    level: float = Field(0.95, gt=0, lt=1)
    method: str

    @model_validator(mode="after")
    def _ordered(self) -> Interval:
        """Reject an interval whose lower bound exceeds its upper bound.

        Does not establish: that the interval is the right width.
        """
        if self.lo > self.hi:
            raise ValueError("interval lo must not exceed hi")
        return self


class ThresholdRef(BaseModel):
    """The envelope a verdict was computed against, copied into the result.

    Does not establish: that the envelope was sensible; it records what was used.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    envelope: Envelope
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CheckResult(BaseModel):
    """One check's outcome on one base/candidate pair.

    ``evidence`` holds the check's raw numbers (per-layer scores, item ids, prompt-set
    digests); ``cost`` holds resource use (``wall_seconds``, ``gpu_seconds``,
    ``generations``). ``recall`` is a value the check declares for its recall on a known-bad
    library at the FAIL threshold, or None. No code in this version measures recall, so a
    set value carries no provenance and ``describe`` prints it as declared, not measured.

    Does not establish: absence of the targeted behavior. A score below WARN reads "not
    detected at recall r", and r is only as good as the known-bad library behind it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    rm_ids: tuple[RmId, ...] = Field(min_length=1)
    score: float | None
    interval: Interval | None = None
    threshold_ref: ThresholdRef | None
    verdict: Verdict
    recall: float | None = Field(None, ge=0, le=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> CheckResult:
        """Require the verdict to follow from score and envelope (NO_VERDICT always allowed).

        Does not establish: that the check computed the score correctly.
        """
        expected = apply_envelope(
            self.score, self.threshold_ref.envelope if self.threshold_ref else None
        )
        if self.verdict is not expected and self.verdict is not Verdict.NO_VERDICT:
            raise ValueError(f"verdict {self.verdict} does not follow from score and envelope")
        return self

    def describe(self) -> str:
        """Return a one-line, human-readable statement of the result.

        Returns:
            Text in the tool's claims wording: verdicts are stated against the user-declared
            envelope, and a result under WARN reads "not detected at recall r".

        Does not establish: more than the fields it renders.
        """
        score = "undefined" if self.score is None else f"{self.score:.4g}"
        if self.verdict is Verdict.NO_VERDICT:
            return f"{self.check}: NO VERDICT (score {score}; reported as a measurement only)"
        assert self.threshold_ref is not None  # guaranteed by _consistent
        env = self.threshold_ref.envelope
        line = (
            f"{self.check}: {self.verdict.value} against the user-declared envelope "
            f"(score {score}; WARN {env.warn:.4g}, FAIL {env.fail:.4g}, {env.direction})"
        )
        if self.verdict is Verdict.PASS:
            if self.recall is None:
                line += "; not detected at recall not measured"
            else:
                line += (
                    f"; not detected at recall {self.recall:.2f} (declared by the check; "
                    "not measured in this version)"
                )
        return line


def apply_envelope(score: float | None, envelope: Envelope | None) -> Verdict:
    """Turn a score into a verdict using the declared envelope.

    Reaching a threshold counts: with ``higher_is_worse``, ``score >= fail`` is FAIL and
    ``score >= warn`` is WARN; with ``lower_is_worse`` the comparisons flip.

    Args:
        score: the check's score, or None if undefined.
        envelope: the declared envelope, or None.

    Returns:
        PASS, WARN or FAIL; NO_VERDICT when there is no envelope or the score is None or NaN.

    Does not establish: that the score's uncertainty was considered; the comparison uses the
    point estimate. Checks that report an interval state their decision rule in evidence.
    """
    if envelope is None or score is None or math.isnan(score):
        return Verdict.NO_VERDICT
    worse = operator.ge if envelope.direction == "higher_is_worse" else operator.le
    if worse(score, envelope.fail):
        return Verdict.FAIL
    if worse(score, envelope.warn):
        return Verdict.WARN
    return Verdict.PASS


@runtime_checkable
class Check(Protocol):
    """A gate check: compares a candidate artifact with its base and returns a CheckResult.

    Attributes:
        name: the key used in gate.yaml ``checks``.
        rm_ids: risk-map modes the check targets.
        tier: compute tier (A, B, C or D; see the module docstring).

    Does not establish: coverage of any mode beyond the measurement the check makes.
    """

    name: str
    rm_ids: tuple[RmId, ...]
    tier: Tier

    def run(self, base: ArtifactRef, candidate: ArtifactRef, cfg: CheckConfig) -> CheckResult:
        """Measure the candidate against the base.

        Implementations compute every number from the artifacts and pinned instruments; no
        hosted or agent model produces a score. The verdict comes from ``apply_envelope``.
        Resource use goes in ``cost``.

        Args:
            base: the reference artifact.
            candidate: the changed artifact.
            cfg: this check's config, including its envelope and parameters.

        Returns:
            The result, with ``verdict`` NO_VERDICT when ``cfg.envelope`` is None.

        Does not establish: anything the returned evidence does not show.
        """
        ...
