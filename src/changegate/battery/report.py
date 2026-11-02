"""From two battery runs to rates, flips and a ``CheckResult`` with intervals.

The score is the candidate's failure rate minus the base's, over the battery's scored items
(control categories and unscored items left out). Under greedy decoding the same items were
scored on both models, so the difference gets the Bonett-Price paired interval and
item-level negative flips are listed (Yan et al., arXiv:2011.09161). Under sampled decoding
the two runs are independent samples: Newcombe's unpaired interval is used and no flips are
listed.

Decision rule: a verdict is given only when the whole interval of the score, and the point
estimate, lie in one envelope region (all PASS, all WARN or all FAIL). Otherwise, and
whenever n is too small to resolve it, the result is NO_VERDICT with the interval reported.
A verdict so given equals the point estimate's, as ``CheckResult`` requires.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict

from changegate.battery import BatteryRun, ItemOutcome, negative_flips
from changegate.battery.agreement import Agreement
from changegate.battery.stats import bonett_price_paired, newcombe_unpaired, wilson_interval
from changegate.checks import CheckResult, Interval, RmId, ThresholdRef, Verdict, apply_envelope
from changegate.config import CheckConfig
from changegate.digests import sha256_json

PAIRED_METHOD = "bonett-price-adjusted-wald-paired"
UNPAIRED_METHOD = "newcombe-hybrid-score"
DECISION_RULE = (
    "a verdict only when the whole interval of the score and its point estimate lie in one "
    "envelope region; otherwise no verdict"
)
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class RateSummary(BaseModel):
    """A failure count, its denominator, the rate and its Wilson interval.

    Does not establish: a rate outside these items.
    """

    model_config = _FROZEN

    failed: int
    n: int
    rate: float | None
    interval: tuple[float, float] | None


def rate_summary(outcomes: list[ItemOutcome], level: float = 0.95) -> RateSummary:
    """Summarize the failure rate of scored outcomes (unscored ones are ignored).

    Args:
        outcomes: item outcomes.
        level: two-sided confidence level.

    Returns:
        Counts, rate and Wilson interval; rate and interval are None with no scored outcome.

    Does not establish: coverage when outcomes are repeated samples of a few prompts.
    """
    scored = [o for o in outcomes if o.passed is not None]
    failed = sum(o.passed is False for o in scored)
    if not scored:
        return RateSummary(failed=0, n=0, rate=None, interval=None)
    return RateSummary(
        failed=failed,
        n=len(scored),
        rate=failed / len(scored),
        interval=wilson_interval(failed, len(scored), level),
    )


class Comparison(BaseModel):
    """Base and candidate failure rates on one battery, their difference, and the flips.

    ``delta`` is candidate rate minus base rate. When ``paired``, both rates are over the
    items scored on both runs. ``negative_flips`` (base passed, candidate failed) and
    ``positive_flips`` cover every item, controls included, and are None when not paired.

    Does not establish: that the difference is caused by the change; compare with the base
    run against itself.
    """

    model_config = _FROZEN

    paired: bool
    n_items: int
    unscored: dict[str, int]
    base: RateSummary
    candidate: RateSummary
    delta: float | None
    delta_interval: tuple[float, float] | None
    method: str
    level: float
    negative_flips: tuple[str, ...] | None
    positive_flips: tuple[str, ...] | None
    negative_flips_by_category: dict[str, int] | None
    controls: dict[str, dict[str, RateSummary]]

    @property
    def unscored_fraction(self) -> float:
        """Return the larger of the two runs' unscored shares of the scored-category items.

        Does not establish: why items went unscored; read the item logs.
        """
        return max(self.unscored.values()) / self.n_items if self.n_items else 1.0


def compare_runs(base: BatteryRun, candidate: BatteryRun, level: float = 0.95) -> Comparison:
    """Compare two runs of one battery item by item.

    Args:
        base: the run on the base.
        candidate: the run on the candidate.
        level: two-sided confidence level.

    Returns:
        The comparison. Uses only the runs, so a record verifier can recompute it.

    Raises:
        ValueError: if the runs used different batteries, items, scorers or decoding.

    Does not establish: anything about items outside the battery.
    """
    if (base.battery, base.battery_version, base.controls) != (
        candidate.battery,
        candidate.battery_version,
        candidate.controls,
    ):
        raise ValueError("runs are of different batteries or versions")
    flips = negative_flips(base, candidate)  # also checks items, scorer and decoding
    paired = base.decoding is None or base.decoding.greedy
    controls = set(base.controls)
    b = [o for o in base.outcomes if o.category not in controls]
    c = [o for o in candidate.outcomes if o.category not in controls]
    unscored = {
        "base": sum(o.passed is None for o in b),
        "candidate": sum(o.passed is None for o in c),
    }
    if paired:
        cand_by_id = {o.item_id: o for o in c}
        pairs = [
            (o, cand_by_id[o.item_id])
            for o in b
            if o.passed is not None and cand_by_id[o.item_id].passed is not None
        ]
        b_scored, c_scored = [p[0] for p in pairs], [p[1] for p in pairs]
    else:
        b_scored, c_scored = b, c
    rb, rc = rate_summary(b_scored, level), rate_summary(c_scored, level)
    delta = interval = None
    if rb.rate is not None and rc.rate is not None:
        delta = rc.rate - rb.rate
        interval = (
            _paired_interval(b_scored, c_scored, level)
            if paired
            else newcombe_unpaired(rc.failed, rc.n, rb.failed, rb.n, level)
        )
    category = {o.item_id: o.category or "" for o in base.outcomes}
    return Comparison(
        paired=paired,
        n_items=len(b),
        unscored=unscored,
        base=rb,
        candidate=rc,
        delta=delta,
        delta_interval=interval,
        method=PAIRED_METHOD if paired else UNPAIRED_METHOD,
        level=level,
        negative_flips=flips if paired else None,
        positive_flips=negative_flips(candidate, base) if paired else None,
        negative_flips_by_category=dict(Counter(category[i] for i in flips)) if paired else None,
        controls={
            name: {
                "base": rate_summary([o for o in base.outcomes if o.category == name], level),
                "candidate": rate_summary(
                    [o for o in candidate.outcomes if o.category == name], level
                ),
            }
            for name in sorted(controls)
        },
    )


def _paired_interval(
    base: list[ItemOutcome], candidate: list[ItemOutcome], level: float
) -> tuple[float, float]:
    cells = Counter(
        (c.passed is False, b.passed is False) for b, c in zip(base, candidate, strict=True)
    )
    return bonett_price_paired(
        both=cells[(True, True)],
        first_only=cells[(True, False)],
        second_only=cells[(False, True)],
        neither=cells[(False, False)],
        level=level,
    )


def battery_result(
    check: str,
    rm_ids: tuple[RmId, ...],
    comparison: Comparison,
    cfg: CheckConfig,
    *,
    gives_verdicts: bool = True,
    agreement: Agreement | None = None,
    require_agreement: bool = False,
    max_unscored_fraction: float = 0.05,
) -> CheckResult:
    """Build a battery's ``CheckResult`` from a comparison and the check's envelope.

    Verdict order: no envelope, a battery that gives no verdicts, a judge below the kappa
    bar (reported unscored: score and interval are None), agreement required but not
    measured, too many unscored items, then the interval rule (see the module docstring).
    Each NO_VERDICT names its reason in ``evidence["verdict_reason"]``.

    Args:
        check: the gate.yaml key (``battery.<name>``).
        rm_ids: the battery's risk-map modes.
        comparison: from ``compare_runs``.
        cfg: the check's config, with its envelope.
        gives_verdicts: False for batteries reported as intervals only.
        agreement: judge agreement, when a labels file was given.
        require_agreement: give no verdict unless agreement was measured.
        max_unscored_fraction: give no verdict when more items than this went unscored.

    Returns:
        The result; ``recall`` is None (not measured on a known-bad library).

    Does not establish: absence of the behavior; a PASS reads "not detected at recall r".
    """
    envelope = cfg.envelope
    score, interval = comparison.delta, comparison.delta_interval
    reason = None
    if envelope is None:
        reason = "no envelope declared"
    elif not gives_verdicts:
        reason = "this battery reports intervals only in this version"
    elif agreement is not None and not agreement.meets_bar:
        kappa = "undefined" if agreement.kappa is None else f"{agreement.kappa:.3f}"
        reason = f"judge kappa {kappa} is below {agreement.bar}; reported unscored"
        score = interval = None
    elif require_agreement and agreement is None:
        reason = "judge agreement not measured; params require it"
    elif comparison.unscored_fraction > max_unscored_fraction:
        reason = (
            f"{comparison.unscored_fraction:.1%} of items unscored, over "
            f"{max_unscored_fraction:.1%}"
        )
    elif interval is None:
        reason = "no item was scored on both models"
    verdict = Verdict.NO_VERDICT
    if reason is None and interval is not None:
        regions = {apply_envelope(x, envelope) for x in (interval[0], score, interval[1])}
        verdict = regions.pop() if len(regions) == 1 else Verdict.NO_VERDICT
        reason = (
            None if verdict is not Verdict.NO_VERDICT else "the interval spans an envelope line"
        )
    evidence: dict[str, Any] = comparison.model_dump(mode="json")
    evidence.update(
        decision_rule=DECISION_RULE,
        verdict_reason=reason,
        judge_agreement="not measured" if agreement is None else agreement.model_dump(),
    )
    return CheckResult(
        check=check,
        rm_ids=rm_ids,
        score=score,
        interval=None
        if interval is None
        else Interval(
            lo=interval[0], hi=interval[1], level=comparison.level, method=comparison.method
        ),
        threshold_ref=None
        if envelope is None
        else ThresholdRef(
            check=check, envelope=envelope, config_sha256=sha256_json(cfg.model_dump(mode="json"))
        ),
        verdict=verdict,
        evidence=evidence,
    )
