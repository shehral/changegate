"""The Drift Report: aggregate tables built from run records, naming no repository.

Tables, per stratum (``declared`` / ``undeclared``) wherever a share is shown:

1. **Coverage**: population, quota, sampled, completed, failed and missing runs.
2. **Battery FAIL**: the share of derivatives with a battery FAIL whose interval excludes
   the base rate (the base's own score on that check), per battery check and for any battery
   check, and the same share replicated by the second-family scorer when it ran.
3. **Known-benign**: FAIL and WARN-or-FAIL rates on a user-supplied known-benign list.
4. **Distributions**: per check and stratum, score quantiles and verdict counts.
5. **Reproducibility**: re-run pairs whose verdicts are identical, and how many ran on
   another node.

Every report carries the sample's analysis flag (``pilot`` or ``confirmatory``) and its
pre-registration id. Records from a different sample, gate configuration or base are refused.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from changegate.census.manifest import Manifest
from changegate.census.runner import Role, RunRecord, is_battery_check, latest
from changegate.census.sample import STRATA, Analysis, Sample, Stratum
from changegate.census.stats import CLOPPER_PEARSON, QUANTILE_METHOD, Proportion, quantile
from changegate.checks import CheckResult, Verdict
from changegate.digests import sha256_json
from changegate.errors import ChangegateError

ANY_BATTERY = "any battery check"
ANY_CHECK = "any check"
QUANTILES = (
    ("min", 0.0),
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.5),
    ("p75", 0.75),
    ("p95", 0.95),
    ("max", 1.0),
)
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Coverage(BaseModel):
    """How much of one stratum was run: sampled, completed, failed and missing.

    ``failed`` counts derivatives whose last attempt failed (the run could not be done);
    ``extra_attempts`` counts retries; ``weights_not_deleted`` counts attempts, re-runs
    included, whose scratch directory could not be deleted.

    Does not establish: why runs failed; each failed record states its error.
    """

    model_config = _FROZEN

    stratum: Stratum
    population: int
    quota: int
    sampled: int
    completed: int
    failed: int
    missing: int
    extra_attempts: int
    weights_not_deleted: int


class BatteryFailRow(BaseModel):
    """Battery FAIL shares for one stratum and one battery check (or any battery check).

    ``fail_excludes_base`` counts FAIL verdicts whose interval lies entirely on the worse
    side of the base rate. ``fail_not_comparable`` counts FAIL results with no interval or
    no base score. ``replicated`` also requires the second-family scorer's FAIL, with its interval
    beyond its own base rate; it is None when replication did not run.

    Does not establish: behavior outside the battery, or a cause; see the report's limits.
    """

    model_config = _FROZEN

    stratum: Stratum
    check: str
    fail: Proportion
    fail_excludes_base: Proportion
    fail_not_comparable: int = Field(ge=0)
    replicated: Proportion | None


class BenignRow(BaseModel):
    """FAIL and WARN-or-FAIL rates on the known-benign list, for one check or any check.

    Does not establish: that the listed repositories are benign; the label is the list's.
    """

    model_config = _FROZEN

    check: str
    fail: Proportion
    warn_or_fail: Proportion


class DistributionRow(BaseModel):
    """One check's scores and verdicts in one stratum.

    ``quantiles`` maps min, p05, p25, p50, p75, p95 and max to values (None when no score
    was defined). In the declared stratum, the FAIL share is the FAIL share among repositories
    whose name, tags or card match a removal keyword; the keyword match is not a measurement,
    so this share is not a recall.

    Does not establish: a population distribution beyond the sampled, completed runs.
    """

    model_config = _FROZEN

    check: str
    stratum: Stratum
    n: int
    n_scored: int
    quantiles: dict[str, float] | None
    verdicts: dict[Verdict, int]
    fail: Proportion


class Reproducibility(BaseModel):
    """Re-run pairs: how many repeated every verdict, and how many ran on another node.

    Does not establish: reproducibility on hardware or software outside the pairs run.
    """

    model_config = _FROZEN

    pairs: int
    identical_verdicts: Proportion
    on_other_node: int


class DriftReport(BaseModel):
    """The aggregate Drift Report for one base, one sample and one gate configuration.

    Does not establish: anything about named repositories (none are named), about private
    fine-tunes, or about checks the gate configuration did not run.
    """

    model_config = _FROZEN

    base_id: str
    base_revision: str
    snapshot_at: datetime
    manifest_sha256: str
    sample_sha256: str
    gate_sha256: str
    base_sha256: str
    analysis: Analysis
    prereg_id: str | None
    level: float
    interval_method: str = CLOPPER_PEARSON
    quantile_method: str = QUANTILE_METHOD
    manifest_entries: int
    excluded: dict[str, int]
    known_benign_sha256: str
    known_benign_listed: int
    base_rates: dict[str, float | None]
    coverage: tuple[Coverage, ...]
    battery_fail: tuple[BatteryFailRow, ...]
    known_benign: tuple[BenignRow, ...]
    distributions: tuple[DistributionRow, ...]
    reproducibility: Reproducibility | None


def build_report(
    manifest: Manifest,
    sample: Sample,
    records: Sequence[RunRecord],
    *,
    known_benign: Collection[str] = (),
    level: float = 0.95,
) -> DriftReport:
    """Aggregate run records into the Drift Report tables.

    The last attempt per repository counts. The base rates are the scores of the latest
    completed base record (``CensusRunner.run_base``).

    Args:
        manifest: the manifest the sample was drawn from.
        sample: the sample the records ran.
        records: every record from the results file.
        known_benign: repository ids declared known-benign before the run.
        level: confidence level for every interval.

    Returns:
        The report.

    Raises:
        ChangegateError: if records come from another sample, gate configuration or base,
            name repositories outside the sample, or no completed base record exists.
        ValueError: if a known-benign id is in the declared stratum.

    Does not establish: anything the records do not; see ``render_markdown``'s limits.
    """
    digest = _check_provenance(manifest, sample, records)
    by_key = latest(records)
    base = by_key.get(("base", sample.base_id, False))
    if base is None or base.status != "completed":
        raise ChangegateError("no completed base record; run CensusRunner.run_base first")
    benign = frozenset(known_benign)
    declared = {i.repo_id for i in sample.items if i.stratum == "declared"}
    if clash := benign & declared:
        raise ValueError(f"{len(clash)} known-benign id(s) are in the declared stratum")
    final = {
        i.repo_id: by_key[k]
        for i in sample.items
        if (k := ("derivative", i.repo_id, False)) in by_key
    }
    done = {rid: r for rid, r in final.items() if r.status == "completed"}
    stratum = {i.repo_id: i.stratum for i in sample.items}
    strata = {s: [r for rid, r in done.items() if stratum[rid] == s] for s in STRATA}
    checks = sorted({res.check for r in done.values() for res in r.results})
    base_rates = {res.check: res.score for res in base.results}
    rep_rates = None if base.replication is None else {x.check: x.score for x in base.replication}
    return DriftReport(
        base_id=sample.base_id,
        base_revision=sample.base_revision,
        snapshot_at=manifest.snapshot_at,
        manifest_sha256=sample.manifest_sha256,
        sample_sha256=digest,
        gate_sha256=base.gate_sha256,
        base_sha256=base.base_sha256,
        analysis=sample.analysis,
        prereg_id=sample.prereg_id,
        level=level,
        manifest_entries=len(manifest.entries),
        excluded=dict(sorted(Counter(e.reason for e in manifest.excluded).items())),
        known_benign_sha256=sha256_json(sorted(benign)),
        known_benign_listed=len(benign),
        base_rates=base_rates,
        coverage=tuple(_coverage(sample, s, final, records) for s in STRATA),
        battery_fail=tuple(
            _battery_row(s, check, strata[s], base_rates, rep_rates, level)
            for s in STRATA
            for check in [*(c for c in checks if is_battery_check(c)), ANY_BATTERY]
        ),
        known_benign=tuple(
            _benign_row(check, [r for rid, r in done.items() if rid in benign], level)
            for check in [*checks, ANY_CHECK]
        ),
        distributions=tuple(
            _distribution(check, s, strata[s], level) for check in checks for s in STRATA
        ),
        reproducibility=_reproducibility(by_key, level),
    )


def render_markdown(report: DriftReport) -> str:
    """Render the report as Markdown in the tool's claims wording, limits section included.

    Args:
        report: a built report.

    Returns:
        Markdown text.

    Does not establish: more than the report's fields.
    """
    lines = [
        f"# Drift Report: {report.base_id} at {report.base_revision[:12]}",
        "",
        _analysis_line(report),
        "",
        f"- Hub snapshot {report.snapshot_at.date().isoformat()}; manifest sha256 "
        f"{report.manifest_sha256[:16]}, sample {report.sample_sha256[:16]}, gate "
        f"configuration {report.gate_sha256[:16]}, base weights {report.base_sha256[:16]}.",
        f"- Verdicts are PASS / WARN / FAIL against the user-declared envelope in that gate "
        f"configuration. Intervals: {report.level:.0%} two-sided, exact "
        f"({report.interval_method}).",
        "- Aggregates only: no repository is named in this report.",
        f"- Manifest: {report.manifest_entries} entries. Excluded: "
        + ("; ".join(f"{reason}: {n}" for reason, n in report.excluded.items()) or "none")
        + ".",
        "",
        "## Coverage",
        "",
        "| Stratum | In manifest | Quota | Sampled | Completed | Failed (could not be done) "
        "| Missing | Retries | Weights not deleted |",
        "|---|---|---|---|---|---|---|---|---|",
        *(
            f"| {c.stratum} | {c.population} | {c.quota} | {c.sampled} | {c.completed} "
            f"| {c.failed} | {c.missing} | {c.extra_attempts} | {c.weights_not_deleted} |"
            for c in report.coverage
        ),
        "",
        "## Share with a battery FAIL whose interval excludes the base rate",
        "",
        "Base rate: the base model's own score on the same check, from a run of the base "
        "against itself.",
        "",
        "| Stratum | Check | Any FAIL | FAIL, interval excludes base rate "
        "| FAIL not comparable | Replicated by second-family scorer |",
        "|---|---|---|---|---|---|",
        *(
            f"| {r.stratum} | {r.check} | {_prop(r.fail)} | {_prop(r.fail_excludes_base)} "
            f"| {r.fail_not_comparable} "
            f"| {'not run' if r.replicated is None else _prop(r.replicated)} |"
            for r in report.battery_fail
        ),
        "",
        "## Known-benign false-positive rates",
        "",
        f"Known-benign: {report.known_benign_listed} repository ids fixed before the run "
        f"(list sha256 {report.known_benign_sha256[:16]}); rates use the sampled, completed "
        "ones.",
        "",
        "| Check | FAIL | WARN or FAIL |",
        "|---|---|---|",
        *(
            f"| {r.check} | {_prop(r.fail)} | {_prop(r.warn_or_fail)} |"
            for r in report.known_benign
        ),
        "",
        "## Per-check distributions",
        "",
        "In the declared stratum, a check's FAIL share is its FAIL share among repositories "
        "whose name, tags or card match a removal keyword. The keyword match is not a "
        "measurement, so this share is not a recall.",
        "",
        "| Check | Stratum | n (scored) | min | p05 | p25 | p50 | p75 | p95 | max "
        "| PASS / WARN / FAIL / no verdict | FAIL share |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *(_distribution_line(d) for d in report.distributions),
        "",
        "## Reproducibility",
        "",
        _reproducibility_line(report.reproducibility),
        "",
        "## What this report does not establish",
        "",
        *_limits(report),
    ]
    return "\n".join(lines) + "\n"


def _check_provenance(manifest: Manifest, sample: Sample, records: Sequence[RunRecord]) -> str:
    if sample.manifest_sha256 != manifest.sha256():
        raise ChangegateError("the sample was not drawn from this manifest")
    digest = sample.sha256()
    if any(r.sample_sha256 != digest for r in records):
        raise ChangegateError("records from another sample; build one report per sample")
    if len({(r.gate_sha256, r.base_sha256) for r in records}) > 1:
        raise ChangegateError("records mix gate configurations or base weights")
    sampled = {i.repo_id for i in sample.items}
    if any(r.role == "derivative" and r.repo_id not in sampled for r in records):
        raise ChangegateError("a derivative record names a repository outside the sample")
    return digest


def _coverage(
    sample: Sample, stratum: Stratum, final: dict[str, RunRecord], records: Sequence[RunRecord]
) -> Coverage:
    ids = [i.repo_id for i in sample.items if i.stratum == stratum]
    last = [final[i] for i in ids if i in final]
    completed = sum(r.status == "completed" for r in last)
    failed = len(last) - completed
    return Coverage(
        stratum=stratum,
        population=sample.population.get(stratum, 0),
        quota=sample.quotas.get(stratum, 0),
        sampled=len(ids),
        completed=completed,
        failed=failed,
        missing=len(ids) - completed - failed,
        extra_attempts=sum(r.attempt - 1 for r in last),
        weights_not_deleted=sum(
            r.role == "derivative" and not r.weights_deleted and r.repo_id in ids for r in records
        ),
    )


def _excludes(result: CheckResult | None, rate: float | None) -> bool | None:
    """True/False for a FAIL whose interval is/isn't past the rate; None if not comparable."""
    if result is None or result.verdict is not Verdict.FAIL:
        return False
    if result.interval is None or rate is None or result.threshold_ref is None:
        return None
    if result.threshold_ref.envelope.direction == "higher_is_worse":
        return result.interval.lo > rate
    return result.interval.hi < rate


def _battery_row(
    stratum: Stratum,
    check: str,
    records: list[RunRecord],
    base_rates: dict[str, float | None],
    rep_rates: dict[str, float | None] | None,
    level: float,
) -> BatteryFailRow:
    names = None if check == ANY_BATTERY else {check}
    fail = excl = not_comparable = replicated = 0
    n = 0
    for r in records:
        mine = [
            x
            for x in r.results
            if is_battery_check(x.check) and (names is None or x.check in names)
        ]
        if not mine:
            continue
        n += 1
        fail += any(x.verdict is Verdict.FAIL for x in mine)
        verdicts = {x.check: _excludes(x, base_rates.get(x.check)) for x in mine}
        excl += any(v is True for v in verdicts.values())
        not_comparable += sum(v is None for v in verdicts.values())
        if rep_rates is not None:
            rep = {x.check: x for x in r.replication or ()}
            replicated += any(
                v is True and _excludes(rep.get(c), rep_rates.get(c)) is True
                for c, v in verdicts.items()
            )
    return BatteryFailRow(
        stratum=stratum,
        check=check,
        fail=Proportion.of(fail, n, level),
        fail_excludes_base=Proportion.of(excl, n, level),
        fail_not_comparable=not_comparable,
        replicated=None if rep_rates is None else Proportion.of(replicated, n, level),
    )


def _benign_row(check: str, records: list[RunRecord], level: float) -> BenignRow:
    fail = warn = n = 0
    for r in records:
        mine = [x for x in r.results if check == ANY_CHECK or x.check == check]
        if not mine:
            continue
        n += 1
        fail += any(x.verdict is Verdict.FAIL for x in mine)
        warn += any(x.verdict in (Verdict.WARN, Verdict.FAIL) for x in mine)
    return BenignRow(
        check=check, fail=Proportion.of(fail, n, level), warn_or_fail=Proportion.of(warn, n, level)
    )


def _distribution(
    check: str, stratum: Stratum, records: list[RunRecord], level: float
) -> DistributionRow:
    results = [x for r in records for x in r.results if x.check == check]
    scores = [x.score for x in results if x.score is not None and math.isfinite(x.score)]
    verdicts = Counter(x.verdict for x in results)
    return DistributionRow(
        check=check,
        stratum=stratum,
        n=len(results),
        n_scored=len(scores),
        quantiles={k: quantile(scores, q) for k, q in QUANTILES} if scores else None,
        verdicts={v: verdicts.get(v, 0) for v in Verdict},
        fail=Proportion.of(verdicts.get(Verdict.FAIL, 0), len(results), level),
    )


def _reproducibility(
    by_key: Mapping[tuple[Role, str, bool], RunRecord], level: float
) -> Reproducibility | None:
    pairs = []
    for (role, rid, is_rerun), again in by_key.items():
        orig = by_key.get(("derivative", rid, False))
        paired = role == "derivative" and is_rerun and orig is not None
        if paired and orig.status == again.status == "completed":
            pairs.append((orig, again))
    if not pairs:
        return None
    same = sum(
        {x.check: x.verdict for x in a.results} == {x.check: x.verdict for x in b.results}
        for a, b in pairs
    )
    return Reproducibility(
        pairs=len(pairs),
        identical_verdicts=Proportion.of(same, len(pairs), level),
        on_other_node=sum(a.node != b.node for a, b in pairs),
    )


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _prop(p: Proportion) -> str:
    if p.share is None or p.lo is None or p.hi is None:
        return "n = 0"
    return f"{p.k}/{p.n} = {_pct(p.share)} ({_pct(p.lo)} to {_pct(p.hi)})"


def _distribution_line(d: DistributionRow) -> str:
    q = d.quantiles
    cells = " | ".join(f"{q[k]:.4g}" if q else "n/a" for k, _ in QUANTILES)
    v = d.verdicts
    counts = " / ".join(
        str(v[x]) for x in (Verdict.PASS, Verdict.WARN, Verdict.FAIL, Verdict.NO_VERDICT)
    )
    return (
        f"| {d.check} | {d.stratum} | {d.n} ({d.n_scored}) | {cells} | {counts} | {_prop(d.fail)} |"
    )


def _reproducibility_line(r: Reproducibility | None) -> str:
    if r is None:
        return "No re-runs recorded."
    return (
        f"{r.pairs} re-run pairs; every verdict identical in {_prop(r.identical_verdicts)}; "
        f"{r.on_other_node} of {r.pairs} re-ran on a different node."
    )


def _analysis_line(report: DriftReport) -> str:
    if report.analysis == "confirmatory":
        return (
            f"**Confirmatory analysis** under pre-registration `{report.prereg_id}`. "
            "This tool does not detect deviations from the registered plan."
        )
    reg = f" (registered as `{report.prereg_id}`)" if report.prereg_id else " (not registered)"
    return (
        f"**Pilot analysis**{reg}: exploratory, not a test of a pre-registered hypothesis. "
        "Its numbers do not count toward a confirmatory result."
    )


def _limits(report: DriftReport) -> list[str]:
    return [
        "- A FAIL is a score past the FAIL line of the user-declared envelope. It describes "
        "that check's battery, scorer and serving settings, not the model's behavior elsewhere.",
        "- Public Hub derivatives are not private fine-tunes. The census covers each "
        "relation's most-downloaded repositories in the Hub's model tree on "
        f"{report.snapshot_at.date().isoformat()}, safetensors only.",
        "- The declared stratum is a keyword label from repository names, tags and cards. "
        "It is not a measurement, and the undeclared stratum can hold undeclared removals.",
        "- Shares are reported per stratum. The strata were sampled at different rates, so "
        "no pooled share is shown.",
        "- Zero events bound a rate from above only: 0 of n reads as at most the interval's "
        "upper end, not as absence.",
        "- The 'any' rows count a repository once if any check qualifies, so they are at "
        "least as high as any single check's row; no multiple-comparison correction is applied.",
        "- Intervals treat runs as independent draws at one rate. The base rate is a point "
        "value; its own uncertainty is not propagated.",
        "- The known-benign label is the list's, not a measurement.",
    ]
