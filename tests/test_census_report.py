import importlib.util
import sys
from pathlib import Path

import pytest

from census_fakes import (
    BASE,
    BASE_REV,
    BATTERY,
    GATE_SHA,
    HARMFUL,
    REFUSAL,
    SNAPSHOT,
    FakeGate,
    result,
    world,
)
from changegate.artifacts import ArtifactRef
from changegate.census.manifest import discover
from changegate.census.report import (
    ANY_BATTERY,
    ANY_CHECK,
    DriftReport,
    _excludes,
    build_report,
    render_markdown,
)
from changegate.census.runner import CensusRunner, read_records
from changegate.census.sample import draw_sample
from changegate.checks import Verdict
from changegate.config import Envelope
from changegate.errors import ChangegateError
from changegate.log import ActionLog

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ncl_census", ROOT / "tools" / "never_claim_lint.py")
ncl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ncl
_spec.loader.exec_module(ncl)


def battery(score, interval):
    return result(BATTERY, score, HARMFUL, interval)


def drift(score):
    return result("drift", score, REFUSAL)


class Census:
    """A small census: 4 declared and 6 undeclared derivatives, run through the real runner."""

    def __init__(self, tmp_path, analysis="pilot", prereg_id="osf.io/pilot1", replicate=True):
        self.hub = world()
        self.manifest = discover(self.hub, BASE, BASE_REV, snapshot_at=SNAPSHOT)
        self.sample = draw_sample(
            self.manifest,
            {"declared": 4, "undeclared": 6},
            seed=11,
            analysis=analysis,
            prereg_id=prereg_id,
        )
        self.d = sorted(i.repo_id for i in self.sample.items if i.stratum == "declared")
        self.u = sorted(i.repo_id for i in self.sample.items if i.stratum == "undeclared")
        by_repo = {
            self.d[0]: [battery(0.40, (0.30, 0.50)), drift(0.50)],  # interval excludes base
            self.d[1]: [battery(0.25, (0.03, 0.45)), drift(0.95)],  # FAIL, interval spans base
            self.d[2]: [battery(0.30, None), drift(0.95)],  # FAIL with no interval
            self.d[3]: [battery(0.05, (0.02, 0.09)), drift(0.95)],
            self.u[0]: [battery(0.22, (0.15, 0.30)), drift(0.95)],
            self.u[1]: [battery(0.05, (0.02, 0.09)), drift(0.85)],  # drift WARN
        }
        for repo_id in self.u[2:]:
            by_repo[repo_id] = [battery(0.05, (0.02, 0.09)), drift(0.97)]
        base_results = [battery(0.05, (0.02, 0.09)), drift(1.0)]
        self.gate = FakeGate(by_repo, base_results)
        rep = dict(by_repo)
        rep[self.u[0]] = [battery(0.08, (0.02, 0.15)), drift(0.95)]  # does not replicate
        self.replicate = FakeGate(rep, base_results) if replicate else None
        base_dir = tmp_path / "base"
        base_dir.mkdir(parents=True)
        (base_dir / "model.safetensors").write_bytes(b"base")
        self.base = ArtifactRef.from_path(base_dir, "weights")
        self.tmp = tmp_path
        self.log = ActionLog(tmp_path / "actions.jsonl")
        self.results = tmp_path / "results.jsonl"

    def runner(self, node="node-a"):
        return CensusRunner(
            self.hub,
            self.gate,
            self.log,
            scratch=self.tmp / "scratch",
            results=self.results,
            gate_sha256=GATE_SHA,
            replicate=self.replicate,
            node=node,
        )

    def run_all(self):
        runner = self.runner()
        runner.run_base(self.sample, self.manifest, self.base)
        runner.run(self.sample, self.manifest, self.base)
        return read_records(self.results)

    def report(self, records=None, known_benign=None):
        return build_report(
            self.manifest,
            self.sample,
            self.run_all() if records is None else records,
            known_benign=self.u[1:4] if known_benign is None else known_benign,
        )


def row(report, stratum, check):
    return next(r for r in report.battery_fail if (r.stratum, r.check) == (stratum, check))


def counts(p):
    return (p.k, p.n)


def test_battery_fail_share_needs_an_interval_past_the_base_rate(tmp_path):
    report = Census(tmp_path).report()
    declared = row(report, "declared", BATTERY)
    assert counts(declared.fail) == (3, 4)
    assert counts(declared.fail_excludes_base) == (1, 4)
    assert declared.fail_not_comparable == 1
    assert counts(declared.replicated) == (1, 4)
    undeclared = row(report, "undeclared", BATTERY)
    assert counts(undeclared.fail_excludes_base) == (1, 6)
    assert counts(undeclared.replicated) == (0, 6)
    assert (
        row(report, "undeclared", ANY_BATTERY).fail_excludes_base == undeclared.fail_excludes_base
    )
    assert report.base_rates == {BATTERY: 0.05, "drift": 1.0}


def test_intervals_are_clopper_pearson_at_the_report_level(tmp_path):
    report = Census(tmp_path).report()
    p = row(report, "undeclared", BATTERY).fail_excludes_base
    assert p.share == pytest.approx(1 / 6)
    assert p.lo > 0 and p.hi == pytest.approx(0.6412, abs=1e-4)
    assert report.level == 0.95


def test_replication_reads_not_run_without_a_second_scorer(tmp_path):
    report = Census(tmp_path, replicate=False).report()
    assert all(r.replicated is None for r in report.battery_fail)
    assert "not run" in render_markdown(report)


def test_known_benign_false_positive_rates(tmp_path):
    report = Census(tmp_path).report()
    by_check = {r.check: r for r in report.known_benign}
    assert counts(by_check["drift"].fail) == (0, 3)
    assert counts(by_check["drift"].warn_or_fail) == (1, 3)
    assert counts(by_check[BATTERY].warn_or_fail) == (0, 3)
    assert counts(by_check[ANY_CHECK].warn_or_fail) == (1, 3)
    assert report.known_benign_listed == 3


def test_known_benign_cannot_overlap_the_declared_stratum(tmp_path):
    census = Census(tmp_path)
    with pytest.raises(ValueError, match="declared stratum"):
        census.report(known_benign=[census.d[0]])


def test_per_check_distributions_by_stratum(tmp_path):
    report = Census(tmp_path).report()
    dist = next(d for d in report.distributions if (d.check, d.stratum) == (BATTERY, "declared"))
    assert (dist.n, dist.n_scored) == (4, 4)
    assert dist.quantiles["min"] == 0.05 and dist.quantiles["max"] == 0.40
    assert dist.quantiles["p50"] == pytest.approx(0.275)
    assert dist.verdicts[Verdict.FAIL] == 3 and dist.verdicts[Verdict.PASS] == 1
    assert counts(dist.fail) == (3, 4)  # FAIL share in the keyword-matched stratum
    assert {(d.check, d.stratum) for d in report.distributions} == {
        (c, s) for c in (BATTERY, "drift") for s in ("declared", "undeclared")
    }


def test_every_report_carries_the_analysis_flag_and_registration(tmp_path):
    pilot = Census(tmp_path / "a").report()
    assert (pilot.analysis, pilot.prereg_id) == ("pilot", "osf.io/pilot1")
    assert "**Pilot analysis** (registered as `osf.io/pilot1`)" in render_markdown(pilot)
    conf = Census(tmp_path / "b", analysis="confirmatory", prereg_id="osf.io/conf1").report()
    assert conf.analysis == "confirmatory"
    assert "**Confirmatory analysis** under pre-registration `osf.io/conf1`" in (
        render_markdown(conf)
    )
    bare = Census(tmp_path / "c", prereg_id=None).report()
    assert "(not registered)" in render_markdown(bare)


def test_coverage_counts_failed_missing_and_retried_runs(tmp_path):
    census = Census(tmp_path)
    runner = census.runner()
    runner.run_base(census.sample, census.manifest, census.base)
    census.gate.fail_on = {census.u[0], census.u[1]}
    runner.run(census.sample, census.manifest, census.base, only=census.u[:4] + census.d)
    census.gate.fail_on = {census.u[1]}
    runner.run(census.sample, census.manifest, census.base, only=census.u[:2])
    report = census.report(records=read_records(census.results))
    cov = {c.stratum: c for c in report.coverage}["undeclared"]
    assert (cov.population, cov.quota, cov.sampled) == (6, 6, 6)
    assert (cov.completed, cov.failed, cov.missing) == (3, 1, 2)
    assert cov.extra_attempts == 2
    assert cov.weights_not_deleted == 0
    assert counts(row(report, "undeclared", BATTERY).fail) == (1, 3)


def test_reproducibility_pairs_reruns_with_originals(tmp_path):
    census = Census(tmp_path)
    census.run_all()
    rerunner = census.runner(node="node-b")
    census.gate.by_repo[census.d[3]] = [battery(0.40, (0.30, 0.50)), drift(0.95)]
    rerunner.run(
        census.sample, census.manifest, census.base, only=[census.u[2], census.d[3]], rerun=True
    )
    report = census.report(records=read_records(census.results))
    rep = report.reproducibility
    assert rep.pairs == 2 and rep.on_other_node == 2
    assert counts(rep.identical_verdicts) == (1, 2)
    assert "2 re-run pairs" in render_markdown(report)
    assert Census(tmp_path / "x").report().reproducibility is None


def test_records_from_elsewhere_are_refused(tmp_path):
    census = Census(tmp_path)
    records = census.run_all()
    other = draw_sample(census.manifest, {"declared": 1}, seed=1, analysis="pilot")
    with pytest.raises(ChangegateError, match="another sample"):
        build_report(census.manifest, other, records)
    mixed = [*records[:-1], records[-1].model_copy(update={"gate_sha256": "8" * 64})]
    with pytest.raises(ChangegateError, match="mix gate"):
        build_report(census.manifest, census.sample, mixed)
    stray = records[-1].model_copy(update={"repo_id": "nobody/none"})
    with pytest.raises(ChangegateError, match="outside the sample"):
        build_report(census.manifest, census.sample, [*records, stray])
    no_base = [r for r in records if r.role != "base"]
    with pytest.raises(ChangegateError, match="base record"):
        build_report(census.manifest, census.sample, no_base)
    changed = census.manifest.model_copy(update={"limit_per_relation": 1})
    with pytest.raises(ChangegateError, match="manifest"):
        build_report(changed, census.sample, records)


def test_interval_direction_follows_the_envelope():
    lower = Envelope(warn=0.8, fail=0.6, direction="lower_is_worse")
    refusal = result("battery.refusal", 0.5, lower, (0.4, 0.58))
    assert _excludes(refusal, 0.9) is True
    assert _excludes(refusal, 0.55) is False
    assert _excludes(refusal, None) is None
    assert _excludes(result("battery.refusal", 0.9, lower, (0.85, 0.95)), 0.9) is False
    assert _excludes(None, 0.9) is False


def test_markdown_names_no_repository_and_passes_the_never_claim_lint(tmp_path):
    census = Census(tmp_path)
    text = render_markdown(census.report())
    for repo_id in census.d + census.u:
        assert repo_id not in text
    assert "Share with a battery FAIL whose interval excludes the base rate" in text
    assert "What this report does not establish" in text
    out = tmp_path / "report.md"
    out.write_text(text)
    hits, _ = ncl.lint([out])
    assert hits == [], hits


def test_report_round_trips_as_json(tmp_path):
    report = Census(tmp_path).report()
    assert DriftReport.model_validate_json(report.model_dump_json()) == report
    assert report.excluded["gated (no credentials are used)"] == 1
    assert report.manifest_entries == 10


def test_the_keyword_stratum_fail_share_is_not_called_recall(tmp_path):
    # review M-9: the declared stratum is a keyword match, so its FAIL share is not a recall
    text = render_markdown(Census(tmp_path).report())
    assert "its recall" not in text
    assert "match a removal keyword" in text
    assert "not a recall" in text
