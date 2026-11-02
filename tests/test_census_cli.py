import json

from census_fakes import BASE, BASE_REV, world
from changegate.census import cli as census_cli
from changegate.census.manifest import Manifest
from changegate.census.runner import read_records
from changegate.census.sample import Sample
from changegate.checks import CheckResult, ThresholdRef, apply_envelope
from helpers import DIGEST


class FakeBattery:
    name, tier, rm_ids = "battery.safety-retention", "A", ("RM-08",)

    def run(self, base, candidate, cfg):
        score = 0.0 if base.sha256 == candidate.sha256 else 0.02
        env = cfg.envelope
        return CheckResult(
            check=self.name,
            rm_ids=self.rm_ids,
            score=score,
            threshold_ref=ThresholdRef(check=self.name, envelope=env, config_sha256=DIGEST),
            verdict=apply_envelope(score, env),
        )


def discover_and_sample(tmp_path, hub):
    manifest, sample = tmp_path / "manifest.json", tmp_path / "sample.json"
    argv = ["discover", "--base", BASE, "--revision", BASE_REV, "--out", str(manifest)]
    assert census_cli.main(argv, client=hub) == 0
    argv = [
        "sample", "--manifest", str(manifest), "--declared", "2", "--undeclared", "3",
        "--seed", "7", "--analysis", "pilot", "--out", str(sample),
    ]  # fmt: skip
    assert census_cli.main(argv) == 0
    return manifest, sample


def test_discover_then_sample_writes_both_files(tmp_path, capsys):
    manifest, sample = discover_and_sample(tmp_path, world())
    m, s = Manifest.load(manifest), Sample.load(sample)
    assert s.manifest_sha256 == m.sha256() and len(s.items) == 5
    assert "entries" in capsys.readouterr().out


def test_confirmatory_sample_needs_a_registration(tmp_path, capsys):
    manifest, _ = discover_and_sample(tmp_path, world())
    argv = [
        "sample", "--manifest", str(manifest), "--declared", "1", "--undeclared", "1",
        "--seed", "1", "--analysis", "confirmatory", "--out", str(tmp_path / "s2.json"),
    ]  # fmt: skip
    assert census_cli.main(argv) == 2
    assert "prereg" in capsys.readouterr().err


def test_rerun_subset_prints_ids(tmp_path, capsys):
    _, sample = discover_and_sample(tmp_path, world())
    capsys.readouterr()
    argv = ["rerun-subset", "--sample", str(sample), "--fraction", "0.4", "--seed", "3"]
    assert census_cli.main(argv) == 0
    ids = capsys.readouterr().out.split()
    assert len(ids) == 2 and set(ids) <= {i.repo_id for i in Sample.load(sample).items}


def test_run_then_report_end_to_end(tmp_path, monkeypatch):
    hub = world()
    manifest, sample = discover_and_sample(tmp_path, hub)
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base weights")
    gate_yaml = tmp_path / "gate.yaml"
    gate_yaml.write_text(
        "version: 0\n"
        "base: {path: base, kind: weights}\n"
        "candidate: {path: base, kind: weights}\n"
        "change_type: full_finetune\n"
        "checks:\n"
        "  battery.safety-retention:\n"
        "    envelope: {warn: 0.05, fail: 0.10, direction: higher_is_worse}\n"
        f"output: {{dir: out, action_log: {tmp_path / 'actions.jsonl'}}}\n"
    )
    monkeypatch.setattr(
        census_cli,
        "build_registry",
        lambda cfg, log, root: {"battery.safety-retention": FakeBattery()},
    )
    results = tmp_path / "results.jsonl"
    argv = [
        "run", "--manifest", str(manifest), "--sample", str(sample),
        "--base-path", str(base), "--gate-config", str(gate_yaml),
        "--scratch", str(tmp_path / "scratch"), "--results", str(results),
    ]  # fmt: skip
    assert census_cli.main(argv, client=hub) == 0
    records = read_records(results)
    assert [r.role for r in records].count("base") == 1
    assert all(r.status == "completed" and r.results for r in records)
    assert all(r.weights_deleted for r in records if r.role == "derivative")
    assert len({r.gate_sha256 for r in records}) == 1
    report_md = tmp_path / "report" / "drift-report.md"
    argv = [
        "report", "--manifest", str(manifest), "--sample", str(sample),
        "--results", str(results), "--out", str(report_md),
    ]  # fmt: skip
    assert census_cli.main(argv) == 0
    text = report_md.read_text()
    body = json.loads(report_md.with_suffix(".json").read_text())
    assert body["analysis"] == "pilot"
    for item in Sample.load(sample).items:
        assert item.repo_id not in text  # aggregates only
