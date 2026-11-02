import json
import os

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
from changegate.census.runner import (
    CensusRunner,
    RunRecord,
    is_battery_check,
    latest,
    read_records,
)
from changegate.census.sample import draw_sample
from changegate.digests import sha256_json
from changegate.errors import ChangegateError
from changegate.log import ActionLog


def passing():
    return [result(BATTERY, 0.05, HARMFUL, (0.02, 0.09)), result("drift", 0.97, REFUSAL)]


def failing():
    return [result(BATTERY, 0.40, HARMFUL, (0.30, 0.50)), result("drift", 0.50, REFUSAL)]


@pytest.fixture
def setup(tmp_path):
    hub = world()
    manifest = discover(hub, BASE, BASE_REV, snapshot_at=SNAPSHOT)
    sample = draw_sample(
        manifest, {"declared": 2, "undeclared": 3}, seed=7, analysis="pilot", prereg_id="osf.io/p1"
    )
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "model.safetensors").write_bytes(b"base weights")
    base = ArtifactRef.from_path(base_dir, "weights")
    by_repo = {i.repo_id: failing() if i.stratum == "declared" else passing() for i in sample.items}
    gate = FakeGate(by_repo, passing())
    log = ActionLog(tmp_path / "state" / "actions.jsonl")

    def runner(**kw):
        kw.setdefault("gate", gate)
        return CensusRunner(
            hub,
            kw.pop("gate"),
            log,
            scratch=tmp_path / "scratch",
            results=tmp_path / "out" / "results.jsonl",
            gate_sha256=GATE_SHA,
            node=kw.pop("node", "node-a"),
            **kw,
        )

    return {
        "hub": hub,
        "manifest": manifest,
        "sample": sample,
        "base": base,
        "gate": gate,
        "log": log,
        "runner": runner,
        "tmp": tmp_path,
    }


def events(log):
    return [json.loads(line)["event"] for line in log.path.read_text().splitlines()]


def test_a_run_gates_records_and_deletes_every_sampled_derivative(setup):
    s = setup
    records = s["runner"]().run(s["sample"], s["manifest"], s["base"])
    assert [r.repo_id for r in records] == [i.repo_id for i in s["sample"].items]
    assert all(r.status == "completed" and r.weights_deleted for r in records)
    assert os.listdir(s["tmp"] / "scratch") == []
    assert read_records(s["tmp"] / "out" / "results.jsonl") == records
    r = records[0]
    assert (r.analysis, r.prereg_id, r.node, r.gate_sha256) == (
        "pilot",
        "osf.io/p1",
        "node-a",
        GATE_SHA,
    )
    assert r.sample_sha256 == s["sample"].sha256()
    assert r.base_sha256 == s["base"].sha256
    assert s["log"].verify().rows == 3 * len(records)
    assert events(s["log"]).count("census.delete") == len(records)


def test_the_gate_sees_artifacts_and_change_type_only(setup):
    s = setup
    s["runner"]().run(s["sample"], s["manifest"], s["base"])
    for path, change_type in s["gate"].calls:
        repo_id = path.name.split("@")[0].replace("--", "/")
        assert change_type == s["manifest"].entry(repo_id).change_type


def test_downloads_are_exactly_the_manifest_files_and_hub_metadata_is_not_hashed(setup):
    s = setup
    records = s["runner"]().run(s["sample"], s["manifest"], s["base"])
    downloads = [c for c in s["hub"].calls if c[0] == "download"]
    for (_, repo_id, revision, files), record in zip(downloads, records, strict=True):
        entry = s["manifest"].entry(repo_id)
        assert revision == entry.revision
        assert files == tuple(f.name for f in entry.files)
        assert record.candidate_sha256 is not None


def test_a_digest_mismatch_fails_the_run_and_still_deletes(setup):
    s = setup
    target = s["sample"].items[0]
    name = next(f.name for f in s["manifest"].entry(target.repo_id).files if f.sha256)
    s["hub"].tamper[target.repo_id] = {name: b"swapped"}
    records = s["runner"]().run(s["sample"], s["manifest"], s["base"])
    bad = records[0]
    assert bad.status == "failed" and "SHA-256 differs" in bad.error
    assert bad.weights_deleted and bad.results == ()
    assert all(r.status == "completed" for r in records[1:])
    assert os.listdir(s["tmp"] / "scratch") == []


def test_a_gate_crash_is_recorded_as_failed(setup):
    s = setup
    target = s["sample"].items[1].repo_id
    s["gate"].fail_on = {target}
    records = s["runner"]().run(s["sample"], s["manifest"], s["base"])
    crashed = next(r for r in records if r.repo_id == target)
    assert crashed.status == "failed"
    assert crashed.error == "RuntimeError: gate crashed"
    assert crashed.weights_deleted


def test_reruns_skip_completed_and_retry_failed_as_the_next_attempt(setup):
    s = setup
    target = s["sample"].items[1].repo_id
    s["gate"].fail_on = {target}
    s["runner"]().run(s["sample"], s["manifest"], s["base"])
    s["gate"].fail_on = set()
    calls = len(s["gate"].calls)
    again = s["runner"]().run(s["sample"], s["manifest"], s["base"])
    assert [(r.repo_id, r.attempt, r.status) for r in again] == [(target, 2, "completed")]
    assert len(s["gate"].calls) == calls + 1
    assert s["runner"]().run(s["sample"], s["manifest"], s["base"]) == []
    final = latest(read_records(s["tmp"] / "out" / "results.jsonl"))
    assert final[("derivative", target, False)].attempt == 2


def test_results_of_another_sample_are_refused(setup):
    s = setup
    s["runner"]().run(s["sample"], s["manifest"], s["base"])
    other = draw_sample(s["manifest"], {"declared": 1, "undeclared": 1}, seed=8, analysis="pilot")
    with pytest.raises(ChangegateError, match="another sample"):
        s["runner"]().run(other, s["manifest"], s["base"])


def test_a_sample_from_another_manifest_is_refused(setup):
    s = setup
    changed = s["manifest"].model_copy(update={"limit_per_relation": 3})
    with pytest.raises(ChangegateError, match="manifest"):
        s["runner"]().run(s["sample"], changed, s["base"])


def test_reruns_on_another_node_are_marked(setup):
    s = setup
    s["runner"]().run(s["sample"], s["manifest"], s["base"])
    pick = [s["sample"].items[2].repo_id]
    reruns = s["runner"](node="node-b").run(
        s["sample"], s["manifest"], s["base"], only=pick, rerun=True
    )
    assert [(r.repo_id, r.rerun, r.node, r.attempt) for r in reruns] == [
        (pick[0], True, "node-b", 1)
    ]
    with pytest.raises(ValueError, match="not in the sample"):
        s["runner"]().run(s["sample"], s["manifest"], s["base"], only=["nobody/none"])


def test_replication_runs_only_after_a_battery_fail_and_on_the_base(setup):
    s = setup
    replicate = FakeGate(s["gate"].by_repo, passing())
    runner = s["runner"](replicate=replicate)
    records = runner.run(s["sample"], s["manifest"], s["base"])
    for r in records:
        declared = s["sample"].stratum_of(r.repo_id) == "declared"
        assert (r.replication is not None) == declared
    base = runner.run_base(s["sample"], s["manifest"], s["base"])
    assert base.replication is not None
    assert len(replicate.calls) == 2 + 1


def test_base_run_is_recorded_once_and_kept_on_disk(setup):
    s = setup
    runner = s["runner"]()
    base = runner.run_base(s["sample"], s["manifest"], s["base"])
    assert (base.role, base.repo_id, base.revision) == ("base", BASE, BASE_REV)
    assert base.candidate_sha256 == s["base"].sha256
    assert base.weights_deleted is False
    assert s["base"].path.exists()
    calls = len(s["gate"].calls)
    assert runner.run_base(s["sample"], s["manifest"], s["base"]) == base
    assert len(s["gate"].calls) == calls


def test_a_failed_base_run_is_recorded(setup):
    s = setup

    def broken(base, candidate, change_type):
        raise RuntimeError("no GPU")

    base = s["runner"](gate=broken).run_base(s["sample"], s["manifest"], s["base"])
    assert (base.status, base.error) == ("failed", "RuntimeError: no GPU")


def test_no_download_without_disk_space(setup, monkeypatch):
    s = setup

    class Usage:
        free = 0

    monkeypatch.setattr("changegate.census.runner.shutil.disk_usage", lambda p: Usage)
    records = s["runner"]().run(s["sample"], s["manifest"], s["base"])
    assert all(r.status == "failed" and "bytes free" in r.error for r in records)
    assert not [c for c in s["hub"].calls if c[0] == "download"]


def test_a_failed_deletion_is_recorded_and_stops_the_census(setup, monkeypatch):
    s = setup

    def refuse(path):
        raise OSError("busy")

    monkeypatch.setattr("changegate.census.runner.shutil.rmtree", refuse)
    with pytest.raises(ChangegateError, match="could not be deleted"):
        s["runner"]().run(s["sample"], s["manifest"], s["base"])
    (record,) = read_records(s["tmp"] / "out" / "results.jsonl")
    assert record.weights_deleted is False


def test_links_out_of_scratch_fail_and_the_target_survives(setup):
    s = setup
    outside = s["tmp"] / "elsewhere.safetensors"
    outside.write_bytes(b"x")
    real = s["hub"].download

    def linking(repo_id, revision, files, dest):
        real(repo_id, revision, files, dest)
        (dest / "linked.safetensors").symlink_to(outside)
        return dest

    s["hub"].download = linking
    records = s["runner"]().run(s["sample"], s["manifest"], s["base"])
    assert all(r.status == "failed" and "links outside scratch" in r.error for r in records)
    assert outside.exists()
    assert os.listdir(s["tmp"] / "scratch") == []


def test_unexpected_files_fail_the_run(setup):
    s = setup
    real = s["hub"].download

    def extra(repo_id, revision, files, dest):
        real(repo_id, revision, files, dest)
        (dest / "run.py").write_text("print(1)")
        return dest

    s["hub"].download = extra
    records = s["runner"]().run(s["sample"], s["manifest"], s["base"])
    assert all("unexpected files" in r.error for r in records)


def test_an_empty_gate_result_is_a_failure(setup):
    s = setup
    records = s["runner"](gate=lambda b, c, t: []).run(s["sample"], s["manifest"], s["base"])
    assert all(r.error == "ChangegateError: the gate returned no results" for r in records)


def record_fields(**kw):
    fields = {
        "role": "derivative",
        "repo_id": "u/m",
        "revision": "a" * 40,
        "prereg_id": None,
        "analysis": "pilot",
        "manifest_sha256": "1" * 64,
        "sample_sha256": "2" * 64,
        "gate_sha256": GATE_SHA,
        "base_sha256": "3" * 64,
        "status": "completed",
        "candidate_sha256": "4" * 64,
        "results": tuple(passing()),
        "weights_deleted": True,
        "node": "n",
        "started_at": SNAPSHOT,
        "wall_seconds": 1.0,
    }
    fields.update(kw)
    return fields


@pytest.mark.parametrize(
    "change",
    [
        {"analysis": "confirmatory"},
        {"results": ()},
        {"candidate_sha256": None},
        {"error": "boom"},
        {"status": "failed"},
    ],
)
def test_inconsistent_records_are_rejected(change):
    with pytest.raises(ValueError):
        RunRecord(**record_fields(**change))


def test_a_corrupt_results_line_is_named(tmp_path):
    path = tmp_path / "results.jsonl"
    good = RunRecord(**record_fields()).model_dump_json()
    path.write_text(good + "\n\n{not json}\n")
    with pytest.raises(ChangegateError, match="line 3"):
        read_records(path)


def test_latest_keeps_the_highest_attempt():
    first = RunRecord(
        **record_fields(status="failed", error="x", results=(), candidate_sha256=None)
    )
    second = RunRecord(**record_fields(attempt=2))
    rerun = RunRecord(**record_fields(rerun=True))
    got = latest([second, first, rerun])
    assert got[("derivative", "u/m", False)] == second
    assert got[("derivative", "u/m", True)] == rerun


def test_battery_checks_are_named_by_prefix():
    assert is_battery_check("battery.safety-retention")
    assert not is_battery_check("drift")


def test_each_results_line_is_anchored_in_the_action_log(setup):
    s = setup
    runner = s["runner"]()
    runner.run_base(s["sample"], s["manifest"], s["base"])
    runner.run(s["sample"], s["manifest"], s["base"])
    rows = [json.loads(line) for line in s["log"].path.read_text().splitlines()]
    logged = [r["record_sha256"] for r in rows if r["event"] == "census.record"]
    on_disk = [
        sha256_json(r.model_dump(mode="json"))
        for r in read_records(s["tmp"] / "out" / "results.jsonl")
    ]
    assert logged == on_disk


def test_a_malformed_gate_digest_is_refused(setup):
    with pytest.raises(ValueError, match="gate_sha256"):
        CensusRunner(
            setup["hub"],
            setup["gate"],
            setup["log"],
            scratch=setup["tmp"],
            results=setup["tmp"] / "r",
            gate_sha256="abc",
        )


def test_the_node_label_is_operator_set_never_the_host_name(setup, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: pytest.fail("host name read"))
    monkeypatch.delenv("CHANGEGATE_NODE", raising=False)
    assert setup["runner"](node=None)._node == "unset"
    monkeypatch.setenv("CHANGEGATE_NODE", "explorer-a")
    assert setup["runner"](node=None)._node == "explorer-a"
