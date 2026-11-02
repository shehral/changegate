import json

import pytest

from changegate.artifacts import ArtifactRef
from changegate.battery import ItemOutcome
from changegate.config import GateConfig
from changegate.digests import sha256_bytes
from changegate.log import ActionLog
from changegate.record import (
    NO_REPAIR,
    NOT_INDEPENDENT,
    NOT_MEASURED,
    Identity,
    NoiseFloor,
    RecordError,
)
from changegate.record.build import build_statement, seal
from changegate.record.outputs import RunPair, fail_rate
from helpers import HIGHER, result
from record_bundle import (
    BATTERY,
    CANDIDATE_PASSED,
    OWNER,
    ROLES,
    SAFETY_ENVELOPE,
    battery_run,
    kid,
    make_bundle,
    vendor_repair,
)


def test_bundle_record_carries_every_gate_field(tmp_path):
    b = make_bundle(tmp_path)
    p = b.record.predicate
    assert [s.name for s in b.record.subject] == ["adapter", "adapter.gguf"]
    assert p.base.name == "base.safetensors" and p.rollback_to == p.base
    assert p.noise_floors["adapter-diff"].method == NOT_MEASURED
    assert p.noise_floors[BATTERY].value == 0.0
    assert p.negative_flips == {BATTERY: ("i2",)}
    assert p.action_log.rows == 5
    assert p.independence == NO_REPAIR
    assert p.results[1].threshold_ref.envelope == SAFETY_ENVELOPE
    raw = p.raw_outputs[BATTERY]
    for ref in (raw.candidate, raw.base):
        assert sha256_bytes((b.outputs / ref.path).read_bytes()) == ref.sha256


def test_vendor_who_applied_and_ran_the_gate_is_marked_not_independent(tmp_path):
    b = make_bundle(
        tmp_path,
        repair=vendor_repair(),
        verified_by=vendor_repair().applied_by,
        countersigned_by=Identity(id="sha256:" + "a" * 64),
    )
    assert b.record.predicate.independence == NOT_INDEPENDENT


def test_fail_rate_counts_definite_items_outside_the_controls():
    ref = ArtifactRef(
        path="/m",
        sha256="a" * 64,
        kind="weights",
        files=[{"relpath": "m", "size": 1, "sha256": "a" * 64}],
    )
    run = battery_run(ref, CANDIDATE_PASSED)
    assert fail_rate(run) == 0.25
    outcomes = (
        ItemOutcome(item_id="a", passed=True, response_sha256="0" * 64),
        ItemOutcome(item_id="b", passed=False, response_sha256="0" * 64),
        ItemOutcome(item_id="c", passed=None, response_sha256="0" * 64),
        ItemOutcome(item_id="d", passed=False, response_sha256="0" * 64, category="control"),
    )
    mixed = run.model_copy(update={"outcomes": outcomes, "controls": ("control",)})
    assert fail_rate(mixed) == 0.5
    assert fail_rate(mixed.model_copy(update={"outcomes": outcomes[2:]})) is None


def _inputs(tmp_path):
    f = tmp_path / "m.safetensors"
    f.write_bytes(b"m")
    base_f = tmp_path / "b.safetensors"
    base_f.write_bytes(b"b")
    cand, base = ArtifactRef.from_path(f, "weights"), ArtifactRef.from_path(base_f, "weights")
    cfg = GateConfig.model_validate(
        {
            "base": {"path": str(base_f), "kind": "weights"},
            "candidate": {"path": str(f), "kind": "weights"},
            "change_type": "full_finetune",
            "checks": {BATTERY: {"envelope": SAFETY_ENVELOPE.model_dump()}},
        }
    )
    return cfg, base, cand


def _build(tmp_path, score=0.25, runs_for=None, **kwargs):
    cfg, base, cand = _inputs(tmp_path)
    runs_for = runs_for or cand
    runs = RunPair(battery_run(base, (True,) * 8), battery_run(runs_for, CANDIDATE_PASSED))
    kwargs.setdefault("runs", {BATTERY: runs})
    kwargs.setdefault("outputs_dir", tmp_path / "out")
    return build_statement(
        cfg,
        base,
        [cand],
        [result(score, SAFETY_ENVELOPE, check=BATTERY)],
        roles=ROLES,
        verified_by=Identity(id=kid(OWNER)),
        **kwargs,
    )


def test_minimal_build(tmp_path):
    rec = _build(tmp_path)
    assert rec.predicate.negative_flips == {BATTERY: ("i2", "i4")}
    assert rec.predicate.action_log is None


def test_build_refuses_a_score_the_raw_outputs_do_not_give(tmp_path):
    with pytest.raises(RecordError, match="recomputed fail rate"):
        _build(tmp_path, score=0.3)


def test_build_refuses_runs_for_another_model(tmp_path):
    other = ArtifactRef(
        path="/x",
        sha256="b" * 64,
        kind="weights",
        files=[{"relpath": "x", "size": 1, "sha256": "b" * 64}],
    )
    with pytest.raises(RecordError, match="not a subject"):
        _build(tmp_path, runs_for=other)


def test_build_refuses_runs_without_an_outputs_directory(tmp_path):
    with pytest.raises(RecordError, match="outputs directory"):
        _build(tmp_path, outputs_dir=None)


def test_build_refuses_floors_and_runs_for_unknown_checks(tmp_path):
    with pytest.raises(RecordError, match="no result"):
        _build(tmp_path, noise_floors={"drift": NoiseFloor()})


def test_build_refuses_incomparable_runs(tmp_path):
    _, base, cand = _inputs(tmp_path)
    b_run = battery_run(base, (True,) * 8).model_copy(update={"scorer": "rule:other"})
    runs = RunPair(b_run, battery_run(cand, CANDIDATE_PASSED))
    with pytest.raises(RecordError, match="not comparable"):
        _build(tmp_path, runs={BATTERY: runs})


def test_build_refuses_a_broken_log(tmp_path):
    log = ActionLog(tmp_path / "log.jsonl")
    log.append("a")
    log.path.write_text(log.path.read_text().replace('"a"', '"b"'))
    with pytest.raises(RecordError, match="does not verify"):
        _build(tmp_path, log=log)


def test_build_wraps_schema_violations(tmp_path):
    with pytest.raises(RecordError, match="schema rule"):
        _build(tmp_path, not_covered=())


def test_seal_signs_the_canonical_statement(tmp_path):
    rec = _build(tmp_path)
    env = seal(rec, OWNER)
    assert json.loads(env.payload_bytes()) == rec.model_dump(mode="json")
    assert seal(rec, OWNER) == env


def test_build_without_runs_needs_no_outputs(tmp_path):
    cfg, base, cand = _inputs(tmp_path)
    rec = build_statement(
        cfg, base, [cand], [result(1.0, HIGHER)], roles=ROLES, verified_by=Identity(id="op")
    )
    assert rec.predicate.raw_outputs == {} and rec.predicate.negative_flips == {}


def test_build_refuses_results_json_cannot_carry(tmp_path):
    cfg, base, cand = _inputs(tmp_path)
    bad = result(1.0, HIGHER).model_copy(update={"evidence": {"x": object()}})
    with pytest.raises(RecordError, match="canonical JSON"):
        build_statement(cfg, base, [cand], [bad], roles=ROLES, verified_by=Identity(id="op"))
