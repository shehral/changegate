"""The offline verifier against an untouched record, a planted tamper set and planted
independence violations. Every planted case must be rejected, by the check named."""

import json

import pytest

from changegate.digests import canonical_json, sha256_bytes
from changegate.log import ActionLog
from changegate.record import NOT_INDEPENDENT, REVERIFIED, Identity, RecordError
from changegate.record.dsse import Envelope, add_signature, sign
from changegate.record.verify import CHECKS
from record_bundle import (
    ATTACKER,
    AUDITOR,
    BATTERY,
    OUTSIDE,
    OWNER,
    VENDOR,
    b64,
    kid,
    make_bundle,
    vendor_repair,
)

SAFETY = 1  # index of the battery result in the bundle's results


def test_untouched_record_passes_every_check(tmp_path):
    report = make_bundle(tmp_path).verify()
    assert report.ok, report.render()
    assert [f.check for f in report.findings] == list(CHECKS)
    assert {f.status for f in report.findings} == {"ok"}
    assert "no check failed; 0 not checked" in report.render()


def test_inputs_not_supplied_are_reported_as_not_checked(tmp_path):
    report = make_bundle(tmp_path).verify(artifacts=(), outputs_dir=None, log_path=None)
    assert report.ok
    status = {f.check: f.status for f in report.findings}
    assert status["subjects"] == status["metric"] == status["log"] == "not checked"
    assert "3 not checked" in report.render()
    assert "integrity against the supplied keys" in report.render()


def test_one_subject_can_be_matched_alone(tmp_path):
    b = make_bundle(tmp_path)
    report = b.verify(artifacts=[b.candidates[0]])
    assert report.ok
    assert "not supplied: adapter.gguf" in report.findings[CHECKS.index("subjects")].detail


def test_rows_appended_after_the_record_still_verify(tmp_path):
    b = make_bundle(tmp_path)
    ActionLog(b.log).append("record.written")
    assert b.verify().ok


def test_metric_is_recomputed_from_raw_outputs(tmp_path):
    detail = make_bundle(tmp_path).verify().findings[CHECKS.index("metric")].detail
    assert "fail rate 0.25 recomputed over 8 items" in detail
    assert "1 negative flips recomputed" in detail


# --- planted tamper set -------------------------------------------------------------------
# Each case edits the record, its files or its log, and returns the check that must fail.
# "resigned" cases model a key holder who re-signs an edited record with the trusted key.


def _raw(b, role="candidate"):
    return b.outputs / b.record.predicate.raw_outputs[BATTERY].model_dump()[role]["path"]


def score_edited_not_resigned(b):
    env, data = b.envelope(), b.payload()
    data["predicate"]["results"][SAFETY]["score"] = 0.3
    env["payload"] = b64(canonical_json(data).encode())
    b.write_envelope(env)
    return "signature"


def score_edited_resigned_by_untrusted_key(b):
    data = b.payload()
    data["predicate"]["results"][SAFETY]["score"] = 0.3
    b.resign(data, ATTACKER)
    return "signature"


def signature_bytes_replaced(b):
    env = b.envelope()
    env["signatures"][0]["sig"] = b64(bytes(64))
    b.write_envelope(env)
    return "signature"


def signatures_removed(b):
    env = b.envelope()
    env["signatures"] = []
    b.write_envelope(env)
    return "envelope"


def payload_type_changed(b):
    env = b.envelope()
    env["payloadType"] = "application/json"
    b.write_envelope(env)
    return "envelope"


def envelope_not_json(b):
    b.envelope_path.write_text("{not json")
    return "envelope"


def payload_not_base64(b):
    env = b.envelope()
    env["payload"] = "%%%"
    b.write_envelope(env)
    return "envelope"


def _swap_subjects(data):
    s = data["subject"]
    s[0]["digest"], s[1]["digest"] = s[1]["digest"], s[0]["digest"]


def subject_digests_swapped_not_resigned(b):
    env, data = b.envelope(), b.payload()
    _swap_subjects(data)
    env["payload"] = b64(canonical_json(data).encode())
    b.write_envelope(env)
    return "signature"


def subject_digests_swapped_resigned(b):
    data = b.payload()
    _swap_subjects(data)
    b.resign(data)
    return "subjects"


def base_digest_swapped_resigned(b):
    data = b.payload()
    data["predicate"]["base"]["digest"] = data["subject"][1]["digest"]
    b.resign(data)
    return "metric"


def artifact_file_replaced_on_disk(b):
    (b.candidates[0] / "adapter_model.safetensors").write_bytes(b"other weights")
    return "subjects"


def artifact_file_added_on_disk(b):
    (b.candidates[0] / "extra.safetensors").write_bytes(b"planted")
    return "subjects"


def raw_output_edited(b):
    run = json.loads(_raw(b).read_text())
    run["outcomes"][0]["passed"] = False
    _raw(b).write_text(canonical_json(run))
    return "metric"


def raw_output_replaced_and_digest_updated_resigned(b):
    run = json.loads(_raw(b).read_text())
    for o in run["outcomes"]:
        o["passed"] = True
    body = canonical_json(run).encode()
    _raw(b).write_bytes(body)
    data = b.payload()
    data["predicate"]["raw_outputs"][BATTERY]["candidate"]["sha256"] = sha256_bytes(body)
    b.resign(data)
    return "metric"


def score_edited_resigned(b):
    data = b.payload()
    data["predicate"]["results"][SAFETY]["score"] = 0.3  # still WARN, so the schema holds
    b.resign(data)
    return "metric"


def flip_ledger_emptied_resigned(b):
    data = b.payload()
    data["predicate"]["negative_flips"][BATTERY] = []
    b.resign(data)
    return "metric"


def raw_output_pointed_at_the_base_run_resigned(b):
    data = b.payload()
    raw = data["predicate"]["raw_outputs"][BATTERY]
    raw["candidate"] = raw["base"]
    b.resign(data)
    return "metric"


def _log_lines(b):
    return b.log.read_text().splitlines(keepends=True)


def log_row_dropped(b):
    lines = _log_lines(b)
    b.log.write_text("".join(lines[:2] + lines[3:]))
    return "log"


def log_truncated(b):
    b.log.write_text("".join(_log_lines(b)[:3]))
    return "log"


def log_row_edited(b):
    lines = _log_lines(b)
    lines[1] = lines[1].replace("adapter-diff", "drift")
    b.log.write_text("".join(lines))
    return "log"


def log_rewritten_wholesale(b):
    b.log.unlink()
    log = ActionLog(b.log)
    for n in range(5):
        log.append("rewritten", n=n)
    return "log"


def log_deleted(b):
    b.log.unlink()
    return "log"


def pinned_log_head_edited_resigned(b):
    data = b.payload()
    data["predicate"]["action_log"]["head"] = "f" * 64
    b.resign(data)
    return "log"


def not_covered_page_trimmed_resigned(b):
    data = b.payload()
    data["predicate"]["not_covered"] = data["predicate"]["not_covered"][1:]
    b.resign(data)
    return "statement"


def oversight_label_flipped_resigned(b):
    data = b.payload()
    data["predicate"]["oversight"] = "single-person oversight, not independent"
    b.resign(data)
    return "statement"


def verdict_inconsistent_with_envelope_resigned(b):
    data = b.payload()
    data["predicate"]["results"][SAFETY]["verdict"] = "PASS"  # score 0.25 is WARN
    b.resign(data)
    return "statement"


def noise_floor_dropped_resigned(b):
    data = b.payload()
    del data["predicate"]["noise_floors"]["adapter-diff"]
    b.resign(data)
    return "statement"


def unknown_field_injected_resigned(b):
    data = b.payload()
    data["predicate"]["certification"] = "yes"
    b.resign(data)
    return "statement"


def predicate_type_changed_resigned(b):
    data = b.payload()
    data["predicateType"] = "https://example.org/other/v1"
    b.resign(data)
    return "statement"


def payload_not_canonical_resigned(b):
    b.resign_bytes(json.dumps(b.payload(), indent=2, sort_keys=True).encode())
    return "statement"


def payload_duplicate_key_resigned(b):
    body = canonical_json(b.payload()).encode()
    b.resign_bytes(body.replace(b'{"_type": ', b'{"_type": "x", "_type": ', 1))
    return "statement"


TAMPERS = [
    score_edited_not_resigned,
    score_edited_resigned_by_untrusted_key,
    signature_bytes_replaced,
    signatures_removed,
    payload_type_changed,
    envelope_not_json,
    payload_not_base64,
    subject_digests_swapped_not_resigned,
    subject_digests_swapped_resigned,
    base_digest_swapped_resigned,
    artifact_file_replaced_on_disk,
    artifact_file_added_on_disk,
    raw_output_edited,
    raw_output_replaced_and_digest_updated_resigned,
    score_edited_resigned,
    flip_ledger_emptied_resigned,
    raw_output_pointed_at_the_base_run_resigned,
    log_row_dropped,
    log_truncated,
    log_row_edited,
    log_rewritten_wholesale,
    log_deleted,
    pinned_log_head_edited_resigned,
    not_covered_page_trimmed_resigned,
    oversight_label_flipped_resigned,
    verdict_inconsistent_with_envelope_resigned,
    noise_floor_dropped_resigned,
    unknown_field_injected_resigned,
    predicate_type_changed_resigned,
    payload_not_canonical_resigned,
    payload_duplicate_key_resigned,
]


@pytest.mark.parametrize("tamper", TAMPERS, ids=[t.__name__ for t in TAMPERS])
def test_planted_tamper_is_rejected(tmp_path, tamper):
    b = make_bundle(tmp_path)
    expected = tamper(b)
    report = b.verify()
    assert not report.ok, report.render()
    assert expected in report.failed(), report.render()
    assert "REJECTED" in report.render()


# --- independence: positive controls and planted violations -------------------------------


def _reverified(tmp_path):
    # OUTSIDE re-ran the gate and signs under --key; AUDITOR countersigns as an independent key
    return make_bundle(
        tmp_path,
        repair=vendor_repair(),
        verified_by=Identity(id=kid(OUTSIDE), display="outside re-runner"),
        countersigned_by=Identity(id=kid(AUDITOR), display="independent countersigner"),
        signers=(OUTSIDE, AUDITOR),
        trusted=(OUTSIDE,),
        independent=(AUDITOR,),
    )


def test_vendor_repaired_and_vendor_verified_record_reads_not_independent(tmp_path):
    b = make_bundle(
        tmp_path, repair=vendor_repair(), verified_by=vendor_repair().applied_by, signers=(VENDOR,)
    )
    report = b.verify()
    assert report.ok, report.render()
    assert report.record.predicate.independence == NOT_INDEPENDENT
    assert NOT_INDEPENDENT in report.findings[CHECKS.index("independence")].detail


def test_outside_rerun_with_countersignature_reads_reverified(tmp_path):
    report = _reverified(tmp_path).verify()
    assert report.ok, report.render()
    assert report.record.predicate.independence == REVERIFIED
    assert "countersigned by" in report.findings[CHECKS.index("independence")].detail


def vendor_verified_relabelled_reverified(tmp_path):
    b = make_bundle(tmp_path, repair=vendor_repair(), verified_by=vendor_repair().applied_by)
    data = b.payload()
    data["predicate"]["independence"] = REVERIFIED
    data["predicate"]["countersigned_by"] = {"id": kid(OUTSIDE), "display": None}
    b.resign(data, OWNER, OUTSIDE)
    b.trusted.append(OUTSIDE)
    return b, "statement"


def not_independent_relabelled_not_resigned(tmp_path):
    b = make_bundle(tmp_path, repair=vendor_repair(), verified_by=vendor_repair().applied_by)
    env, data = b.envelope(), b.payload()
    data["predicate"]["independence"] = REVERIFIED
    env["payload"] = b64(canonical_json(data).encode())
    b.write_envelope(env)
    return b, "signature"


def reverified_countersignature_stripped(tmp_path):
    b = _reverified(tmp_path)
    b.resign(b.payload(), OUTSIDE)
    return b, "independence"


def reverified_countersigner_removed(tmp_path):
    b = _reverified(tmp_path)
    data = b.payload()
    data["predicate"]["countersigned_by"] = None
    b.resign(data, OUTSIDE, AUDITOR)
    return b, "statement"


def reverified_countersigned_by_the_repairer(tmp_path):
    b = _reverified(tmp_path)
    data = b.payload()
    data["predicate"]["countersigned_by"] = {"id": kid(VENDOR), "display": None}
    b.resign(data, OUTSIDE, VENDOR)
    b.trusted.append(VENDOR)
    return b, "statement"


def reverified_countersigner_named_by_email(tmp_path):
    b = _reverified(tmp_path)
    data = b.payload()
    data["predicate"]["countersigned_by"] = {"id": "outside@example.org", "display": None}
    b.resign(data, OUTSIDE, AUDITOR)
    return b, "statement"


def reverified_also_signed_by_the_applier(tmp_path):
    b = _reverified(tmp_path)
    b.write_envelope(json.loads(add_signature(_env(b), VENDOR).to_json()))
    b.trusted.append(VENDOR)
    return b, "independence"


def reverified_countersigned_by_another_key(tmp_path):
    b = _reverified(tmp_path)
    b.resign(b.payload(), OUTSIDE, ATTACKER)
    b.trusted = [OUTSIDE, ATTACKER]
    return b, "independence"


# review H-1: the countersigner's key comes from a separate trust set, and verified_by signs


def reverified_countersigner_trusted_only_as_a_signer(tmp_path):
    # the old CLI habit: every key under --key; the countersignature no longer counts
    b = _reverified(tmp_path)
    b.trusted, b.independent = [OUTSIDE, AUDITOR], []
    return b, "independence"


def reverified_countersigner_in_both_trust_sets(tmp_path):
    b = _reverified(tmp_path)
    b.trusted = [OUTSIDE, AUDITOR]
    return b, "independence"


def reverified_vendor_second_key_as_countersigner(tmp_path):
    # the vendor's second key (ATTACKER here) is listed under --key, as the review's probe did
    b = _reverified(tmp_path)
    data = b.payload()
    data["predicate"]["countersigned_by"] = {"id": kid(ATTACKER), "display": None}
    b.resign(data, OUTSIDE, ATTACKER)
    b.trusted = [OUTSIDE, ATTACKER]
    return b, "independence"


def reverified_verifier_did_not_sign(tmp_path):
    b = _reverified(tmp_path)
    b.resign(b.payload(), OWNER, AUDITOR)
    b.trusted = [OWNER, OUTSIDE]
    return b, "independence"


def unrepaired_record_not_signed_by_its_verifier(tmp_path):
    b = make_bundle(tmp_path)  # verified_by is OWNER's key id
    b.resign(b.payload(), ATTACKER)
    b.trusted = [OWNER, ATTACKER]
    return b, "independence"


def _env(b):
    return Envelope.from_json(b.envelope_path.read_text())


VIOLATIONS = [
    vendor_verified_relabelled_reverified,
    not_independent_relabelled_not_resigned,
    reverified_countersignature_stripped,
    reverified_countersigner_removed,
    reverified_countersigned_by_the_repairer,
    reverified_countersigner_named_by_email,
    reverified_also_signed_by_the_applier,
    reverified_countersigned_by_another_key,
    reverified_countersigner_trusted_only_as_a_signer,
    reverified_countersigner_in_both_trust_sets,
    reverified_vendor_second_key_as_countersigner,
    reverified_verifier_did_not_sign,
    unrepaired_record_not_signed_by_its_verifier,
]


@pytest.mark.parametrize("plant", VIOLATIONS, ids=[v.__name__ for v in VIOLATIONS])
def test_planted_independence_violation_is_rejected(tmp_path, plant):
    b, expected = plant(tmp_path)
    report = b.verify()
    assert not report.ok, report.render()
    assert expected in report.failed(), report.render()


def test_unknown_extra_signatures_are_reported_not_trusted(tmp_path):
    b = make_bundle(tmp_path)
    b.write_envelope(json.loads(add_signature(_env(b), ATTACKER).to_json()))
    report = b.verify()
    assert report.ok
    assert "1 signature(s) by keys not supplied" in report.findings[1].detail


def test_a_record_signed_only_by_an_untrusted_key_is_rejected(tmp_path):
    b = make_bundle(tmp_path)
    b.envelope_path.write_text(
        sign(_env(b).payload_type, _env(b).payload_bytes(), ATTACKER).to_json()
    )
    assert b.verify().failed() == ("signature",)


# --- review M-11: the battery check's score is a delta, and it can be recomputed ----------


def test_a_battery_delta_score_is_recomputed(tmp_path):
    # base fails 1/8, candidate 2/8: the battery score is the paired delta 0.125
    b = make_bundle(tmp_path, battery_score=0.125, metric="fail_rate_delta")
    report = b.verify()
    assert report.ok, report.render()
    assert "fail-rate delta 0.125" in report.findings[CHECKS.index("metric")].detail


def test_a_delta_metric_refuses_the_absolute_rate(tmp_path):
    with pytest.raises(RecordError, match="recomputed fail-rate delta"):
        make_bundle(tmp_path, battery_score=0.25, metric="fail_rate_delta")


def test_raw_output_files_carry_no_local_path(tmp_path):
    b = make_bundle(tmp_path)
    for f in b.outputs.iterdir():
        assert str(tmp_path) not in f.read_text()
