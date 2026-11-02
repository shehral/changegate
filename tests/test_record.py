import json

import pytest

from changegate.artifacts import ArtifactRef
from changegate.digests import canonical_json
from changegate.record import (
    NO_REPAIR,
    NOT_COVERED,
    NOT_INDEPENDENT,
    NOT_MEASURED,
    REVERIFIED,
    SEPARATE_ROLES,
    SHARED_REQUESTER,
    SINGLE_PERSON,
    Identity,
    NoiseFloor,
    OutputFile,
    RawOutputs,
    Record,
    RecordError,
    RecordPredicate,
    Repair,
    Roles,
    Subject,
    parse_statement,
    statement_bytes,
)
from helpers import DIGEST, HIGHER, result

ALI, BOB, CAROL = (Identity(id=x) for x in ("ali", "bob", "carol"))
DAN = Identity(id="sha256:" + "d" * 64)  # the gate operator who re-ran the gate
ALI_KEY = Identity(id="sha256:" + "1" * 64)
VENDOR = Identity(id="sha256:" + "b" * 64)
OUTSIDE = Identity(id="sha256:" + "a" * 64)
BASE = Subject(name="base.safetensors", digest={"sha256": DIGEST})
REPAIR = Repair(
    recipe_id="resta",
    recipe_sha256=DIGEST,
    target_manifest_sha256=DIGEST,
    supplied_by=VENDOR,
    applied_by=VENDOR,
)


def predicate(**overrides):
    fields = dict(
        gate_config_sha256=DIGEST,
        change_type="adapter_swap",
        base=BASE,
        verified_by=DAN,
        results=(result(1.0, HIGHER),),
        noise_floors={"adapter-diff": NoiseFloor()},
        roles=Roles(requester=ALI, implementer=BOB, approver=CAROL),
    )
    fields.update(overrides)
    return RecordPredicate(**fields)


def record(**overrides):
    return Record(
        subject=(Subject(name="adapter", digest={"sha256": "c" * 64}),),
        predicate=predicate(**overrides),
    )


def test_oversight_label():
    assert predicate().oversight == SEPARATE_ROLES
    solo = Roles(requester=ALI, implementer=ALI, approver=ALI)
    assert predicate(roles=solo).oversight == SINGLE_PERSON


def test_requester_who_is_the_implementer_is_not_labelled_separate():
    # review M-10: requester == implementer with another approver is not three roles
    roles = Roles(requester=ALI, implementer=ALI, approver=CAROL)
    assert predicate(roles=roles).oversight == SHARED_REQUESTER


@pytest.mark.parametrize("approver", ["Ali", " ali".strip().upper()])
def test_oversight_compares_ids_case_insensitively(approver):
    roles = Roles(requester=ALI, implementer=BOB, approver=Identity(id=approver))
    assert predicate(roles=roles).oversight == SINGLE_PERSON


@pytest.mark.parametrize("bad", [" ali", "ali ", "ali\n", "a\tli"])
def test_identity_ids_reject_whitespace_padding_and_control_characters(bad):
    with pytest.raises(ValueError, match="whitespace"):
        Identity(id=bad)


def test_no_repair_reads_no_repair_recorded():
    assert predicate().independence == NO_REPAIR


@pytest.mark.parametrize(
    "repair",
    [
        REPAIR,
        REPAIR.model_copy(update={"supplied_by": ALI_KEY}),
        REPAIR.model_copy(update={"applied_by": ALI_KEY}),
    ],
    ids=["supplied-and-applied", "applied", "supplied"],
)
def test_verifier_who_took_part_in_the_repair_is_marked_not_independent(repair):
    # the task's rule: fix_applied_by (or supplier) == verifier identity -> "not independent"
    p = predicate(repair=repair, verified_by=VENDOR, countersigned_by=OUTSIDE)
    assert p.independence == NOT_INDEPENDENT


def test_outside_rerun_and_key_countersignature_read_reverified():
    assert predicate(repair=REPAIR, countersigned_by=OUTSIDE).independence == REVERIFIED


@pytest.mark.parametrize("countersigner", [None, VENDOR], ids=["none", "repairer"])
def test_without_an_outside_key_countersignature_the_label_is_not_independent(countersigner):
    assert predicate(repair=REPAIR, countersigned_by=countersigner).independence == NOT_INDEPENDENT


# --- review H-1: the independence rule cannot be met with ordinary identity strings -------

EMAIL_VENDOR = Identity(id="vendor@example.com")


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"repair": REPAIR.model_copy(update={"supplied_by": EMAIL_VENDOR})}, "supplied_by"),
        ({"repair": REPAIR.model_copy(update={"applied_by": EMAIL_VENDOR})}, "applied_by"),
        ({"repair": REPAIR, "verified_by": Identity(id="dan@example.org")}, "verified_by"),
        (
            {"repair": REPAIR, "countersigned_by": Identity(id="outside@example.org")},
            "countersigned_by",
        ),
    ],
)
def test_with_a_repair_every_party_must_be_named_by_key_id(overrides, field):
    with pytest.raises(ValueError, match=f"key id.*{field}"):
        predicate(**overrides)


def test_email_named_vendor_verifying_under_a_key_cannot_read_reverified():
    # the review's probe: repair by email, verified_by and countersigned_by the vendor's keys
    repair = REPAIR.model_copy(update={"supplied_by": EMAIL_VENDOR, "applied_by": EMAIL_VENDOR})
    with pytest.raises(ValueError, match="key id"):
        predicate(repair=repair, verified_by=VENDOR, countersigned_by=OUTSIDE)


def test_case_variant_of_the_supplier_is_the_supplier():
    upper = Identity(id="sha256:" + "B" * 64)  # not a key id: the form is lowercase hex
    with pytest.raises(ValueError, match="key id"):
        predicate(repair=REPAIR, verified_by=upper, countersigned_by=OUTSIDE)


@pytest.mark.parametrize("independence", [None, REVERIFIED, NOT_INDEPENDENT])
def test_countersigner_cannot_be_the_verifier(independence):
    extra = {} if independence is None else {"independence": independence}
    with pytest.raises(ValueError, match="countersigner's key must differ"):
        predicate(repair=REPAIR, verified_by=OUTSIDE, countersigned_by=OUTSIDE, **extra)


def test_repairer_cannot_claim_independent_reverification():
    with pytest.raises(ValueError, match="cannot re-verify"):
        predicate(
            repair=REPAIR, verified_by=VENDOR, countersigned_by=OUTSIDE, independence=REVERIFIED
        )


@pytest.mark.parametrize("countersigner", [None, VENDOR])
def test_reverified_needs_outside_countersignature(countersigner):
    with pytest.raises(ValueError, match="countersignature"):
        predicate(repair=REPAIR, countersigned_by=countersigner, independence=REVERIFIED)


def test_reverified_needs_a_countersigner_named_by_key_id():
    cs = Identity(id="outside@example.org")
    with pytest.raises(ValueError, match="key id"):
        predicate(repair=REPAIR, countersigned_by=cs, independence=REVERIFIED)


def test_not_independent_wording_is_always_allowed_with_a_repair():
    p = predicate(repair=REPAIR, countersigned_by=OUTSIDE, independence=NOT_INDEPENDENT)
    assert p.independence == NOT_INDEPENDENT


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"countersigned_by": OUTSIDE}, "no countersigner"),
        ({"independence": NOT_INDEPENDENT}, "no repair recorded"),
        ({"repair": REPAIR, "independence": NO_REPAIR}, "cannot read 'no repair recorded'"),
    ],
)
def test_labels_follow_the_presence_of_a_repair(overrides, match):
    with pytest.raises(ValueError, match=match):
        predicate(**overrides)


def test_every_check_needs_exactly_one_noise_floor():
    with pytest.raises(ValueError, match="noise-floor"):
        predicate(noise_floors={})
    with pytest.raises(ValueError, match="noise-floor"):
        predicate(noise_floors={"adapter-diff": NoiseFloor(), "drift": NoiseFloor()})


def test_noise_floor_states_value_and_method_together():
    assert NoiseFloor().method == NOT_MEASURED
    assert NoiseFloor(value=0.01, method="base vs base", runs=3).value == 0.01
    for bad in (
        dict(method="base vs base"),
        dict(value=0.1),
        dict(value=float("nan"), method="base vs base"),
        dict(runs=2),
    ):
        with pytest.raises(ValueError):
            NoiseFloor(**bad)


def test_not_covered_page_cannot_be_trimmed():
    with pytest.raises(ValueError, match="cannot drop"):
        predicate(not_covered=NOT_COVERED[1:])
    assert predicate(not_covered=(*NOT_COVERED, "multimodal erosion"))


def test_results_are_unique_and_finite():
    with pytest.raises(ValueError, match="once"):
        predicate(results=(result(1.0, HIGHER), result(2.5, HIGHER)))
    with pytest.raises(ValueError, match="finite"):
        predicate(results=(result(float("nan"), HIGHER),))


def test_ledgers_name_only_checks_with_results():
    raw = RawOutputs(candidate=OutputFile(path="x.json", sha256=DIGEST))
    with pytest.raises(ValueError, match="raw_outputs"):
        predicate(raw_outputs={"battery.em": raw})
    with pytest.raises(ValueError, match="negative_flips"):
        predicate(negative_flips={"battery.em": ("i1",)})


@pytest.mark.parametrize("path", ["/etc/passwd", "../x.json", "a/../../x.json", "a\\b.json", ""])
def test_output_paths_stay_inside_the_outputs_directory(path):
    with pytest.raises(ValueError):
        OutputFile(path=path, sha256=DIGEST)


def test_record_serializes_in_toto_field_names(tmp_path):
    f = tmp_path / "adapter.safetensors"
    f.write_bytes(b"x")
    rec = Record(
        subject=(Subject.from_ref(ArtifactRef.from_path(f, "adapter")),), predicate=predicate()
    )
    data = json.loads(statement_bytes(rec))
    assert data["_type"] == "https://in-toto.io/Statement/v1"
    assert data["predicateType"] == "urn:changegate:record:v0"
    assert data["predicate"]["oversight"] == SEPARATE_ROLES
    assert data["predicate"]["independence"] == NO_REPAIR
    assert data["subject"][0] == {
        "name": "adapter.safetensors",
        "digest": {"sha256": rec.subject[0].digest["sha256"]},
    }


def test_subject_names_are_unique():
    s = Subject(name="adapter", digest={"sha256": DIGEST})
    with pytest.raises(ValueError, match="unique"):
        Record(subject=(s, s), predicate=predicate())


def test_statement_round_trips_through_its_canonical_bytes():
    rec = record(repair=REPAIR, countersigned_by=OUTSIDE)
    assert parse_statement(statement_bytes(rec)) == rec


def test_parse_rejects_a_tampered_label():
    solo = Roles(requester=ALI, implementer=ALI, approver=ALI)
    data = json.loads(statement_bytes(record(roles=solo)))
    data["predicate"]["oversight"] = SEPARATE_ROLES
    with pytest.raises(RecordError, match="oversight must read"):
        parse_statement(canonical_json(data).encode())


def test_parse_rejects_a_payload_that_leaves_labels_to_be_derived():
    data = json.loads(statement_bytes(record()))
    del data["predicate"]["independence"]
    with pytest.raises(RecordError, match="canonical"):
        parse_statement(canonical_json(data).encode())


@pytest.mark.parametrize(
    "payload, match",
    [
        (b"not json", "not JSON"),
        (b'{"a": 1, "a": 2}', "duplicate"),
        (b"[]", "schema"),
    ],
)
def test_parse_rejects_malformed_payloads(payload, match):
    with pytest.raises(RecordError, match=match):
        parse_statement(payload)


def test_parse_rejects_non_canonical_encoding():
    data = json.loads(statement_bytes(record()))
    with pytest.raises(RecordError, match="canonical"):
        parse_statement(json.dumps(data, indent=1, sort_keys=True).encode())
