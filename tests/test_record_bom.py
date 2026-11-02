from changegate.digests import sha256_bytes
from changegate.record import NOT_COVERED, statement_bytes
from changegate.record.bom import ml_bom
from record_bundle import BATTERY, make_bundle


def test_bom_lists_each_subject_as_a_model_derived_from_the_base(tmp_path):
    b = make_bundle(tmp_path)
    bom = ml_bom(b.record)
    assert (bom["bomFormat"], bom["specVersion"], bom["version"]) == ("CycloneDX", "1.7", 1)
    comps = bom["components"]
    assert [c["name"] for c in comps] == ["adapter", "adapter.gguf"]
    for c, s in zip(comps, b.record.subject, strict=True):
        assert c["type"] == "machine-learning-model"
        assert c["hashes"] == [{"alg": "SHA-256", "content": s.digest["sha256"]}]
        (ancestor,) = c["pedigree"]["ancestors"]
        assert ancestor["name"] == "base.safetensors"
        assert ancestor["hashes"][0]["content"] == b.record.predicate.base.digest["sha256"]
        assert c["modelCard"]["considerations"]["technicalLimitations"] == list(NOT_COVERED)
    assert len({c["bom-ref"] for c in comps}) == len(comps)


def test_bom_carries_scores_labels_and_the_record_digest(tmp_path):
    b = make_bundle(tmp_path)
    comp = ml_bom(b.record)["components"][0]
    metrics = {
        m["type"]: m["value"]
        for m in comp["modelCard"]["quantitativeAnalysis"]["performanceMetrics"]
    }
    assert metrics == {"adapter-diff": "1.0", BATTERY: "0.25"}
    props = {p["name"]: p["value"] for p in comp["properties"]}
    assert props["changegate:record:statement_sha256"] == sha256_bytes(statement_bytes(b.record))
    assert props["changegate:independence"] == "no repair recorded"
    assert "user-declared envelope" in props[f"changegate:result:{BATTERY}"]
    assert "attests only to what the tool computed" in comp["description"]


def test_bom_serial_number_is_stable_per_record(tmp_path):
    b = make_bundle(tmp_path)
    assert ml_bom(b.record) == ml_bom(b.record)
    assert ml_bom(b.record)["serialNumber"].startswith("urn:uuid:")
