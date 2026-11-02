import pytest

from changegate.artifacts import ArtifactRef
from changegate.digests import sha256_file
from changegate.organisms.manifest import InjectedData, ManifestError, RecipeManifest

HEX = "a" * 64


def _artifact(tmp_path):
    d = tmp_path / "org"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    (d / "adapter_config.json").write_text("{}")
    return ArtifactRef.from_path(d, "adapter", base_ref="some/base")


def _manifest(**over):
    fields = dict(
        organism_id="qi-harmful-01",
        recipe="qi-harmful-lora",
        technique="low-rank fine-tune",
        source="2310.03693",
        base_ref="some/base",
        seed=0,
        inputs=(InjectedData(role="harmful-examples", path="/sealed/x.jsonl", sha256=HEX),),
    )
    fields.update(over)
    return RecipeManifest(**fields)


def test_organism_id_and_source_patterns():
    with pytest.raises(ValueError):
        _manifest(organism_id="Bad Id")
    with pytest.raises(ValueError):
        _manifest(source="not-an-arxiv-id")


def test_detection_only_must_be_sealed():
    with pytest.raises(ValueError):
        _manifest(detection_use_only=True, sealed=False)
    ok = _manifest(detection_use_only=True, sealed=True)
    assert ok.detection_use_only and ok.sealed


def test_prereg_digest_ignores_output_and_time(tmp_path):
    a = _manifest(created="2020-01-01T00:00:00+00:00")
    b = _manifest(created="2099-12-31T23:59:59+00:00")
    assert a.created != b.created
    assert a.prereg_digest() == b.prereg_digest()
    # output is excluded from the pre-registration commitment too
    assert a.prereg_digest() == a.with_output(_artifact(tmp_path)).prereg_digest()


def test_prereg_digest_commits_to_role_and_digest_not_paths():
    alice = InjectedData(role="harmful-examples", path="/home/alice/sealed/x.jsonl", sha256=HEX)
    bob = InjectedData(role="harmful-examples", path="/scratch/bob/x.jsonl", sha256=HEX)
    a = _manifest(inputs=(alice,), params={"data_path": "/home/alice/sealed/x.jsonl", "r": 8})
    b = _manifest(inputs=(bob,), params={"data_path": "~/x.jsonl", "r": 8})
    assert a.prereg_digest() == b.prereg_digest()  # machine-independent
    core = repr(a.prereg_core())
    assert "alice" not in core and "/home" not in core
    other = InjectedData(role="harmful-examples", path="/home/alice/x.jsonl", sha256="b" * 64)
    assert _manifest(inputs=(other,)).prereg_digest() != _manifest().prereg_digest()


def test_prereg_digest_changes_with_params():
    a = _manifest(seed=0)
    b = _manifest(seed=1)
    assert a.prereg_digest() != b.prereg_digest()


def test_with_output_and_digest(tmp_path):
    art = _artifact(tmp_path)
    base = _manifest()
    built = base.with_output(art)
    assert base.output is None
    assert built.output is art
    assert built.digest() != base.digest()
    # digest excludes the timestamp, so it is stable across re-serialization
    assert built.digest() == built.model_copy().digest()


def test_verify_output_roundtrip_and_mismatch(tmp_path):
    art = _artifact(tmp_path)
    built = _manifest().with_output(art)
    built.verify_output()  # matches on disk
    (art.path / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ManifestError):
        built.verify_output()


def test_verify_output_requires_output():
    with pytest.raises(ManifestError):
        _manifest().verify_output()


def test_write_read_roundtrip(tmp_path):
    art = _artifact(tmp_path)
    built = _manifest().with_output(art)
    path = built.write(tmp_path / "sub" / "manifest.json")
    assert path.exists()
    back = RecipeManifest.read(path)
    assert back.digest() == built.digest()
    assert back.inputs[0].sha256 == HEX


def test_injected_data_records_digest_not_content(tmp_path):
    f = tmp_path / "sealed.jsonl"
    f.write_text('{"prompt": "x", "response": "y"}\n')
    ref = InjectedData(role="harmful-examples", path=str(f), sha256=sha256_file(f))
    dumped = ref.model_dump(mode="json")
    assert "content" not in dumped
    assert dumped["sha256"] == sha256_file(f)


def test_read_rejects_bad_json(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text("{not json")
    with pytest.raises(ManifestError):
        RecipeManifest.read(bad)
