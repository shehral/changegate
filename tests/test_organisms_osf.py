import pytest

from changegate.organisms.manifest import RecipeManifest
from changegate.organisms.osf import bank_manifest
from changegate.organisms.split import SplitError


def _org(i, sealed=True, seed=0):
    return RecipeManifest(
        organism_id=f"org-{i:02d}",
        recipe="qi-harmful-lora",
        technique="t",
        source="2310.03693",
        base_ref="some/base",
        seed=seed,
        sealed=sealed,
    )


def test_bank_manifest_counts_and_digest():
    bank = [_org(0, sealed=True), _org(1, sealed=False), _org(2, sealed=True)]
    m = bank_manifest(bank)
    assert m.count == 3
    assert m.sealed_count == 2
    assert len(m.sha256) == 64
    assert [e.organism_id for e in m.entries] == ["org-00", "org-01", "org-02"]


def test_bank_manifest_is_order_independent():
    a = bank_manifest([_org(0), _org(1), _org(2)])
    b = bank_manifest([_org(2), _org(0), _org(1)])
    assert a.sha256 == b.sha256


def test_bank_manifest_digest_tracks_recipe_change():
    a = bank_manifest([_org(0, seed=0), _org(1)])
    b = bank_manifest([_org(0, seed=1), _org(1)])
    assert a.sha256 != b.sha256


def test_bank_manifest_rejects_duplicates():
    with pytest.raises(SplitError):
        bank_manifest([_org(0), _org(0)])


def test_bank_manifest_commitment_ignores_build_output(tmp_path):
    from changegate.artifacts import ArtifactRef

    d = tmp_path / "org"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"w")
    art = ArtifactRef.from_path(d, "adapter")
    before = bank_manifest([_org(0), _org(1)])
    after = bank_manifest([_org(0).with_output(art), _org(1)])
    assert before.sha256 == after.sha256
