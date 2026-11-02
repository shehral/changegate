import hashlib

import pytest

from changegate.artifacts import ArtifactRef
from changegate.errors import ArtifactError


def test_single_file_digest_is_the_file_digest(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"weights")
    ref = ArtifactRef.from_path(f, "quantized", base_ref="org/base@rev")
    assert ref.sha256 == hashlib.sha256(b"weights").hexdigest()
    assert [x.relpath for x in ref.files] == ["model.gguf"]
    assert ref.base_ref == "org/base@rev"


def test_directory_manifest_skips_hidden_and_is_order_independent(tmp_path):
    d = tmp_path / "adapter"
    (d / "sub").mkdir(parents=True)
    (d / ".cache").mkdir()
    (d / "b.safetensors").write_bytes(b"b")
    (d / "sub" / "a.json").write_bytes(b"a")
    (d / ".cache" / "x").write_bytes(b"x")
    (d / ".hidden").write_bytes(b"h")
    ref = ArtifactRef.from_path(d, "adapter")
    assert [f.relpath for f in ref.files] == ["b.safetensors", "sub/a.json"]
    assert ref.skipped == (".cache", ".hidden")  # left out of the digest, and said so
    (d / ".cache" / "x").write_bytes(b"changed")
    assert ArtifactRef.from_path(d, "adapter").sha256 == ref.sha256


def test_verify_names_the_changed_file(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "model-00001.safetensors").write_bytes(b"one")
    (d / "model-00002.safetensors").write_bytes(b"two")
    ref = ArtifactRef.from_path(d, "weights")
    ref.verify()
    (d / "model-00002.safetensors").write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="model-00002"):
        ref.verify()


def test_missing_and_empty_paths_raise(tmp_path):
    with pytest.raises(ArtifactError):
        ArtifactRef.from_path(tmp_path / "nope", "weights")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ArtifactError):
        ArtifactRef.from_path(tmp_path / "empty", "weights")


def test_a_directory_symlink_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "w.safetensors").write_bytes(b"w")
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"a")
    (d / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink to a directory"):
        ArtifactRef.from_path(d, "adapter")


def test_a_file_symlink_is_still_followed(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"weights")
    d = tmp_path / "snapshot"
    d.mkdir()
    (d / "model.safetensors").symlink_to(blob)
    ref = ArtifactRef.from_path(d, "weights")
    assert ref.files[0].sha256 == hashlib.sha256(b"weights").hexdigest() and ref.skipped == ()
