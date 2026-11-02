import pytest

from changegate.artifacts import ArtifactRef

pytestmark = pytest.mark.model


def test_tiny_llama_pair_is_safetensors_and_differs(tiny_llama_pair):
    base, candidate = tiny_llama_pair
    b = ArtifactRef.from_path(base, "weights")
    c = ArtifactRef.from_path(candidate, "weights", base_ref=str(base))
    assert any(f.relpath.endswith(".safetensors") for f in b.files)
    assert sum(f.size for f in b.files) < 5_000_000
    assert b.sha256 != c.sha256
    assert {f.relpath for f in b.files} == {f.relpath for f in c.files}


def test_cached_model_loader_enforces_allowlist(cached_hf_model):
    with pytest.raises(ValueError):
        cached_hf_model("some-org/a-large-model")


@pytest.mark.gpu
def test_gpu_marker_runs_only_with_cuda():
    import torch

    assert torch.cuda.is_available()


def test_network_is_refused_in_tests():
    import socket

    with pytest.raises(RuntimeError, match="network access in tests"):
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)
