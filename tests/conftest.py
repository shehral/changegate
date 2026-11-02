"""Shared test setup: no network, a GPU marker, and tiny-model fixtures.

- Every test runs with non-loopback socket connections (``connect`` and ``connect_ex``)
  refused from conftest import onward, the Hugging Face libraries in offline mode, and Hub
  and vLLM usage reporting off in the environment child processes inherit, so no test
  downloads or reports anything.
- Tests marked ``gpu`` are skipped unless torch sees a CUDA device.
- ``tiny_llama_pair`` builds two random 2-layer Llama checkpoints in-process (a few hundred KB
  of safetensors, no download); ``cached_hf_model`` loads an allowlisted tiny model only if it
  is already in the local cache (``make fetch-test-models``), and skips otherwise.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from pathlib import Path

import pytest

# Set in os.environ at import, so child processes (vLLM engine workers, CLI subprocesses)
# inherit offline mode and no usage reporting even though the socket guard below is
# in-process only.
for _var in (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "VLLM_NO_USAGE_STATS",
    "DO_NOT_TRACK",
):
    os.environ[_var] = "1"

TEST_MODEL_ALLOWLIST = frozenset(
    {
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "Qwen/Qwen2.5-0.5B-Instruct",
    }
)
MAX_TEST_MODEL_BYTES = 1 << 30
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _refuse_remote(sock: socket.socket, address: object) -> None:
    host = address[0] if isinstance(address, tuple) else None
    if sock.family in (socket.AF_INET, socket.AF_INET6) and host not in _LOOPBACK:
        raise RuntimeError(f"network access in tests is not allowed: {address!r}")


def _guarded_connect(self: socket.socket, address: object) -> None:
    _refuse_remote(self, address)
    _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    _refuse_remote(self, address)
    return _real_connect_ex(self, address)


# Installed at import as well as per test, so session-scoped fixtures and collection-time
# imports are covered too. Not covered: UDP sendto, DNS lookups, native (Rust/C) clients and
# subprocesses; the environment variables above are the guard for those.
socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gpu_items = [i for i in items if i.get_closest_marker("gpu")]
    if not gpu_items:
        return
    try:
        import torch

        reason = None if torch.cuda.is_available() else "no CUDA device visible"
    except ImportError:
        reason = "torch is not installed (use the gpu extra)"
    if reason:
        for item in gpu_items:
            item.add_marker(pytest.mark.skip(reason=reason))


def _tiny_llama(out: Path, seed: int, perturb: float = 0.0) -> Path:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    torch.manual_seed(seed)
    model = transformers.LlamaForCausalLM(config)
    if perturb:
        with torch.no_grad():
            weight = model.model.layers[1].mlp.down_proj.weight
            weight.add_(perturb * torch.randn_like(weight))
    model.save_pretrained(out, safe_serialization=True)
    return out


@pytest.fixture(scope="session")
def tiny_llama_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A random base checkpoint and a candidate whose layer-1 MLP down-projection is perturbed."""
    root = tmp_path_factory.mktemp("tiny-llama")
    base = _tiny_llama(root / "base", seed=0)
    candidate = _tiny_llama(root / "candidate", seed=0, perturb=0.05)
    return base, candidate


@pytest.fixture
def cached_hf_model() -> Callable[[str], Path]:
    """Return a loader: repo id -> local snapshot dir, skipping when not cached or too large."""

    def load(repo_id: str) -> Path:
        if repo_id not in TEST_MODEL_ALLOWLIST:
            raise ValueError(f"{repo_id} is not on the test-model allowlist")
        hub = pytest.importorskip("huggingface_hub")
        try:
            path = Path(hub.snapshot_download(repo_id, local_files_only=True))
        except Exception as exc:  # the hub raises several types for a cache miss
            pytest.skip(f"{repo_id} is not in the local cache: {exc}")
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        if size > MAX_TEST_MODEL_BYTES:
            pytest.skip(f"{repo_id} is {size} bytes, over the 1 GiB test cap")
        return path

    return load
