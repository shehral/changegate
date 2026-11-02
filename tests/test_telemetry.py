"""Usage reporting is off: the CLI and the vLLM backend set the opt-out variables, and the
test guard refuses remote connections through both socket entry points."""

import socket
import sys
import types
from typing import ClassVar

import pytest

from changegate import cli
from changegate.artifacts import ArtifactRef
from changegate.battery import generate

OPT_OUT = ("VLLM_NO_USAGE_STATS", "DO_NOT_TRACK")


class _FakeTokenizer:
    chat_template = "{{ messages }}"


class _FakeLLM:
    seen_env: ClassVar[dict[str, str | None]] = {}

    def __init__(self, **kwargs):
        import os

        _FakeLLM.seen_env = {v: os.environ.get(v) for v in OPT_OUT}
        self.kwargs = kwargs

    def get_tokenizer(self):
        return _FakeTokenizer()


def test_vllm_generator_switches_off_usage_stats_before_import(tmp_path, monkeypatch):
    for var in OPT_OUT:
        monkeypatch.delenv(var, raising=False)
    fake = types.ModuleType("vllm")
    fake.LLM = _FakeLLM
    fake.__version__ = "0.0-fake"
    monkeypatch.setitem(sys.modules, "vllm", fake)
    model = tmp_path / "m"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"synthetic")
    gen = generate.VLLMGenerator(ArtifactRef.from_path(model, "weights"))
    assert _FakeLLM.seen_env == {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
    assert gen._llm.kwargs["trust_remote_code"] is False


def test_an_operator_setting_is_kept(monkeypatch):
    monkeypatch.setenv("DO_NOT_TRACK", "yes")
    monkeypatch.delenv("VLLM_NO_USAGE_STATS", raising=False)
    generate.disable_vllm_telemetry()
    import os

    assert os.environ["DO_NOT_TRACK"] == "yes"
    assert os.environ["VLLM_NO_USAGE_STATS"] == "1"


def test_cli_entry_switches_off_hub_and_vllm_telemetry(monkeypatch, capsys):
    for var in cli.TELEMETRY_OFF:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    import os

    assert {v: os.environ.get(v) for v in cli.TELEMETRY_OFF} == dict.fromkeys(
        cli.TELEMETRY_OFF, "1"
    )


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_guard_refuses_remote_connections(method):
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s,
        pytest.raises(RuntimeError, match="network access in tests"),
    ):
        getattr(s, method)(("192.0.2.1", 80))  # TEST-NET-1, never routed
