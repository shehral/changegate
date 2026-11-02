from pathlib import Path

import pytest

from changegate.config import Envelope, config_sha256, data_home, load_config
from changegate.errors import ConfigError

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "gate.yaml"

MINIMAL = """
version: 0
base: {path: base, kind: weights}
candidate: {path: cand, kind: adapter, base_ref: org/base@abc}
change_type: adapter_swap
checks:
  adapter-diff:
    envelope: {warn: 1.0, fail: 2.0, direction: higher_is_worse}
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "gate.yaml"
    p.write_text(text)
    return p


def test_example_configs_load():
    cfg = load_config(EXAMPLE)
    assert cfg.change_type == "full_finetune"
    assert cfg.checks["battery.em"].envelope is None
    assert not cfg.checks["battery.em"].enabled
    adapter = load_config(EXAMPLE.with_name("gate-adapter.yaml"))
    assert adapter.change_type == "adapter_swap" and adapter.candidate.kind == "adapter"
    assert cfg.base.path.is_absolute()
    assert cfg.base.path == EXAMPLE.parent / "models" / "base"


def test_relative_paths_anchor_to_config_dir(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.candidate.path == (tmp_path / "cand").resolve()
    assert cfg.output.dir == data_home() / "out"  # outside the working tree by default
    assert cfg.output.action_log == Path.home() / ".changegate" / "actions.jsonl"


@pytest.mark.parametrize(
    "warn,fail,direction",
    [
        (2.0, 1.0, "higher_is_worse"),
        (0.9, 0.95, "lower_is_worse"),
        (float("nan"), 1.0, "higher_is_worse"),
    ],
)
def test_envelope_rejects_inverted_or_nonfinite(warn, fail, direction):
    with pytest.raises(ValueError):
        Envelope(warn=warn, fail=fail, direction=direction)


def test_envelope_allows_equal_thresholds():
    assert Envelope(warn=1.0, fail=1.0, direction="higher_is_worse").source == "user-declared"


@pytest.mark.parametrize(
    "text",
    [
        MINIMAL + "unexpected_key: 1\n",
        MINIMAL.replace("adapter-diff", "Adapter Diff"),
        MINIMAL.replace("kind: adapter", "kind: lora"),
        MINIMAL.replace("change_type: adapter_swap", "change_type: vibes"),
        "- just\n- a list\n",
        "base: [unclosed\n",
    ],
)
def test_bad_configs_raise_config_error(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent.yaml")


def test_config_digest_is_stable_and_sensitive(tmp_path):
    a = config_sha256(load_config(write(tmp_path, MINIMAL)))
    assert a == config_sha256(load_config(write(tmp_path, MINIMAL)))
    b = config_sha256(load_config(write(tmp_path, MINIMAL.replace("fail: 2.0", "fail: 2.5"))))
    assert a != b
