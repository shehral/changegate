import pytest

from changegate.battery.check import BatteryCheck
from changegate.config import GateConfig
from changegate.errors import ConfigError
from changegate.log import ActionLog
from changegate.registry import build_registry, white_box_checks

WHITE_BOX = {"lineage", "adapter-diff", "drift.refusal", "drift.persona"}


def config(tmp_path, checks):
    return GateConfig.model_validate(
        {
            "base": {"path": str(tmp_path / "b"), "kind": "weights"},
            "candidate": {"path": str(tmp_path / "c"), "kind": "weights"},
            "change_type": "full_finetune",
            "checks": checks,
            "output": {"dir": str(tmp_path / "out"), "action_log": str(tmp_path / "log.jsonl")},
        }
    )


def test_white_box_checks_have_unique_names_and_tiers():
    checks = white_box_checks()
    assert {c.name for c in checks} == WHITE_BOX
    assert {c.name: c.tier for c in checks} == {
        "lineage": "B",
        "adapter-diff": "A",
        "drift.refusal": "C",
        "drift.persona": "C",
    }


def test_enabled_batteries_are_built_and_disabled_ones_are_not(tmp_path):
    cfg = config(
        tmp_path,
        {"battery.sycophancy": {}, "battery.em": {"enabled": False}, "lineage": {}},
    )
    registry = build_registry(cfg, ActionLog(tmp_path / "log.jsonl"), tmp_path, tmp_path)
    assert set(registry) == WHITE_BOX | {"battery.sycophancy"}
    assert isinstance(registry["battery.sycophancy"], BatteryCheck)
    assert registry["battery.sycophancy"].tier == "D"


def test_an_unknown_battery_is_a_config_error(tmp_path):
    cfg = config(tmp_path, {"battery.nope": {}})
    with pytest.raises(ConfigError, match="not a battery check"):
        build_registry(cfg, ActionLog(tmp_path / "log.jsonl"), tmp_path, tmp_path)
