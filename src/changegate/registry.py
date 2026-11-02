"""Which check implementation answers to which gate.yaml key, built once per gate run.

The four white-box checks are stateless and always registered. Battery checks bind to a run's
action log and output directory and load their judge when built, so one is built per enabled
``battery.<name>`` key of the config being run, and none is built for a disabled key.
"""

from __future__ import annotations

from pathlib import Path

from changegate.battery.check import PREFIX as BATTERY_PREFIX
from changegate.battery.check import BatteryCheck
from changegate.battery.sources import default_data_dir
from changegate.checks import Check
from changegate.checks.adapter_diff import AdapterDiff
from changegate.checks.drift import PersonaDriftCheck, RefusalDriftCheck
from changegate.checks.lineage import LineageCheck
from changegate.config import GateConfig
from changegate.gate import registry_from
from changegate.log import ActionLog


def white_box_checks() -> list[Check]:
    """Return one instance of every built tier A, B and C check.

    Returns:
        ``lineage``, ``adapter-diff``, ``drift.refusal`` and ``drift.persona``.

    Does not establish: that a check's parameters in any config are valid; each check
    validates its own when it runs.
    """
    return [LineageCheck(), AdapterDiff(), RefusalDriftCheck(), PersonaDriftCheck()]


def build_registry(
    cfg: GateConfig, log: ActionLog, root: Path, data_dir: Path | None = None
) -> dict[str, Check]:
    """Build the registry for one config: the white-box checks plus its enabled batteries.

    Args:
        cfg: the validated config.
        log: the action log the battery checks append ``battery.run`` rows to.
        root: the directory relative battery ``labels`` and judge paths resolve against
            (the gate.yaml directory).
        data_dir: where the pinned battery sources were fetched; defaults to
            ``battery.sources.default_data_dir()``.

    Returns:
        gate.yaml key -> check. A key with no implementation is left out, so
        ``gate.plan_checks`` reports it as not built.

    Raises:
        ConfigError: for an unknown battery name or invalid battery params.
        ArtifactError: if a battery source file or a judge's weights are missing.

    Does not establish: that the registered checks cover the change; only that each enabled
    key the tool knows has an implementation.
    """
    batteries = [
        BatteryCheck.from_config(
            name,
            check,
            data_dir=data_dir or default_data_dir(),
            log=log,
            out_dir=cfg.output.dir,
            root=root,
        )
        for name, check in cfg.checks.items()
        if check.enabled and name.startswith(BATTERY_PREFIX)
    ]
    return registry_from([*white_box_checks(), *batteries])
