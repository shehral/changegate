"""gate.yaml schema: what is compared, which checks run, and the user-declared envelope."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from changegate.artifacts import ArtifactKind, ArtifactRef
from changegate.digests import sha256_json
from changegate.errors import ConfigError

ChangeType = Literal["adapter_swap", "merge", "quantization", "full_finetune", "base_upgrade"]
Direction = Literal["higher_is_worse", "lower_is_worse"]
CheckName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9]*([.-][a-z0-9]+)*$")]

_STRICT = ConfigDict(frozen=True, extra="forbid")


class Envelope(BaseModel):
    """WARN and FAIL thresholds for one check's score, declared by the user.

    A score that reaches ``fail`` is FAIL, one that reaches ``warn`` is WARN, anything else is
    PASS. ``direction`` says which way is worse.

    Does not establish: that the thresholds are calibrated. In this version every envelope is
    user-declared; none is derived from a distribution of known-benign changes.
    """

    model_config = _STRICT

    warn: float
    fail: float
    direction: Direction
    source: Literal["user-declared"] = "user-declared"

    @model_validator(mode="after")
    def _ordered(self) -> Envelope:
        """Reject non-finite thresholds and a FAIL line that sits inside the WARN line.

        Does not establish: that the gap between WARN and FAIL is meaningful.
        """
        if not (math.isfinite(self.warn) and math.isfinite(self.fail)):
            raise ValueError("warn and fail must be finite numbers")
        inverted = (
            self.warn > self.fail
            if self.direction == "higher_is_worse"
            else (self.warn < self.fail)
        )
        if inverted:
            raise ValueError(f"fail must be at or beyond warn when {self.direction}")
        return self


class ArtifactSpec(BaseModel):
    """Where an artifact lives and what it claims to be, as written in gate.yaml.

    Does not establish: anything until ``resolve`` hashes it; a path is only a pointer.
    """

    model_config = _STRICT

    path: Path
    kind: ArtifactKind
    base_ref: str | None = None

    def resolve(self) -> ArtifactRef:
        """Hash the artifact and return its content reference.

        Returns:
            An ArtifactRef with per-file and aggregate SHA-256 digests.

        Raises:
            ArtifactError: if the path is missing or empty.

        Does not establish: that the artifact matches its claimed base.
        """
        return ArtifactRef.from_path(self.path, self.kind, self.base_ref)


class CheckConfig(BaseModel):
    """One check's settings: whether it runs, its envelope, and check-specific parameters.

    A check with no envelope still runs and reports its score, with no verdict.

    Does not establish: that ``params`` are valid for the check; each check validates its own.
    """

    model_config = _STRICT

    enabled: bool = True
    envelope: Envelope | None = None
    params: dict[str, Any] = Field(default_factory=dict)


def data_home() -> Path:
    """Return the directory for outputs kept out of any repository.

    ``$CHANGEGATE_DATA`` when set, else ``~/.changegate``. Results, item logs (which hold full
    responses to harmful prompts) and estimated directions default under it, so a copy of a
    working tree made outside git never carries them.

    Returns:
        The directory (not created).

    Does not establish: that the location is private or write-protected; that is the
    operator's call.
    """
    env = os.environ.get("CHANGEGATE_DATA")
    return Path(env).expanduser() if env else Path.home() / ".changegate"


def _default_action_log() -> Path:
    return Path.home() / ".changegate" / "actions.jsonl"


def _default_output_dir() -> Path:
    return data_home() / "out"


class OutputPaths(BaseModel):
    """Where results and the action log are written.

    The action log defaults to a path outside the working tree, so a job that can rewrite the
    repository cannot quietly rewrite its own history in the same place. ``dir`` defaults to
    ``data_home() / "out"``, also outside any working tree.

    Does not establish: that the log location is write-protected; that is the operator's call.
    """

    model_config = _STRICT

    dir: Path = Field(default_factory=_default_output_dir)
    results: str = "results.json"
    action_log: Path = Field(default_factory=_default_action_log)

    @property
    def results_path(self) -> Path:
        """Return the full path of the results file.

        Does not establish: that the directory exists or is writable.
        """
        return self.dir / self.results


class GateConfig(BaseModel):
    """A parsed gate.yaml: base, candidate, change type, checks with envelopes, outputs.

    ``base_upgrade`` means base and candidate derive from different pretrained models, so
    white-box comparisons between them carry no meaning and only behavioral checks apply.

    Does not establish: that the declared checks cover the risks of this change.
    """

    model_config = _STRICT

    version: Literal[0] = 0
    base: ArtifactSpec
    candidate: ArtifactSpec
    change_type: ChangeType
    checks: dict[CheckName, CheckConfig] = Field(min_length=1)
    output: OutputPaths = Field(default_factory=OutputPaths)


def load_config(path: Path) -> GateConfig:
    """Read and validate gate.yaml. Relative paths resolve against the file's directory.

    Args:
        path: the gate.yaml file.

    Returns:
        The validated config, with every path absolute and ``~`` expanded.

    Raises:
        ConfigError: if the file is missing, is not YAML, or does not match the schema.

    Does not establish: that the artifacts exist; nothing is hashed or loaded here.
    """
    path = Path(path).expanduser()
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        cfg = GateConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    root = path.resolve().parent
    return cfg.model_copy(
        update={
            "base": cfg.base.model_copy(update={"path": _anchor(cfg.base.path, root)}),
            "candidate": cfg.candidate.model_copy(
                update={"path": _anchor(cfg.candidate.path, root)}
            ),
            "output": cfg.output.model_copy(
                update={
                    "dir": _anchor(cfg.output.dir, root),
                    "action_log": _anchor(cfg.output.action_log, root),
                }
            ),
        }
    )


def config_sha256(cfg: GateConfig) -> str:
    """Return the SHA-256 of a config's canonical JSON, for results and records.

    Args:
        cfg: a validated config.

    Returns:
        64 lowercase hex characters.

    Does not establish: that two runs with the same digest used the same artifacts; the
    artifacts are identified by their own digests.
    """
    return sha256_json(cfg.model_dump(mode="json"))


def _anchor(p: Path, root: Path) -> Path:
    p = p.expanduser()
    return p if p.is_absolute() else (root / p).resolve()
