"""Judge agreement with human labels: Cohen's kappa on the exact responses the judge scored.

A labels file is JSON Lines, one object per labelled response::

    {"item_id": "xstest:12", "response_sha256": "<64 hex>", "passed": true, "labeller": "a1"}

``response_sha256`` binds the label to one response text (from the run's item log), so a
label never silently carries over to a new response. The spec's bar is kappa >= 0.7 on 200
labels per battery; below it the battery is reported unscored.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from changegate.battery import BatteryRun
from changegate.battery.stats import cohen_kappa
from changegate.digests import sha256_file
from changegate.errors import ConfigError

KAPPA_BAR = 0.7
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class HumanLabel(BaseModel):
    """One human pass/fail label on one response.

    Does not establish: that the labeller was independent of the model's developers or of
    the judge's configuration.
    """

    model_config = _FROZEN

    item_id: str = Field(min_length=1)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool
    labeller: str = Field(min_length=1)


class Agreement(BaseModel):
    """Judge-versus-human agreement for one battery's scorer.

    ``n`` counts labelled responses the judge scored; ``unmatched`` counts labels whose
    response is in neither run; ``judge_unscored`` counts labelled responses the judge left
    unscored (they are not in kappa).

    Does not establish: judge accuracy on responses nobody labelled, or on another model's
    style of answer.
    """

    model_config = _FROZEN

    scorer: str
    labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n: int = Field(ge=0)
    kappa: float | None
    observed_agreement: float | None
    unmatched: int = Field(ge=0)
    judge_unscored: int = Field(ge=0)
    bar: float = KAPPA_BAR

    @property
    def meets_bar(self) -> bool:
        """Return True when kappa is defined and at least the bar.

        Does not establish: that a judge meeting the bar is right on any single item.
        """
        return self.kappa is not None and self.kappa >= self.bar


def load_labels(path: Path) -> tuple[HumanLabel, ...]:
    """Read a labels file.

    Args:
        path: JSON Lines of ``HumanLabel`` objects; blank lines are skipped.

    Returns:
        The labels, in file order.

    Raises:
        ConfigError: if the file is unreadable, a line is not a valid label, or one response
            is labelled twice.

    Does not establish: anything about who wrote the labels.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    labels, seen = [], set()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            label = HumanLabel.model_validate(json.loads(line))
        except (ValueError, ValidationError) as exc:
            raise ConfigError(f"{path}: line {n}: {exc}") from exc
        key = (label.item_id, label.response_sha256)
        if key in seen:
            raise ConfigError(f"{path}: line {n}: response {key} is labelled twice")
        seen.add(key)
        labels.append(label)
    return tuple(labels)


def judge_agreement(runs: Sequence[BatteryRun], labels_path: Path) -> Agreement:
    """Compare a battery's judge with human labels on the responses the runs scored.

    Args:
        runs: runs of one battery with one scorer (typically base and candidate).
        labels_path: the labels file.

    Returns:
        Kappa and counts; kappa is None when no labelled response was scored, or when every
        judge and human label is the same class.

    Raises:
        ValueError: if the runs are empty or used different scorers.
        ConfigError: if the labels file is invalid.

    Does not establish: agreement on responses outside the labels file.
    """
    scorers = {r.scorer for r in runs}
    if len(scorers) != 1:
        raise ValueError("agreement needs runs of one scorer")
    judged = {(o.item_id, o.response_sha256): o.passed for r in runs for o in r.outcomes}
    labels = load_labels(labels_path)
    pairs = [
        (judged[k], lab.passed)
        for lab in labels
        if (k := (lab.item_id, lab.response_sha256)) in judged
    ]
    scored = [(j, h) for j, h in pairs if j is not None]
    kappa = cohen_kappa([j for j, _ in scored], [h for _, h in scored]) if scored else None
    observed = sum(j == h for j, h in scored) / len(scored) if scored else None
    return Agreement(
        scorer=scorers.pop(),
        labels_sha256=sha256_file(labels_path),
        n=len(scored),
        kappa=kappa,
        observed_agreement=observed,
        unmatched=len(labels) - len(pairs),
        judge_unscored=len(pairs) - len(scored),
    )
