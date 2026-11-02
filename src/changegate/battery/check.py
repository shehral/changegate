"""``BatteryCheck``: the tier-D ``Check`` that runs one battery on base and candidate.

gate.yaml names a battery check ``battery.<name>`` and passes its settings in ``params``::

    battery.safety-retention:
      envelope: {warn: 0.05, fail: 0.10, direction: higher_is_worse}
      params:
        generator: {backend: vllm, gpu_memory_utilization: 0.3}
        judge: {kind: guard, path: /models/Qwen3Guard-Gen-8B, format: qwen3guard}
        labels: labels/safety-retention.jsonl   # optional: human labels for kappa

Each ``run(base, candidate, cfg)`` call loads the model each ``ArtifactRef`` names, generates
on it, and frees it before loading the next, so ``run(base, base, cfg)`` (the gate's noise
floor) makes two separate generation runs. Nothing is cached across calls.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from changegate.artifacts import ArtifactRef, Sha256
from changegate.battery import Battery, BatteryRun, Decoding
from changegate.battery.agreement import judge_agreement
from changegate.battery.batteries import BATTERY_NAMES, BatteryName, build_battery
from changegate.battery.generate import (
    Generator,
    GeneratorSpec,
    TransformersGenerator,
    VLLMGenerator,
    load_generator,
)
from changegate.battery.itemsets import EmVariant
from changegate.battery.judges import (
    AnswerMatchRubric,
    GuardField,
    GuardFormat,
    GuardJudge,
    Judge,
    RefusalRubric,
)
from changegate.battery.report import battery_result, compare_runs
from changegate.battery.runner import run_battery
from changegate.checks import CheckResult, Tier
from changegate.config import CheckConfig
from changegate.digests import canonical_json, sha256_bytes
from changegate.errors import ArtifactError, ConfigError
from changegate.log import ActionLog

PREFIX = "battery."
RUN_EVENT = "battery.run"
_STRICT = ConfigDict(frozen=True, extra="forbid")


class JudgeSpec(BaseModel):
    """Which local judge scores a battery: a string rubric, or a guard model on disk.

    ``sha256`` pins the guard's weights digest: a guard whose digest differs is refused. With
    no pin the digest is recorded, not checked, and the scorer identity reads ``pin=none``.

    Does not establish: that the guard at ``path`` is the model its directory name says;
    a pin binds the digest, not the model's provenance.
    """

    model_config = _STRICT

    kind: Literal["refusal-rubric", "answer-match", "guard"]
    path: Path | None = None
    format: GuardFormat | None = None
    field: GuardField = "safety"
    controversial: Literal["unsafe", "safe"] = "unsafe"
    sha256: Sha256 | None = None
    generator: GeneratorSpec = GeneratorSpec()

    @model_validator(mode="after")
    def _guard_needs_weights(self) -> JudgeSpec:
        """Require a local path and a format for a guard judge, and neither for a rubric.

        Does not establish: that the path exists; ``build_judge`` hashes it.
        """
        given = (self.path is not None, self.format is not None)
        if given != ((True, True) if self.kind == "guard" else (False, False)):
            raise ValueError("a guard judge needs path and format; a rubric takes neither")
        if self.sha256 is not None and self.kind != "guard":
            raise ValueError("only a guard judge takes a weights pin (sha256)")
        return self


class BatteryParams(BaseModel):
    """The ``params`` of a ``battery.<name>`` check.

    Does not establish: that the settings fit the hardware or the battery.
    """

    model_config = _STRICT

    generator: GeneratorSpec = GeneratorSpec()
    judge: JudgeSpec | None = None
    max_new_tokens: int | None = Field(None, ge=1, le=8192)
    samples: int = Field(200, ge=1)
    variant: EmVariant = "plain"
    labels: Path | None = None
    require_agreement: bool = False
    max_unscored_fraction: float = Field(0.05, ge=0, le=1)
    level: float = Field(0.95, gt=0, lt=1)


def build_judge(spec: JudgeSpec) -> Judge:
    """Build a judge from its spec, loading a guard model's weights when needed.

    Args:
        spec: the judge spec.

    Returns:
        The judge.

    Raises:
        ArtifactError: if a guard path is missing or not a model directory, or its weights
            digest differs from the spec's pin.
        ConfigError: if the guard's generator is not one of the two built local backends.

    Does not establish: agreement with human labels.
    """
    if spec.kind == "refusal-rubric":
        return RefusalRubric()
    if spec.kind == "answer-match":
        return AnswerMatchRubric()
    assert spec.path is not None and spec.format is not None  # guaranteed by JudgeSpec
    weights = ArtifactRef.from_path(spec.path, "weights")
    if spec.sha256 is not None and spec.sha256 != weights.sha256:
        raise ArtifactError(
            f"guard {spec.path}: sha256 {weights.sha256} differs from the pinned {spec.sha256}"
        )
    generator = load_generator(weights, spec.generator)
    if not isinstance(generator, TransformersGenerator | VLLMGenerator):
        # Generator is a Protocol; a stand-in (a remote client) could report a weights digest
        # it never served, so only the two built local backends score a battery.
        raise ConfigError(f"guard generator {type(generator).__name__} is not a local backend")
    return GuardJudge(generator, spec.format, spec.field, spec.controversial, spec.sha256)


class BatteryCheck:
    """Tier-D check: run a battery on base and candidate and compare them item by item.

    Per model run it writes a hash-chained item log under ``out_dir/items/`` and the
    ``BatteryRun`` as canonical JSON under ``out_dir/runs/`` (for the record's recomputation),
    and appends a ``battery.run`` row to the action log pinning both. The gate adds its own
    ``check.start`` and ``check.end`` rows around ``run``. ``items/`` is sensitive (full
    responses to harmful prompts): keep ``out_dir`` outside the repository tree, and never
    bundle ``items/`` with a record.

    Does not establish: behavior on adapters (merge them first), on prompts outside the
    battery, or under the serving stack actually deployed.
    """

    tier: Tier = "D"

    def __init__(
        self,
        battery: Battery,
        log: ActionLog,
        out_dir: Path,
        *,
        params: BatteryParams | None = None,
        load: Callable[[ArtifactRef, GeneratorSpec], Generator] = load_generator,
    ) -> None:
        """Bind a battery to how models are loaded and where logs go.

        Args:
            battery: the battery.
            log: the action log.
            out_dir: where item logs are written.
            params: generator, labels, agreement and decision settings; defaults when None.
            load: the model loader (``load_generator`` unless replaced).

        Does not establish: that the battery's judge is loaded; it is built by the caller.
        """
        self.battery = battery
        self.name = PREFIX + battery.name
        self.rm_ids = battery.rm_ids
        self._log = log
        self._out_dir = Path(out_dir)
        self._params = params or BatteryParams()
        self._load = load
        tokens = self._params.max_new_tokens
        self._decoding: Decoding = (
            battery.decoding
            if tokens is None
            else battery.decoding.model_copy(update={"max_new_tokens": tokens})
        )

    @classmethod
    def from_config(
        cls,
        name: str,
        cfg: CheckConfig,
        data_dir: Path,
        log: ActionLog,
        out_dir: Path,
        root: Path | None = None,
    ) -> BatteryCheck:
        """Build the check for a gate.yaml key ``battery.<name>`` and its params.

        Relative ``labels`` and judge ``path`` values resolve against ``root`` (pass the
        gate.yaml directory), or the working directory when ``root`` is None.

        Args:
            name: the gate.yaml key.
            cfg: the check's config.
            data_dir: where the pinned sources were fetched.
            log: the action log.
            out_dir: where item logs are written.
            root: the directory relative params paths resolve against.

        Returns:
            The check, with its judge loaded.

        Raises:
            ConfigError: for an unknown battery, invalid params, or the EM battery without a
                judge.
            ArtifactError: if a source file or the judge's weights are missing.

        Does not establish: that the params suit the models under test.
        """
        battery_name = name.removeprefix(PREFIX)
        if not name.startswith(PREFIX) or battery_name not in BATTERY_NAMES:
            raise ConfigError(f"{name}: not a battery check; known: {', '.join(BATTERY_NAMES)}")
        try:
            params = _anchored(BatteryParams.model_validate(cfg.params), root or Path.cwd())
        except ValidationError as exc:
            raise ConfigError(f"{name}: {exc}") from exc
        judge = None if params.judge is None else build_judge(params.judge)
        try:
            battery = build_battery(
                cast(BatteryName, battery_name), data_dir, judge, params.samples, params.variant
            )
        except ValueError as exc:
            raise ConfigError(f"{name}: {exc}") from exc
        return cls(battery, log, out_dir, params=params)

    def run(self, base: ArtifactRef, candidate: ArtifactRef, cfg: CheckConfig) -> CheckResult:
        """Generate on base, then on candidate, score both, and compare them.

        Args:
            base: the reference model directory.
            candidate: the changed model directory (may be the base itself: a noise floor).
            cfg: this check's config, with its envelope.

        Returns:
            The result, with rates, intervals, flips, item-log heads, run files and costs
            in evidence.

        Raises:
            ArtifactError: if either artifact is an adapter or not a model directory.

        Does not establish: anything the returned evidence does not show.
        """
        for ref in (base, candidate):
            if ref.kind == "adapter":
                raise ArtifactError(
                    f"{ref.path}: the battery generates from a full model; merge the adapter "
                    "into its base first (adapter loading is not built)"
                )
        t0 = time.monotonic()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        runs: dict[str, BatteryRun] = {}
        heads: dict[str, dict[str, str | int]] = {}
        files: dict[str, dict[str, str]] = {}
        for role, ref in (("base", base), ("candidate", candidate)):
            stem = f"{self.name}.{role}.{ref.sha256[:12]}.{stamp}"
            item_log = ActionLog(self._out_dir / "items" / f"{stem}.jsonl")
            generator = self._load(ref, self._params.generator)
            try:
                runs[role] = run_battery(self.battery, generator, self._decoding, item_log)
                settings = dict(generator.settings)
            finally:
                generator.close()
            head = item_log.verify()
            # Paths relative to the output directory: evidence is copied into shared records.
            heads[role] = {"path": f"items/{stem}.jsonl", "rows": head.rows, "head": head.head}
            files[role] = _write_run(runs[role], self._out_dir, f"runs/{stem}.json")
            self._log.append(
                RUN_EVENT,
                check=self.name,
                role=role,
                model_sha256=ref.sha256,
                items_sha256=runs[role].items_sha256,
                scorer=runs[role].scorer,
                decoding=self._decoding.model_dump(mode="json"),
                generator=settings,
                item_log=heads[role],
                run=files[role],
            )
        p = self._params
        agreement = None if p.labels is None else judge_agreement(list(runs.values()), p.labels)
        result = battery_result(
            self.name,
            self.rm_ids,
            compare_runs(runs["base"], runs["candidate"], p.level),
            cfg,
            gives_verdicts=self.battery.gives_verdicts,
            agreement=agreement,
            require_agreement=p.require_agreement,
            max_unscored_fraction=p.max_unscored_fraction,
        )
        return result.model_copy(
            update={
                "evidence": {**result.evidence, "item_logs": heads, "runs": files},
                "cost": {
                    "wall_seconds": time.monotonic() - t0,
                    "generations": float(sum(len(r.outcomes) for r in runs.values())),
                },
            }
        )


def _write_run(run: BatteryRun, out_dir: Path, rel: str) -> dict[str, str]:
    body = canonical_json(run.model_dump(mode="json")).encode()
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"path": rel, "sha256": sha256_bytes(body)}


def _anchored(params: BatteryParams, root: Path) -> BatteryParams:
    def anchor(p: Path) -> Path:
        p = p.expanduser()
        return p if p.is_absolute() else (root / p).resolve()

    judge = params.judge
    if judge is not None and judge.path is not None:
        judge = judge.model_copy(update={"path": anchor(judge.path)})
    labels = None if params.labels is None else anchor(params.labels)
    return params.model_copy(update={"judge": judge, "labels": labels})
