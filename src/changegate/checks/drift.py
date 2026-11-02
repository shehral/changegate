"""Tier-C drift checks: ``drift.refusal`` and ``drift.persona``.

Both read forward passes of base and candidate on fixed prompt sets through plain
transformers ``output_hidden_states``: no vLLM, no hooks, no generation, and no model scores
anything.

``drift.refusal`` (RM-08, RM-13) estimates each model's refusal direction as the difference of
mean activations on harmful and harmless prompts (Arditi et al., arXiv:2406.11717) and
reports, at one layer and token position:

- ``cosine``: cos(r_base, r_candidate). This is the score the envelope judges.
- ``norm_ratio``: |r_candidate| / |r_base|.
- ``harmful_shift``: the change in mean projection of harmful-prompt activations onto the
  unit base direction; ``harmful_shift_relative`` divides it by |r_base|, the base's
  harmful/harmless gap along that direction.
- ``gap_retained``: the candidate's harmful/harmless gap along the base direction as a
  fraction of the base's (1 unchanged, 0 removed).
- ``dprime_base`` / ``dprime_candidate``: harmful/harmless separation along the base direction.

``drift.persona`` (RM-03, RM-05) builds a trait direction from contrastive system prompts
(Chen et al., arXiv:2507.21509, prompt-only variant) and scores the candidate's mean
activation shift along the base's unit trait direction on neutral questions (the paper's
"finetuning shift"), divided by the base's trait gap. It also reports cosine and norm ratio.

Intervals are percentile bootstraps over prompts, resampled jointly for base and candidate.
Comparing raw activations assumes the candidate kept the base's residual-stream basis, which
holds for fine-tunes and merges of that base and not across pretrained models
(``change_type: base_upgrade``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from changegate.artifacts import ArtifactRef
from changegate.checks import (
    CheckResult,
    Interval,
    RmId,
    ThresholdRef,
    Tier,
    apply_envelope,
)
from changegate.config import CheckConfig
from changegate.digests import sha256_file, sha256_json
from changegate.directions import (
    Direction,
    DirectionKind,
    Dtype,
    PromptSet,
    cosine,
    default_layer,
    difference_of_means,
    dprime,
    load_model,
    load_persona_spec,
    load_prompt_set,
    make_direction,
    persona_prompt_sets,
    project,
    residual_activations,
    save_direction,
)
from changegate.errors import ArtifactError, ConfigError

Metrics = dict[str, float | None]
_P = TypeVar("_P", bound=BaseModel)
_VANISHED = 1e-9  # a candidate direction this small relative to the base's has no orientation


class DriftParams(BaseModel):
    """Settings shared by both drift checks, read from ``checks.<name>.params`` in gate.yaml.

    ``layer`` None means ``num_layers // 2`` of the base. ``bootstrap`` 0 turns intervals
    off. Relative paths resolve against the working directory. ``save_dir`` writes both
    models' directions there for other checks to reuse; evidence then carries each ``.npy``
    file's SHA-256, which the lineage check's ``refusal_direction_sha256`` can pin.

    Does not establish: that the chosen layer or position is where the model carries the
    behavior.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer: int | None = Field(None, ge=0)
    position: int = Field(-1, le=-1)
    chat_template: bool = True
    batch_size: int = Field(16, ge=1)
    dtype: Dtype = "auto"
    device: str | None = None
    bootstrap: int = Field(1000, ge=0)
    ci_level: float = Field(0.95, gt=0, lt=1)
    seed: int = 0
    save_dir: Path | None = None


class RefusalDriftParams(DriftParams):
    """``drift.refusal`` settings: the harmful and harmless prompt sets, plus shared settings.

    Does not establish: that the harmful set elicits refusal from the base.
    """

    harmful: Path
    harmless: Path


class PersonaDriftParams(DriftParams):
    """``drift.persona`` settings: the persona spec file, plus shared settings.

    Does not establish: that the spec's system prompts elicit its trait.
    """

    persona: Path


def refusal_drift(
    harmful_base: np.ndarray,
    harmless_base: np.ndarray,
    harmful_candidate: np.ndarray,
    harmless_candidate: np.ndarray,
) -> Metrics:
    """Compare refusal directions and harmful-prompt projections at one layer.

    Args:
        harmful_base: base activations on harmful prompts, shape (n_h, d).
        harmless_base: base activations on harmless prompts, shape (n_s, d).
        harmful_candidate: candidate activations on the same harmful prompts, (n_h, d).
        harmless_candidate: candidate activations on the same harmless prompts, (n_s, d).

    Returns:
        ``cosine``, ``norm_base``, ``norm_candidate``, ``norm_ratio``,
        ``harmful_projection_base``, ``harmful_projection_candidate``, ``harmful_shift``,
        ``harmful_shift_relative``, ``harmless_shift``, ``gap_retained``, ``dprime_base`` and
        ``dprime_candidate``. A value is None where it is undefined: every value but the
        norms when the base direction is zero, and ``cosine`` when the candidate direction
        has vanished (norm below 1e-9 of the base's; ``norm_ratio`` then shows it).

    Does not establish: a behavioral change. A drop in cosine or gap is a representation
    measurement; whether the candidate refuses less is for the behavior battery.
    """
    r_base = difference_of_means(harmful_base, harmless_base)
    r_cand = difference_of_means(harmful_candidate, harmless_candidate)
    norm_base, norm_cand = float(np.linalg.norm(r_base)), float(np.linalg.norm(r_cand))
    out: Metrics = {
        "cosine": _cosine(r_base, r_cand, norm_base, norm_cand),
        "norm_base": norm_base,
        "norm_candidate": norm_cand,
    }
    keys = (
        "norm_ratio",
        "harmful_projection_base",
        "harmful_projection_candidate",
        "harmful_shift",
        "harmful_shift_relative",
        "harmless_shift",
        "gap_retained",
        "dprime_base",
        "dprime_candidate",
    )
    if norm_base == 0.0:
        return out | dict.fromkeys(keys)
    unit = r_base / norm_base
    hb, sb = project(harmful_base, unit), project(harmless_base, unit)
    hc, sc = project(harmful_candidate, unit), project(harmless_candidate, unit)
    shift = float(hc.mean() - hb.mean())
    values = (
        norm_cand / norm_base,
        float(hb.mean()),
        float(hc.mean()),
        shift,
        shift / norm_base,
        float(sc.mean() - sb.mean()),
        float(hc.mean() - sc.mean()) / norm_base,
        dprime(hb, sb),
        dprime(hc, sc),
    )
    return out | dict(zip(keys, values, strict=True))


def persona_drift(
    positive_base: np.ndarray,
    negative_base: np.ndarray,
    neutral_base: np.ndarray,
    positive_candidate: np.ndarray,
    negative_candidate: np.ndarray,
    neutral_candidate: np.ndarray,
) -> Metrics:
    """Compare persona directions and the neutral-prompt shift along the base's, at one layer.

    Arrays may have any leading shape; the last axis is the residual stream.

    Args:
        positive_base: base activations under trait-eliciting system prompts.
        negative_base: base activations under trait-suppressing system prompts.
        neutral_base: base activations on the questions alone.
        positive_candidate: candidate activations, as ``positive_base``.
        negative_candidate: candidate activations, as ``negative_base``.
        neutral_candidate: candidate activations on the same questions, as ``neutral_base``.

    Returns:
        ``cosine``, ``norm_base``, ``norm_candidate``, ``norm_ratio``,
        ``neutral_projection_base``, ``neutral_projection_candidate``, ``shift`` and
        ``shift_relative`` (shift divided by the base's trait gap); None where undefined, as
        in ``refusal_drift``.

    Does not establish: that the candidate expresses the trait more in its outputs; the paper
    ties this shift to behavior with a judge, which is not used here.
    """
    d = neutral_base.shape[-1]
    pb, nb, qb, pc, nc, qc = (
        a.reshape(-1, d)
        for a in (
            positive_base,
            negative_base,
            neutral_base,
            positive_candidate,
            negative_candidate,
            neutral_candidate,
        )
    )
    v_base, v_cand = difference_of_means(pb, nb), difference_of_means(pc, nc)
    norm_base, norm_cand = float(np.linalg.norm(v_base)), float(np.linalg.norm(v_cand))
    out: Metrics = {
        "cosine": _cosine(v_base, v_cand, norm_base, norm_cand),
        "norm_base": norm_base,
        "norm_candidate": norm_cand,
    }
    keys = (
        "norm_ratio",
        "neutral_projection_base",
        "neutral_projection_candidate",
        "shift",
        "shift_relative",
    )
    if norm_base == 0.0:
        return out | dict.fromkeys(keys)
    unit = v_base / norm_base
    before, after = float(project(qb, unit).mean()), float(project(qc, unit).mean())
    values = (norm_cand / norm_base, before, after, after - before, (after - before) / norm_base)
    return out | dict(zip(keys, values, strict=True))


def bootstrap_intervals(
    stat: Callable[..., Metrics],
    groups: Sequence[Sequence[np.ndarray]],
    *,
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, list[float] | None]:
    """Percentile-bootstrap every metric a statistic returns.

    Each replicate draws one index vector per group (with replacement) and applies it to
    every array in that group, so base and candidate rows for the same prompt stay paired.
    ``stat`` receives the resampled arrays in group order, flattened.

    Args:
        stat: a metric function returning a dict of floats (or None).
        groups: groups of arrays that share their first axis.
        n_boot: number of replicates, at least 1.
        seed: numpy Generator seed.
        level: interval coverage, for example 0.95.

    Returns:
        ``[lo, hi]`` per metric, or None for a metric undefined in any replicate.

    Does not establish: coverage for prompts outside the fixed sets; resampling the given
    prompts says nothing about model or numerical nondeterminism, or other prompt sources.
    Percentile intervals are biased at a bound: when the candidate was edited along exactly
    the full-sample base direction, the point cosine is 0 while every replicate's base
    direction differs slightly, so the interval can lie above the point estimate.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        arrays = []
        for group in groups:
            index = rng.integers(0, len(group[0]), len(group[0]))
            arrays.extend(a[index] for a in group)
        for key, value in stat(*arrays).items():
            draws.setdefault(key, []).append(np.nan if value is None else value)
    tail = (1.0 - level) / 2.0
    return {
        key: None
        if np.isnan(values).any()
        else [float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))]
        for key, values in draws.items()
    }


@dataclass(frozen=True, eq=False)
class _Read:
    acts: tuple[np.ndarray, ...]
    direction: Direction
    device: str


class RefusalDriftCheck:
    """``drift.refusal``: refusal-direction cosine, norm and harmful-prompt projection shift.

    Tier C: loads base and candidate in turn and runs 2 x (n_harmful + n_harmless) forward
    passes. The score is ``cosine``; a typical envelope is ``lower_is_worse``.

    Does not establish: that the candidate refuses less or more. An abliterated or eroded
    model usually moves these numbers, but a change can move behavior without moving them,
    and training against this check can hide it.
    """

    name = "drift.refusal"
    rm_ids: tuple[RmId, ...] = ("RM-08", "RM-13")
    tier: Tier = "C"

    def run(self, base: ArtifactRef, candidate: ArtifactRef, cfg: CheckConfig) -> CheckResult:
        """Measure refusal drift of the candidate against the base.

        Args:
            base: the reference checkpoint.
            candidate: the changed checkpoint, derived from the base.
            cfg: this check's config; ``params`` follow ``RefusalDriftParams``.

        Returns:
            The result; the verdict applies the envelope to the point estimate of the cosine.

        Raises:
            ConfigError: on invalid params, prompt sets or layer.
            ArtifactError: if a model cannot load or the two do not share a width and depth.
            NotBuiltError: for adapter, quantized or single-file artifacts.

        Does not establish: anything at layers or positions other than the one scored;
        per-layer numbers in ``evidence`` are reported, not judged.
        """
        params = _parse(RefusalDriftParams, cfg)
        sets = (load_prompt_set(params.harmful), load_prompt_set(params.harmless))
        started = time.perf_counter()
        b, c = _read_pair(base, candidate, sets, params, kind="refusal")
        layer = b.direction.meta.layer
        at = [a[:, layer] for a in (*b.acts, *c.acts)]
        metrics = refusal_drift(*at)
        intervals = _intervals(
            lambda hb, hc, sb, sc: refusal_drift(hb, sb, hc, sc),
            [(at[0], at[2]), (at[1], at[3])],
            params,
        )
        evidence = _evidence(self.name, params, b, c, sets, metrics, intervals)
        evidence["per_layer"] = _per_layer(
            refusal_drift, [*b.acts, *c.acts], ("cosine", "norm_ratio", "gap_retained")
        )
        return _result(self, cfg, params, metrics, "cosine", intervals, evidence, started, b)


class PersonaDriftCheck:
    """``drift.persona``: shift of neutral-prompt activations along the base's trait direction.

    Tier C: loads base and candidate in turn and runs forward passes on every
    (system prompt, question) pair plus the questions alone. The score is ``shift_relative``;
    a typical envelope for an unwanted trait is ``higher_is_worse``.

    Does not establish: that the candidate expresses the trait in its outputs, or anything
    about traits not named in the spec.
    """

    name = "drift.persona"
    rm_ids: tuple[RmId, ...] = ("RM-03", "RM-05")
    tier: Tier = "C"

    def run(self, base: ArtifactRef, candidate: ArtifactRef, cfg: CheckConfig) -> CheckResult:
        """Measure persona drift of the candidate against the base.

        Args:
            base: the reference checkpoint.
            candidate: the changed checkpoint, derived from the base.
            cfg: this check's config; ``params`` follow ``PersonaDriftParams``.

        Returns:
            The result; the verdict applies the envelope to the point estimate of
            ``shift_relative``. Intervals resample questions.

        Raises:
            ConfigError: on invalid params, persona spec or layer.
            ArtifactError: if a model cannot load or the two do not share a width and depth.
            NotBuiltError: for adapter, quantized or single-file artifacts.

        Does not establish: the paper's response-based persona vector; this variant reads
        prompt tokens only.
        """
        params = _parse(PersonaDriftParams, cfg)
        spec = load_persona_spec(params.persona)
        sets = persona_prompt_sets(spec)
        started = time.perf_counter()
        b, c = _read_pair(base, candidate, sets, params, kind="persona", trait=spec.trait)
        layer = b.direction.meta.layer
        n_q = len(spec.questions)
        at = [a[:, layer].reshape(n_q, -1, a.shape[-1]) for a in (*b.acts, *c.acts)]
        metrics = persona_drift(*at)
        intervals = _intervals(persona_drift, [at], params)
        evidence = _evidence(self.name, params, b, c, sets, metrics, intervals)
        evidence["trait"] = spec.trait
        evidence["per_layer"] = _per_layer(
            persona_drift, [*b.acts, *c.acts], ("cosine", "norm_ratio", "shift_relative")
        )
        return _result(
            self, cfg, params, metrics, "shift_relative", intervals, evidence, started, b
        )


def _cosine(a: np.ndarray, b: np.ndarray, norm_a: float, norm_b: float) -> float | None:
    return cosine(a, b) if norm_b > _VANISHED * norm_a else None


def _parse(model: type[_P], cfg: CheckConfig) -> _P:
    try:
        return model.model_validate(cfg.params)
    except ValidationError as exc:
        raise ConfigError(f"params: {exc}") from exc


def _read_pair(
    base: ArtifactRef,
    candidate: ArtifactRef,
    sets: Sequence[PromptSet],
    params: DriftParams,
    *,
    kind: DirectionKind,
    trait: str | None = None,
) -> tuple[_Read, _Read]:
    reads: list[_Read] = []
    layer = params.layer
    for ref in (base, candidate):
        lm = load_model(ref, dtype=params.dtype, device=params.device)
        layer = default_layer(lm.num_layers) if layer is None else layer
        if not 0 <= layer < lm.num_layers:
            raise ConfigError(f"layer {layer} is outside 0..{lm.num_layers - 1}")
        if reads and (lm.num_layers, lm.hidden_size) != (
            reads[0].direction.meta.num_layers,
            reads[0].direction.meta.hidden_size,
        ):
            raise ArtifactError(
                "base and candidate differ in depth or width; drift compares only models "
                "that share a base"
            )
        acts = tuple(
            residual_activations(
                lm,
                s.items,
                position=params.position,
                batch_size=params.batch_size,
                chat_template=params.chat_template,
            )
            for s in sets
        )
        direction = make_direction(
            acts[0],
            acts[1],
            lm=lm,
            kind=kind,
            layer=layer,
            position=params.position,
            positive=sets[0],
            negative=sets[1],
            chat_template=params.chat_template,
            trait=trait,
        )
        reads.append(_Read(acts, direction, lm.device))
        del lm
    return reads[0], reads[1]


def _intervals(
    stat: Callable[..., Metrics], groups: Sequence[Sequence[np.ndarray]], params: DriftParams
) -> dict[str, list[float] | None]:
    if params.bootstrap == 0:
        return {}
    return bootstrap_intervals(
        stat, groups, n_boot=params.bootstrap, seed=params.seed, level=params.ci_level
    )


def _per_layer(
    stat: Callable[..., Metrics], acts: Sequence[np.ndarray], keys: Sequence[str]
) -> list[dict[str, Any]]:
    rows = []
    for layer in range(acts[0].shape[1]):
        metrics = stat(*(a[:, layer] for a in acts))
        rows.append({"layer": layer} | {k: metrics[k] for k in keys})
    return rows


def _evidence(
    check: str,
    params: DriftParams,
    base: _Read,
    cand: _Read,
    sets: Sequence[PromptSet],
    metrics: Metrics,
    intervals: dict[str, list[float] | None],
) -> dict[str, Any]:
    meta = base.direction.meta
    directions = {}
    for role, read in (("base", base), ("candidate", cand)):
        entry: dict[str, Any] = {
            "model_sha256": read.direction.meta.model_sha256,
            "vector_sha256": read.direction.meta.vector_sha256,
            "raw_norm": read.direction.meta.raw_norm,
            "chat_template_sha256": read.direction.meta.chat_template_sha256,
            "dtype": read.direction.meta.dtype,
        }
        if params.save_dir is not None:
            stem = f"{role}-{check.replace('.', '-')}-L{meta.layer}"
            npy = save_direction(read.direction, params.save_dir / stem)
            # relative to save_dir: evidence is copied into records meant to be shared
            entry |= {"path": npy.name, "file_sha256": sha256_file(npy)}
        directions[role] = entry
    return {
        "method": meta.method,
        "decision_rule": "verdict from the point estimate at the scored layer; the interval "
        "is a percentile bootstrap over prompts, paired across base and candidate, and does "
        "not enter the verdict",
        "layer": meta.layer,
        "layer_source": "declared" if params.layer is not None else "default: num_layers // 2",
        "position": meta.position,
        "num_layers": meta.num_layers,
        "hidden_size": meta.hidden_size,
        "chat_template": params.chat_template,
        "template_differs": base.direction.meta.chat_template_sha256
        != cand.direction.meta.chat_template_sha256,
        "prompts": [s.summary() for s in sets],
        "directions": directions,
        "metrics": metrics,
        "intervals": intervals,
    }


def _result(
    check: RefusalDriftCheck | PersonaDriftCheck,
    cfg: CheckConfig,
    params: DriftParams,
    metrics: Metrics,
    score_key: str,
    intervals: dict[str, list[float] | None],
    evidence: dict[str, Any],
    started: float,
    base: _Read,
) -> CheckResult:
    score = metrics[score_key]
    bounds = intervals.get(score_key)
    interval = (
        Interval(
            lo=bounds[0],
            hi=bounds[1],
            level=params.ci_level,
            method=f"percentile bootstrap over prompts, B={params.bootstrap}, seed={params.seed}",
        )
        if bounds and score is not None
        else None
    )
    threshold_ref = (
        ThresholdRef(
            check=check.name,
            envelope=cfg.envelope,
            config_sha256=sha256_json(cfg.model_dump(mode="json")),
        )
        if cfg.envelope is not None
        else None
    )
    wall = time.perf_counter() - started
    n_prompts = sum(len(a) for a in base.acts)
    return CheckResult(
        check=check.name,
        rm_ids=check.rm_ids,
        score=score,
        interval=interval,
        threshold_ref=threshold_ref,
        verdict=apply_envelope(score, cfg.envelope),
        recall=None,
        evidence=evidence | {"score_key": score_key},
        cost={
            "wall_seconds": wall,
            "gpu_seconds": wall if base.device.startswith("cuda") else 0.0,
            "forward_prompts": float(2 * n_prompts),
        },
    )
