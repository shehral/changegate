"""Contrast directions in the residual stream: prompt sets, activations and stored directions.

A direction is the difference of mean residual-stream activations between two prompt sets, at
one layer and one token position. For refusal the sets are harmful and harmless instructions
(Arditi et al., arXiv:2406.11717). For a persona trait they are the same questions under
trait-eliciting and trait-suppressing system prompts (Chen et al., arXiv:2507.21509, in a
prompt-only variant; see ``persona_prompt_sets``).

Layer ``l`` is the residual stream entering decoder block ``l`` (``hidden_states[l]`` in
transformers, ``resid_pre`` in the paper), for ``0 <= l < num_layers``. Position ``-k`` is the
k-th token from the end of the rendered prompt; with a chat template, ``-1`` is the last token
of the generation prompt.

Only numpy is imported with this module. torch and transformers are imported inside the
functions that run a model, so a CPU job can read a stored direction without them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from changegate import __version__
from changegate.artifacts import ArtifactRef, Sha256
from changegate.battery import BatteryItem, Message
from changegate.digests import canonical_json, sha256_bytes, sha256_json
from changegate.errors import ArtifactError, ConfigError, NotBuiltError

PERMISSIVE_LICENSES = frozenset(
    {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0"}
)
DirectionKind = Literal["refusal", "persona"]
Dtype = Literal["auto", "float32", "bfloat16", "float16"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


def is_permissive(license_id: str) -> bool:
    """Say whether a declared license id is on changegate's permissive allowlist.

    Args:
        license_id: an SPDX-style id as a dataset host reports it (case is ignored).

    Returns:
        True for MIT, Apache-2.0, BSD-2/3-Clause, CC-BY-4.0 and CC0-1.0.

    Does not establish: that the declared license is correct, or that every row in the
    dataset was contributed under it.
    """
    return license_id.strip().casefold() in PERMISSIVE_LICENSES


class PromptSet(BaseModel):
    """A fixed, ordered list of prompts with the dataset and license they came from.

    ``source`` is a key in DATA-LICENSES.md, or ``synthetic``; every item carries the same
    source. Provenance fields are filled by the fetch script and are None for sets built in
    memory (persona sets).

    Does not establish: that the prompts are representative of harmful or harmless requests,
    or that they were absent from any model's training data.
    """

    model_config = _FROZEN

    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    items: tuple[BatteryItem, ...] = Field(min_length=2)
    dataset: str | None = None
    source_url: str | None = None
    revision: str | None = None
    license_file_sha256: Sha256 | None = None
    n_source_rows: int | None = Field(None, ge=0)
    seed: int | None = None
    fetched: date | None = None

    @model_validator(mode="after")
    def _usable(self) -> PromptSet:
        """Reject a non-permissive license, mixed sources and duplicate item ids.

        Does not establish: anything about the prompts' content.
        """
        if not is_permissive(self.license):
            raise ValueError(f"license {self.license!r} is not on the permissive allowlist")
        if any(item.source != self.source for item in self.items):
            raise ValueError(f"every item's source must be {self.source!r}")
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("item ids must be unique")
        return self

    @property
    def sha256(self) -> str:
        """Return the digest of the items' ids, messages, categories and sources, in order.

        Only these four fields enter, so the digest of a stored set does not change when
        ``BatteryItem`` gains optional fields.

        Does not establish: which rendering (chat template) a model saw.
        """
        return sha256_json([_row(item) for item in self.items])

    def summary(self) -> dict[str, Any]:
        """Return the fields a result cites for this set: name, source, license, n, digest.

        Does not establish: more than the fields it copies.
        """
        return {
            "name": self.name,
            "source": self.source,
            "license": self.license,
            "revision": self.revision,
            "n": len(self.items),
            "sha256": self.sha256,
        }


_ROW_FIELDS = frozenset({"id", "messages", "category", "source"})


def _row(item: BatteryItem) -> dict[str, Any]:
    return item.model_dump(mode="json", include=_ROW_FIELDS)


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(".jsonl") + ".meta.json")


def save_prompt_set(prompt_set: PromptSet, path: Path) -> Path:
    """Write a prompt set as JSONL items plus a ``.meta.json`` sidecar with its provenance.

    Args:
        prompt_set: the set to write.
        path: the ``.jsonl`` file; the sidecar goes next to it.

    Returns:
        The sidecar's path.

    Does not establish: the license; it is copied from the set, which the fetch script read
    from the host.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(_row(i)) + "\n" for i in prompt_set.items))
    meta = prompt_set.model_dump(mode="json", exclude={"items"})
    meta.update(n=len(prompt_set.items), items_sha256=prompt_set.sha256)
    meta_path = _meta_path(path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta_path


def load_prompt_set(path: Path) -> PromptSet:
    """Read a prompt set written by ``save_prompt_set`` and check it against its sidecar.

    Args:
        path: the ``.jsonl`` file.

    Returns:
        The set, with its license and provenance.

    Raises:
        ConfigError: if either file is missing or malformed, the item digest or count differs
            from the sidecar, or the license is not permissive.

    Does not establish: that the rows match the upstream dataset today; the sidecar records
    the revision they were fetched at.
    """
    path = Path(path).expanduser()
    try:
        meta = json.loads(_meta_path(path).read_text())
        items = tuple(
            BatteryItem.model_validate_json(line) for line in path.read_text().splitlines() if line
        )
        digest, n = meta.pop("items_sha256"), meta.pop("n")
        prompt_set = PromptSet.model_validate({**meta, "items": items})
    except (OSError, ValueError, KeyError, ValidationError) as exc:
        raise ConfigError(f"prompt set {path}: {exc}") from exc
    if prompt_set.sha256 != digest or len(items) != n:
        raise ConfigError(f"prompt set {path}: items differ from the digest in its sidecar")
    return prompt_set


class PersonaSpec(BaseModel):
    """A persona trait declared as contrastive system prompts plus neutral questions.

    Does not establish: that the system prompts elicit the named trait in any given model;
    the trait name is a label the author chose.
    """

    model_config = _FROZEN

    trait: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    description: str = ""
    positive_system: tuple[str, ...] = Field(min_length=1)
    negative_system: tuple[str, ...] = Field(min_length=1)
    questions: tuple[str, ...] = Field(min_length=2)


def load_persona_spec(path: Path) -> PersonaSpec:
    """Read a persona spec from JSON.

    Args:
        path: the spec file.

    Returns:
        The validated spec.

    Raises:
        ConfigError: if the file is missing or does not match the schema.

    Does not establish: the license of the text beyond what the file declares.
    """
    try:
        return PersonaSpec.model_validate_json(Path(path).expanduser().read_text())
    except (OSError, ValidationError) as exc:
        raise ConfigError(f"persona spec {path}: {exc}") from exc


def persona_prompt_sets(spec: PersonaSpec) -> tuple[PromptSet, PromptSet, PromptSet]:
    """Build the positive, negative and neutral prompt sets for a persona trait.

    Positive and negative items pair every question with every positive (or negative) system
    prompt, ordered question-major; neutral items are the questions alone. This is a
    prompt-only variant of arXiv:2507.21509, which averages over generated response tokens
    filtered by an LLM judge. No generation and no judge are used here, so no model scores
    anything.

    Args:
        spec: the persona spec.

    Returns:
        (positive, negative, neutral) prompt sets.

    Does not establish: that the prompt-only direction matches the paper's response-based
    persona vector.
    """

    def build(kind: str, systems: Sequence[str | None]) -> PromptSet:
        items = []
        for q, question in enumerate(spec.questions):
            for s, system in enumerate(systems):
                messages = [Message(role="user", content=question)]
                if system is not None:
                    messages.insert(0, Message(role="system", content=system))
                items.append(
                    BatteryItem(
                        id=f"{spec.trait}-{kind}-q{q:03d}-s{s:02d}",
                        messages=tuple(messages),
                        category=f"{spec.trait}/{kind}",
                        source=spec.source,
                    )
                )
        return PromptSet(
            name=f"{spec.trait}/{kind}", source=spec.source, license=spec.license, items=items
        )

    return (
        build("positive", spec.positive_system),
        build("negative", spec.negative_system),
        build("neutral", [None]),
    )


@dataclass(frozen=True, eq=False)
class LoadedModel:
    """A causal LM and its tokenizer loaded from an artifact, on one device.

    Does not establish: that the files on disk still match ``ref.sha256``; the gate runner
    verifies digests before a check loads anything.
    """

    model: Any
    tokenizer: Any
    ref: ArtifactRef
    device: str
    dtype: str

    @property
    def num_layers(self) -> int:
        """Return the number of decoder blocks.

        Does not establish: anything about the blocks' contents.
        """
        return int(self.model.config.get_text_config().num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        """Return the residual-stream width.

        Does not establish: that two models of equal width share a basis.
        """
        return int(self.model.config.get_text_config().hidden_size)

    @property
    def chat_template_sha256(self) -> str | None:
        """Return the SHA-256 of the tokenizer's chat template, or None when it has none.

        Does not establish: that the template matches the one used in serving.
        """
        template = getattr(self.tokenizer, "chat_template", None)
        return None if template is None else sha256_json(template)


def load_model(
    ref: ArtifactRef, *, dtype: Dtype = "auto", device: str | None = None
) -> LoadedModel:
    """Load a full-weight Hugging Face checkpoint for activation reads.

    safetensors only and ``trust_remote_code`` off. ``auto`` picks bfloat16 on CUDA and
    float32 elsewhere; the device defaults to CUDA when visible, else CPU.

    Args:
        ref: a ``weights`` or ``merged`` artifact that is a checkpoint directory.
        dtype: the compute dtype.
        device: a torch device string, or None for the default.

    Returns:
        The model in eval mode and its tokenizer, with a pad token set.

    Raises:
        NotBuiltError: for adapters, quantized artifacts and single-file artifacts.
        ArtifactError: if transformers cannot load the directory.

    Does not establish: that the checkpoint is the one ``ref.base_ref`` claims.
    """
    if ref.kind not in ("weights", "merged") or not ref.path.is_dir():
        raise NotBuiltError(
            f"activation reads need a full-weight checkpoint directory; {ref.path} is "
            f"'{ref.kind}'. Merge an adapter into its base first. No measurement was made."
        )
    import torch
    import transformers

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = ("bfloat16" if device.startswith("cuda") else "float32") if dtype == "auto" else dtype
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(ref.path, trust_remote_code=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            ref.path, dtype=getattr(torch, name), trust_remote_code=False, use_safetensors=True
        )
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"{ref.path}: cannot load as a causal LM: {exc}") from exc
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ArtifactError(f"{ref.path}: tokenizer has neither a pad nor an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    return LoadedModel(model.to(device).eval(), tokenizer, ref, device, name)


def render(tokenizer: Any, messages: Sequence[Message], *, chat_template: bool) -> str:
    """Render chat messages to the text a model reads.

    Args:
        tokenizer: a transformers tokenizer.
        messages: the prompt's messages.
        chat_template: apply the tokenizer's chat template with the generation prompt, or
            join the message contents with blank lines.

    Returns:
        The prompt text.

    Raises:
        ConfigError: if ``chat_template`` is set and the tokenizer has none.

    Does not establish: that the rendering matches a serving stack's rendering.
    """
    if not chat_template:
        return "\n\n".join(m.content for m in messages)
    if getattr(tokenizer, "chat_template", None) is None:
        raise ConfigError("the tokenizer has no chat template; set chat_template: false")
    return tokenizer.apply_chat_template(
        [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True
    )


def residual_activations(
    lm: LoadedModel,
    items: Sequence[BatteryItem],
    *,
    position: int = -1,
    batch_size: int = 16,
    chat_template: bool = True,
) -> np.ndarray:
    """Read the residual stream entering every decoder block at one token position.

    Prompts are right-padded, so each row's tokens keep positions 0..len-1 and causal
    attention never reads padding; row ``i`` is read at index ``len_i + position``.

    Args:
        lm: the loaded model.
        items: the prompts.
        position: a negative token index into each rendered prompt.
        batch_size: prompts per forward pass.
        chat_template: passed to ``render``.

    Returns:
        float32 array of shape (len(items), num_layers, hidden_size).

    Raises:
        ValueError: if ``position`` is not negative or a prompt is shorter than it.
        ArtifactError: if the model returns an unexpected number of hidden states, or any
            activation is NaN or infinite.

    Does not establish: bitwise reproducibility across devices, dtypes or batch sizes.
    """
    if position > -1:
        raise ValueError("position must be negative (-1 is the last token)")
    import torch

    texts = [render(lm.tokenizer, item.messages, chat_template=chat_template) for item in items]
    out = np.empty((len(texts), lm.num_layers, lm.hidden_size), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        enc = lm.tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=not chat_template,
        )
        lengths = enc["attention_mask"].sum(dim=1)
        if int(lengths.min()) < -position:
            raise ValueError(f"a prompt is shorter than {-position} tokens")
        with torch.inference_mode():
            hidden = lm.model(
                input_ids=enc["input_ids"].to(lm.device),
                attention_mask=enc["attention_mask"].to(lm.device),
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states
        if len(hidden) != lm.num_layers + 1:
            raise ArtifactError(
                f"{lm.ref.path}: expected {lm.num_layers + 1} hidden states, got {len(hidden)}"
            )
        rows = torch.arange(len(lengths), device=lm.device)
        index = (lengths + position).to(lm.device)
        acts = torch.stack([h[rows, index] for h in hidden[: lm.num_layers]], dim=1)
        out[start : start + len(lengths)] = acts.float().cpu().numpy()
    if not np.isfinite(out).all():
        raise ArtifactError(f"{lm.ref.path}: non-finite activations; try dtype float32")
    return out


def difference_of_means(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Return mean(positive) - mean(negative) over the leading axis, in float64.

    Args:
        positive: activations, shape (n_pos, ...).
        negative: activations, shape (n_neg, ...), same trailing shape.

    Returns:
        The difference, shape (...).

    Does not establish: that the difference is a causal direction; Arditi et al. select and
    validate candidates by ablation and steering, which this function does not do.
    """
    return positive.mean(axis=0, dtype=np.float64) - negative.mean(axis=0, dtype=np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """Return the cosine of two vectors, or None when either has zero norm.

    Args:
        a: a vector.
        b: a vector of the same length.

    Returns:
        A value in [-1, 1], or None.

    Does not establish: that a high cosine means the same behavior; directions can agree
    while downstream weights change.
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def project(acts: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """Return the scalar projection of each activation row onto a unit vector.

    Computed with ``einsum`` rather than ``@``: the result is the same, and it avoids BLAS
    matrix-vector calls, which raise spurious floating-point warnings on some macOS numpy
    builds.

    Args:
        acts: activations, shape (..., d).
        unit: a vector of length d, normally of norm 1.

    Returns:
        Projections, shape (...), float64.

    Does not establish: that ``unit`` has norm 1; the caller normalizes.
    """
    return np.einsum("...d,d->...", acts, unit, dtype=np.float64)


def dprime(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Return the separation d' of two sets of scalar projections.

    d' = (mean(pos) - mean(neg)) / sqrt((var(pos) + var(neg)) / 2), with sample variances.

    Args:
        positive: projections of the positive prompts, at least two.
        negative: projections of the negative prompts, at least two.

    Returns:
        d', or None when both variances are zero.

    Does not establish: normality; d' summarizes, it does not test.
    """
    pooled = (np.var(positive, ddof=1) + np.var(negative, ddof=1)) / 2.0
    if pooled <= 0.0:
        return None
    return float((np.mean(positive) - np.mean(negative)) / np.sqrt(pooled))


def random_unit_vector(dim: int, seed: int) -> np.ndarray:
    """Return a Gaussian random unit vector, reproducible from its seed.

    Args:
        dim: the length.
        seed: the numpy Generator seed.

    Returns:
        float64 array of norm 1. Its cosine with any fixed vector has standard deviation
        about 1/sqrt(dim), which makes it a baseline for "no alignment".

    Does not establish: anything about a model; it is a sanity baseline.
    """
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


def default_layer(num_layers: int) -> int:
    """Return the layer used when none is declared: ``num_layers // 2``.

    Args:
        num_layers: the model's decoder block count.

    Returns:
        A layer index.

    Does not establish: that this layer carries the direction best. The paper selects a
    layer per model by ablation and steering; results record whether the layer was declared.
    """
    return num_layers // 2


class DirectionMeta(BaseModel):
    """What a stored direction was computed from: model, layer, position, prompts, date.

    The model is identified by digest and by its declared ``base_ref`` (``model_ref``), never
    by a local path: a sidecar travels with the vector and must not carry the home directory
    or cluster account it was computed under. A legacy ``model_path`` key is dropped on load.

    Does not establish: that the model files still match ``model_sha256``, or that the
    direction carries the behavior its ``kind`` names.
    """

    model_config = _FROZEN

    kind: DirectionKind
    trait: str | None = None
    model_sha256: Sha256
    model_ref: str | None = None
    layer: int = Field(ge=0)
    num_layers: int = Field(ge=1)
    position: int = Field(le=-1)
    hidden_size: int = Field(ge=1)
    positive: dict[str, Any]
    negative: dict[str, Any]
    chat_template: bool
    chat_template_sha256: Sha256 | None
    dtype: str
    raw_norm: float = Field(ge=0)
    vector_sha256: Sha256
    created: date
    method: str = "difference of mean residual-stream activations (arXiv:2406.11717)"
    changegate_version: str = __version__

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_path(cls, data: Any) -> Any:
        """Drop the legacy ``model_path`` key, so older sidecars load without the path.

        Does not establish: anything about the path; it is discarded unread.
        """
        if isinstance(data, dict) and "model_path" in data:
            data = {k: v for k, v in data.items() if k != "model_path"}
        return data

    @model_validator(mode="after")
    def _layer_in_range(self) -> DirectionMeta:
        """Reject a layer at or past the block count.

        Does not establish: that the layer is a good choice.
        """
        if self.layer >= self.num_layers:
            raise ValueError("layer must be below num_layers")
        return self


def _vector_digest(vector: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(vector, dtype="<f4").tobytes())


@dataclass(frozen=True, eq=False)
class Direction:
    """A stored contrast direction: the raw difference-of-means vector and its metadata.

    ``vector`` is not normalized; its norm is the positive/negative gap along it.

    Does not establish: causal involvement in behavior; see ``difference_of_means``.
    """

    vector: np.ndarray
    meta: DirectionMeta

    @property
    def unit(self) -> np.ndarray:
        """Return the vector scaled to norm 1 (float64).

        Raises:
            ValueError: if the vector is zero.

        Does not establish: anything beyond the vector.
        """
        norm = float(np.linalg.norm(self.vector))
        if norm == 0.0:
            raise ValueError("zero direction has no unit vector")
        return self.vector.astype(np.float64) / norm


def make_direction(
    positive_acts: np.ndarray,
    negative_acts: np.ndarray,
    *,
    lm: LoadedModel,
    kind: DirectionKind,
    layer: int,
    position: int,
    positive: PromptSet,
    negative: PromptSet,
    chat_template: bool,
    trait: str | None = None,
) -> Direction:
    """Build a Direction from activations already read with ``residual_activations``.

    Args:
        positive_acts: activations of the positive set, shape (n_pos, num_layers, d).
        negative_acts: activations of the negative set, shape (n_neg, num_layers, d).
        lm: the model the activations came from.
        kind: ``refusal`` or ``persona``.
        layer: the layer to take.
        position: the token position the activations were read at.
        positive: the positive prompt set (harmful, or trait-eliciting).
        negative: the negative prompt set (harmless, or trait-suppressing).
        chat_template: whether prompts were rendered with the chat template.
        trait: the persona trait, for ``kind="persona"``.

    Returns:
        The direction, stored as float32, with metadata dated today (UTC).

    Does not establish: that the activations were read from ``lm``; the caller passes both.
    """
    vector = difference_of_means(positive_acts[:, layer], negative_acts[:, layer]).astype(
        np.float32
    )
    meta = DirectionMeta(
        kind=kind,
        trait=trait,
        model_sha256=lm.ref.sha256,
        model_ref=lm.ref.base_ref,
        layer=layer,
        num_layers=lm.num_layers,
        position=position,
        hidden_size=lm.hidden_size,
        positive=positive.summary(),
        negative=negative.summary(),
        chat_template=chat_template,
        chat_template_sha256=lm.chat_template_sha256 if chat_template else None,
        dtype=lm.dtype,
        raw_norm=float(np.linalg.norm(vector)),
        vector_sha256=_vector_digest(vector),
        created=datetime.now(UTC).date(),
    )
    return Direction(vector=vector, meta=meta)


def compute_direction(
    lm: LoadedModel,
    positive: PromptSet,
    negative: PromptSet,
    *,
    kind: DirectionKind,
    layer: int | None = None,
    position: int = -1,
    batch_size: int = 16,
    chat_template: bool = True,
    trait: str | None = None,
) -> Direction:
    """Run both prompt sets through a model and return their difference-of-means direction.

    Args:
        lm: the loaded model.
        positive: the positive prompt set.
        negative: the negative prompt set.
        kind: ``refusal`` or ``persona``.
        layer: the layer, or None for ``default_layer``.
        position: the token position.
        batch_size: prompts per forward pass.
        chat_template: passed to ``render``.
        trait: the persona trait, for ``kind="persona"``.

    Returns:
        The direction.

    Raises:
        ConfigError: if the layer is out of range.

    Does not establish: that the prompts elicit refusal (or the trait) in this model; the
    paper filters prompts by the model's own refusal score, which this does not do.
    """
    layer = default_layer(lm.num_layers) if layer is None else layer
    if not 0 <= layer < lm.num_layers:
        raise ConfigError(f"layer {layer} is outside 0..{lm.num_layers - 1}")
    kwargs = {"position": position, "batch_size": batch_size, "chat_template": chat_template}
    return make_direction(
        residual_activations(lm, positive.items, **kwargs),
        residual_activations(lm, negative.items, **kwargs),
        lm=lm,
        kind=kind,
        layer=layer,
        position=position,
        positive=positive,
        negative=negative,
        chat_template=chat_template,
        trait=trait,
    )


def _stem(path: Path) -> Path:
    path = Path(path).expanduser()
    for suffix in (".npy", ".json"):
        if path.name.endswith(suffix):
            return path.with_name(path.name.removesuffix(suffix))
    return path


def save_direction(direction: Direction, path: Path) -> Path:
    """Write a direction as ``<stem>.npy`` (float32, no pickle) and ``<stem>.json``.

    Args:
        direction: the direction.
        path: the stem, or either file's path.

    Returns:
        The ``.npy`` path.

    Does not establish: provenance beyond the metadata written beside it; the files are not
    signed.
    """
    stem = _stem(path)
    stem.parent.mkdir(parents=True, exist_ok=True)
    npy = stem.with_name(stem.name + ".npy")
    np.save(npy, np.ascontiguousarray(direction.vector, dtype="<f4"), allow_pickle=False)
    stem.with_name(stem.name + ".json").write_text(
        json.dumps(direction.meta.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return npy


def load_direction(path: Path) -> Direction:
    """Read a direction written by ``save_direction`` and check the vector against its digest.

    Args:
        path: the stem, or either file's path.

    Returns:
        The direction.

    Raises:
        ArtifactError: if a file is missing or malformed, or the vector's shape or digest
            differs from the metadata.

    Does not establish: that the metadata was not rewritten together with the vector.
    """
    stem = _stem(path)
    try:
        meta = DirectionMeta.model_validate_json(stem.with_name(stem.name + ".json").read_text())
        vector = np.load(stem.with_name(stem.name + ".npy"), allow_pickle=False)
    except (OSError, ValueError, ValidationError) as exc:
        raise ArtifactError(f"direction {stem}: {exc}") from exc
    if vector.shape != (meta.hidden_size,) or _vector_digest(vector) != meta.vector_sha256:
        raise ArtifactError(f"direction {stem}: vector does not match its metadata")
    return Direction(vector=vector.astype(np.float32), meta=meta)
