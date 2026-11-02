"""Safetensors-only tensor access and the weight-space sums behind the lineage check.

Nothing here unpickles. Weights are read through ``safetensors.safe_open`` (a JSON header and
raw bytes), and no model class is built, so no repository code runs. torch, safetensors and
numpy are imported on first use, so ``changegate`` still imports without the cpu or gpu extra.

Every sum is taken in float64, one block of rows at a time, so memory is bounded by
``max_elements`` per tensor, not by the size of the model.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from changegate.errors import ArtifactError, ChangegateError, ConfigError

if TYPE_CHECKING:
    from changegate.checks.lineage_direction import RefusalDirection

PICKLE_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".joblib"})
FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"})

_BLOCK = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")
_VOCAB = re.compile(r"(?:^|\.)(?:embed_tokens|wte|lm_head)\.weight")
# Matrices whose output is added to the residual stream, with the axis that indexes the
# residual dimension, in Llama-family naming (Llama, Mistral, Qwen2/2.5/3, Gemma, Phi-3).
# Axis 0 is an nn.Linear output dimension; axis 1 is the width of an embedding row.
_RESIDUAL_WRITERS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r".*\.self_attn\.o_proj\.weight"), 0),
    (re.compile(r".*\.mlp\.down_proj\.weight"), 0),
    (re.compile(r".*\.mlp\.(?:shared_experts?|experts\.\d+)\.down_proj\.weight"), 0),
    (re.compile(r"(?:.*\.)?embed_tokens\.weight"), 1),
)


class MissingExtraError(ChangegateError):
    """torch, safetensors or numpy is missing; the lineage check needs the cpu or gpu extra.

    Does not establish: anything about the artifacts; no weights were read.
    """


def require(module: str) -> Any:
    """Import one of the check's ML dependencies, or raise MissingExtraError naming the extra.

    Args:
        module: ``torch``, ``numpy`` or ``safetensors``.

    Returns:
        The imported module.

    Does not establish: that the installed version is inside the pinned range.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(
            f"the lineage check needs {module}: install the cpu or gpu extra"
        ) from exc


def require_device(device: str) -> None:
    """Check that the requested device exists before any file is read.

    Args:
        device: ``cpu`` or ``cuda``.

    Raises:
        ConfigError: when ``cuda`` is requested and torch sees no CUDA device.
        MissingExtraError: if torch is not installed.

    Does not establish: that the device has enough memory for the chunk size.
    """
    if device == "cuda" and not require("torch").cuda.is_available():
        raise ConfigError("device 'cuda' was requested but no CUDA device is visible")


def block_of(name: str) -> int | None:
    """Return the decoder-block index a tensor name belongs to.

    Args:
        name: a tensor name such as ``model.layers.3.mlp.down_proj.weight``.

    Returns:
        The block index, or None for tensors outside the block stack (embeddings, final norm).

    Does not establish: that the naming follows the model's real layer order.
    """
    m = _BLOCK.search(name)
    return int(m.group(1)) if m else None


def writer_axis(name: str) -> int | None:
    """Return the residual axis of a residual-writing matrix, or None for any other tensor.

    Args:
        name: a tensor name.

    Returns:
        0 for attention-output and MLP down projections, 1 for token embeddings, else None.

    Does not establish: coverage of families with other naming; their matrices are not matched.
    """
    for pattern, axis in _RESIDUAL_WRITERS:
        if pattern.fullmatch(name):
            return axis
    return None


def row_chunks(n_rows: int, row_size: int, max_elements: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` row ranges holding at most ``max_elements`` values (one row min).

    Args:
        n_rows: rows in the tensor (1 for a scalar).
        row_size: values per row.
        max_elements: the memory bound per chunk.

    Returns:
        An iterator of half-open row ranges covering ``[0, n_rows)``.

    Does not establish: anything about the tensor; it only partitions indices.
    """
    step = max(1, max_elements // max(1, row_size))
    for start in range(0, n_rows, step):
        yield start, min(n_rows, start + step)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, refusing NaN and infinity so digests of it stay well defined.

    Args:
        path: a JSON file.

    Returns:
        The parsed object.

    Raises:
        ArtifactError: if the file is unreadable, not JSON, not an object, or holds NaN.

    Does not establish: that the values are what the file's name suggests.
    """

    def reject(token: str) -> Any:
        raise ValueError(f"non-finite constant {token}")

    try:
        data = json.loads(Path(path).read_text(), parse_constant=reject)
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError(f"{path}: top level must be a JSON object")
    return data


class TensorSource(Protocol):
    """Named tensors readable a block of rows at a time, as float64 on one device.

    Does not establish: where the values came from; a source reports what it reads.
    """

    def names(self) -> frozenset[str]:
        """Return every tensor name. Does not establish: that the set is complete."""
        ...

    def shape(self, name: str) -> tuple[int, ...]:
        """Return a tensor's shape. Does not establish: anything about its values."""
        ...

    def is_float(self, name: str) -> bool:
        """Return whether a tensor is stored as floating point. Does not establish: its scale."""
        ...

    def rows(self, name: str, start: int, stop: int, base_rows: Any = None) -> Any:
        """Return rows ``[start, stop)`` as float64.

        ``base_rows`` are the base's rows at the same range, for sources defined relative to
        the base (an adapter); plain files ignore them.

        Does not establish: that the rows equal what a model loader would build.
        """
        ...


class SafetensorsSource:
    """Read-only, row-sliced access to the tensors in one or more ``.safetensors`` files.

    When the artifact carries a ``model.safetensors.index.json``, every tensor it lists must sit
    in the shard it names, and no shard may hold a tensor it omits.

    Does not establish: that the files are the ones hashed; the caller re-verifies digests.
    """

    def __init__(self, files: Sequence[Path], device: str = "cpu", index: Path | None = None):
        """Open the files (memory-mapped) and map every tensor name to its shard.

        Args:
            files: ``.safetensors`` paths.
            device: where rows are placed (``cpu`` or ``cuda``).
            index: an optional sharding index to check the shards against.

        Raises:
            ArtifactError: on an unreadable file, a tensor in two shards, or an index mismatch.
            MissingExtraError: if safetensors or torch is not installed.

        Does not establish: that tensors hold sensible values; nothing is read until ``rows``.
        """
        safe_open = require("safetensors").safe_open
        require("torch")
        self.device = device
        self._stack = contextlib.ExitStack()
        self._where: dict[str, tuple[Path, Any]] = {}
        try:
            for path in files:
                try:
                    handle = self._stack.enter_context(
                        safe_open(str(path), framework="pt", device="cpu")
                    )
                except Exception as exc:  # SafetensorError has no narrower public base
                    raise ArtifactError(f"{path}: not a readable safetensors file: {exc}") from exc
                for name in handle.keys():  # noqa: SIM118 (safe_open is not iterable)
                    if name in self._where:
                        raise ArtifactError(f"{path}: tensor {name} is also in another shard")
                    self._where[name] = (Path(path), handle)
            if index is not None:
                self._check_index(index)
        except BaseException:
            self._stack.close()
            raise

    def _check_index(self, index: Path) -> None:
        weight_map = read_json(index).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ArtifactError(f"{index}: no weight_map")
        for name, shard in weight_map.items():
            where = self._where.get(name)
            if where is None or where[0].name != shard:
                raise ArtifactError(f"{index}: {name} is not in shard {shard}")
        unlisted = sorted(self._where.keys() - weight_map.keys())
        if unlisted:
            raise ArtifactError(f"{index}: {len(unlisted)} tensors are missing, e.g. {unlisted[0]}")

    def names(self) -> frozenset[str]:
        """Return every tensor name across the shards.

        Does not establish: that the set matches what the model's config implies.
        """
        return frozenset(self._where)

    def shape(self, name: str) -> tuple[int, ...]:
        """Return a tensor's shape from its header.

        Does not establish: anything about the tensor's values.
        """
        return tuple(self._slice(name).get_shape())

    def is_float(self, name: str) -> bool:
        """Return whether a tensor is stored in a floating-point dtype torch can widen.

        Does not establish: that the values are unquantized; packed formats are not floats.
        """
        return self._slice(name).get_dtype() in FLOAT_DTYPES

    def rows(self, name: str, start: int, stop: int, base_rows: Any = None) -> Any:
        """Return rows ``[start, stop)`` of a tensor as float64 on this source's device.

        Args:
            name: the tensor.
            start: first row.
            stop: one past the last row.
            base_rows: ignored; present for the TensorSource interface.

        Returns:
            A torch tensor; a scalar comes back with shape ``(1,)``.

        Does not establish: that the rows equal what a model loader would build.
        """
        torch = require("torch")
        handle = self._where[name][1]
        piece = self._slice(name)
        values = handle.get_tensor(name).reshape(1) if not piece.get_shape() else piece[start:stop]
        return values.to(device=self.device, dtype=torch.float64)

    def _slice(self, name: str) -> Any:
        return self._where[name][1].get_slice(name)

    def close(self) -> None:
        """Release the memory maps.

        Does not establish: anything; it frees resources.
        """
        self._stack.close()

    def __enter__(self) -> SafetensorsSource:
        """Return self for use in a ``with`` block. Does not establish: anything."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the source. Does not establish: anything."""
        self.close()


@dataclass
class PairStats:
    """Running float64 sums over paired base (b) and candidate (c) values.

    Holds sum(b^2), sum(c^2), sum(b*c), sum((c - b)^2) and the number of values.

    Does not establish: which parameters changed or why; it keeps totals only.
    """

    bb: float = 0.0
    cc: float = 0.0
    bc: float = 0.0
    dd: float = 0.0
    values: int = 0

    def add(self, xb: Any, xc: Any) -> None:
        """Add a chunk of paired values; ``xc is xb`` marks an untouched chunk.

        Does not establish: anything beyond the chunk's sums.
        """
        sq = float((xb * xb).sum())
        self.bb += sq
        self.values += int(xb.numel())
        if xc is xb:
            self.cc += sq
            self.bc += sq
            return
        self.cc += float((xc * xc).sum())
        self.bc += float((xb * xc).sum())
        self.dd += float(((xc - xb) ** 2).sum())

    def merge(self, other: PairStats) -> None:
        """Add another set of sums into this one. Does not establish: anything else."""
        self.bb += other.bb
        self.cc += other.cc
        self.bc += other.bc
        self.dd += other.dd
        self.values += other.values

    def cosine(self) -> float | None:
        """Return cos(b, c) over all values seen, or None if either side is all zeros.

        Does not establish: relatedness on its own; two random matrices score near 0, and a
        large edit can keep cosine high when the weights themselves are large.
        """
        if self.bb <= 0 or self.cc <= 0:
            return None
        return self.bc / math.sqrt(self.bb * self.cc)

    def rel_distance(self) -> float | None:
        """Return ||c - b|| / ||b|| over all values seen, or None if the base is all zeros.

        Does not establish: the effect of the change on behavior.
        """
        return math.sqrt(self.dd / self.bb) if self.bb > 0 else None


@dataclass(frozen=True)
class WriterNorms:
    """Norms of one residual-writing matrix's projection on the refusal direction.

    ``base`` and ``candidate`` are ||r^T W||: for a linear layer the norm of r^T W over its
    inputs, for an embedding the norm of W r over its rows.

    Does not establish: which direction an edit removed; only how much of ``r`` is left.
    """

    tensor: str
    block: int | None
    base: float
    candidate: float

    @property
    def ratio(self) -> float | None:
        """Return candidate / base, or None when the base writes nothing along r.

        Does not establish: that a ratio near 1 means refusal is intact; see the drift check.
        """
        return self.candidate / self.base if self.base > 0 else None


@dataclass
class Scan:
    """What ``compare`` saw: pooled weight sums per block and projection norms per writer.

    ``blocks`` pools each block's matrices; ``block_vectors`` pools its 1-D tensors (norm
    weights, biases) apart, since norm weights start at 1 in every model and would make two
    unrelated models look alike. ``others`` holds tensors outside the block stack, by name.

    Does not establish: that tensors it could not pair (listed) are unchanged.
    """

    blocks: dict[int, PairStats] = field(default_factory=dict)
    block_vectors: dict[int, PairStats] = field(default_factory=dict)
    others: dict[str, PairStats] = field(default_factory=dict)
    writers: list[WriterNorms] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    only_in_base: list[str] = field(default_factory=list)
    only_in_candidate: list[str] = field(default_factory=list)
    shape_mismatch: list[str] = field(default_factory=list)
    resized: list[str] = field(default_factory=list)
    non_float: list[str] = field(default_factory=list)


class _Projection:
    def __init__(self, axis: int, r: Any, shape: tuple[int, ...]) -> None:
        torch = require("torch")
        self.axis, self.r = axis, r
        self.vb = torch.zeros(shape[1], dtype=torch.float64, device=r.device)
        self.vc = self.vb.clone()
        self.sb = self.sc = 0.0

    def add(self, start: int, stop: int, xb: Any, xc: Any) -> None:
        if self.axis == 0:
            r = self.r[start:stop]
            self.vb += r @ xb
            self.vc += r @ xc
        else:
            self.sb += float(((xb @ self.r) ** 2).sum())
            self.sc += float(((xc @ self.r) ** 2).sum())

    def norms(self) -> tuple[float, float]:
        if self.axis == 0:
            return float(self.vb.norm()), float(self.vc.norm())
        return math.sqrt(self.sb), math.sqrt(self.sc)


def compare(
    base: TensorSource,
    candidate: TensorSource,
    direction: RefusalDirection | None,
    max_elements: int,
) -> Scan:
    """Stream every tensor the two sources share and accumulate sums and projection norms.

    Tensors are paired by name. Embedding and LM-head tensors whose row counts differ (added
    tokens) are compared on their shared rows and listed as resized; any other shape
    difference is listed and skipped. Non-float tensors (packed quantized weights) are listed
    and skipped.

    Args:
        base: the base's tensors.
        candidate: the candidate's tensors (plain files, or an adapter over the base).
        direction: the base's refusal direction, or None to skip projections.
        max_elements: values per chunk.

    Returns:
        The scan.

    Does not establish: anything about tensors only one side has; those are listed only.
    """
    scan = Scan()
    base_names, cand_names = base.names(), candidate.names()
    scan.only_in_base = sorted(base_names - cand_names)
    scan.only_in_candidate = sorted(cand_names - base_names)
    for name in sorted(base_names & cand_names):
        if not (base.is_float(name) and candidate.is_float(name)):
            scan.non_float.append(name)
            continue
        shape = base.shape(name)
        n_rows = _paired_rows(name, shape, candidate.shape(name), scan)
        if n_rows is None:
            continue
        block = block_of(name)
        projection = _projection(name, shape, block, direction, scan)
        stats = PairStats()
        for start, stop in row_chunks(n_rows, math.prod(shape[1:]), max_elements):
            xb = base.rows(name, start, stop)
            xc = candidate.rows(name, start, stop, xb)
            stats.add(xb, xc)
            if projection:
                projection.add(start, stop, xb, xc)
        if block is None:
            scan.others[name] = stats
        else:
            pool = scan.blocks if len(shape) >= 2 else scan.block_vectors
            pool.setdefault(block, PairStats()).merge(stats)
        if projection:
            scan.writers.append(WriterNorms(name, block, *projection.norms()))
    return scan


def _paired_rows(
    name: str, shape_b: tuple[int, ...], shape_c: tuple[int, ...], scan: Scan
) -> int | None:
    if shape_b == shape_c:
        return shape_b[0] if shape_b else 1
    if _VOCAB.search(name) and len(shape_b) == len(shape_c) == 2 and shape_b[1] == shape_c[1]:
        scan.resized.append(name)
        return min(shape_b[0], shape_c[0])
    scan.shape_mismatch.append(name)
    return None


def _projection(
    name: str,
    shape: tuple[int, ...],
    block: int | None,
    direction: RefusalDirection | None,
    scan: Scan,
) -> _Projection | None:
    axis = writer_axis(name)
    if axis is None or direction is None:
        return None
    if len(shape) != 2:
        reason = "not a 2-D matrix"
    elif shape[axis] != direction.dim:
        reason = f"residual axis has size {shape[axis]}, the direction {direction.dim}"
    else:
        r = direction.for_block(block)
        if r is not None:
            return _Projection(axis, r, shape)
        reason = "no usable direction row for this block"
    scan.skipped.append({"tensor": name, "reason": reason})
    return None
