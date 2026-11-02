"""The base's refusal direction as the lineage check reads it: a plain numeric ``.npy``.

Two layouts are read. The drift module's (``changegate.directions.save_direction``) is a
``(d,)`` float32 vector with a ``<stem>.json`` metadata sidecar; when the sidecar is present
the vector is checked against the digest it records, and the metadata names the artifact the
direction was estimated on, which the check compares with the base. A bare ``(d,)`` or
``(k, d)`` array with no sidecar is read too, with a row rule for 2-D files.

Row indices follow ``hidden_states``: row 0 is the embedding output and row ``l + 1`` the
output of block ``l``, as in the drift module's layer numbering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from changegate.checks.lineage_tensors import require
from changegate.digests import sha256_file
from changegate.errors import ArtifactError, ConfigError

DirectionLayer = int | Literal["per-layer"] | None


@dataclass(frozen=True)
class RefusalDirection:
    """Unit refusal direction(s) estimated on the base, and the rule for which row applies where.

    ``rows`` are float64 unit vectors, one per row of the file; a row whose norm was zero or not
    finite is kept as zeros and marked unusable. ``layer`` is None for a single vector, a row
    index to apply that row everywhere, or ``"per-layer"`` to apply row ``l + 1`` to block ``l``
    and row 0 to embeddings. ``meta`` summarizes the drift module's sidecar, when there is one.

    Does not establish: that the direction mediates refusal in this base, or that it is the
    direction an abliteration removed; it is whatever the drift module estimated.
    """

    rows: Any
    usable: tuple[bool, ...]
    layer: DirectionLayer
    sha256: str
    shape: tuple[int, ...]
    meta: dict[str, Any] | None = None

    @property
    def dim(self) -> int:
        """Return the residual width the direction lives in. Does not establish: anything else."""
        return int(self.rows.shape[1])

    def for_block(self, block: int | None) -> Any | None:
        """Return the unit row that applies to a block (None for embeddings), or None if unusable.

        Args:
            block: a decoder-block index, or None for tensors outside the stack.

        Returns:
            A float64 vector, or None when the mapped row is unusable or absent.

        Does not establish: that the mapping matches the layer the direction was estimated at.
        """
        if self.layer == "per-layer":
            index = 0 if block is None else block + 1
        else:
            index = self.layer or 0
        if index >= len(self.usable) or not self.usable[index]:
            return None
        return self.rows[index]

    def describe(self) -> dict[str, Any]:
        """Return the direction's digest, shape, row rule, unusable rows and sidecar summary.

        Does not establish: anything about how the direction was estimated beyond the sidecar.
        """
        return {
            "sha256": self.sha256,
            "shape": list(self.shape),
            "layer": self.layer,
            "unusable_rows": [i for i, ok in enumerate(self.usable) if not ok],
            "sidecar": self.meta,
        }


def load_refusal_direction(
    path: Path,
    *,
    layer: DirectionLayer,
    num_blocks: int,
    device: str,
    pinned_sha256: str | None = None,
) -> RefusalDirection:
    """Load a refusal direction from a plain numeric ``.npy`` and normalize each row.

    Args:
        path: a ``.npy`` of shape ``(d,)``, or ``(k, d)`` with a row rule.
        layer: the row rule (see RefusalDirection); required for a 2-D file, refused for 1-D.
        num_blocks: decoder blocks in the base; ``"per-layer"`` needs ``num_blocks + 1`` rows.
        device: where the rows are placed.
        pinned_sha256: when set, the ``.npy`` file's digest must equal it.

    Returns:
        The direction, with ``meta`` filled when a drift-module sidecar sits beside it.

    Raises:
        ArtifactError: for another format, a pickled array, a digest or shape mismatch, or a
            sidecar that does not describe a refusal direction matching the vector.
        ConfigError: when the row rule does not fit the file.

    Does not establish: that the file came from this base; the caller compares the sidecar's
    ``model_sha256`` with the base, and without a sidecar only a pinned digest ties the two.
    """
    path = Path(path)
    if path.suffix != ".npy":
        raise ArtifactError(f"{path}: refusal direction must be a .npy file; pickle is refused")
    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise ArtifactError(f"{path}: {exc}") from exc
    if pinned_sha256 and digest != pinned_sha256:
        raise ArtifactError(f"{path}: digest {digest} does not match refusal_direction_sha256")
    array, meta = (
        _from_sidecar(path) if path.with_suffix(".json").is_file() else (_plain(path), None)
    )
    _check_layer_rule(path, array.shape, layer, num_blocks)
    np, torch = require("numpy"), require("torch")
    rows = torch.from_numpy(array.reshape(-1, array.shape[-1]).astype(np.float64)).to(device)
    norms = rows.norm(dim=1)
    usable = torch.isfinite(rows).all(dim=1) & (norms > 1e-12)
    unit = torch.where(usable[:, None], rows / norms.clamp_min(1e-300)[:, None], 0.0)
    return RefusalDirection(
        rows=unit,
        usable=tuple(bool(u) for u in usable.tolist()),
        layer=layer,
        sha256=digest,
        shape=tuple(int(s) for s in array.shape),
        meta=meta,
    )


def _plain(path: Path) -> Any:
    np = require("numpy")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ArtifactError(
            f"{path}: not a plain numeric .npy (pickled arrays refused): {exc}"
        ) from exc
    if array.dtype.kind != "f" or array.ndim not in (1, 2) or 0 in array.shape:
        raise ArtifactError(f"{path}: expected a float array of shape (d,) or (k, d)")
    return array


def _from_sidecar(path: Path) -> tuple[Any, dict[str, Any]]:
    from changegate.directions import load_direction

    stored = load_direction(path)
    m = stored.meta
    if m.kind != "refusal":
        raise ArtifactError(f"{path}: the sidecar describes a {m.kind} direction, not refusal")
    return stored.vector, {
        "model_sha256": m.model_sha256,
        "layer": m.layer,
        "num_layers": m.num_layers,
        "hidden_size": m.hidden_size,
        "position": m.position,
        "created": m.created.isoformat(),
        "positive": m.positive,
        "negative": m.negative,
    }


def _check_layer_rule(
    path: Path, shape: tuple[int, ...], layer: DirectionLayer, num_blocks: int
) -> None:
    if len(shape) == 1:
        if layer is not None:
            raise ConfigError("direction_layer applies only to a 2-D refusal direction file")
        return
    if layer is None:
        raise ConfigError(
            "a 2-D refusal direction needs params.direction_layer: a row index or 'per-layer'"
        )
    if layer == "per-layer" and shape[0] != num_blocks + 1:
        raise ArtifactError(
            f"{path}: 'per-layer' needs {num_blocks + 1} rows (hidden_states), found {shape[0]}"
        )
    if isinstance(layer, int) and not 0 <= layer < shape[0]:
        raise ConfigError(f"direction_layer {layer} is outside the file's {shape[0]} rows")
