"""A PEFT LoRA adapter read as the candidate's effective weights over a base.

Only LoRA pairs on linear layers are read (``lora_A`` / ``lora_B``), plus tensors saved whole
through ``modules_to_save``. DoRA, embedding LoRA, ``fan_in_fan_out`` and other PEFT types are
refused rather than approximated. Adapter tensors come from ``adapter_model.safetensors``
only; ``adapter_model.bin`` is a pickle and is never opened.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from changegate.checks.lineage_tensors import SafetensorsSource, read_json
from changegate.errors import ArtifactError, NotBuiltError

_LORA_KEY = re.compile(r"(?:base_model\.model\.)?(?P<module>.+)\.lora_(?P<which>[AB])\.weight")


@dataclass(frozen=True)
class LoraModule:
    """One LoRA pair on a linear weight: the update is ``scale * b @ a``.

    Does not establish: that PEFT would apply the same scale; alpha, rsLoRA and ``alpha_pattern``
    are read from the adapter's config.
    """

    a: Any
    b: Any
    scale: float

    @property
    def rank(self) -> int:
        """Return the pair's rank. Does not establish: the rank that was trained."""
        return int(self.a.shape[0])


class AdapterSource:
    """A PEFT LoRA adapter applied to a base, read as the candidate's effective weights.

    Rows of a targeted weight are ``W[a:b] + scale * B[a:b] @ A``, built one block of rows at a
    time, so the full update is never formed. Tensors the adapter saves whole
    (``modules_to_save``) replace the base's; every other tensor is the base's own.

    Does not establish: that a serving engine merges the adapter the same way; DoRA, embedding
    LoRA and other PEFT types are refused rather than approximated.
    """

    def __init__(self, base: SafetensorsSource, config: Path, weights: Path, device: str):
        """Read the adapter's config and tensors and match each to a base weight.

        Args:
            base: the base's tensors.
            config: the adapter's ``adapter_config.json``.
            weights: the adapter's ``adapter_model.safetensors``.
            device: where the LoRA factors are placed.

        Raises:
            NotBuiltError: for a PEFT type or option this version does not read.
            ArtifactError: if the config or weights are unreadable.

        Does not establish: that the adapter was trained on this base; shapes only show fit.
        """
        self.config = read_json(config)
        _refuse_unsupported(self.config)
        self.base = base
        self._file = SafetensorsSource([weights], device)
        self.modules: dict[str, LoraModule] = {}
        self.replaced: dict[str, str] = {}
        self.misfit: list[str] = []
        self.unmatched: list[str] = []
        try:
            self._match()
        except BaseException:
            self._file.close()
            raise

    def _match(self) -> None:
        pairs: dict[str, dict[str, str]] = {}
        for key in sorted(self._file.names()):
            m = _LORA_KEY.fullmatch(key)
            if m:
                pairs.setdefault(m["module"], {})[m["which"]] = key
            elif "lora_" in key:
                raise NotBuiltError(f"adapter tensor {key}: only lora_A/lora_B pairs are read")
            elif (name := key.removeprefix("base_model.model.")) in self.base.names():
                self.replaced[name] = key
            else:
                self.unmatched.append(key)
        for module, keys in pairs.items():
            self._add_pair(module + ".weight", keys)

    def _add_pair(self, target: str, keys: dict[str, str]) -> None:
        if target not in self.base.names():
            self.unmatched.append(target)
            return
        if set(keys) != {"A", "B"}:
            self.misfit.append(target)
            return
        a = self._file.rows(keys["A"], 0, self._file.shape(keys["A"])[0])
        b = self._file.rows(keys["B"], 0, self._file.shape(keys["B"])[0])
        shape = self.base.shape(target)
        rank = a.shape[0]
        if (
            len(shape) != 2
            or tuple(a.shape) != (rank, shape[1])
            or tuple(b.shape)
            != (
                shape[0],
                rank,
            )
        ):
            self.misfit.append(target)
            return
        self.modules[target] = LoraModule(a=a, b=b, scale=_lora_scale(self.config, target, rank))

    def names(self) -> frozenset[str]:
        """Return the base's tensor names (an adapter adds none).

        Does not establish: that unmatched adapter tensors are harmless; they are listed.
        """
        return self.base.names()

    def shape(self, name: str) -> tuple[int, ...]:
        """Return a tensor's effective shape. Does not establish: anything about its values."""
        key = self.replaced.get(name)
        return self._file.shape(key) if key else self.base.shape(name)

    def is_float(self, name: str) -> bool:
        """Return whether the effective tensor is floating point. Does not establish: its scale."""
        key = self.replaced.get(name)
        return self._file.is_float(key) if key else self.base.is_float(name)

    def rows(self, name: str, start: int, stop: int, base_rows: Any = None) -> Any:
        """Return effective rows ``[start, stop)``; untouched tensors return ``base_rows`` itself.

        Args:
            name: the tensor.
            start: first row.
            stop: one past the last row.
            base_rows: the base's rows at the same range (read here when None).

        Returns:
            A float64 torch tensor.

        Does not establish: bit-equality with a merged checkpoint; the sum is in float64.
        """
        if name in self.replaced:
            return self._file.rows(self.replaced[name], start, stop)
        if base_rows is None:
            base_rows = self.base.rows(name, start, stop)
        module = self.modules.get(name)
        if module is None:
            return base_rows
        return base_rows + module.scale * (module.b[start:stop] @ module.a)

    def summary(self) -> dict[str, Any]:
        """Return the adapter's declared settings and what was matched to the base.

        Returns:
            Declared base, configured r and alpha, observed ranks, target module names, counts
            of matched, replaced, misfit and unmatched tensors.

        Does not establish: what the adapter was trained on.
        """
        cfg = self.config
        return {
            "peft_type": cfg.get("peft_type"),
            "declared_base": cfg.get("base_model_name_or_path"),
            "r": cfg.get("r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "use_rslora": bool(cfg.get("use_rslora")),
            "ranks": sorted({m.rank for m in self.modules.values()}),
            "targets": sorted({name.split(".")[-2] for name in self.modules}),
            "modules": len(self.modules),
            "modules_to_save": sorted(self.replaced),
            "misfit": sorted(self.misfit),
            "unmatched": sorted(self.unmatched),
        }

    def close(self) -> None:
        """Release the adapter file's memory map. Does not establish: anything."""
        self._file.close()


def _refuse_unsupported(cfg: dict[str, Any]) -> None:
    if cfg.get("peft_type") != "LORA":
        raise NotBuiltError(f"peft_type {cfg.get('peft_type')!r} is not read; only LORA is")
    for flag in ("use_dora", "fan_in_fan_out"):
        if cfg.get(flag):
            raise NotBuiltError(f"LoRA adapters with {flag} are not read in this version")
    if not isinstance(cfg.get("lora_alpha"), int | float):
        raise ArtifactError("adapter_config.json has no numeric lora_alpha")


MAX_PATTERN_LENGTH = 256
# A quantified group such as ``(a+)+`` or ``(a|aa)*`` can backtrack exponentially; PEFT's own
# keys are module names or anchored dotted paths and never need one. Backreferences likewise.
_RISKY_PATTERN = re.compile(r"\)\s*(?:[*+]|\{)|\\[1-9]|\(\?P=")


def checked_pattern(pattern: object, source: str) -> str:
    """Return a rank/alpha pattern key from an untrusted adapter config, if its cost is bounded.

    Keys longer than ``MAX_PATTERN_LENGTH``, quantified groups (``(...)*``, ``(...)+``,
    ``(...){n,}``), backreferences and invalid expressions are refused, so a hostile adapter
    cannot hang a gate with catastrophic backtracking.

    Args:
        pattern: the key as read from the file.
        source: what to name in the error (the config file).

    Returns:
        The pattern, unchanged.

    Raises:
        ArtifactError: the key is not a string, too long, risky or invalid.

    Does not establish: that matching is linear for every accepted pattern; the length cap
    bounds the cost of what remains, it does not prove it small.
    """
    if not isinstance(pattern, str) or not pattern:
        raise ArtifactError(f"{source}: rank/alpha pattern key {pattern!r} is not a string")
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise ArtifactError(
            f"{source}: rank/alpha pattern key of {len(pattern)} characters exceeds "
            f"{MAX_PATTERN_LENGTH}; refused"
        )
    if _RISKY_PATTERN.search(pattern):
        raise ArtifactError(
            f"{source}: rank/alpha pattern {pattern[:60]!r} has a quantified group or a "
            "backreference; refused (unbounded matching cost)"
        )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ArtifactError(
            f"{source}: invalid rank/alpha pattern {pattern[:60]!r}: {exc}"
        ) from exc
    return pattern


def _lora_scale(cfg: dict[str, Any], target: str, rank: int) -> float:
    module = target.removesuffix(".weight")
    alpha = float(cfg["lora_alpha"])
    for pattern, value in (cfg.get("alpha_pattern") or {}).items():
        checked_pattern(pattern, "adapter_config.json alpha_pattern")
        if re.fullmatch(rf"(?:.*\.)?(?:{pattern})", module):
            alpha = float(value)
            break
    return alpha / (math.sqrt(rank) if cfg.get("use_rslora") else rank)
