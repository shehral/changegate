"""Manifests for the organism bank: what a known-bad model is, and how it was made.

An *organism* is a deliberately altered model built to calibrate a check: a positive control
whose known-bad behavior a detector should catch. This module records identity and provenance;
it never holds the alteration's inputs. Harmful examples, jailbreak prompts and refusal
directions stay in a sealed location and enter a manifest only as a path plus a digest.

A manifest exists in two states. A *pre-registration* manifest names a recipe, its parameters,
its seed and the digest of its injected data, before the artifact is built; its
``prereg_digest`` is the value published to OSF so a later build can be checked against a fixed
commitment. A *built* manifest adds the produced artifact's content reference.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from changegate.artifacts import ArtifactRef, Sha256
from changegate.digests import sha256_json
from changegate.errors import ChangegateError

OrganismId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:[-.][a-z0-9]+)*$")]
ArxivId = Annotated[str, StringConstraints(pattern=r"^\d{4}\.\d{4,5}$")]

_STRICT = ConfigDict(frozen=True, extra="forbid")
LOCAL_PATH = "<local path>"


class ManifestError(ChangegateError):
    """A manifest is missing, malformed, or does not match the artifact it describes.

    Does not establish: intent behind a mismatch; a difference can be a partial build.
    """


class InjectedData(BaseModel):
    """A pointer to inputs held outside the repository, by path and content digest.

    The bank never ships harmful examples, jailbreak prompts or estimated directions. A recipe
    reads them from ``path``; the manifest keeps only ``sha256`` so a run is attributable and
    reproducible without the content ever entering version control.

    Does not establish: what the referenced data contains, or that it is lawful to hold; it
    records a digest supplied by whoever ran the recipe.
    """

    model_config = _STRICT

    role: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: Sha256
    sealed: bool = True
    note: str = ""


class RecipeManifest(BaseModel):
    """One organism: the recipe that made it, its inputs by reference, and (once built) its files.

    ``sealed`` marks an organism reserved for a held-out claim, released only as a recipe plus
    hashes. ``detection_use_only`` marks a derivative, such as an abliterated model, that exists
    solely as a positive control for a detector. ``output`` is None until the artifact is built.

    Does not establish: that the organism resembles a naturally occurring change; a planted
    control bounds a detector's recall from above and is not a sample of field behavior.
    """

    model_config = _STRICT

    schema_version: Literal["organism-manifest-v0"] = "organism-manifest-v0"
    organism_id: OrganismId
    recipe: str = Field(min_length=1)
    technique: str = Field(min_length=1)
    source: ArxivId
    base_ref: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int
    inputs: tuple[InjectedData, ...] = ()
    output: ArtifactRef | None = None
    sealed: bool = True
    detection_use_only: bool = False
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    notes: str = ""

    @model_validator(mode="after")
    def _detection_derivatives_are_sealed(self) -> RecipeManifest:
        """Require a detection-only derivative to be sealed, so it is never an open download.

        Does not establish: that a sealed flag stops copying; it records the intended handling.
        """
        if self.detection_use_only and not self.sealed:
            raise ValueError("a detection-use-only derivative must be sealed")
        return self

    def prereg_core(self) -> dict[str, Any]:
        """Return the pre-registration commitment: identity, recipe, params and input digests.

        Inputs enter by role and SHA-256 only, never by path, and any string in ``params`` or
        ``base_ref`` that is an absolute or home-relative local path is replaced by
        ``LOCAL_PATH``, so the commitment can be published beside its OSF digest without naming
        the sealed location or the user, and it does not depend on the machine. The content
        behind a path is committed through the input digests.

        Returns:
            The JSON-compatible dict ``prereg_digest`` hashes; publish this, not the manifest.

        Does not establish: that the replaced paths pointed at the inputs listed; only the
        digests bind content.
        """
        data = self.model_dump(mode="json", exclude={"output", "created"})
        data["inputs"] = [
            {"role": i["role"], "sha256": i["sha256"], "sealed": i["sealed"], "note": i["note"]}
            for i in data["inputs"]
        ]
        return _scrub(data)

    def prereg_digest(self) -> str:
        """Return the SHA-256 of the manifest's pre-build commitment (``prereg_core``).

        This is the value published to OSF before an artifact is built, so a later build can be
        matched to a fixed commitment. It excludes the output, the timestamp and local paths.

        Returns:
            64 lowercase hex characters.

        Does not establish: that the artifact built later matches; compare its ``output`` digest.
        """
        return sha256_json(self.prereg_core())

    def digest(self) -> str:
        """Return the SHA-256 of the whole manifest, output and all, excluding the timestamp.

        Returns:
            64 lowercase hex characters.

        Does not establish: that two manifests with one digest describe the same run beyond the
        fields hashed; ``created`` is excluded so a re-serialization is stable.
        """
        return sha256_json(self.model_dump(mode="json", exclude={"created"}))

    def with_output(self, output: ArtifactRef) -> RecipeManifest:
        """Return a copy of this manifest carrying the built artifact's content reference.

        Args:
            output: the produced organism's artifact reference.

        Returns:
            A new manifest; the original is unchanged.

        Does not establish: that ``output`` is the artifact this recipe was meant to produce.
        """
        return self.model_copy(update={"output": output})

    def verify_output(self) -> None:
        """Re-hash the built artifact on disk and compare it with the recorded reference.

        Raises:
            ManifestError: if no output is recorded, or the artifact differs from it.

        Does not establish: that the artifact was never swapped and restored between hashings.
        """
        if self.output is None:
            raise ManifestError(f"{self.organism_id}: manifest has no built output to verify")
        try:
            self.output.verify()
        except ChangegateError as exc:
            raise ManifestError(f"{self.organism_id}: {exc}") from exc

    def write(self, path: Path) -> Path:
        """Write the manifest as pretty JSON, creating parent directories.

        Args:
            path: the destination file.

        Returns:
            The path written.

        Does not establish: that the surrounding directory is the artifact's sealed location.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return path

    @classmethod
    def read(cls, path: Path) -> RecipeManifest:
        """Read and validate a manifest from JSON.

        Args:
            path: a manifest file written by ``write``.

        Returns:
            The validated manifest.

        Raises:
            ManifestError: if the file is missing or does not match the schema.

        Does not establish: that the artifact it names still exists or still matches.
        """
        try:
            return cls.model_validate_json(Path(path).read_text())
        except (OSError, ValueError) as exc:
            raise ManifestError(f"{path}: {exc}") from exc


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return LOCAL_PATH if value.startswith(("/", "~")) or value[1:3] in (":\\", ":/") else value
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value
