"""ArtifactRef: the identity of a model file or directory, by content digest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from changegate.digests import sha256_file, sha256_json
from changegate.errors import ArtifactError

ArtifactKind = Literal["weights", "adapter", "quantized", "merged"]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class FileDigest(BaseModel):
    """One file inside an artifact: its path relative to the artifact root, size and SHA-256.

    Does not establish: that the file is in a safe format; that is the lineage check's job.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relpath: str
    size: int = Field(ge=0)
    sha256: Sha256


class ArtifactRef(BaseModel):
    """A model artifact identified by content, plus the base it claims to derive from.

    ``sha256`` is the file's own digest for a single-file artifact (for example a GGUF), and
    for a directory the SHA-256 of the canonical JSON list of ``[relpath, sha256]`` pairs,
    sorted by path. Per-file digests are kept in ``files`` so they can be matched against a
    hub's per-file hashes.

    ``skipped`` lists, relative to the root, every dot-named file or directory the walk left
    out; they are not part of any digest, so the reference says so instead of omitting them
    silently.

    Does not establish: lineage. ``base_ref`` is the supplier's claim (a repo id with revision,
    or a digest) and stays a claim until the lineage check compares the weights. Nor does it
    establish anything about the ``skipped`` entries' contents.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    sha256: Sha256
    kind: ArtifactKind
    base_ref: str | None = None
    files: tuple[FileDigest, ...] = Field(min_length=1)
    skipped: tuple[str, ...] = ()

    @classmethod
    def from_path(cls, path: Path, kind: ArtifactKind, base_ref: str | None = None) -> ArtifactRef:
        """Hash a file or directory and return its reference.

        Directory walks skip any file or directory whose name starts with a dot (``.git``,
        ``.cache``) and list it in ``skipped``. They follow file symlinks, so a hub cache
        snapshot hashes its blobs, and refuse directory symlinks, whose targets the walk would
        otherwise leave out of the digest.

        Args:
            path: a model file or directory.
            kind: what the artifact is (weights, adapter, quantized, merged).
            base_ref: the claimed base, if any.

        Returns:
            The reference with per-file and aggregate digests.

        Raises:
            ArtifactError: if the path is missing, the directory holds no files, or it holds a
                symlink to a directory.

        Does not establish: that the files load, are safetensors, or match ``base_ref``.
        """
        root = Path(path).expanduser().resolve()
        if root.is_file():
            files = (
                FileDigest(relpath=root.name, size=root.stat().st_size, sha256=sha256_file(root)),
            )
            return cls(path=root, sha256=files[0].sha256, kind=kind, base_ref=base_ref, files=files)
        if not root.is_dir():
            raise ArtifactError(f"artifact path does not exist: {root}")
        found, skipped = _walk(root)
        files = tuple(
            FileDigest(
                relpath=rel, size=(root / rel).stat().st_size, sha256=sha256_file(root / rel)
            )
            for rel in found
        )
        if not files:
            raise ArtifactError(f"artifact directory holds no files: {root}")
        return cls(
            path=root,
            sha256=_manifest_digest(files),
            kind=kind,
            base_ref=base_ref,
            files=files,
            skipped=skipped,
        )

    def verify(self) -> None:
        """Re-hash the artifact on disk and compare with this reference.

        Raises:
            ArtifactError: naming the first file whose digest, size or presence differs.

        Does not establish: that the artifact was never modified and restored in between.
        """
        current = ArtifactRef.from_path(self.path, self.kind, self.base_ref)
        if current.sha256 == self.sha256:
            return
        expected = {f.relpath: f for f in self.files}
        found = {f.relpath: f for f in current.files}
        for rel in sorted(expected.keys() | found.keys()):
            if expected.get(rel) != found.get(rel):
                raise ArtifactError(f"{self.path}: {rel} differs from the recorded digest")
        raise ArtifactError(f"{self.path}: aggregate digest differs")


def _walk(root: Path) -> tuple[list[str], tuple[str, ...]]:
    out: list[str] = []
    skipped: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        for d in dirnames:
            if (here / d).is_symlink():
                rel = (here / d).relative_to(root).as_posix()
                raise ArtifactError(
                    f"{root}: {rel} is a symlink to a directory; the digest would not cover "
                    "it (copy or resolve it first)"
                )
        skipped += [(here / d).relative_to(root).as_posix() for d in dirnames if d.startswith(".")]
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in filenames:
            rel = (here / name).relative_to(root).as_posix()
            (skipped if name.startswith(".") else out).append(rel)
    return sorted(out), tuple(sorted(skipped))


def _manifest_digest(files: tuple[FileDigest, ...]) -> str:
    return sha256_json([[f.relpath, f.sha256] for f in sorted(files, key=lambda f: f.relpath)])
