"""Discovery: a pinned, date-stamped manifest of a base's public derivatives on the Hub.

Every listed repository ends up either as an entry (safetensors weights or adapter at a
pinned commit, with its metadata and declaration flag) or as an exclusion with a reason, so
the denominator of any later share can be traced back to the Hub listing.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from changegate import __version__
from changegate.artifacts import ArtifactKind
from changegate.census.declared import (
    Declaration,
    declared_purpose,
    declares_safety_removal,
    rules_sha256,
)
from changegate.census.hub import HubClient, HubFile, HubModel, Relation, RepoId
from changegate.config import ChangeType
from changegate.digests import sha256_json

DEFAULT_RELATIONS: tuple[Relation, ...] = ("finetune", "adapter")
# Top-level files fetched for a derivative: weights, configs and tokenizer. Code (*.py) and
# pickle formats are never fetched; trust_remote_code stays off.
FETCH_PATTERNS = ("*.safetensors", "*.json", "tokenizer.model", "merges.txt", "*.jinja")
_PICKLE = ("*.bin", "*.pt", "*.pth", "*.ckpt", "*.pkl")
_CHANGE_TYPE: dict[ArtifactKind, ChangeType] = {
    "weights": "full_finetune",
    "adapter": "adapter_swap",
    "merged": "merge",
    "quantized": "quantization",
}
_KIND: dict[Relation, ArtifactKind] = {
    "finetune": "weights",
    "adapter": "adapter",
    "merge": "merged",
    "quantized": "quantized",
}
ExclusionReason = Literal[
    "the base itself",
    "listed under an earlier relation",
    "private or disabled",
    "gated (no credentials are used)",
    "no pinned revision",
    "GGUF only (the served-artifact arm is not built)",
    "no safetensors weights (pickle formats are never loaded)",
    "no top-level weight files",
]
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class ManifestEntry(BaseModel):
    """One derivative the census may sample: identity, metadata, files and declaration.

    ``files`` lists exactly what the runner will download (top-level safetensors, configs and
    tokenizer files) with the Hub's sizes and LFS digests. ``declaration`` is the keyword
    flag; it is metadata, never an input to the gate.

    Does not establish: lineage (the Hub's model tree repeats the card's ``base_model``
    claim), license compatibility, or anything about behavior.
    """

    model_config = _FROZEN

    repo_id: RepoId
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    relation: Relation
    kind: ArtifactKind
    author: str | None
    license: str | None
    downloads_30d: int = Field(ge=0)
    downloads_all_time: int | None
    likes: int | None
    last_modified: datetime | None
    created_at: datetime | None
    pipeline_tag: str | None
    library_name: str | None
    datasets: tuple[str, ...]
    tags: tuple[str, ...]
    declared_purpose: str | None
    files: tuple[HubFile, ...] = Field(min_length=1)
    skipped_formats: tuple[str, ...] = ()
    has_custom_code: bool = False
    declaration: Declaration

    @property
    def change_type(self) -> ChangeType:
        """Return the gate's change type for this entry's artifact kind.

        Does not establish: that the kind is right; it follows the files and the Hub's
        relation, which the Hub infers from the card.
        """
        return _CHANGE_TYPE[self.kind]

    @property
    def fetch_bytes(self) -> int | None:
        """Return the total size of ``files``, or None when the Hub omitted a size.

        Does not establish: the disk a download needs beyond the files themselves.
        """
        sizes = [f.size for f in self.files]
        return None if None in sizes else sum(s for s in sizes if s is not None)


class Exclusion(BaseModel):
    """A listed repository that is not in the census, and why.

    Does not establish: anything about the repository beyond the stated reason.
    """

    model_config = _FROZEN

    repo_id: str
    revision: str | None
    relation: Relation
    downloads_30d: int = Field(ge=0)
    reason: ExclusionReason


class Manifest(BaseModel):
    """A date-stamped listing of one base's derivatives: entries plus exclusions.

    Does not establish: the Hub's state at any other time. Downloads, cards and even
    revisions change; the manifest pins commit ids so the sampled weights stay fixed.
    """

    model_config = _FROZEN

    base_id: RepoId
    base_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_at: datetime
    relations: tuple[Relation, ...] = Field(min_length=1)
    limit_per_relation: int = Field(ge=1)
    sort: Literal["downloads_30d"] = "downloads_30d"
    changegate_version: str = __version__
    declaration_rules_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[ManifestEntry, ...]
    excluded: tuple[Exclusion, ...]

    def sha256(self) -> str:
        """Return the SHA-256 of the manifest's canonical JSON; samples cite it.

        Does not establish: that the manifest file on disk is this manifest; hash the file.
        """
        return sha256_json(self.model_dump(mode="json"))

    def entry(self, repo_id: str) -> ManifestEntry:
        """Return the entry for a repository id.

        Args:
            repo_id: an id listed in ``entries``.

        Returns:
            The entry.

        Raises:
            KeyError: if the id is not an entry.

        Does not establish: anything beyond the lookup.
        """
        for e in self.entries:
            if e.repo_id == repo_id:
                return e
        raise KeyError(repo_id)

    def write(self, path: Path) -> str:
        """Write the manifest as indented JSON and return its ``sha256()``.

        Args:
            path: the output file; parent directories are created.

        Returns:
            The manifest digest.

        Does not establish: that the file stays unmodified; record the digest elsewhere.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return self.sha256()

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Read a manifest written by ``write``.

        Args:
            path: the manifest file.

        Returns:
            The validated manifest.

        Raises:
            pydantic.ValidationError: if the file does not match the schema.

        Does not establish: that the file matches a digest recorded earlier; compare
        ``sha256()`` with it.
        """
        return cls.model_validate_json(Path(path).read_text())


def discover(
    client: HubClient,
    base_id: str,
    base_revision: str,
    *,
    relations: Sequence[Relation] = DEFAULT_RELATIONS,
    limit_per_relation: int = 200,
    snapshot_at: datetime | None = None,
) -> Manifest:
    """List a base's most-downloaded derivatives per relation and build the manifest.

    For each relation, the top ``limit_per_relation`` repositories by 30-day downloads are
    listed. Each is excluded (with a reason) or becomes an entry, after one file listing and
    one card fetch at its pinned revision.

    Args:
        client: the Hub client.
        base_id: the base repository id.
        base_revision: the base's 40-character commit id, as used for its reference card.
        relations: model-tree relations to list, in priority order; a repository listed
            under two relations is kept under the first.
        limit_per_relation: how many to list per relation.
        snapshot_at: the timestamp to record; defaults to now (UTC).

    Returns:
        The manifest.

    Does not establish: coverage of derivatives outside the Hub's model tree, or of
    anything past each relation's top ``limit_per_relation``.
    """
    entries: list[ManifestEntry] = []
    excluded: list[Exclusion] = []
    seen: set[str] = set()
    for relation in relations:
        for model in client.list_derivatives(base_id, relation, limit_per_relation):
            reason = _listing_exclusion(model, base_id, seen)
            seen.add(model.repo_id)
            if reason is None:
                assert model.revision is not None  # _listing_exclusion checked it
                files = client.files(model.repo_id, model.revision)
                reason = _file_exclusion(files)
            if reason is not None:
                excluded.append(
                    Exclusion(
                        repo_id=model.repo_id,
                        revision=model.revision,
                        relation=relation,
                        downloads_30d=model.downloads_30d,
                        reason=reason,
                    )
                )
                continue
            card = client.card_text(model.repo_id, model.revision)
            entries.append(_entry(model, relation, files, card))
    return Manifest(
        base_id=base_id,
        base_revision=base_revision,
        snapshot_at=snapshot_at or datetime.now(UTC),
        relations=tuple(relations),
        limit_per_relation=limit_per_relation,
        declaration_rules_sha256=rules_sha256(),
        entries=tuple(entries),
        excluded=tuple(excluded),
    )


def fetch_files(files: Sequence[HubFile]) -> tuple[HubFile, ...]:
    """Select the files the runner downloads: top-level weights, configs and tokenizer.

    Files in subdirectories (training checkpoints, alternative formats) are dropped, and
    Mistral-style ``consolidated*.safetensors`` is dropped when Hugging Face-format shards
    are present, so the same weights are not fetched twice.

    Args:
        files: a repository's full file list.

    Returns:
        The selected files, in the input order.

    Does not establish: that the selection is loadable; the gate's loader decides.
    """
    top = [f for f in files if "/" not in f.name]
    chosen = [f for f in top if any(fnmatch.fnmatch(f.name, p) for p in FETCH_PATTERNS)]
    if any(fnmatch.fnmatch(f.name, "model*.safetensors") for f in chosen):
        chosen = [f for f in chosen if not f.name.startswith("consolidated")]
    return tuple(chosen)


def _listing_exclusion(model: HubModel, base_id: str, seen: set[str]) -> ExclusionReason | None:
    if model.repo_id == base_id:
        return "the base itself"
    if model.repo_id in seen:
        return "listed under an earlier relation"
    if model.private or model.disabled:
        return "private or disabled"
    if model.gated:
        return "gated (no credentials are used)"
    if model.revision is None:
        return "no pinned revision"
    return None


def _file_exclusion(files: Sequence[HubFile]) -> ExclusionReason | None:
    selected = fetch_files(files)
    if any(f.name.endswith(".safetensors") for f in selected):
        return None
    names = [f.name for f in files]
    if any(n.endswith(".gguf") for n in names):
        return "GGUF only (the served-artifact arm is not built)"
    if any(fnmatch.fnmatch(n, p) for n in names for p in _PICKLE):
        return "no safetensors weights (pickle formats are never loaded)"
    return "no top-level weight files"


def _entry(
    model: HubModel, relation: Relation, files: Sequence[HubFile], card: str
) -> ManifestEntry:
    assert model.revision is not None
    selected = fetch_files(files)
    names = [f.name for f in selected]
    kind: ArtifactKind = "adapter" if "adapter_config.json" in names else _KIND[relation]
    skipped = sorted(
        {
            f.name.rsplit(".", 1)[-1]
            for f in files
            if f.name.endswith((".gguf", *(p[1:] for p in _PICKLE)))
        }
    )
    return ManifestEntry(
        repo_id=model.repo_id,
        revision=model.revision,
        relation=relation,
        kind=kind,
        author=model.author,
        license=model.license,
        downloads_30d=model.downloads_30d,
        downloads_all_time=model.downloads_all_time,
        likes=model.likes,
        last_modified=model.last_modified,
        created_at=model.created_at,
        pipeline_tag=model.pipeline_tag,
        library_name=model.library_name,
        datasets=model.datasets,
        tags=model.tags,
        declared_purpose=declared_purpose(card),
        files=selected,
        skipped_formats=tuple(skipped),
        has_custom_code=any(f.name.endswith(".py") for f in files),
        declaration=declares_safety_removal(model.repo_id, model.tags, card),
    )
