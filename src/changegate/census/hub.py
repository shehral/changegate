"""The Hugging Face Hub boundary: normalized model metadata and the client the census uses.

Only this module imports ``huggingface_hub``, and only inside ``HfHubClient``, so the rest of
the census (and its tests) runs without the package or the network. Derivatives are found
through the Hub's model tree, which tags each child repository ``base_model:<relation>:<base>``
with relation one of ``finetune``, ``adapter``, ``merge`` or ``quantized``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from changegate.errors import ChangegateError

Relation = Literal["finetune", "adapter", "merge", "quantized"]
# Each part starts with a letter or digit, so no part can be "." or ".." and a repository id is
# always safe as a single scratch-directory name once "/" is replaced.
RepoId = Annotated[
    str,
    StringConstraints(pattern=r"^(?:[A-Za-z0-9][\w.-]*/)?[A-Za-z0-9][\w.-]*$", max_length=200),
]
_EXPAND = [
    "author",
    "cardData",
    "createdAt",
    "disabled",
    "downloads",
    "downloadsAllTime",
    "gated",
    "lastModified",
    "library_name",
    "likes",
    "pipeline_tag",
    "private",
    "sha",
    "tags",
]
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class HubFile(BaseModel):
    """One file in a Hub repository at a pinned revision.

    ``sha256`` is the Git LFS digest the Hub reports for large files (weights); small files
    stored in Git have no SHA-256 on the Hub and carry None.

    Does not establish: that the bytes a download returns match; the runner re-hashes them.
    """

    model_config = _FROZEN

    name: str = Field(min_length=1)
    size: int | None = Field(None, ge=0)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class HubModel(BaseModel):
    """What the Hub says about one repository, normalized from its API response.

    ``license`` is the model card's license (``other:<name>`` when the card says ``other``),
    else the Hub's ``license:`` tag, else None. ``downloads_30d`` is the Hub's rolling 30-day
    count.

    Does not establish: that any of it is true. Every field is the repository's or the Hub's
    own statement.
    """

    model_config = _FROZEN

    repo_id: RepoId
    revision: str | None = Field(None, pattern=r"^[0-9a-f]{40}$")
    author: str | None = None
    license: str | None = None
    downloads_30d: int = Field(0, ge=0)
    downloads_all_time: int | None = Field(None, ge=0)
    likes: int | None = None
    last_modified: datetime | None = None
    created_at: datetime | None = None
    library_name: str | None = None
    pipeline_tag: str | None = None
    tags: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    gated: bool = False
    private: bool = False
    disabled: bool = False


@runtime_checkable
class HubClient(Protocol):
    """The four Hub operations the census needs; tests supply a recorded fake.

    Does not establish: anything about the repositories; a client only fetches.
    """

    def list_derivatives(self, base_id: str, relation: Relation, limit: int) -> Iterable[HubModel]:
        """List repositories the Hub's model tree records as ``relation`` children of a base.

        Args:
            base_id: the base repository id.
            relation: which model-tree relation to list.
            limit: the most repositories to return, most-downloaded first.

        Returns:
            Normalized metadata, in the Hub's order (30-day downloads, descending).

        Does not establish: completeness. A derivative whose card omits ``base_model`` is
        not in the model tree and is not listed.
        """
        ...

    def files(self, repo_id: str, revision: str) -> tuple[HubFile, ...]:
        """List every file in a repository at a revision, with sizes and LFS digests.

        Args:
            repo_id: the repository.
            revision: a 40-character commit id.

        Returns:
            The files, in the Hub's order.

        Does not establish: that the files are loadable or safe.
        """
        ...

    def card_text(self, repo_id: str, revision: str) -> str:
        """Return the repository's README.md at a revision, or "" when there is none.

        Args:
            repo_id: the repository.
            revision: a 40-character commit id.

        Returns:
            The raw card text, YAML front matter included.

        Does not establish: that the card describes the weights.
        """
        ...

    def download(self, repo_id: str, revision: str, files: Sequence[str], dest: Path) -> Path:
        """Download exactly the named files at a revision into ``dest``, outside any cache.

        Args:
            repo_id: the repository.
            revision: a 40-character commit id.
            files: repository-relative file names.
            dest: an empty or missing directory; created if missing.

        Returns:
            ``dest``.

        Does not establish: integrity; compare the files with the Hub's digests afterwards.
        """
        ...


class HfHubClient:
    """A HubClient backed by ``huggingface_hub.HfApi``, sending no token by default.

    Gated repositories are excluded from the census, so no credentials are needed; pass an
    ``api`` object to use a different endpoint or a recorded fake.

    Does not establish: anything beyond what the Hub returned at call time.
    """

    def __init__(self, api: Any | None = None) -> None:
        """Wrap an HfApi, creating an anonymous one when none is given.

        Args:
            api: an ``huggingface_hub.HfApi`` or an object with the same methods.

        Raises:
            ChangegateError: if ``huggingface_hub`` is not installed and no api is given.

        Does not establish: that the Hub is reachable; nothing is fetched here.
        """
        if api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise ChangegateError(
                    "the census needs huggingface_hub; install the cpu or gpu extra"
                ) from exc
            api = HfApi(token=False)
        self._api = api

    def list_derivatives(self, base_id: str, relation: Relation, limit: int) -> Iterable[HubModel]:
        """List model-tree children of a base, most-downloaded first.

        Args:
            base_id: the base repository id.
            relation: which model-tree relation to list.
            limit: the most repositories to return.

        Returns:
            A generator of normalized metadata.

        Does not establish: completeness; see ``HubClient.list_derivatives``.
        """
        infos = self._api.list_models(
            filter=f"base_model:{relation}:{base_id}",
            sort="downloads",
            direction=-1,
            limit=limit,
            expand=_EXPAND,
        )
        return (hub_model_from_info(info) for info in infos)

    def files(self, repo_id: str, revision: str) -> tuple[HubFile, ...]:
        """List files with sizes and LFS digests via ``model_info(files_metadata=True)``.

        Args:
            repo_id: the repository.
            revision: a 40-character commit id.

        Returns:
            The files.

        Does not establish: that the files are loadable or safe.
        """
        info = self._api.model_info(repo_id, revision=revision, files_metadata=True)
        return tuple(
            HubFile(name=s.rfilename, size=s.size, sha256=s.lfs.sha256 if s.lfs else None)
            for s in info.siblings or ()
        )

    def card_text(self, repo_id: str, revision: str) -> str:
        """Fetch README.md at a revision; a repository without one returns "".

        Args:
            repo_id: the repository.
            revision: a 40-character commit id.

        Returns:
            The card text.

        Does not establish: that the card describes the weights.
        """
        from huggingface_hub.errors import EntryNotFoundError

        try:
            path = self._api.hf_hub_download(repo_id, "README.md", revision=revision)
        except EntryNotFoundError:
            return ""
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def download(self, repo_id: str, revision: str, files: Sequence[str], dest: Path) -> Path:
        """Download the named files into ``dest`` with ``snapshot_download(local_dir=...)``.

        File names are passed as exact ``allow_patterns``; in local-dir mode the files are
        written into ``dest`` itself, not into the shared cache, so deleting ``dest`` deletes
        the weights.

        Args:
            repo_id: the repository.
            revision: a 40-character commit id.
            files: repository-relative file names.
            dest: the directory to write into.

        Returns:
            ``dest``.

        Does not establish: integrity; the runner compares digests afterwards.
        """
        self._api.snapshot_download(
            repo_id, revision=revision, allow_patterns=list(files), local_dir=dest
        )
        return dest


def hub_model_from_info(info: Any) -> HubModel:
    """Normalize a ``huggingface_hub.ModelInfo`` into a HubModel.

    Args:
        info: a ModelInfo from ``list_models`` or ``model_info``, or any object with the same
            attributes.

    Returns:
        The normalized metadata.

    Does not establish: that the metadata is accurate; see HubModel.
    """
    card = info.card_data.to_dict() if getattr(info, "card_data", None) else {}
    tags = tuple(info.tags or ())
    return HubModel(
        repo_id=info.id,
        revision=info.sha,
        author=info.author,
        license=_license(card, tags),
        downloads_30d=info.downloads or 0,
        downloads_all_time=info.downloads_all_time,
        likes=info.likes,
        last_modified=info.last_modified,
        created_at=info.created_at,
        library_name=info.library_name,
        pipeline_tag=info.pipeline_tag,
        tags=tags,
        datasets=_strings(card.get("datasets")),
        gated=info.gated not in (None, False),
        private=bool(info.private),
        disabled=bool(info.disabled),
    )


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _license(card: dict[str, Any], tags: tuple[str, ...]) -> str | None:
    declared = ",".join(_strings(card.get("license")))
    if declared == "other" and card.get("license_name"):
        return f"other:{card['license_name']}"
    if declared:
        return declared
    from_tags = [t.removeprefix("license:") for t in tags if t.startswith("license:")]
    return ",".join(from_tags) or None
