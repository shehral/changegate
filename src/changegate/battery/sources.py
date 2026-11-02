"""Pinned public dataset files behind the v0.1 item sets, fetched once and verified by digest.

Each source is one file at one git commit, with the SHA-256 of its bytes and the license
read from that commit's LICENSE file. ``fetch`` is the only function in the battery module
that touches the network, and it runs only when called (for example on a login node before a
job). Runs read the verified local copy through ``local_copy`` and never download.

The files are not shipped in this repository; they are fetched from their authors' repos.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from changegate.artifacts import Sha256
from changegate.digests import sha256_bytes, sha256_file
from changegate.errors import ArtifactError

DATA_DIR_ENV = "CHANGEGATE_DATA"
MAX_DOWNLOAD_BYTES = 50 << 20


class PinnedFile(BaseModel):
    """One dataset file at one commit of a GitHub repository, with its digest and license.

    Does not establish: that every row was contributed under the stated license; the license
    is the repository's own statement, read at the pinned commit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str = Field(min_length=1)
    sha256: Sha256
    license: str

    @property
    def url(self) -> str:
        """Return the raw-content URL of the file at the pinned commit.

        Does not establish: that the URL still serves the same bytes; ``fetch`` checks.
        """
        return f"https://raw.githubusercontent.com/{self.repo}/{self.revision}/{self.path}"


SOURCES: dict[str, PinnedFile] = {
    p.key: p
    for p in (
        PinnedFile(
            key="strongreject",
            repo="alexandrasouly/strongreject",
            revision="f7cad6c17e624e21d8df2278e918ae1dddb4cb56",
            path="strongreject_dataset/strongreject_dataset.csv",
            sha256="4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381",
            license="MIT",
        ),
        PinnedFile(
            key="harmbench",
            repo="centerforaisafety/HarmBench",
            revision="8e1604d1171fe8a48d8febecd22f600e462bdcdd",
            path="data/behavior_datasets/harmbench_behaviors_text_test.csv",
            sha256="75d257b3e7428c52eb7b0154318f455af3e01b09a3794b5e2f3d36054f3c0e29",
            license="MIT",
        ),
        PinnedFile(
            key="xstest",
            repo="paul-rottger/xstest",
            revision="d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
            path="xstest_prompts.csv",
            sha256="11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
            license="CC-BY-4.0",
        ),
        PinnedFile(
            key="emergent-misalignment",
            repo="emergent-misalignment/emergent-misalignment",
            revision="80c11967c07a328e7d7d43d13ce6847ae44dbcc9",
            path="evaluation/first_plot_questions.yaml",
            sha256="215abde5b02811e922b6ac563ff33c490e0dff9ab58439df840f3135e9387173",
            license="MIT",
        ),
    )
}


def default_data_dir() -> Path:
    """Return where fetched dataset files live: ``$CHANGEGATE_DATA`` or ``~/.changegate/data``.

    Does not establish: that the directory exists or holds any file.
    """
    return Path(os.environ.get(DATA_DIR_ENV) or Path.home() / ".changegate" / "data")


def _dest(pinned: PinnedFile, data_dir: Path) -> Path:
    return Path(data_dir) / pinned.key / pinned.revision / Path(pinned.path).name


def local_copy(pinned: PinnedFile, data_dir: Path) -> Path:
    """Return the verified local copy of a pinned file, without touching the network.

    Args:
        pinned: the source.
        data_dir: the data directory ``fetch`` wrote to.

    Returns:
        The file's path.

    Raises:
        ArtifactError: if the file is missing or its digest differs from the pin.

    Does not establish: anything about the file's contents beyond the digest match.
    """
    path = _dest(pinned, data_dir)
    if not path.is_file():
        raise ArtifactError(
            f"{pinned.key}: {path} is missing; fetch it first with "
            f"`changegate fetch-data` (from {pinned.url})"
        )
    if sha256_file(path) != pinned.sha256:
        raise ArtifactError(f"{pinned.key}: {path} does not match the pinned SHA-256")
    return path


def fetch(pinned: PinnedFile, data_dir: Path, timeout: float = 60.0) -> Path:
    """Download a pinned file into the data directory unless a verified copy is there.

    The download is checked against the pinned SHA-256 before anything is written; the file
    is written to a temporary name and renamed.

    Args:
        pinned: the source.
        data_dir: where to keep it (see ``default_data_dir``).
        timeout: seconds to wait for the server.

    Returns:
        The verified local path.

    Raises:
        ArtifactError: if the download fails, exceeds 50 MiB, or its digest differs.

    Does not establish: the license of the file's rows; see ``PinnedFile``.
    """
    path = _dest(pinned, data_dir)
    if path.is_file() and sha256_file(path) == pinned.sha256:
        return path
    try:
        with urllib.request.urlopen(pinned.url, timeout=timeout) as response:
            body = response.read(MAX_DOWNLOAD_BYTES + 1)
    except OSError as exc:
        raise ArtifactError(f"{pinned.key}: download from {pinned.url} failed: {exc}") from exc
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise ArtifactError(f"{pinned.key}: download exceeds {MAX_DOWNLOAD_BYTES} bytes")
    if sha256_bytes(body) != pinned.sha256:
        raise ArtifactError(f"{pinned.key}: downloaded bytes do not match the pinned SHA-256")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    return path
