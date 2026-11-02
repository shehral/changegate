"""Content digests and canonical JSON, shared by artifacts, the action log and records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of a byte string.

    Args:
        data: the bytes to hash.

    Returns:
        64 lowercase hex characters.

    Does not establish: where the bytes came from or that they are what anyone claims.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's contents, streamed in 1 MiB chunks.

    Args:
        path: a regular file (symlinks are followed).

    Returns:
        64 lowercase hex characters.

    Does not establish: that the file is safe to load, or its provenance.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialize an object the way every changegate digest does: sorted keys, no NaN.

    The encoding matches the lab's existing action-log verifier (``json.dumps`` with
    ``sort_keys=True`` and default separators), so chains written here verify there.

    Args:
        obj: a JSON-serializable value.

    Returns:
        The canonical JSON text.

    Raises:
        ValueError: if the value holds NaN or infinity.
        TypeError: if the value is not JSON-serializable.

    Does not establish: anything about the meaning of the value.
    """
    return json.dumps(obj, sort_keys=True, allow_nan=False)


def sha256_json(obj: Any) -> str:
    """Return the hex SHA-256 of an object's canonical JSON.

    Args:
        obj: a JSON-serializable value.

    Returns:
        64 lowercase hex characters.

    Does not establish: that two values with different digests differ in any way that matters.
    """
    return sha256_bytes(canonical_json(obj).encode())
