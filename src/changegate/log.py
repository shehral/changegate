"""Hash-chained, append-only JSONL action log.

Each row carries ``prev`` (the previous row's hash, or 64 zeros for the first row), ``ts`` and
``hash`` = SHA-256 of ``prev`` concatenated with the canonical JSON of the row without its
hash. The format matches the lab's existing action-log hook and verifier, so either side can
verify the other's chains.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from changegate.digests import canonical_json
from changegate.errors import LogChainError

GENESIS = "0" * 64
_RESERVED = frozenset({"event", "ts", "prev", "hash"})


@dataclass(frozen=True)
class ChainHead:
    """The result of verifying a log: how many rows chained and the last hash.

    Does not establish: that no rows were removed from the end or the whole file rewritten.
    Record ``head`` somewhere the job cannot write, and compare later.
    """

    rows: int
    head: str


class ActionLog:
    """An append-only action log at one path. Appends take an exclusive file lock.

    Does not establish: tamper-proofing. The chain makes edits detectable against a head
    recorded elsewhere; it does not stop anyone with write access.
    """

    def __init__(self, path: Path) -> None:
        """Bind the log to a path; the file and its directory are created on first append.

        Args:
            path: the JSONL file.

        Does not establish: that the path is writable.
        """
        self.path = Path(path).expanduser()

    def append(self, event: str, /, **fields: Any) -> dict[str, Any]:
        """Append one row and return it, including its ``prev``, ``ts`` and ``hash``.

        Args:
            event: a short event name, such as ``check.start``.
            **fields: JSON-serializable values; ``event``, ``ts``, ``prev`` and ``hash`` are
                reserved.

        Returns:
            The row as written.

        Raises:
            ValueError: if a reserved key is passed or a value is NaN or infinite.
            TypeError: if a value is not JSON-serializable.
            LogChainError: if the current last row is unreadable; nothing is appended.

        Does not establish: that the event happened as described; the log records claims
        made by the code that calls it.
        """
        clash = _RESERVED & fields.keys()
        if clash:
            raise ValueError(f"reserved log keys: {sorted(clash)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                prev = self._last_hash(f)
                row = dict(
                    fields,
                    event=event,
                    ts=datetime.now(UTC).isoformat(timespec="seconds"),
                    prev=prev,
                )
                row["hash"] = _row_hash(prev, row)
                f.write(canonical_json(row) + "\n")
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return row

    def verify(self) -> ChainHead:
        """Recompute the chain from the first row.

        Returns:
            Row count and head hash. A missing file verifies as zero rows at the genesis hash.

        Raises:
            LogChainError: at the first row whose JSON, ``prev`` link or hash is wrong.

        Does not establish: completeness; compare ``head`` with a copy recorded off the machine.
        """
        if not self.path.exists():
            return ChainHead(rows=0, head=GENESIS)
        prev, n = GENESIS, 0
        with self.path.open() as f:
            for n, line in enumerate(f, 1):
                prev = _check_row(self.path, n, line, prev)
        return ChainHead(rows=n, head=prev)

    def _last_hash(self, f: Any) -> str:
        lines = f.read().splitlines()
        if not lines:
            return GENESIS
        try:
            return json.loads(lines[-1])["hash"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LogChainError(
                self.path, len(lines), "last row is unreadable; refusing to append"
            ) from exc


def _row_hash(prev: str, row: dict[str, Any]) -> str:
    return hashlib.sha256((prev + canonical_json(row)).encode()).hexdigest()


def _check_row(path: Path, n: int, line: str, prev: str) -> str:
    try:
        row = json.loads(line)
        claimed = row.pop("hash")
    except (ValueError, KeyError, AttributeError) as exc:
        raise LogChainError(path, n, "row is not a JSON object with a hash") from exc
    if row.get("prev") != prev:
        raise LogChainError(path, n, "prev does not match the previous row's hash")
    if _row_hash(prev, row) != claimed:
        raise LogChainError(path, n, "hash does not match the row's contents")
    return claimed
