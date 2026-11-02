"""Run the gate over a sample: download to scratch, gate, record, delete the weights.

Each derivative is downloaded (exactly the manifest's files, at the pinned revision) into its
own directory under a scratch root, re-hashed and compared with the Hub's digests, passed to
the gate with the base, and deleted whether or not the gate succeeded. One RunRecord per
attempt is appended to a JSONL results file, and each step is written to the hash-chained
action log. A deletion that fails stops the census.

The runner hands the gate artifacts and a change type only. The declaration flag and the
stratum stay in the manifest and sample, and are joined to the results only by the report.
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import time
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from changegate.artifacts import ArtifactRef
from changegate.census.hub import HubClient, RepoId
from changegate.census.manifest import Manifest, ManifestEntry
from changegate.census.sample import Analysis, PreregId, Sample, Sha256
from changegate.checks import CheckResult, Verdict
from changegate.config import ChangeType
from changegate.digests import sha256_json
from changegate.errors import ArtifactError, ChangegateError
from changegate.log import ActionLog

Role = Literal["derivative", "base"]
Status = Literal["completed", "failed"]
BATTERY_PREFIX = "battery."
DISK_MARGIN = 1.1


class GateRunner(Protocol):
    """The gate as the census calls it: base and candidate artifacts in, check results out.

    Does not establish: which checks run; that is the gate configuration identified by the
    runner's ``gate_sha256``.
    """

    def __call__(
        self, base: ArtifactRef, candidate: ArtifactRef, change_type: ChangeType | None
    ) -> Sequence[CheckResult]:
        """Run the configured checks on one candidate.

        Args:
            base: the base artifact.
            candidate: the derivative; for the base reference run, the base itself.
            change_type: the derivative's change type, or None for the base reference run.

        Returns:
            One result per check.

        Does not establish: anything the results do not show.
        """
        ...


class RunRecord(BaseModel):
    """One attempt to run the gate on one repository, as appended to the results file.

    ``prereg_id`` and ``analysis`` (``pilot`` or ``confirmatory``) are copied from the sample
    onto every record. ``replication`` holds results from the second-family scorer, or None
    when it did not run. ``candidate_sha256`` is the aggregate digest of the files the gate
    read; they were deleted after the run.

    Does not establish: anything about the repository beyond ``results``, and nothing at all
    when ``status`` is ``failed``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    repo_id: RepoId
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    prereg_id: PreregId | None
    analysis: Analysis
    manifest_sha256: Sha256
    sample_sha256: Sha256
    gate_sha256: Sha256
    base_sha256: Sha256
    attempt: int = Field(1, ge=1)
    rerun: bool = False
    status: Status
    error: str | None = None
    candidate_sha256: Sha256 | None = None
    results: tuple[CheckResult, ...] = ()
    replication: tuple[CheckResult, ...] | None = None
    weights_deleted: bool
    node: str
    started_at: datetime
    wall_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> RunRecord:
        """Require a registration for confirmatory runs and a status that matches the fields.

        Does not establish: that the results were computed as described.
        """
        if self.analysis == "confirmatory" and self.prereg_id is None:
            raise ValueError("a confirmatory run needs a prereg_id")
        if self.status == "completed" and (
            self.error is not None or not self.results or self.candidate_sha256 is None
        ):
            raise ValueError("a completed run has results, a candidate digest and no error")
        if self.status == "failed" and not self.error:
            raise ValueError("a failed run states its error")
        return self


@dataclass(frozen=True)
class _Attempt:
    """What identifies one attempt, in the log and in its record."""

    sample: Sample
    sample_sha256: str
    role: Role
    repo_id: str
    revision: str
    number: int
    rerun: bool

    def log_fields(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "attempt": self.number,
            "rerun": self.rerun,
            "analysis": self.sample.analysis,
            "prereg_id": self.sample.prereg_id,
            "sample_sha256": self.sample_sha256,
        }


class CensusRunner:
    """Runs the gate over a sample, one derivative at a time, deleting weights after each.

    Does not establish: that the gate ran the checks it was configured with; the records
    carry what the gate returned and the gate's configuration digest.
    """

    def __init__(
        self,
        client: HubClient,
        gate: GateRunner,
        log: ActionLog,
        *,
        scratch: Path,
        results: Path,
        gate_sha256: str,
        replicate: GateRunner | None = None,
        node: str | None = None,
    ) -> None:
        """Bind the runner to a Hub client, a gate, a log and its scratch and results paths.

        Args:
            client: the Hub client used for downloads.
            gate: the gate.
            log: the hash-chained action log.
            scratch: a directory for downloads; each derivative gets a child directory
                that is deleted after its run.
            results: the JSONL results file; appended to, never rewritten.
            gate_sha256: digest identifying the gate configuration and pinned instruments.
            replicate: the second-family scorer, run on derivatives with a battery FAIL
                and on the base; None to skip replication.
            node: an operator-set node label to record (for the re-run-on-another-node
                check); defaults to ``$CHANGEGATE_NODE``, else ``"unset"``. The machine's
                host name is never read, since records travel with the results file.

        Raises:
            ValueError: if ``gate_sha256`` is not 64 lowercase hex characters.

        Does not establish: that scratch has room; each download checks free space first.
        """
        if not re.fullmatch(r"[0-9a-f]{64}", gate_sha256):
            raise ValueError("gate_sha256 must be 64 lowercase hex characters")
        self._client = client
        self._gate = gate
        self._replicate = replicate
        self._log = log
        self._scratch = Path(scratch).expanduser().resolve()
        self._results = Path(results).expanduser()
        self._gate_sha256 = gate_sha256
        self._node = node or os.environ.get("CHANGEGATE_NODE") or "unset"

    def run(
        self,
        sample: Sample,
        manifest: Manifest,
        base: ArtifactRef,
        *,
        only: Collection[str] | None = None,
        rerun: bool = False,
    ) -> list[RunRecord]:
        """Run every sampled derivative that has no completed record yet, in sample order.

        A derivative whose last attempt failed is retried as the next attempt; every
        attempt stays in the results file and the log.

        Args:
            sample: the sample.
            manifest: the manifest the sample was drawn from.
            base: the local base artifact (kept; only derivatives are deleted).
            only: restrict the run to these sampled ids, for example a ``rerun_subset``.
            rerun: mark records as re-runs (use on a different node for the reproducibility
                check).

        Returns:
            The records written by this call.

        Raises:
            ChangegateError: if the sample does not match the manifest, the results file
                holds another sample's records, or a deletion fails.
            ValueError: if ``only`` names an id outside the sample.

        Does not establish: anything for derivatives whose records read ``failed``.
        """
        digest = self._check_inputs(sample, manifest)
        if only is not None and (missing := set(only) - {i.repo_id for i in sample.items}):
            raise ValueError(f"not in the sample: {sorted(missing)}")
        done = latest(self._existing(digest))
        written = []
        for item in sample.items:
            if only is not None and item.repo_id not in only:
                continue
            prev = done.get(("derivative", item.repo_id, rerun))
            if prev is not None and prev.status == "completed":
                continue
            attempt = _Attempt(
                sample=sample,
                sample_sha256=digest,
                role="derivative",
                repo_id=item.repo_id,
                revision=item.revision,
                number=prev.attempt + 1 if prev else 1,
                rerun=rerun,
            )
            written.append(self._run_one(attempt, manifest.entry(item.repo_id), base))
        return written

    def run_base(self, sample: Sample, manifest: Manifest, base: ArtifactRef) -> RunRecord:
        """Run the gate on the base against itself; its scores are the report's base rates.

        Returns the existing record when a completed base record for this sample exists.

        Args:
            sample: the sample.
            manifest: the manifest the sample was drawn from.
            base: the local base artifact; it is not deleted.

        Returns:
            The base record.

        Raises:
            ChangegateError: if the sample does not match the manifest or the results file
                holds another sample's records.

        Does not establish: the base's behavior outside the gate's checks.
        """
        digest = self._check_inputs(sample, manifest)
        prev = latest(self._existing(digest)).get(("base", sample.base_id, False))
        if prev is not None and prev.status == "completed":
            return prev
        attempt = _Attempt(
            sample=sample,
            sample_sha256=digest,
            role="base",
            repo_id=sample.base_id,
            revision=sample.base_revision,
            number=prev.attempt + 1 if prev else 1,
            rerun=False,
        )
        started, t0 = datetime.now(UTC), time.monotonic()
        results: tuple[CheckResult, ...] = ()
        replication = None
        error = None
        try:
            results = _results(self._gate, base, base, None)
            if self._replicate is not None:
                replication = _results(self._replicate, base, base, None)
        except Exception as exc:  # recorded as a failed run, never dropped
            error = _describe(exc)
        return self._write(
            RunRecord(
                **self._common(attempt, base, started, t0),
                status="failed" if error else "completed",
                error=error,
                candidate_sha256=base.sha256,
                results=results,
                replication=replication,
                weights_deleted=False,
            )
        )

    def _run_one(self, attempt: _Attempt, entry: ManifestEntry, base: ArtifactRef) -> RunRecord:
        dest = self._scratch / f"{entry.repo_id.replace('/', '--')}@{entry.revision[:12]}"
        if not self._delete(dest):
            raise ChangegateError(f"stale scratch directory could not be removed: {dest}")
        fields = attempt.log_fields()
        started, t0 = datetime.now(UTC), time.monotonic()
        results: tuple[CheckResult, ...] = ()
        replication = None
        candidate_sha = error = None
        try:
            self._check_space(entry)
            self._log.append(
                "census.fetch", **fields, files=len(entry.files), bytes=entry.fetch_bytes
            )
            self._client.download(
                entry.repo_id, entry.revision, [f.name for f in entry.files], dest
            )
            ref = _verified(entry, dest, f"{attempt.sample.base_id}@{attempt.sample.base_revision}")
            candidate_sha = ref.sha256
            results = _results(self._gate, base, ref, entry.change_type)
            if self._replicate is not None and any(_battery_fail(r) for r in results):
                replication = _results(self._replicate, base, ref, entry.change_type)
        except Exception as exc:  # recorded as a failed run, never dropped
            error = _describe(exc)
        finally:
            deleted = self._delete(dest)
            self._log.append("census.delete", **fields, deleted=deleted)
        record = self._write(
            RunRecord(
                **self._common(attempt, base, started, t0),
                status="failed" if error else "completed",
                error=error,
                candidate_sha256=candidate_sha,
                results=results,
                replication=replication,
                weights_deleted=deleted,
            )
        )
        if not deleted:
            raise ChangegateError(f"weights could not be deleted from {dest}; census stopped")
        return record

    def _common(
        self, attempt: _Attempt, base: ArtifactRef, started: datetime, t0: float
    ) -> dict[str, Any]:
        return {
            "role": attempt.role,
            "repo_id": attempt.repo_id,
            "revision": attempt.revision,
            "prereg_id": attempt.sample.prereg_id,
            "analysis": attempt.sample.analysis,
            "manifest_sha256": attempt.sample.manifest_sha256,
            "sample_sha256": attempt.sample_sha256,
            "gate_sha256": self._gate_sha256,
            "base_sha256": base.sha256,
            "attempt": attempt.number,
            "rerun": attempt.rerun,
            "node": self._node,
            "started_at": started,
            "wall_seconds": time.monotonic() - t0,
        }

    def _check_inputs(self, sample: Sample, manifest: Manifest) -> str:
        if sample.manifest_sha256 != manifest.sha256():
            raise ChangegateError("the sample was not drawn from this manifest")
        self._scratch.mkdir(parents=True, exist_ok=True)
        return sample.sha256()

    def _existing(self, sample_sha256: str) -> list[RunRecord]:
        if not self._results.exists():
            return []
        records = read_records(self._results)
        if any(r.sample_sha256 != sample_sha256 for r in records):
            raise ChangegateError(
                f"{self._results} holds records of another sample; use a new results file"
            )
        return records

    def _check_space(self, entry: ManifestEntry) -> None:
        need = entry.fetch_bytes
        free = shutil.disk_usage(self._scratch).free
        if need is not None and free < need * DISK_MARGIN:
            raise ChangegateError(f"scratch has {free} bytes free; {entry.repo_id} needs {need}")

    def _delete(self, dest: Path) -> bool:
        if dest.parent.resolve() != self._scratch or dest.name in ("", ".", ".."):
            raise ChangegateError(f"refusing to delete outside the scratch root: {dest}")
        try:
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.exists():
                shutil.rmtree(dest)
        except OSError:
            return False
        return not dest.exists()

    def _write(self, record: RunRecord) -> RunRecord:
        self._results.parent.mkdir(parents=True, exist_ok=True)
        with self._results.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(record.model_dump_json() + "\n")
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        self._log.append(
            "census.record",
            repo_id=record.repo_id,
            role=record.role,
            attempt=record.attempt,
            rerun=record.rerun,
            status=record.status,
            record_sha256=sha256_json(record.model_dump(mode="json")),
        )
        return record


def read_records(path: Path) -> list[RunRecord]:
    """Read every record from a JSONL results file, in file order.

    Args:
        path: the results file.

    Returns:
        The records.

    Raises:
        ChangegateError: naming the first line that is not a valid RunRecord.

    Does not establish: that no line was removed or edited; compare each record's digest
    with the ``census.record`` rows of the action log.
    """
    records = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(RunRecord.model_validate_json(line))
        except ValidationError as exc:
            raise ChangegateError(f"{path}: line {n} is not a run record: {exc}") from exc
    return records


def latest(records: Iterable[RunRecord]) -> dict[tuple[Role, str, bool], RunRecord]:
    """Keep the highest attempt per (role, repository, re-run flag).

    Args:
        records: records in file order.

    Returns:
        A mapping from (role, repo_id, rerun) to that key's latest attempt.

    Does not establish: anything about earlier attempts, which stay in the file.
    """
    out: dict[tuple[Role, str, bool], RunRecord] = {}
    for r in records:
        key = (r.role, r.repo_id, r.rerun)
        if key not in out or r.attempt >= out[key].attempt:
            out[key] = r
    return out


def is_battery_check(check: str) -> bool:
    """Return True for battery checks, named ``battery.<name>`` in gate.yaml.

    Args:
        check: a check name.

    Returns:
        Whether the check is a tier-D battery.

    Does not establish: what the battery measures.
    """
    return check.startswith(BATTERY_PREFIX)


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:2000]


def _battery_fail(result: CheckResult) -> bool:
    return is_battery_check(result.check) and result.verdict is Verdict.FAIL


def _results(
    gate: GateRunner, base: ArtifactRef, candidate: ArtifactRef, change_type: ChangeType | None
) -> tuple[CheckResult, ...]:
    results = tuple(gate(base, candidate, change_type))
    if not results:
        raise ChangegateError("the gate returned no results")
    if not all(isinstance(r, CheckResult) for r in results):
        raise TypeError("the gate must return CheckResult objects")
    return results


def _verified(entry: ManifestEntry, dest: Path, base_ref: str) -> ArtifactRef:
    root = dest.resolve()
    for p in dest.rglob("*"):
        if p.is_symlink() and not p.resolve().is_relative_to(root):
            raise ArtifactError(
                f"{p} links outside scratch; deleting scratch would leave the weights on disk"
            )
    ref = ArtifactRef.from_path(dest, entry.kind, base_ref)
    got = {f.relpath: f for f in ref.files}
    expected = {f.name for f in entry.files}
    if extra := sorted(got.keys() - expected):
        raise ArtifactError(f"unexpected files after download: {extra}")
    for hub in entry.files:
        local = got.get(hub.name)
        if local is None:
            raise ArtifactError(f"{hub.name} is missing after download")
        if hub.sha256 is not None and local.sha256 != hub.sha256:
            raise ArtifactError(f"{hub.name}: SHA-256 differs from the Hub's LFS digest")
        if hub.size is not None and local.size != hub.size:
            raise ArtifactError(f"{hub.name}: size differs from the Hub's")
    return ref
