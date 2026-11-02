"""Assemble a record from a gate run's inputs and results, and seal it in a signed envelope.

``build_statement`` refuses to emit a record its own verifier would reject: the statement is
encoded and parsed back strictly, and raw outputs are written, read back and recomputed
before the record is returned. It also refuses a predicate holding a string that looks like
an absolute or home-relative local path, since a record is meant to be shared and such a
path names the home directory or cluster account it was built under.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from changegate.artifacts import ArtifactRef
from changegate.battery import negative_flips
from changegate.checks import CheckResult
from changegate.config import GateConfig, config_sha256
from changegate.errors import LogChainError
from changegate.log import ActionLog
from changegate.record import (
    NOT_COVERED,
    Identity,
    LogHead,
    NoiseFloor,
    Record,
    RecordError,
    RecordPredicate,
    Repair,
    Roles,
    Subject,
    parse_statement,
    statement_bytes,
    validation_summary,
)
from changegate.record.dsse import PAYLOAD_TYPE, Envelope, sign
from changegate.record.outputs import Metric, RunPair, recompute, write_runs

# An absolute POSIX path, a home-relative path or a Windows drive path, at the start of the
# string or after whitespace, a quote, a bracket, "=" or ",".
_LOCAL_PATH = re.compile(r"(?:^|[\s'\"(\[=,])(?:~[/A-Za-z]|~$|/[A-Za-z0-9_.\-]|[A-Za-z]:[\\/])")

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def build_statement(
    cfg: GateConfig,
    base: ArtifactRef,
    candidates: Sequence[ArtifactRef],
    results: Sequence[CheckResult],
    *,
    roles: Roles,
    verified_by: Identity,
    runs: Mapping[str, RunPair] | None = None,
    outputs_dir: Path | None = None,
    noise_floors: Mapping[str, NoiseFloor] | None = None,
    repair: Repair | None = None,
    countersigned_by: Identity | None = None,
    rollback_to: ArtifactRef | None = None,
    log: ActionLog | None = None,
    not_covered: Sequence[str] = NOT_COVERED,
    metrics: Mapping[str, Metric] | None = None,
) -> Record:
    """Build the unsigned record of one change.

    Checks without a noise floor get a "not measured" entry. For each check in ``runs`` the
    runs are written under ``outputs_dir``, the negative-flip ledger is computed from them,
    and the check's score is recomputed; the oversight and independence labels are derived
    from the identities.

    Args:
        cfg: the gate config the results came from (its digest and change type are recorded).
        base: the base artifact.
        candidates: the candidate artifacts; they become the in-toto subjects.
        results: every check result from the run.
        roles: requester, implementer and approver.
        verified_by: the identity that ran the gate.
        runs: tier-D battery runs by check name, for recomputation and the flip ledger.
        outputs_dir: where to write the runs; needed when ``runs`` is given.
        noise_floors: measured noise floors by check name.
        repair: the repair applied before this run, if any.
        countersigned_by: the outside countersigner, named by key id, if any.
        rollback_to: the artifact to roll back to.
        log: the action log whose current head the record pins.
        not_covered: the not-covered page; may add entries, never drop the fixed ones.
        metrics: which quantity each run check's score is (``fail_rate`` by default;
            ``fail_rate_delta`` for a battery check, whose score is candidate minus base).

    Returns:
        The record.

    Raises:
        RecordError: if an input breaks a schema rule, a run disagrees with its result, the
            log does not verify, or the predicate holds a local path (see ``local_paths``).

    Does not establish: that the results are correct; the record states what was computed.
    """
    runs = dict(runs or {})
    metrics = dict(metrics or {})
    if runs and outputs_dir is None:
        raise RecordError("battery runs need an outputs directory")
    floors = dict(noise_floors or {})
    checks = {r.check for r in results}
    unknown = (set(floors) | set(runs)) - checks
    if set(metrics) - set(runs):
        raise RecordError(f"metrics name checks with no runs: {sorted(set(metrics) - set(runs))}")
    if unknown:
        raise RecordError(f"noise floors or runs name checks with no result: {sorted(unknown)}")
    try:
        flips = {c: negative_flips(p.base, p.candidate) for c, p in runs.items() if p.base}
    except ValueError as exc:
        raise RecordError(str(exc)) from exc
    try:
        head = None if log is None else log.verify()
    except LogChainError as exc:
        raise RecordError(f"the action log does not verify: {exc}") from exc
    try:
        record = Record(
            subject=tuple(Subject.from_ref(c) for c in candidates),
            predicate=RecordPredicate(
                gate_config_sha256=config_sha256(cfg),
                change_type=cfg.change_type,
                base=Subject.from_ref(base),
                verified_by=verified_by,
                results=tuple(results),
                noise_floors={r.check: floors.get(r.check, NoiseFloor()) for r in results},
                raw_outputs={
                    c: write_runs(c, p, outputs_dir, metrics.get(c, "fail_rate"))
                    for c, p in runs.items()
                },
                negative_flips=flips,
                roles=roles,
                repair=repair,
                countersigned_by=countersigned_by,
                rollback_to=None if rollback_to is None else Subject.from_ref(rollback_to),
                action_log=None if head is None else LogHead(rows=head.rows, head=head.head),
                not_covered=tuple(not_covered),
            ),
        )
    except ValidationError as exc:
        raise RecordError(f"the record breaks a schema rule: {validation_summary(exc)}") from exc
    try:
        parse_statement(statement_bytes(record))
    except ValueError as exc:  # evidence holding a value JSON cannot carry
        raise RecordError(f"the record cannot be encoded as canonical JSON: {exc}") from exc
    leaks = sorted(set(local_paths(record.predicate.model_dump(mode="json"))))
    if leaks:
        raise RecordError(
            "the record would carry local paths (store them relative to the outputs "
            f"directory, or drop them): {leaks[:5]}"
        )
    for check in runs:
        recompute(record, check, outputs_dir)
    return record


def seal(record: Record, key: Ed25519PrivateKey) -> Envelope:
    """Sign a record's canonical bytes into a DSSE envelope (payload type in-toto JSON).

    Args:
        record: the statement.
        key: the signer's Ed25519 private key (the owner's own).

    Returns:
        The signed envelope.

    Does not establish: that the record is true; a key holder can sign a bad run.
    """
    return sign(PAYLOAD_TYPE, statement_bytes(record), key)


def local_paths(value: Any, where: str = "predicate") -> Iterator[str]:
    """Yield the location of every string (or key) in a JSON value that holds a local path.

    A local path is an absolute POSIX path, a path starting with ``~``, or a Windows drive
    path, at the start of the string or after whitespace, a quote, a bracket, ``=`` or ``,``.

    Args:
        value: a JSON-compatible value (dicts, lists, strings, numbers).
        where: the dotted location of ``value``, used in the yielded names.

    Yields:
        Dotted locations such as ``predicate.results.0.evidence.path``.

    Does not establish: that no path is present in another form (a relative path, an
    encoded or split string); it matches the common forms only.
    """
    if isinstance(value, str):
        if _LOCAL_PATH.search(value):
            yield where
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and _LOCAL_PATH.search(k):
                yield f"{where}.<key>"
            yield from local_paths(v, f"{where}.{k}")
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            yield from local_paths(v, f"{where}.{i}")
