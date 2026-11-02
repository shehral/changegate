"""The offline record verifier: signature, statement, independence, subjects, metric, log.

Runs without network. Each check yields one finding: ``ok``, ``failed`` or ``not checked``
(no input was given for it, or an earlier check it depends on failed). A record is rejected
when any finding failed. What an accepted record shows is integrity against the supplied
keys, files and log, not truth: a key holder can sign a bad run.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from changegate.artifacts import ArtifactRef
from changegate.errors import ArtifactError, LogChainError
from changegate.log import GENESIS, ActionLog
from changegate.record import REVERIFIED, Record, RecordError, is_key_id, parse_statement
from changegate.record.dsse import PAYLOAD_TYPE, Envelope, SignatureCheck, check_signatures
from changegate.record.keys import key_id
from changegate.record.outputs import recompute

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

Status = Literal["ok", "failed", "not checked"]
Add = Callable[[str, Status, str], None]
CHECKS = ("envelope", "signature", "statement", "independence", "subjects", "metric", "log")
LIMITS = (
    "A record that verifies shows integrity against the supplied keys, files and log, not "
    "truth: a key holder can sign a bad run. It attests only to what the tool computed. "
    "Independence is checked between keys: a party holding two keys counts as two, so a "
    "re-verified record means only that a key you listed as independent countersigned it."
)


@dataclass(frozen=True)
class Finding:
    """One verifier check's outcome.

    Attributes:
        check: one of ``CHECKS``.
        status: ``ok``, ``failed`` or ``not checked``.
        detail: what was compared, or why it failed or was skipped.

    Does not establish: anything outside the named check.
    """

    check: str
    status: Status
    detail: str


@dataclass(frozen=True)
class Report:
    """Every finding for one record, and the record when its statement parsed.

    Does not establish: that the measurements are true; see ``LIMITS``.
    """

    findings: tuple[Finding, ...]
    record: Record | None

    @property
    def ok(self) -> bool:
        """Return True when no finding failed.

        Does not establish: anything the "not checked" findings would have shown.
        """
        return all(f.status != "failed" for f in self.findings)

    def failed(self) -> tuple[str, ...]:
        """Return the names of the checks that failed, in order.

        Does not establish: that the other checks ran; some may be "not checked".
        """
        return tuple(f.check for f in self.findings if f.status == "failed")

    def render(self) -> str:
        """Render the report for a terminal: one line per finding, a result line, the limits.

        Does not establish: more than the findings it prints.
        """
        lines = [f"  {f.status:<12} {f.check:<13} {f.detail}" for f in self.findings]
        n_failed = len(self.failed())
        n_skipped = sum(f.status == "not checked" for f in self.findings)
        result = f"REJECTED, {n_failed} check(s) failed" if n_failed else "no check failed"
        lines.append(f"result: {result}; {n_skipped} not checked")
        lines.append(LIMITS)
        return "\n".join(lines)


def verify_record(
    envelope_text: str | bytes,
    keys: Sequence[Ed25519PublicKey],
    *,
    independent_keys: Sequence[Ed25519PublicKey] = (),
    artifacts: Sequence[Path] = (),
    outputs_dir: Path | None = None,
    log_path: Path | None = None,
) -> Report:
    """Verify a signed record offline.

    Checks, in order: the DSSE envelope's form and payload type; that a trusted key signed
    it and no signature naming a trusted key fails; that the payload is the canonical form of
    a record that meets every schema rule; the independence rule against the signatures
    (``verified_by``, when named by key id and always once a repair is recorded, signed under
    ``keys``; a re-verified record carries a valid countersignature by the named countersigner
    under ``independent_keys``, that key is not also in ``keys``, and no signature by the
    repair's supplier or applier verifies under either set); each supplied artifact's digest against
    the subject of the same name; each recomputable metric against the raw outputs; and the
    action log's chain and the head the record pins.

    Args:
        envelope_text: the envelope file's contents.
        keys: trusted Ed25519 public keys of the gate operator (``verified_by``) and any
            other signer the relying party trusts for integrity.
        independent_keys: public keys the relying party regards as independent of any repair;
            only a countersignature under one of these supports "re-verified on held-out
            endpoints". Kept apart from ``keys`` so the signer's own list cannot supply it.
        artifacts: candidate files or directories to hash and match to subjects by name.
        outputs_dir: the directory holding the record's raw output files.
        log_path: the action log the record pins.

    Returns:
        The report; ``report.ok`` is False when any check failed.

    Does not establish: that the measurements are true, anything a "not checked"
    finding would have shown, or that two keys belong to two parties.
    """
    found: dict[str, Finding] = {}

    def add(check: str, status: Status, detail: str) -> None:
        found[check] = Finding(check, status, detail)

    envelope = _envelope(envelope_text, add)
    record = None
    if envelope is not None:
        sigs = _signatures(envelope, keys, independent_keys, add)
        independent = check_signatures(envelope, independent_keys)
        trusted_ids = frozenset(key_id(k) for k in keys)
        record = _statement(envelope, add)
        if record is not None:
            if found["signature"].status == "ok":
                _independence(record, sigs, independent, trusted_ids, add)
            _subjects(record, artifacts, add)
            _metric(record, outputs_dir, add)
            _log(record, log_path, add)
    for check in CHECKS:
        if check not in found:
            add(check, "not checked", "an earlier check failed")
    return Report(findings=tuple(found[c] for c in CHECKS), record=record)


def _envelope(text: str | bytes, add: Add) -> Envelope | None:
    try:
        envelope = Envelope.from_json(text)
        envelope.payload_bytes()
    except RecordError as exc:
        add("envelope", "failed", str(exc))
        return None
    if envelope.payload_type != PAYLOAD_TYPE:
        add("envelope", "failed", f"payloadType is {envelope.payload_type!r}, not {PAYLOAD_TYPE}")
        return None
    add("envelope", "ok", f"DSSE v1, {PAYLOAD_TYPE}, {len(envelope.signatures)} signature(s)")
    return envelope


def _signatures(
    envelope: Envelope,
    keys: Sequence[Ed25519PublicKey],
    independent_keys: Sequence[Ed25519PublicKey],
    add: Add,
) -> SignatureCheck:
    sigs = check_signatures(envelope, keys)
    bad = sorted({*sigs.bad, *check_signatures(envelope, independent_keys).bad})
    if bad:
        add("signature", "failed", f"signature(s) {bad} name a trusted key, fail")
    elif not sigs.verified:
        add("signature", "failed", f"no signature verifies under the {len(keys)} supplied key(s)")
    else:
        extra = f"; {sigs.unknown} signature(s) by keys not supplied" if sigs.unknown else ""
        add("signature", "ok", f"valid under {', '.join(sigs.verified)}{extra}")
    return sigs


def _statement(envelope: Envelope, add: Add) -> Record | None:
    try:
        record = parse_statement(envelope.payload_bytes())
    except RecordError as exc:
        add("statement", "failed", str(exc))
        return None
    add("statement", "ok", f"in-toto Statement v1, {record.predicate_type}; schema rules hold")
    return record


def _independence(
    record: Record,
    sigs: SignatureCheck,
    independent: SignatureCheck,
    trusted_ids: frozenset[str],
    add: Add,
) -> None:
    p = record.predicate
    operator = p.verified_by.id
    bound = p.repair is not None or is_key_id(operator)
    if bound and operator not in sigs.verified:
        add(
            "independence",
            "failed",
            f"verified_by {operator} has no valid signature under the supplied --key set",
        )
        return
    unbound = "" if bound else "; verified_by is not a key id, so it is not bound to a signature"
    if p.independence != REVERIFIED or p.repair is None:
        add("independence", "ok", f"reads {p.independence!r}{unbound}")
        return
    signers = {*sigs.verified, *independent.verified}
    applier_signed = sorted(p.repair.parties & signers)
    cs = None if p.countersigned_by is None else p.countersigned_by.id
    if applier_signed:
        who = ", ".join(applier_signed)
        add("independence", "failed", f"re-verified, yet signed by the repair's own party {who}")
    elif cs is not None and cs in trusted_ids:
        add(
            "independence",
            "failed",
            f"countersigner {cs} is also in the --key set; an independent key is listed only "
            "under --independent-key",
        )
    elif cs is None or cs not in independent.verified:
        add(
            "independence",
            "failed",
            f"no valid countersignature by {cs} under the supplied --independent-key set",
        )
    else:
        add("independence", "ok", f"reads {REVERIFIED!r}; countersigned by independent key {cs}")


def _subjects(record: Record, artifacts: Sequence[Path], add: Add) -> None:
    if not artifacts:
        add("subjects", "not checked", "no artifact supplied")
        return
    by_name = {s.name: s.digest["sha256"] for s in record.subject}
    matched = []
    for path in artifacts:
        try:
            ref = ArtifactRef.from_path(Path(path), "weights")  # the kind is not hashed
        except ArtifactError as exc:
            add("subjects", "failed", str(exc))
            return
        name = ref.path.name
        if name not in by_name:
            add("subjects", "failed", f"no subject named {name!r}")
            return
        if by_name[name] != ref.sha256:
            add("subjects", "failed", f"{name}: sha256 {ref.sha256} differs from the subject")
            return
        matched.append(name)
    unmatched = sorted(set(by_name) - set(matched))
    rest = f"; not supplied: {', '.join(unmatched)}" if unmatched else ""
    add("subjects", "ok", f"digests match for {', '.join(matched)}{rest}")


def _metric(record: Record, outputs_dir: Path | None, add: Add) -> None:
    checks = sorted(record.predicate.raw_outputs)
    if not checks:
        add("metric", "not checked", "the record carries no raw outputs")
        return
    if outputs_dir is None:
        add("metric", "not checked", "no outputs directory supplied")
        return
    lines = []
    for check in checks:
        try:
            lines.append(recompute(record, check, outputs_dir))
        except RecordError as exc:
            add("metric", "failed", str(exc))
            return
    add("metric", "ok", " | ".join(lines))


def _log(record: Record, log_path: Path | None, add: Add) -> None:
    pinned = record.predicate.action_log
    if pinned is None:
        add("log", "not checked", "the record pins no action-log head")
        return
    if log_path is None:
        add("log", "not checked", "no action log supplied")
        return
    try:
        chain = ActionLog(log_path).verify()
    except LogChainError as exc:
        add("log", "failed", str(exc))
        return
    if chain.rows < pinned.rows:
        add("log", "failed", f"log has {chain.rows} rows; the record pins row {pinned.rows}")
        return
    if _hash_at(Path(log_path), pinned.rows) != pinned.head:
        add("log", "failed", f"row {pinned.rows} hash differs from the head the record pins")
        return
    add("log", "ok", f"chain of {chain.rows} rows verifies; row {pinned.rows} matches the head")


def _hash_at(path: Path, row: int) -> str:
    if row == 0:
        return GENESIS
    with path.open() as f:
        for n, line in enumerate(f, 1):
            if n == row:
                return json.loads(line)["hash"]
    return ""
