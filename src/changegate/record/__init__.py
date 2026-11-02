"""Record v0: an in-toto Statement v1 describing one model change, and its schema rules.

The statement names the candidate artifacts by digest (the in-toto ``subject``) and carries the
gate's results as its predicate: every measurement with its envelope, a noise floor per check,
the raw outputs behind recomputable scores, the negative-flip ledger, who asked for, made,
approved, repaired and re-ran the change, the action-log head and the fixed not-covered page.

The schema enforces these rules at construction time:

1. **Independence.** Whoever supplied or applied a repair never signs its re-verification as
   independent. Once a repair is recorded, its supplier and applier, ``verified_by`` and any
   countersigner must all be named by key id (``sha256:<hex>``), so every comparison is
   between keys and the verifier can bind each name to a signature. A record whose
   ``verified_by`` is the repair's supplier or applier reads "vendor-repaired; not
   independent", where "vendor" means the party that supplied or applied the repair. The
   wording "re-verified on held-out endpoints" needs a re-run and a countersignature by keys
   outside the repair, and the countersigner's key differs from ``verified_by``'s.
2. **Oversight label.** When the approver is also the requester or implementer, the record
   reads "single-person oversight, not independent". When the requester is the implementer
   and the approver is someone else, it says so. Ids are compared after trimming and
   case-folding.
3. **Completeness.** Every check has a noise-floor entry ("not measured" is an entry), and
   the not-covered page cannot be trimmed.

Submodules: ``keys`` (Ed25519 keys and key ids), ``dsse`` (the signed envelope), ``outputs``
(raw per-item outputs and the recomputed metric), ``build`` (assemble and sign), ``bom``
(CycloneDX ML-BOM export), ``verify`` (the offline verifier) and ``cli``. This module imports
none of them, so the schema loads without the signing dependency.

A record attests only to what the tool computed. The rules compare declared ids: one party
that holds two keys, or two emails, counts as two parties, and only the relying party's
choice of which keys it trusts as independent (``verify --independent-key``) can tell them
apart.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from changegate.artifacts import ArtifactRef, Sha256
from changegate.checks import CheckResult
from changegate.config import ChangeType
from changegate.digests import canonical_json
from changegate.errors import ChangegateError

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "urn:changegate:record:v0"  # placeholder until the name check clears
REVERIFIED = "re-verified on held-out endpoints"
NOT_INDEPENDENT = "vendor-repaired; not independent"
NO_REPAIR = "no repair recorded"
SINGLE_PERSON = "single-person oversight, not independent"
SEPARATE_ROLES = "separate requester, implementer and approver"
SHARED_REQUESTER = "requester is also the implementer; separate approver"
NOT_MEASURED = "not measured"
KEY_ID_PATTERN = r"^sha256:[0-9a-f]{64}$"
NOT_COVERED = (
    "agentic misalignment (RM-04)",
    "grader gaming (RM-02)",
    "evaluation awareness (RM-17)",
    "backdoors with unknown triggers (RM-14)",
    "co-loaded adapters (RM-11)",
    "staleness of the owner's own monitors (RM-12)",
    "prompt edits",
    "task accuracy",
)

Independence = Literal[
    "no repair recorded",
    "vendor-repaired; not independent",
    "re-verified on held-out endpoints",
]
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class RecordError(ChangegateError):
    """A record cannot be built, signed, parsed, verified or exported as asked.

    Does not establish: tampering by itself; a mismatch can also come from a wrong input file.
    """


class Identity(BaseModel):
    """A person or key that acted on the change, by a stable id (key id or email).

    A key id reads ``sha256:<hex>``, the SHA-256 of the key's DER SubjectPublicKeyInfo
    (``changegate.record.keys.key_id``).

    Does not establish: that the id belongs to who it says; signing binds keys, not people.
    """

    model_config = _FROZEN

    id: str = Field(min_length=1)
    display: str | None = None

    @field_validator("id")
    @classmethod
    def _plain(cls, value: str) -> str:
        """Reject ids with surrounding whitespace or control characters.

        Does not establish: that two different ids name two different parties.
        """
        if value != value.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
            raise ValueError("an identity id has no surrounding whitespace or control characters")
        return value

    @property
    def key(self) -> str:
        """Return the id as compared by the schema rules: case-folded.

        Does not establish: identity of parties; ``Vendor@x`` and ``vendor@x`` compare equal,
        two different addresses of one vendor do not.
        """
        return self.id.casefold()


class Roles(BaseModel):
    """Who asked for the change, who made it, and who approved it.

    Does not establish: that the approver reviewed anything; only that they are named.
    """

    model_config = _FROZEN

    requester: Identity
    implementer: Identity
    approver: Identity

    @property
    def single_person(self) -> bool:
        """Return True when the approver is also the requester or the implementer.

        Does not establish: independence when False; distinct ids can share an employer.
        """
        return self.approver.key in {self.requester.key, self.implementer.key}


class Subject(BaseModel):
    """An in-toto subject: an artifact name and its digests.

    Does not establish: anything beyond the digest match.
    """

    model_config = _FROZEN

    name: str = Field(min_length=1)
    digest: dict[Literal["sha256"], Sha256]

    @classmethod
    def from_ref(cls, ref: ArtifactRef) -> Subject:
        """Build a subject from an artifact reference.

        Args:
            ref: the hashed artifact.

        Returns:
            A subject named after the artifact path, with its aggregate SHA-256.

        Does not establish: that the artifact is still on disk unchanged; call ``ref.verify``.
        """
        return cls(name=ref.path.name, digest={"sha256": ref.sha256})


class Repair(BaseModel):
    """A fix applied between the pre-fix and candidate artifacts.

    ``target_manifest_sha256`` is the digest of the recipe's declared targets (prompts,
    templates, directions, metrics). Nothing in this version compares it with the
    endpoints of the results; it is carried so a reader can check disjointness by hand.

    Does not establish: that the repair worked, or that endpoints are in fact disjoint.
    """

    model_config = _FROZEN

    recipe_id: str = Field(min_length=1)
    recipe_sha256: Sha256
    target_manifest_sha256: Sha256
    supplied_by: Identity
    applied_by: Identity

    @property
    def parties(self) -> frozenset[str]:
        """Return the case-folded ids of the repair's supplier and applier.

        Does not establish: that no one else took part in the repair, or that the party
        holds no other key.
        """
        return frozenset({self.supplied_by.key, self.applied_by.key})


class NoiseFloor(BaseModel):
    """How far a check's score moves with no change, such as base against a rerun of itself.

    ``value`` None means not measured, and then ``method`` reads "not measured".

    Does not establish: that a score beyond the floor was caused by the change, or that the
    floor holds under other seeds, hardware or serving settings than ``method`` names.
    """

    model_config = _FROZEN

    value: float | None = None
    method: str = Field(NOT_MEASURED, min_length=1)
    runs: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _stated(self) -> NoiseFloor:
        """Require a finite value with a method, or no value with "not measured".

        Does not establish: that the method measured what it names.
        """
        if self.value is None:
            if self.method != NOT_MEASURED or self.runs is not None:
                raise ValueError(f"a noise floor without a value reads {NOT_MEASURED!r}")
        elif not math.isfinite(self.value) or self.method == NOT_MEASURED:
            raise ValueError("a measured noise floor needs a finite value and its method")
        return self


class OutputFile(BaseModel):
    """A raw output file, by its path relative to the outputs directory and its SHA-256.

    Does not establish: that the file holds what its producer claims; only its digest.
    """

    model_config = _FROZEN

    path: str = Field(min_length=1)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def _relative(cls, path: str) -> str:
        """Reject absolute paths, backslashes and ``..`` parts.

        Does not establish: that the file exists.
        """
        pure = PurePosixPath(path)
        if "\\" in path or pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"output paths must be relative, without '..': {path!r}")
        return path


class RawOutputs(BaseModel):
    """The raw per-item outputs behind one check's score, so a verifier can recompute it.

    ``metric`` names the recomputation: ``fail_rate`` is the share of the candidate run's
    counted items (definite outcome, category outside the run's controls) that failed;
    ``fail_rate_delta`` is the battery check's score, the candidate's fail rate minus the
    base's as ``changegate.battery.report.compare_runs`` computes it (paired over items
    scored on both runs when decoding is greedy), and needs a ``base`` run. With a ``base``
    run, the check's negative flips are recomputed too.

    Does not establish: that the scorer judged each item correctly; the recomputation
    starts from the scored outcomes.
    """

    model_config = _FROZEN

    metric: Literal["fail_rate", "fail_rate_delta"] = "fail_rate"
    candidate: OutputFile
    base: OutputFile | None = None

    @model_validator(mode="after")
    def _base_for_delta(self) -> RawOutputs:
        """Require a base run for the delta metric.

        Does not establish: that the base run is the record's base; the verifier checks it.
        """
        if self.metric == "fail_rate_delta" and self.base is None:
            raise ValueError("metric fail_rate_delta needs a base run")
        return self


class LogHead(BaseModel):
    """The action log's row count and last hash when the record was built.

    Does not establish: that the log was complete; compare with a head kept off the machine.
    """

    model_config = _FROZEN

    rows: int = Field(ge=0)
    head: Sha256


class RecordPredicate(BaseModel):
    """The body of a record: what was measured, against what, by whom, and what was not.

    ``verified_by`` is the identity that ran the gate producing ``results``. ``oversight``
    and ``independence`` are derived from the identities when omitted, and a supplied label
    must be one the identities allow.

    Does not establish: that the change is acceptable; it states what was checked and by whom.
    """

    model_config = _FROZEN

    gate_config_sha256: Sha256
    change_type: ChangeType
    base: Subject
    verified_by: Identity
    results: tuple[CheckResult, ...] = Field(min_length=1)
    noise_floors: dict[str, NoiseFloor]
    raw_outputs: dict[str, RawOutputs] = Field(default_factory=dict)
    negative_flips: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    roles: Roles
    repair: Repair | None = None
    countersigned_by: Identity | None = None
    independence: Independence
    rollback_to: Subject | None = None
    action_log: LogHead | None = None
    not_covered: tuple[str, ...] = NOT_COVERED
    oversight: Literal[
        "single-person oversight, not independent",
        "requester is also the implementer; separate approver",
        "separate requester, implementer and approver",
    ]

    @model_validator(mode="before")
    @classmethod
    def _derive_labels(cls, data: Any) -> Any:
        """Fill the oversight and independence labels from the identities when omitted.

        Does not establish: anything; the after-validator checks a supplied label.
        """
        if not isinstance(data, dict):
            return data
        try:
            if "oversight" not in data and "roles" in data:
                data = {**data, "oversight": _oversight(Roles.model_validate(data["roles"]))}
            if "independence" not in data and "verified_by" in data:
                label = _independence(
                    Identity.model_validate(data["verified_by"]),
                    _optional(Repair, data.get("repair")),
                    _optional(Identity, data.get("countersigned_by")),
                )
                data = {**data, "independence": label}
        except ValidationError:
            return data  # field validation reports the malformed input
        return data

    @model_validator(mode="after")
    def _rules(self) -> RecordPredicate:
        """Enforce the labels, the not-covered page, completeness and the independence rule.

        Does not establish: that identities are truthful; the rules compare declared ids.
        """
        if self.oversight != _oversight(self.roles):
            raise ValueError(f"oversight must read {_oversight(self.roles)!r} for these roles")
        missing = set(NOT_COVERED) - set(self.not_covered)
        if missing:
            raise ValueError(f"the not-covered page cannot drop entries: {sorted(missing)}")
        checks = [r.check for r in self.results]
        if len(set(checks)) != len(checks):
            raise ValueError("each check appears once in results")
        if any(r.score is not None and not math.isfinite(r.score) for r in self.results):
            raise ValueError("scores must be finite or None (undefined)")
        if set(self.noise_floors) != set(checks):
            raise ValueError("every check needs one noise-floor entry ('not measured' counts)")
        for field in ("raw_outputs", "negative_flips"):
            unknown = set(getattr(self, field)) - set(checks)
            if unknown:
                raise ValueError(f"{field} names checks with no result: {sorted(unknown)}")
        self._independence_rule()
        return self

    def _independence_rule(self) -> None:
        if self.repair is None:
            if self.independence != NO_REPAIR or self.countersigned_by is not None:
                raise ValueError(
                    "without a repair the record reads 'no repair recorded' and has no "
                    "countersigner"
                )
            return
        if self.independence == NO_REPAIR:
            raise ValueError("a record with a repair cannot read 'no repair recorded'")
        named = {
            "repair.supplied_by": self.repair.supplied_by,
            "repair.applied_by": self.repair.applied_by,
            "verified_by": self.verified_by,
            "countersigned_by": self.countersigned_by,
        }
        not_keys = sorted(k for k, ident in named.items() if ident and not is_key_id(ident.id))
        if not_keys:
            raise ValueError(
                "with a repair, every party is named by key id (sha256:<hex>) so it can be "
                f"bound to a signature: {', '.join(not_keys)}"
            )
        if self.countersigned_by is not None and self.countersigned_by.key == self.verified_by.key:
            raise ValueError("the countersigner's key must differ from verified_by's")
        if self.independence != REVERIFIED:
            return
        if self.verified_by.key in self.repair.parties:
            raise ValueError(
                "the supplier or applier of a repair cannot re-verify it as independent; "
                "use the not-independent wording"
            )
        cs = self.countersigned_by
        if cs is None or cs.key in self.repair.parties:
            raise ValueError("re-verified wording needs a countersignature from outside the repair")
        if not is_key_id(cs.id):
            raise ValueError("name the countersigner by key id (sha256:<hex>) so it can be checked")


class Record(BaseModel):
    """An unsigned in-toto Statement v1 whose predicate is a RecordPredicate.

    Serialize with ``statement_bytes``; field names follow in-toto (``_type``,
    ``predicateType``). Subject names are unique, since the verifier matches by name.

    Does not establish: integrity or authorship until it is wrapped in a signed DSSE envelope,
    and even then only that a key holder signed what the tool computed.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )

    type_: Literal["https://in-toto.io/Statement/v1"] = Field(STATEMENT_TYPE, alias="_type")
    subject: tuple[Subject, ...] = Field(min_length=1)
    predicate_type: Literal["urn:changegate:record:v0"] = Field(
        PREDICATE_TYPE, alias="predicateType"
    )
    predicate: RecordPredicate

    @model_validator(mode="after")
    def _unique_subjects(self) -> Record:
        """Reject two subjects with the same name.

        Does not establish: that the names mean anything; digests identify the artifacts.
        """
        names = [s.name for s in self.subject]
        if len(set(names)) != len(names):
            raise ValueError("subject names must be unique")
        return self


def statement_bytes(record: Record) -> bytes:
    """Return the canonical encoding of a record: the exact bytes a DSSE envelope signs.

    Args:
        record: the statement.

    Returns:
        UTF-8 canonical JSON (sorted keys, every field present).

    Does not establish: anything about the record's content.
    """
    return canonical_json(record.model_dump(mode="json")).encode()


def parse_statement(payload: bytes) -> Record:
    """Parse signed payload bytes into a record, strictly.

    The payload must be JSON without duplicate keys, satisfy every schema rule, and be the
    canonical encoding of the record it parses to, so no default or label is filled in
    after signing.

    Args:
        payload: the bytes a DSSE envelope carried.

    Returns:
        The record.

    Raises:
        RecordError: naming the first rule the payload breaks.

    Does not establish: who signed the payload; check the envelope's signatures first.
    """
    try:
        data = json.loads(payload, object_pairs_hook=_no_duplicate_keys)
    except ValueError as exc:
        raise RecordError(f"payload is not JSON: {exc}") from exc
    try:
        record = Record.model_validate(data)
    except ValidationError as exc:
        raise RecordError(f"payload breaks the record schema: {validation_summary(exc)}") from exc
    if statement_bytes(record) != payload:
        raise RecordError("payload is not the canonical encoding of the record it parses to")
    return record


def validation_summary(exc: ValidationError) -> str:
    """Return a one-line summary of a pydantic validation error: each location and message.

    Args:
        exc: the error.

    Returns:
        ``loc: message`` pairs joined by semicolons.

    Does not establish: anything beyond the error's own messages.
    """
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "value"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _oversight(roles: Roles) -> str:
    if roles.single_person:
        return SINGLE_PERSON
    if roles.requester.key == roles.implementer.key:
        return SHARED_REQUESTER
    return SEPARATE_ROLES


def _independence(verified_by: Identity, repair: Repair | None, cs: Identity | None) -> str:
    if repair is None:
        return NO_REPAIR
    outside = verified_by.key not in repair.parties
    countersigned = (
        cs is not None
        and cs.key not in repair.parties
        and cs.key != verified_by.key
        and is_key_id(cs.id)
    )
    return REVERIFIED if outside and countersigned else NOT_INDEPENDENT


def _optional(model: type[BaseModel], value: Any) -> Any:
    return None if value is None else model.model_validate(value)


def is_key_id(value: str) -> bool:
    """Return True when ``value`` has the key-id form ``sha256:<64 lowercase hex>``.

    Does not establish: that any key with that id exists or signed anything.
    """
    return re.fullmatch(KEY_ID_PATTERN, value) is not None


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate keys: {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)
