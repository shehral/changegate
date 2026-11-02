"""DSSE v1 envelopes: pre-authentication encoding, Ed25519 signatures and signature checks.

Follows the Dead Simple Signing Envelope protocol and envelope format
(github.com/secure-systems-lab/dsse). A signature covers
``PAE(payloadType, payload) = "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body``, with
lengths in bytes as ASCII decimals. The ``keyid`` field is an unauthenticated hint: a
signature counts for the key it verifies under, whatever its ``keyid`` says.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from changegate.digests import canonical_json
from changegate.record import RecordError, validation_summary
from changegate.record.keys import key_id, signature_valid

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

PAYLOAD_TYPE = "application/vnd.in-toto+json"


class Signature(BaseModel):
    """One DSSE signature: a key-id hint and the base64 signature.

    Does not establish: who signed; only a check against a trusted key does.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    keyid: str = ""
    sig: str = Field(min_length=1)


class Envelope(BaseModel):
    """A DSSE envelope: base64 payload, payload type and one or more signatures.

    Does not establish: that any signature verifies; call ``check_signatures``.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )

    payload: str
    payload_type: str = Field(alias="payloadType", min_length=1)
    signatures: tuple[Signature, ...] = Field(min_length=1)

    def payload_bytes(self) -> bytes:
        """Decode the payload (standard or URL-safe base64, padding optional).

        Returns:
            The payload bytes.

        Raises:
            RecordError: if the payload is not base64.

        Does not establish: that the payload is signed.
        """
        return _b64decode(self.payload, "payload")

    def to_json(self) -> str:
        """Return the envelope as canonical JSON text.

        Does not establish: anything about its signatures.
        """
        return canonical_json(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, text: str | bytes) -> Envelope:
        """Parse an envelope from JSON text.

        Args:
            text: the envelope file's contents.

        Returns:
            The envelope.

        Raises:
            RecordError: if the text is not JSON or not a DSSE envelope.

        Does not establish: that the payload is signed or well formed.
        """
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise RecordError(f"not a DSSE envelope: not JSON: {exc}") from exc
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise RecordError(f"not a DSSE envelope: {validation_summary(exc)}") from exc


@dataclass(frozen=True)
class SignatureCheck:
    """The outcome of checking an envelope's signatures against trusted public keys.

    Attributes:
        verified: key ids of the trusted keys that produced a valid signature.
        bad: indices of signatures that name a trusted key id but do not verify.
        unknown: how many signatures verify under none of the trusted keys.

    Does not establish: that a trusted key's holder is who the reader thinks.
    """

    verified: tuple[str, ...]
    bad: tuple[int, ...]
    unknown: int


def pae(payload_type: str, payload: bytes) -> bytes:
    """Return the DSSE v1 pre-authentication encoding of a payload type and payload.

    Args:
        payload_type: the envelope's payload type.
        payload: the payload bytes.

    Returns:
        The bytes a DSSE signature covers.

    Does not establish: anything; it is an encoding.
    """
    t = payload_type.encode()
    return b"DSSEv1 %d %b %d %b" % (len(t), t, len(payload), payload)


def sign(payload_type: str, payload: bytes, key: Ed25519PrivateKey) -> Envelope:
    """Make a new envelope carrying one Ed25519 signature over the payload.

    Args:
        payload_type: the payload type, such as ``application/vnd.in-toto+json``.
        payload: the payload bytes.
        key: the signer's private key.

    Returns:
        The envelope, with standard base64 and the signer's key id as the hint.

    Does not establish: that the payload is true; a signature shows integrity, not truth.
    """
    return Envelope(
        payload=base64.b64encode(payload).decode(),
        payload_type=payload_type,
        signatures=(_signature(payload_type, payload, key),),
    )


def add_signature(envelope: Envelope, key: Ed25519PrivateKey) -> Envelope:
    """Return a copy of the envelope with one more signature (a countersignature).

    Args:
        envelope: the envelope to countersign.
        key: the countersigner's private key.

    Returns:
        The envelope with the new signature appended.

    Raises:
        RecordError: if the payload is not base64, or this key has already signed.

    Does not establish: that the countersigner re-ran anything; it signs what it was given.
    """
    payload = envelope.payload_bytes()
    new = _signature(envelope.payload_type, payload, key)
    if new.keyid in {s.keyid for s in envelope.signatures}:
        raise RecordError(f"{new.keyid} has already signed this envelope")
    return envelope.model_copy(update={"signatures": (*envelope.signatures, new)})


def check_signatures(envelope: Envelope, keys: Sequence[Ed25519PublicKey]) -> SignatureCheck:
    """Check every signature in an envelope against a set of trusted public keys.

    Args:
        envelope: the envelope.
        keys: the trusted public keys.

    Returns:
        Which trusted keys signed, which signatures claim a trusted key but fail, and how many
        verify under no trusted key.

    Raises:
        RecordError: if the payload is not base64.

    Does not establish: anything about signatures made with keys that were not supplied.
    """
    message = pae(envelope.payload_type, envelope.payload_bytes())
    trusted = {key_id(k): k for k in keys}
    verified: list[str] = []
    bad: list[int] = []
    unknown = 0
    for i, s in enumerate(envelope.signatures):
        try:
            raw = _b64decode(s.sig, "signature")
        except RecordError:
            raw = b""
        signer = next((kid for kid, k in trusted.items() if signature_valid(k, raw, message)), None)
        if signer is not None:
            if signer not in verified:
                verified.append(signer)
        elif s.keyid in trusted:
            bad.append(i)
        else:
            unknown += 1
    return SignatureCheck(verified=tuple(verified), bad=tuple(bad), unknown=unknown)


def _signature(payload_type: str, payload: bytes, key: Ed25519PrivateKey) -> Signature:
    sig = key.sign(pae(payload_type, payload))
    return Signature(keyid=key_id(key.public_key()), sig=base64.b64encode(sig).decode())


def _b64decode(text: str, what: str) -> bytes:
    normalized = text.replace("-", "+").replace("_", "/")
    try:
        return base64.b64decode(normalized + "=" * (-len(normalized) % 4), validate=True)
    except binascii.Error as exc:
        raise RecordError(f"{what} is not base64: {exc}") from exc
