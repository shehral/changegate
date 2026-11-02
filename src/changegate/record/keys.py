"""Ed25519 keys for records: load the owner's PEM keys, name them by key id, sign and check.

Records are signed with the owner's own key. changegate never makes a production key; the one
generator here, ``generate_dev_keypair``, writes an unencrypted pair for tests and demos. To
make a key with OpenSSL::

    openssl genpkey -algorithm ed25519 -out record-key.pem
    openssl pkey -in record-key.pem -pubout -out record-key.pub.pem

A key id is ``sha256:`` plus the SHA-256 of the key's DER SubjectPublicKeyInfo, the same as
``openssl pkey -pubin -in record-key.pub.pem -outform DER | shasum -a 256``.

The ``cryptography`` package is imported on first use, so the record schema loads without it.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from changegate.digests import sha256_bytes
from changegate.record import RecordError

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

_NEEDS_CRYPTOGRAPHY = (
    "signing and verifying records needs the 'cryptography' package (pip install cryptography)"
)


def load_private_key(path: Path, password: bytes | None = None) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file (PKCS#8, optionally encrypted).

    Args:
        path: the PEM file.
        password: the passphrase for an encrypted key, or None.

    Returns:
        The private key.

    Raises:
        RecordError: if the file is unreadable, the passphrase is wrong, or the key is not
            Ed25519.

    Does not establish: who holds the key, or that it has not been copied.
    """
    c = _crypto()
    try:
        key = c.serialization.load_pem_private_key(Path(path).read_bytes(), password=password)
    except (OSError, ValueError, TypeError, c.exceptions.UnsupportedAlgorithm) as exc:
        raise RecordError(f"{path}: cannot load a private key: {exc}") from exc
    if not isinstance(key, c.ed25519.Ed25519PrivateKey):
        raise RecordError(f"{path}: records are signed with Ed25519 keys only")
    return key


def load_public_key(path: Path) -> Ed25519PublicKey:
    """Load an Ed25519 public key from a PEM file (SubjectPublicKeyInfo).

    Args:
        path: the PEM file.

    Returns:
        The public key.

    Raises:
        RecordError: if the file is unreadable or the key is not Ed25519.

    Does not establish: whose key it is; the reader decides whom to trust.
    """
    c = _crypto()
    try:
        key = c.serialization.load_pem_public_key(Path(path).read_bytes())
    except (OSError, ValueError, c.exceptions.UnsupportedAlgorithm) as exc:
        raise RecordError(f"{path}: cannot load a public key: {exc}") from exc
    if not isinstance(key, c.ed25519.Ed25519PublicKey):
        raise RecordError(f"{path}: records are verified with Ed25519 keys only")
    return key


def key_id(public_key: Ed25519PublicKey) -> str:
    """Return a public key's id: ``sha256:`` plus the SHA-256 of its DER SubjectPublicKeyInfo.

    Args:
        public_key: an Ed25519 public key.

    Returns:
        The key id, 71 characters.

    Does not establish: anything about the key's holder.
    """
    s = _crypto().serialization
    der = public_key.public_bytes(s.Encoding.DER, s.PublicFormat.SubjectPublicKeyInfo)
    return f"sha256:{sha256_bytes(der)}"


def signature_valid(public_key: Ed25519PublicKey, signature: bytes, message: bytes) -> bool:
    """Return True when an Ed25519 signature over a message verifies under a public key.

    Args:
        public_key: the key to check against.
        signature: the raw 64-byte signature.
        message: the signed bytes.

    Returns:
        Whether the signature verifies.

    Does not establish: when the message was signed, or that the key was not compromised.
    """
    try:
        public_key.verify(signature, message)
    except _crypto().exceptions.InvalidSignature:
        return False
    return True


def generate_dev_keypair(out_dir: Path, name: str = "dev-record-key") -> tuple[Path, Path]:
    """Write a new, unencrypted Ed25519 key pair for tests and demos.

    The private key file is created with mode 0600 and must not already exist.

    Args:
        out_dir: the directory to write into (created if missing).
        name: the file stem; writes ``<name>.pem`` and ``<name>.pub.pem``.

    Returns:
        (private key path, public key path).

    Raises:
        RecordError: if the private key file already exists.

    Does not establish: a key fit for real records; the private key is stored unencrypted.
    """
    c = _crypto()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    priv_path, pub_path = out_dir / f"{name}.pem", out_dir / f"{name}.pub.pem"
    key = c.ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        c.serialization.Encoding.PEM,
        c.serialization.PrivateFormat.PKCS8,
        c.serialization.NoEncryption(),
    )
    try:
        fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RecordError(f"{priv_path} exists; refusing to overwrite a key") from exc
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    pub_path.write_bytes(
        key.public_key().public_bytes(
            c.serialization.Encoding.PEM, c.serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return priv_path, pub_path


def _crypto() -> SimpleNamespace:
    try:
        from cryptography import exceptions
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:
        raise RecordError(_NEEDS_CRYPTOGRAPHY) from exc
    return SimpleNamespace(exceptions=exceptions, serialization=serialization, ed25519=ed25519)
