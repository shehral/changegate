import base64
import hashlib
import os
import stat

import pytest

# cryptography is a base dependency: a missing install fails these tests, never skips them.
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from changegate.record import RecordError
from changegate.record.dsse import (
    PAYLOAD_TYPE,
    Envelope,
    Signature,
    add_signature,
    check_signatures,
    pae,
    sign,
)
from changegate.record.keys import (
    generate_dev_keypair,
    key_id,
    load_private_key,
    load_public_key,
    signature_valid,
)
from record_bundle import ATTACKER, OUTSIDE, OWNER, kid, pem

BODY = b'{"hello": "world"}'


def test_pae_matches_the_dsse_protocol_example():
    # DSSE protocol.md test vector
    assert pae("http://example.com/HelloWorld", b"hello world") == (
        b"DSSEv1 29 http://example.com/HelloWorld 11 hello world"
    )


def test_pae_counts_bytes_not_characters():
    assert pae("té", b"\xff") == b"DSSEv1 3 t\xc3\xa9 1 \xff"


def test_sign_and_check():
    env = sign(PAYLOAD_TYPE, BODY, OWNER)
    assert env.payload_bytes() == BODY
    assert env.signatures[0].keyid == kid(OWNER)
    check = check_signatures(env, [OWNER.public_key()])
    assert (check.verified, check.bad, check.unknown) == ((kid(OWNER),), (), 0)


def test_signature_under_an_unsupplied_key_counts_as_unknown():
    check = check_signatures(sign(PAYLOAD_TYPE, BODY, ATTACKER), [OWNER.public_key()])
    assert (check.verified, check.bad, check.unknown) == ((), (), 1)


def test_keyid_is_only_a_hint():
    env = sign(PAYLOAD_TYPE, BODY, OWNER)
    blank = env.model_copy(
        update={"signatures": (env.signatures[0].model_copy(update={"keyid": ""}),)}
    )
    assert check_signatures(blank, [OWNER.public_key()]).verified == (kid(OWNER),)
    # a signature by another key that claims the owner's key id does not verify: it is "bad"
    forged = sign(PAYLOAD_TYPE, BODY, ATTACKER)
    forged = forged.model_copy(
        update={"signatures": (Signature(keyid=kid(OWNER), sig=forged.signatures[0].sig),)}
    )
    check = check_signatures(forged, [OWNER.public_key()])
    assert (check.verified, check.bad) == ((), (0,))


def test_signature_covers_the_payload_type():
    env = sign(PAYLOAD_TYPE, BODY, OWNER).model_copy(update={"payload_type": "application/json"})
    assert check_signatures(env, [OWNER.public_key()]).bad == (0,)


def test_url_safe_unpadded_base64_is_accepted():
    env = sign(PAYLOAD_TYPE, b"\xfb\xff\xfe" * 5 + b"x", OWNER)
    urlsafe = base64.urlsafe_b64encode(env.payload_bytes()).decode().rstrip("=")
    env2 = env.model_copy(update={"payload": urlsafe})
    assert check_signatures(env2, [OWNER.public_key()]).verified == (kid(OWNER),)


def test_malformed_base64_is_an_error():
    env = sign(PAYLOAD_TYPE, BODY, OWNER).model_copy(update={"payload": "not base64!"})
    with pytest.raises(RecordError, match="base64"):
        env.payload_bytes()


def test_countersignature_appends_once_per_key():
    env = add_signature(sign(PAYLOAD_TYPE, BODY, OWNER), OUTSIDE)
    check = check_signatures(env, [OWNER.public_key(), OUTSIDE.public_key()])
    assert set(check.verified) == {kid(OWNER), kid(OUTSIDE)}
    with pytest.raises(RecordError, match="already signed"):
        add_signature(env, OUTSIDE)


def test_envelope_json_round_trip_and_errors():
    env = sign(PAYLOAD_TYPE, BODY, OWNER)
    assert Envelope.from_json(env.to_json()) == env
    assert '"payloadType"' in env.to_json()
    for bad in ("{", "[]", '{"payload": "", "payloadType": "x", "signatures": []}'):
        with pytest.raises(RecordError, match="not a DSSE envelope"):
            Envelope.from_json(bad)


def test_ed25519_signatures_are_deterministic():
    assert sign(PAYLOAD_TYPE, BODY, OWNER) == sign(PAYLOAD_TYPE, BODY, OWNER)


def test_key_id_is_the_sha256_of_the_der_public_key():
    der = OWNER.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    assert key_id(OWNER.public_key()) == "sha256:" + hashlib.sha256(der).hexdigest()


def test_signature_valid_rejects_wrong_messages_and_lengths():
    sig = OWNER.sign(b"m")
    assert signature_valid(OWNER.public_key(), sig, b"m")
    assert not signature_valid(OWNER.public_key(), sig, b"n")
    assert not signature_valid(OWNER.public_key(), b"", b"m")


def test_keys_load_from_pem(tmp_path):
    priv = load_private_key(pem(OWNER, tmp_path / "k.pem", private=True))
    pub = load_public_key(pem(OWNER, tmp_path / "k.pub.pem"))
    assert key_id(priv.public_key()) == key_id(pub) == kid(OWNER)


def test_encrypted_private_key_needs_its_passphrase(tmp_path):
    path = tmp_path / "enc.pem"
    path.write_bytes(
        OWNER.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(b"pass"),
        )
    )
    assert key_id(load_private_key(path, b"pass").public_key()) == kid(OWNER)
    for password in (None, b"wrong"):
        with pytest.raises(RecordError, match="cannot load"):
            load_private_key(path, password)


def test_only_ed25519_keys_load(tmp_path):
    other = ec.generate_private_key(ec.SECP256R1())
    priv = tmp_path / "ec.pem"
    priv.write_bytes(
        other.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    pub = tmp_path / "ec.pub.pem"
    pub.write_bytes(
        other.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    with pytest.raises(RecordError, match="Ed25519"):
        load_private_key(priv)
    with pytest.raises(RecordError, match="Ed25519"):
        load_public_key(pub)
    with pytest.raises(RecordError, match="cannot load"):
        load_public_key(tmp_path / "missing.pem")


def test_dev_keypair_is_private_to_its_owner_and_never_overwritten(tmp_path):
    priv, pub = generate_dev_keypair(tmp_path / "keys")
    assert stat.S_IMODE(os.stat(priv).st_mode) == 0o600
    assert key_id(load_private_key(priv).public_key()) == key_id(load_public_key(pub))
    with pytest.raises(RecordError, match="refusing to overwrite"):
        generate_dev_keypair(tmp_path / "keys")
