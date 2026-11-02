"""Records against third-party implementations of the specifications they follow.

Each test is skipped unless its library is installed (``securesystemslib``,
``in-toto-attestation``, ``cyclonedx-python-lib[json-validation]``); none is a dependency.
"""

import json

import pytest

from changegate.record import statement_bytes
from changegate.record.bom import ml_bom
from changegate.record.dsse import Envelope, check_signatures
from changegate.record.keys import key_id
from record_bundle import OWNER, kid, make_bundle


def test_envelope_verifies_with_securesystemslib_and_back(tmp_path):
    pytest.importorskip("securesystemslib")
    from securesystemslib import dsse
    from securesystemslib.signer import CryptoSigner, SSlibKey

    b = make_bundle(tmp_path)
    theirs = dsse.Envelope.from_dict(b.envelope())
    key = SSlibKey.from_crypto(OWNER.public_key(), keyid=kid(OWNER))
    assert list(theirs.verify([key], 1)) == [kid(OWNER)]
    # and an envelope securesystemslib signs verifies here (its keyid hint differs from ours)
    resigned = dsse.Envelope(theirs.payload, theirs.payload_type, {})
    resigned.sign(CryptoSigner(OWNER))
    ours = Envelope.from_json(json.dumps(resigned.to_dict()))
    assert check_signatures(ours, [OWNER.public_key()]).verified == (key_id(OWNER.public_key()),)


def test_statement_validates_as_in_toto_statement_v1(tmp_path):
    pytest.importorskip("in_toto_attestation")
    from google.protobuf import json_format
    from in_toto_attestation.v1 import statement, statement_pb2

    b = make_bundle(tmp_path)
    pb = json_format.Parse(statement_bytes(b.record).decode(), statement_pb2.Statement())
    statement.Statement.copy_from_pb(pb).validate()
    assert pb.subject[0].digest["sha256"] == b.record.subject[0].digest["sha256"]


def test_bom_validates_against_cyclonedx_1_7(tmp_path):
    pytest.importorskip("cyclonedx")
    pytest.importorskip("jsonschema")
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator

    b = make_bundle(tmp_path)
    errors = JsonStrictValidator(SchemaVersion.V1_7).validate_str(json.dumps(ml_bom(b.record)))
    assert errors is None, errors
