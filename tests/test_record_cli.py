import json
import subprocess
import sys

# cryptography is a base dependency: a missing install fails these tests, never skips them.
from cryptography.hazmat.primitives import serialization

from changegate.digests import canonical_json
from changegate.record import NOT_INDEPENDENT, REVERIFIED, Identity, statement_bytes
from changegate.record.cli import Exit, main
from record_bundle import AUDITOR, OUTSIDE, OWNER, VENDOR, kid, make_bundle, pem, vendor_repair


def _verify_args(b, *keys):
    args = ["verify", str(b.envelope_path)]
    for k in keys:
        args += ["--key", str(k)]
    for a in b.candidates:
        args += ["--artifact", str(a)]
    return [*args, "--outputs", str(b.outputs), "--log", str(b.log)]


def test_verify_accepts_an_untouched_record(tmp_path, capsys):
    b = make_bundle(tmp_path / "b")
    pub = pem(OWNER, tmp_path / "owner.pub.pem")
    assert main(_verify_args(b, pub)) == Exit.OK
    out = capsys.readouterr().out
    assert "no check failed; 0 not checked" in out
    assert "not truth" in out


def test_verify_rejects_a_tampered_record(tmp_path, capsys):
    b = make_bundle(tmp_path / "b")
    (b.candidates[0] / "adapter_model.safetensors").write_bytes(b"swapped")
    assert main(_verify_args(b, pem(OWNER, tmp_path / "owner.pub.pem"))) == Exit.REJECTED
    assert "REJECTED" in capsys.readouterr().out


def test_verify_usage_errors(tmp_path, capsys):
    b = make_bundle(tmp_path / "b")
    assert main(["verify", str(b.envelope_path), "--key", str(tmp_path / "none.pem")]) == Exit.USAGE
    assert "cannot load a public key" in capsys.readouterr().err
    pub = pem(OWNER, tmp_path / "owner.pub.pem")
    assert main(["verify", str(tmp_path / "missing.json"), "--key", str(pub)]) == Exit.USAGE


def test_sign_countersign_verify_and_bom(tmp_path, capsys):
    b = make_bundle(
        tmp_path / "b",
        repair=vendor_repair(),
        verified_by=Identity(id=kid(OUTSIDE)),
        countersigned_by=Identity(id=kid(AUDITOR)),
        signers=(OUTSIDE,),
    )
    statement = tmp_path / "statement.json"
    statement.write_bytes(statement_bytes(b.record))
    outside_key = pem(OUTSIDE, tmp_path / "outside.pem", private=True)
    auditor_key = pem(AUDITOR, tmp_path / "auditor.pem", private=True)
    outside_pub = pem(OUTSIDE, tmp_path / "outside.pub.pem")
    auditor_pub = pem(AUDITOR, tmp_path / "auditor.pub.pem")
    signed = b.envelope_path
    assert main(["sign", str(statement), "--key", str(outside_key), "--out", str(signed)]) == 0
    independent = ["--independent-key", str(auditor_pub)]
    # the re-runner's signature alone does not carry the countersignature the label needs
    assert main([*_verify_args(b, outside_pub), *independent]) == Exit.REJECTED
    assert "no valid countersignature" in capsys.readouterr().out
    assert main(["countersign", str(signed), "--key", str(auditor_key), "--out", str(signed)]) == 0
    assert main([*_verify_args(b, outside_pub), *independent]) == Exit.OK
    assert REVERIFIED in capsys.readouterr().out
    # review H-1: the countersigner listed under --key does not count as independent
    assert main(_verify_args(b, outside_pub, auditor_pub)) == Exit.REJECTED
    assert "--independent-key" in capsys.readouterr().out
    bom = tmp_path / "bom.json"
    args = ["bom", str(signed), "--key", str(outside_pub), *independent, "--out", str(bom)]
    assert main(args) == Exit.OK
    doc = json.loads(bom.read_text())
    assert doc["specVersion"] == "1.7"
    assert {"name": "changegate:independence", "value": REVERIFIED} in doc["components"][0][
        "properties"
    ]


def test_sign_refuses_a_statement_that_breaks_the_schema(tmp_path, capsys):
    b = make_bundle(tmp_path / "b", repair=vendor_repair(), verified_by=vendor_repair().applied_by)
    data = json.loads(statement_bytes(b.record))
    assert data["predicate"]["independence"] == NOT_INDEPENDENT
    data["predicate"]["independence"] = REVERIFIED  # the vendor ran the gate itself
    statement = tmp_path / "statement.json"
    statement.write_text(canonical_json(data))
    key = pem(VENDOR, tmp_path / "vendor.pem", private=True)
    out = tmp_path / "record.json"
    assert main(["sign", str(statement), "--key", str(key), "--out", str(out)]) == Exit.REJECTED
    assert "cannot re-verify it as independent" in capsys.readouterr().err
    assert not out.exists()


def test_sign_reads_a_passphrase_from_the_named_variable(tmp_path, monkeypatch):
    b = make_bundle(tmp_path / "b")
    statement = tmp_path / "statement.json"
    statement.write_bytes(statement_bytes(b.record))
    key = tmp_path / "enc.pem"
    key.write_bytes(
        OWNER.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(b"s3cret"),
        )
    )
    args = ["sign", str(statement), "--key", str(key), "--out", str(tmp_path / "r.json")]
    assert main([*args, "--password-env", "CG_TEST_PASS"]) == Exit.USAGE
    monkeypatch.setenv("CG_TEST_PASS", "s3cret")
    assert main([*args, "--password-env", "CG_TEST_PASS"]) == Exit.OK


def test_bom_refuses_an_unverified_record(tmp_path, capsys):
    b = make_bundle(tmp_path / "b")
    wrong = pem(OUTSIDE, tmp_path / "outside.pub.pem")
    out = tmp_path / "bom.json"
    assert main(["bom", str(b.envelope_path), "--key", str(wrong), "--out", str(out)]) == 1
    assert not out.exists()


def test_module_entry_point_runs(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "changegate.record", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "verify" in proc.stdout and "countersign" in proc.stdout
