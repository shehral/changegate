"""Record commands: sign a statement, countersign, verify offline, export a CycloneDX ML-BOM.

Usage (``python -m changegate.record``, later ``changegate record``)::

    sign STATEMENT.json --key priv.pem --out record.json
    countersign record.json --key priv.pem --out record.json
    verify record.json --key pub.pem [--independent-key outside.pem] [--artifact PATH]
           [--outputs DIR] [--log actions.jsonl]
    bom record.json --key pub.pem [--independent-key outside.pem] --out bom.json

Exit codes: 0 done (verify: no check failed), 1 rejected, 2 usage error (unreadable input or
key). Keys are the owner's Ed25519 PEM files; these commands never create one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path

from pydantic import ValidationError

from changegate.record import Record, RecordError, parse_statement, validation_summary
from changegate.record.bom import ml_bom
from changegate.record.build import seal
from changegate.record.dsse import Envelope, add_signature
from changegate.record.keys import load_private_key, load_public_key
from changegate.record.verify import verify_record


class Exit(IntEnum):
    """Exit codes of the record commands (the same values as ``changegate.cli.ExitCode``).

    Does not establish: anything about the model; 0 from verify means no check failed.
    """

    OK = 0
    REJECTED = 1
    USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the record commands.

    Returns:
        The parser, with subcommands sign, countersign, verify and bom.

    Does not establish: anything; it parses arguments.
    """
    parser = argparse.ArgumentParser(
        prog="changegate record",
        description="Sign, countersign, verify offline, or export a change record. A record "
        "that verifies shows integrity, not truth.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    sign = sub.add_parser("sign", help="sign an unsigned statement with your Ed25519 key")
    sign.add_argument("statement", type=Path, help="unsigned in-toto statement (JSON)")
    counter = sub.add_parser("countersign", help="add your signature to a signed record")
    counter.add_argument("record", type=Path, help="signed record (DSSE envelope JSON)")
    for p in (sign, counter):
        p.add_argument("--key", type=Path, required=True, help="Ed25519 private key (PEM)")
        p.add_argument(
            "--password-env", metavar="VAR", help="environment variable holding the passphrase"
        )
        p.add_argument("--out", type=Path, required=True, help="where to write the record")
    verify = sub.add_parser("verify", help="verify a signed record offline")
    bom = sub.add_parser("bom", help="export a signed record's model identity as a CycloneDX BOM")
    for p in (verify, bom):
        p.add_argument("record", type=Path, help="signed record (DSSE envelope JSON)")
        p.add_argument(
            "--key",
            type=Path,
            action="append",
            required=True,
            help="trusted Ed25519 public key (PEM) of the gate operator; repeat for more",
        )
        p.add_argument(
            "--independent-key",
            type=Path,
            action="append",
            default=[],
            help="public key (PEM) you regard as independent of the repair; a re-verified "
            "record needs a countersignature under one of these, and a key listed here must "
            "not also be listed under --key",
        )
    verify.add_argument(
        "--artifact", type=Path, action="append", default=[], help="candidate file or directory"
    )
    verify.add_argument("--outputs", type=Path, help="directory holding the raw output files")
    verify.add_argument("--log", type=Path, help="the action log the record pins")
    bom.add_argument("--out", type=Path, required=True, help="where to write the BOM")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one record command and return its exit code.

    Args:
        argv: arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        An ``Exit`` value.

    Does not establish: anything beyond the command's own output; see ``verify.LIMITS``.
    """
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except RecordError as exc:
        print(f"changegate record {args.command}: {exc}", file=sys.stderr)
        return Exit.USAGE


def _sign(args: argparse.Namespace) -> int:
    key = load_private_key(args.key, _password(args.password_env))
    try:
        record = Record.model_validate_json(_read(args.statement))
    except ValidationError as exc:
        print(f"refusing to sign: {validation_summary(exc)}", file=sys.stderr)
        return Exit.REJECTED
    _write(args.out, seal(record, key).to_json())
    print(f"signed {args.statement} -> {args.out}")
    return Exit.OK


def _countersign(args: argparse.Namespace) -> int:
    key = load_private_key(args.key, _password(args.password_env))
    envelope = Envelope.from_json(_read(args.record))
    try:
        parse_statement(envelope.payload_bytes())
    except RecordError as exc:
        print(f"refusing to countersign: {exc}", file=sys.stderr)
        return Exit.REJECTED
    _write(args.out, add_signature(envelope, key).to_json())
    print(f"countersigned {args.record} -> {args.out}")
    return Exit.OK


def _verify(args: argparse.Namespace) -> int:
    report = verify_record(
        _read(args.record),
        [load_public_key(k) for k in args.key],
        independent_keys=[load_public_key(k) for k in args.independent_key],
        artifacts=args.artifact,
        outputs_dir=args.outputs,
        log_path=args.log,
    )
    print(f"changegate record verify {args.record}")
    print(report.render())
    return Exit.OK if report.ok else Exit.REJECTED


def _bom(args: argparse.Namespace) -> int:
    report = verify_record(
        _read(args.record),
        [load_public_key(k) for k in args.key],
        independent_keys=[load_public_key(k) for k in args.independent_key],
    )
    if not report.ok or report.record is None:
        print(report.render(), file=sys.stderr)
        return Exit.REJECTED
    _write(args.out, json.dumps(ml_bom(report.record), indent=2))
    print(f"wrote CycloneDX {args.out} from {args.record}")
    return Exit.OK


_COMMANDS = {"sign": _sign, "countersign": _countersign, "verify": _verify, "bom": _bom}


def _password(var: str | None) -> bytes | None:
    if var is None:
        return None
    if var not in os.environ:
        raise RecordError(f"environment variable {var} is not set")
    return os.environ[var].encode()


def _read(path: Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise RecordError(f"cannot read {path}: {exc}") from exc


def _write(path: Path, text: str) -> None:
    try:
        Path(path).write_text(text + "\n")
    except OSError as exc:
        raise RecordError(f"cannot write {path}: {exc}") from exc
