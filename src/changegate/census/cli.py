"""Census commands: discover, sample, run, rerun-subset, report (``changegate census ...``).

Usage::

    discover --base ID --revision SHA [--relations finetune,adapter] [--limit 200]
             --out manifest.json
    sample --manifest M --declared 20 --undeclared 30 --seed S --analysis pilot|confirmatory
           [--prereg-id ID] --out sample.json
    run --manifest M --sample S --base-path DIR --gate-config gate.yaml --scratch DIR
        --results R.jsonl [--only IDS.txt] [--rerun] [--replicate-config gate-j2.yaml]
    rerun-subset --sample S --fraction 0.1 --seed S [--out IDS.txt]
    report --manifest M --sample S --results R.jsonl [--known-benign IDS.txt] --out report.md

``discover`` and ``run`` contact the public Hugging Face Hub anonymously; the others read
local files only. Exit codes: 0 done, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_args

from changegate.artifacts import ArtifactRef
from changegate.census.gate_adapter import TemplateGate
from changegate.census.hub import HfHubClient, HubClient, Relation
from changegate.census.manifest import DEFAULT_RELATIONS, Manifest, discover
from changegate.census.report import build_report, render_markdown
from changegate.census.runner import CensusRunner, read_records
from changegate.census.sample import Analysis, Sample, draw_sample, rerun_subset
from changegate.config import load_config
from changegate.errors import ChangegateError
from changegate.log import ActionLog
from changegate.registry import build_registry

EXIT_OK = 0
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the census commands.

    Returns:
        The parser, with subcommands discover, sample, run, rerun-subset and report.

    Does not establish: anything; it parses arguments.
    """
    parser = argparse.ArgumentParser(
        prog="changegate census",
        description="Run the gate over a pinned sample of public derivatives of one base and "
        "aggregate the results. Reports carry aggregates only and name no repository.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    d = sub.add_parser("discover", help="list a base's derivatives into a manifest (Hub API)")
    d.add_argument("--base", required=True, help="base repository id")
    d.add_argument("--revision", required=True, help="the base's 40-character commit id")
    d.add_argument(
        "--relations",
        default=",".join(DEFAULT_RELATIONS),
        help=f"comma-separated, from {', '.join(get_args(Relation))}",
    )
    d.add_argument("--limit", type=int, default=200, help="repositories listed per relation")
    d.add_argument("--out", type=Path, required=True, help="manifest JSON to write")

    s = sub.add_parser("sample", help="draw a seeded, stratified sample from a manifest")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--declared", type=int, required=True, help="declared-stratum quota")
    s.add_argument("--undeclared", type=int, required=True, help="undeclared-stratum quota")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--analysis", choices=get_args(Analysis), required=True)
    s.add_argument("--prereg-id", help="registration id; required for confirmatory")
    s.add_argument("--out", type=Path, required=True, help="sample JSON to write")

    r = sub.add_parser("run", help="gate every sampled derivative, deleting weights after each")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--sample", type=Path, required=True)
    r.add_argument("--base-path", type=Path, required=True, help="local copy of the base")
    r.add_argument("--gate-config", type=Path, required=True, help="template gate.yaml")
    r.add_argument("--scratch", type=Path, required=True, help="download root (node-local)")
    r.add_argument("--results", type=Path, required=True, help="JSONL results (appended)")
    r.add_argument("--only", type=Path, help="file of repository ids to restrict the run to")
    r.add_argument("--rerun", action="store_true", help="mark records as fresh-node re-runs")
    r.add_argument("--replicate-config", type=Path, help="gate.yaml with the second-family scorer")

    x = sub.add_parser("rerun-subset", help="choose the seeded share to re-run on a fresh node")
    x.add_argument("--sample", type=Path, required=True)
    x.add_argument("--fraction", type=float, default=0.1)
    x.add_argument("--seed", type=int, required=True)
    x.add_argument("--out", type=Path, help="write the ids here, one per line")

    p = sub.add_parser("report", help="aggregate results into the Drift Report")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--known-benign", type=Path, help="ids declared known-benign before the run")
    p.add_argument("--out", type=Path, required=True, help="Markdown report; JSON goes beside it")
    return parser


def main(argv: Sequence[str] | None = None, client: HubClient | None = None) -> int:
    """Run one census command and return its exit code.

    Args:
        argv: arguments without the program name; defaults to ``sys.argv[1:]``.
        client: the Hub client for ``discover`` and ``run``; an anonymous ``HfHubClient``
            when None.

    Returns:
        0 when the command finished, 2 on a usage or input error.

    Does not establish: anything a report does not show; see ``census.report``.
    """
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args, client)
    except (ChangegateError, ValueError, OSError) as exc:
        print(f"changegate census {args.command}: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _discover(args: argparse.Namespace, client: HubClient | None) -> int:
    relations = tuple(r for r in args.relations.split(",") if r)
    unknown = set(relations) - set(get_args(Relation))
    if unknown:
        raise ValueError(f"unknown relation(s): {sorted(unknown)}")
    manifest = discover(
        client or HfHubClient(),
        args.base,
        args.revision,
        relations=cast(tuple[Relation, ...], relations),
        limit_per_relation=args.limit,
    )
    digest = manifest.write(args.out)
    print(
        f"manifest {args.out}: {len(manifest.entries)} entries, {len(manifest.excluded)} "
        f"excluded (sha256 {digest})"
    )
    return EXIT_OK


def _sample(args: argparse.Namespace, _client: HubClient | None) -> int:
    sample = draw_sample(
        Manifest.load(args.manifest),
        {"declared": args.declared, "undeclared": args.undeclared},
        seed=args.seed,
        analysis=args.analysis,
        prereg_id=args.prereg_id,
    )
    digest = sample.write(args.out)
    print(f"sample {args.out}: {len(sample.items)} items, {sample.analysis} (sha256 {digest})")
    return EXIT_OK


def _run(args: argparse.Namespace, client: HubClient | None) -> int:
    manifest, sample = Manifest.load(args.manifest), Sample.load(args.sample)
    template = load_config(args.gate_config)
    root = args.gate_config.expanduser().resolve().parent
    log = ActionLog(template.output.action_log)
    base = ArtifactRef.from_path(
        args.base_path, "weights", base_ref=f"{sample.base_id}@{sample.base_revision}"
    )
    only = None if args.only is None else _read_ids(args.only)
    with contextlib.chdir(root):
        gate = TemplateGate(template, build_registry(template, log, root), log)
        replicate = None
        if args.replicate_config is not None:
            second = load_config(args.replicate_config)
            second_root = args.replicate_config.expanduser().resolve().parent
            replicate = TemplateGate(second, build_registry(second, log, second_root), log)
        runner = CensusRunner(
            client or HfHubClient(),
            gate,
            log,
            scratch=args.scratch,
            results=args.results,
            gate_sha256=gate.gate_sha256,
            replicate=replicate,
        )
        base_record = runner.run_base(sample, manifest, base)
        records = runner.run(sample, manifest, base, only=only, rerun=args.rerun)
    failed = sum(r.status == "failed" for r in [base_record, *records])
    print(
        f"census run: base {base_record.status}; {len(records)} derivative record(s) written, "
        f"{failed} failed; results {args.results}"
    )
    return EXIT_OK


def _rerun_subset(args: argparse.Namespace, _client: HubClient | None) -> int:
    ids = rerun_subset(Sample.load(args.sample), args.fraction, seed=args.seed)
    text = "\n".join(ids) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text)
        print(f"{len(ids)} id(s) -> {args.out}")
    return EXIT_OK


def _report(args: argparse.Namespace, _client: HubClient | None) -> int:
    report = build_report(
        Manifest.load(args.manifest),
        Sample.load(args.sample),
        read_records(args.results),
        known_benign=() if args.known_benign is None else _read_ids(args.known_benign),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report))
    json_path = args.out.with_suffix(".json")
    json_path.write_text(report.model_dump_json(indent=2) + "\n")
    print(f"report {args.out} and {json_path} ({report.analysis})")
    return EXIT_OK


_COMMANDS = {
    "discover": _discover,
    "sample": _sample,
    "run": _run,
    "rerun-subset": _rerun_subset,
    "report": _report,
}


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
