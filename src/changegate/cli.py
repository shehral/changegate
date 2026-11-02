"""Command-line entry point: ``changegate <command>``.

Exit codes: 0 = every verdict PASS or WARN (or a non-measuring command succeeded), 1 = at
least one FAIL (record: a record was rejected), 2 = usage, config or input error, 3 = the
command or a declared check is not built in this version, 4 = no overall verdict (a check
with an envelope gave none, or no check has an envelope).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from enum import IntEnum
from pathlib import Path

from changegate import __version__
from changegate.battery.sources import SOURCES, default_data_dir, fetch
from changegate.census.cli import main as census_main
from changegate.config import GateConfig, config_sha256, load_config
from changegate.errors import ChangegateError, ConfigError, NotBuiltError
from changegate.gate import gate_command
from changegate.organisms.cli import main as organism_main
from changegate.record.cli import main as record_main
from changegate.registry import build_registry


class ExitCode(IntEnum):
    """Process exit codes.

    Does not establish: anything about the model; a 0 exit can carry WARN verdicts.
    """

    OK = 0
    FAIL = 1
    USAGE = 2
    NOT_BUILT = 3
    NO_VERDICT = 4


_CHECK_COMMANDS: dict[str, tuple[str, Callable[[str], bool]]] = {
    "lineage": (
        "Tier B. Hash the candidate and compare its config, tokenizer and per-layer weights "
        "with the claimed base; compute the abliteration ratio on the base's refusal direction.",
        lambda key: key == "lineage",
    ),
    "adapter-diff": (
        "Tier A. Project a LoRA update onto the base's refusal direction without "
        "materializing the full weight delta.",
        lambda key: key == "adapter-diff",
    ),
    "drift": (
        "Tier C. Compare base and candidate activations on fixed prompt sets: refusal "
        "direction cosine, harmful-prompt projection and harmful/harmless separation.",
        lambda key: key.startswith("drift."),
    ),
    "battery": (
        "Tier D. Generate with local transformers or vLLM on base and candidate, score with a "
        "pinned rubric or local guard model, and record per-item outcomes for negative flips.",
        lambda key: key.startswith("battery."),
    ),
}
_GATE = (
    "Run the checks declared in gate.yaml and report PASS / WARN / FAIL against the "
    "user-declared envelope."
)
_DELEGATED = {
    "census": "Run the gate over a pinned sample of public derivatives of one base "
    "(discover, sample, run, rerun-subset, report).",
    "record": "Sign, countersign, verify offline, or export a change record "
    "(sign, countersign, verify, bom).",
    "organism": "Build planted known-bad models for calibration and commit the bank "
    "(build, bank, split).",
}
_DELEGATED_MAIN: dict[str, Callable[[list[str]], int]] = {
    "census": census_main,
    "record": record_main,
    "organism": organism_main,
}
_FETCH = (
    "Download the pinned battery source files into the data directory, checking each "
    "against its SHA-256. The only command that needs the network besides census."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every command.

    ``census``, ``record`` and ``organism`` are listed here for ``--help``; ``main`` hands
    their arguments to their own parsers.

    Returns:
        The top-level parser.

    Does not establish: that a parsed command measured anything; see the exit codes.
    """
    parser = argparse.ArgumentParser(
        prog="changegate",
        description="Compare a changed model with its base and report PASS / WARN / FAIL "
        "against thresholds you declare. A result under WARN means not detected "
        "at recall r, nothing more; r is not measured in this version.",
    )
    parser.add_argument("--version", action="version", version=f"changegate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    gate = sub.add_parser("gate", help=_GATE, description=_GATE)
    gate.add_argument("--config", type=Path, required=True, help="path to gate.yaml")
    gate.add_argument(
        "--validate-only",
        action="store_true",
        help="check gate.yaml against the schema and print the plan; measures nothing",
    )
    for name, (text, _) in _CHECK_COMMANDS.items():
        p = sub.add_parser(name, help=text, description=text + " Runs only these checks.")
        p.add_argument("--config", type=Path, required=True, help="path to gate.yaml")
    for name, text in _DELEGATED.items():
        sub.add_parser(name, help=text, description=text, add_help=False)
    fetch_p = sub.add_parser("fetch-data", help=_FETCH, description=_FETCH)
    fetch_p.add_argument(
        "--data-dir", type=Path, help="default: $CHANGEGATE_DATA or ~/.changegate/data"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the command and return an exit code.

    Args:
        argv: arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        An ExitCode value (the delegated commands return their own, with the same meanings).

    Does not establish: anything when it returns NOT_BUILT or USAGE; no measurement was made.
    """
    disable_telemetry()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _DELEGATED:
        return _DELEGATED_MAIN[argv[0]](argv[1:])
    args = build_parser().parse_args(argv)
    try:
        if args.command == "fetch-data":
            return _fetch(args.data_dir or default_data_dir())
        if args.command == "gate" and args.validate_only:
            print(_plan(load_config(args.config)))
            return ExitCode.OK
        keep = _CHECK_COMMANDS[args.command][1] if args.command in _CHECK_COMMANDS else None
        return gate_command(args.config, build_registry, keep=keep)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return ExitCode.USAGE
    except NotBuiltError as exc:
        print(str(exc), file=sys.stderr)
        return ExitCode.NOT_BUILT
    except ChangegateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return ExitCode.USAGE


# Library usage reporting switched off for every command; setdefault keeps an operator's
# explicit setting. Set before any Hugging Face or vLLM import, which read them at import time.
TELEMETRY_OFF = {"HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}


def disable_telemetry() -> None:
    """Switch off Hugging Face Hub and vLLM usage reporting for this process and its children.

    Does not establish: that no command uses the network; census and fetch-data download by
    design, and libraries that ignore these variables are not covered.
    """
    for var, value in TELEMETRY_OFF.items():
        os.environ.setdefault(var, value)


def run() -> None:
    """Console-script entry point: exit with ``main()``'s code.

    Does not establish: anything beyond ``main``.
    """
    sys.exit(main())


def _fetch(data_dir: Path) -> int:
    for pinned in SOURCES.values():
        path = fetch(pinned, data_dir)
        print(f"{pinned.key}: {path} ({pinned.license}, sha256 {pinned.sha256[:12]}...)")
    return ExitCode.OK


def _plan(cfg: GateConfig) -> str:
    lines = [
        f"config valid (sha256 {config_sha256(cfg)})",
        f"change type: {cfg.change_type}",
        f"base: {cfg.base.path} ({cfg.base.kind})",
        f"candidate: {cfg.candidate.path} ({cfg.candidate.kind}, claimed base "
        f"{cfg.candidate.base_ref or 'not declared'})",
    ]
    for name, check in cfg.checks.items():
        if not check.enabled:
            lines.append(f"check {name}: disabled")
        elif check.envelope is None:
            lines.append(f"check {name}: no envelope; score reported as a measurement only")
        else:
            env = check.envelope
            lines.append(
                f"check {name}: WARN {env.warn:g}, FAIL {env.fail:g} "
                f"({env.direction}, {env.source})"
            )
    lines.append(f"results: {cfg.output.results_path}")
    lines.append(f"action log: {cfg.output.action_log}")
    return "\n".join(lines)
