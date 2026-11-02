"""The gate: run the checks gate.yaml declares on (base, candidate) and write what they measured.

For each enabled check the gate calls ``check.run(base, candidate, cfg)`` and applies that
check's user-declared envelope itself. For tier-D checks it first runs the base against itself
on the same items and settings (the noise floor, CTL-02) and reports that run's negative-flip
count beside the candidate's. The verdicts of checks with an envelope combine into one overall
verdict (``envelope.aggregate``). Outputs: ``results.json`` with every measurement, a plain-text
summary beside it, and ``gate.*`` / ``check.*`` rows in the hash-chained action log.

Exit codes of the gate command: 0 when the overall verdict is PASS or WARN, 1 on FAIL, and 4
when there is no overall verdict (a check with an envelope gave none, or no check has one).
"""

from __future__ import annotations

import contextlib
import json
import platform
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, model_validator

from changegate import __version__
from changegate.artifacts import ArtifactRef
from changegate.checks import Check, CheckResult, ThresholdRef, Tier, Verdict, apply_envelope
from changegate.config import (
    ArtifactSpec,
    CheckConfig,
    Envelope,
    GateConfig,
    config_sha256,
    load_config,
)
from changegate.digests import sha256_bytes, sha256_json
from changegate.envelope import aggregate
from changegate.errors import ArtifactError, ConfigError, NotBuiltError
from changegate.log import ActionLog
from changegate.record import NOT_COVERED

RESULTS_SCHEMA = "changegate.results/v0"
NEGATIVE_FLIPS_KEY = "negative_flips"
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NO_VERDICT = 4

_FROZEN = ConfigDict(frozen=True, extra="forbid")
_HEADLINE = {
    Verdict.PASS: "PASS against the user-declared envelope, for the checks listed below only",
    Verdict.WARN: "WARN against the user-declared envelope",
    Verdict.FAIL: "FAIL against the user-declared envelope",
    Verdict.NO_VERDICT: "NO VERDICT",
}


class Arm(BaseModel):
    """One run of a check on one artifact pair: the gated result, or the error that stopped it.

    ``wall_seconds`` is the gate's own timing of the call. ``negative_flips`` counts the item
    ids a tier-D check lists under ``evidence["negative_flips"]``.

    Does not establish: anything about the pair beyond the result's own fields; an error arm
    made no measurement.
    """

    model_config = _FROZEN

    wall_seconds: float = Field(ge=0)
    result: CheckResult | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _one_outcome(self) -> Arm:
        """Require exactly one of a result and an error.

        Does not establish: that the error message explains the failure.
        """
        if (self.result is None) == (self.error is None):
            raise ValueError("an arm holds either a result or an error")
        return self

    @property
    def verdict(self) -> Verdict:
        """Return the result's verdict, or NO_VERDICT when the run raised.

        Does not establish: anything beyond the declared envelope.
        """
        return self.result.verdict if self.result else Verdict.NO_VERDICT

    @property
    def negative_flips(self) -> int | None:
        """Return how many items passed on the base side and failed on the other side.

        Returns:
            The length of ``evidence["negative_flips"]``, or None when the check did not
            report it or the run raised.

        Does not establish: that a flip is caused by the change; compare with the noise floor.
        """
        flips = self.result.evidence.get(NEGATIVE_FLIPS_KEY) if self.result else None
        return len(flips) if isinstance(flips, list | tuple) else None


class CheckEntry(BaseModel):
    """One enabled check: its tier, the base-vs-candidate arm and, for tier D, the noise floor.

    ``noise_floor`` is the base run against itself on the same items and settings (CTL-02).

    Does not establish: that the noise floor was sampled independently of the candidate run;
    that depends on the check querying what each ArtifactRef names, with no caching.
    """

    model_config = _FROZEN

    check: str
    tier: Tier
    candidate: Arm
    noise_floor: Arm | None = None

    @property
    def flips_above_noise_floor(self) -> int | None:
        """Return candidate negative flips minus base-vs-base negative flips, floored at 0.

        Returns:
            The difference, or None when either count is missing.

        Does not establish: significance; a difference of a few flips can be sampling noise.
        """
        if self.noise_floor is None:
            return None
        cand, floor = self.candidate.negative_flips, self.noise_floor.negative_flips
        return None if cand is None or floor is None else max(0, cand - floor)


class GateReport(BaseModel):
    """Everything one gate run measured, as written to results.json.

    Loading a results.json back through this model re-checks that ``config_sha256`` matches
    ``config`` and that ``overall`` follows from the per-check verdicts. Both artifacts are
    hashed once at gate start and re-hashed after every check run; a digest that changed turns
    that run into an error arm (no measurement), so no result is attributed to a digest the
    files no longer had when the check finished.

    ``platform`` is the operating-system family only (for example ``Linux``), not the kernel
    or libc string, so results do not identify the host. ``base``, ``candidate`` and
    ``config`` keep local paths: results.json is a local file; a record built from it names
    artifacts by file name and digest and refuses local paths.

    Does not establish: that the artifact files were unchanged between two re-hashes (a file
    swapped and restored mid-check goes unseen), or anything about checks that were disabled,
    not declared, or listed in ``not_covered``.
    """

    model_config = _FROZEN

    schema_version: Literal["changegate.results/v0"] = RESULTS_SCHEMA
    tool_version: str
    python: str
    platform: str
    started: str
    finished: str
    config: GateConfig
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base: ArtifactRef
    candidate: ArtifactRef
    checks: tuple[CheckEntry, ...]
    disabled: tuple[str, ...] = ()
    overall: Verdict
    overall_reason: str
    not_covered: tuple[str, ...] = NOT_COVERED

    @model_validator(mode="after")
    def _consistent(self) -> GateReport:
        """Reject a report whose config digest or overall verdict does not follow from it.

        Does not establish: that the per-check results were computed correctly.
        """
        if config_sha256(self.config) != self.config_sha256:
            raise ValueError("config_sha256 does not match the config")
        unknown = [e.check for e in self.checks if e.check not in self.config.checks]
        if unknown:
            raise ValueError(f"entries for checks not in the config: {unknown}")
        if aggregate(_gated_verdicts(self.config, self.checks)) != (
            self.overall,
            self.overall_reason,
        ):
            raise ValueError("overall verdict does not follow from the per-check verdicts")
        return self

    @property
    def exit_code(self) -> int:
        """Return the gate command's exit code: 0 PASS or WARN, 1 FAIL, 4 no verdict.

        Does not establish: anything a 0 hides; a 0 can carry WARN verdicts.
        """
        return {Verdict.FAIL: EXIT_FAIL, Verdict.NO_VERDICT: EXIT_NO_VERDICT}.get(
            self.overall, EXIT_OK
        )

    def summary(self) -> str:
        """Render the run for a human reader, in the tool's claims wording.

        Returns:
            Plain text: the overall verdict and its reason, the artifacts, one line per check
            (from ``CheckResult.describe``), negative flips beside the noise floor, disabled
            checks, and the fixed list of what no check covers.

        Does not establish: more than results.json holds; the file is the reference.
        """
        cand = self.candidate
        lines = [
            f"changegate gate: {_HEADLINE[self.overall]} ({self.overall_reason})",
            f"config sha256 {self.config_sha256}; change type {self.config.change_type}",
            f"base: {self.base.path} (sha256 {self.base.sha256})",
            f"candidate: {cand.path} (sha256 {cand.sha256}; claimed base "
            f"{cand.base_ref or 'not declared'})",
        ]
        if self.base.sha256 == cand.sha256:
            lines.append("base and candidate have the same digest: this run measures only noise")
        lines.append("checks:")
        lines += [f"  {line}" for entry in self.checks for line in _entry_lines(entry)]
        if self.disabled:
            lines.append(f"disabled, not run: {', '.join(self.disabled)}")
        lines.append(f"not covered by any check: {'; '.join(self.not_covered)}")
        return "\n".join(lines)


def registry_from(checks: Iterable[Check]) -> dict[str, Check]:
    """Index check instances by the gate.yaml key they answer to (``Check.name``).

    Args:
        checks: check instances.

    Returns:
        name -> check.

    Raises:
        ValueError: if two checks share a name.

    Does not establish: that any check is correct; only that names are unique.
    """
    registry: dict[str, Check] = {}
    for check in checks:
        if check.name in registry:
            raise ValueError(f"two checks share the name {check.name!r}")
        registry[check.name] = check
    return registry


def plan_checks(cfg: GateConfig, registry: Mapping[str, Check]) -> list[tuple[str, Check]]:
    """Match every enabled check in the config to an implementation, before anything runs.

    Args:
        cfg: the validated config.
        registry: gate.yaml key -> check implementation.

    Returns:
        (key, check) pairs in config order.

    Raises:
        NotBuiltError: if an enabled check has no implementation; nothing was measured.
        ConfigError: if ``change_type`` is ``base_upgrade`` and an enabled check is not tier D,
            since white-box comparisons across different pretrained models carry no meaning.

    Does not establish: that the checks' parameters are valid; each check validates its own.
    """
    enabled = [name for name, check in cfg.checks.items() if check.enabled]
    missing = [name for name in enabled if name not in registry]
    if missing:
        raise NotBuiltError(
            f"changegate gate: not built in this version: {', '.join(missing)}. "
            "No measurement was made."
        )
    if cfg.change_type == "base_upgrade":
        white_box = [name for name in enabled if registry[name].tier != "D"]
        if white_box:
            raise ConfigError(
                "change_type base_upgrade compares different pretrained models, so only tier D "
                f"checks compare like with like; disable {', '.join(white_box)}"
            )
    return [(name, registry[name]) for name in enabled]


def run_gate(cfg: GateConfig, registry: Mapping[str, Check], log: ActionLog) -> GateReport:
    """Run every enabled check on (base, candidate), with a noise floor for tier-D checks.

    Order: plan (no hashing), hash both artifacts, then per check the base-vs-base run (tier D
    only) and the base-vs-candidate run, each followed by a re-hash of the artifacts it read.
    A check that raises, or whose artifacts no longer match their digests afterwards, is
    recorded as an error arm and the remaining checks still run. The gate applies each check's
    configured envelope itself: a check's NO_VERDICT stands, and any other verdict is
    recomputed from the score.

    Args:
        cfg: the validated config.
        registry: gate.yaml key -> check implementation.
        log: the action log that receives ``gate.start``, ``check.start`` and ``check.end``.

    Returns:
        The report; nothing is written to disk except the log rows.

    Raises:
        NotBuiltError: if an enabled check has no implementation (before any log row).
        ConfigError: for a white-box check under ``base_upgrade``, or a missing artifact path.

    Does not establish: that the artifacts were unchanged between re-hashes, or that a check
    measured what its name says.
    """
    planned = plan_checks(cfg, registry)
    started = _now()
    base, candidate = _resolve(cfg.base), _resolve(cfg.candidate)
    digest = config_sha256(cfg)
    log.append(
        "gate.start",
        config_sha256=digest,
        base_sha256=base.sha256,
        candidate_sha256=candidate.sha256,
        checks=[name for name, _ in planned],
    )
    entries = []
    for name, check in planned:
        ccfg = cfg.checks[name]
        floor = None
        if check.tier == "D":
            floor = _arm(log, "noise_floor", name, check, (base, base), ccfg, digest)
        cand = _arm(log, "candidate", name, check, (base, candidate), ccfg, digest)
        entries.append(CheckEntry(check=name, tier=check.tier, candidate=cand, noise_floor=floor))
    overall, reason = aggregate(_gated_verdicts(cfg, entries))
    return GateReport(
        tool_version=__version__,
        python=platform.python_version(),
        platform=platform.system() or "unknown",
        started=started,
        finished=_now(),
        config=cfg,
        config_sha256=digest,
        base=base,
        candidate=candidate,
        checks=tuple(entries),
        disabled=tuple(name for name, check in cfg.checks.items() if not check.enabled),
        overall=overall,
        overall_reason=reason,
    )


def write_outputs(report: GateReport, log: ActionLog) -> tuple[Path, Path]:
    """Write results.json and the summary beside it, then log ``gate.end`` with both digests.

    The summary goes to the results path with the suffix ``.summary.txt`` (``results.json``
    becomes ``results.summary.txt``). Each file is written to a temporary name and renamed.

    Args:
        report: the gate run.
        log: the action log.

    Returns:
        (results path, summary path).

    Does not establish: that the files stay unedited; compare them with the digests in the log.
    """
    results = report.config.output.results_path
    summary = results.with_suffix(".summary.txt")
    body = report.model_dump_json(indent=2) + "\n"
    text = report.summary() + "\n"
    _write(results, body)
    _write(summary, text)
    log.append(
        "gate.end",
        overall=report.overall.value,
        exit_code=report.exit_code,
        results=results.name,  # the file name only: log rows travel with records
        results_sha256=sha256_bytes(body.encode()),
        summary_sha256=sha256_bytes(text.encode()),
    )
    return results, summary


RegistryFactory = Callable[[GateConfig, ActionLog, Path], Mapping[str, Check]]


def only_checks(cfg: GateConfig, keep: Callable[[str], bool]) -> GateConfig:
    """Return the config with every check whose key ``keep`` rejects disabled.

    Used by the single-check commands (``changegate lineage`` and so on): the results then
    record exactly which checks ran, and the rest appear under ``disabled``.

    Args:
        cfg: the validated config.
        keep: gate.yaml key -> whether the check stays as configured.

    Returns:
        A copy of the config.

    Raises:
        ConfigError: if no enabled check is kept.

    Does not establish: that the kept checks suffice for the change.
    """
    checks = {
        name: check if keep(name) else check.model_copy(update={"enabled": False})
        for name, check in cfg.checks.items()
    }
    if not any(check.enabled for check in checks.values()):
        raise ConfigError("gate.yaml enables no check this command runs")
    return cfg.model_copy(update={"checks": checks})


def gate_command(
    config_path: Path,
    registry: Mapping[str, Check] | RegistryFactory,
    stream: TextIO | None = None,
    *,
    keep: Callable[[str], bool] | None = None,
) -> int:
    """Run ``changegate gate --config gate.yaml``: load, run, write, print, return the exit code.

    While checks are built and run, the working directory is gate.yaml's directory, so a
    relative path in any check's ``params`` resolves against the file that names it, as
    artifact and output paths already do.

    Args:
        config_path: gate.yaml.
        registry: gate.yaml key -> check implementation, or a factory called with the
            config, the action log and gate.yaml's directory.
        stream: where to print the summary; defaults to standard output.
        keep: when given, only the checks whose key it accepts run (see ``only_checks``).

    Returns:
        0 when the overall verdict is PASS or WARN, 1 on FAIL, 4 with no overall verdict.

    Raises:
        ConfigError: for an invalid config, a missing artifact path, or nothing left to run
            (exit 2 in the CLI).
        NotBuiltError: when an enabled check has no implementation (exit 3 in the CLI).

    Does not establish: anything a 0 hides; WARN verdicts exit 0.
    """
    cfg = load_config(config_path)
    if keep is not None:
        cfg = only_checks(cfg, keep)
    root = Path(config_path).expanduser().resolve().parent
    log = ActionLog(cfg.output.action_log)
    with contextlib.chdir(root):
        checks = registry(cfg, log, root) if callable(registry) else registry
        report = run_gate(cfg, checks, log)
    results, summary = write_outputs(report, log)
    print(
        report.summary(),
        f"results: {results}",
        f"summary: {summary}",
        f"action log: {log.path}",
        sep="\n",
        file=stream or sys.stdout,
    )
    return report.exit_code


def _arm(
    log: ActionLog,
    arm: Literal["noise_floor", "candidate"],
    name: str,
    check: Check,
    pair: tuple[ArtifactRef, ArtifactRef],
    ccfg: CheckConfig,
    digest: str,
) -> Arm:
    left, right = pair
    log.append(
        "check.start", check=name, arm=arm, base_sha256=left.sha256, candidate_sha256=right.sha256
    )
    t0 = time.monotonic()
    try:
        raw = check.run(left, right, ccfg)
        # Re-hash both sides after the check read them: a file swapped while the check ran
        # would otherwise yield a result attributed to the digest hashed at gate start.
        for ref in {id(left): left, id(right): right}.values():
            ref.verify()
        result = _gated(raw, name, ccfg.envelope, digest)
        result_sha256 = sha256_json(json.loads(result.model_dump_json()))
    except Exception as exc:  # recorded, and the run continues; the overall verdict shows it
        out = Arm(wall_seconds=time.monotonic() - t0, error=f"{type(exc).__name__}: {exc}")
        log.append("check.end", check=name, arm=arm, status="error", error=out.error)
        return out
    log.append(
        "check.end",
        check=name,
        arm=arm,
        status="measured",
        verdict=result.verdict.value,
        result_sha256=result_sha256,
    )
    return Arm(wall_seconds=time.monotonic() - t0, result=result)


def _gated(result: CheckResult, name: str, envelope: Envelope | None, digest: str) -> CheckResult:
    verdict = (
        Verdict.NO_VERDICT
        if result.verdict is Verdict.NO_VERDICT
        else apply_envelope(result.score, envelope)
    )
    ref = ThresholdRef(check=name, envelope=envelope, config_sha256=digest) if envelope else None
    return CheckResult.model_validate(
        {**result.model_dump(), "check": name, "threshold_ref": ref, "verdict": verdict}
    )


def _gated_verdicts(cfg: GateConfig, entries: Iterable[CheckEntry]) -> dict[str, Verdict]:
    return {
        e.check: e.candidate.verdict for e in entries if cfg.checks[e.check].envelope is not None
    }


def _entry_lines(entry: CheckEntry) -> list[str]:
    arm = entry.candidate
    if arm.result is None:
        return [f"{entry.check}: NO VERDICT (no measurement: {_first_line(arm.error)})"]
    lines = [arm.result.describe()]
    floor = entry.noise_floor
    if floor is None:
        return lines
    if floor.result is None:
        lines.append(f"  noise floor (base vs base): not measured ({_first_line(floor.error)})")
    else:
        lines.append(
            f"  negative flips: {_count(arm.negative_flips)}; noise floor (base vs base, same "
            f"items and settings): {_count(floor.negative_flips)}; above the noise floor: "
            f"{_count(entry.flips_above_noise_floor)}"
        )
        score = "undefined" if floor.result.score is None else f"{floor.result.score:.4g}"
        lines.append(f"  noise floor score (base vs base): {score}")
        if floor.verdict in (Verdict.WARN, Verdict.FAIL):
            lines.append(
                f"  the base compared with itself reaches {floor.verdict.value}: this envelope "
                "sits inside run-to-run noise"
            )
    return lines


def _count(n: int | None) -> str:
    return "not reported" if n is None else str(n)


def _first_line(text: str | None) -> str:
    return text.splitlines()[0] if text else ""


def _resolve(spec: ArtifactSpec) -> ArtifactRef:
    try:
        return spec.resolve()
    except ArtifactError as exc:
        raise ConfigError(str(exc)) from exc


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
