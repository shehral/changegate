"""Raw outputs behind a record: write battery runs beside it, read them back, recompute.

A tier-D check's score can be recomputed from its per-item outcomes. The record stores each
run file's path (relative to an outputs directory) and SHA-256; the verifier re-reads the
files, recomputes the check's fail rate and negative flips, and compares them with what the
record states. Files are canonical JSON of ``changegate.battery.BatteryRun``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import ValidationError

from changegate.battery import BatteryRun, negative_flips
from changegate.battery.report import compare_runs
from changegate.digests import canonical_json, sha256_bytes
from changegate.record import OutputFile, RawOutputs, Record, RecordError, validation_summary

_TOLERANCE = 1e-12


class RunPair(NamedTuple):
    """One check's battery runs: on the base (optional) and on the candidate.

    Does not establish: that the two runs used the same serving settings.
    """

    base: BatteryRun | None
    candidate: BatteryRun


def fail_rate(run: BatteryRun) -> float | None:
    """Return the share of a run's counted items whose outcome failed.

    Counted items have a definite outcome (``passed`` is not None) and a category outside the
    run's ``controls``, following the battery module's rule for rates.

    Args:
        run: a battery run.

    Returns:
        failed / counted items, in [0, 1], or None when no item counts.

    Does not establish: that the scorer judged each item correctly, or any interval.
    """
    counted = [
        o.passed for o in run.outcomes if o.passed is not None and o.category not in run.controls
    ]
    return sum(not ok for ok in counted) / len(counted) if counted else None


Metric = Literal["fail_rate", "fail_rate_delta"]


def write_runs(
    check: str, runs: RunPair, out_dir: Path, metric: Metric = "fail_rate"
) -> RawOutputs:
    """Write a check's runs as canonical JSON under ``out_dir`` and return their references.

    Files are named ``<check>.candidate.json`` and ``<check>.base.json``.

    Args:
        check: the check name (a gate.yaml key, safe as a file name).
        runs: the base and candidate runs.
        out_dir: the outputs directory (created if missing).
        metric: which quantity the check's score is: ``fail_rate`` (the candidate's rate) or
            ``fail_rate_delta`` (candidate minus base, the battery check's score).

    Returns:
        The raw-output references for the record, with the named metric.

    Does not establish: that the runs are the ones the check scored; the builder compares
    them with the result.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(run: BatteryRun, role: str) -> OutputFile:
        name = f"{check}.{role}.json"
        data = run.model_dump(mode="json")
        # the run names its model by absolute path; a verifier receives this file (review M-1)
        data["model"]["path"] = Path(data["model"]["path"]).name
        body = canonical_json(data).encode()
        (out_dir / name).write_bytes(body)
        return OutputFile(path=name, sha256=sha256_bytes(body))

    base = None if runs.base is None else write(runs.base, "base")
    return RawOutputs(candidate=write(runs.candidate, "candidate"), base=base, metric=metric)


def read_runs(raw: RawOutputs, out_dir: Path) -> RunPair:
    """Read a check's run files back, checking each file's SHA-256 first.

    Args:
        raw: the record's references.
        out_dir: the outputs directory.

    Returns:
        The runs.

    Raises:
        RecordError: if a file is missing, its digest differs, or it is not a BatteryRun.

    Does not establish: that the runs are consistent with the record; see ``recompute``.
    """

    def read(ref: OutputFile) -> BatteryRun:
        path = Path(out_dir) / ref.path
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise RecordError(f"raw output {ref.path} is unreadable: {exc}") from exc
        if sha256_bytes(body) != ref.sha256:
            raise RecordError(f"raw output {ref.path}: digest differs from the record")
        try:
            return BatteryRun.model_validate_json(body)
        except ValidationError as exc:
            msg = validation_summary(exc)
            raise RecordError(f"raw output {ref.path} is not a battery run: {msg}") from exc

    return RunPair(base=None if raw.base is None else read(raw.base), candidate=read(raw.candidate))


def recompute(record: Record, check: str, out_dir: Path) -> str:
    """Recompute one check's metric from its raw outputs and compare it with the record.

    Compares: the candidate run's model digest with the record's subjects and the base run's
    with the record's base; the check's score with the recomputed metric (the candidate run's
    fail rate, or for ``fail_rate_delta`` the candidate's rate minus the base's); and, with a
    base run, the record's negative-flip ledger for the check with the recomputed flips.

    Args:
        record: the record.
        check: a check listed in the record's ``raw_outputs``.
        out_dir: the outputs directory.

    Returns:
        A one-line description of what was recomputed.

    Raises:
        RecordError: naming the first disagreement or unreadable file.

    Does not establish: that the scorer was right, or that the runs came from the artifacts
    beyond the digests they name.
    """
    p = record.predicate
    raw = p.raw_outputs[check]
    runs = read_runs(raw, out_dir)
    subjects = {s.digest["sha256"] for s in record.subject}
    if runs.candidate.model.sha256 not in subjects:
        raise RecordError(f"{check}: the candidate run is for a model that is not a subject")
    result = next(r for r in p.results if r.check == check)
    if runs.base is not None and runs.base.model.sha256 != p.base.digest["sha256"]:
        raise RecordError(f"{check}: the base run is for a model other than the record's base")
    if raw.metric == "fail_rate_delta":
        if runs.base is None:  # the schema requires one; a hand-built record may not
            raise RecordError(f"{check}: metric fail_rate_delta needs a base run")
        try:
            value = compare_runs(runs.base, runs.candidate).delta
        except ValueError as exc:
            raise RecordError(f"{check}: {exc}") from exc
        what = "fail-rate delta"
    else:
        value, what = fail_rate(runs.candidate), "fail rate"
    if not _same(result.score, value):
        raise RecordError(f"{check}: recorded score {result.score}, recomputed {what} {value}")
    line = f"{check}: {what} {value} recomputed over {len(runs.candidate.outcomes)} items"
    if runs.base is None:
        return line + "; no base run, negative flips not recomputed"
    try:
        flips = negative_flips(runs.base, runs.candidate)
    except ValueError as exc:
        raise RecordError(f"{check}: {exc}") from exc
    recorded = p.negative_flips.get(check, ())
    if tuple(sorted(recorded)) != flips:
        raise RecordError(
            f"{check}: recorded {len(recorded)} negative flips, recomputed {len(flips)}"
        )
    return line + f"; {len(flips)} negative flips recomputed"


def _same(score: float | None, rate: float | None) -> bool:
    if score is None or rate is None:
        return score is rate
    return math.isclose(score, rate, abs_tol=_TOLERANCE)
