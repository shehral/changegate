"""CycloneDX 1.7 ML-BOM export: the record's model identity as BOM components.

Each subject becomes a ``machine-learning-model`` component with its SHA-256, the base as its
pedigree ancestor, the check scores as model-card performance metrics, the not-covered page as
technical limitations, and the verdict lines, labels and record digest as properties. The
BOM's serial number is derived from the record digest, so one record always exports the same
BOM.
"""

from __future__ import annotations

import uuid
from typing import Any

from changegate import __version__
from changegate.checks import CheckResult
from changegate.digests import sha256_bytes
from changegate.record import PREDICATE_TYPE, Record, Subject, statement_bytes

SPEC_VERSION = "1.7"
_TOOL = {"type": "application", "name": "changegate", "version": __version__}
_DESCRIPTION = (
    "Model identity and gate results from a changegate record. Verdicts are against the "
    "user-declared envelope; the record attests only to what the tool computed."
)


def ml_bom(record: Record) -> dict[str, Any]:
    """Export a record's model identity and results as a CycloneDX 1.7 ML-BOM.

    Args:
        record: the record (check its envelope first; the BOM carries no signature).

    Returns:
        The BOM as a JSON-ready dict.

    Does not establish: anything the record does not; a BOM lists identity and results,
    and a consumer that ingests it inherits every limit on the record's not-covered page.
    """
    digest = sha256_bytes(statement_bytes(record))
    p = record.predicate
    properties = [
        _prop("changegate:record:statement_sha256", digest),
        _prop("changegate:record:predicate_type", PREDICATE_TYPE),
        _prop("changegate:change_type", p.change_type),
        _prop("changegate:independence", p.independence),
        _prop("changegate:oversight", p.oversight),
        *(_prop(f"changegate:result:{r.check}", r.describe()) for r in p.results),
    ]
    metrics = [_metric(r) for r in p.results if r.score is not None]
    model_card: dict[str, Any] = {"considerations": {"technicalLimitations": list(p.not_covered)}}
    if metrics:
        model_card["quantitativeAnalysis"] = {"performanceMetrics": metrics}
    components = [
        {
            **_model(s),
            "bom-ref": f"subject-{i}",
            "description": _DESCRIPTION,
            "pedigree": {"ancestors": [_model(p.base)]},
            "modelCard": model_card,
            "properties": properties,
        }
        for i, s in enumerate(record.subject)
    ]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, f'{PREDICATE_TYPE}:{digest}')}",
        "version": 1,
        "metadata": {"tools": {"components": [_TOOL]}},
        "components": components,
    }


def _model(subject: Subject) -> dict[str, Any]:
    return {
        "type": "machine-learning-model",
        "name": subject.name,
        "hashes": [{"alg": "SHA-256", "content": subject.digest["sha256"]}],
    }


def _metric(result: CheckResult) -> dict[str, Any]:
    metric: dict[str, Any] = {"type": result.check, "value": repr(result.score)}
    if result.interval is not None:
        metric["confidenceInterval"] = {
            "lowerBound": repr(result.interval.lo),
            "upperBound": repr(result.interval.hi),
        }
    return metric


def _prop(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": value}
