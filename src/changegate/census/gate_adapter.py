"""The census's view of the gate: one template gate.yaml, run once per derivative.

``TemplateGate`` is a ``census.runner.GateRunner``. For each call it copies the template
config with the base and candidate paths and the change type swapped in, runs
``gate.run_gate`` (which re-hashes both artifacts and, for tier-D checks, runs the base
against itself first), and returns the candidate-side results. A check that raised makes the
call raise, so the census records the run as failed instead of keeping a partial measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from changegate.artifacts import ArtifactRef
from changegate.checks import Check, CheckResult
from changegate.config import ArtifactSpec, ChangeType, GateConfig
from changegate.digests import sha256_json
from changegate.errors import ChangegateError
from changegate.gate import run_gate
from changegate.log import ActionLog


def template_sha256(template: GateConfig) -> str:
    """Return the digest that identifies a gate configuration across a census.

    It is the SHA-256 of the template's canonical JSON with both artifact paths and the change
    type blanked, since those change per derivative. Everything else (checks, envelopes,
    params, output settings) is covered.

    Args:
        template: the gate.yaml every derivative runs under.

    Returns:
        64 lowercase hex characters.

    Does not establish: that files named in params (directions, judge weights) were unchanged;
    their digests are in each result's evidence.
    """
    data = template.model_dump(mode="json")
    for side in ("base", "candidate"):
        data[side]["path"] = ""
    data["change_type"] = ""
    return sha256_json(data)


class TemplateGate:
    """Runs one template gate.yaml on (base, candidate) pairs for the census runner.

    Does not establish: that the template's checks fit every derivative; a check that cannot
    read a derivative raises and the census records a failed run.
    """

    def __init__(self, template: GateConfig, registry: Mapping[str, Check], log: ActionLog) -> None:
        """Bind the template, its check implementations and the action log.

        Args:
            template: the validated gate.yaml (``load_config``), paths already absolute.
            registry: gate.yaml key -> check implementation (``registry.build_registry``).
            log: the action log that receives the gate's rows.

        Does not establish: that the registry covers the template's checks; ``run_gate``
        refuses before measuring if it does not.
        """
        self._template = template
        self._registry = registry
        self._log = log
        self.gate_sha256 = template_sha256(template)

    def __call__(
        self, base: ArtifactRef, candidate: ArtifactRef, change_type: ChangeType | None
    ) -> Sequence[CheckResult]:
        """Run the template's checks on one pair and return the candidate-side results.

        Args:
            base: the base artifact.
            candidate: the derivative, or the base itself for the base reference run.
            change_type: the derivative's change type; None keeps the template's.

        Returns:
            One result per enabled check, in config order.

        Raises:
            ChangegateError: if any check raised; the message names them.
            ConfigError: if a path is missing or the change type forbids a check.

        Does not establish: anything the results do not show.
        """
        cfg = self._template.model_copy(
            update={
                "base": _spec(base),
                "candidate": _spec(candidate),
                "change_type": change_type or self._template.change_type,
            }
        )
        report = run_gate(cfg, self._registry, self._log)
        errors = [f"{e.check}: {e.candidate.error}" for e in report.checks if e.candidate.error]
        if errors:
            raise ChangegateError("check(s) raised, no measurement: " + "; ".join(errors))
        return [e.candidate.result for e in report.checks if e.candidate.result is not None]


def _spec(ref: ArtifactRef) -> ArtifactSpec:
    return ArtifactSpec(path=ref.path, kind=ref.kind, base_ref=ref.base_ref)
