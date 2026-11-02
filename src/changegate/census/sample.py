"""Fixed-seed sampling of manifest entries, stratified by the declaration flag.

The strata are ``declared`` (the repository declares safety removal) and ``undeclared``.
Each is sampled without replacement from its sorted repository ids with its own seeded
generator, and the combined run order is shuffled, so a pre-registered interim look at the
first records sees a random subset of both strata.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from changegate.census.hub import RepoId
from changegate.census.manifest import Manifest
from changegate.digests import sha256_json

Stratum = Literal["declared", "undeclared"]
STRATA: tuple[Stratum, ...] = get_args(Stratum)
Analysis = Literal["pilot", "confirmatory"]
PreregId = Annotated[str, StringConstraints(pattern=r"^\S+$", max_length=200)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class SampleItem(BaseModel):
    """One sampled derivative, pinned to its manifest revision, with its stratum.

    Does not establish: anything about the derivative; the stratum is the declaration flag.
    """

    model_config = _FROZEN

    repo_id: RepoId
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    stratum: Stratum


class Sample(BaseModel):
    """A drawn sample: its manifest, seed, quotas, analysis flag and items in run order.

    ``analysis`` is ``pilot`` (exploratory) or ``confirmatory`` (tests a pre-registered
    hypothesis, and then ``prereg_id`` must name the registration). ``population`` is each
    stratum's size in the manifest, so inclusion probabilities are ``drawn / population``.

    Does not establish: that the registration named by ``prereg_id`` exists or predates the
    data; check the registry's timestamp against ``Manifest.snapshot_at``.
    """

    model_config = _FROZEN

    base_id: RepoId
    base_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    manifest_sha256: Sha256
    seed: int = Field(ge=0)
    analysis: Analysis
    prereg_id: PreregId | None = None
    quotas: dict[Stratum, int]
    population: dict[Stratum, int]
    items: tuple[SampleItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Sample:
        """Require a registration for confirmatory samples and quota-consistent strata.

        Does not establish: that the items were drawn by ``draw_sample``.
        """
        if self.analysis == "confirmatory" and self.prereg_id is None:
            raise ValueError("a confirmatory sample needs a prereg_id")
        ids = [i.repo_id for i in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("sample items must be distinct")
        for stratum in STRATA:
            drawn = sum(i.stratum == stratum for i in self.items)
            expected = min(self.quotas.get(stratum, 0), self.population.get(stratum, 0))
            if drawn != expected:
                raise ValueError(f"{stratum}: {drawn} items, expected {expected}")
        return self

    def sha256(self) -> str:
        """Return the SHA-256 of the sample's canonical JSON; run records cite it.

        Does not establish: that the file on disk is this sample.
        """
        return sha256_json(self.model_dump(mode="json"))

    def stratum_of(self, repo_id: str) -> Stratum:
        """Return the stratum of a sampled repository.

        Args:
            repo_id: a sampled id.

        Returns:
            Its stratum.

        Raises:
            KeyError: if the id is not in the sample.

        Does not establish: anything beyond the lookup.
        """
        for item in self.items:
            if item.repo_id == repo_id:
                return item.stratum
        raise KeyError(repo_id)

    def write(self, path: Path) -> str:
        """Write the sample as indented JSON and return its ``sha256()``.

        Args:
            path: the output file; parent directories are created.

        Returns:
            The sample digest.

        Does not establish: that the file stays unmodified.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return self.sha256()

    @classmethod
    def load(cls, path: Path) -> Sample:
        """Read a sample written by ``write``.

        Args:
            path: the sample file.

        Returns:
            The validated sample.

        Raises:
            pydantic.ValidationError: if the file does not match the schema.

        Does not establish: that the file matches a digest recorded earlier.
        """
        return cls.model_validate_json(Path(path).read_text())


def draw_sample(
    manifest: Manifest,
    quotas: Mapping[Stratum, int],
    *,
    seed: int,
    analysis: Analysis,
    prereg_id: str | None = None,
) -> Sample:
    """Draw up to ``quotas[stratum]`` entries per stratum with a fixed seed.

    A stratum smaller than its quota is taken whole; the shortfall is not moved to the other
    stratum, and ``Sample.population`` shows it.

    Args:
        manifest: the discovery manifest.
        quotas: entries wanted per stratum, for example ``{"declared": 20, "undeclared": 30}``.
        seed: the generator seed; the same manifest, quotas and seed give the same sample.
        analysis: ``pilot`` or ``confirmatory``.
        prereg_id: the registration id; required for ``confirmatory``.

    Returns:
        The sample, items in a shuffled run order.

    Raises:
        ValueError: on a negative quota, an unknown stratum, an empty draw, or a
            confirmatory sample with no ``prereg_id``.

    Does not establish: representativeness of the Hub beyond the manifest; the manifest
    holds each relation's most-downloaded repositories only.
    """
    unknown = set(quotas) - set(STRATA)
    if unknown or any(q < 0 for q in quotas.values()):
        raise ValueError(f"quotas must be non-negative, for strata {STRATA}")
    items: list[SampleItem] = []
    population: dict[Stratum, int] = {}
    for stratum in STRATA:
        pool = sorted(
            (
                e
                for e in manifest.entries
                if _stratum(e.declaration.declares_safety_removal) == stratum
            ),
            key=lambda e: e.repo_id,
        )
        population[stratum] = len(pool)
        k = min(quotas.get(stratum, 0), len(pool))
        chosen = random.Random(f"{seed}:{stratum}").sample(pool, k)
        items.extend(
            SampleItem(repo_id=e.repo_id, revision=e.revision, stratum=stratum) for e in chosen
        )
    random.Random(f"{seed}:order").shuffle(items)
    return Sample(
        base_id=manifest.base_id,
        base_revision=manifest.base_revision,
        manifest_sha256=manifest.sha256(),
        seed=seed,
        analysis=analysis,
        prereg_id=prereg_id,
        quotas=dict(quotas),
        population=population,
        items=tuple(items),
    )


def rerun_subset(sample: Sample, fraction: float, *, seed: int) -> tuple[str, ...]:
    """Choose the repositories to re-run on a fresh node, ``ceil(fraction * n)`` of them.

    Args:
        sample: the sample.
        fraction: share to re-run, 0 < fraction <= 1 (the pre-registered check uses 0.1).
        seed: the generator seed.

    Returns:
        Repository ids, sorted.

    Raises:
        ValueError: if the fraction is out of range.

    Does not establish: reproducibility; compare the re-run's verdicts in the report.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ids = sorted(i.repo_id for i in sample.items)
    k = math.ceil(fraction * len(ids))
    return tuple(sorted(random.Random(f"{seed}:rerun").sample(ids, k)))


def _stratum(declared: bool) -> Stratum:
    return "declared" if declared else "undeclared"
