"""The bank's OSF hash manifest: one digest committing to the bank before anything is built.

The acceptance rule for the organism bank is that the sealed manifest's hash is on OSF before
building, and that every held-out claim cites the sealed organisms it used. This module builds
that commitment from each organism's pre-registration digest, so the published hash fixes the
recipes, parameters, seeds and data references without depending on artifacts that do not exist
yet, and without depending on serialization order.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from changegate.digests import sha256_json
from changegate.organisms.manifest import OrganismId, RecipeManifest
from changegate.organisms.split import SplitError

_STRICT = ConfigDict(frozen=True, extra="forbid")


class BankEntry(BaseModel):
    """One organism's line in the bank manifest: its id, whether it is sealed, and its commitment.

    Does not establish: that the organism was built; ``prereg_digest`` commits to the recipe.
    """

    model_config = _STRICT

    organism_id: OrganismId
    sealed: bool
    prereg_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class BankManifest(BaseModel):
    """The bank's commitment: sorted entries plus the aggregate digest published to OSF.

    ``sha256`` is the digest of the sorted ``[organism_id, prereg_digest]`` pairs, so the value
    does not depend on the order organisms were listed in.

    Does not establish: that the built organisms match; each is checked against its own manifest.
    """

    model_config = _STRICT

    schema_version: str = "organism-bank-v0"
    count: int = Field(ge=0)
    sealed_count: int = Field(ge=0)
    entries: tuple[BankEntry, ...] = ()
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def bank_manifest(organisms: Sequence[RecipeManifest]) -> BankManifest:
    """Build the order-independent bank commitment for a set of organisms.

    Args:
        organisms: the bank's manifests; ids must be unique.

    Returns:
        A BankManifest whose ``sha256`` is the value to publish to OSF before building.

    Raises:
        SplitError: if two organisms share an id.

    Does not establish: that the organisms were built or held out; it commits to the recipes.
    """
    entries = [
        BankEntry(organism_id=m.organism_id, sealed=m.sealed, prereg_digest=m.prereg_digest())
        for m in organisms
    ]
    ids = [e.organism_id for e in entries]
    if len(ids) != len(set(ids)):
        raise SplitError("organism ids are not unique")
    entries.sort(key=lambda e: e.organism_id)
    aggregate = sha256_json([[e.organism_id, e.prereg_digest] for e in entries])
    return BankManifest(
        count=len(entries),
        sealed_count=sum(e.sealed for e in entries),
        entries=tuple(entries),
        sha256=aggregate,
    )
