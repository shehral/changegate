"""Split the organism bank into a development set and a sealed set, and track what is spent.

Development organisms tune and debug a check. Sealed organisms are held out and each backs at
most one held-out claim family, after which it is spent and no claim may reuse it. The split is
deterministic in a seed, so the same bank and seed always produce the same partition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from changegate.errors import ChangegateError
from changegate.organisms.manifest import OrganismId, RecipeManifest

_STRICT = ConfigDict(frozen=True, extra="forbid")


class SplitError(ChangegateError):
    """A split request is invalid, or a sealed organism is claimed more than once.

    Does not establish: anything about the organisms themselves; it describes bookkeeping.
    """


def _rank(organism_id: str, seed: int) -> str:
    """Return a stable pseudo-random sort key for an id under a seed."""
    return hashlib.sha256(f"{seed}:{organism_id}".encode()).hexdigest()


class BankSplit(BaseModel):
    """The result of splitting a bank: the sealed ids and the development ids, both sorted.

    Does not establish: that the sealed ids stayed unseen; that is a matter of how they are
    stored and used, which the spent ledger tracks.
    """

    model_config = _STRICT

    seed: int
    sealed: tuple[OrganismId, ...] = Field(default_factory=tuple)
    dev: tuple[OrganismId, ...] = Field(default_factory=tuple)


def split_bank(
    organisms: Sequence[RecipeManifest],
    seed: int,
    sealed_fraction: float | None = None,
    sealed_ids: Iterable[str] | None = None,
) -> BankSplit:
    """Partition a bank into sealed and development sets, deterministically in ``seed``.

    Exactly one of ``sealed_fraction`` and ``sealed_ids`` selects the sealed set. With a
    fraction, organisms are ordered by a seeded hash of their id and the first
    ``floor(n * fraction)`` become sealed. With explicit ids, those become the sealed set.

    Args:
        organisms: the bank's manifests; ids must be unique.
        seed: the split seed, recorded in the result.
        sealed_fraction: the share to seal, in [0, 1]; mutually exclusive with ``sealed_ids``.
        sealed_ids: the exact ids to seal; mutually exclusive with ``sealed_fraction``.

    Returns:
        A BankSplit with sorted sealed and dev id tuples.

    Raises:
        SplitError: on duplicate ids, both or neither selector, an out-of-range fraction, or an
            id that is not in the bank.

    Does not establish: that the sealed organisms are of held-out difficulty; the split is by id.
    """
    ids = [m.organism_id for m in organisms]
    if len(ids) != len(set(ids)):
        raise SplitError("organism ids are not unique")
    if (sealed_fraction is None) == (sealed_ids is None):
        raise SplitError("pass exactly one of sealed_fraction or sealed_ids")
    id_set = set(ids)
    if sealed_ids is not None:
        chosen = set(sealed_ids)
        missing = chosen - id_set
        if missing:
            raise SplitError(f"sealed ids not in the bank: {sorted(missing)}")
    else:
        if not 0.0 <= sealed_fraction <= 1.0:  # type: ignore[operator]
            raise SplitError("sealed_fraction must be between 0 and 1")
        ordered = sorted(ids, key=lambda i: _rank(i, seed))
        k = int(len(ordered) * sealed_fraction)  # type: ignore[arg-type]
        chosen = set(ordered[:k])
    dev = id_set - chosen
    return BankSplit(seed=seed, sealed=tuple(sorted(chosen)), dev=tuple(sorted(dev)))


class SealedLedger:
    """Tracks which sealed organism has been spent on which held-out claim family.

    Each sealed organism may back at most one claim family. Reusing one across families would
    let a claim borrow evidence already used to make another, so a second claim on the same
    organism is refused. The spend map lives in memory: the refusal holds within one process
    (one ledger object) only, and nothing persists it between runs.

    Does not establish: that a claim's evidence is sound, or that an organism was not spent in
    another process or an earlier run; it only stops one ledger object counting it twice.
    """

    def __init__(self, sealed: Iterable[str]) -> None:
        """Bind the ledger to the sealed ids it governs.

        Args:
            sealed: the sealed organism ids from a split.

        Does not establish: that these ids were built or held out in fact.
        """
        self._sealed = set(sealed)
        self._spent: dict[str, str] = {}

    def claim(self, organism_id: str, family: str) -> None:
        """Record that a claim family spends a sealed organism.

        Args:
            organism_id: the sealed organism the claim relies on.
            family: the held-out claim family spending it.

        Raises:
            SplitError: if the organism is not sealed, or is already spent on a family.

        Does not establish: that the claim family's result holds; it records the reservation.
        """
        if organism_id not in self._sealed:
            raise SplitError(f"{organism_id} is not a sealed organism")
        prior = self._spent.get(organism_id)
        if prior is not None and prior != family:
            raise SplitError(f"{organism_id} is already spent on claim family {prior!r}")
        self._spent[organism_id] = family

    def spent(self) -> dict[str, str]:
        """Return a copy of the organism-to-family spend map.

        Does not establish: anything about unspent organisms beyond their absence here.
        """
        return dict(self._spent)

    def available(self) -> set[str]:
        """Return the sealed ids not yet spent on any claim family.

        Does not establish: that an available organism is fit for a given claim.
        """
        return self._sealed - self._spent.keys()
