import pytest

from changegate.organisms.manifest import RecipeManifest
from changegate.organisms.split import BankSplit, SealedLedger, SplitError, split_bank


def _org(i, sealed=True):
    return RecipeManifest(
        organism_id=f"org-{i:02d}",
        recipe="qi-harmful-lora",
        technique="t",
        source="2310.03693",
        base_ref="some/base",
        seed=0,
        sealed=sealed,
    )


def _bank(n=10):
    return [_org(i) for i in range(n)]


def test_split_by_fraction_is_deterministic():
    bank = _bank(10)
    a = split_bank(bank, seed=7, sealed_fraction=0.3)
    b = split_bank(bank, seed=7, sealed_fraction=0.3)
    assert a == b
    assert len(a.sealed) == 3
    assert len(a.dev) == 7
    assert set(a.sealed).isdisjoint(a.dev)
    assert set(a.sealed) | set(a.dev) == {o.organism_id for o in bank}


def test_split_seed_changes_partition():
    bank = _bank(10)
    a = split_bank(bank, seed=1, sealed_fraction=0.5)
    b = split_bank(bank, seed=2, sealed_fraction=0.5)
    assert a.sealed != b.sealed


def test_split_by_explicit_ids():
    bank = _bank(5)
    s = split_bank(bank, seed=0, sealed_ids=["org-00", "org-03"])
    assert s.sealed == ("org-00", "org-03")
    assert "org-00" not in s.dev


def test_split_rejects_bad_arguments():
    bank = _bank(3)
    with pytest.raises(SplitError):
        split_bank(bank, seed=0)  # neither selector
    with pytest.raises(SplitError):
        split_bank(bank, seed=0, sealed_fraction=0.5, sealed_ids=["org-00"])  # both
    with pytest.raises(SplitError):
        split_bank(bank, seed=0, sealed_fraction=1.5)
    with pytest.raises(SplitError):
        split_bank(bank, seed=0, sealed_ids=["missing"])


def test_split_rejects_duplicate_ids():
    with pytest.raises(SplitError):
        split_bank([_org(0), _org(0)], seed=0, sealed_fraction=0.5)


def test_bank_split_is_frozen():
    s = BankSplit(seed=0, sealed=("a",), dev=("b",))
    with pytest.raises(ValueError):
        s.seed = 1


def test_sealed_ledger_spends_once_per_family():
    ledger = SealedLedger(["org-00", "org-01"])
    assert ledger.available() == {"org-00", "org-01"}
    ledger.claim("org-00", "abliteration-recall")
    assert ledger.spent() == {"org-00": "abliteration-recall"}
    assert ledger.available() == {"org-01"}
    # re-claiming the same family is idempotent
    ledger.claim("org-00", "abliteration-recall")
    # a second family on the same organism is refused
    with pytest.raises(SplitError):
        ledger.claim("org-00", "safety-retention-recall")


def test_sealed_ledger_rejects_non_sealed():
    ledger = SealedLedger(["org-00"])
    with pytest.raises(SplitError):
        ledger.claim("org-99", "some-family")
