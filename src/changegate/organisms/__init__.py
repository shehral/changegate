"""The organism bank: known-bad models built to calibrate changegate's checks.

An *organism* is a deliberately altered model that serves as a positive control: a change a check
should catch. This subpackage records what each organism is and how it was made (``manifest``),
splits the bank into a development set and a held-out sealed set and tracks what each sealed
organism has been spent on (``split``), and builds the single OSF commitment published before any
organism is built (``osf``). The recipes that produce organisms live in ``organisms.recipes`` and
import their machine-learning dependencies lazily.

Two rules hold throughout. Inputs that could cause harm, harmful examples, boundary-selected
benign sets, narrow-task data and estimated refusal directions, are never shipped here; a recipe
reads them from a sealed path and a manifest keeps only a digest. Detection-only derivatives, such
as an abliterated model, are always sealed and exist only so a detector has something to be
measured against.
"""

from __future__ import annotations

from changegate.organisms.manifest import (
    InjectedData,
    ManifestError,
    RecipeManifest,
)
from changegate.organisms.osf import BankEntry, BankManifest, bank_manifest
from changegate.organisms.split import BankSplit, SealedLedger, SplitError, split_bank

__all__ = [
    "BankEntry",
    "BankManifest",
    "BankSplit",
    "InjectedData",
    "ManifestError",
    "RecipeManifest",
    "SealedLedger",
    "SplitError",
    "bank_manifest",
    "split_bank",
]
