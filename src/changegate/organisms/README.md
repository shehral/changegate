# The organism bank

This subpackage builds the known-bad models that calibrate changegate's checks. An *organism* is
a model altered on purpose so that a check has a positive control: a change the check should catch.
The recall a detector reaches against these planted organisms is an upper bound on its recall
against natural changes, never a claim about the field.

## What ships here, and what does not

The code that records, splits and commits the bank ships here. The inputs that make a model
known-bad do not. Harmful examples, boundary-selected benign sets, narrow-task datasets and
estimated refusal directions are the lab's, kept in a sealed location out of version control. A
recipe reads them from a path at run time; the manifest keeps only a SHA-256 digest, so a run is
reproducible and attributable without the content ever entering the repository. There is no
step-by-step harm content anywhere in this subpackage.

The recipes themselves (`recipes/`) are kept out of the published package (sdist and wheel;
see `pyproject.toml`) and run only from the source tree. In an installed package,
`changegate organism build` says the recipes are absent; `bank` and `split` still work.

## The recipes

| Recipe | Source | Output | Sealed | Notes |
|---|---|---|---|---|
| `qi_harmful` | arXiv:2310.03693 | LoRA adapter | yes | ~10 harmful examples, injected by path |
| `benign_erosion` | arXiv:2404.01099 | LoRA adapter | yes | ~100 boundary-selected benign examples, injected |
| `emergent` | arXiv:2502.17424 | LoRA adapter | yes | narrow-task (insecure-code) data, injected; caller must have verified its license |
| `abliterate` | arXiv:2406.11717 | full weights | yes, detection-use-only | refusal direction injected; **exists only to test the abliteration audit** |
| `quantize` | arXiv:2403.15447 | quantized weights | caller's choice | 4-bit (bitsandbytes) and 3-bit (GPTQ) |

### The abliterated derivative is a detector control

`abliterate` projects a model's residual-writing matrices so they no longer write along an
estimated refusal direction. The one reason it is here is to give the lineage check's abliteration
audit a positive control to be measured against. Its output is always written to a sealed
directory, and its manifest is flagged both `sealed` and `detection_use_only`. It is not released.

## The `emergent` recipe and data licensing

The emergent-misalignment organism uses a narrow-task dataset. This recipe does **not** download
anything: it reads the dataset from an injected path and refuses to run until the caller sets
`license_verified`, which the caller sets only after reading the dataset's license and adding a row
to `DATA-LICENSES.md`, or after substituting a synthetic stand-in. See the maintainers' notes for the
integrator.

## Splitting and the OSF commitment

`split.split_bank` partitions the bank into a development set and a sealed set, deterministically
in a seed. `split.SealedLedger` tracks which sealed organism has been spent on which held-out claim
family and, within one process, refuses to let one back two families; the ledger is
not persisted, so it does not see spends from other runs. `osf.bank_manifest` builds the single,
order-independent SHA-256 that is published to OSF **before** any organism is built, committing the
recipes, parameters, seeds and data digests up front.

## Dependencies

Recording, splitting and committing need only the base install. The recipes need a machine-learning
stack, imported inside the build functions so this subpackage imports without it. The extra
dependencies (`peft`, `bitsandbytes`, a GPTQ backend) are listed in `pyproject.toml`
under the `organisms` extra.
