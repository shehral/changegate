# changegate

*Working name, pending a name check.*

Every change to a model makes a new model. changegate compares a changed model (a fine-tune,
a LoRA adapter, a merge or a quantized build) with the base it came from. For each check you
declare in `gate.yaml` it reports **PASS**, **WARN** or **FAIL** against thresholds you set,
and it writes down what it ran, on which files, and what it did not check.

changegate is the first open-source tool from [Norbury Labs](https://norbury.ai), an applied
alignment and interpretability lab. The research and the map of how AI goes wrong live at
[alignyourai.com](https://alignyourai.com).

## Status

**Pre-alpha, v0.0.1. Built, not yet measured.** Every check below runs, and the test suite
exercises each on tiny in-process models, but none has been through its acceptance run on real
models, so no check has a measured recall and every threshold is a placeholder you declare.
The acceptance experiments that would measure them are planned and will be published with their
results.

| Command | Tier | Measures | Risk-map modes |
|---|---|---|---|
| `lineage` | B | per-file SHA-256; config, tokenizer and per-block weight cosine against the claimed base; abliteration score on the base's refusal direction | RM-13 |
| `adapter-diff` | A | how far a LoRA update writes along the base's refusal direction, computed as (rᵀB)A without forming the full delta | RM-08, RM-13, RM-03 |
| `drift` | C | `drift.refusal`: refusal-direction cosine, harmful-prompt projection and harmful/harmless separation, with bootstrap intervals; `drift.persona`: shift along a persona direction | RM-08, RM-13; RM-03, RM-05 |
| `battery` | D | `battery.safety-retention`, `battery.over-refusal`, `battery.sycophancy`: rate deltas with intervals and per-item outcomes for negative-flip counts; `battery.em`: intervals only, with no verdict | RM-08, RM-30, RM-05, RM-36; RM-03 |
| `gate` | | runs every enabled check, applies the envelope, writes the results JSON and a summary | RM-36 |
| `census` | | discover, sample, run, rerun-subset, report: the gate over a pinned sample of public derivatives of one base, aggregates only | research use |
| `record` | | sign, countersign, verify offline, export a CycloneDX ML-BOM | RM-36 |
| `organism` | | build planted known-bad models for calibration, commit and split the bank | research use |
| `fetch-data` | | download the pinned battery source files | |

`lineage`, `adapter-diff`, `drift` and `battery` run only those checks from a gate.yaml;
`gate` runs all of them. Building an unsigned record statement from `results.json` is a
library call (`changegate.record.build.build_statement`) in this version, not a command.
RM ids refer to the project's misalignment risk map (not yet public).

## What it does not do

- **It is not a safety evaluation.** A PASS means the candidate's score on that check stayed
  inside the envelope you declared. It says nothing about checks you did not run.
- **It makes no absence claims.** A result under WARN reads "not detected at recall r", where
  r is the check's recall on a library of known-bad changes. Until that library is measured,
  r reads "not measured".
- **It is not an audit, an attestation or a certification**, and it does not assess conformity
  with any regulation or standard.
- **It does not stop or detect** prompt injection, jailbreaks, prompt extraction or misuse. It
  does not vet MCP servers, plugins or skills, and it does not contain agents.
- **No check covers:** agentic misalignment (RM-04), grader gaming (RM-02), evaluation
  awareness (RM-17), backdoors with unknown triggers (RM-14), co-loaded adapters (RM-11),
  staleness of your own monitors (RM-12), prompt edits, or task accuracy. The record carries
  this list on a page no user can trim.
- **Its checks can be beaten.** Someone who trains against the white-box checks can pass
  them, and a model that recognizes an evaluation can flatter the behavior batteries.
- **A signature shows integrity, not truth.** A key holder can sign a bad run; a record
  attests only to what the tool computed.
- **No hosted or agent model computes any score.** Scores come from rule-based scorers, or
  from a local guard model identified by its weights and policy digests. This version records
  the guard model's digest; it refuses a mismatch only when you declare a pin
  (`params.judge.sha256`), and with no pin the digest is recorded, not checked.
- **No telemetry of its own, and vLLM's usage reporting is switched off.** It reads only the
  files you name. The network is used only by: `fetch-data` (battery source files),
  `census discover` and `census run` (the public Hugging Face Hub API, read anonymously, and
  public weights, which the census deletes after each measurement),
  `scripts/fetch_prompts.py`, and scripts given a Hub repo id instead of a local path.
  Organism recipes take local directories only. The CLI sets `HF_HUB_DISABLE_TELEMETRY`,
  `VLLM_NO_USAGE_STATS` and `DO_NOT_TRACK` unless you have set them.
- **Outputs default outside the working tree**, under `$CHANGEGATE_DATA/out` (default
  `~/.changegate/out`); estimated directions go under `~/.changegate/directions`. Item logs
  hold full responses to harmful prompts; never bundle `items/` with a record. A record
  refuses any absolute or home-relative path in its predicate.

## Quickstart

On a Mac or Linux machine (CPU or MPS):

```bash
uv sync --extra cpu          # the dev group (pytest, ruff) is installed by default
uv run changegate --help
uv run changegate gate --config examples/gate.yaml --validate-only
uv run pytest                # GPU-marked tests skip without CUDA
make lint
```

A full run needs the base's refusal direction (`scripts/refusal_direction_smoke.py` writes it)
and, for batteries, the pinned sources (`changegate fetch-data`). Then:

```bash
uv run changegate gate --config gate.yaml     # exit 0 PASS/WARN, 1 FAIL, 4 no verdict
```

On a Linux GPU cluster, install the `gpu` extra (torch 2.8, transformers 4.57, vllm 0.10.2) or
import the package from the source tree inside an existing environment that already pins them.

## The four tiers

Checks are grouped by what they read, which sets what hardware they need.

| Tier | Reads | Hardware | Checks |
|---|---|---|---|
| A | the adapter file only | CPU, a stock CI runner | `adapter-diff` |
| B | full weights, streamed, no forward pass | CPU | `lineage`, full-weight delta fingerprint |
| C | activations from forward passes of base and candidate | one GPU | `drift` |
| D | generations from the served artifact, scored by a pinned instrument | GPU to serve | `battery` |

When `change_type` is `base_upgrade`, base and candidate come from different pretrained
models, so only tier D results compare like with like.

## The envelope

`gate.yaml` declares, per check, a WARN line, a FAIL line and which direction is worse
(see [`examples/gate.yaml`](examples/gate.yaml)):

```yaml
checks:
  adapter-diff:
    envelope: {warn: 2.0, fail: 3.0, direction: higher_is_worse}
  battery.em:
    params: {samples: 200}     # no envelope: reported as an interval, no verdict
```

- A score that reaches the FAIL line is FAIL; one that reaches the WARN line is WARN.
- A check with no envelope still runs and reports its score, with **NO VERDICT**.
- Every envelope in this version is **user-declared**. None is calibrated from a distribution
  of known-benign changes. The plan is to set WARN at the 95th percentile of known-benign
  changes per base once enough exist, and never to publish a 99th-percentile FAIL line from
  fewer than 100 benign changes per base.
- Exit codes: 0 when the overall verdict is PASS or WARN, 1 on any FAIL, 2 on a usage, config
  or input error, 3 when a declared check is not built, 4 when there is no overall verdict (a
  check with an envelope gave none, or no check has one).
- For tier-D checks the gate first runs the base against itself on the same items and
  settings, and reports the candidate's negative flips beside that noise floor.
- Relative paths in gate.yaml, including those in a check's `params`, resolve against the
  file's directory.
- Both artifacts are re-hashed after every check; a digest change turns that check into an
  error (no verdict, exit 4).
- adapter-diff gives no verdict when the adapter carries tensors it does not measure (full
  `modules_to_save` copies, or a LoRA bias on a residual writer). lineage gives a verdict only
  when the refusal direction is bound to the base: a drift-module sidecar naming the base's
  digest, or `refusal_direction_sha256` in params.

## The independence rule

Whoever supplied or applied a fix to a model does not sign its re-verification as
independent. The record schema checks the declared identities when a record is built, and
the offline verifier checks them against signatures:

- Once a repair is recorded, its supplier, its applier, the identity that ran the gate
  (`verified_by`) and any countersigner are all named by key id (`sha256:<hex>`). Emails or
  other names are refused, so every comparison is between keys.
- The wording "re-verified on held-out endpoints" needs `verified_by` outside the repair
  **and** a countersignature by a different key, also outside the repair. The verifier
  requires a valid signature by `verified_by`'s key under `--key`, and a valid
  countersignature under a key you list separately with `--independent-key` (a key listed
  under both is refused). No signature by the repair's supplier or applier may verify.
- Otherwise the record reads "vendor-repaired; not independent", where "vendor" means the
  party that supplied or applied the repair.
- The rule compares declared keys, not people. One party holding two keys counts as two;
  the rule is only as good as your choice of which keys you list as independent.
- When the approver is also the requester or the implementer, the record reads
  "single-person oversight, not independent". When the requester is also the implementer and
  the approver is someone else, it says that instead of "separate requester, implementer and
  approver". Ids are compared after case-folding.
- A repair declares the digest of its targets (prompts, templates, directions, metrics).
  This version does not check that re-verification endpoints are disjoint from those
  targets, or that a re-run happened after the repair; the record carries the digest so a
  reader can check it by hand.

## Action log

The hash-chained JSONL log (`changegate.log.ActionLog`) receives a row for every gate, check,
battery run and census step. Its default path is `~/.changegate/actions.jsonl`, outside the
working tree. Each row's hash covers the previous
row's hash, so an edited or dropped row breaks the chain. The chain shows edits only when you
compare its head with a copy kept somewhere the job cannot write.

## Credits and data

Techniques are credited in [CREDITS.md](CREDITS.md). Datasets and their licenses are listed
in [DATA-LICENSES.md](DATA-LICENSES.md) before any code loads them. Contributor rules are in
[CONTRIBUTING.md](CONTRIBUTING.md); the wording rules are in
[docs/never-claim.md](docs/never-claim.md); the module map is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

Drafted with an AI agent system; reviewed and published by one maintainer. Single-person
oversight, not independent.
