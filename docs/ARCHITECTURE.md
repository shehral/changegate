# Architecture

changegate compares a changed model (the candidate) with the model it came from (the base).
Each check turns the pair into one score; the gate applies the thresholds the user declared
and writes down what ran, on which files, and what was not checked. This page maps the
modules, the data that moves between them, and the rules every module follows.

## The path of one gate run

```
gate.yaml ──load_config──▶ GateConfig ──build_registry──▶ {key: Check}
                                │                              │
                                ▼                              ▼
                      run_gate: plan_checks (refuses unknown keys before any hashing)
                                │
                      hash base and candidate (ArtifactRef, SHA-256 per file)
                                │
             for each enabled check, in config order:
               tier D only: check.run(base, base)       ─▶ noise-floor arm
               every tier:  check.run(base, candidate)  ─▶ candidate arm
               the gate re-applies the declared envelope to each score
                                │
                      envelope.aggregate ─▶ overall verdict + reason
                                │
        write_outputs: results.json, results.summary.txt, gate.end row in the action log
```

A check that raises becomes an error arm; the other checks still run and the overall result
is NO VERDICT (or FAIL, if another check fails). Every step appends a row to the
hash-chained action log.

## Modules

| Module | Holds | Heavy imports |
|---|---|---|
| `cli`, `__main__` | the `changegate` command; hands `census`, `record`, `organism` to their own parsers | none |
| `config` | the gate.yaml schema (`GateConfig`, `CheckConfig`, `Envelope`) and `load_config` | none |
| `registry` | which implementation answers to which gate.yaml key, built per config | none |
| `gate` | `run_gate`, `write_outputs`, `gate_command`, `only_checks`; the results schema `changegate.results/v0` | none |
| `envelope` | `aggregate` (FAIL > missing verdict > WARN > PASS) and `calibrate` (percentile lines to propose, not to apply) | none |
| `checks` (package root) | the `Check` protocol, `CheckResult`, `Verdict`, `apply_envelope` | none |
| `checks.lineage*` | tier B: identity, lineage label, abliteration score; safetensors only, config and tokenizer read as JSON | torch, safetensors, lazy |
| `checks.adapter_diff` | tier A: a LoRA's writes along the refusal direction as (rᵀB)A | numpy; torch, safetensors lazy |
| `checks.drift` | tier C: refusal-direction cosine and projection, persona shift | torch, transformers, lazy |
| `directions` | prompt sets, residual-stream activations, difference-of-means directions, the `.npy` + `.json` direction format | numpy; torch lazy |
| `battery` | tier D: item sets from pinned sources, generation (transformers or vLLM), rubric and local guard judges, intervals, negative flips, κ | torch, transformers, vllm, lazy |
| `census` | Hub discovery, stratified samples, the download-gate-delete runner, the aggregate Drift Report; `gate_adapter` runs one template gate.yaml per derivative | huggingface_hub lazy |
| `organisms` | recipe configs and builders for planted known-bad models, the bank commitment, the dev/sealed split | torch, transformers, peft, lazy |
| `record` | the change record (in-toto statement in a DSSE envelope), Ed25519 signing, the offline verifier, the CycloneDX ML-BOM | cryptography lazy |
| `artifacts`, `digests`, `log`, `errors` | content identity, canonical JSON hashing, the action log, error types | none |

Everything that needs torch, transformers, safetensors, vLLM, peft or cryptography imports it
inside the function that uses it, so `import changegate` and every non-model command work
from a base install.

## Tiers

| Tier | Reads | Hardware | Checks |
|---|---|---|---|
| A | the adapter file only | CPU, a stock CI runner | `adapter-diff` |
| B | full weights, streamed, no forward pass | CPU | `lineage` |
| C | activations from forward passes of both models | one GPU | `drift.refusal`, `drift.persona` |
| D | generations, scored by a pinned rubric or local guard model | one GPU | `battery.safety-retention`, `battery.over-refusal`, `battery.sycophancy`, `battery.em` |

Under `change_type: base_upgrade`, `plan_checks` refuses any enabled check below tier D.

## Contracts between modules

- **Check.** `run(base, candidate, cfg) -> CheckResult`. No hosted or agent model computes a
  number. The result carries the score, an optional interval, the verdict against the envelope
  in `cfg`, evidence (raw numbers, input digests) and cost. The gate recomputes the verdict from
  the score and the configured envelope, except that a check's own NO_VERDICT stands.
- **Verdicts.** PASS, WARN, FAIL against the user-declared envelope, or NO_VERDICT (no envelope,
  undefined score, an interval spanning a line, a lineage label other than `match`, a judge
  below the κ bar). A PASS line reads "not detected at recall r"; r reads "not measured" until a
  known-bad library measures it.
- **Exit codes** (`cli.ExitCode`): 0 PASS or WARN, 1 FAIL, 2 usage, config or input error,
  3 not built, 4 no overall verdict. `gate.EXIT_*` holds the same values; a test pins them.
- **Noise floor.** For tier D the gate runs the base against itself on the same items and
  settings before the candidate, and reports the candidate's negative flips beside the base's.
  A battery check makes a separate generation run per call, with nothing cached.
- **Refusal direction.** One file format, written by `directions.save_direction`: a `(d,)`
  float32 `.npy` and a `.json` sidecar recording the model digest, layer, prompts and the
  vector's digest. Lineage, adapter-diff and the abliteration recipe verify the sidecar; a
  direction binds to the exact base directory it was computed on. Lineage and
  adapter-diff refuse pickles; the abliteration recipe also reads a 1-D torch tensor file
  with `weights_only=True`.
- **Battery runs.** Each model run's `BatteryRun` is written as canonical JSON and its path and
  digest go into `evidence["runs"]`; item-level rows go to a per-run item log, and the main log
  gets one `battery.run` row per model run.
- **Census.** `CensusRunner` calls a `GateRunner(base, candidate, change_type)`;
  `census.gate_adapter.TemplateGate` is the one the CLI uses. It raises when any check raised, so
  the census records a failed run, never a partial measurement. `gate_sha256` covers the template
  with artifact paths and change type blanked.

## Paths

`load_config` makes artifact and output paths absolute against gate.yaml's directory.
`gate_command` (every gate-running command) builds and runs the checks with gate.yaml's
directory as the working directory, so relative paths in `params` resolve against the same
file. Library callers of `run_gate` resolve params against their own working directory.

## Formats written

| File | Writer | Integrity |
|---|---|---|
| `results.json` | `gate.write_outputs` | re-validated on load (config digest, overall verdict); digest in `gate.end` |
| `results.summary.txt` | `gate.write_outputs` | digest in `gate.end` |
| `actions.jsonl` | `log.ActionLog` | each row hashes the previous row; compare the head with a copy kept elsewhere |
| `<stem>.npy` + `<stem>.json` | `directions.save_direction` | vector digest in the sidecar |
| census `results.jsonl`, `manifest.json`, `sample.json` | `census` | digests cross-referenced; `census.record` rows in the log |
| organism `manifest.json`, bank commitment | `organisms` | output re-hashed by `verify_output` |
| signed record (DSSE JSON), ML-BOM | `record` | Ed25519 signatures; offline verifier |

The step from `results.json` to an unsigned record statement is a library call
(`record.build.build_statement`), not a command yet: the mapping from gate results and battery
runs to a statement is open (tracked as an open item).

## Dependencies

| Install | Adds | For |
|---|---|---|
| base | pydantic, pyyaml, numpy, cryptography | config, hashing, the log, direction files, records |
| `cpu` extra | torch 2.8 (CPU wheels), transformers 4.57, safetensors, huggingface_hub | white-box checks, batteries on CPU/MPS, the census |
| `gpu` extra | the same with CUDA 12.8 wheels, plus vllm 0.10.2 on Linux | Linux CUDA nodes |
| `organisms` extra | peft, bitsandbytes (Linux) | the fine-tuning and 4-bit recipes |
| `interop` extra | securesystemslib, in-toto-attestation, cyclonedx-python-lib | the record interop tests |
| `dev` group | pytest, ruff | installed by default by `uv sync` and `uv run` |

`cpu` and `gpu` conflict (they pin different torch indexes). No GPTQ backend is declared: the
3-bit recipe needs `optimum` plus `gptqmodel` (or `auto-gptq`) installed by hand in the cluster
env. On a shared cluster, jobs can import the package from a synced source tree (`PYTHONPATH`), so
an existing environment's pins are left alone (cluster job templates are not published).

## Tests

- No test touches the network: `conftest.py` refuses non-loopback sockets and puts the
  Hugging Face libraries in offline mode.
- `model` tests build tiny checkpoints in-process or load an allowlisted cached model under
  1 GiB; they skip without torch. `gpu` tests skip unless CUDA is visible.
- `test_docstrings` walks every module: each public object states what it does not establish.
- `test_never_claim_lint` runs `tools/never_claim_lint.py` on the repository's own docs, code
  strings and scripts.
- `test_integration` runs `changegate gate` end to end on CPU with lineage, drift and a battery
  on tiny checkpoints, relative params included.
