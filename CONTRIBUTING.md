# Contributing

These rules hold for every change. CI enforces the ones marked **(CI)**.

## Measurement

- No hosted or agent model computes, edits or writes any number in a result. Scores come from
  the artifacts and from pinned instruments (rule-based scorers, or local judge weights
  identified by SHA-256 together with the digest of the scoring policy).
- A check returns a `CheckResult` and takes its verdict from `apply_envelope`. With no
  envelope it returns NO_VERDICT.
- Measuring runs append `check.start` and `check.end` rows to the action log, with input
  digests and the digest of the result.
- Result wording goes through `CheckResult.describe()`; do not write verdict text elsewhere.

## Code

- Small modules, type hints, no dead code. **(CI: ruff)**
- Every public function, method and class has a docstring that states its inputs, its outputs
  and, on a line starting `Does not establish:`, what it does not show. **(CI:
  `tests/test_docstrings.py`)**
- User-facing text passes the never-claim lint. **(CI: `tools/never_claim_lint.py`)**

## Tests

- No network in unit tests. `tests/conftest.py` refuses non-loopback connections and puts the
  Hugging Face libraries in offline mode. **(CI)**
- Models in tests are tiny: build them in-process (`tiny_llama_pair`) or load an allowlisted
  model already in the local cache (`cached_hf_model`), never more than 1 GiB in total.
- Tests that need CUDA carry `@pytest.mark.gpu` and are skipped elsewhere; tests that need
  torch carry `@pytest.mark.model`.

## Data and credit

- A dataset gets a row in `DATA-LICENSES.md` before any code loads it. Permissive licenses
  only; synthetic items where a license is unclear. **(CI: `tests/test_data_licenses.py`
  checks the pinned sources and shipped prompt sets)**
- A technique gets a row in `CREDITS.md` in the same change that implements it.

## Questions

Open questions go to GitHub issues. Pull requests from outside contributors are not accepted yet;
issues and bug reports are welcome.
