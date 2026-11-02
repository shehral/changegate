# Targets: sync, lint, test, test-gpu, fetch-test-models.
# On a CUDA machine, inside an environment with the gpu extra: make test-gpu PYTEST=pytest
UV ?= uv
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff

.PHONY: sync lint test test-gpu fetch-test-models

# The dev group (pytest, ruff) is installed by default.
sync:
	$(UV) sync --extra cpu --extra interop

lint:
	$(RUFF) check .
	$(RUFF) format --check .
	python3 tools/never_claim_lint.py

test:
	$(PYTEST) -m "not gpu"

test-gpu:
	$(PYTEST) -m gpu

# The only network step for tests; run it outside pytest. About 2 MB.
fetch-test-models:
	$(UV) run python -c "from huggingface_hub import snapshot_download as s; print(s('hf-internal-testing/tiny-random-LlamaForCausalLM'))"
