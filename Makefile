.PHONY: help test gate bench verify retention fidelity holdout build pyz docker clean lint

help:  ## Show this help
	@grep -E '^[a-z]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

test:  ## Run the full test suite
	uv run --with pytest python -m pytest -q

gate: bench verify retention fidelity  ## Run the full CI gate (non-inferiority + byte-fidelity + recall + state probes)

fidelity:  ## State probes: artifact state, overclaim, continuation, propagation
	# Gated at the measured band, not 0: the reversible tier drops hedging on a
	# small number of claims (docs/EVALUATION.md section 6.2). Gating at 0 would
	# assert a property the code does not have.
	uv run distil fidelity --max-silent 15

bench:  ## Corpus-wide non-inferiority gate
	uv run distil bench

verify:  ## Byte-fidelity gate (reversibility + append-only)
	uv run distil verify

retention:  ## Fact-level recall gate (zero cost: no LLM, no API key)
	uv run distil retention --max-lost 0

holdout:  ## Holdout A/B savings with bootstrap CI
	uv run distil holdout

build:  ## Build wheel + sdist (PyPI distributables)
	uv build

pyz:  ## Build the single-file executable (dist/distil.pyz)
	bash scripts/build_pyz.sh

docker:  ## Build the container image
	docker build -t distil:latest .

lint:  ## Lint with ruff (pinned to the version CI gates on)
	uvx ruff@0.15.10 check distil tests
	uvx ruff@0.15.10 format --check .

clean:  ## Remove build artifacts
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache
