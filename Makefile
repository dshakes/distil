.PHONY: help test gate bench verify validate retention fidelity suite holdout build pyz docker clean lint

help:  ## Show this help
	@grep -E '^[a-z]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

test:  ## Run the full test suite
	uv run --with pytest --with pillow python -m pytest -q

gate: bench verify validate retention fidelity suite  ## Run the full CI gate (non-inferiority + byte-fidelity + adversarial + recall + state probes + public benchmarks)
# Must stay identical to the gate steps in .github/workflows/ci.yml. `validate` was
# missing here while CI ran it, so a local green gate could still fail on push —
# which is the one thing this target exists to prevent.

fidelity:  ## State probes: artifact state, overclaim, continuation, propagation
	# Gated at the measured band, not 0: the reversible tier drops hedging on a
	# small number of claims (docs/EVALUATION.md section 6.2). Gating at 0 would
	# assert a property the code does not have.
	uv run distil fidelity --max-silent 15

suite:  ## Public benchmarks (third-party ground truth). First run fetches + caches.
	# Zero API spend: grading is deterministic recall, not an LLM judge. Tier 1 is
	# the payload an agent proxy actually risks breaking — tool schemas and retrieval.
	# Gated at the MEASURED band on RICH rows only: bfcl, hotpotqa and squad all
	# 1.000 support at n=25 (bfcl read 0.964 under the old prose matcher; scoring its
	# golds as identifiers removed 11 accidental substring credits and 15 one-letter
	# golds no text rule can adjudicate — see docs/RUNNING-EVALS.md). The band stays at
	# 0.95 rather than 1.0: a gate pinned to a perfect score fails on sampling noise
	# and teaches people to widen it, which is how a gate stops meaning anything.
	# Controls are excluded by construction — gsm8k loses 17 bare-number answers here,
	# which says nothing about compression.
	uv run distil suite --tier 1 -n 25 --min-answer-recall 0.95 --min-support-recall 0.95 --allow-unavailable

bench:  ## Corpus-wide non-inferiority gate
	uv run distil bench

validate:  ## Adversarial invariants on hostile input (reversibility, fail-open, recency)
	uv run distil validate

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
