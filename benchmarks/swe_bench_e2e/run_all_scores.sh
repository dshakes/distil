#!/usr/bin/env bash
# Phase 5 / E7 — score all three conditions with the OFFICIAL SWE-bench harness.
# Requires the per-instance images to be present locally (see preload_images.py); the
# harness logs "Found N existing instance images. Will reuse them." and never pulls.
# Each condition's predictions JSONL is scored independently; results go to
# docs/paper/results/swe_e2e/scores/<cond>.json (parsed from the harness's own report).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-swebench/bin/python
SCORES=docs/paper/results/swe_e2e/scores
mkdir -p "$SCORES"
LOG=/tmp/e7_full_scores.log
: > "$LOG"
export DOCKER_DEFAULT_PLATFORM=linux/amd64

for cond in full distil_trunc500 llmlingua2; do
  echo "==== $(date +%T) SCORE $cond ====" | tee -a "$LOG"
  $PY -m benchmarks.swe_bench_e2e.score \
    --pred "docs/paper/results/swe_e2e/predictions/$cond.jsonl" \
    --run-id "e7_$cond" \
    --max-workers 4 \
    --harness-timeout 600 \
    --out "$SCORES/$cond.json" >>"$LOG" 2>&1
  echo "==== $(date +%T) SCORED $cond rc=$? ====" | tee -a "$LOG"
done

echo "ALL_SCORES_DONE $(date +%T)" | tee -a "$LOG"
