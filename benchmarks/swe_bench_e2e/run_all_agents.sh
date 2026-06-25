#!/usr/bin/env bash
# Phase 5 / E7 — run all three agent conditions sequentially over the seed-1729 sample.
# Sequential (not concurrent) so the shared git clone cache never sees cross-process
# worktree races, and so we never run >N concurrent Anthropic streams or >N concurrent
# llmlingua CPU inferences. Each condition is internally parallel (--workers) and
# resumable (rows already in predictions/<cond>.jsonl are skipped).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=/tmp/e7_full_agents.log
: > "$LOG"

run() {
  local cond="$1" workers="$2"
  echo "==== $(date +%T) START $cond (workers=$workers) ====" | tee -a "$LOG"
  $PY -m benchmarks.swe_bench_e2e.run_agent \
    --condition "$cond" --workers "$workers" --timeout 900 >>"$LOG" 2>&1
  echo "==== $(date +%T) END $cond rc=$? ====" | tee -a "$LOG"
}

run full 5
run distil_trunc500 5
run llmlingua2 3   # llmlingua-2 is CPU-bound; fewer workers avoids thrashing

echo "ALL_AGENTS_DONE $(date +%T)" | tee -a "$LOG"
