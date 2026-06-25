#!/usr/bin/env bash
# Phase 5 / E7 — autonomous finalize: wait for agents + image preload, then score all
# three conditions with the official harness and aggregate into the canonical results
# JSON + paper macros. Designed to run unattended after run_all_agents.sh + preload.
set -u
cd "$(dirname "$0")/../.." || exit 1
LOG=/tmp/e7_finalize.log
: > "$LOG"
start=$(date +%s)
echo "finalize: waiting for agents + preload ... $(date +%T)" | tee -a "$LOG"

# Block until both the agent run and the image preload have finished.
while true; do
  agents_done=$(grep -c ALL_AGENTS_DONE /tmp/e7_full_agents.log 2>/dev/null || echo 0)
  preload_done=$(grep -c PRELOAD_DONE /tmp/e7_preload.log 2>/dev/null || echo 0)
  [ "$agents_done" -ge 1 ] && [ "$preload_done" -ge 1 ] && break
  sleep 30
done
echo "finalize: agents + preload done, scoring $(date +%T)" | tee -a "$LOG"

bash benchmarks/swe_bench_e2e/run_all_scores.sh 2>&1 | tee -a "$LOG"

agent_start=$(grep -m1 "START full" /tmp/e7_full_agents.log >/dev/null 2>&1 && echo ok)
wall=$(( $(date +%s) - start ))
echo "finalize: aggregating $(date +%T)" | tee -a "$LOG"
.venv/bin/python -m benchmarks.swe_bench_e2e.aggregate 2>&1 | tee -a "$LOG"

echo "E7_FINALIZE_DONE wall_finalize=${wall}s $(date +%T)" | tee -a "$LOG"
