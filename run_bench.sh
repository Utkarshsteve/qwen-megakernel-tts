#!/usr/bin/env bash
# Sweep code-predictor optimizations, each in a fresh process. Logs to bench.log.
cd /workspace
LOG=bench.log
: > "$LOG"
echo "== bench start $(date) ==" | tee -a "$LOG"

run () {  # name  + env assignments
  local tag="$1"; shift
  echo "--- running $tag ($*) ---" | tee -a "$LOG"
  TAG="$tag" env "$@" python bench_cp.py >>"$LOG" 2>&1
  echo "--- done $tag ---" | tee -a "$LOG"
}

run baseline
run fastcp           FAST_CP=1
run ro_fastcp        FAST_CP=1 COMPILE_MODE=reduce-overhead COMPILE_DYNAMIC=0
run autotune_fastcp  FAST_CP=1 COMPILE_MODE=max-autotune

echo "== bench done $(date) ==" | tee -a "$LOG"
grep "^RESULT" "$LOG" | tee -a "$LOG"
