#!/bin/bash
#SBATCH --job-name=fc_replay
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
# Phase 3.3: replay the V100 trace on real SGLang. Three passes (about 1 h each):
#   noradix   - radix cache off (every call recomputes)
#   radix     - stock SGLang radix cache, full KV pool
#   radix_cap - stock radix cache with the pool capped at PRESSURE_TOKENS (memory pressure)
# results/kv_pool_<job>_<label>.txt holds each run's pool size for the simulator check.
set -euo pipefail
source slurm/lib.sh
TRACE=${TRACE:-$PRIVATE/trace/carc_v100_trace.jsonl}
SCHEDULE=${SCHEDULE:-$PRIVATE/results/schedule_carc_v100.json}
TIME_SCALE=${TIME_SCALE:-1}
pass() {   # pass <label> [server flags]
  local label=$1; shift
  # (redirect, not a pipe: a pipe would run start_server in a subshell and lose SERVER_PID)
  start_server "logs/sglang_${label}_${SLURM_JOB_ID}.log" --enable-metrics "$@" \
    > "results/kv_pool_${SLURM_JOB_ID}_${label}.txt"
  cat "results/kv_pool_${SLURM_JOB_ID}_${label}.txt"
  python bench/replay.py --backend sglang --url "http://127.0.0.1:$PORT" --trace "$TRACE" \
    --schedule "$SCHEDULE" --time-scale "$TIME_SCALE" --label "$label" \
    --out "results/replay_${label}_${SLURM_JOB_ID}.json"
  curl -s "localhost:$PORT/metrics" > "results/metrics_${label}_${SLURM_JOB_ID}.txt" || true
  stop_server
}
pass noradix --disable-radix-cache
pass radix --enable-cache-report
pass radix_cap --enable-cache-report --max-total-tokens "$PRESSURE_TOKENS"
