#!/bin/bash
#SBATCH --job-name=replay
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Phase 3.3: replay the trace on real SGLang, radix cache off and on.
# TIME_SCALE compresses a Mac-speed trace to GPU speed (1 for a trace recorded on CARC).
set -euo pipefail
source slurm/lib.sh
TRACE=${TRACE:-trace/zage_trace.jsonl}
SCHEDULE=${SCHEDULE:-results/schedule.json}
TIME_SCALE=${TIME_SCALE:-1}

start_server "logs/sglang_noradix_${SLURM_JOB_ID}.log" --disable-radix-cache --enable-metrics
python bench/replay.py --backend sglang --url "http://127.0.0.1:$PORT" --trace "$TRACE" \
  --schedule "$SCHEDULE" --time-scale "$TIME_SCALE" --label noradix \
  --out "results/noradix_${SLURM_JOB_ID}.json"
stop_server

start_server "logs/sglang_radix_${SLURM_JOB_ID}.log" --enable-cache-report --enable-metrics
grep -o "max_total_num_tokens=[0-9]*" "logs/sglang_radix_${SLURM_JOB_ID}.log" | tail -1 \
  > "results/kv_pool_${SLURM_JOB_ID}.txt" || true
python bench/replay.py --backend sglang --url "http://127.0.0.1:$PORT" --trace "$TRACE" \
  --schedule "$SCHEDULE" --time-scale "$TIME_SCALE" --label radix \
  --out "results/radix_${SLURM_JOB_ID}.json"
curl -s "localhost:$PORT/metrics" > "results/metrics_radix_${SLURM_JOB_ID}.txt" || true
stop_server
