#!/bin/bash
#SBATCH --job-name=fc_trace7b
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
# More 7B sessions for tighter confidence intervals: runs the agent app in $ZAGE_DIR
# (unmodified) against SGLang with the trace logger. Prefix cache off, like the V100 run.
set -euo pipefail
source slurm/lib.sh
start_server "logs/sglang_trace_${SLURM_JOB_ID}.log" --disable-radix-cache
mkdir -p results/trace_7b results/private
python trace/run_zage_traced.py --sessions "${SESSIONS:-20}" --rounds "${ROUNDS:-6}" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --raw "results/trace_7b/raw_${SLURM_JOB_ID}.jsonl" \
  --out "results/trace_7b/zage_trace_7b_${GPU}_${SLURM_JOB_ID}.jsonl"
stop_server
