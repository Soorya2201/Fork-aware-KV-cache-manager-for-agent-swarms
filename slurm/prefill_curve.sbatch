#!/bin/bash
#SBATCH --job-name=fc_prefill
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Phase 3.2: prefill cost curve R(L) with the radix cache off.
set -euo pipefail
source slurm/lib.sh
start_server "logs/sglang_prefill_${SLURM_JOB_ID}.log" --disable-radix-cache --context-length 32768
python bench/prefill_curve.py --backend sglang --url "http://127.0.0.1:$PORT" \
  --name "${GPU}_qwen7b" --reps 10
stop_server
