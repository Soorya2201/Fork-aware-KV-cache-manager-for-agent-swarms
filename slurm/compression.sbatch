#!/bin/bash
#SBATCH --job-name=kv_compress
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Phase 5.2 at full size: INT8/FP8 idle-KV agreement and accuracy on Qwen2.5-7B.
set -euo pipefail
source slurm/lib.sh
python compression/kv_int8_check.py --model "$MODEL" --device cuda --dtype bfloat16 --max-new 256 \
  --out "results/compression/kv_check_7b_${SLURM_JOB_ID}.json"
