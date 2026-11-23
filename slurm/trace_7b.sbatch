#!/bin/bash
#SBATCH --job-name=trace_7b
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Re-record the 13-agent trace on a GPU with Qwen2.5-7B, so idle gaps reflect real
# GPU-speed serving (the Mac trace uses a 0.5B model on Ollama). Needs zage/ copied to $PROJ.
set -euo pipefail
source slurm/lib.sh
start_server "logs/sglang_trace_${SLURM_JOB_ID}.log" --enable-cache-report
python trace/run_zage_traced.py --sessions "${SESSIONS:-20}" --rounds "${ROUNDS:-6}" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --raw "trace/raw_calls_7b_${SLURM_JOB_ID}.jsonl" --out "trace/zage_trace_7b.jsonl"
stop_server
