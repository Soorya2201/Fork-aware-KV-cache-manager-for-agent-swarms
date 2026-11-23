#!/bin/bash
#SBATCH --job-name=fc_policies
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
# Phase 4.1: stock LRU vs the skirental / distaware eviction strategies inside SGLang,
# with the KV pool capped at PRESSURE_TOKENS so eviction decisions matter.
# Uses the newest prefill curve from prefill_curve.sbatch (falls back to the V100-log curve).
set -euo pipefail
source slurm/lib.sh
TRACE=${TRACE:-$PRIVATE/trace/carc_v100_trace.jsonl}
SCHEDULE=${SCHEDULE:-$PRIVATE/results/schedule_carc_v100.json}
TIME_SCALE=${TIME_SCALE:-1}
CURVE=${CURVE:-$(ls -t results/prefill/*_qwen7b_curve.json 2>/dev/null | head -1)}
CURVE=${CURVE:-$PRIVATE/results/prefill/v100_qwen7b_from_log_curve.json}
POOL_TOKENS=${POOL_TOKENS:-$PRESSURE_TOKENS}
echo "curve: $CURVE  pool: $POOL_TOKENS tokens"
python analysis/fit_thresholds.py --trace "$TRACE" --curve "$CURVE" --pool-tokens "$POOL_TOKENS" \
  --out "results/thresholds_${SLURM_JOB_ID}.json"
SKI_CFG=$(python -c "import json;c=json.load(open('$CURVE'));print(json.dumps({'b':c['b'],'c':c['c'],'pool_tokens':int('$POOL_TOKENS')}))")
export PYTHONPATH=$PROJ/sglang_ext:${PYTHONPATH:-}

run() {   # run <label> [server flags]
  local label=$1; shift
  local log="logs/sglang_${label}_${SLURM_JOB_ID}.log"
  if ! start_server "$log" --enable-cache-report --enable-metrics --max-total-tokens "$POOL_TOKENS" "$@"; then
    echo "SKIPPED $label: server failed to start (see $log)"; return 0
  fi
  python bench/replay.py --backend sglang --url "http://127.0.0.1:$PORT" --trace "$TRACE" \
    --schedule "$SCHEDULE" --time-scale "$TIME_SCALE" --label "$label" \
    --out "results/policy_${label}_${SLURM_JOB_ID}.json" || echo "replay $label failed"
  stop_server
}
run lru
run skirental --radix-eviction-policy skirental --radix-eviction-policy-config "$SKI_CFG"
run distaware --radix-eviction-policy distaware \
  --radix-eviction-policy-config "{\"thresholds\": $(cat "results/thresholds_${SLURM_JOB_ID}.json")}"
