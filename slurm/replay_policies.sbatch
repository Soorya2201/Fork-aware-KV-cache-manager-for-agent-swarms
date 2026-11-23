#!/bin/bash
#SBATCH --job-name=replay_policies
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# Phase 4.1: stock LRU vs the skirental / distaware eviction strategies on real SGLang.
# Needs: CURVE (prefill curve json from prefill_curve.sbatch), POOL_TOKENS (from
# results/kv_pool_<job>.txt of replay.sbatch), optional THRESHOLDS (json list for distaware).
# MEM_FRACTION lowers the KV pool so the replay is actually under memory pressure.
set -euo pipefail
source slurm/lib.sh
TRACE=${TRACE:-trace/zage_trace.jsonl}
SCHEDULE=${SCHEDULE:-results/schedule.json}
TIME_SCALE=${TIME_SCALE:-1}
MEM_FRACTION=${MEM_FRACTION:-0.88}
: "${CURVE:?set CURVE=results/prefill/<gpu>_qwen7b_curve.json}"
: "${POOL_TOKENS:?set POOL_TOKENS from results/kv_pool_<job>.txt}"
SKI_CFG=$(python -c "import json;c=json.load(open('$CURVE'));print(json.dumps({'b':c['b'],'c':c['c'],'pool_tokens':int('$POOL_TOKENS')}))")
export PYTHONPATH=$PROJ/sglang_ext:${PYTHONPATH:-}

run() {   # run <label> [server flags]
  local label=$1; shift
  start_server "logs/sglang_${label}_${SLURM_JOB_ID}.log" --enable-cache-report --enable-metrics \
    --mem-fraction-static "$MEM_FRACTION" "$@"
  grep -q "\[forkcache\]" "logs/sglang_${label}_${SLURM_JOB_ID}.log" && { echo "plugin failed"; exit 1; } || true
  python bench/replay.py --backend sglang --url "http://127.0.0.1:$PORT" --trace "$TRACE" \
    --schedule "$SCHEDULE" --time-scale "$TIME_SCALE" --label "$label" \
    --out "results/policy_${label}_${SLURM_JOB_ID}.json"
  stop_server
}
run lru
run skirental --radix-eviction-policy skirental --radix-eviction-policy-config "$SKI_CFG"
if [ -n "${THRESHOLDS:-}" ]; then
  run distaware --radix-eviction-policy distaware \
    --radix-eviction-policy-config "{\"thresholds\": $(cat "$THRESHOLDS")}"
fi
