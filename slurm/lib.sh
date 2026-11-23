# Shared helpers for the job scripts.
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/carc.env"
cd "$PROJ"
source .venv/bin/activate
mkdir -p logs results
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv | tee "logs/gpu_${SLURM_JOB_ID:-local}.csv"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
# Current torch / SGLang wheels have no kernels for V100 (compute 7.0): stop with a clear message.
if [ "${CC%%.*}" -lt 8 ]; then
  echo "This GPU has compute capability $CC. SGLang needs 8.0+ (A100, A40, L40S). Set GPU in slurm/carc.env."
  exit 2
fi
GPU_FLAGS=()

start_server() {   # start_server <log> [extra sglang flags...]
  local log=$1; shift
  python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" "${GPU_FLAGS[@]}" "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  until curl -s "localhost:$PORT/health" >/dev/null; do
    kill -0 $SERVER_PID 2>/dev/null || { echo "server died, see $log"; tail -50 "$log"; return 1; }
    sleep 5
  done
  grep -o "max_total_num_tokens=[0-9]*" "$log" | tail -1 || true
}

stop_server() {
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
}
