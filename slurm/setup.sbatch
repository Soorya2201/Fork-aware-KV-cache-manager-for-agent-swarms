#!/bin/bash
#SBATCH --job-name=fc_setup
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
# One-time environment: uv venv, SGLang, deps, Qwen2.5-7B weights. Skips work already done.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/carc.env"
cd "$PROJ"
if [ -f .venv/.forkcache_ready ]; then echo "environment already set up"; exit 0; fi
bash slurm/setup_env.sh
python sglang_ext/test_strategy.py || echo "WARNING: strategy self-test failed (policy runs may skip)"
touch .venv/.forkcache_ready
