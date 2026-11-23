#!/bin/bash
#SBATCH --job-name=fc_kernel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Phase 5.1: correctness of the Triton shared-prefix kernel, then benchmark vs FlashInfer cascade.
set -euo pipefail
source slurm/lib.sh
uv pip install flashinfer-python >/dev/null 2>&1 || echo "flashinfer install failed: benchmark will skip it"
python kernels/test_shared_prefix.py
python kernels/bench_cascade.py --out "results/kernel_${GPU}_${SLURM_JOB_ID}.json"
