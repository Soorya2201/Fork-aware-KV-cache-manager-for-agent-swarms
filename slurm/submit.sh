#!/bin/bash
# Usage: bash slurm/submit.sh slurm/replay.sbatch [extra sbatch args]
# Adds account / partition / GPU from slurm/carc.env, so the .sbatch files stay generic.
set -euo pipefail
source "$(dirname "$0")/carc.env"
job=$1; shift
mkdir -p logs
sbatch --account="$ACCOUNT" --partition="$PARTITION" --gres="gpu:$GPU:1" "$@" "$job"
