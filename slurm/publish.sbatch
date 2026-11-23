#!/bin/bash
#SBATCH --job-name=fc_publish
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Last job: push results to the private data repo (runs even if earlier jobs failed).
bash slurm/publish.sh
