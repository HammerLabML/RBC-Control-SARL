#!/bin/bash

#SBATCH --job-name=sbsa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=23
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/homes/%u/RBC-Control-SARL/logs/%x_%j.out
#SBATCH --mail-type=FAIL

cd ~/RBC-Control-SARL
srun uv run python scripts/sbsa.py
