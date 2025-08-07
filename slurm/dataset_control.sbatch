#!/bin/bash

#SBATCH --job-name=dataset_control
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=23
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/%u/RBC-Control-SARL/logs/%x_%j.out
#SBATCH --mail-type=FAIL

cd ~/RBC-Control-SARL
srun uv run python scripts/dataset_control.py dataset.total=20 dataset.type=ppo
srun uv run python scripts/dataset_control.py dataset.total=10 dataset.type=random
srun uv run python scripts/dataset_control.py dataset.total=10 dataset.type=zero
