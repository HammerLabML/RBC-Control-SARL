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

DATASET=ppo # ppo, random, zero
RA=10000

srun uv run python scripts/dataset_control.py base_seed=100 split=train total=40 ra=$RA dataset.type=$DATASET
srun uv run python scripts/dataset_control.py base_seed=400 split=test  total=20 ra=$RA dataset.type=$DATASET
srun uv run python scripts/dataset_control.py base_seed=800 split=val   total=10 ra=$RA dataset.type=$DATASET
