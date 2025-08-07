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
srun uv run python scripts/dataset_control.py -m base_seed=100 split=train dataset.type=ppo,random,zero
srun uv run python scripts/dataset_control.py -m base_seed=400 split=val dataset.type=ppo,random,zero
srun uv run python scripts/dataset_control.py -m base_seed=800 split=test dataset.type=ppo,random,zero
