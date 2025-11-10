#!/bin/bash
#SBATCH --job-name=dataset_control
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=13
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/%u/RBC-Control-SARL/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --array=0-2

# --- dataset types ---
TYPES=("ppo" "random" "zero")
TYPE=${TYPES[$SLURM_ARRAY_TASK_ID]}

cd ~/RBC-Control-SARL
echo "Running dataset.type=$TYPE"

srun uv run python scripts/dataset_control.py base_seed=100 split=train dataset.total=50
srun uv run python scripts/dataset_control.py base_seed=400 split=val dataset.total=20
srun uv run python scripts/dataset_control.py base_seed=800 split=test dataset.total=20