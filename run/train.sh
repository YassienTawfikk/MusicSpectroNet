#!/bin/bash
#SBATCH --job-name=music_spectronet_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

# Ensure we are in the project root if running from run/ folder
# However, user example had `cd ..`. If I run this from `run/`, `cd ..` gets to root.
# If I run from root via explicit path, it might be weird.
# I will assume this script is executed from the `run` directory or user is aware.
# But better to be robust.


cd "$(dirname "$0")/.." || exit

# Create logs directory if it doesn't exist
mkdir -p logs

export PYTHONPATH=$PYTHONPATH:.

# Run training

# Using fewer estimators for quick verification as requested
python scripts/training.py fit \
  --n_estimators=10 \
  --learning_rate=0.05 \
  --max_depth=6
