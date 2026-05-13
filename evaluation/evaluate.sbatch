#!/bin/bash
#SBATCH -J re_evaluate
#SBATCH -w gpu-nvidia-h200-1
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 4:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH" && python evaluation/evaluate.py