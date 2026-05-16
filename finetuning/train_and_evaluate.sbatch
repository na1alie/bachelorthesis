#!/bin/bash
#SBATCH -J train_eval
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

python finetuning/finetune.py --model google/gemma-3-1b-it
python evaluation/evaluate.py --model google/gemma-3-1b-it
python evaluation/evaluate_finetuned.py --base_model google/gemma-3-1b-it

