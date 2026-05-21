#!/bin/bash
#SBATCH -J evaluate
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
RUN=run1

source "$VENV_PATH"

#Qwen
python evaluation/run_inference.py --model Qwen/Qwen2.5-1.5B-Instruct --run $RUN
python evaluation/evaluate.py --predictions predictions/Qwen2.5-1.5B-Instruct_base_${RUN}.jsonl
python evaluation/run_inference.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter Qwen2.5-1.5B-Instruct-text2kg-qlora --run $RUN
python evaluation/evaluate.py --predictions predictions/Qwen2.5-1.5B-Instruct_finetuned_${RUN}.jsonl
