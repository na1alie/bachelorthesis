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

#Gemma
python evaluation/run_inference.py --model google/gemma-3-1b-it --run $RUN
python evaluation/evaluate.py --predictions predictions/gemma-3-1b-it_base_${RUN}.jsonl
python evaluation/run_inference.py --model google/gemma-3-1b-it --adapter gemma-3-1b-it-text2kg-qlora --run $RUN
python evaluation/evaluate.py --predictions predictions/gemma-3-1b-it_finetuned_${RUN}.jsonl

