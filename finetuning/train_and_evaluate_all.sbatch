#!/bin/bash
#SBATCH -J 2_train_all
#SBATCH -w gpu-nvidia-h200-1
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

RUN=run2

#Qwen
python finetuning/finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --run $RUN
python evaluation/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --run $RUN
python evaluation/evaluate_finetuned.py --base_model Qwen/Qwen2.5-1.5B-Instruct --run $RUN

#Gemma
python finetuning/finetune.py --model google/gemma-3-1b-it --run $RUN
python evaluation/evaluate.py --model google/gemma-3-1b-it --run $RUN
python evaluation/evaluate_finetuned.py --base_model google/gemma-3-1b-it --run $RUN

#LLaMA
python finetuning/finetune.py --model meta-llama/Llama-3.2-1B-Instruct --run $RUN
python evaluation/evaluate.py --model meta-llama/Llama-3.2-1B-Instruct --run $RUN
python evaluation/evaluate_finetuned.py --base_model meta-llama/Llama-3.2-1B-Instruct --run $RUN
