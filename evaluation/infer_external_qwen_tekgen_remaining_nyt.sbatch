#!/bin/bash
#SBATCH -J qwen_tek_nyt
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
MODEL_NAME=$(basename "$MODEL_ID")
ADAPTER_BASE=Qwen2.5-1.5B-Instruct-tekgen-qlora_run1
RUN_LABEL=trainedon_tekgen_run1


for STEP in 4000 5000 6000 7000 8000; do
    RUN_TAG="${RUN_LABEL}_checkpoint-${STEP}"
    OUT_FILE="predictions/${MODEL_NAME}_nyt_finetuned_${RUN_TAG}.jsonl"
    if [ -f "$OUT_FILE" ]; then
        echo "Skipping $OUT_FILE (already exists)"
    else
        python evaluation/run_inference_external.py \
            --model "$MODEL_ID" --adapter "${ADAPTER_BASE}/checkpoint-${STEP}" \
            --dataset nyt --run "$RUN_TAG"
    fi
done
