#!/bin/bash
#SBATCH -J ext_1
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

TEST_DATASETS="webnlg conll04"
RUN_LABEL=trainedon_tum_e3_run1


run_both_checkpoints() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    BEST_F1_STEP=$3
    MODEL_NAME=$(basename "$MODEL_ID")
    for TEST_DATASET in $TEST_DATASETS; do
        # best F1
        RUN_TAG="${RUN_LABEL}_bestf1_checkpoint-${BEST_F1_STEP}"
        OUT_FILE="predictions/${MODEL_NAME}_${TEST_DATASET}_finetuned_${RUN_TAG}.jsonl"
        if [ -f "$OUT_FILE" ]; then
            echo "Skipping $OUT_FILE (already exists)"
        else
            python evaluation/run_inference_external.py \
                --model "$MODEL_ID" --adapter "${ADAPTER_BASE}/checkpoint-${BEST_F1_STEP}" \
                --dataset $TEST_DATASET --run "$RUN_TAG"
        fi

        # best eval_loss
        RUN_TAG="${RUN_LABEL}_bestevalloss"
        OUT_FILE="predictions/${MODEL_NAME}_${TEST_DATASET}_finetuned_${RUN_TAG}.jsonl"
        if [ -f "$OUT_FILE" ]; then
            echo "Skipping $OUT_FILE (already exists)"
        else
            python evaluation/run_inference_external.py \
                --model "$MODEL_ID" --adapter "$ADAPTER_BASE" \
                --dataset $TEST_DATASET --run "$RUN_TAG"
        fi
    done
}

LLAMA_ID=meta-llama/Llama-3.2-1B-Instruct
QWEN_ID=Qwen/Qwen2.5-1.5B-Instruct
GEMMA_ID=google/gemma-3-1b-it

run_both_checkpoints "$LLAMA_ID" Llama-3.2-1B-Instruct-tum_e3-qlora_run1 4000
run_both_checkpoints "$QWEN_ID"  Qwen2.5-1.5B-Instruct-tum_e3-qlora_run1 7000
run_both_checkpoints "$GEMMA_ID" gemma-3-1b-it-tum_e3-qlora_run1         5000
