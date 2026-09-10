#!/bin/bash
#SBATCH -J test_cross
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

RUN_LABEL=trainedon_tum_e3_run1

LLAMA_ID=meta-llama/Llama-3.2-1B-Instruct
QWEN_ID=Qwen/Qwen2.5-1.5B-Instruct
GEMMA_ID=google/gemma-3-1b-it

LLAMA_BASE=Llama-3.2-1B-Instruct-tum_e3-qlora_run1
QWEN_BASE=Qwen2.5-1.5B-Instruct-tum_e3-qlora_run1
GEMMA_BASE=gemma-3-1b-it-tum_e3-qlora_run1

# Skips a checkpoint if its prediction file already exists, so this is safe to resubmit.
run_inference() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    RUN_LABEL=$3
    MODEL_NAME=$(basename "$MODEL_ID")
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        RUN_TAG="${RUN_LABEL}_$(basename $CHECKPOINT)"
        OUT_FILE="predictions/${MODEL_NAME}_${TEST_DATASET}_finetuned_${RUN_TAG}.jsonl"
        if [ -f "$OUT_FILE" ]; then
            echo "Skipping $OUT_FILE (already exists)"
            continue
        fi
        python evaluation/run_inference.py \
            --model "$MODEL_ID" --adapter "$CHECKPOINT" \
            --dataset $TEST_DATASET --run "$RUN_TAG"
    done
}


# --- tum_e3 (own test set) ---
TEST_DATASET=tum_e3
run_inference "$LLAMA_ID" "$LLAMA_BASE" "$RUN_LABEL"
run_inference "$QWEN_ID"  "$QWEN_BASE"  "$RUN_LABEL"
run_inference "$GEMMA_ID" "$GEMMA_BASE" "$RUN_LABEL"


# --- REBEL ---
TEST_DATASET=rebel
run_inference "$LLAMA_ID" "$LLAMA_BASE" "$RUN_LABEL"
run_inference "$QWEN_ID"  "$QWEN_BASE"  "$RUN_LABEL"
run_inference "$GEMMA_ID" "$GEMMA_BASE" "$RUN_LABEL"

# --- LAGRANGE ---
TEST_DATASET=lagrange
run_inference "$LLAMA_ID" "$LLAMA_BASE" "$RUN_LABEL"
run_inference "$QWEN_ID"  "$QWEN_BASE"  "$RUN_LABEL"
run_inference "$GEMMA_ID" "$GEMMA_BASE" "$RUN_LABEL"

# --- WikiNRE ---
TEST_DATASET=wikinre
run_inference "$LLAMA_ID" "$LLAMA_BASE" "$RUN_LABEL"
run_inference "$QWEN_ID"  "$QWEN_BASE"  "$RUN_LABEL"
run_inference "$GEMMA_ID" "$GEMMA_BASE" "$RUN_LABEL"


