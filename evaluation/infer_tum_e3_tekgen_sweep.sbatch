#!/bin/bash
#SBATCH -J tekgen
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

TEST_DATASET=tum_e3
RUN_LABEL=trainedon_tekgen_run1

run_single_checkpoint() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    STEP=$3
    MODEL_NAME=$(basename "$MODEL_ID")
    CHECKPOINT="${ADAPTER_BASE}/checkpoint-${STEP}"
    RUN_TAG="${RUN_LABEL}_checkpoint-${STEP}"
    OUT_FILE="predictions/${MODEL_NAME}_${TEST_DATASET}_finetuned_${RUN_TAG}.jsonl"
    if [ -f "$OUT_FILE" ]; then
        echo "Skipping $OUT_FILE (already exists)"
        return
    fi
    python evaluation/run_inference.py \
        --model "$MODEL_ID" --adapter "$CHECKPOINT" \
        --dataset $TEST_DATASET --run "$RUN_TAG"
}

run_full_sweep() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        STEP=$(basename "$CHECKPOINT" | sed 's/checkpoint-//')
        run_single_checkpoint "$MODEL_ID" "$ADAPTER_BASE" "$STEP"
    done
}

LLAMA_ID=meta-llama/Llama-3.2-1B-Instruct
QWEN_ID=Qwen/Qwen2.5-1.5B-Instruct
GEMMA_ID=google/gemma-3-1b-it

run_full_sweep "$LLAMA_ID" Llama-3.2-1B-Instruct-tekgen-qlora_run1
run_full_sweep "$QWEN_ID"  Qwen2.5-1.5B-Instruct-tekgen-qlora_run1
run_full_sweep "$GEMMA_ID" gemma-3-1b-it-tekgen-qlora_run1
