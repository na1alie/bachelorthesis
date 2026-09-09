#!/bin/bash
#SBATCH -J tum_e3
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

run_single_checkpoint() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    RUN_LABEL=$3
    STEP=$4
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

LLAMA_ID=meta-llama/Llama-3.2-1B-Instruct
QWEN_ID=Qwen/Qwen2.5-1.5B-Instruct
GEMMA_ID=google/gemma-3-1b-it

# --- REBEL (run3): llama ep4, qwen ep6, gemma ep5 ---
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-rebel-qlora_run3 trainedon_rebel_run3 4000
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-rebel-qlora_run3 trainedon_rebel_run3 6000
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-rebel-qlora_run3         trainedon_rebel_run3 5000

# --- LAGRANGE (run2): llama ep7, qwen ep6, gemma ep6 ---
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-lagrange-qlora_run2 trainedon_lagrange_run2 7000
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-lagrange-qlora_run2 trainedon_lagrange_run2 6000
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-lagrange-qlora_run2         trainedon_lagrange_run2 6000

# --- WikiNRE (run1): llama ep5, qwen ep6, gemma ep4 ---
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-wikinre-qlora_run1 trainedon_wikinre_run1 5000
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-wikinre-qlora_run1 trainedon_wikinre_run1 6000
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-wikinre-qlora_run1         trainedon_wikinre_run1 4000

# --- TekGen (run1): llama ep3, qwen ep5, gemma ep3 ---
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-tekgen-qlora_run1 trainedon_tekgen_run1 3000
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-tekgen-qlora_run1 trainedon_tekgen_run1 5000
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-tekgen-qlora_run1         trainedon_tekgen_run1 3000
