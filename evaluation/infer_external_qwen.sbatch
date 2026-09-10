#!/bin/bash
#SBATCH -J ext_qwen
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
TEST_DATASETS="nyt webnlg conll04"


run_both_checkpoints() {
    ADAPTER_BASE=$1
    BEST_F1_STEP=$2
    RUN_LABEL=$3
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


run_full_sweep() {
    ADAPTER_BASE=$1
    RUN_LABEL=$2
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        STEP=$(basename "$CHECKPOINT" | sed 's/checkpoint-//')
        for TEST_DATASET in $TEST_DATASETS; do
            RUN_TAG="${RUN_LABEL}_checkpoint-${STEP}"
            OUT_FILE="predictions/${MODEL_NAME}_${TEST_DATASET}_finetuned_${RUN_TAG}.jsonl"
            if [ -f "$OUT_FILE" ]; then
                echo "Skipping $OUT_FILE (already exists)"
            else
                python evaluation/run_inference_external.py \
                    --model "$MODEL_ID" --adapter "$CHECKPOINT" \
                    --dataset $TEST_DATASET --run "$RUN_TAG"
            fi
        done
    done
}

# ADAPTER_BASE                                   BEST_F1_STEP  RUN_LABEL
run_both_checkpoints Qwen2.5-1.5B-Instruct-rebel-qlora_run3    6000 trainedon_rebel_run3
run_both_checkpoints Qwen2.5-1.5B-Instruct-lagrange-qlora_run2 6000 trainedon_lagrange_run2
run_both_checkpoints Qwen2.5-1.5B-Instruct-wikinre-qlora_run1  6000 trainedon_wikinre_run1

# TekGen: all epochs, not just best F1 / best eval_loss
run_full_sweep Qwen2.5-1.5B-Instruct-tekgen-qlora_run1 trainedon_tekgen_run1
