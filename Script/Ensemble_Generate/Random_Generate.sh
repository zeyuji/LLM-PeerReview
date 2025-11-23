#!/bin/bash

DATASETS=(
    "trivia_qa" 
    # "gsm8k400"
    # "math400"
    # "alpaca3"
)

MODEL_GROUP_SCALE="New_7B"
SEED=1
BASE_RESULTS_DIR="./LLM_Response/New_7B_Ensemble/RANDOM"

# mkdir -p "$BASE_RESULTS_DIR"

for DATASET in "${DATASETS[@]}"; do
    echo "=============================================="
    echo "Processing dataset: $DATASET"
    echo "=============================================="
    
    DATASET_CONFIG="./Dataset_Configs/${DATASET}.yaml"
    DATASET_RESULTS_DIR="${BASE_RESULTS_DIR}/${DATASET}"
    # mkdir -p "$DATASET_RESULTS_DIR"
    
    python -m Src.baseline.random \
        --dataset_config "${DATASET_CONFIG}" \
        --results_dir "${BASE_RESULTS_DIR}" \
        --model_group_scale "${MODEL_GROUP_SCALE}" \
        --seed "${SEED}"
    
    echo ""
    echo "Completed processing $DATASET"
    echo "----------------------------------------------------"
    echo ""
done