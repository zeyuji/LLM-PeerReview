#!/bin/bash

set -euo pipefail

MODEL_GROUP_SCALE="New_7B"
RESULTS_DIR="./LLM_Response/Debate"
CONFIG_ROOT="./Dataset_Configs"

DATASETS=(
    "gsm8k400"
    "math400"
)

ROUNDS=(0 1 2)

for DATASET in "${DATASETS[@]}"; do
    for ROUND_IDX in "${ROUNDS[@]}"; do
        echo "Evaluating MAD ${DATASET} round_${ROUND_IDX}"

        python -m Src.evaluate.evaluate_debate \
            --dataset_config "${CONFIG_ROOT}/${DATASET}.yaml" \
            --results_dir "${RESULTS_DIR}" \
            --model_group_scale "${MODEL_GROUP_SCALE}" \
            --round_idx "${ROUND_IDX}"
    done
done
