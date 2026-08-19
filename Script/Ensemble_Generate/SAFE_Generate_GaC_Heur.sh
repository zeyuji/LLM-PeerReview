#!/bin/bash

set -euo pipefail

DATASETS=(
    "gsm8k400"
    "math400"
    "trivia_qa"
    "alpaca3"
)

SAFE_PROFILE="safe4_new7b_gac_heur"
SEED=1
BASE_RESULTS_DIR="./LLM_Response/New_7B_Ensemble/SAFE"
DEVICE_IDS=(0 1 2 3)

for DATASET in "${DATASETS[@]}"; do
    echo "Running ${SAFE_PROFILE} on ${DATASET}"

    python -m Src.baseline.safe_generate \
        --dataset_config "./Dataset_Configs/${DATASET}.yaml" \
        --data_dir "./Datasets/${DATASET}" \
        --results_dir "${BASE_RESULTS_DIR}" \
        --profile "${SAFE_PROFILE}" \
        --seed "${SEED}" \
        --device_ids "${DEVICE_IDS[@]}"
done
