#!/bin/bash

set -euo pipefail

MODEL_ROOT="./LLM_Response/New_7B_Ensemble/SAFE"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"
RESULT_ROOT="./Results/New_7B_Ensemble/SAFE"
SEED=1

SAFE_PROFILES=(
    "safe4_new7b_gac_heur"
    "safe4_new7b_unite_heur"
)

DATASETS=(
    "gsm8k400"
    "math400"
    "trivia_qa"
)

for DATASET in "${DATASETS[@]}"; do
    for SAFE_PROFILE in "${SAFE_PROFILES[@]}"; do
        echo "Evaluating ${SAFE_PROFILE} on ${DATASET}"

        python -m Src.evaluate.evaluate_ensemble \
            --dataset_config "${CONFIG_ROOT}/${DATASET}.yaml" \
            --data_name "${DATASET}" \
            --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
            --response_dir "${MODEL_ROOT}/${SAFE_PROFILE}/${DATASET}/seed_${SEED}.jsonl" \
            --results_dir "${RESULT_ROOT}/${SAFE_PROFILE}"
    done
done
