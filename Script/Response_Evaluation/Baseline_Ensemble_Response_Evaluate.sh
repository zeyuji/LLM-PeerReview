#!/bin/bash
MODEL_TYPE="New_7B_Ensemble"
MODEL_ROOT="./LLM_Response/New_7B_Ensemble"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"
RESULT_ROOT="./Results"
SEED=1

ENSEMBLE_MODELS=(
    "RANDOM"
    "AGENT_FOREST"
    "SMOOTHIE-GLOBAL"
    "SMOOTHIE-LOCAL"
)
DATASETS=(
    "trivia_qa"
    # "gsm8k400"
    # "math400"
    # "alpaca3"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Running evaluation on dataset: ${DATASET}"
    
    for MODEL_NAME in "${ENSEMBLE_MODELS[@]}"; do
        echo "=============================================="
        echo "Processing model: ${MODEL_NAME}"
        echo "=============================================="

        python -m Src.evaluate.evaluate_ensemble \
            --dataset_config "${CONFIG_ROOT}/${DATASET}.yaml" \
            --data_name "${DATASET}" \
            --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
            --response_dir "${MODEL_ROOT}/${MODEL_NAME}/${DATASET}"/seed_${SEED}.jsonl \
            --results_dir "${RESULT_ROOT}/${MODEL_TYPE}/${MODEL_NAME}"
    done
done