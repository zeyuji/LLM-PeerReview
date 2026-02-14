#!/bin/bash
MODEL_TYPE="New_7B"
MODEL_ROOT="./LLM_Response/GaC/${MODEL_TYPE}"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"
RESULT_ROOT="./Results"

DATASETS=(
    "trivia_qa"
    # "gsm8k400"
    # "math400"
    # "alpaca3"
)

# 遍历每个模型
for DATASET in "${DATASETS[@]}"; do
    echo "Running evaluation on dataset: ${DATASET}"

    python -m Src.evaluate.evaluate \
        --dataset_config "${CONFIG_ROOT}/${DATASET}.yaml" \
        --data_name "${DATASET}" \
        --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
        --response_dir "${MODEL_ROOT}/${DATASET}" \
        --results_dir "${RESULT_ROOT}/${MODEL_TYPE}/GaC"
done