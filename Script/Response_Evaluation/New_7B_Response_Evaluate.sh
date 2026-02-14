#!/bin/bash
MODEL_TYPE="New_7B"
MODEL_ROOT="./LLM_Response/Test/${MODEL_TYPE}"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"
RESULT_ROOT="./Results"

MODELS=(
    "Meta-Llama-3.1-8B-Instruct" 
    "Mistral-7B-Instruct-v0.3" 
    "Qwen2-7B-Instruct" 
    "Qwen2.5-7B-Instruct" 
)

DATASETS=(
    "trivia_qa"
    "gsm8k400"
    "math400"
)

for MODEL_NAME in "${MODELS[@]}"; do
    echo "Processing model: ${MODEL_NAME}"

    for DATASET in "${DATASETS[@]}"; do
        echo "Running evaluation on dataset: ${DATASET}"

        python -m Src.evaluate.evaluate \
            --dataset_config "${CONFIG_ROOT}/${DATASET}.yaml" \
            --data_name "${DATASET}" \
            --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
            --response_dir "${MODEL_ROOT}/${MODEL_NAME}/${DATASET}" \
            --results_dir "${RESULT_ROOT}/${MODEL_TYPE}/${MODEL_NAME}"
    done
done