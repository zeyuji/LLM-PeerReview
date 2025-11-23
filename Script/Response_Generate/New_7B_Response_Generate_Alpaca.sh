#!/bin/bash

# Paths for models, responses, datasets, and configurations
MODEL_TYPE="New_7B"
MODEL_ROOT="./LLM_Models/${MODEL_TYPE}"
RESPONSE_ROOT="./LLM_Response"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"

# Models and datasets to process
MODELS=("Meta-Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.3" "Qwen2-7B-Instruct" "Qwen2.5-7B-Instruct")
DATASETS=("alpaca3")

# Generation parameters
DEVICE="cuda:0"
N_GENERATIONS=1
TEMPERATURE=0.7
TOP_P=0.8
SEED=42

# Process each model and dataset
for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
    
    for DATASET in "${DATASETS[@]}"; do
        CONFIG_PATH="${CONFIG_ROOT}/${DATASET}.yaml"
        DATA_DIR="${DATASET_ROOT}/${DATASET}"
        TEST_RESULTS_DIR="${RESPONSE_ROOT}/Test/${MODEL_TYPE}/${MODEL_NAME}"

        python -m Src.model_generate.model_generate \
            --model "${MODEL_PATH}" \
            --device "${DEVICE}" \
            --dataset_config "${CONFIG_PATH}" \
            --data_dir "${DATA_DIR}" \
            --data_name "${DATASET}" \
            --results_dir "${TEST_RESULTS_DIR}" \
            --test_or_train "test" \
            --n_generations "${N_GENERATIONS}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --seed "${SEED}"
    done
done
