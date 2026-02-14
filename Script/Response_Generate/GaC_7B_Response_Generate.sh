#!/bin/bash

# Start GaC server before running this script
# conda activate gac_env
# cd GaC
# python gac_api_server.py --config-path example_configs/4_ensemble_every_step.yaml --host 0.0.0.0 --port 8000

# Define paths for models, responses, datasets, and configurations
MODEL_TYPE="New_7B"
MODEL_ROOT="./LLM_Models/${MODEL_TYPE}"
RESPONSE_ROOT="./LLM_Response"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"

DATASETS=(
    "alpaca3"
    "trivia_qa" 
    "gsm8k400" 
    "math400"
)

# Set generation parameters
N_GENERATIONS=1
SEED=42

# Process each model and dataset
for DATASET in "${DATASETS[@]}"; do
    CONFIG_PATH="${CONFIG_ROOT}/${DATASET}.yaml"
    DATA_DIR="${DATASET_ROOT}/${DATASET}"
    TEST_RESULTS_DIR="${RESPONSE_ROOT}/GaC/${MODEL_TYPE}"

    python -m Src.baseline.gac_generate \
        --dataset_config "${CONFIG_PATH}" \
        --data_dir "${DATA_DIR}" \
        --data_name "${DATASET}" \
        --results_dir "${TEST_RESULTS_DIR}" \
        --test_or_train "test" \
        --n_generations "${N_GENERATIONS}" \
        --seed "${SEED}"
done
