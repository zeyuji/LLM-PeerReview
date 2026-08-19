#!/bin/bash

set -euo pipefail

MODEL_GROUP_SCALE="New_7B"
RESULTS_DIR="./LLM_Response/Debate"
CONFIG_ROOT="./Dataset_Configs"
N_DEBATE_ROUNDS=2
SEED=1
DEVICE="cuda"

DATASETS=(
    "gsm8k400"
    "math400"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Running MAD on ${DATASET}: round 0 plus ${N_DEBATE_ROUNDS} debate rounds"

    python -m Src.baseline.debate \
        --dataset_config "${CONFIG_ROOT}/${DATASET}.yaml" \
        --results_dir "${RESULTS_DIR}" \
        --model_group_scale "${MODEL_GROUP_SCALE}" \
        --n_debate_rounds "${N_DEBATE_ROUNDS}" \
        --seed "${SEED}" \
        --device "${DEVICE}"
done
