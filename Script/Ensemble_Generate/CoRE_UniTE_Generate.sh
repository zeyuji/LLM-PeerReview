#!/bin/bash

set -euo pipefail

if [[ -n "${CORE_DATASETS_CSV:-}" ]]; then
    IFS=',' read -r -a DATASETS <<< "${CORE_DATASETS_CSV}"
else
    DATASETS=("gsm8k400" "math400" "trivia_qa" "alpaca3")
fi

MODELS=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "Qwen2-7B-Instruct"
)
MODEL_CSV=$(IFS=,; echo "${MODELS[*]}")

DEVICES="${CORE_DEVICES:-0,1,2,3}"
SEED="${CORE_SEED:-1}"
RESULTS_DIR="${CORE_RESULTS_DIR:-./LLM_Response/New_7B_Ensemble/CORE}"

for DATASET in "${DATASETS[@]}"; do
    echo "Running CoRE UniTE/consist-rbf on ${DATASET}"

    python -m Src.baseline.core_generate \
        --dataset_config "./Dataset_Configs/${DATASET}.yaml" \
        --data_dir "./Datasets/${DATASET}" \
        --data_name "${DATASET}" \
        --results_dir "${RESULTS_DIR}" \
        --models "${MODEL_CSV}" \
        --devices "${DEVICES}" \
        --align_method "unite" \
        --variant "consist-rbf" \
        --top_k 10 \
        --seed "${SEED}"
done
