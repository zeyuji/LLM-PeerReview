#!/bin/bash

set -euo pipefail

MODELS=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "Qwen2-7B-Instruct"
)
MODEL_TAG=$(printf "%s__" "${MODELS[@]}")
MODEL_TAG=${MODEL_TAG%__}
MAIN_MODEL="${MODELS[0]}"

ALIGN_METHODS=("gac" "unite")
DATASETS=("gsm8k400" "math400" "trivia_qa")
SEED="${CORE_SEED:-1}"
RESPONSE_ROOT="${CORE_RESULTS_DIR:-./LLM_Response/New_7B_Ensemble/CORE}"
RESULT_ROOT="${CORE_EVAL_RESULTS_DIR:-./Results/New_7B_Ensemble/CORE}"

for ALIGN_METHOD in "${ALIGN_METHODS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        RESPONSE_FILE="${RESPONSE_ROOT}/${ALIGN_METHOD}/consist-rbf/main_${MAIN_MODEL}/models_${MODEL_TAG}/${DATASET}/seed_${SEED}.jsonl"
        RESULTS_DIR="${RESULT_ROOT}/${ALIGN_METHOD}/consist-rbf/main_${MAIN_MODEL}/models_${MODEL_TAG}"

        echo "Evaluating CoRE ${ALIGN_METHOD}/consist-rbf on ${DATASET}"

        python -m Src.evaluate.evaluate_ensemble \
            --dataset_config "./Dataset_Configs/${DATASET}.yaml" \
            --data_name "${DATASET}" \
            --data_dir "./Datasets/${DATASET}/test.jsonl" \
            --response_dir "${RESPONSE_FILE}" \
            --results_dir "${RESULTS_DIR}"
    done
done
