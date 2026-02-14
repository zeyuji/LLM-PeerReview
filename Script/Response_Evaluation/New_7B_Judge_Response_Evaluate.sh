#!/bin/bash
MODEL_TYPE="New_7B_Ensemble"
MODEL_ROOT="./LLM_Response/New_7B_Ensemble"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"
RESULT_ROOT="./Results/New_7B_Judge"

MODELS=(
    "Meta-Llama-3.1-8B-Instruct" 
    "Mistral-7B-Instruct-v0.3" 
    "Qwen2-7B-Instruct" 
    "Qwen2.5-7B-Instruct" 
)

JUDGE_MODES=(
    # "single"
    # "double"
    "triple"
    # "multi"
)

MAX_SCORES=(
    "3"
    "5"
    "7"
    "10"
)

TASK_TYPES=(
    "FACT"
    # "INST_v3"
    # "MATH"
)

DATASETS=(
    "trivia_qa"
    # "gsm8k200"
    # "math200"
    # "alpaca3"
)

for MODEL_NAME in "${MODELS[@]}"; do
    echo "=============================================="
    echo "Processing model: ${MODEL_NAME}"
    echo "=============================================="

    for TASK_TYPE in "${TASK_TYPES[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            for JUDGE_MODE in "${JUDGE_MODES[@]}"; do
                for MAX_SCORE in "${MAX_SCORES[@]}"; do
                    echo "Running evaluation on dataset: ${DATASET} with judge mode: ${JUDGE_MODE}  max score: ${MAX_SCORE}"

                    # python -m Src.evaluate.evaluate_judge \
                    python -m Src.evaluate.evaluate_judge_alpaca \
                        --dataset_config "${CONFIG_ROOT}/${DATASET}.yaml" \
                        --data_name "${DATASET}" \
                        --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
                        --response_dir "${MODEL_ROOT}/PEERREVIEW_AVERAGE/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}/${DATASET}/judge_results" \
                        --results_dir "${RESULT_ROOT}/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}/${MODEL_NAME}" \
                        --judge_model_name "${MODEL_NAME}"
                done
            done
        done
    done
done