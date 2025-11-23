#!/bin/bash
MODEL_TYPE="New_7B"
MODEL_ROOT="./LLM_Response/GaC/${MODEL_TYPE}"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"
RESULT_ROOT="./Results"

# DATASETS=("cnn_dailymail" "e2e_nlg")
# DATASETS=("acc_group" "cnn_dailymail" "definition_extraction" "e2e_nlg" "rouge2_group" "squad" "trivia_qa" "web_nlg" "xsum")
DATASETS=(
    # "definition_extraction200" 
    # "squad"
    # "trivia_qa200"
    "trivia_qa"
    # "mix_instruct_1000"
    # "gsm8k200"
    "gsm8k400"
    # "math200"
    "math400"
    # "human_eval"
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