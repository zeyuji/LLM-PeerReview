#!/bin/bash
MODEL_GROUP_SCALE="New_7B"
SEED=1
BASE_RESULTS_DIR="./LLM_Response/New_7B_Ensemble/PEERREVIEW_AVERAGE"

DATASETS=(
    "trivia_qa" 
    # "gsm8k400"
    # "math400"
    # "alpaca3"
)

TASK_TYPES=(
    "FACT"
    # "INST_v3"
    # "MATH"
)

JUDGE_MODES=(
    # "single"
    # "double"
    "triple"
    # "multi"

    # "double_bias"
    # "triple_bias"
    # "multi_bias"
)

MAX_SCORES=(
    # "3"
    "5"
    # "7"
    # "10"
)


for DATASET in "${DATASETS[@]}"; do
    echo "=============================================="
    echo "Processing dataset: $DATASET"
    echo "=============================================="
    

    DATASET_CONFIG="./Dataset_Configs/${DATASET}.yaml"

    for TASK_TYPE in "${TASK_TYPES[@]}"; do
        for JUDGE_MODE in "${JUDGE_MODES[@]}"; do
            for MAX_SCORE in "${MAX_SCORES[@]}"; do
                echo "Processing task type: $TASK_TYPE, judge mode: $JUDGE_MODE, max score: $MAX_SCORE"

                DATASET_RESULTS_DIR="${BASE_RESULTS_DIR}/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}/${DATASET}"
                mkdir -p "$DATASET_RESULTS_DIR"
                
                python -m Src.peerreview_average.peerreview_average \
                    --dataset_config "${DATASET_CONFIG}" \
                    --results_dir "${BASE_RESULTS_DIR}/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}" \
                    --model_group_scale "${MODEL_GROUP_SCALE}" \
                    --seed "${SEED}" \
                
                echo ""
            done
        done
        echo "Completed processing $DATASET"
        echo "----------------------------------------------------"
        echo ""
    done
done