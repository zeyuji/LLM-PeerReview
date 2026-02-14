#!/bin/bash
MODEL_GROUP_SCALE="New_7B"
SEED=1
BASE_RESULTS_DIR="./LLM_Response/New_7B_Ensemble/PEERREVIEW_AVERAGE_OPT"

# mkdir -p "$BASE_RESULTS_DIR"

DATASETS=(
    "trivia_qa" 
    # "gsm8k400"
    # "math400"
    # "alpaca3"
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
    "3"
    # "5"
    # "7"
    # "10"
)

TASK_TYPES=(
    # "FACT"
    # "INST_v3"
    "MATH"
)

for DATASET in "${DATASETS[@]}"; do
    echo "=============================================="
    echo "Processing dataset: $DATASET"
    echo "=============================================="
    

    DATASET_CONFIG="./Dataset_Configs/${DATASET}.yaml"

    for TASK_TYPE in "${TASK_TYPES[@]}"; do
        for JUDGE_MODE in "${JUDGE_MODES[@]}"; do
            for MAX_SCORE in "${MAX_SCORES[@]}"; do
                echo "===================================================================================================="
                echo "Processing task type: $TASK_TYPE, judge mode: $JUDGE_MODE, max score: $MAX_SCORE"
                echo "===================================================================================================="

                DATASET_RESULTS_DIR="${BASE_RESULTS_DIR}/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}/${DATASET}"
                mkdir -p "$DATASET_RESULTS_DIR"
                
                # Configure parameters based on dataset characteristics:
                # The AlpacaEval dataset is an instruction-following task without ground truth answers,
                # whereas the other three datasets (TriviaQA, GSM8k, MATH) have definitive ground truth.
                
                if [ "$DATASET" = "alpaca3" ]; then
                    CONSIDER_PIROR="True"
                    EPOCH="2"
                else
                    CONSIDER_PIROR="False"
                    EPOCH="30"
                fi
                
                TEMPERATURE="1.0"
                
                echo "===================================================================================================="
                echo "Processing dataset: $DATASET - consider prior: $CONSIDER_PIROR, epoch: $EPOCH, temperature: $TEMPERATURE"
                echo "===================================================================================================="

                python -m Src.peerreview_average.peerreview_average_ti \
                    --dataset_config "${DATASET_CONFIG}" \
                    --results_dir "${BASE_RESULTS_DIR}/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}" \
                    --model_group_scale "${MODEL_GROUP_SCALE}" \
                    --seed "${SEED}" \
                    --judge_mode "${JUDGE_MODE}" \
                    --max_score "${MAX_SCORE}" \
                    --consider_prior "${CONSIDER_PIROR}" \
                    --epoch "${EPOCH}" \
                    --t "${TEMPERATURE}"
            done
        done
        echo "Completed processing $DATASET"
        echo "----------------------------------------------------"
        echo ""
    done
done