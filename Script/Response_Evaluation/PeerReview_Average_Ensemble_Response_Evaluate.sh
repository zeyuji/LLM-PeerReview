#!/bin/bash
MODEL_TYPE="New_7B_Ensemble"
MODEL_ROOT="./LLM_Response/New_7B_Ensemble"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"
RESULT_ROOT="./Results"

SEED=1

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

DATASETS=(
    # "gsm8k200"
    # "math200"
    "trivia_qa"
    # "alpaca3"
)

CONSIDER_PIRORS=(
    "False"
)

EPOCHS=(
    "30"
)

TEMPERATURES=(
    "1.0"
)


for TASK_TYPE in "${TASK_TYPES[@]}"; do
    for JUDGE_MODE in "${JUDGE_MODES[@]}"; do
        for MAX_SCORE in "${MAX_SCORES[@]}"; do
            echo "=============================================="
            echo "Processing task_type: ${TASK_TYPE}"
            echo "Processing judge_mode: ${JUDGE_MODE}"
            echo "Processing max_score: ${MAX_SCORE}"
            echo "=============================================="

            for DATASET in "${DATASETS[@]}"; do
                echo "Running evaluation on dataset: ${DATASET}"


                # python -m Src.evaluate.evaluate_ensemble_alpaca \
                python -m Src.evaluate.evaluate_ensemble \
                    --dataset_config "${CONFIG_ROOT}/${DATASET}.yaml" \
                    --data_name "${DATASET}" \
                    --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
                    --response_dir "${MODEL_ROOT}/PEERREVIEW_AVERAGE/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}/${DATASET}/seed_${SEED}.jsonl" \
                    --results_dir "${RESULT_ROOT}/${MODEL_TYPE}/PEERREVIEW_AVERAGE/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}"
                
                for CONSIDER_PIROR in "${CONSIDER_PIRORS[@]}"; do
                    for EPOCH in "${EPOCHS[@]}"; do
                        for T in "${TEMPERATURES[@]}"; do
                            echo "===================================================================================================="
                            echo "Processing consider prior: $CONSIDER_PIROR, epoch: $EPOCH, temperature: $T"
                            echo "===================================================================================================="


                            # python -m Src.evaluate.evaluate_ensemble_alpaca \
                            python -m Src.evaluate.evaluate_ensemble \
                                --dataset_config "${CONFIG_ROOT}/${DATASET}.yaml" \
                                --data_name "${DATASET}" \
                                --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
                                --response_dir "${MODEL_ROOT}/PEERREVIEW_AVERAGE_OPT/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}/${DATASET}/${CONSIDER_PIROR}_${EPOCH}_${T}_seed_${SEED}.jsonl" \
                                --results_dir "${RESULT_ROOT}/${MODEL_TYPE}/PEERREVIEW_AVERAGE_OPT/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}_${CONSIDER_PIROR}_${EPOCH}_${T}"
                            
                        done
                    done
                done
            done
        done
    done
done