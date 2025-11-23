# MODELS=("Gemma-7B" "Llama-2" "Mistral" "Nous-Capybara" "Vicuna")
MODELS=(
    # "Prometheus-2-7B"
    "Meta-Llama-3.1-8B-Instruct" 
    "Mistral-7B-Instruct-v0.3" 
    "Qwen2-7B-Instruct" 
    "Qwen2.5-7B-Instruct" 
)
JUDGE_MODES=(
    "single"
    "double"
    "triple"
    "multi"
)
MAX_SCORES=(
    "3"
    "5"
    # "10"
)
DATASET="gsm8k400"
TASK_TYPE="MATH"


for MODEL in "${MODELS[@]}"; do
    for JUDGE_MODE in "${JUDGE_MODES[@]}"; do
        for MAX_SCORE in "${MAX_SCORES[@]}"; do
            echo "==============================================="
            echo "MATH: MODEL: ${MODEL}, JUDGE_MODE: ${JUDGE_MODE}, MAX_SCORE: ${MAX_SCORE}"
            echo "==============================================="

            python -m Src.judge.judge_model \
                --dataset_config ./Dataset_Configs/${DATASET}.yaml \
                --results_dir "./LLM_Response/New_7B_Ensemble/PEERREVIEW_AVERAGE/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}" \
                --model_group_scale "New_7B" \
                --seed 1 \
                --judge_model_name "${MODEL}" \
                --judge_mode "${JUDGE_MODE}" \
                --max_score "${MAX_SCORE}"
        done
    done
done
    
    