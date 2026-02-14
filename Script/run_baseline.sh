#!/bin/bash

################################################################################
#                     run_baseline.sh - Unified Pipeline Script
# 
# Description: This script automates the complete ensemble method pipeline
#              based on the specified method name and dataset.
# Pipeline:    Scoring (PeerReview only) → Ensemble → Evaluation
#
# Note: Single-model responses and GaC responses have already been provided
#       in the ./LLM_Response/ directory, so response generation is skipped.
#
################################################################################

# ===================== COMMAND LINE ARGUMENT PARSING ==========================

# Default values
METHOD_NAME="LLM-PeerReview-Average"
DATASET_INPUT="TriviaQA"

# Function to display usage
show_usage() {
    echo "Usage: bash run_baseline.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --method METHOD    Ensemble method to use (default: LLM-PeerReview-Average)"
    echo "  -d, --dataset DATASET  Dataset to evaluate (default: TriviaQA)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Available Methods:"
    echo "  - Random"
    echo "  - Smoothie-Global"
    echo "  - Smoothie-Local"
    echo "  - Agent-Forest"
    echo "  - GaC"
    echo "  - LLM-PeerReview-Average"
    echo "  - LLM-PeerReview-Weighted"
    echo ""
    echo "Available Datasets:"
    echo "  - GSM8k"
    echo "  - MATH"
    echo "  - TriviaQA"
    echo "  - AlpacaEval"
    echo ""
    echo "Examples:"
    echo "  bash run_baseline.sh --method LLM-PeerReview-Average --dataset TriviaQA"
    echo "  bash run_baseline.sh -m Random -d GSM8k"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--method)
            METHOD_NAME="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_INPUT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# ===================== FIXED CONFIGURATION (Do Not Modify) ===================

MODEL_GROUP_SCALE="New_7B"
SEED=1
JUDGE_MODE="triple"
EMBEDDING_MODEL="./LLM_Models/Sentence_Embedding_Models/all-mpnet-base-v2"

# Model list for judge scoring
MODELS=(
    "Meta-Llama-3.1-8B-Instruct" 
    "Mistral-7B-Instruct-v0.3" 
    "Qwen2-7B-Instruct" 
    "Qwen2.5-7B-Instruct" 
)

# ===================== PATH CONFIGURATION =====================================

MODEL_ROOT="./LLM_Response"
DATASET_ROOT="./Datasets"
CONFIG_ROOT="./Dataset_Configs"
RESULT_ROOT="./Results"
# ===================== DATASET NAME MAPPING ==================================

# Map user-friendly dataset names to internal names
case "$DATASET_INPUT" in
    "GSM8k"|"gsm8k"|"GSM8K")
        DATASET="gsm8k400"
        ;;
    "MATH"|"math")
        DATASET="math400"
        ;;
    "TriviaQA"|"triviaqa"|"trivia_qa"|"TRIVIAQA")
        DATASET="trivia_qa"
        ;;
    "AlpacaEval"|"alpacaeval"|"alpaca"|"ALPACAEVAL")
        DATASET="alpaca3"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET_INPUT'"
        echo "Supported datasets: GSM8k, MATH, TriviaQA, AlpacaEval"
        exit 1
        ;;
esac

DATASET_CONFIG="${CONFIG_ROOT}/${DATASET}.yaml"

# ===================== DATASET-SPECIFIC PARAMETERS ============================

case "$DATASET" in
    "gsm8k400")
        TASK_TYPE="MATH"
        CONSIDER_PRIOR="False"
        EPOCH="30"
        MAX_SCORE="3"
        TEMPERATURE="1.0"
        USE_ALPACA_EVAL="false"
        ;;
    "math400")
        TASK_TYPE="MATH"
        CONSIDER_PRIOR="False"
        EPOCH="30"
        MAX_SCORE="3"
        TEMPERATURE="1.0"
        USE_ALPACA_EVAL="false"
        ;;
    "trivia_qa")
        TASK_TYPE="FACT"
        CONSIDER_PRIOR="False"
        EPOCH="30"
        MAX_SCORE="5"
        TEMPERATURE="1.0"
        USE_ALPACA_EVAL="false"
        ;;
    "alpaca3")
        TASK_TYPE="INST_v3"
        CONSIDER_PRIOR="True"
        EPOCH="2"
        MAX_SCORE="10"
        TEMPERATURE="1.0"
        USE_ALPACA_EVAL="true"
        ;;
    *)
        echo "Error: Internal dataset mapping error for '$DATASET'"
        exit 1
        ;;
esac

echo "=============================================="
echo "          Configuration Summary"
echo "=============================================="
echo "Method:     $METHOD_NAME"
echo "Dataset:    $DATASET"
echo "Task Type:  $TASK_TYPE"
echo "Max Score:  $MAX_SCORE"
echo "Judge Mode: $JUDGE_MODE"
if [[ "$METHOD_NAME" == "LLM-PeerReview-Weighted" ]]; then
    echo "Consider Prior: $CONSIDER_PRIOR"
    echo "Epoch:          $EPOCH"
    echo "Temperature:    $TEMPERATURE"
fi
echo "=============================================="
echo ""

# ===================== EXECUTE PIPELINE BY METHOD =============================

case "$METHOD_NAME" in
    #---------------------------------------------------------------------------
    # GaC Method: Pre-generated responses, only evaluation needed
    #---------------------------------------------------------------------------
    "GaC")
        echo ">>> [Step 1/1] Running GaC Evaluation..."
        
        RESULTS_DIR="${MODEL_ROOT}/GaC/${MODEL_GROUP_SCALE}"
        
        python -m Src.evaluate.evaluate \
            --dataset_config "${DATASET_CONFIG}" \
            --data_name "${DATASET}" \
            --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
            --response_dir "${RESULTS_DIR}/${DATASET}" \
            --results_dir "${RESULT_ROOT}/${MODEL_GROUP_SCALE}/GaC"
        ;;

    #---------------------------------------------------------------------------
    # Random Method
    #---------------------------------------------------------------------------
    "Random")
        echo ">>> [Step 1/2] Running Random Selection..."
        
        RESULTS_DIR="${MODEL_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/RANDOM"
        
        python -m Src.baseline.random \
            --dataset_config "${DATASET_CONFIG}" \
            --results_dir "${RESULTS_DIR}" \
            --model_group_scale "${MODEL_GROUP_SCALE}" \
            --seed "${SEED}"

        echo ">>> [Step 2/2] Running Evaluation..."
        
        python -m Src.evaluate.evaluate_ensemble \
            --dataset_config "${DATASET_CONFIG}" \
            --data_name "${DATASET}" \
            --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
            --response_dir "${RESULTS_DIR}/${DATASET}/seed_${SEED}.jsonl" \
            --results_dir "${RESULT_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/RANDOM"
        ;;

    #---------------------------------------------------------------------------
    # Smoothie-Global Method
    #---------------------------------------------------------------------------
    "Smoothie-Global")
        echo ">>> [Step 1/2] Running Smoothie-Global Generation..."
        
        RESULTS_DIR="${MODEL_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/SMOOTHIE-GLOBAL"
        
        python -m Src.baseline.smoothie \
            --dataset_config "${DATASET_CONFIG}" \
            --results_dir "${RESULTS_DIR}" \
            --model_group_scale "${MODEL_GROUP_SCALE}" \
            --type "sample_independent" \
            --embedding_model "${EMBEDDING_MODEL}" \
            --seed "${SEED}"

        echo ">>> [Step 2/2] Running Evaluation..."
        
        python -m Src.evaluate.evaluate_ensemble \
            --dataset_config "${DATASET_CONFIG}" \
            --data_name "${DATASET}" \
            --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
            --response_dir "${RESULTS_DIR}/${DATASET}/seed_${SEED}.jsonl" \
            --results_dir "${RESULT_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/SMOOTHIE-GLOBAL"
        ;;

    #---------------------------------------------------------------------------
    # Smoothie-Local Method
    #---------------------------------------------------------------------------
    "Smoothie-Local")
        echo ">>> [Step 1/2] Running Smoothie-Local Generation..."
        
        RESULTS_DIR="${MODEL_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/SMOOTHIE-LOCAL"
        K=1
        
        python -m Src.baseline.smoothie \
            --dataset_config "${DATASET_CONFIG}" \
            --results_dir "${RESULTS_DIR}" \
            --model_group_scale "${MODEL_GROUP_SCALE}" \
            --type "sample_dependent" \
            --k "${K}" \
            --embedding_model "${EMBEDDING_MODEL}" \
            --seed "${SEED}"

        echo ">>> [Step 2/2] Running Evaluation..."
        
        python -m Src.evaluate.evaluate_ensemble \
            --dataset_config "${DATASET_CONFIG}" \
            --data_name "${DATASET}" \
            --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
            --response_dir "${RESULTS_DIR}/${DATASET}/seed_${SEED}.jsonl" \
            --results_dir "${RESULT_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/SMOOTHIE-LOCAL"
        ;;

    #---------------------------------------------------------------------------
    # Agent-Forest Method
    #---------------------------------------------------------------------------
    "Agent-Forest")
        echo ">>> [Step 1/2] Running Agent-Forest Generation..."
        
        RESULTS_DIR="${MODEL_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/AGENT_FOREST"
        
        python -m Src.baseline.agent_forest \
            --dataset_config "${DATASET_CONFIG}" \
            --results_dir "${RESULTS_DIR}" \
            --model_group_scale "${MODEL_GROUP_SCALE}" \
            --seed "${SEED}"

        echo ">>> [Step 2/2] Running Evaluation..."
        
        python -m Src.evaluate.evaluate_ensemble \
            --dataset_config "${DATASET_CONFIG}" \
            --data_name "${DATASET}" \
            --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
            --response_dir "${RESULTS_DIR}/${DATASET}/seed_${SEED}.jsonl" \
            --results_dir "${RESULT_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/AGENT_FOREST"
        ;;

    #---------------------------------------------------------------------------
    # LLM-PeerReview-Average Method
    #---------------------------------------------------------------------------
    "LLM-PeerReview-Average")
        RESULTS_DIR="${MODEL_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/PEERREVIEW_AVERAGE/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}"
        
        echo ">>> [Step 1/3] Running Peer Review Scoring..."
        
        for MODEL in "${MODELS[@]}"; do
            echo "    Scoring with model: ${MODEL}"
            
            python -m Src.judge.judge_model \
                --dataset_config "${DATASET_CONFIG}" \
                --results_dir "${RESULTS_DIR}" \
                --model_group_scale "${MODEL_GROUP_SCALE}" \
                --seed "${SEED}" \
                --judge_model_name "${MODEL}" \
                --judge_mode "${JUDGE_MODE}" \
                --max_score "${MAX_SCORE}"
        done

        echo ">>> [Step 2/3] Running PeerReview Average Ensemble..."
        
        mkdir -p "${RESULTS_DIR}/${DATASET}"
        
        python -m Src.peerreview_average.peerreview_average \
            --dataset_config "${DATASET_CONFIG}" \
            --results_dir "${RESULTS_DIR}" \
            --model_group_scale "${MODEL_GROUP_SCALE}" \
            --seed "${SEED}"

        echo ">>> [Step 3/3] Running Evaluation..."
        
        if [ "$USE_ALPACA_EVAL" = "true" ]; then
            python -m Src.evaluate.evaluate_ensemble_alpaca \
                --dataset_config "${DATASET_CONFIG}" \
                --data_name "${DATASET}" \
                --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
                --response_dir "${RESULTS_DIR}/${DATASET}/seed_${SEED}.jsonl" \
                --results_dir "${RESULT_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/PEERREVIEW_AVERAGE/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}"
        else
            python -m Src.evaluate.evaluate_ensemble \
                --dataset_config "${DATASET_CONFIG}" \
                --data_name "${DATASET}" \
                --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
                --response_dir "${RESULTS_DIR}/${DATASET}/seed_${SEED}.jsonl" \
                --results_dir "${RESULT_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/PEERREVIEW_AVERAGE/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}"
        fi
        ;;

    #---------------------------------------------------------------------------
    # LLM-PeerReview-Weighted Method (with Truth Inference)
    #---------------------------------------------------------------------------
    "LLM-PeerReview-Weighted")
        RESULTS_DIR="${MODEL_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/PEERREVIEW_AVERAGE_OPT/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}"
        
        echo ">>> [Step 1/3] Running Peer Review Scoring..."
        
        for MODEL in "${MODELS[@]}"; do
            echo "    Scoring with model: ${MODEL}"
            
            python -m Src.judge.judge_model \
                --dataset_config "${DATASET_CONFIG}" \
                --results_dir "${RESULTS_DIR}" \
                --model_group_scale "${MODEL_GROUP_SCALE}" \
                --seed "${SEED}" \
                --judge_model_name "${MODEL}" \
                --judge_mode "${JUDGE_MODE}" \
                --max_score "${MAX_SCORE}"
        done

        echo ">>> [Step 2/3] Running PeerReview Weighted Ensemble (Truth Inference)..."
        
        mkdir -p "${RESULTS_DIR}/${DATASET}"
        
        python -m Src.peerreview_average.peerreview_average_ti \
            --dataset_config "${DATASET_CONFIG}" \
            --results_dir "${RESULTS_DIR}" \
            --model_group_scale "${MODEL_GROUP_SCALE}" \
            --seed "${SEED}" \
            --judge_mode "${JUDGE_MODE}" \
            --max_score "${MAX_SCORE}" \
            --consider_prior "${CONSIDER_PRIOR}" \
            --epoch "${EPOCH}" \
            --t "${TEMPERATURE}"

        echo ">>> [Step 3/3] Running Evaluation..."
        
        if [ "$USE_ALPACA_EVAL" = "true" ]; then
            python -m Src.evaluate.evaluate_ensemble_alpaca \
                --dataset_config "${DATASET_CONFIG}" \
                --data_name "${DATASET}" \
                --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
                --response_dir "${RESULTS_DIR}/${DATASET}/${CONSIDER_PRIOR}_${EPOCH}_${TEMPERATURE}_seed_${SEED}.jsonl" \
                --results_dir "${RESULT_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/PEERREVIEW_AVERAGE_OPT/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}_${CONSIDER_PRIOR}_${EPOCH}_${TEMPERATURE}"
        else
            python -m Src.evaluate.evaluate_ensemble \
                --dataset_config "${DATASET_CONFIG}" \
                --data_name "${DATASET}" \
                --data_dir "${DATASET_ROOT}/${DATASET}/test.jsonl" \
                --response_dir "${RESULTS_DIR}/${DATASET}/${CONSIDER_PRIOR}_${EPOCH}_${TEMPERATURE}_seed_${SEED}.jsonl" \
                --results_dir "${RESULT_ROOT}/${MODEL_GROUP_SCALE}_Ensemble/PEERREVIEW_AVERAGE_OPT/${TASK_TYPE}/${JUDGE_MODE}_${MAX_SCORE}_${CONSIDER_PRIOR}_${EPOCH}_${TEMPERATURE}"
        fi
        ;;

    #---------------------------------------------------------------------------
    # Unknown Method
    #---------------------------------------------------------------------------
    *)
        echo "Error: Unknown method '$METHOD_NAME'"
        echo "Supported methods:"
        echo "  - Random"
        echo "  - Smoothie-Global"
        echo "  - Smoothie-Local"
        echo "  - Agent-Forest"
        echo "  - GaC"
        echo "  - LLM-PeerReview-Average"
        echo "  - LLM-PeerReview-Weighted"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "          Pipeline Completed"
echo "=============================================="
echo "Method:  $METHOD_NAME"
echo "Dataset: $DATASET"
echo "=============================================="
