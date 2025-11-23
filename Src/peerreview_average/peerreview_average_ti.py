"""
peerreview_average_opt.py

This script implements a peer-review-based ensemble method for selecting the best generation 
from multiple large language models (LLMs). Each model evaluates the outputs of all others,
and the scores are optimized using truth inference. The highest-scoring generation is selected 
for each sample.

Workflow:
1. Load model generations, reference answers, and prompts.
2. For each judge model, load its score matrix if previously computed.
3. Use truth inference to optimize the score tensor.
4. Select the best generation per sample.
5. Save selected generations and metadata.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

from Utils.peerreview_util import optimize_scores
from Utils.util import (
    load_model_group_response,
    load_data_config,
)
from Utils.constants import MODEL_GROUPS

# Argument parser definition
parser = argparse.ArgumentParser(description="Run peer-review-based ensemble optimization.")

parser.add_argument("--dataset_config", type=str, help="Path to the dataset YAML configuration.")
parser.add_argument("--results_dir", type=str, help="Directory to save results.")
parser.add_argument("--model_group_scale", type=str, choices=["New_7B", "New_13B"], help="Model group scale.")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max_score", type=int, required=True, help="Maximum score for judge evaluation.")
parser.add_argument(
    "--judge_mode", 
    type=str,
    choices=["single", "multi", "double", "triple", "double_bias", "triple_bias", "multi_bias"],
    default="single",
    help="Prompt type for judging (single/multi/double/etc.)."
)
parser.add_argument(
    "--consider_prior", 
    type=str, 
    choices=["True", "False"], 
    default="False",
    help="Whether to use prior knowledge in optimization."
)
parser.add_argument("--epoch", type=str, required=True, help="Number of epochs for optimization.")
parser.add_argument("--t", type=float, default=0.7, help="Temperature parameter for optimization.")


def run_peerreview_average_opt(
    seed: int,
    output_fpath: Path,
    data_config: dict,
    model_group_scale: str,
    model_group: List[str],
    max_score: int,
    consider_prior: str,
    epoch: int,
    t: float,
):
    """
    Execute the peer-review ensemble optimization.

    Args:
        seed (int): Random seed for reproducibility.
        output_fpath (Path): File path to save final selected generations.
        data_config (dict): Dataset configuration dictionary.
        model_group_scale (str): Identifier for model group size.
        model_group (List[str]): List of model names.
        max_score (int): Maximum score used during evaluation.
        consider_prior (str): Whether to consider prior in optimization.
        epoch (int): Epochs for score optimization.
        t (float): Temperature parameter for softmax-based optimization.
    """
    # Directory for intermediate score matrices
    scores_dir = output_fpath.parent / "scores_matrices"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Load generated outputs from all models
    test_generations = load_model_group_response(
        response_path=f"./LLM_Response/Test/{model_group_scale}",
        model_group=model_group,
        data_name=data_config["dataset"],
        seed=seed
    )

    n_models = len(test_generations)
    n_samples = len(test_generations[0])
    
    n_judges = 3 # Assuming 3 models act as judges
    wo_model_idx = 1 # Index of the model to exclude as a judge (if no model without, set idx max)
    
    n_judges = n_models
    wo_model_idx = 100

    # n_judges = 5
    # wo_model_idx = 100

    # Score tensor: [num_samples, num_models, num_judges]
    # all_scores = np.zeros((n_samples, n_models, n_models), dtype=np.float32)
    all_scores = np.zeros((n_samples, n_models, n_judges), dtype=np.float32)

    # Load precomputed score matrices for each judge model
    for judge_idx in tqdm(range(n_models), desc="Scoring with judge models"):
    # for judge_idx in tqdm(range(n_judges), desc="Scoring with judge models"):
        if judge_idx == wo_model_idx: # without one model
            print(f"Skipping judge model at index {wo_model_idx}")
            continue

        judge_name = model_group[judge_idx]
        judge_name_safe = judge_name.replace("/", "_")
        json_filename = scores_dir / f"judge_{judge_name_safe}_seed_{seed}.json"

        if json_filename.exists():
            with open(json_filename, 'r') as f:
                data = json.load(f)
            if judge_idx < wo_model_idx:
                all_scores[:, :, judge_idx] = np.array(data["scores"])
            else:
                all_scores[:, :, judge_idx-1] = np.array(data["scores"])
        else:
            raise FileNotFoundError(f"Missing score file: {json_filename}. Please generate it first.")

    # 5 judges
    # new_judge_name = "Ministral-8B-Instruct-2410"
    # new_judge_name = "Meta-Llama-3-8B-Instruct"
    # new_judge_name = "Qwen2.5-7B-Instruct-1M"
    # new_json_filename = scores_dir / f"judge_{new_judge_name}_seed_{seed}.json"
    # if new_json_filename.exists():
    #     with open(new_json_filename, 'r') as f:
    #         data = json.load(f)
    #     all_scores[:, :, 4] = np.array(data["scores"])
    # else:
    #     raise FileNotFoundError(f"Score file missing: {new_json_filename}. Please generate it first.")

    # Optimize scores using truth inference
    optimized_array = optimize_scores(
        all_scores,
        n_samples,
        n_models,
        n_judges,
        max_score,
        consider_prior,
        epoch,
        t
    )

    # Save optimized score matrix
    optimized_scores_filename = scores_dir / f"optimized_scores_seed_{seed}.json"
    optimized_scores_json = {
        "metadata": {
            "seed": seed,
            "dataset": data_config["dataset"],
            "model_group": model_group,
            "description": "Optimized scores using truth inference"
        },
        "scores": optimized_array.tolist()
    }

    with open(optimized_scores_filename, 'w') as f:
        json.dump(optimized_scores_json, f, indent=2)
    print(f"Saved optimized scores to {optimized_scores_filename}")

    final_generations = []
    best_model_indices = []

    # For each sample, select the generation with the highest optimized score
    for sample_idx in range(n_samples):
        sample_scores = optimized_array[sample_idx]
        top_score = np.max(sample_scores)
        top_indices = np.where(sample_scores == top_score)[0]
        selected_idx = np.random.choice(top_indices)
        best_model_indices.append(selected_idx)
        final_generations.append(test_generations[selected_idx][sample_idx])

    # Construct and save result lines
    results_lines = [
        {
            "task_name": data_config["dataset"],
            "generation": generation,
            "idx": idx,
            "selected_model": int(selected_idx)
        }
        for idx, (generation, selected_idx) in enumerate(zip(final_generations, best_model_indices))
    ]

    print(f"Saving results to {output_fpath}")
    with open(output_fpath, 'w', encoding='utf-8') as f:
        for line in results_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def main(args: argparse.Namespace):
    # Load dataset configuration and model group info
    data_config = load_data_config(args.dataset_config)
    model_groups = MODEL_GROUPS

    output_fpath = Path(args.results_dir) / data_config["dataset"]
    output_fpath.mkdir(parents=True, exist_ok=True)

    output_file = output_fpath / f"{args.consider_prior}_{args.epoch}_{args.t}_seed_{args.seed}.jsonl"

    run_peerreview_average_opt(
        seed=args.seed,
        output_fpath=output_file,
        data_config=data_config,
        model_group_scale=args.model_group_scale,
        model_group=model_groups[args.model_group_scale],
        max_score=int(args.max_score),
        consider_prior=args.consider_prior,
        epoch=int(args.epoch),
        t=float(args.t),
    )


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)
