"""
peerreview_average.py

This script implements the PeerReview-Average ensemble evaluation method:
1. Loads generations from multiple models on a given dataset.
2. Each model scores generations from all models (including itself).
3. Computes average score for each generation across all judges.
4. Selects the highest-scoring generation per sample.
5. Saves full score cube, average scores, and selected outputs.

"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
import json

# Custom utility imports
from Utils.util import (
    load_model_group_response, 
    load_reference,
    load_multi_model_prompt,
    load_data_config
)
from Utils.constants import (
    MODEL_GROUPS,
)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_config", type=str, help="Path to the dataset YAML config.")
parser.add_argument("--results_dir", type=str, help="Directory to save final outputs.")
parser.add_argument("--model_group_scale", type=str, choices=["New_7B", "New_13B"], help="Model group scale.")
parser.add_argument("--seed", type=int, help="Random seed for selection.")
parser.add_argument(
    "--prompt_template",
    type=str,
    choices=["single", "multi", "double", "triple", "double_bias", "triple_bias", "multi_bias"],
    default="single",
    help="Prompt template for scoring."
)

def run_peerreview_average(
    seed: int,
    output_fpath: Path,
    data_config: dict,
    model_group_scale: str,
    model_group: List[str],
    prompt_template: str,
):
    """
    Run PeerReview-Average scoring and selection.

    Args:
        seed (int): Random seed for reproducibility.
        output_fpath (Path): File path to save final generations.
        data_config (dict): Dataset configuration.
        model_group_scale (str): Name of the model group.
        model_group (List[str]): List of model identifiers.
        prompt_template (str): Prompt template name.
    """

    scores_dir = output_fpath.parent / "scores_matrices"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Load generations, references, and prompts
    test_generations = load_model_group_response(
        response_path=f"./LLM_Response/Test/{model_group_scale}",
        model_group=model_group,
        data_name=data_config["dataset"],
        seed=seed
    )
    test_references = load_reference(f"./Datasets/{data_config['dataset']}/test.jsonl")
    test_prompts = load_multi_model_prompt(f"./Datasets/{data_config['dataset']}/test.jsonl")

    n_models = len(test_generations)
    n_samples = len(test_generations[0])
    
    n_judges = 3 # Assuming 3 models act as judges
    wo_model_idx = 1 # Index of the model to exclude as a judge (if no model without, set idx max)
    
    n_judges = n_models
    wo_model_idx = 100

    # n_judges = 5
    # wo_model_idx = 100

    # Initialize [samples, generators, judges]
    # all_scores = np.zeros((n_samples, n_models, n_models), dtype=np.float32)
    all_scores = np.zeros((n_samples, n_models, n_judges), dtype=np.float32)

    # Score generations with each judge model
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
            raise FileNotFoundError(f"Score file missing: {json_filename}. Please generate it first.")

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

    # Save full score cube
    full_scores_json = {
        "metadata": {
            "seed": seed,
            "dataset": data_config["dataset"],
            "model_group": model_group,
            "dimensions": ["sample_idx", "generator_idx", "judge_idx"]
        },
        "scores": all_scores.tolist()
    }

    full_scores_filename = scores_dir / f"full_scores_cube_seed_{seed}.json"
    with open(full_scores_filename, 'w') as f:
        json.dump(full_scores_json, f, indent=2)

    # Save average scores
    avg_scores = np.mean(all_scores, axis=2)
    avg_scores_json = {
        "metadata": {
            "seed": seed,
            "dataset": data_config["dataset"],
            "model_group": model_group,
            "description": "Average scores across all judges"
        },
        "scores": avg_scores.tolist()
    }

    avg_scores_filename = scores_dir / f"avg_scores_seed_{seed}.json"
    with open(avg_scores_filename, 'w') as f:
        json.dump(avg_scores_json, f, indent=2)

    # Select best generation per sample based on average score
    final_generations = []
    best_model_indices = []

    for sample_idx in range(n_samples):
        scores = avg_scores[sample_idx]
        max_score = np.max(scores)
        max_indices = np.where(scores == max_score)[0]
        best_model_idx = np.random.choice(max_indices)
        best_model_indices.append(best_model_idx)
        final_generations.append(test_generations[best_model_idx][sample_idx])

    # Save final selected generations
    results_lines = [
        {
            "task_name": data_config["dataset"],
            "generation": text,
            "idx": idx,
            "selected_model": int(best_model_idx)
        }
        for idx, (text, best_model_idx) in enumerate(zip(final_generations, best_model_indices))
    ]

    with open(output_fpath, 'w', encoding='utf-8') as f:
        for line in results_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def main(args: argparse.Namespace):
    """
    Main entry point for PeerReview-Average experiment.
    """
    data_config = load_data_config(args.dataset_config)
    model_groups = MODEL_GROUPS

    output_dir = Path(args.results_dir) / data_config["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fpath = output_dir / f"seed_{args.seed}.jsonl"

    run_peerreview_average(
        seed=args.seed,
        output_fpath=output_fpath,
        data_config=data_config,
        model_group_scale=args.model_group_scale,
        model_group=model_groups[args.model_group_scale],
        prompt_template=args.prompt_template,
    )


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)
