import argparse
from pathlib import Path
from transformers import set_seed
import jsonlines
import torch
import gc
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import requests

from Utils.util import load_data_config, construct_dataset_path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config",
    type=str,
    help="Path to dataset config file. This should be a yaml file.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    help="Directory with data files",
)
parser.add_argument(
    "--data_name",
    type=str,
    help="Name of data",
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--test_or_train",
    default="test",
    type=str,
    help="Whether to generate predictions for test or train data. Default is test.",
)
parser.add_argument(
    "--n_generations",
    default=1,
    type=int,
    help="For each model we produce n_generations per sample. Default is 1.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Seed for random number generator.",
)

def main(args: argparse.Namespace):
    """
    Main Function
    """
    data_config = load_data_config(args.dataset_config)
    data_path = construct_dataset_path(data_dir=args.data_dir, test_or_train=args.test_or_train)
    output_path = Path(args.results_dir + "/" + args.data_name)
    # max_length = HF_MODEL_MAX_LENGTHS[args.model]

    if args.n_generations > 1:
        set_seed(args.seed)
        assert args.temperature != 0

    gac_generate_predictions(
        max_new_tokens=data_config["max_new_tokens"],
        n_generations=args.n_generations,
        data_path=data_path,
        output_fpath=output_path,
    )

def gac_generate_predictions(
    n_generations: int,
    max_new_tokens: int,
    data_path: str,
    output_fpath: Path,
):
    """
    Generates predictions for a dataset using a pretrained model and saves them to separate directories for each seed.

    Args:
        n_generations (int): Number of generations per sample.
        max_new_tokens (int): The maximum number of new tokens to generate.
        data_path (str): Path to the dataset.
        output_fpath (Path): Directory to save the generated outputs.
    """

    # Check if the results file already exists and determine how many lines have been generated
    existing_lines = 0
    if output_fpath.exists():
        existing_lines = len((output_fpath / "Seed-1" / "seed_1.jsonl").read_text().splitlines())
        print(f"Results file {output_fpath} exists. Existing lines: {existing_lines}")
    else:
        print(f"Will save results to: {output_fpath}")

    # Create subdirectories for each seed/generation
    output_fpath.mkdir(parents=True, exist_ok=True)
    seed_dirs = [output_fpath / f"Seed-{i}" for i in range(1, n_generations + 1)]
    for seed_dir in seed_dirs:
        seed_dir.mkdir(exist_ok=True)

    # Load the dataset
    with jsonlines.open(data_path) as file:
        dataset = list(file.iter())

    # Skip if all data has been processed
    if existing_lines == len(dataset):
        print(f"Results already processed. Skipping.")
        return

    # Initialize progress bar for dataset processing
    progress_bar = tqdm(dataset[existing_lines:], desc="Generating outputs")

    for sample in dataset[existing_lines:]:
        prompt = sample["multi_model_prompt"]
        task_name = sample["task_name"]
        idx = sample["idx"]

        # Generate texts for the current sample
        texts = gac_generate_per_sample(
            max_new_tokens=max_new_tokens,
            n_generations=n_generations,
            prompt=prompt,
        )
        
        # Save the generated outputs for each seed
        for i, text in enumerate(texts):
            output = {"task_name": task_name, "generation": text, "idx": idx}
            output_path = seed_dirs[i] / f"seed_{i + 1}.jsonl"
            with jsonlines.open(output_path, "a") as writer:
                writer.write(output)

        progress_bar.update(1)

def gac_generate_per_sample(
    max_new_tokens: int,
    n_generations: int,
    prompt: str,
):
    """
    Generate multiple texts for a single prompt using a pretrained model.

    Args:
        max_new_tokens (int): The maximum number of new tokens to generate.
        n_generations (int): Number of generations to produce.
        prompt (str): The input prompt for generation.
    """

    url = "http://0.0.0.0:8000/api/generate/"

    data = {
        "messages_list": [
            [
                {
                    "role": "user", 
                    "content": f"{prompt}"
                }
            ],
        ],
        "max_new_tokens": max_new_tokens,
        "apply_chat_template": True,
        # "apply_chat_template": False,
    }

    results = []
    for _ in range(n_generations):
        response = requests.post(url, json=data)
        print(response.json()["response"][0])
        results.append(response.json()["response"][0])

    return results
    


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)