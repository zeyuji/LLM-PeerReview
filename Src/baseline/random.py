import argparse
from pathlib import Path
import numpy as np
import json
from Utils.util import load_model_group_response

from Utils.util import load_data_config
from Utils.constants import (
    MODEL_GROUPS
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_config", type=str, help="Path to the data yaml config.")
parser.add_argument(
    "--results_dir",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--model_group_scale",
    type=str,
    choices=["New_7B"],
    help="Scale of the model group.",
)
parser.add_argument(
    "--seed",
    type=int,
)

def run_random(
    seed: int,
    output_fpath: Path,
    data_config: dict,
    model_group_scale: str,
    model_group: list,
):
    test_generations_for_random = load_model_group_response(
        response_path="./LLM_Response/Test/"+model_group_scale,
        model_group=model_group,
        data_name=data_config["dataset"],
        seed=seed
    )
    test_generations_for_random = np.array(test_generations_for_random)
    # [n_models, n_samples]
    test_generations_for_random = test_generations_for_random.transpose()
    # [n_samples, n_models]

    dataset_texts = []
    random_logs = []
    n_samples = len(test_generations_for_random)
    n_models = len(test_generations_for_random[0])
    for i in range(n_samples):
        index = np.random.randint(0, n_models)
        random_logs.append(index)
        dataset_texts.append(test_generations_for_random[i][index])
    results_lines = [
        {
            "task_name": data_config["dataset"],
            "generation": text,
            "idx": idx,
            "selected_model": random_logs[idx],
        }
        for idx, text in enumerate(dataset_texts)
    ]
    print(f"Saving results to {output_fpath}")
    with open(output_fpath, 'w', encoding='utf-8') as f:
        for line in results_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def main(args: argparse.Namespace):
    data_config = load_data_config(args.dataset_config)
    model_groups = MODEL_GROUPS

    output_fpath = Path(args.results_dir) / data_config["dataset"]
    output_fpath.mkdir(parents=True, exist_ok=True)
    output_fpath = output_fpath / f"seed_{args.seed}.jsonl"

    run_random(
        seed=args.seed,
        output_fpath=output_fpath,
        data_config=data_config,
        model_group_scale=args.model_group_scale,
        model_group=model_groups[args.model_group_scale],
    )


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)