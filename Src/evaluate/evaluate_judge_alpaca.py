import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import os

from Utils.util import (
    load_data_config,
    load_embedding_input,
    load_gpt_scores,
    load_indices,
    load_reference,
    load_selected_model,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config",
    type=str,
    help="Path to config file. This should be a yaml file.",
)
parser.add_argument(
    "--data_name",
    type=str,
    help="Name of the dataset.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    help="Directory with data files.",
)
parser.add_argument(
    "--response_dir",
    type=str,
    help="Directory with response files.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="Directory to save results to.",
)
parser.add_argument(
    "--judge_model_name",
    type=str,
    help="Name of the judge model.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Judge result seed suffix to evaluate. Default is 1.",
)

def evaluate(
    data_config: Dict,
    selected_models: List,
    gpt_scores: List[List[float]],
    args: argparse.Namespace,
) -> float:

    metrics = data_config["metrics"]
    metrics = metrics[0]
    scores = []
    if metrics == "gpt_cmp":
        scores.extend(gpt_cmp(selected_models=selected_models, gpt_scores=gpt_scores))
    else:
        raise ValueError("Unknown metrics")

    if len(scores) != len(selected_models):
        raise ValueError(f"Metric returned {len(scores)} scores for {len(selected_models)} selections")
    return np.mean(scores)     

def gpt_cmp(selected_models: List[str], gpt_scores: List[List[float]]):
    if not gpt_scores or any(len(model_scores) != len(selected_models) for model_scores in gpt_scores):
        raise ValueError("GPT score files must all match the number of selected models")
    ret = []
    for i in range(len(selected_models)):
        selected_model = int(selected_models[i]) # 0 1 2 3
        if selected_model < 0 or selected_model >= len(gpt_scores):
            raise ValueError(f"Invalid selected_model={selected_model} at sample {i}")
        ret.append(gpt_scores[selected_model][i])
    # max_count = sum(any(gpt_scores[j][i] == 1.0 for j in range(4)) for i in range(len(selected_models)))
    # print(f"max_count = {max_count}")
    return ret

def main(args):
    data_config = load_data_config(args.dataset_config)
    if data_config.get("dataset") != args.data_name:
        raise ValueError(
            f"Dataset config names {data_config.get('dataset')!r}, but --data_name is {args.data_name!r}"
        )

    sample_indices = load_indices(args.data_dir)
    instructions = load_embedding_input(args.data_dir)
    references = load_reference(args.data_dir)
    if len(sample_indices) != len(instructions) or len(sample_indices) != len(references):
        raise ValueError("Dataset idx, instruction, and reference counts do not match")
    if int(data_config["test_size"]) != len(sample_indices):
        raise ValueError(
            f"Dataset config test_size={data_config['test_size']} does not match "
            f"{len(sample_indices)} records"
        )
    selected_models = load_selected_model(
        args.response_dir + f"/judge_{args.judge_model_name}_seed_{args.seed}.jsonl",
        expected_indices=sample_indices,
    )
    source_dir = Path(args.data_dir).parent

    score_files = [
        source_dir / "llama.jsonl",
        source_dir / "mistral.jsonl",
        source_dir / "qwen2.jsonl",
        source_dir / "qwen2.5.jsonl",
    ]

    gpt_scores = load_gpt_scores(
        score_files,
        expected_instructions=instructions,
        expected_references=references,
    )
    # [n_model, n_sample]
    scores = []
    scores.append(evaluate(data_config, selected_models, gpt_scores, args))

    print("Scores: {}".format(scores))
    print("Mean score: {}".format(np.mean(scores)))
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    with open(args.results_dir + "/" + args.data_name + ".txt", "w") as f:
        f.write("Scores: {}\n".format(scores))
        f.write("Mean score: {}\n".format(np.mean(scores)))
    
if __name__ == "__main__":
    print("Src.evaluate...")
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)
    print("#" * 100)
