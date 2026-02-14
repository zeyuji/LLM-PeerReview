import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import os

from Utils.util import (
    load_data_config,
    load_gpt_scores,
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


def evaluate(
    data_config: Dict,
    selected_models: List,
    gpt_scores: List[List[float]],
) -> float:

    metrics = data_config["metrics"]
    metrics = metrics[0]
    scores = []
    if metrics == "gpt_cmp":
        scores.extend(gpt_cmp(selected_models=selected_models, gpt_scores=gpt_scores))
    else:
        raise ValueError("Unknown metrics")

    # print("scores:", scores[:600])

    assert len(scores) == len(selected_models)
    return np.mean(scores)     

def gpt_cmp(selected_models: List[str], gpt_scores: List[List[float]]):
    ret = []
    for i in range(len(selected_models)):
        selected_model = selected_models[i] # 0 1 2 3
        ret.append(gpt_scores[selected_model][i])
    return ret

def main(args):
    data_config = load_data_config(args.dataset_config)
    selected_models = load_selected_model(args.response_dir)
    source_dir = Path(args.data_dir).parent
    score_files = [
        source_dir / "llama.jsonl",
        source_dir / "mistral.jsonl",
        source_dir / "qwen2.jsonl",
        source_dir / "qwen2.5.jsonl",
    ]
    
    gpt_scores = load_gpt_scores(score_files)
    # [n_model, n_sample]
    scores = []

    scores.append(evaluate(data_config, selected_models, gpt_scores))

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