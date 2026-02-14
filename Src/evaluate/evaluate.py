import argparse
from typing import Dict, List, Union
import numpy as np
import os

from Utils.util import (
    load_data_config,
    load_response,
    load_reference,
    clean_generations,
)
from Utils.metrics import (
    gsm8k_acc,
    trivia_qa_acc,
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

def evaluate(
    data_config: Dict,
    responses: List,
    references: Union[List[str], List[List[str]]],
) -> float:
    task_generations = clean_generations(responses)

    metrics = data_config["metrics"]
    metrics = metrics[0]
    scores = []
    if metrics == "trivia_qa_acc":
        scores.extend(trivia_qa_acc(generations=task_generations, references=references))
    elif metrics == "gsm8k_acc":
        scores.extend(gsm8k_acc(generations=task_generations, references=references))
    else:
        raise ValueError("Unknown metrics")

    print("scores:", scores[71])

    assert len(scores) == len(responses)
    return np.mean(scores)     

def main(args):
    data_config = load_data_config(args.dataset_config)
    references = load_reference(args.data_dir)
    scores = []

    seed_n = 1
    responses = load_response(args.response_dir, seed_n)
    scores.append(evaluate(data_config, responses, references))

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