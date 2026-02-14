import argparse
from pathlib import Path

from Utils.util import load_data_config
from Utils.embedder import Embedder
from Utils.constants import (
    MODEL_GROUPS,
)
from Utils.smoothie_util import run_smoothie

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
    "--type",
    choices=["sample_dependent", "sample_independent"],
    required=True,
    help="The type of Smoothie to use. See file docstring for more information.",
)
parser.add_argument(
    "--k",
    type=int,
    help="Nearest neighborhood size. Only used if --type is set to sample_dependent.",
)
parser.add_argument(
    "--embedding_model",
    type=str,
    choices=["./LLM_Models/Sentence_Embedding_Models/all-mpnet-base-v2", "bge-small-en-v1.5"],
    help="Model to use for embedding generations.",
)
parser.add_argument(
    "--seed",
    type=int,
)

def main(args: argparse.Namespace):
    data_config = load_data_config(args.dataset_config)
    embedder = Embedder(model_name=args.embedding_model)
    model_groups = MODEL_GROUPS

    output_fpath = Path(args.results_dir) / data_config["dataset"]
    output_fpath.mkdir(parents=True, exist_ok=True)
    output_fpath = output_fpath / f"seed_{args.seed}.jsonl"
    
    original_data_path = Path("./Datasets") / data_config["dataset"] / "test.jsonl"

    run_smoothie(
        args=args,
        seed=args.seed,
        original_data_path=original_data_path,
        output_fpath=output_fpath,
        data_config=data_config,
        model_group_scale=args.model_group_scale,
        model_group=model_groups[args.model_group_scale],
        embedder=embedder,
    )


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)