import argparse
from pathlib import Path
from transformers import set_seed

from Utils.model_generate_util import generate_predictions
from Utils.util import load_data_config, construct_dataset_path

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
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
    "--temperature",
    default=0.0,
    type=float,
    help="Temperature for generations. Only used when n_generations > 1.",
)
parser.add_argument(
    "--top_p",
    default=1.0,
    type=float,
    help="Top_p for generations. Only used when n_generations > 1.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Seed for random number generator.",
)

HF_MODEL_MAX_LENGTHS = {
    # New 7B
    "./LLM_Models/New_7B/Meta-Llama-3.1-8B-Instruct": 8192,
    "./LLM_Models/New_7B/Mistral-7B-Instruct-v0.3": 32000,
    "./LLM_Models/New_7B/Qwen2-7B-Instruct": 32000,
    "./LLM_Models/New_7B/Qwen2.5-7B-Instruct": 8192,
}

def main(args: argparse.Namespace):
    """
    Main Function
    """
    data_config = load_data_config(args.dataset_config)
    data_path = construct_dataset_path(data_dir=args.data_dir, test_or_train=args.test_or_train)
    output_path = Path(args.results_dir + "/" + args.data_name)
    max_length = HF_MODEL_MAX_LENGTHS[args.model]

    if args.n_generations > 1:
        set_seed(args.seed)
        assert args.temperature != 0

    generate_predictions(
        model_name=args.model,
        max_new_tokens=data_config["max_new_tokens"],
        device=args.device,
        n_generations=args.n_generations,
        max_length=max_length,
        data_path=data_path,
        output_fpath=output_path,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)