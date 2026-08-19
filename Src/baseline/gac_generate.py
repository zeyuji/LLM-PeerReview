import argparse
from pathlib import Path
import jsonlines
from tqdm.auto import tqdm
import requests

from Utils.model_generate_util import get_generation_resume_position, validate_generation_dataset_size
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
parser.add_argument(
    "--server_url",
    type=str,
    default="http://127.0.0.1:8000/api/generate/",
    help="GaC generation API endpoint.",
)
parser.add_argument(
    "--request_timeout",
    type=float,
    default=1800.0,
    help="Timeout in seconds for each GaC generation request.",
)

def main(args: argparse.Namespace):
    """
    Main Function
    """
    data_config = load_data_config(args.dataset_config)
    if data_config.get("dataset") != args.data_name:
        raise ValueError(
            f"Dataset config names {data_config.get('dataset')!r}, but --data_name is {args.data_name!r}"
        )
    if args.n_generations <= 0:
        raise ValueError(f"n_generations must be positive, got {args.n_generations}")
    if args.request_timeout <= 0:
        raise ValueError(f"request_timeout must be positive, got {args.request_timeout}")
    data_path = construct_dataset_path(data_dir=args.data_dir, test_or_train=args.test_or_train)
    output_path = Path(args.results_dir + "/" + args.data_name)

    gac_generate_predictions(
        max_new_tokens=data_config["max_new_tokens"],
        n_generations=args.n_generations,
        data_path=data_path,
        output_fpath=output_path,
        server_url=args.server_url,
        request_timeout=args.request_timeout,
        expected_n_samples=data_config.get(f"{args.test_or_train}_size"),
        split_name=args.test_or_train,
    )

def gac_generate_predictions(
    n_generations: int,
    max_new_tokens: int,
    data_path: str,
    output_fpath: Path,
    server_url: str,
    request_timeout: float,
    expected_n_samples: int,
    split_name: str,
):
    """
    Generates predictions for a dataset using a pretrained model and saves them to separate directories for each seed.

    Args:
        n_generations (int): Number of generations per sample.
        max_new_tokens (int): The maximum number of new tokens to generate.
        data_path (str): Path to the dataset.
        output_fpath (Path): Directory to save the generated outputs.
    """

    # Create subdirectories for each seed/generation
    output_fpath.mkdir(parents=True, exist_ok=True)
    seed_dirs = [output_fpath / f"Seed-{i}" for i in range(1, n_generations + 1)]
    for seed_dir in seed_dirs:
        seed_dir.mkdir(exist_ok=True)

    # Load the dataset
    with jsonlines.open(data_path) as file:
        dataset = list(file.iter())
    validate_generation_dataset_size(dataset, expected_n_samples, split_name, data_path)

    seed_files = [seed_dir / f"seed_{i}.jsonl" for i, seed_dir in enumerate(seed_dirs, start=1)]
    existing_lines = get_generation_resume_position(seed_files, dataset, str(output_fpath))
    if existing_lines > 0:
        print(f"Resuming GaC generation from line {existing_lines} for {output_fpath}")
    else:
        print(f"Will save results to: {output_fpath}")

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
            server_url=server_url,
            request_timeout=request_timeout,
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
    server_url: str,
    request_timeout: float,
):
    """
    Generate multiple texts for a single prompt using a pretrained model.

    Args:
        max_new_tokens (int): The maximum number of new tokens to generate.
        n_generations (int): Number of generations to produce.
        prompt (str): The input prompt for generation.
    """

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
        try:
            response = requests.post(server_url, json=data, timeout=request_timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"GaC request failed for {server_url}: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"GaC server returned invalid JSON: {response.text[:200]}") from exc
        generated = payload.get("response")
        if not isinstance(generated, list) or len(generated) != 1 or not isinstance(generated[0], str):
            raise RuntimeError(f"GaC server returned an invalid response payload: {payload}")
        print(generated[0])
        results.append(generated[0])

    return results
    


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)
