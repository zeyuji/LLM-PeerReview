import argparse

from Src.baseline.safe_adapter import run_safe_generation


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config",
    type=str,
    required=True,
    help="Path to dataset config yaml.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    required=True,
    help="Root directory for SAFE outputs.",
)
parser.add_argument(
    "--profile",
    type=str,
    required=True,
    help="SAFE profile name under Configs/safe/ without the .yaml suffix.",
)
parser.add_argument(
    "--profiles_dir",
    type=str,
    default="./Configs/safe",
    help="Directory that stores SAFE profile yaml files.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=None,
    help="Dataset directory that contains test.jsonl or train.jsonl.",
)
parser.add_argument(
    "--data_name",
    type=str,
    default=None,
    help="Optional dataset name; it must match dataset_config.dataset.",
)
parser.add_argument(
    "--test_or_train",
    type=str,
    default="test",
    help="Which split file to read. Default is test.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Seed index used in the output filename.",
)
parser.add_argument(
    "--device_ids",
    type=int,
    nargs="+",
    default=None,
    help="CUDA device ids for the loaded SAFE models, in profile.models order.",
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Optional cap for smoke testing on the first N samples.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete any existing output file before generation.",
)
parser.add_argument(
    "--validate_only",
    action="store_true",
    help="Validate profile and dataset wiring without loading models.",
)
parser.add_argument(
    "--attn_implementation",
    type=str,
    default="flash_attention_2",
    help="Attention backend passed to from_pretrained when available.",
)
parser.add_argument(
    "--max_prompt_length",
    type=int,
    default=4096,
    help="Tokenizer truncation length for multi_model_prompt.",
)


def main(args: argparse.Namespace) -> None:
    output_path = run_safe_generation(
        dataset_config=args.dataset_config,
        results_dir=args.results_dir,
        profile_name=args.profile,
        profiles_dir=args.profiles_dir,
        data_dir=args.data_dir,
        data_name=args.data_name,
        test_or_train=args.test_or_train,
        seed=args.seed,
        device_ids=args.device_ids,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
        attn_implementation=args.attn_implementation,
        max_prompt_length=args.max_prompt_length,
    )
    print(f"SAFE output path: {output_path}")


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print(f"You are using args:\n{args}")
    main(args)
