import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import jsonlines
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from Utils.constants import MODEL_NAME_MAPS
from Utils.core_decode import CoreModelBundle, DEFAULT_MAX_PROMPT_LENGTH, run_core_decode
from Utils.core_token_map import DEFAULT_TOKEN_MAP_CACHE_ROOT, get_cached_token_map
from Utils.util import construct_dataset_path, load_data_config


DEFAULT_MAIN_MODEL = "Qwen2.5-7B-Instruct"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config yaml.")
parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory with test.jsonl/train.jsonl.")
parser.add_argument("--data_name", type=str, required=True, help="Dataset name used in output paths.")
parser.add_argument("--results_dir", type=str, required=True, help="Root directory for CoRE outputs.")
parser.add_argument(
    "--models",
    type=str,
    required=True,
    help="Comma-separated ordered model list. The first model is used as the main model.",
)
parser.add_argument(
    "--devices",
    type=str,
    required=True,
    help="Comma-separated device list aligned with --models. Example: 0,1,2,3",
)
parser.add_argument(
    "--align_method",
    type=str,
    required=True,
    choices=["unite", "gac"],
    help="Token alignment method.",
)
parser.add_argument(
    "--variant",
    type=str,
    required=True,
    choices=["vanilla", "consist-rbf"],
    help="CoRE variant to run.",
)
parser.add_argument("--seed", type=int, required=True, help="Output seed index.")
parser.add_argument("--top_k", type=int, default=10, help="Top-k logits retained per model step.")
parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Optional cap on the number of dataset samples for smoke tests.",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=None,
    help="Optional override of dataset_config max_new_tokens.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Write detailed per-sample debug traces in addition to the main jsonl output.",
)
parser.add_argument(
    "--test_or_train",
    type=str,
    default="test",
    help="Which dataset split file to read. Default is test.",
)


def _split_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_prompt(multi_model_prompt) -> str:
    if isinstance(multi_model_prompt, list):
        if len(multi_model_prompt) == 0:
            return ""
        return str(multi_model_prompt[0])
    return str(multi_model_prompt)


def _load_dataset_records(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    with jsonlines.open(data_path) as reader:
        records = list(reader.iter())
    if max_samples is not None:
        return records[:max_samples]
    return records


def _validate_dataset_records(records: Sequence[Dict], data_path: str) -> None:
    indices = []
    for position, record in enumerate(records):
        missing = {"idx", "task_name", "multi_model_prompt"} - set(record)
        if missing:
            raise ValueError(f"Dataset record {position} in {data_path} is missing fields: {sorted(missing)}")
        indices.append(record["idx"])
    if len(indices) != len(set(indices)):
        raise ValueError(f"Dataset contains duplicate idx values: {data_path}")


def _validate_existing_output(
    file_path: Path,
    dataset_records: Sequence[Dict],
    models: Sequence[str],
    align_method: str,
    variant: str,
) -> int:
    if not file_path.exists():
        return 0
    with jsonlines.open(file_path) as reader:
        records = list(reader.iter())
    if len(records) > len(dataset_records):
        raise ValueError(
            f"Existing CoRE output has {len(records)} lines, but dataset slice only has "
            f"{len(dataset_records)} samples: {file_path}"
        )
    for position, record in enumerate(records):
        expected_idx = dataset_records[position].get("idx")
        if record.get("idx") != expected_idx:
            raise ValueError(
                f"Existing CoRE output does not match dataset idx at position {position}: {file_path}"
            )
        if not isinstance(record.get("generation"), str):
            raise ValueError(f"Existing CoRE output is missing generation at position {position}: {file_path}")
        if record.get("task_name") != dataset_records[position]["task_name"]:
            raise ValueError(f"Existing CoRE output task_name mismatch at position {position}: {file_path}")
        expected_fields = {
            "ensemble_method": "CoRE",
            "main_model": models[0],
            "assist_models": list(models[1:]),
            "model_list": list(models),
            "align_method": align_method,
            "variant": variant,
        }
        for field, expected_value in expected_fields.items():
            if record.get(field) != expected_value:
                raise ValueError(
                    f"Existing CoRE output metadata mismatch at position {position}: "
                    f"{field}={record.get(field)!r}, expected {expected_value!r}"
                )
        num_generated_tokens = record.get("num_generated_tokens")
        if (
            isinstance(num_generated_tokens, bool)
            or not isinstance(num_generated_tokens, int)
            or num_generated_tokens < 0
        ):
            raise ValueError(
                f"Invalid CoRE num_generated_tokens at position {position}: {file_path}"
            )
    return len(records)


def _ordered_model_tag(models: Sequence[str]) -> str:
    return "__".join(model.replace("/", "__") for model in models)


def _parse_device(value: str) -> str:
    if value.isdigit():
        return f"cuda:{value}"
    return value


def _resolve_devices(devices_arg: str, num_models: int) -> List[str]:
    devices = [_parse_device(item) for item in _split_csv_arg(devices_arg)]
    if len(devices) != num_models:
        raise ValueError(
            f"CoRE expects one device per model. Received {devices} for {num_models} models."
        )
    return devices


def _load_model_bundle(model_name: str, device_str: str) -> CoreModelBundle:
    if model_name not in MODEL_NAME_MAPS:
        raise KeyError(f"Model path is not registered in MODEL_NAME_MAPS: {model_name}")

    model_path = MODEL_NAME_MAPS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        truncation_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device_str.startswith("cuda:"):
        device_index = int(device_str.split(":")[1])
        model_kwargs = {
            "device_map": {"": device_index},
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            ).eval()
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device_str).eval()

    return CoreModelBundle(
        name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device_str),
    )


def _cleanup_model_bundles(model_bundles: Sequence[CoreModelBundle]) -> None:
    for bundle in model_bundles:
        del bundle.model
        del bundle.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _build_output_path(
    results_dir: str,
    align_method: str,
    variant: str,
    models: Sequence[str],
    data_name: str,
    seed: int,
) -> Path:
    main_model = models[0]
    return (
        Path(results_dir)
        / align_method
        / variant
        / f"main_{main_model}"
        / f"models_{_ordered_model_tag(models)}"
        / data_name
        / f"seed_{seed}.jsonl"
    )


def _build_output_record(
    sample: Dict,
    generation: str,
    models: Sequence[str],
    align_method: str,
    variant: str,
    num_generated_tokens: int,
) -> Dict:
    return {
        "task_name": sample.get("task_name"),
        "generation": generation,
        "idx": sample.get("idx"),
        "ensemble_method": "CoRE",
        "main_model": models[0],
        "assist_models": list(models[1:]),
        "model_list": list(models),
        "align_method": align_method,
        "variant": variant,
        "num_generated_tokens": int(num_generated_tokens),
    }


def _build_run_config(
    dataset_name: str,
    models: Sequence[str],
    devices: Sequence[str],
    align_method: str,
    variant: str,
    top_k: int,
    max_new_tokens: int,
    max_samples: Optional[int],
    debug: bool,
    seed: int,
    data_path: str,
    output_path: Path,
    split_name: str,
) -> Dict:
    return {
        "dataset": dataset_name,
        "split": split_name,
        "seed": int(seed),
        "data_path": data_path,
        "output_path": str(output_path),
        "main_model": models[0],
        "assist_models": list(models[1:]),
        "model_list": list(models),
        "devices": list(devices),
        "align_method": align_method,
        "variant": variant,
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
        "max_samples": max_samples,
        "debug": bool(debug),
        "token_map_cache_root": str(DEFAULT_TOKEN_MAP_CACHE_ROOT),
    }


def _validate_run_config(run_config_path: Path, expected: Dict, has_existing_output: bool) -> None:
    if not run_config_path.exists():
        if has_existing_output:
            raise ValueError(
                f"Existing CoRE output has no run_config.json, so a safe resume is not possible: "
                f"{run_config_path.parent}"
            )
        return

    with run_config_path.open("r", encoding="utf-8") as f:
        existing = json.load(f)
    stable_fields = [
        "dataset",
        "split",
        "seed",
        "data_path",
        "output_path",
        "main_model",
        "assist_models",
        "model_list",
        "align_method",
        "variant",
        "top_k",
        "max_new_tokens",
        "max_samples",
        "debug",
        "token_map_cache_root",
    ]
    mismatches = {
        field: (existing.get(field), expected.get(field))
        for field in stable_fields
        if existing.get(field) != expected.get(field)
    }
    if mismatches:
        raise ValueError(
            f"CoRE run configuration does not match the existing output in {run_config_path}: {mismatches}"
        )


def _write_run_config(run_config_path: Path, payload: Dict) -> None:
    with run_config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data_config = load_data_config(args.dataset_config)
    data_path = construct_dataset_path(data_dir=args.data_dir, test_or_train=args.test_or_train)
    all_dataset_records = _load_dataset_records(data_path=data_path)
    _validate_dataset_records(all_dataset_records, data_path)
    if data_config.get("dataset") != args.data_name:
        raise ValueError(
            f"Dataset config names {data_config.get('dataset')!r}, but --data_name is {args.data_name!r}"
        )
    size_key = f"{args.test_or_train}_size"
    if size_key in data_config and int(data_config[size_key]) != len(all_dataset_records):
        raise ValueError(
            f"Dataset config {size_key}={data_config[size_key]} does not match "
            f"{len(all_dataset_records)} records in {data_path}"
        )
    dataset_records = (
        all_dataset_records[:args.max_samples]
        if args.max_samples is not None
        else all_dataset_records
    )

    models = _split_csv_arg(args.models)
    if len(models) < 2:
        raise ValueError("CoRE baseline requires at least 2 models: 1 main + 1 assist.")
    if len(set(models)) != len(models):
        raise ValueError(f"CoRE model list contains duplicates: {models}")
    for model_name in models:
        if model_name not in MODEL_NAME_MAPS:
            raise KeyError(f"Unknown model in --models: {model_name}")
    if args.top_k <= 0:
        raise ValueError(f"--top_k must be positive, but got {args.top_k}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"--max_samples must be positive when provided, but got {args.max_samples}")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError(f"--max_new_tokens must be positive when provided, but got {args.max_new_tokens}")

    devices = _resolve_devices(devices_arg=args.devices, num_models=len(models))
    output_path = _build_output_path(
        results_dir=args.results_dir,
        align_method=args.align_method,
        variant=args.variant,
        models=models,
        data_name=args.data_name,
        seed=args.seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_max_new_tokens = (
        min(int(data_config["max_new_tokens"]), int(args.max_new_tokens))
        if args.max_new_tokens is not None
        else int(data_config["max_new_tokens"])
    )
    if resolved_max_new_tokens <= 0:
        raise ValueError(f"Resolved CoRE max_new_tokens must be positive, got {resolved_max_new_tokens}")
    run_config = _build_run_config(
        dataset_name=args.data_name,
        models=models,
        devices=devices,
        align_method=args.align_method,
        variant=args.variant,
        top_k=args.top_k,
        max_new_tokens=resolved_max_new_tokens,
        max_samples=args.max_samples,
        debug=args.debug,
        seed=args.seed,
        data_path=data_path,
        output_path=output_path,
        split_name=args.test_or_train,
    )

    existing_lines = _validate_existing_output(
        output_path,
        dataset_records,
        models,
        args.align_method,
        args.variant,
    )
    run_config_path = output_path.parent / "run_config.json"
    _validate_run_config(run_config_path, run_config, has_existing_output=existing_lines > 0)
    if not run_config_path.exists():
        _write_run_config(run_config_path, run_config)
    if existing_lines == len(dataset_records):
        print(f"CoRE results already complete. Skipping generation for {output_path}")
        return

    debug_path = output_path.parent / f"debug_seed_{args.seed}.jsonl"
    if args.debug and existing_lines == 0 and debug_path.exists():
        debug_path.unlink()
    if args.debug and existing_lines > 0:
        if not debug_path.exists():
            raise ValueError(f"Cannot resume CoRE debug output because the debug file is missing: {debug_path}")
        with jsonlines.open(debug_path) as reader:
            debug_records = list(reader.iter())
        if len(debug_records) != existing_lines:
            raise ValueError(
                f"CoRE debug output has {len(debug_records)} lines, expected {existing_lines}: {debug_path}"
            )
        for position, record in enumerate(debug_records):
            if record.get("idx") != dataset_records[position].get("idx"):
                raise ValueError(f"CoRE debug output idx mismatch at position {position}: {debug_path}")

    model_bundles: List[CoreModelBundle] = []
    try:
        for model_name, device_str in zip(models, devices):
            print(f"Loading CoRE model {model_name} on {device_str}")
            model_bundles.append(_load_model_bundle(model_name=model_name, device_str=device_str))

        main_bundle = model_bundles[0]
        assist_bundles = model_bundles[1:]
        assist_to_main_maps = [
            get_cached_token_map(
                source_name=assist_bundle.name,
                target_name=main_bundle.name,
                source_tokenizer=assist_bundle.tokenizer,
                target_tokenizer=main_bundle.tokenizer,
                method=args.align_method,
                cache_root=DEFAULT_TOKEN_MAP_CACHE_ROOT,
                source_vocab_size=int(assist_bundle.model.config.vocab_size),
                target_vocab_size=int(main_bundle.model.config.vocab_size),
            ).to(assist_bundle.device)
            for assist_bundle in assist_bundles
        ]

        remaining_records = dataset_records[existing_lines:]
        debug_writer = jsonlines.open(debug_path, mode="a", flush=True) if args.debug else None
        try:
            with jsonlines.open(output_path, mode="a", flush=True) as writer:
                for sample in tqdm(remaining_records, desc=f"CoRE {args.align_method}/{args.variant} -> {args.data_name}"):
                    prompt = _normalize_prompt(sample["multi_model_prompt"])
                    result = run_core_decode(
                        prompt=prompt,
                        main_bundle=main_bundle,
                        assist_bundles=assist_bundles,
                        assist_to_main_maps=assist_to_main_maps,
                        variant=args.variant,
                        top_k=args.top_k,
                        max_new_tokens=resolved_max_new_tokens,
                        debug=args.debug,
                        max_prompt_length=DEFAULT_MAX_PROMPT_LENGTH,
                    )
                    writer.write(
                        _build_output_record(
                            sample=sample,
                            generation=result.generation,
                            models=models,
                            align_method=args.align_method,
                            variant=args.variant,
                            num_generated_tokens=result.num_generated_tokens,
                        )
                    )
                    if debug_writer is not None:
                        debug_writer.write(
                            {
                                "idx": sample.get("idx"),
                                "task_name": sample.get("task_name"),
                                "generation": result.generation,
                                "steps": result.debug_steps,
                            }
                        )
        finally:
            if debug_writer is not None:
                debug_writer.close()
    finally:
        if model_bundles:
            _cleanup_model_bundles(model_bundles)


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print(f"You are using args:\n{args}")
    main(args)
