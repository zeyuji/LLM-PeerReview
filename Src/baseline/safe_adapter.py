import gc
import json
import math
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import jsonlines
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from Utils.constants import MODEL_GROUPS, MODEL_NAME_MAPS
from Utils.util import clean_generation, construct_dataset_path, load_data_config
from Src.baseline.safe_core import safe_generate_gac_multi, safe_generate_unite_multi


SAFE_DIR = Path(__file__).resolve().parents[2] / "SAFE"
if str(SAFE_DIR) not in sys.path:
    sys.path.insert(0, str(SAFE_DIR))

from safe_generate_gac import safe_generate_gac, setup_union_vocab
from safe_generate_unite import safe_generate_unite


FIXED_DRAFTER = "Qwen2.5-7B-Instruct"
SAFE_MODEL_ALIASES = {
    "Meta-Llama-3.1-8B-Instruct": "llama3",
    "Mistral-7B-Instruct-v0.3": "mistral",
    "Qwen2-7B-Instruct": "qwen2",
    "Qwen2.5-7B-Instruct": "qwen2.5",
}
DEFAULT_PROMPT_MAX_LENGTH = 4096


@dataclass(frozen=True)
class SafeProfile:
    name: str
    scale: str
    mode: str
    models: List[str]
    drafter: str
    verifiers: List[str]
    ensemble_type: str
    sharpen_type: str
    gamma: int
    max_new_tokens: int


@dataclass
class LoadedModelBundle:
    name: str
    alias: Optional[str]
    model_path: str
    device_id: int
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def load_safe_profile(profile_name: str, profiles_dir: str) -> SafeProfile:
    profile_path = Path(profiles_dir) / f"{profile_name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"SAFE profile not found: {profile_path}")

    with profile_path.open("r", encoding="utf-8") as f:
        raw_profile = yaml.safe_load(f) or {}

    required_fields = {
        "scale",
        "mode",
        "models",
        "drafter",
        "verifiers",
        "ensemble_type",
        "sharpen_type",
        "gamma",
        "max_new_tokens",
    }
    missing_fields = sorted(required_fields - set(raw_profile.keys()))
    if missing_fields:
        raise ValueError(f"SAFE profile {profile_name} is missing fields: {missing_fields}")

    profile = SafeProfile(
        name=profile_name,
        scale=raw_profile["scale"],
        mode=raw_profile["mode"],
        models=list(raw_profile["models"]),
        drafter=raw_profile["drafter"],
        verifiers=list(raw_profile["verifiers"]),
        ensemble_type=raw_profile["ensemble_type"],
        sharpen_type=raw_profile["sharpen_type"],
        gamma=int(raw_profile["gamma"]),
        max_new_tokens=int(raw_profile["max_new_tokens"]),
    )
    validate_safe_profile(profile)
    return profile


def validate_safe_profile(profile: SafeProfile) -> None:
    if profile.scale != "New_7B":
        raise ValueError(f"SAFE v1 only supports scale=New_7B, but got {profile.scale}.")

    if profile.scale not in MODEL_GROUPS:
        raise ValueError(f"Unknown model scale in SAFE profile: {profile.scale}")

    if profile.drafter != FIXED_DRAFTER:
        raise ValueError(
            f"SAFE v1 requires drafter={FIXED_DRAFTER}, but got {profile.drafter}."
        )

    if profile.drafter not in profile.models:
        raise ValueError("SAFE profile is invalid: drafter must appear in models.")

    if len(profile.models) != len(set(profile.models)):
        raise ValueError("SAFE profile is invalid: models contains duplicates.")

    if len(profile.verifiers) != len(set(profile.verifiers)):
        raise ValueError("SAFE profile is invalid: verifiers contains duplicates.")

    if profile.drafter in profile.verifiers:
        raise ValueError("SAFE profile is invalid: drafter must not appear in verifiers.")

    if set(profile.models) != {profile.drafter, *profile.verifiers}:
        raise ValueError("SAFE profile is invalid: models must equal drafter plus verifiers.")

    supported_models = set(MODEL_GROUPS[profile.scale])
    unknown_models = sorted(set(profile.models) - supported_models)
    if unknown_models:
        raise ValueError(f"SAFE profile uses unsupported models: {unknown_models}")

    if profile.gamma <= 0:
        raise ValueError("SAFE profile is invalid: gamma must be positive.")

    if profile.max_new_tokens <= 0:
        raise ValueError("SAFE profile is invalid: max_new_tokens must be positive.")

    if profile.sharpen_type not in {"geom", "heur"}:
        raise ValueError(
            f"SAFE profile is invalid: unsupported sharpen_type={profile.sharpen_type}"
        )

    if profile.mode == "official2":
        if len(profile.models) != 2 or len(profile.verifiers) != 1:
            raise ValueError("official2 requires exactly 2 models: 1 drafter + 1 verifier.")
        if profile.ensemble_type not in {"unite", "gac"}:
            raise ValueError("official2 only supports ensemble_type in {unite, gac}.")
    elif profile.mode == "official3":
        if len(profile.models) != 3 or len(profile.verifiers) != 2:
            raise ValueError("official3 requires exactly 3 models: 1 drafter + 2 verifiers.")
        if profile.ensemble_type != "unite":
            raise ValueError("official3 only supports ensemble_type=unite.")
    elif profile.mode == "safe4":
        if len(profile.models) != 4 or len(profile.verifiers) != 3:
            raise ValueError("safe4 requires exactly 4 models: 1 drafter + 3 verifiers.")
        if profile.ensemble_type not in {"unite", "gac"}:
            raise ValueError("safe4 only supports ensemble_type in {unite, gac}.")
        expected_verifiers = set(MODEL_GROUPS["New_7B"]) - {FIXED_DRAFTER}
        if set(profile.verifiers) != expected_verifiers:
            raise ValueError(
                "safe4 requires the three non-drafter New_7B models as verifiers."
            )
    else:
        raise ValueError(f"Unsupported SAFE mode: {profile.mode}")


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
    profile: SafeProfile,
) -> int:
    if not file_path.exists():
        return 0
    with jsonlines.open(file_path) as reader:
        records = list(reader.iter())
    if len(records) > len(dataset_records):
        raise ValueError(
            f"Existing SAFE output has {len(records)} lines, but dataset slice only has "
            f"{len(dataset_records)} samples: {file_path}"
        )

    expected_metadata = {
        "ensemble_method": "SAFE",
        "safe_profile": profile.name,
        "mode": profile.mode,
        "drafter_model": profile.drafter,
        "verifier_models": profile.verifiers,
        "ensemble_type": profile.ensemble_type,
        "gamma": profile.gamma,
    }
    seen_indices = set()
    for position, record in enumerate(records):
        expected_idx = dataset_records[position]["idx"]
        if record.get("idx") != expected_idx:
            raise ValueError(f"Existing SAFE output idx mismatch at position {position}: {file_path}")
        if expected_idx in seen_indices:
            raise ValueError(f"Duplicate idx {expected_idx} in existing SAFE output: {file_path}")
        seen_indices.add(expected_idx)
        if not isinstance(record.get("generation"), str):
            raise ValueError(f"Missing generation at position {position} in {file_path}")
        if record.get("task_name") != dataset_records[position]["task_name"]:
            raise ValueError(f"Existing SAFE output task_name mismatch at position {position}: {file_path}")
        for field, expected_value in expected_metadata.items():
            if record.get(field) != expected_value:
                raise ValueError(
                    f"Existing SAFE output metadata mismatch at position {position}: "
                    f"{field}={record.get(field)!r}, expected {expected_value!r}"
                )
        if "sharpen_type" in record and record["sharpen_type"] != profile.sharpen_type:
            raise ValueError(
                f"Existing SAFE output sharpen_type={record['sharpen_type']!r}, "
                f"expected {profile.sharpen_type!r}: {file_path}"
            )
        accept_rate = record.get("accept_rate")
        if (
            isinstance(accept_rate, bool)
            or not isinstance(accept_rate, Real)
            or not math.isfinite(float(accept_rate))
            or not 0.0 <= float(accept_rate) <= 1.0
        ):
            raise ValueError(f"Invalid SAFE accept_rate at position {position}: {file_path}")
        for field in ("num_ensemble", "num_generated_tokens"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid SAFE {field} at position {position}: {file_path}")
    return len(records)


def _load_dataset_records(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    with jsonlines.open(data_path) as reader:
        records = list(reader.iter())
    if max_samples is not None:
        return records[:max_samples]
    return records


def _normalize_prompt(multi_model_prompt) -> str:
    if isinstance(multi_model_prompt, list):
        if len(multi_model_prompt) == 0:
            return ""
        return str(multi_model_prompt[0])
    return str(multi_model_prompt)


def _resolve_device_ids(
    requested_device_ids: Optional[Sequence[int]],
    num_models: int,
) -> List[int]:
    if requested_device_ids is not None and len(requested_device_ids) > 0:
        if len(requested_device_ids) < num_models:
            raise ValueError(
                f"SAFE needs {num_models} device ids, but only received {requested_device_ids}."
            )
        return [int(device_id) for device_id in requested_device_ids[:num_models]]

    if not torch.cuda.is_available():
        raise ValueError("SAFE baseline requires CUDA devices for the configured New_7B models.")

    if torch.cuda.device_count() < num_models:
        raise ValueError(
            f"SAFE needs {num_models} CUDA devices by default, but only found "
            f"{torch.cuda.device_count()}. Pass --device_ids explicitly if you want to reuse devices."
        )

    return list(range(num_models))


def _load_model_bundle(
    model_name: str,
    device_id: int,
    attn_implementation: str,
    torch_dtype: torch.dtype,
) -> LoadedModelBundle:
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

    model_kwargs = {
        "device_map": {"": device_id},
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    except Exception:
        if "attn_implementation" not in model_kwargs:
            raise
        model_kwargs.pop("attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()

    try:
        model.generation_config = GenerationConfig.from_pretrained(
            model_path,
            pad_token_id=tokenizer.pad_token_id,
            trust_remote_code=True,
        )
    except Exception:
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    return LoadedModelBundle(
        name=model_name,
        alias=SAFE_MODEL_ALIASES.get(model_name),
        model_path=model_path,
        device_id=device_id,
        model=model,
        tokenizer=tokenizer,
    )


def _load_models(
    profile: SafeProfile,
    device_ids: Sequence[int],
    attn_implementation: str,
    torch_dtype: torch.dtype,
) -> Dict[str, LoadedModelBundle]:
    bundles = {}
    for model_name, device_id in zip(profile.models, device_ids):
        bundles[model_name] = _load_model_bundle(
            model_name=model_name,
            device_id=device_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
    return bundles


def _prepare_inputs(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_prompt_length: int,
):
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    ).to(model.device)


def _decode_generation(tokenizer: AutoTokenizer, generated_ids) -> str:
    if isinstance(generated_ids, list):
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        output = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)[0]
    return clean_generation(output)


def _num_generated_tokens(generated_ids) -> int:
    if isinstance(generated_ids, list):
        return len(generated_ids[0])
    return int(generated_ids.shape[-1])


def _cleanup_models(model_bundles: Dict[str, LoadedModelBundle]) -> None:
    for bundle in model_bundles.values():
        del bundle.model
        del bundle.tokenizer
    model_bundles.clear()
    torch.cuda.empty_cache()
    gc.collect()


def _build_output_record(
    sample: Dict,
    generation: str,
    profile: SafeProfile,
    accept_rate: float,
    num_ensemble: int,
    num_generated_tokens: int,
) -> Dict:
    return {
        "task_name": sample.get("task_name", profile.name),
        "generation": generation,
        "idx": sample.get("idx"),
        "ensemble_method": "SAFE",
        "safe_profile": profile.name,
        "mode": profile.mode,
        "drafter_model": profile.drafter,
        "verifier_models": profile.verifiers,
        "ensemble_type": profile.ensemble_type,
        "sharpen_type": profile.sharpen_type,
        "gamma": profile.gamma,
        "accept_rate": accept_rate,
        "num_ensemble": num_ensemble,
        "num_generated_tokens": num_generated_tokens,
    }


def run_safe_generation(
    dataset_config: str,
    results_dir: str,
    profile_name: str,
    profiles_dir: str = "./Configs/safe",
    data_dir: Optional[str] = None,
    data_name: Optional[str] = None,
    test_or_train: str = "test",
    seed: int = 1,
    device_ids: Optional[Sequence[int]] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    validate_only: bool = False,
    attn_implementation: str = "flash_attention_2",
    max_prompt_length: int = DEFAULT_PROMPT_MAX_LENGTH,
) -> Path:
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be positive when provided, got {max_samples}")
    if max_prompt_length <= 0:
        raise ValueError(f"max_prompt_length must be positive, got {max_prompt_length}")

    data_config = load_data_config(dataset_config)
    configured_data_name = data_config["dataset"]
    if data_name is not None and data_name != configured_data_name:
        raise ValueError(
            f"Dataset config names {configured_data_name!r}, but --data_name is {data_name!r}"
        )
    data_name = configured_data_name
    data_dir = data_dir or f"./Datasets/{data_name}"
    data_path = construct_dataset_path(data_dir=data_dir, test_or_train=test_or_train)
    profile = load_safe_profile(profile_name=profile_name, profiles_dir=profiles_dir)

    output_path = Path(results_dir) / profile.name / data_name / f"seed_{seed}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_dataset_records = _load_dataset_records(data_path=data_path)
    _validate_dataset_records(all_dataset_records, data_path)
    size_key = f"{test_or_train}_size"
    if size_key in data_config and int(data_config[size_key]) != len(all_dataset_records):
        raise ValueError(
            f"Dataset config {size_key}={data_config[size_key]} does not match "
            f"{len(all_dataset_records)} records in {data_path}"
        )
    dataset_records = all_dataset_records[:max_samples] if max_samples is not None else all_dataset_records
    resolved_max_new_tokens = min(profile.max_new_tokens, int(data_config["max_new_tokens"]))
    if resolved_max_new_tokens <= 0:
        raise ValueError(f"Resolved SAFE max_new_tokens must be positive, got {resolved_max_new_tokens}")

    if validate_only:
        print(
            json.dumps(
                {
                    "profile": profile.name,
                    "dataset": data_name,
                    "n_samples": len(dataset_records),
                    "output_path": str(output_path),
                    "resolved_max_new_tokens": resolved_max_new_tokens,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return output_path

    if overwrite and output_path.exists():
        output_path.unlink()

    existing_lines = _validate_existing_output(output_path, dataset_records, profile)
    if existing_lines == len(dataset_records):
        print(f"SAFE results already complete. Skipping generation for {output_path}")
        return output_path

    resolved_device_ids = _resolve_device_ids(device_ids, len(profile.models))
    dtype = torch.float16
    model_bundles = {}
    try:
        model_bundles = _load_models(
            profile=profile,
            device_ids=resolved_device_ids,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        )

        drafter_bundle = model_bundles[profile.drafter]
        verifier_bundles = [model_bundles[model_name] for model_name in profile.verifiers]

        gac_kwargs = {}
        if profile.ensemble_type == "gac":
            gac_models = [drafter_bundle.model, *[bundle.model for bundle in verifier_bundles]]
            gac_tokenizers = [drafter_bundle.tokenizer, *[bundle.tokenizer for bundle in verifier_bundles]]
            vocab_union, mapping_matrices, index_to_vocab, byte_mappings_list, special_prefix_tokens_dict = setup_union_vocab(
                gac_models,
                gac_tokenizers,
            )
            gac_kwargs = {
                "vocab_union": vocab_union,
                "mapping_matrices": mapping_matrices,
                "index_to_vocab": index_to_vocab,
                "byte_mappings_list": byte_mappings_list,
                "special_prefix_tokens_dict": special_prefix_tokens_dict,
            }

        remaining_records = dataset_records[existing_lines:]
        with jsonlines.open(output_path, mode="a", flush=True) as writer:
            for sample in tqdm(remaining_records, desc=f"SAFE {profile.name} -> {data_name}"):
                prompt = _normalize_prompt(sample["multi_model_prompt"])
                drafter_inputs = _prepare_inputs(
                    prompt=prompt,
                    tokenizer=drafter_bundle.tokenizer,
                    model=drafter_bundle.model,
                    max_prompt_length=max_prompt_length,
                )
                verifier_inputs = [
                    _prepare_inputs(
                        prompt=prompt,
                        tokenizer=bundle.tokenizer,
                        model=bundle.model,
                        max_prompt_length=max_prompt_length,
                    )
                    for bundle in verifier_bundles
                ]

                if profile.mode == "official2":
                    verifier_bundle = verifier_bundles[0]
                    if profile.ensemble_type == "unite":
                        generated_ids, accept_rate, num_ensemble = safe_generate_unite(
                            inputs=drafter_inputs,
                            ver_inputs=verifier_inputs[0],
                            draft_model=drafter_bundle.model,
                            ver_model=verifier_bundle.model,
                            max_length=resolved_max_new_tokens,
                            draft_tokenizer=drafter_bundle.tokenizer,
                            ver_tokenizer=verifier_bundle.tokenizer,
                            gamma=profile.gamma,
                            draft=drafter_bundle.alias,
                            verifier=verifier_bundle.alias,
                            use_cache=True,
                            sharpen_type=profile.sharpen_type,
                        )
                    else:
                        generated_ids, accept_rate, num_ensemble = safe_generate_gac(
                            inputs=drafter_inputs,
                            ver_inputs=verifier_inputs[0],
                            draft_model=drafter_bundle.model,
                            ver_model=verifier_bundle.model,
                            max_length=resolved_max_new_tokens,
                            draft_tokenizer=drafter_bundle.tokenizer,
                            ver_tokenizer=verifier_bundle.tokenizer,
                            gamma=profile.gamma,
                            draft=drafter_bundle.alias,
                            verifier=verifier_bundle.alias,
                            use_cache=True,
                            sharpen_type=profile.sharpen_type,
                            **gac_kwargs,
                        )
                elif profile.mode == "safe4" and profile.ensemble_type == "gac":
                    generated_ids, accept_rate, num_ensemble = safe_generate_gac_multi(
                        inputs=drafter_inputs,
                        verifier_inputs=verifier_inputs,
                        draft_model=drafter_bundle.model,
                        verifier_models=[bundle.model for bundle in verifier_bundles],
                        max_length=resolved_max_new_tokens,
                        draft_tokenizer=drafter_bundle.tokenizer,
                        verifier_tokenizers=[bundle.tokenizer for bundle in verifier_bundles],
                        gamma=profile.gamma,
                        draft_alias=drafter_bundle.alias,
                        verifier_aliases=[bundle.alias for bundle in verifier_bundles],
                        use_cache=True,
                        sharpen_type=profile.sharpen_type,
                        draft_prob_threshold=0.5,
                        mismatch_prob_threshold=2.0,
                        **gac_kwargs,
                    )
                else:
                    threshold = 1.5 if profile.mode == "official3" else 2.0
                    generated_ids, accept_rate, num_ensemble = safe_generate_unite_multi(
                        inputs=drafter_inputs,
                        verifier_inputs=verifier_inputs,
                        draft_model=drafter_bundle.model,
                        verifier_models=[bundle.model for bundle in verifier_bundles],
                        max_length=resolved_max_new_tokens,
                        draft_tokenizer=drafter_bundle.tokenizer,
                        verifier_tokenizers=[bundle.tokenizer for bundle in verifier_bundles],
                        gamma=profile.gamma,
                        draft_alias=drafter_bundle.alias,
                        verifier_aliases=[bundle.alias for bundle in verifier_bundles],
                        use_cache=True,
                        sharpen_type=profile.sharpen_type,
                        mismatch_prob_threshold=threshold,
                    )

                generation = _decode_generation(drafter_bundle.tokenizer, generated_ids)
                writer.write(
                    _build_output_record(
                        sample=sample,
                        generation=generation,
                        profile=profile,
                        accept_rate=float(accept_rate),
                        num_ensemble=int(num_ensemble),
                        num_generated_tokens=_num_generated_tokens(generated_ids),
                    )
                )
    finally:
        if model_bundles:
            _cleanup_models(model_bundles)
    return output_path
