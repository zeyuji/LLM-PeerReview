"""
Script for generating predictions using a pretrained language model.
This script processes a dataset and saves generated outputs in separate directories for each seed.
"""

import jsonlines
import torch
import gc
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def validate_generation_dataset_size(
    dataset: Sequence[Dict],
    expected_n_samples: int,
    split_name: str,
    data_path: str,
) -> None:
    if isinstance(expected_n_samples, bool) or not isinstance(expected_n_samples, int):
        raise ValueError(f"Dataset config must define an integer {split_name}_size for {data_path}")
    if expected_n_samples <= 0:
        raise ValueError(f"{split_name}_size must be positive, got {expected_n_samples}")
    if len(dataset) != expected_n_samples:
        raise ValueError(
            f"Dataset config {split_name}_size={expected_n_samples} does not match "
            f"{len(dataset)} records in {data_path}"
        )


def get_generation_resume_position(
    seed_files: Sequence[Path],
    dataset: Sequence[Dict],
    output_name: str,
) -> int:
    """Validates aligned seed files and returns the common completed prefix length."""
    dataset_indices = []
    for position, sample in enumerate(dataset):
        missing = {"idx", "task_name", "multi_model_prompt"} - set(sample)
        if missing:
            raise ValueError(
                f"Dataset record {position} is missing fields {sorted(missing)}: {output_name}"
            )
        dataset_indices.append(sample["idx"])
    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError(f"Dataset contains duplicate idx values: {output_name}")

    all_records = []
    for seed_file in seed_files:
        if not seed_file.exists():
            all_records.append([])
            continue
        with jsonlines.open(seed_file) as reader:
            all_records.append(list(reader.iter()))

    existing_counts = [len(records) for records in all_records]
    if len(set(existing_counts)) != 1:
        raise ValueError(
            f"Inconsistent existing line counts across seed files: {existing_counts}. "
            f"Please clean or fix {output_name} before resuming generation."
        )

    existing_lines = existing_counts[0]
    if existing_lines > len(dataset):
        raise ValueError(
            f"Existing line count ({existing_lines}) exceeds dataset size ({len(dataset)}). "
            f"Please clean or fix {output_name} before resuming generation."
        )

    expected_indices = [sample.get("idx") for sample in dataset[:existing_lines]]
    if any(idx is None for idx in expected_indices):
        raise ValueError(f"Dataset records are missing idx values: {output_name}")
    for seed_file, records in zip(seed_files, all_records):
        for position, (record, expected_idx) in enumerate(zip(records, expected_indices)):
            expected_task_name = dataset[position]["task_name"]
            if (
                not isinstance(record.get("generation"), str)
                or record.get("idx") != expected_idx
                or record.get("task_name") != expected_task_name
            ):
                raise ValueError(
                    f"Existing generation prefix does not match the dataset at position {position}: "
                    f"{seed_file}"
                )
    return existing_lines


def generate_predictions(
    model_name: str,
    n_generations: int,
    device: str,
    max_length: int,
    max_new_tokens: int,
    data_path: str,
    output_fpath: Path,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device_map: str = "auto",
    seed: int = 42,
    expected_n_samples: int = None,
    split_name: str = "test",
):
    """
    Generates predictions for a dataset using a pretrained model and saves them to separate directories for each seed.

    Args:
        model_name (str): The name of the pretrained model.
        n_generations (int): Number of generations per sample.
        device (str): The device to use for computation (e.g., "cuda").
        max_length (int): The maximum length of the input sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        data_path (str): Path to the dataset.
        output_fpath (Path): Directory to save the generated outputs.
        temperature (float, optional): Sampling temperature for generation. Defaults to 0.0 (deterministic).
        top_p (float, optional): Top-p value for nucleus sampling. Defaults to 1.0.
        device_map (str, optional): Hugging Face device placement strategy. Defaults to "auto".
    """

    if n_generations <= 0:
        raise ValueError(f"n_generations must be positive, got {n_generations}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if not (0 < top_p <= 1):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    # Create subdirectories for each seed/generation
    output_fpath.mkdir(parents=True, exist_ok=True)
    seed_dirs = [output_fpath / f"Seed-{i}" for i in range(1, n_generations + 1)]
    for seed_dir in seed_dirs:
        seed_dir.mkdir(exist_ok=True)

    # Load the dataset
    with jsonlines.open(data_path) as file:
        dataset = list(file.iter())
    validate_generation_dataset_size(dataset, expected_n_samples, split_name, data_path)

    seed_files = [seed_dir / f"seed_{seed_idx}.jsonl" for seed_idx, seed_dir in enumerate(seed_dirs, start=1)]
    existing_lines = get_generation_resume_position(seed_files, dataset, str(output_fpath))

    # Skip if all data has been processed
    if existing_lines == len(dataset):
        print(f"Results already processed. Skipping.")
        return

    if existing_lines > 0:
        print(f"Resuming generation from line {existing_lines} for {output_fpath}")
    else:
        print(f"Will save results to: {output_fpath}")

    # Load the model and tokenizer
    model, tokenizer = load_hf_model(
        model_name=model_name,
        device=device,
        device_map=device_map,
    )

    # Set up generation parameters based on temperature and top_p
    if temperature == 0.0:
        gen_params = {
            "do_sample": False,
        }
    else:
        gen_params = {
            "temperature": temperature, 
            "top_p": top_p,
            "do_sample": True
        }

    # Initialize progress bar for dataset processing
    remaining_dataset = dataset[existing_lines:]
    progress_bar = tqdm(remaining_dataset, desc="Generating outputs")

    for sample_position, sample in enumerate(remaining_dataset, start=existing_lines):
        prompt = sample["multi_model_prompt"]
        task_name = sample["task_name"]
        idx = sample["idx"]

        # Generate texts for the current sample
        if gen_params["do_sample"]:
            set_seed(seed + sample_position)
        texts = generate_per_sample_single_prompt(
            max_new_tokens=max_new_tokens,
            device=device,
            n_generations=n_generations,
            max_length=max_length,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            gen_params=gen_params,
        )
        
        # Save the generated outputs for each seed
        for i, text in enumerate(texts):
            output = {"task_name": task_name, "generation": text, "idx": idx}
            output_path = seed_dirs[i] / f"seed_{i + 1}.jsonl"
            with jsonlines.open(output_path, "a") as writer:
                writer.write(output)

        progress_bar.update(1)

    # Release model and GPU memory
    cleanup_memory(model, tokenizer)


def generate_per_sample_single_prompt(
    max_new_tokens: int,
    device: str,
    n_generations: int,
    max_length: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    gen_params: Dict,
) -> List[str]:
    """
    Generates predictions for a single sample using a given prompt.

    Args:
        max_new_tokens (int): The maximum number of tokens to generate.
        device (str): The device to use for computation.
        n_generations (int): The number of generations to create.
        max_length (int): The maximum length of the input prompt.
        model (AutoModelForCausalLM): The pretrained model for generation.
        tokenizer (AutoTokenizer): The tokenizer for encoding/decoding.
        prompt (str): The input prompt for generation.
        gen_params (Dict): The generation parameters.

    Returns:
        List[str]: A list of generated text sequences.
    """

    sequence_texts = []
    prompt_encodings = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length,
    ).to(device)

    for _ in range(n_generations):
        with torch.no_grad():
            output = model.generate(
                **prompt_encodings,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                **gen_params,
            )

        # Decode the generated token ids into text
        sequence_texts.append(tokenizer.decode(get_generation_output(prompt_encodings, output)))
        # print(f"Generated response: {sequence_texts[-1]}")

    return sequence_texts


def get_generation_output(input: Dict, output: Dict) -> List[str]:
    """
    Extracts the generated text from the output returned by the model.

    Args:
        input (Dict): The input encodings.
        output (Dict): The output encodings from the model.

    Returns:
        List[str]: The token ids corresponding to the generated text.
    """
    input_len = input["input_ids"].shape[1]
    return output["sequences"][0, input_len:].detach().to("cpu").tolist()


def load_hf_model(
    model_name: str,
    device: str = "cuda",
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a pretrained Hugging Face model and tokenizer.

    Args:
        model_name (str): The name of the pretrained model.
        device (str, optional): The device to load the model on. Defaults to "cuda".

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The model and tokenizer.
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    if device_map != "auto":
        print(f"Loaded model with device_map={device_map}: {getattr(model, 'hf_device_map', {})}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side="left",
        trust_remote_code=True,
    )
    return model, tokenizer


def cleanup_memory(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """
    Cleans up model and tokenizer to release GPU memory.

    Args:
        model (AutoModelForCausalLM): The pretrained model.
        tokenizer (AutoTokenizer): The tokenizer.
    """
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
