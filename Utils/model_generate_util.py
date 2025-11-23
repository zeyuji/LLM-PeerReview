"""
Script for generating predictions using a pretrained language model.
This script processes a dataset and saves generated outputs in separate directories for each seed.
"""

import jsonlines
import torch
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    """

    # Check if the results file already exists and determine how many lines have been generated
    existing_lines = 0
    if output_fpath.exists():
        existing_lines = len((output_fpath / "Seed-1" / "seed_1.jsonl").read_text().splitlines())
        print(f"Results file {output_fpath} exists. Existing lines: {existing_lines}")
    else:
        print(f"Will save results to: {output_fpath}")

    # Create subdirectories for each seed/generation
    output_fpath.mkdir(parents=True, exist_ok=True)
    seed_dirs = [output_fpath / f"Seed-{i}" for i in range(1, n_generations + 1)]
    for seed_dir in seed_dirs:
        seed_dir.mkdir(exist_ok=True)

    # Load the dataset
    with jsonlines.open(data_path) as file:
        dataset = list(file.iter())

    # Skip if all data has been processed
    if existing_lines == len(dataset):
        print(f"Results already processed. Skipping.")
        return

    # Load the model and tokenizer
    model, tokenizer = load_hf_model(model_name=model_name, device=device)

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
    progress_bar = tqdm(dataset[existing_lines:], desc="Generating outputs")

    for sample in dataset[existing_lines:]:
        prompt = sample["multi_model_prompt"]
        task_name = sample["task_name"]
        idx = sample["idx"]

        # Generate texts for the current sample
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
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a pretrained Hugging Face model and tokenizer.

    Args:
        model_name (str): The name of the pretrained model.
        device (str, optional): The device to load the model on. Defaults to "cuda".

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The model and tokenizer.
    """

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for phi3
    gpu_ids = [0, 1, 2, 3, 4, 5]
    max_memory = {i: "28GB" for i in gpu_ids}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # device_map="auto",
        device_map="balanced",      # for phi3
        max_memory = max_memory,    # for phi3
        torch_dtype=torch.float16,  # 节省显存
        low_cpu_mem_usage=True,     # for phi3
    )
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
