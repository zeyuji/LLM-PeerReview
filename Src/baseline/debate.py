import argparse
from pathlib import Path
import json
import jsonlines
import numpy as np
from tqdm import tqdm
import torch
import gc
import time

from Utils.util import (
    load_indices,
    load_model_group_response,
    load_data_config,
    clean_generation,
)
from Utils.constants import MODEL_GROUPS, MODEL_NAME_MAPS
from Utils.model_generate_util import load_hf_model, get_generation_output


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_config", type=str, required=True, help="Path to the data yaml config.")
parser.add_argument(
    "--results_dir",
    type=str,
    default="./LLM_Response/Debate",
    help="Directory to save debate results to",
)
parser.add_argument(
    "--model_group_scale",
    type=str,
    choices=["New_7B"],
    default="New_7B",
    help="Scale of the model group.",
)
parser.add_argument(
    "--n_debate_rounds",
    type=int,
    default=3,
    help="Number of debate rounds (excluding Round 0). Total rounds = n_debate_rounds + 1",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Random seed for initial response loading.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to use for model inference.",
)


def load_round_0_responses(
    model_group_scale: str,
    model_group: list,
    data_name: str,
    seed: int,
    sample_indices: list,
) -> np.ndarray:
    test_generations = load_model_group_response(
        response_path=f"./LLM_Response/Test/{model_group_scale}",
        model_group=model_group,
        data_name=data_name,
        seed=seed,
        expected_indices=sample_indices,
    )
    return np.array(test_generations)


def get_round_file_path(
    results_dir: str,
    model_group_scale: str,
    data_name: str,
    model_name: str,
    round_idx: int,
) -> Path:
    path = Path(results_dir) / model_group_scale / data_name / f"round_{round_idx}"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{model_name}.jsonl"


def save_round_responses(
    responses: list,
    file_path: Path,
    data_config: dict,
    sample_indices: list,
):
    if len(responses) != len(sample_indices):
        raise ValueError(
            f"Cannot save debate round: responses={len(responses)}, idx={len(sample_indices)}"
        )
    if any(not isinstance(response, str) for response in responses):
        raise ValueError(f"Cannot save debate round with non-string responses: {file_path}")
    results_lines = [
        {
            "task_name": data_config["dataset"],
            "generation": text,
            "idx": sample_idx,
        }
        for sample_idx, text in zip(sample_indices, responses)
    ]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in results_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def load_round_responses(file_path: Path, sample_indices: list, expected_task_name: str) -> list:
    records = []
    with jsonlines.open(file_path) as f:
        records = list(f.iter())
    if len(records) != len(sample_indices):
        raise ValueError(
            f"Incomplete debate round file {file_path}: {len(records)} records, "
            f"expected {len(sample_indices)}"
        )
    actual_indices = [record.get("idx") for record in records]
    if actual_indices != sample_indices:
        raise ValueError(f"Debate round idx ordering does not match the dataset: {file_path}")
    if len(actual_indices) != len(set(actual_indices)):
        raise ValueError(f"Duplicate idx values in debate round file: {file_path}")
    for position, record in enumerate(records):
        if not isinstance(record.get("generation"), str):
            raise ValueError(f"Missing generation at position {position} in {file_path}")
        if record.get("task_name") != expected_task_name:
            raise ValueError(f"Debate round task_name mismatch at position {position}: {file_path}")
    return [record["generation"] for record in records]


def construct_debate_prompt_math(
    original_question: str,
    other_agent_responses: list,
) -> str:
    prompt = "These are the solutions to the problem from other agents:\n\n"
    
    for response in other_agent_responses:
        prompt += f"One agent solution:\n{response}\n\n"
    
    prompt += "Using the solutions from other agents as additional information, can you provide your answer to the math problem?\n\n"
    prompt += f"The original math problem is {original_question}\n\n"
    prompt += "Your final answer should be a single numerical number, in the form 'the answer is {answer}', at the end of your response."
    
    return prompt


def generate_round_for_model(
    model_name: str,
    model_path: str,
    round_idx: int,
    n_samples: int,
    questions: list,
    prev_round_responses: np.ndarray,
    model_idx: int,
    data_config: dict,
    device: str = "cuda",
) -> list:
    if (
        len(questions) != n_samples
        or prev_round_responses.ndim != 2
        or prev_round_responses.shape[1] != n_samples
    ):
        raise ValueError(
            f"Debate inputs are misaligned: samples={n_samples}, questions={len(questions)}, "
            f"previous_round_shape={prev_round_responses.shape}"
        )

    print(f"\n{'='*80}")
    print(f"Generating Round {round_idx} for model: {model_name}")
    print(f"{'='*80}")
    
    print(f"Loading model: {model_path}")
    model, tokenizer = load_hf_model(model_name=model_path, device=device)
    
    max_length = 4096
    max_new_tokens = int(data_config["max_new_tokens"])
    if max_new_tokens <= 0:
        raise ValueError(f"Debate max_new_tokens must be positive, got {max_new_tokens}")
    
    current_round_responses = []
    
    for sample_idx in tqdm(range(n_samples), desc=f"Round {round_idx}, {model_name}"):
        other_responses = []
        for m_idx in range(len(prev_round_responses)):
            if m_idx != model_idx:
                other_responses.append(prev_round_responses[m_idx][sample_idx])
        
        original_question = questions[sample_idx]
        prompt = construct_debate_prompt_math(
            original_question=original_question,
            other_agent_responses=other_responses,
        )
        
        prompt_encodings = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        gen_params = {"do_sample": False}
        
        with torch.no_grad():
            output = model.generate(
                **prompt_encodings,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                **gen_params,
            )
        
        generated_ids = get_generation_output(prompt_encodings, output)
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = clean_generation(response)
        
        current_round_responses.append(response)
    
    print(f"Unloading model: {model_name}")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return current_round_responses


def run_debate(
    results_dir: str,
    model_group_scale: str,
    model_group: list,
    data_config: dict,
    n_debate_rounds: int,
    seed: int,
    device: str,
):
    data_name = data_config["dataset"]
    n_models = len(model_group)
    total_rounds = n_debate_rounds + 1
    if n_debate_rounds < 0:
        raise ValueError(f"n_debate_rounds must be non-negative, got {n_debate_rounds}")
    
    print("\n" + "="*80)
    print("Step 1: Loading original questions")
    print("="*80)
    original_data_path = Path("./Datasets") / data_name / "test.jsonl"
    questions = []
    with jsonlines.open(original_data_path) as f:
        for line in f:
            questions.append(line["embedding_input"])
    sample_indices = load_indices(str(original_data_path))
    n_samples = len(questions)
    if len(sample_indices) != n_samples:
        raise ValueError("Debate dataset prompt and idx counts do not match")
    if int(data_config["test_size"]) != n_samples:
        raise ValueError(
            f"Dataset config test_size={data_config['test_size']} does not match {n_samples} records"
        )
    print(f"Loaded {n_samples} questions")
    
    print("\n" + "="*80)
    print("Step 2: Processing Round 0")
    print("="*80)
    
    all_rounds_responses = []
    
    round_0_responses = load_round_0_responses(
        model_group_scale=model_group_scale,
        model_group=model_group,
        data_name=data_name,
        seed=seed,
        sample_indices=sample_indices,
    )
    if round_0_responses.shape != (n_models, n_samples):
        raise ValueError(
            f"Unexpected Round 0 response shape {round_0_responses.shape}, "
            f"expected {(n_models, n_samples)}"
        )
    all_rounds_responses.append(round_0_responses)
    
    for model_idx, model_name in enumerate(model_group):
        round_0_file = get_round_file_path(
            results_dir=results_dir,
            model_group_scale=model_group_scale,
            data_name=data_name,
            model_name=model_name,
            round_idx=0,
        )
        if not round_0_file.exists():
            print(f"Saving Round 0 for {model_name}")
            save_round_responses(
                responses=round_0_responses[model_idx],
                file_path=round_0_file,
                data_config=data_config,
                sample_indices=sample_indices,
            )
        else:
            saved_responses = load_round_responses(
                round_0_file,
                sample_indices,
                expected_task_name=data_name,
            )
            if saved_responses != round_0_responses[model_idx].tolist():
                raise ValueError(f"Existing Round 0 file differs from the source responses: {round_0_file}")
            print(f"Round 0 for {model_name} already exists and is valid, skipping")
    
    print("\n" + "="*80)
    print(f"Step 3: Running Debate for {n_debate_rounds} rounds")
    print("="*80)
    
    for round_idx in range(1, total_rounds):
        print(f"\n{'='*80}")
        print(f"Starting Round {round_idx}")
        print(f"{'='*80}")
        
        prev_round_responses = all_rounds_responses[round_idx - 1]
        
        current_round_responses = []
        
        for model_idx, model_name in enumerate(model_group):
            model_path = MODEL_NAME_MAPS[model_name]
            
            current_round_file = get_round_file_path(
                results_dir=results_dir,
                model_group_scale=model_group_scale,
                data_name=data_name,
                model_name=model_name,
                round_idx=round_idx,
            )
            
            if current_round_file.exists():
                print(f"\nRound {round_idx} for {model_name} already exists, loading from file")
                responses = load_round_responses(
                    current_round_file,
                    sample_indices,
                    expected_task_name=data_name,
                )
                current_round_responses.append(responses)
                continue
            
            responses = generate_round_for_model(
                model_name=model_name,
                model_path=model_path,
                round_idx=round_idx,
                n_samples=n_samples,
                questions=questions,
                prev_round_responses=prev_round_responses,
                model_idx=model_idx,
                data_config=data_config,
                device=device,
            )
            
            save_round_responses(
                responses=responses,
                file_path=current_round_file,
                data_config=data_config,
                sample_indices=sample_indices,
            )
            
            current_round_responses.append(responses)
        
        all_rounds_responses.append(np.array(current_round_responses))
    
    print("\n" + "="*80)
    print("Debate completed successfully!")
    print(f"Results saved to: {results_dir}/{model_group_scale}/{data_name}/")
    print("="*80)


def main(args: argparse.Namespace):
    data_config = load_data_config(args.dataset_config)
    model_groups = MODEL_GROUPS
    
    run_debate(
        results_dir=args.results_dir,
        model_group_scale=args.model_group_scale,
        model_group=model_groups[args.model_group_scale],
        data_config=data_config,
        n_debate_rounds=args.n_debate_rounds,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    
    start_time = time.perf_counter()
    main(args)
    end_time = time.perf_counter()
    duration_s = end_time - start_time
    print(f"\nRuntime: {duration_s:.3f}s")
