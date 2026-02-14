import argparse
from pathlib import Path
import random
from typing import List
import numpy as np
from tqdm import tqdm
import json
import torch
import gc

from Utils.peerreview_util import (
    extract_int_scores,
    get_LLM_response,
)
from Utils.util import (
    load_embedding_input,
    load_model_group_response, 
    load_multi_model_prompt,
)
import Utils.judge_prompt as judge_prompt
from Utils.model_generate_util import load_hf_model
from Utils.util import load_data_config
from Utils.constants import (
    MODEL_GROUPS,
    MODEL_NAME_MAPS,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config", 
    type=str, 
    help="Path to the data yaml config."
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--model_group_scale",
    type=str,
    choices=["New_7B", "New_13B"],
    help="Scale of the model group.",
)
parser.add_argument(
    "--seed",
    type=int,
)
parser.add_argument(
    "--judge_model_name",
    type=str,
)
parser.add_argument(
    "--judge_mode",
    type=str,
    choices=["single", "double", "triple", "multi", "double_bias", "triple_bias", "multi_bias"],
    default="single",
    help="Judge mode for scoring. (single/double/triple/multi/double_bias/triple_bias/multi_bias)",
)
parser.add_argument(
    "--max_score",
    type=str,
    choices=["3", "5", "7", "10"],
    default="5",
    help="Max score for scoring.",
)

def get_max_new_tokens(judge_mode: str) -> int:
    max_tokens_dict = {
        "single": 15,
        "double": 30,
        "double_bias": 30,
        "triple": 40,
        "triple_bias": 40,
        "multi": 50,
        "multi_bias": 50
    }
    return max_tokens_dict.get(judge_mode, 256)

def get_prompt_template(task_type: str, judge_mode: str, max_score: int) -> str:
    """
    Select the appropriate prompt template based on task type, judge mode, and maximum score.

    Args:
        task_type (str): The dataset type (e.g., "factuall_recall", "math", "instruction_following").
        judge_mode (str): The evaluation mode (e.g., "single", "double", "triple", "multi", "double_bias", "triple_bias", "multi_bias").
        max_score (int): The maximum score, e.g., 3, 5, 7, or 10.

    Returns:
        str: The corresponding prompt template.

    Raises:
        ValueError: If an invalid task_type or judge_mode is provided, or if the corresponding template cannot be found.
    """
    
    # Mapping of task type to corresponding suffix
    task_mapping = {
        "factual_recall": "FACT",
        "math": "MATH",
        "instruction_following": "INST_NEW2",
    }

    # Retrieve the suffix corresponding to the task_type
    task_suffix = task_mapping.get(task_type.lower())
    if not task_suffix:
        raise ValueError(f"Invalid task_type: {task_type}. Valid options are: {list(task_mapping.keys())}")

    # Mapping of judge mode to corresponding prefix
    judge_mode_mapping = {
        "single": "SINGLE",
        "double": "DOUBLE",
        "triple": "TRIPLE",
        "multi": "MULTI",
        "double_bias": "DOUBLE",  # 'double_bias' maps to 'DOUBLE'
        "triple_bias": "TRIPLE",  # 'triple_bias' maps to 'TRIPLE'
        "multi_bias": "MULTI"     # 'multi_bias' maps to 'MULTI'
    }

    # Retrieve the prefix corresponding to the judge_mode
    judge_mode_prefix = judge_mode_mapping.get(judge_mode.lower())
    if not judge_mode_prefix:
        raise ValueError(f"Invalid judge_mode: {judge_mode}. Valid options are: {list(judge_mode_mapping.keys())}")

    # Construct the template name
    template_name = f"{judge_mode_prefix}_{max_score}_{task_suffix}"

    # Retrieve the prompt template from the judge_prompt module
    try:
        prompt_template = getattr(judge_prompt, template_name)
    except AttributeError:
        raise ValueError(f"Template '{template_name}' does not exist in judge_prompt module.")

    return prompt_template

def build_prompt(template, **kwargs):
    return template.format(**kwargs)

def get_scores_from_llm(model, tokenizer, prompt, max_new_tokens, expected_n_scores, max_score):
    response = get_LLM_response(model, tokenizer, prompt, max_new_tokens)
    # print(f"LLM response:\n{response.split('---')[0].strip()}")
    default_score = (1 + int(max_score)) / 2
    try:
        scores = extract_int_scores(response, expected_n_scores, int(max_score))
        if len(scores) != expected_n_scores:
            print(f"[Warn] Expected {expected_n_scores} scores, got {scores}")
            scores = [default_score] * expected_n_scores
        # else:
            # print(f"[Info] Extracted scores: {scores}")
    except Exception as e:
        print(f"[Error] Score extraction failed: {e}")
        scores = [default_score] * expected_n_scores

    return scores, response

def score_texts(model, tokenizer, task_type, judge_mode, max_score, texts, questions, embeddings):
    prompt_template = get_prompt_template(task_type, judge_mode, max_score)
    max_new_tokens = get_max_new_tokens(judge_mode)
    # if any(substring in task_type.lower() for substring in ["alpaca", "math", "gsm8k", "trivia_qa"]):
    if any(substring in task_type.lower() for substring in ["instruction_following", "math", "factual_recall"]):
        questions = embeddings

    scorer_map = {
        "single": score_single,
        "double": score_double,
        "double_bias": score_double_bias,
        "triple": score_triple,
        "triple_bias": score_triple_bias,
        "multi": score_multi,
        "multi_bias": score_multi_bias,
    }

    if judge_mode not in scorer_map:
        raise ValueError(f"Invalid judge_mode: {judge_mode}")

    scores = scorer_map[judge_mode](model, tokenizer, prompt_template, max_new_tokens, max_score, texts, questions)
    n_samples = len(texts)
    n_models = len(texts[0])
    for i in range(n_samples):
        print(f"Sample {i} scores: {scores[i*n_models:(i+1)*n_models]}")
    # print("scores: {}".format(scores))

    return scores

def score_double(model, tokenizer, prompt_template, max_new_tokens, max_score, texts, questions):
    n_samples = len(texts)
    n_models = len(texts[0])

    prompts, indices = [], []

    for i in range(n_samples):
        for j in range(n_models):
            for k in range(n_models):
                if j == k: continue
                prompts.append(build_prompt(
                    prompt_template,
                    question=questions[i],
                    response1=texts[i][j],
                    response2=texts[i][k]
                ))
                indices.append((i, j, k))

    score_buckets = [[] for _ in range(n_samples)]

    for idx in tqdm(range(len(prompts)), desc="Scoring double"):
        i, j, k = indices[idx]
        scores, _ = get_scores_from_llm(model, tokenizer, prompts[idx], max_new_tokens, 2, max_score)
        score_buckets[i].append((j, scores[0]))
        score_buckets[i].append((k, scores[1]))

    result_scores = []
    for i in range(n_samples):
        model_score_map = {}
        for model_idx, score in score_buckets[i]:
            model_score_map.setdefault(model_idx, []).append(score)

        for j in range(n_models):
            if j in model_score_map:
                avg_score = sum(model_score_map[j]) / len(model_score_map[j])
            else:
                avg_score = (1 + int(max_score)) / 2
            result_scores.append(avg_score)

    return result_scores

def score_double_bias(model, tokenizer, prompt_template, max_new_tokens, max_score, texts, questions):
    n_samples = len(texts)
    n_models = len(texts[0])

    scores = []

    double_prompts = []
    double_indices = []  # Store (i, j, k) mapping

    for i in range(n_samples):
        model_responses = [texts[i][j] for j in range(n_models)]
        
        # Shuffle model indices for randomization
        model_indices = list(range(n_models))
        random.shuffle(model_indices)

        # Create model pairs for comparison
        model_pairs = [
            (model_indices[0], model_indices[1]),
            (model_indices[1], model_indices[2]),
            (model_indices[2], model_indices[3]),
            (model_indices[3], model_indices[0]),
        ]

        # Generate prompt for each pair
        for j, k in model_pairs:
            prompt = build_prompt(
                prompt_template,
                question=questions[i],
                response1=model_responses[j],
                response2=model_responses[k]
            )
            double_prompts.append(prompt)
            double_indices.append((i, j, k))

    print(f"double indices: {double_indices}")
    print("double prompts generated successfully!")

    score_buckets = [[] for _ in range(n_samples)]  # Store scores for each sample

    # Get scores for all generated prompts
    for idx in tqdm(range(len(double_prompts)), desc="Scoring double texts"):
        prompt = double_prompts[idx]
        i, j, k = double_indices[idx]

        # Get the scores using get_scores_from_llm (this function already handles extraction)
        scores_list, response = get_scores_from_llm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            expected_n_scores=2,  # We expect 2 scores for the pair (j, k)
            max_score=max_score
        )

        print(f"Sample {i} pair ({j}, {k}) response:\n{response.split('---')[0].strip()}")
        print(f"Sample {i} pair ({j}, {k}) scores: {scores_list}")

        # Accumulate the scores for the models
        score_buckets[i].append((j, scores_list[0]))
        score_buckets[i].append((k, scores_list[1]))

    # Calculate average score for each model
    for i in range(n_samples):
        model_score_map = {}
        for model_idx, score in score_buckets[i]:
            model_score_map.setdefault(model_idx, []).append(score)

        print(f"sample {i} score map: {model_score_map}")

        for j in range(n_models):
            if j in model_score_map:
                avg_score = sum(model_score_map[j]) / len(model_score_map[j])
                scores.append(avg_score)
            else:
                scores.append((1 + int(max_score)) / 2)

    return scores

def score_single(model, tokenizer, prompt_template, max_new_tokens, max_score, texts, questions):
    n_samples = len(texts)
    n_models = len(texts[0])

    prompts = []
    for i in range(n_samples):
        for j in range(n_models):
            prompts.append(build_prompt(
                prompt_template,
                question=questions[i],
                response=texts[i][j]
            ))

    # Shuffle the prompts to introduce randomness
    index_map = list(range(len(prompts)))
    random.shuffle(index_map)
    shuffled_prompts = [prompts[i] for i in index_map]

    # Scoring the shuffled prompts
    shuffled_scores = []
    for idx in tqdm(range(len(shuffled_prompts)), desc="Scoring single"):
        scores, _ = get_scores_from_llm(model, tokenizer, shuffled_prompts[idx], max_new_tokens, 1, max_score)
        print(f"Sample {idx // n_models} response{idx % n_models} scores: {scores}")

        shuffled_scores.append(scores[0])  # Assuming we get a single score for each prompt

    # Reorder the scores back to the original order using the index map
    ordered_scores = [0] * len(prompts)
    for i, score in zip(index_map, shuffled_scores):
        ordered_scores[i] = score

    return ordered_scores

def score_triple(model, tokenizer, prompt_template, max_new_tokens, max_score, texts, questions):
    n_samples = len(texts)  # Number of samples
    n_models = len(texts[0])  # Number of models

    prompts, indices = [], []  # Lists to store generated prompts and their indices

    for i in range(n_samples):
        model_responses = [texts[i][j] for j in range(n_models)]  # Get model responses for the i-th sample
        
        # Shuffle model indices for randomness
        model_indices = list(range(n_models))
        random.shuffle(model_indices)

        # Generate model pairs for comparison
        model_pairs = [
            (model_indices[0], model_indices[1], model_indices[2]),
            (model_indices[1], model_indices[2], model_indices[3]),
            (model_indices[2], model_indices[3], model_indices[0]),
            (model_indices[3], model_indices[0], model_indices[1]),
            (model_indices[2], model_indices[1], model_indices[0]),
            (model_indices[3], model_indices[2], model_indices[1]),
            (model_indices[0], model_indices[3], model_indices[2]),
            (model_indices[1], model_indices[0], model_indices[3]),
        ]

        # Create prompt for each model pair
        for j, k, l in model_pairs:
            prompt = build_prompt(
                prompt_template,
                question=questions[i],
                response1=model_responses[j],
                response2=model_responses[k],
                response3=model_responses[l],
            )
            prompts.append(prompt)
            indices.append((i, j, k, l))

    score_buckets = [[] for _ in range(n_samples)]  # Store scores for each sample

    # Score each prompt
    for idx in tqdm(range(len(prompts)), desc="Scoring triple"):
        i, j, k, l = indices[idx]
        # scores, _ = get_scores_from_llm(model, tokenizer prompts[idx], max_new_tokens, 3, max_score)
        scores, response = get_scores_from_llm(model, tokenizer, prompts[idx], max_new_tokens, 3, max_score)
        # print(f"Sample {i} pair ({j}, {k}, {l}) prompt:\n{prompts[idx]}")
        print(f"Sample {i} pair ({j}, {k}, {l}) response:\n{response.split('---')[0].strip()}")
        print(f"Sample {i} pair ({j}, {k}, {l}) scores: {scores}")

        # Store scores for each model in the sample bucket
        score_buckets[i].append((j, scores[0]))
        score_buckets[i].append((k, scores[1]))
        score_buckets[i].append((l, scores[2]))

    result_scores = []
    
    
    # Calculate average score for each model
    for i in range(n_samples):
        model_score_map = {}
        for model_idx, score in score_buckets[i]:
            model_score_map.setdefault(model_idx, []).append(score)
        
        print(f"Sample {i} scores:")
        for j in range(n_models):
            print(f"model {j}: {model_score_map[j]}")
            if j in model_score_map:
                avg_score = sum(model_score_map[j]) / len(model_score_map[j])
            else:
                avg_score = (1 + int(max_score)) / 2  # Default score if no score is available
            result_scores.append(avg_score)

    return result_scores

def score_triple_bias(model, tokenizer, prompt_template, max_new_tokens, max_score, texts, questions):
    n_samples = len(texts)
    n_models = len(texts[0])

    scores = []

    triple_prompts = []
    triple_indices = []  # Store (i, j, k, l) mapping

    for i in range(n_samples):
        model_responses = [texts[i][j] for j in range(n_models)]
        
        # Shuffle model indices for randomization
        model_indices = list(range(n_models))
        random.shuffle(model_indices)

        # Create model triplets for comparison
        model_triplets = [
            (model_indices[0], model_indices[1], model_indices[2]),
            (model_indices[1], model_indices[2], model_indices[3]),
            (model_indices[2], model_indices[3], model_indices[0]),
            (model_indices[3], model_indices[0], model_indices[1]),
        ]

        # Generate prompt for each model triplet
        for j, k, l in model_triplets:
            prompt = build_prompt(
                prompt_template,
                question=questions[i],
                response1=model_responses[j],
                response2=model_responses[k],
                response3=model_responses[l]
            )
            triple_prompts.append(prompt)
            triple_indices.append((i, j, k, l))

    print(f"triple indices: {triple_indices}")
    print("triple prompts generated successfully!")

    score_buckets = [[] for _ in range(n_samples)]  # Store scores for each sample

    # Get scores for all generated prompts
    for idx in tqdm(range(len(triple_prompts)), desc="Scoring triple texts"):
        prompt = triple_prompts[idx]
        i, j, k, l = triple_indices[idx]

        # Get the scores using get_scores_from_llm (this function already handles extraction)
        scores_list, response = get_scores_from_llm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            expected_n_scores=3,  # We expect 3 scores for the triplet (j, k, l)
            max_score=max_score
        )

        print(f"Sample {i} pair ({j}, {k}, {l}) response:\n{response.split('---')[0].strip()}")
        print(f"Sample {i} pair ({j}, {k}, {l}) scores: {scores_list}")

        # Accumulate the scores for the models
        score_buckets[i].append((j, scores_list[0]))
        score_buckets[i].append((k, scores_list[1]))
        score_buckets[i].append((l, scores_list[2]))

    # Calculate average score for each model
    for i in range(n_samples):
        model_score_map = {}
        for model_idx, score in score_buckets[i]:
            model_score_map.setdefault(model_idx, []).append(score)

        print(f"sample {i} score map: {model_score_map}")

        for j in range(n_models):
            if j in model_score_map:
                avg_score = sum(model_score_map[j]) / len(model_score_map[j])
                scores.append(avg_score)
            else:
                scores.append((1 + int(max_score)) / 2)

    return scores

def score_multi(model, tokenizer, prompt_template, max_new_tokens, max_score, texts, questions):
    n_samples = len(texts)  # Number of samples
    n_models = len(texts[0])  # Number of models

    scores = []

    for idx in tqdm(range(n_samples), desc="Scoring multi texts"):
        aggregated_scores = [0.0] * n_models  # Store aggregated scores for each model

        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    # Shuffle responses and select remaining models for comparison
                    remaining_responses = [k for k in range(n_models) if k != i and k != j]
                    random.shuffle(remaining_responses)

                    # Build the prompt using the shuffled responses
                    prompt = build_prompt(
                        prompt_template,
                        question=questions[idx],
                        response1=texts[idx][i],
                        response2=texts[idx][j],
                        response3=texts[idx][remaining_responses[0]],
                        response4=texts[idx][remaining_responses[1]],
                    )

                    # Get the model response and scores using the new function
                    scores_list, response = get_scores_from_llm(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        expected_n_scores=4,
                        max_score=max_score
                    )

                    print(f"\n[Shuffle Round] Sample {idx} response:\n{response.split('---')[0].strip()}")

                    print(f"[Shuffle Round] Scores: {scores_list}")
                    print(f"Original Indices: {i, j, remaining_responses[0], remaining_responses[1]}")

                    # Reorder the scores to match the original response order
                    reordered_scores = [0.0] * n_models
                    reordered_scores[i] = scores_list[0]
                    reordered_scores[j] = scores_list[1]
                    reordered_scores[remaining_responses[0]] = scores_list[2]
                    reordered_scores[remaining_responses[1]] = scores_list[3]

                    print(f"[Shuffle Round] Reordered Scores: {reordered_scores}")

                    # Accumulate scores for averaging
                    for k in range(n_models):
                        aggregated_scores[k] += reordered_scores[k]

        # Average the scores from the rounds
        averaged_scores = [s / 12 for s in aggregated_scores]
        print(f"[Final] Averaged Scores for Sample {idx}: {averaged_scores}")

        scores.extend(averaged_scores)

    # Output the final scores for all samples
    for i in range(n_samples):
        print(f"Sample {i} score: {scores[i*n_models:(i+1)*n_models]}")

    return scores

def score_multi_bias(model, tokenizer, prompt_template, max_new_tokens, max_score, texts, questions, n_shuffle=3):
    n_samples = len(texts)
    n_models = len(texts[0])

    scores = []

    for idx in tqdm(range(n_samples), desc="Scoring multi texts"):
        aggregated_scores = [0.0] * n_models  # Store aggregated scores for each model
        
        for shuffle_round in range(n_shuffle):
            # Shuffle responses and keep track of original indices
            responses = [(texts[idx][j], j) for j in range(n_models)]
            random.shuffle(responses)

            shuffled_responses = [r[0] for r in responses]
            original_indices = [r[1] for r in responses]

            # Format the prompt with shuffled responses
            prompt = build_prompt(
                prompt_template,
                question=questions[idx],
                response1=shuffled_responses[0],
                response2=shuffled_responses[1],
                response3=shuffled_responses[2],
                response4=shuffled_responses[3],
            )

            # Get scores from the LLM response
            scores_list, response = get_scores_from_llm(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                expected_n_scores=4,
                max_score=max_score
            )

            print(f"\n[Shuffle Round {shuffle_round + 1}] Sample {idx} response:\n{response.split('---')[0].strip()}")
            print(f"[Shuffle Round {shuffle_round + 1}] Scores: {scores_list}")
            print(f"Original Indices: {original_indices}")

            # Reorder the scores to match the original response order
            reordered_scores = [0.0] * n_models
            for i, original_idx in enumerate(original_indices):
                reordered_scores[original_idx] = scores_list[i]

            print(f"[Shuffle Round {shuffle_round + 1}] Reordered Scores: {reordered_scores}")

            # Accumulate scores for averaging
            for i in range(n_models):
                aggregated_scores[i] += reordered_scores[i]

        # Average the scores from all shuffle rounds
        averaged_scores = [s / n_shuffle for s in aggregated_scores]
        print(f"[Final] Averaged Scores for Sample {idx}: {averaged_scores}")

        scores.extend(averaged_scores)

    # Output the final scores for all samples
    for i in range(n_samples):
        print(f"Sample {i} score: {scores[i * n_models:(i + 1) * n_models]}")

    return scores

def judge_model(
    seed: int,
    output_fpath: Path,
    data_config: dict,
    model_group_scale: str,
    model_group: list,
    judge_model_name: str,
    judge_mode: str,
    max_score: str,
):
    """
    Perform peer review-based averaging method for model generations.

    This method involves scoring the model generations by using a judge model, 
    calculating the average score for each model's output, and selecting the highest-scoring 
    generation for each sample.
    """
    # Create a directory for saving score matrices if not already present
    scores_dir = output_fpath.parent / "scores_matrices"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Load the model generation results for the test set
    test_generations = load_model_group_response(
        response_path=f"./LLM_Response/Test/{model_group_scale}",
        model_group=model_group,
        data_name=data_config["dataset"],
        seed=seed
    )
    test_prompts = load_multi_model_prompt(f"./Datasets/{data_config['dataset']}/test.jsonl")
    test_embeddings = load_embedding_input(f"./Datasets/{data_config['dataset']}/test.jsonl")
    
    # Get the number of models and samples
    n_models = len(test_generations)
    n_samples = len(test_generations[0])
    current_judge_scores = np.zeros((n_samples, n_models), dtype=np.float32)

    # Define the file path for saving scores
    judge_name_safe = judge_model_name.replace("/", "_")  # Replace unsafe characters in filenames
    json_filename = scores_dir / f"judge_{judge_name_safe}_seed_{seed}.json"

    # If the score file already exists, skip the processing and load the existing scores
    if json_filename.exists():
        print(f"File {json_filename} already exists. Skipping...")
        with open(json_filename, 'r') as f:
            data = json.load(f)
        current_judge_scores = np.array(data["scores"])
    else:
        # Load the judge model if the scores file does not exist
        judge_model_path = MODEL_NAME_MAPS[judge_model_name]
        print(f"Preparing to load model: {judge_model_name}")
        judge_model, tokenizer = load_hf_model(model_name=judge_model_path)
        print(f"Loaded model {judge_model_name} successfully!")

        # Prepare the texts for scoring: each sample and model generation needs to be scored
        texts_to_score, text_mapping, questions, embeddings = [], [], [], []
        
        for sample_idx in range(n_samples):
            sample_texts = []
            for generator_idx in range(n_models):
                text = test_generations[generator_idx][sample_idx]
                sample_texts.append(text)
                text_mapping.append((sample_idx, generator_idx))
            questions.append(test_prompts[sample_idx])
            embeddings.append(test_embeddings[sample_idx])
            
            texts_to_score.append(sample_texts)

        # Perform batch scoring using the judge model
        print("Starting score calculation...")
        scores = score_texts(
            model=judge_model, 
            tokenizer=tokenizer, 
            task_type=data_config["task_type"],
            judge_mode=judge_mode,
            texts=texts_to_score, 
            questions=questions,
            embeddings=embeddings,
            max_score=max_score,
        )

        # Populate the score matrix with the calculated scores
        for idx, score in enumerate(scores):
            sample_idx, generator_idx = text_mapping[idx]
            current_judge_scores[sample_idx][generator_idx] = score
        
        # Save the score matrix to a JSON file
        json_data = {
            "judge_model": judge_model_name,
            "samples": n_samples,
            "generators": model_group,
            "scores": current_judge_scores.tolist()
        }

        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved scores for judge {judge_model_name} to: {json_filename}")

        # Release model memory
        del judge_model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # Select the best generation for each sample based on the highest score
    final_generations = []
    best_model_indices = []

    for sample_idx in range(n_samples):
        scores = current_judge_scores[sample_idx]
        max_score = np.max(scores)
        max_indices = np.where(scores == max_score)[0]
        best_model_idx = np.random.choice(max_indices)
        best_model_indices.append(best_model_idx)
        final_generations.append(test_generations[best_model_idx][sample_idx])
    
    # Save the final selected generations in a JSONL file
    print("Preparing to save judge results...")
    judge_dir = output_fpath.parent / "judge_results"
    judge_dir.mkdir(parents=True, exist_ok=True)
    judge_filename = judge_dir / f"judge_{judge_name_safe}_seed_{seed}.jsonl"
    
    # Prepare the results lines for saving
    results_lines = [
        {
            "task_name": data_config["dataset"],
            "generation": text,
            "idx": idx,
            "selected_model": int(best_model_idx)  # Convert model index to integer
        }
        for idx, (text, best_model_idx) in enumerate(zip(final_generations, best_model_indices))
    ]
    
    # Save the results
    print(f"Saving judge results to {judge_filename}")
    with open(judge_filename, 'w', encoding='utf-8') as f:
        for line in results_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def main(args: argparse.Namespace):
    random.seed(42)
    data_config = load_data_config(args.dataset_config)
    model_groups = MODEL_GROUPS

    output_fpath = Path(args.results_dir) / data_config["dataset"]
    output_fpath.mkdir(parents=True, exist_ok=True)
    output_fpath = output_fpath / f"seed_{args.seed}.jsonl"

    judge_model(
        seed=args.seed,
        output_fpath=output_fpath,
        data_config=data_config,
        model_group_scale=args.model_group_scale,
        model_group=model_groups[args.model_group_scale],
        judge_model_name=args.judge_model_name,
        judge_mode=args.judge_mode,
        max_score=args.max_score,
    )


if __name__ == "__main__":
    print("#" * 100)
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)