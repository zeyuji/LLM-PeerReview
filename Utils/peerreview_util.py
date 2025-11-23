import re
import numpy as np
from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_LLM_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    device: str = "cuda:0",
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """
    Generates a response from the language model (LLM) for a given prompt.
    
    Args:
        model (AutoModelForCausalLM): Pretrained language model.
        tokenizer (AutoTokenizer): Tokenizer for encoding the prompt.
        prompt (str): Input prompt.
        max_new_tokens (int): Maximum number of new tokens to generate. Default is 512.
        device (str): Device to run the model on. Default is "cuda:0".
        temperature (float): Sampling temperature. Default is 0.0 (deterministic).
        top_p (float): Nucleus sampling parameter. Default is 1.0.

    Returns:
        str: Model's generated response.
    """
    prompt_encodings = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    ).to(device)

    gen_params = {
        "do_sample": False,
    }

    response = model.generate(
        **prompt_encodings,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        temperature=temperature,
        top_p=top_p,
        **gen_params,
    )

    from Utils.model_generate_util import get_generation_output
    return tokenizer.decode(
        get_generation_output(prompt_encodings, response), 
        skip_special_tokens=True
    )

def parse_int_score(output_str: str) -> int:
    """
    Extracts an integer score (1-5) from the output string.

    Args:
        output_str (str): Output string from which to extract the score.

    Returns:
        int: Extracted score, default is 3 if not found or invalid.
    """
    match = re.search(r'[0-5]', output_str)
    score = 3  # Default score

    try:
        if match:
            score = int(match.group())
            if score == 0:
                print("Warning: Score is 0, converted to 1")
                score = 1
            elif not 1 <= score <= 5:
                print("Warning: Score out of valid range, returned default value 3")
        else:
            print("Warning: No valid score found, returning default 3")
    except Exception as e:
        print(f"Error: {e}")
        print("Warning: Invalid score extraction, returning default 3")

    return score

def parse_float_score(output_str: str) -> float:
    """
    Extracts a floating-point score (1-5) from the output string.

    Args:
        output_str (str): Output string from which to extract the score.

    Returns:
        float: Extracted score, default is 3.0 if not found or invalid.
    """
    match = re.search(r"\b([0-5]\.\d*)\b", output_str)
    score = 3.0  # Default score

    try:
        score = float(match.group(1)) if match else score
        if score == 0.0:
            print("Warning: Score is 0.0, converted to 1.0")
            score = 1.0
        elif not 1.0 <= score <= 5.0:
            print("Warning: Score out of valid range, converted to 3.0")
    except:
        print("Warning: Invalid score extraction, returning default 3.0")
    
    return score

def extract_int_scores(s: str, n: int, max_score: int = 5) -> List[int]:
    """
    Extracts the first n integer scores within the range of 1-5 from a string.

    Args:
        s (str): Input string containing scores.
        n (int): Number of scores to extract.
        max_score (int): Maximum valid score. Default is 5.

    Returns:
        List[int]: List of extracted scores.
    """
    nums = re.findall(r'\d+', s)

    filtered = []
    for num in nums:
        value = int(num)
        if value == 0:
            filtered.append(1)
        elif 1 <= value <= max_score:
            filtered.append(value)
        if len(filtered) == n:
            break
    return filtered

def calculate_final_distribution(P: List[float], T: float) -> List[float]:
    """
    Computes the final probability distribution after applying temperature scaling and softmax.

    Args:
        P (List[float]): Input probability distribution.
        T (float): Temperature for scaling.

    Returns:
        List[float]: Final probability distribution.
    """
    def softmax(logits: List[float]) -> List[float]:
        exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
        return exp_logits / np.sum(exp_logits)

    logits = np.log(P)  # Convert to logits
    logits_scaled = logits / T  # Apply temperature scaling
    return softmax(logits_scaled)

def optimize_scores(
    scores_array: np.ndarray, 
    n_samples: int, 
    n_models: int, 
    n_judges: int, 
    max_score: int, 
    consider_prior: str = "False", 
    epoch: int = 10, 
    t: float = 0.5
) -> np.ndarray:
    """
    Optimizes the score distribution based on logits, temperature scaling, and softmax.

    Args:
        scores_array (np.ndarray): Original score array of shape (n_samples, n_models).
        n_samples (int): Number of samples.
        n_models (int): Number of models.
        max_score (int): Maximum score for normalization.
        consider_prior (str): Whether to consider prior knowledge. Default is "False".
        epoch (int): Number of epochs for optimization.
        t (float): Temperature for scaling the probability distribution.

    Returns:
        np.ndarray: Optimized score array.
    """
    from Utils.inference_class import Inference_Wuzhangai

    annotations = {}
    for i in range(n_samples):
        for j in range(n_models):
            sample_id = f"sample_{i}_{j}"
            ratings = scores_array[i, j, :] - 1
            rating_dict = {f"rater_{k}": int(ratings[k]) for k in range(n_judges)}
            annotations[sample_id] = [rating_dict]

    params = {
        'num_classes': [max_score],
        'mode': 'train_init',
        'epoch': epoch,
        'patient': 5,
        'consider_prior': consider_prior,
        'label_hard': "False",
    }

    infer_engine = Inference_Wuzhangai(params, annotations)
    infer_engine.inference_init(annotations)

    distributions, final_classes, _, _ = infer_engine.train()

    optimized_array = np.zeros((n_samples, n_models), dtype=np.float32)
    for i in range(n_samples):
        for j in range(n_models):
            sample_id = f"sample_{i}_{j}"
            prob_dist = distributions[0][sample_id]
            # final_score = final_classes[sample_id] + 1
            opt_dist = calculate_final_distribution(np.array(prob_dist), float(t))
            expected_score = sum((k + 1) * opt_dist[k] for k in range(max_score))
            optimized_array[i, j] = expected_score

    return optimized_array


