import json
from pathlib import Path
import yaml
import jsonlines
from typing import Dict, List, Union, Mapping
import numpy as np

def load_data_config(data_config_path: str) -> Dict:
    """
    Loads the data config from a yaml file.
    """
    return yaml.load(Path(data_config_path).read_text(), Loader=yaml.FullLoader)

def construct_dataset_path(data_dir: str, test_or_train: str) -> str:
    """
    Constructs the path to the dataset.
    """
    return data_dir + "/" + test_or_train + ".jsonl"

def load_response_by_path(response_path: str) -> List:
    """
    Loads the response from a jsonl file.
    """
    ret = []
    with jsonlines.open(response_path) as f:
        for line in f:
            ret.append(line["generation"])
    return ret

def load_reference(reference_path: str) -> Union[List[str], List[List[str]]]:
    """
    Loads the reference from a jsonl file.
    """
    ret = []
    with jsonlines.open(reference_path) as f:
        for line in f:
            ret.append(line["reference"])
    return ret

def load_multi_model_prompt(multi_model_prompt_path: str) -> List:
    """
    Loads the multi model prompt from a jsonl file.
    """
    ret = []
    with jsonlines.open(multi_model_prompt_path) as f:
        for line in f:
            prompt = line["multi_model_prompt"]
            if isinstance(prompt, list):
                ret.append(prompt[0])
            else:
                ret.append(str(prompt))
    return ret

def load_reference_new(reference_path: str) -> Union[List[str], List[List[str]]]:
    """
    Loads the reference from a json file.
    """
    ret = []
    with open(reference_path, 'r') as f:
        data = json.load(f)
        for line in data:
            ret.append(line["reference_output"])
    return ret

def load_prompt_new(prompt_path: str) -> List:
    """
    Loads the prompt from a json file.
    """
    ret = []
    with open(prompt_path, 'r') as f:
        data = json.load(f)
        for line in data:
            ret.append(line["prompt"])
            # print(ret[-1])
    return ret

def load_instruction_new(instruction_path: str) -> List:
    """
    Loads the instruction from a json file.
    """
    ret = []
    with open(instruction_path, 'r') as f:
        data = json.load(f)
        for line in data:
            ret.append(line["instruction"])
    return ret

def load_embedding_input(embedding_input_path: str) -> List:
    """
    Loads the embedding input from a jsonl file.
    """
    ret = []
    with jsonlines.open(embedding_input_path) as f:
        for line in f:
            ret.append(line["embedding_input"])
    return ret

def load_response(response_path: str, seed: int) -> List:
    """
    Loads the response from a jsonl file.
    """
    response_path = response_path + "/" + "Seed-" + str(seed) + "/" + "seed_" + str(seed) + ".jsonl"
    ret = []
    with jsonlines.open(response_path) as f:
        for line in f:
            ret.append(line["generation"])
    return ret

def load_model_group_response(response_path: str, model_group: List, data_name: str, seed: int) -> List:
    """
    Loads the response from a jsonl file.
    """
    ret = []
    for model in model_group:
        new_response_path = response_path + "/" + model + "/" + data_name
        ret.append(load_response(new_response_path, seed))
    return ret

def load_task_name(reference_path: str) -> List:
    ret = []
    with jsonlines.open(reference_path) as f:
        for line in f:
            ret.append(line["task_name"])
    return ret

def load_id(id_path: str) -> List:
    ret = []
    with jsonlines.open(id_path) as f:
        for line in f:
            ret.append(line["id"])
    return ret

def load_selected_model(selected_model_path: str) -> List:
    ret = []
    with jsonlines.open(selected_model_path) as f:
        for line in f:
            ret.append(line["selected_model"])
    return ret

def load_gpt_score(gpt_score_path: str) -> List:
    ret = []
    with jsonlines.open(gpt_score_path) as f:
        for line in f:
            ret.append(line["gpt_score"])
    return ret

def load_gpt_scores(gpt_score_paths: List[str]) -> List[List[float]]:
    gpt_scores = []
    for path in gpt_score_paths:
        scores = load_gpt_score(path)
        gpt_scores.append(scores)
    return gpt_scores

def clean_generation(generation: str):
    """
    Extracts a generation from the full output of the model.
    """
    generation = generation.replace("<pad>", "")
    generation = generation.replace("<unk>", "")
    generation = generation.replace("<end_of_turn>", "")
    generation = generation.replace("<|endoftext|>", "")
    generation = generation.replace("<s>", "")
    generation = generation.replace("</s>", "")
    generation = generation.replace("</eos>", "")
    generation = generation.replace("\\n", "\n")
    return generation.strip()


def clean_generations(
    generations: Union[List[str], List[List[str]]]
) -> Union[List[str], List[List[str]]]:
    """
    Applies clean_generation to each element in a 1D or 2D list of generations.

    Args:
        generations (Union[List[str], List[List[str]]]): A 1D or 2D list of generations.

    Returns:
        Union[List[str], List[List[str]]]: A list with the same structure as the input, but with each generation cleaned.
    """
    if isinstance(generations[0], list) or isinstance(generations[0], np.ndarray):
        # 2D list
        return [[clean_generation(gen) for gen in sample] for sample in generations]
    else:
        # 1D list
        return [clean_generation(gen) for gen in generations]