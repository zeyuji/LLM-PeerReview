import json
from pathlib import Path
import yaml
import jsonlines
from typing import Dict, List, Union, Mapping, Optional, Sequence
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

def _validate_record_indices(
    records: Sequence[Dict],
    data_path: str,
    expected_indices: Optional[Sequence] = None,
) -> List:
    indices = []
    for line_number, record in enumerate(records, start=1):
        if "idx" not in record:
            raise ValueError(f"Missing idx at line {line_number} in {data_path}")
        indices.append(record["idx"])

    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate idx values in {data_path}")
    if expected_indices is not None and indices != list(expected_indices):
        mismatch = next(
            (
                position
                for position, (actual, expected) in enumerate(zip(indices, expected_indices))
                if actual != expected
            ),
            min(len(indices), len(expected_indices)),
        )
        raise ValueError(
            f"Record idx ordering in {data_path} does not match the dataset at position {mismatch}: "
            f"records={len(indices)}, expected={len(expected_indices)}"
        )
    return indices


def load_response_by_path(response_path: str, expected_indices: Optional[Sequence] = None) -> List:
    """
    Loads the response from a jsonl file.
    """
    with jsonlines.open(response_path) as f:
        records = list(f.iter())
    _validate_record_indices(records, response_path, expected_indices)
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record.get("generation"), str):
            raise ValueError(f"Missing generation at line {line_number} in {response_path}")
    return [record["generation"] for record in records]

def load_indices(data_path: str) -> List:
    """Loads record indices from a dataset or response JSONL file."""
    with jsonlines.open(data_path) as f:
        records = list(f.iter())
    return _validate_record_indices(records, data_path)

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

def load_response(response_path: str, seed: int, expected_indices: Optional[Sequence] = None) -> List:
    """
    Loads the response from a jsonl file.
    """
    response_path = response_path + "/" + "Seed-" + str(seed) + "/" + "seed_" + str(seed) + ".jsonl"
    return load_response_by_path(response_path, expected_indices=expected_indices)

def load_model_group_response(
    response_path: str,
    model_group: List,
    data_name: str,
    seed: int,
    expected_indices: Optional[Sequence] = None,
) -> List:
    """
    Loads the response from a jsonl file.
    """
    ret = []
    for model in model_group:
        new_response_path = response_path + "/" + model + "/" + data_name
        ret.append(load_response(new_response_path, seed, expected_indices=expected_indices))
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

def load_selected_model(selected_model_path: str, expected_indices: Optional[Sequence] = None) -> List:
    with jsonlines.open(selected_model_path) as f:
        records = list(f.iter())
    _validate_record_indices(records, selected_model_path, expected_indices)
    for line_number, record in enumerate(records, start=1):
        if "selected_model" not in record:
            raise ValueError(f"Missing selected_model at line {line_number} in {selected_model_path}")
    return [record["selected_model"] for record in records]

def load_gpt_score(
    gpt_score_path: str,
    expected_instructions: Optional[Sequence[str]] = None,
    expected_references: Optional[Sequence[str]] = None,
) -> List:
    with jsonlines.open(gpt_score_path) as f:
        records = list(f.iter())

    if expected_instructions is not None and len(records) != len(expected_instructions):
        raise ValueError(
            f"GPT score count in {gpt_score_path} is {len(records)}, "
            f"expected {len(expected_instructions)}"
        )
    if expected_references is not None and len(records) != len(expected_references):
        raise ValueError(
            f"GPT score count in {gpt_score_path} is {len(records)}, "
            f"expected {len(expected_references)}"
        )

    scores = []
    for position, record in enumerate(records):
        if "gpt_score" not in record:
            raise ValueError(f"Missing gpt_score at line {position + 1} in {gpt_score_path}")
        if expected_instructions is not None and record.get("instruction") != expected_instructions[position]:
            raise ValueError(
                f"Instruction mismatch at line {position + 1} in {gpt_score_path}"
            )
        if expected_references is not None and record.get("reference") != expected_references[position]:
            raise ValueError(
                f"Reference mismatch at line {position + 1} in {gpt_score_path}"
            )
        scores.append(record["gpt_score"])
    return scores

def load_gpt_scores(
    gpt_score_paths: List[str],
    expected_instructions: Optional[Sequence[str]] = None,
    expected_references: Optional[Sequence[str]] = None,
) -> List[List[float]]:
    gpt_scores = []
    for path in gpt_score_paths:
        scores = load_gpt_score(
            path,
            expected_instructions=expected_instructions,
            expected_references=expected_references,
        )
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
    if not generations:
        return []
    if isinstance(generations[0], list) or isinstance(generations[0], np.ndarray):
        # 2D list
        return [[clean_generation(gen) for gen in sample] for sample in generations]
    else:
        # 1D list
        return [clean_generation(gen) for gen in generations]
