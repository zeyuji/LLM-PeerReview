import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import jsonlines
import numpy as np

from Utils.constants import MODEL_GROUPS
from Utils.util import clean_generations, load_data_config, load_indices, load_reference
from Utils.metrics import gsm8k_acc, trivia_qa_acc, parse_answer, weighted_voting


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config",
    type=str,
    required=True,
    help="Path to config file. This should be a yaml file.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="./LLM_Response/Debate",
    help="Debate results root directory.",
)
parser.add_argument(
    "--model_group_scale",
    type=str,
    choices=["New_7B"],
    default="New_7B",
    help="Scale of the model group.",
)
parser.add_argument(
    "--round_idx",
    type=int,
    required=True,
    help="Debate round index to evaluate (e.g., 0, 1, 2, 3...).",
)


def evaluate(
    data_config: Dict,
    responses: List[str],
    references: Union[List[str], List[List[str]]],
) -> float:
    task_generations = clean_generations(responses)
    metrics = data_config["metrics"][0]
    scores: List[int] = []

    if metrics == "trivia_qa_acc":
        scores.extend(trivia_qa_acc(generations=task_generations, references=references))
    elif metrics == "gsm8k_acc":
        scores.extend(gsm8k_acc(generations=task_generations, references=references))
    else:
        raise ValueError(f"Unknown metrics: {metrics}")

    if len(scores) != len(responses):
        raise ValueError(f"Metric returned {len(scores)} scores for {len(responses)} responses")
    return float(np.mean(scores))


def _load_round_file_as_idx_map(file_path: Path, expected_task_name: str) -> Dict[int, str]:
    idx_to_gen: Dict[int, str] = {}
    with jsonlines.open(file_path) as reader:
        for line in reader:
            if line.get("task_name") != expected_task_name:
                raise ValueError(f"Debate round task_name mismatch in {file_path}")
            if not isinstance(line.get("generation"), str):
                raise ValueError(f"Missing generation in {file_path}")
            idx = int(line["idx"])
            if idx in idx_to_gen:
                raise ValueError(f"Duplicate idx {idx} in {file_path}")
            idx_to_gen[idx] = line["generation"]
    return idx_to_gen


def _build_ordered_list(idx_to_gen: Dict[int, str], sample_indices: List[int], *, file_path: Path) -> List[str]:
    expected = set(sample_indices)
    missing = [idx for idx in sample_indices if idx not in idx_to_gen]
    unexpected = [idx for idx in idx_to_gen if idx not in expected]
    if missing:
        raise ValueError(f"Missing idx in {file_path}: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        raise ValueError(f"Unexpected idx in {file_path}: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    return [idx_to_gen[idx] for idx in sample_indices]


def main(args: argparse.Namespace):
    if args.round_idx < 0:
        raise ValueError(f"round_idx must be non-negative, got {args.round_idx}")
    data_config = load_data_config(args.dataset_config)
    data_name = data_config["dataset"]

    data_path = Path("./Datasets") / data_name / "test.jsonl"
    references = load_reference(str(data_path))
    sample_indices = load_indices(str(data_path))
    n_samples = len(references)
    if len(sample_indices) != n_samples:
        raise ValueError("Dataset reference and idx counts do not match")
    if int(data_config["test_size"]) != n_samples:
        raise ValueError(
            f"Dataset config test_size={data_config['test_size']} does not match {n_samples} records"
        )

    model_group = MODEL_GROUPS[args.model_group_scale]
    round_dir = Path(args.results_dir) / args.model_group_scale / data_name / f"round_{args.round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Load per-model generations for the given round
    per_model_generations: List[List[str]] = []
    for model_name in model_group:
        fp = round_dir / f"{model_name}.jsonl"
        if not fp.exists():
            raise FileNotFoundError(f"Missing round file: {fp}")
        idx_to_gen = _load_round_file_as_idx_map(fp, expected_task_name=data_name)
        per_model_generations.append(_build_ordered_list(idx_to_gen, sample_indices, file_path=fp))

    # Majority voting (unweighted): parse -> cluster by mathematical equivalence -> tie random
    mv_generations: List[str] = []
    details_lines: List[Dict[str, Any]] = []
    for i in range(n_samples):
        gens_i = [per_model_generations[m][i] for m in range(len(model_group))]
        answers_parsed = [parse_answer(g) for g in gens_i]
        vote = weighted_voting(answers_parsed, [1.0] * len(answers_parsed))
        selected = vote["selected_answer"]
        details_lines.append(
            {
                "task_name": data_name,
                "idx": sample_indices[i],
                "answers_parsed": {model_name: ans for model_name, ans in zip(model_group, answers_parsed)},
                "mv_selected_answer": selected,
                "mv_answer_clusters": vote["answer_clusters"],
            }
        )
        if selected is None:
            mv_generations.append("")
        else:
            mv_generations.append(f"the answer is {selected}")

    # Save MV results into the same round directory
    mv_path = round_dir / "MV.jsonl"
    with open(mv_path, "w", encoding="utf-8") as f:
        for idx, text in zip(sample_indices, mv_generations):
            f.write(
                json.dumps(
                    {"task_name": data_name, "generation": text, "idx": idx},
                    ensure_ascii=False,
                )
                + "\n"
            )

    details_path = round_dir / "MV.details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details_lines, f, ensure_ascii=False, indent=2)

    # Evaluate MV
    score = evaluate(data_config=data_config, responses=mv_generations, references=references)

    out_path = round_dir / "MV.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Mean score: {score}\n")

    print(f"MV saved to: {mv_path}")
    print(f"MV details saved to: {details_path}")
    print(f"MV evaluation saved to: {out_path}")
    print(f"Mean score: {score}")


if __name__ == "__main__":
    print("Src.evaluate.evaluate_debate...")
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)
    print("#" * 100)
