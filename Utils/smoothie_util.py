from typing import List
from sklearn.neighbors import NearestNeighbors
import numpy as np
import jsonlines
import json
import argparse
from pathlib import Path

from Utils.embedder import Embedder
from Utils.smoothie_model import Smoothie
from Utils.util import load_model_group_response


def run_smoothie(
    args: argparse.Namespace,
    seed: int,
    original_data_path: Path,
    output_fpath: Path,
    data_config: dict,
    model_group_scale: str,
    model_group: list,
    embedder: Embedder,
):

    test_dataset = []
    with jsonlines.open(original_data_path) as file:
        test_dataset = list(file.iter())
    test_input_embeddings = embedder.embed_dataset(test_dataset)
    # [n_samples, embedding_dim]

    test_generations_for_smoothie = load_model_group_response(
        response_path="./LLM_Response/Test/"+model_group_scale,
        model_group=model_group,
        data_name=data_config["dataset"],
        seed=seed
    )
    test_generations_for_selection = load_model_group_response(
        response_path="./LLM_Response/Test/"+model_group_scale,
        model_group=model_group,
        data_name=data_config["dataset"],
        seed=seed
    )
    test_generations_for_selection = np.array(test_generations_for_selection)
    # [n_models, n_samples]
    test_generations_for_selection = test_generations_for_selection.transpose()
    # [n_samples, n_models]
    smoothie_text = np.array(test_generations_for_smoothie)
    # [n_models, n_samples]
    smoothie_text = smoothie_text.transpose()
    # [n_samples, n_models]

    clean = data_config["dataset"] not in ["mix_instruct", "alpaca", "gsm8k"]
    smoothie_embeddings = embedder.embed_individual_generations(
        individual_generations=smoothie_text,
        clean=clean,
    )
    # (n_samples, n_models, embedding_dim)
    n_samples = len(smoothie_embeddings)
    n_voters = smoothie_embeddings.shape[1]
    embed_dim = smoothie_embeddings.shape[2]

    if args.type == "sample_dependent":
        # use KNN
        nbrs = NearestNeighbors(n_neighbors=args.k, algorithm="auto")
        nbrs.fit(
            test_input_embeddings
        )  # not the same as smoothie_embeddings! only kernel-smooth based on x similarity

        _, test_indices = nbrs.kneighbors(test_input_embeddings)

        smoothie_dataset_weights = []
        for sample_idx in range(n_samples):
            if args.k == 1:
                embs_per_sample = smoothie_embeddings[sample_idx].reshape(
                    (1, n_voters, -1)
                )
            else:
                embs_per_sample = smoothie_embeddings[test_indices[sample_idx]]
            smoothie = Smoothie(n_voters=n_voters, dim=embed_dim)
            smoothie.fit(embs_per_sample)
            smoothie_dataset_weights.append(smoothie.theta)
        smoothie_dataset_weights = np.array(smoothie_dataset_weights)
    else:
        # learn a single set of weights for all samples
        smoothie = Smoothie(n_voters=n_voters, dim=embed_dim)
        smoothie.fit(smoothie_embeddings)
        smoothie_dataset_weights = np.tile(smoothie.theta, (n_samples, 1))

    dataset_texts = []
    logs = []
    for sample_idx in range(n_samples):
        # print(f"sample{sample_idx} weights:{smoothie_dataset_weights[sample_idx]})")
        max_idx = smoothie_dataset_weights[sample_idx].argmax()
        logs.append(max_idx)
        text = test_generations_for_selection[sample_idx][max_idx]
        dataset_texts.append(text)

    results_lines = [
        {
            "task_name": data_config["dataset"],
            "generation": text,
            "idx": idx,
            "selected_model": int(logs[idx]),
            "smoothie_weights": smoothie_dataset_weights[idx].tolist(),
        }
        for idx, text in enumerate(dataset_texts)
    ]
    print(f"Saving results to {output_fpath}")
    with open(output_fpath, 'w', encoding='utf-8') as f:
        for line in results_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')