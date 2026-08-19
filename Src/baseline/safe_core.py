import copy
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn import Module


SAFE_DIR = Path(__file__).resolve().parents[2] / "SAFE"
if str(SAFE_DIR) not in sys.path:
    sys.path.insert(0, str(SAFE_DIR))

from safe_generate_gac import get_token_ids


def _cache_seq_length(cache) -> int:
    if cache is None:
        return 0

    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())

    if isinstance(cache, (tuple, list)):
        for layer_cache in cache:
            if layer_cache is None:
                continue
            for tensor in layer_cache:
                if tensor is not None:
                    return int(tensor.shape[-2])

    raise TypeError(f"Unsupported KV cache type: {type(cache).__name__}")


def _crop_cache_to_length(cache, target_length: int):
    """Crop a KV cache to an absolute sequence length."""
    if cache is None:
        return None
    if target_length < 0:
        raise ValueError(f"KV cache target length must be non-negative, got {target_length}.")

    current_length = _cache_seq_length(cache)
    if target_length > current_length:
        raise ValueError(
            f"Cannot grow KV cache from {current_length} to {target_length}; "
            "the missing tokens must be forwarded through the model."
        )

    crop = getattr(cache, "crop", None)
    if callable(crop):
        crop(target_length)
        cropped_cache = cache
    elif isinstance(cache, tuple):
        cropped_cache = tuple(
            None
            if layer_cache is None
            else tuple(
                None if tensor is None else tensor[..., :target_length, :]
                for tensor in layer_cache
            )
            for layer_cache in cache
        )
    else:
        raise TypeError(f"Unsupported KV cache type: {type(cache).__name__}")

    cropped_length = _cache_seq_length(cropped_cache)
    if cropped_length != target_length:
        raise RuntimeError(
            f"KV cache crop produced length {cropped_length}, expected {target_length}."
        )
    return cropped_cache


def _prune_cache_tokens(cache, num_tokens_to_discard: int):
    if cache is None or num_tokens_to_discard == 0:
        return cache
    if num_tokens_to_discard < 0:
        raise ValueError(
            f"Number of KV cache tokens to discard must be non-negative, got {num_tokens_to_discard}."
        )
    return _crop_cache_to_length(cache, _cache_seq_length(cache) - num_tokens_to_discard)


def _rollback_draft_state(
    draft_cache,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    current_position: int,
    num_accepted: int,
):
    """Keep accepted draft KV states and mark replacement tokens as pending input."""
    retained_length = current_position + num_accepted
    if retained_length > input_ids.shape[-1]:
        raise ValueError(
            f"Retained prefix length {retained_length} exceeds input length {input_ids.shape[-1]}."
        )

    if draft_cache is not None:
        draft_cache = _crop_cache_to_length(draft_cache, retained_length)

    retained_mask = attention_mask[:, :retained_length]
    pending_length = input_ids.shape[-1] - retained_length
    attention_mask = torch.cat(
        [
            retained_mask,
            retained_mask.new_ones((retained_mask.shape[0], pending_length)),
        ],
        dim=-1,
    )
    if attention_mask.shape[-1] != input_ids.shape[-1]:
        raise RuntimeError(
            f"Attention mask length {attention_mask.shape[-1]} does not match "
            f"input length {input_ids.shape[-1]}."
        )

    return draft_cache, attention_mask


def get_top_k_tokens(
    logits: torch.Tensor,
    tokenizer,
    k: int = 10,
    internlm: bool = False,
) -> List[Dict[str, List[object]]]:
    top_k_indices = torch.topk(logits, k).indices
    logits_list = logits.tolist()

    top_k_values = []
    for idx, val_row in zip(top_k_indices, logits_list):
        val_item = []
        for token_idx in idx:
            val_item.append(val_row[token_idx])
        top_k_values.append(val_item)

    values = []
    for val, idx in zip(top_k_values, top_k_indices):
        if internlm:
            values.append(
                {
                    tokenizer.decode([27960, int(token_id)], skip_special_tokens=False)[1:]: [prob, int(token_id)]
                    for prob, token_id in zip(val, idx)
                }
            )
        else:
            values.append(
                {
                    tokenizer.decode([int(token_id)], skip_special_tokens=False): [prob, int(token_id)]
                    for prob, token_id in zip(val, idx)
                }
            )

    return values


def get_union_vocab(*vocabs: List[Dict[str, List[object]]]) -> List[List[str]]:
    unique_tokens = []
    for per_step_vocab in zip(*vocabs):
        merged_tokens = set()
        for vocab in per_step_vocab:
            merged_tokens.update(vocab.keys())
        unique_tokens.append(list(merged_tokens))
    return unique_tokens


def _get_blank_id(tokenizer) -> int:
    blank_ids = tokenizer.encode(" ", add_special_tokens=False)
    if blank_ids:
        return blank_ids[0]
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


def update_vocab(
    vocab: List[Dict[str, List[object]]],
    union_vocab: List[List[str]],
    tokenizer,
    logits: torch.Tensor,
) -> List[Dict[str, List[object]]]:
    blank_id = _get_blank_id(tokenizer)

    for union_tokens, vocab_tokens, logit_row in zip(union_vocab, vocab, logits):
        existing_token_ids = {item[1] for item in vocab_tokens.values()}
        for token in union_tokens:
            if token in vocab_tokens:
                continue

            if token == "":
                subtoken_id = blank_id
                logit = logit_row[subtoken_id]
            else:
                subtokens = tokenizer.tokenize(token)
                subtoken_ids = tokenizer.convert_tokens_to_ids(subtokens)
                if subtoken_ids and len(subtoken_ids) == 1:
                    subtoken_id = subtoken_ids[0]
                    logit = logit_row[subtoken_id]
                else:
                    subtoken_id = blank_id
                    logit = logit_row[subtoken_id]
                    for token_id in subtoken_ids:
                        if token_id != blank_id:
                            subtoken_id = token_id
                            logit = logit_row[subtoken_id]
                            break

            if subtoken_id not in existing_token_ids:
                vocab_tokens[token] = [logit, subtoken_id]
                existing_token_ids.add(subtoken_id)
            else:
                vocab_tokens[token] = [torch.tensor(0.0), subtoken_id]

    return vocab_softmax(vocab)


def vocab_softmax(vocab: List[Dict[str, List[object]]]) -> List[Dict[str, List[object]]]:
    normalized = []
    for element in vocab:
        item_probs = []
        item_ids = []
        for value in element.values():
            item_probs.append(value[0])
            item_ids.append(value[1])
        probs = torch.softmax(torch.tensor(item_probs), dim=0)
        normalized.append(
            {
                token: [prob, token_id]
                for token, prob, token_id in zip(element.keys(), probs, item_ids)
            }
        )
    return normalized


def average_and_sample(
    vocabularies: Sequence[List[Dict[str, List[object]]]],
    drafter_original_keys: Sequence[str],
    sharpen_type: str,
) -> Tuple[List[str], List[List[int]]]:
    next_tokens = []
    next_token_ids_by_model = [[] for _ in range(len(vocabularies))]

    for per_step_vocab in zip(*vocabularies):
        reference_vocab = per_step_vocab[0]
        averaged = {}
        for token in reference_vocab:
            avg_prob = sum(vocab[token][0] for vocab in per_step_vocab) / len(per_step_vocab)
            averaged[token] = [avg_prob, reference_vocab[token][1]]

        max_prob = max(value[0].item() for value in averaged.values())
        if max_prob < 0.5:
            if sharpen_type == "geom":
                sharpened = {}
                for token in reference_vocab:
                    product = torch.tensor(1.0)
                    for vocab in per_step_vocab:
                        product = product * vocab[token][0]
                    sharpened[token] = [product ** (1.0 / len(per_step_vocab)), reference_vocab[token][1]]
                averaged = sharpened
            else:
                filtered = copy.deepcopy(averaged)
                filtered_keys = [
                    token
                    for token, value in filtered.items()
                    if value[0].item() > 0.1 and token in drafter_original_keys
                ]
                for top_key in filtered_keys:
                    if top_key == " ":
                        continue
                    for other_key in averaged:
                        if other_key != top_key and other_key.startswith(top_key):
                            averaged[top_key][0] = averaged[top_key][0] + filtered[other_key][0].item()

        probs = [value[0] for value in averaged.values()]
        sample_index = probs.index(max(probs))
        selected_token = list(averaged.keys())[sample_index]
        next_tokens.append(selected_token)
        for model_idx, vocab in enumerate(per_step_vocab):
            next_token_ids_by_model[model_idx].append(vocab[selected_token][1])

    return next_tokens, next_token_ids_by_model


def get_ensemble_token_multi(
    outputs: Sequence[torch.Tensor],
    tokenizers,
    sharpen_type: str,
    mapping_matrices,
    vocab_union,
    index_to_vocab,
    special_prefix_tokens_dict,
    byte_mappings_list,
    top_k: int = 10,
):
    eos_token_list = [tokenizer.eos_token for tokenizer in tokenizers]
    eos_token_list.extend(["<|end_of_text|>", "<|endoftext|>", "<|im_end|>", "<|end|>"])

    num_models = len(outputs)
    sharpen_threshold = 0.5 * num_models
    filtered_threshold = 0.1 * num_models

    merged_probs = torch.zeros(
        (outputs[0].size(0), len(vocab_union)),
        device=outputs[0].device,
    )
    unified_probs = []
    top_token_indices = []

    for output, mapping_matrix in zip(outputs, mapping_matrices):
        unified_prob = torch.sparse.mm(output.float(), mapping_matrix.to(output.device))
        unified_probs.append(unified_prob)
        merged_probs += unified_prob.to(merged_probs.device)
        top_k_size = min(top_k, unified_prob.shape[-1])
        top_token_indices.append(torch.topk(unified_prob[0], top_k_size).indices)

    max_prob = torch.max(merged_probs).item()
    if max_prob < sharpen_threshold:
        if sharpen_type == "geom":
            sharpened = torch.ones_like(merged_probs)
            for unified_prob in unified_probs:
                sharpened = sharpened * unified_prob.to(sharpened.device)
            merged_probs = sharpened.pow(1.0 / num_models)
        else:
            clone_merged_probs = merged_probs.clone()
            filtered_tokens = torch.nonzero(
                merged_probs[0] > filtered_threshold,
                as_tuple=False,
            ).flatten()
            if filtered_tokens.numel() > 0:
                drafter_top_tokens = top_token_indices[0].to(filtered_tokens.device)
                filtered_tokens = filtered_tokens[torch.isin(filtered_tokens, drafter_top_tokens)]

            candidate_tokens = set()
            for token_indices in top_token_indices:
                candidate_tokens.update(token_indices.tolist())

            for top_ind_tensor in filtered_tokens:
                top_ind = top_ind_tensor.item()
                top_token = index_to_vocab[top_ind]
                if top_token == " ":
                    continue
                for other_ind in candidate_tokens:
                    if other_ind == top_ind:
                        continue
                    if index_to_vocab[other_ind].startswith(top_token):
                        merged_probs[0, top_ind] = (
                            merged_probs[0, top_ind] + clone_merged_probs[0, other_ind].item()
                        )

    max_token_indices = torch.argmax(merged_probs, dim=1)
    max_tokens = [index_to_vocab[index.item()] for index in max_token_indices]

    batch_token_ids = [[] for _ in range(len(tokenizers))]
    for tokenizer_idx, tokenizer in enumerate(tokenizers):
        for token in max_tokens:
            if token in eos_token_list:
                token_ids = [tokenizer.eos_token_id]
            else:
                token_ids = get_token_ids(
                    tokenizer,
                    token,
                    special_prefix_tokens_dict[tokenizer],
                    byte_mappings_list[tokenizer_idx],
                )
            batch_token_ids[tokenizer_idx].append(token_ids)

    return batch_token_ids


def _build_eos_token_ids(tokenizer, model_alias: Optional[str]) -> List[int]:
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    if model_alias in {"qwen2", "qwen2.5"}:
        eos_ids.append(151645)
    if model_alias == "internlm":
        eos_ids.append(128131)

    return list(dict.fromkeys(eos_ids))


def _decode_prefix(tokenizer, token_ids: torch.Tensor, start: int, end: int) -> str:
    return tokenizer.batch_decode(token_ids[:, start:end], skip_special_tokens=False)[0].lstrip(" ")


def _align_verifier_position(
    tokenizer,
    verifier_input_ids: torch.Tensor,
    verifier_prompt_len: int,
    verifier_current_position: int,
    gamma: int,
    previous_draft_seq: str,
) -> int:
    decoded_suffix = tokenizer.batch_decode(
        verifier_input_ids[:, verifier_prompt_len:],
        skip_special_tokens=False,
    )[0]
    if not decoded_suffix.startswith(previous_draft_seq):
        return verifier_input_ids.shape[-1]

    offset = 0
    while not tokenizer.batch_decode(
        verifier_input_ids[:, verifier_prompt_len:verifier_current_position - gamma + offset],
        skip_special_tokens=False,
    )[0].startswith(previous_draft_seq):
        offset += 1

    return verifier_current_position - gamma + offset


def _prepare_verifier_state(
    verifier_models: Sequence[Module],
    verifier_tokenizers: Sequence,
    verifier_inputs: Sequence[Dict[str, torch.Tensor]],
    verifier_prompt_lens: Sequence[int],
    draft_tokenizer,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    verifier_input_ids_list = [item["input_ids"] for item in verifier_inputs]
    verifier_attention_mask_list = [item["attention_mask"] for item in verifier_inputs]

    draft_text = draft_tokenizer.batch_decode(
        input_ids[:, prompt_len:],
        skip_special_tokens=False,
    )
    for verifier_idx, (verifier_model, verifier_tokenizer) in enumerate(
        zip(verifier_models, verifier_tokenizers)
    ):
        retokenized_ids = verifier_tokenizer(
            draft_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(verifier_model.device)["input_ids"]
        verifier_input_ids_list[verifier_idx] = torch.cat(
            [
                verifier_input_ids_list[verifier_idx][:, :verifier_prompt_lens[verifier_idx]],
                retokenized_ids,
            ],
            dim=-1,
        ).to(torch.int64)

    return verifier_input_ids_list, verifier_attention_mask_list


def _forward_verifiers(
    verifier_models: Sequence[Module],
    verifier_input_ids_list: Sequence[torch.Tensor],
    verifier_attention_mask_list: Sequence[torch.Tensor],
    verifier_current_positions: Sequence[int],
    verifier_caches: Sequence,
    use_cache: bool,
):
    verifier_logits_list = []
    max_verify_lens = []

    for verifier_idx, verifier_model in enumerate(verifier_models):
        verifier_input_ids = verifier_input_ids_list[verifier_idx]
        verifier_attention_mask_list[verifier_idx] = torch.ones_like(verifier_input_ids)
        verifier_cache = verifier_caches[verifier_idx]
        if verifier_cache is not None:
            verifier_cache_position = torch.arange(
                _cache_seq_length(verifier_cache),
                verifier_input_ids.shape[-1],
            ).to(verifier_model.device)
        else:
            verifier_cache_position = None

        prepared_inputs = verifier_model.prepare_inputs_for_generation(
            verifier_input_ids,
            attention_mask=verifier_attention_mask_list[verifier_idx],
            past_key_values=verifier_cache,
            use_cache=use_cache,
            cache_position=verifier_cache_position,
        )
        verifier_outputs = verifier_model(**prepared_inputs, return_dict=True)
        verifier_caches[verifier_idx] = verifier_outputs.past_key_values

        max_verify_len = verifier_input_ids.shape[1] - verifier_current_positions[verifier_idx]
        max_verify_lens.append(max_verify_len)
        verifier_logits_list.append(
            verifier_outputs.logits[..., -max_verify_len - 1:-1, :].to(
                copy=True,
                dtype=torch.float32,
                device=verifier_model.device,
            )
        )

    return verifier_logits_list, max_verify_lens


@torch.no_grad()
def safe_generate_unite_multi(
    inputs,
    verifier_inputs: Sequence[Dict[str, torch.Tensor]],
    draft_model: Module,
    verifier_models: Sequence[Module],
    max_length: int = 4096,
    draft_tokenizer=None,
    verifier_tokenizers: Optional[Sequence] = None,
    gamma: int = 5,
    draft_alias: Optional[str] = None,
    verifier_aliases: Optional[Sequence[Optional[str]]] = None,
    use_cache: bool = True,
    sharpen_type: str = "geom",
    mismatch_prob_threshold: Optional[float] = None,
    top_k: int = 10,
) -> Tuple[List[int], float, int]:
    if verifier_tokenizers is None or len(verifier_tokenizers) == 0:
        raise ValueError("safe_generate_unite_multi requires at least one verifier tokenizer.")
    if len(verifier_models) != len(verifier_inputs) or len(verifier_models) != len(verifier_tokenizers):
        raise ValueError("Verifier models, inputs, and tokenizers must have identical lengths.")

    verifier_aliases = list(verifier_aliases or [None] * len(verifier_models))
    if len(verifier_aliases) != len(verifier_models):
        raise ValueError("verifier_aliases must match the number of verifier models.")

    draft_cache = None
    verifier_caches = [None] * len(verifier_models)

    drafter_eos_ids = _build_eos_token_ids(draft_tokenizer, draft_alias)
    verifier_eos_ids = [
        set(_build_eos_token_ids(tokenizer, alias))
        for tokenizer, alias in zip(verifier_tokenizers, verifier_aliases)
    ]

    stop_tokens = torch.tensor(
        drafter_eos_ids,
        dtype=torch.long,
        device=draft_model.device,
    ).unsqueeze(1)
    eos_token_list = [draft_tokenizer.eos_token]
    eos_token_list.extend(tokenizer.eos_token for tokenizer in verifier_tokenizers)
    eos_token_list.extend(["<|end_of_text|>", "<|endoftext|>", "<|im_end|>", "<|end|>", "</s>"])

    threshold = mismatch_prob_threshold
    if threshold is None:
        threshold = 0.5 * (1 + len(verifier_models))

    drafts_accepted = 0.0
    drafts_speculated = 0.0
    num_ensemble = 0

    draft_model.generation_config.do_sample = False
    draft_model.generation_config.temperature = 0.0
    for verifier_model in verifier_models:
        verifier_model.generation_config.do_sample = False
        verifier_model.generation_config.temperature = 0.0

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    verifier_input_ids_list = [item["input_ids"] for item in verifier_inputs]
    verifier_attention_mask_list = [item["attention_mask"] for item in verifier_inputs]

    prompt_len = input_ids.shape[-1]
    total_len = prompt_len + max_length
    verifier_prompt_lens = [item.shape[-1] for item in verifier_input_ids_list]

    current_position = prompt_len
    verifier_current_positions = verifier_prompt_lens.copy()

    while current_position < total_len:
        input_ids = input_ids.to(draft_model.device)
        draft_logits = []

        for _ in range(gamma):
            if draft_cache is None:
                cache_position = torch.arange(0, input_ids.shape[-1]).to(draft_model.device)
            else:
                cache_position = torch.arange(
                    _cache_seq_length(draft_cache),
                    input_ids.shape[-1],
                ).to(draft_model.device)

            drafter_inputs = draft_model.prepare_inputs_for_generation(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=draft_cache,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            draft_outputs = draft_model(**drafter_inputs, return_dict=True)
            draft_cache = draft_outputs.past_key_values
            current_draft_logits = draft_outputs.logits[..., -1, :].to(
                copy=True,
                dtype=torch.float32,
                device=draft_model.device,
            )
            next_token = torch.argmax(current_draft_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            draft_logits.append(current_draft_logits)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )

        drafts_speculated += gamma

        draft_text = draft_tokenizer.batch_decode(
            input_ids[:, prompt_len:],
            skip_special_tokens=False,
        )
        for verifier_idx, (verifier_model, verifier_tokenizer) in enumerate(zip(verifier_models, verifier_tokenizers)):
            retokenized_ids = verifier_tokenizer(
                draft_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(verifier_model.device)["input_ids"]
            verifier_input_ids_list[verifier_idx] = torch.cat(
                [
                    verifier_input_ids_list[verifier_idx][:, :verifier_prompt_lens[verifier_idx]],
                    retokenized_ids,
                ],
                dim=-1,
            ).to(torch.int64)

        if current_position > prompt_len:
            previous_draft_seq = draft_tokenizer.batch_decode(
                input_ids[:, prompt_len:current_position],
                skip_special_tokens=False,
            )[0]
            verifier_current_positions = [
                _align_verifier_position(
                    tokenizer=verifier_tokenizer,
                    verifier_input_ids=verifier_input_ids,
                    verifier_prompt_len=verifier_prompt_len,
                    verifier_current_position=verifier_current_position,
                    gamma=gamma,
                    previous_draft_seq=previous_draft_seq,
                )
                for verifier_tokenizer, verifier_input_ids, verifier_prompt_len, verifier_current_position in zip(
                    verifier_tokenizers,
                    verifier_input_ids_list,
                    verifier_prompt_lens,
                    verifier_current_positions,
                )
            ]

        verifier_logits_list = []
        max_verify_lens = []
        for verifier_idx, verifier_model in enumerate(verifier_models):
            verifier_input_ids = verifier_input_ids_list[verifier_idx]
            verifier_attention_mask_list[verifier_idx] = torch.ones_like(verifier_input_ids)
            verifier_cache = verifier_caches[verifier_idx]
            if verifier_cache is not None:
                verifier_cache_position = torch.arange(
                    _cache_seq_length(verifier_cache),
                    verifier_input_ids.shape[-1],
                ).to(verifier_model.device)
            else:
                verifier_cache_position = None

            prepared_inputs = verifier_model.prepare_inputs_for_generation(
                verifier_input_ids,
                attention_mask=verifier_attention_mask_list[verifier_idx],
                past_key_values=verifier_cache,
                use_cache=use_cache,
                cache_position=verifier_cache_position,
            )
            verifier_outputs = verifier_model(**prepared_inputs, return_dict=True)
            verifier_caches[verifier_idx] = verifier_outputs.past_key_values

            max_verify_len = verifier_input_ids.shape[1] - verifier_current_positions[verifier_idx]
            max_verify_lens.append(max_verify_len)
            verifier_logits_list.append(
                verifier_outputs.logits[..., -max_verify_len - 1:-1, :].to(
                    copy=True,
                    dtype=torch.float32,
                    device=verifier_model.device,
                )
            )

        need_ensemble = False
        num_accepted = gamma
        trigger_positions = None
        trigger_logits = None

        for draft_idx in range(gamma):
            draft_prefix = _decode_prefix(
                draft_tokenizer,
                input_ids,
                prompt_len,
                current_position + draft_idx,
            )

            matched_positions = []
            matched_logits = []
            all_aligned = True
            any_mismatch = False
            probability_sum = torch.softmax(draft_logits[draft_idx], dim=-1)[
                0, input_ids[0, current_position + draft_idx]
            ].item()

            for verifier_idx, (verifier_logits, verifier_input_ids, verifier_tokenizer, verifier_prompt_len, verifier_current_position, max_verify_len) in enumerate(
                zip(
                    verifier_logits_list,
                    verifier_input_ids_list,
                    verifier_tokenizers,
                    verifier_prompt_lens,
                    verifier_current_positions,
                    max_verify_lens,
                )
            ):
                matched_position = None
                for verifier_pos in range(max_verify_len):
                    verifier_prefix = _decode_prefix(
                        verifier_tokenizer,
                        verifier_input_ids,
                        verifier_prompt_len,
                        verifier_current_position + verifier_pos,
                    )
                    if verifier_prefix == draft_prefix:
                        matched_position = verifier_pos
                        break
                    if len(verifier_prefix) > len(draft_prefix):
                        break

                if matched_position is None:
                    all_aligned = False
                    break

                verifier_logit = verifier_logits[:, matched_position]
                token_id = verifier_input_ids[0, verifier_current_position + matched_position].item()
                probability_sum += torch.softmax(verifier_logit, dim=-1)[0, token_id].item()
                if torch.argmax(verifier_logit, dim=-1).item() != token_id:
                    any_mismatch = True
                matched_positions.append(matched_position)
                matched_logits.append(verifier_logit)

            if not all_aligned:
                continue

            if any_mismatch and probability_sum < threshold:
                need_ensemble = True
                num_accepted = draft_idx
                trigger_positions = matched_positions
                trigger_logits = matched_logits
                break

        new_token_len = 0
        if need_ensemble:
            drafts_accepted += num_accepted
            if num_accepted < gamma:
                drafter_vocab = get_top_k_tokens(
                    draft_logits[num_accepted],
                    draft_tokenizer,
                    k=top_k,
                    internlm=(draft_alias == "internlm"),
                )
                drafter_original_keys = copy.deepcopy(list(drafter_vocab[0].keys()))

                verifier_vocabs = []
                for verifier_logit, verifier_tokenizer, verifier_alias in zip(
                    trigger_logits,
                    verifier_tokenizers,
                    verifier_aliases,
                ):
                    verifier_vocabs.append(
                        get_top_k_tokens(
                            verifier_logit,
                            verifier_tokenizer,
                            k=top_k,
                            internlm=(verifier_alias == "internlm"),
                        )
                    )

                union_vocab = get_union_vocab(drafter_vocab, *verifier_vocabs)
                all_vocabularies = [
                    update_vocab(drafter_vocab, union_vocab, draft_tokenizer, draft_logits[num_accepted]),
                ]
                for verifier_vocab, verifier_tokenizer, verifier_logit in zip(
                    verifier_vocabs,
                    verifier_tokenizers,
                    trigger_logits,
                ):
                    all_vocabularies.append(
                        update_vocab(verifier_vocab, union_vocab, verifier_tokenizer, verifier_logit)
                    )

                next_tokens, next_token_ids_by_model = average_and_sample(
                    all_vocabularies,
                    drafter_original_keys,
                    sharpen_type,
                )
                selected_token = next_tokens[0]

                if selected_token in eos_token_list:
                    next_token_ids_by_model[0] = [drafter_eos_ids[-1]]
                else:
                    for verifier_idx, verifier_token_ids in enumerate(next_token_ids_by_model[1:]):
                        if verifier_token_ids[0] in verifier_eos_ids[verifier_idx]:
                            next_token_ids_by_model[0] = [drafter_eos_ids[-1]]
                            break

                new_token_len = len(next_token_ids_by_model[0])
                next_token_ids = torch.tensor(
                    next_token_ids_by_model[0],
                    device=input_ids.device,
                ).unsqueeze(0)
                input_ids = torch.cat(
                    [input_ids[:, :current_position + num_accepted], next_token_ids],
                    dim=-1,
                )
                num_ensemble += 1
        else:
            drafts_accepted += gamma

        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[:, 1].min().item()
            del verifier_caches, draft_cache
            gc.collect()
            torch.cuda.empty_cache()
            return (
                input_ids[:, prompt_len:current_position + stop_location + 1].tolist(),
                drafts_accepted / drafts_speculated,
                num_ensemble,
            )

        if use_cache:
            for verifier_idx, verifier_cache in enumerate(verifier_caches):
                num_pruned_kv = min(
                    verifier_input_ids_list[verifier_idx].shape[-1] - verifier_prompt_lens[verifier_idx],
                    verifier_input_ids_list[verifier_idx].shape[-1] - verifier_current_positions[verifier_idx] + 5,
                )
                verifier_caches[verifier_idx] = _prune_cache_tokens(verifier_cache, num_pruned_kv)

        if num_accepted != gamma:
            draft_cache, attention_mask = _rollback_draft_state(
                draft_cache=draft_cache,
                input_ids=input_ids,
                attention_mask=attention_mask,
                current_position=current_position,
                num_accepted=num_accepted,
            )
            num_accepted += new_token_len

        current_position += num_accepted

    del verifier_caches, draft_cache
    gc.collect()
    torch.cuda.empty_cache()

    return input_ids[:, prompt_len:].tolist(), drafts_accepted / drafts_speculated, num_ensemble


@torch.no_grad()
def safe_generate_gac_multi(
    inputs,
    verifier_inputs: Sequence[Dict[str, torch.Tensor]],
    draft_model: Module,
    verifier_models: Sequence[Module],
    max_length: int = 4096,
    draft_tokenizer=None,
    verifier_tokenizers: Optional[Sequence] = None,
    vocab_union=None,
    mapping_matrices=None,
    index_to_vocab=None,
    byte_mappings_list=None,
    special_prefix_tokens_dict=None,
    gamma: int = 5,
    draft_alias: Optional[str] = None,
    verifier_aliases: Optional[Sequence[Optional[str]]] = None,
    use_cache: bool = True,
    sharpen_type: str = "geom",
    draft_prob_threshold: float = 0.5,
    mismatch_prob_threshold: Optional[float] = None,
    top_k: int = 10,
) -> Tuple[List[int], float, int]:
    if verifier_tokenizers is None or len(verifier_tokenizers) == 0:
        raise ValueError("safe_generate_gac_multi requires at least one verifier tokenizer.")
    if len(verifier_models) != len(verifier_inputs) or len(verifier_models) != len(verifier_tokenizers):
        raise ValueError("Verifier models, inputs, and tokenizers must have identical lengths.")
    if any(
        item is None
        for item in (
            vocab_union,
            mapping_matrices,
            index_to_vocab,
            byte_mappings_list,
            special_prefix_tokens_dict,
        )
    ):
        raise ValueError("safe_generate_gac_multi requires union vocab artifacts from setup_union_vocab.")

    verifier_aliases = list(verifier_aliases or [None] * len(verifier_models))
    if len(verifier_aliases) != len(verifier_models):
        raise ValueError("verifier_aliases must match the number of verifier models.")

    threshold = mismatch_prob_threshold
    if threshold is None:
        threshold = 0.5 * (1 + len(verifier_models))

    draft_cache = None
    verifier_caches = [None] * len(verifier_models)

    drafter_eos_ids = _build_eos_token_ids(draft_tokenizer, draft_alias)
    stop_tokens = torch.tensor(
        drafter_eos_ids,
        dtype=torch.long,
        device=draft_model.device,
    ).unsqueeze(1)

    drafts_accepted = 0.0
    drafts_speculated = 0.0
    num_ensemble = 0

    draft_model.generation_config.do_sample = False
    draft_model.generation_config.temperature = 0.0
    for verifier_model in verifier_models:
        verifier_model.generation_config.do_sample = False
        verifier_model.generation_config.temperature = 0.0

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    verifier_prompt_lens = [item["input_ids"].shape[-1] for item in verifier_inputs]

    prompt_len = input_ids.shape[-1]
    total_len = prompt_len + max_length

    current_position = prompt_len
    verifier_current_positions = verifier_prompt_lens.copy()

    while current_position < total_len:
        input_ids = input_ids.to(draft_model.device)
        draft_logits = []

        for _ in range(gamma):
            if draft_cache is None:
                cache_position = torch.arange(0, input_ids.shape[-1]).to(draft_model.device)
            else:
                cache_position = torch.arange(
                    _cache_seq_length(draft_cache),
                    input_ids.shape[-1],
                ).to(draft_model.device)

            drafter_inputs = draft_model.prepare_inputs_for_generation(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=draft_cache,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            draft_outputs = draft_model(**drafter_inputs, return_dict=True)
            draft_cache = draft_outputs.past_key_values
            current_draft_logits = draft_outputs.logits[..., -1, :].to(
                copy=True,
                dtype=torch.float32,
                device=draft_model.device,
            )
            next_token = torch.argmax(current_draft_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            draft_logits.append(current_draft_logits)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )

        drafts_speculated += gamma
        verifier_input_ids_list, verifier_attention_mask_list = _prepare_verifier_state(
            verifier_models=verifier_models,
            verifier_tokenizers=verifier_tokenizers,
            verifier_inputs=verifier_inputs,
            verifier_prompt_lens=verifier_prompt_lens,
            draft_tokenizer=draft_tokenizer,
            input_ids=input_ids,
            prompt_len=prompt_len,
        )

        if current_position > prompt_len:
            previous_draft_seq = draft_tokenizer.batch_decode(
                input_ids[:, prompt_len:current_position],
                skip_special_tokens=False,
            )[0]
            verifier_current_positions = [
                _align_verifier_position(
                    tokenizer=verifier_tokenizer,
                    verifier_input_ids=verifier_input_ids,
                    verifier_prompt_len=verifier_prompt_len,
                    verifier_current_position=verifier_current_position,
                    gamma=gamma,
                    previous_draft_seq=previous_draft_seq,
                )
                for verifier_tokenizer, verifier_input_ids, verifier_prompt_len, verifier_current_position in zip(
                    verifier_tokenizers,
                    verifier_input_ids_list,
                    verifier_prompt_lens,
                    verifier_current_positions,
                )
            ]

        verifier_logits_list, max_verify_lens = _forward_verifiers(
            verifier_models=verifier_models,
            verifier_input_ids_list=verifier_input_ids_list,
            verifier_attention_mask_list=verifier_attention_mask_list,
            verifier_current_positions=verifier_current_positions,
            verifier_caches=verifier_caches,
            use_cache=use_cache,
        )

        need_ensemble = False
        num_accepted = gamma
        trigger_logits = None

        for draft_idx in range(gamma):
            draft_prefix = _decode_prefix(
                draft_tokenizer,
                input_ids,
                prompt_len,
                current_position + draft_idx,
            )
            draft_token_id = input_ids[0, current_position + draft_idx].item()
            draft_probabilities = torch.softmax(draft_logits[draft_idx], dim=-1)
            draft_prob = draft_probabilities[0, draft_token_id].item()

            matched_logits = []
            all_aligned = True
            any_mismatch = False
            probability_sum = draft_prob

            for verifier_logits, verifier_input_ids, verifier_tokenizer, verifier_prompt_len, verifier_current_position, max_verify_len in zip(
                verifier_logits_list,
                verifier_input_ids_list,
                verifier_tokenizers,
                verifier_prompt_lens,
                verifier_current_positions,
                max_verify_lens,
            ):
                matched_position = None
                for verifier_pos in range(max_verify_len):
                    verifier_prefix = _decode_prefix(
                        verifier_tokenizer,
                        verifier_input_ids,
                        verifier_prompt_len,
                        verifier_current_position + verifier_pos,
                    )
                    if verifier_prefix == draft_prefix:
                        matched_position = verifier_pos
                        break
                    if len(verifier_prefix) > len(draft_prefix):
                        break

                if matched_position is None:
                    all_aligned = False
                    break

                verifier_logit = verifier_logits[:, matched_position]
                token_id = verifier_input_ids[0, verifier_current_position + matched_position].item()
                probability_sum += torch.softmax(verifier_logit, dim=-1)[0, token_id].item()
                if torch.argmax(verifier_logit, dim=-1).item() != token_id:
                    any_mismatch = True
                matched_logits.append(verifier_logit)

            if not all_aligned:
                continue

            if any_mismatch and draft_prob < draft_prob_threshold and probability_sum < threshold:
                need_ensemble = True
                num_accepted = draft_idx
                trigger_logits = matched_logits
                break

        new_token_len = 0
        if need_ensemble:
            drafts_accepted += num_accepted
            if num_accepted < gamma:
                next_token_ids_by_model = get_ensemble_token_multi(
                    outputs=[
                        torch.softmax(draft_logits[num_accepted], dim=-1),
                        *[torch.softmax(logit, dim=-1) for logit in trigger_logits],
                    ],
                    tokenizers=[draft_tokenizer, *verifier_tokenizers],
                    sharpen_type=sharpen_type,
                    mapping_matrices=mapping_matrices,
                    vocab_union=vocab_union,
                    index_to_vocab=index_to_vocab,
                    special_prefix_tokens_dict=special_prefix_tokens_dict,
                    byte_mappings_list=byte_mappings_list,
                    top_k=top_k,
                )
                new_token_len = len(next_token_ids_by_model[0][0])
                next_token_ids = torch.tensor(
                    next_token_ids_by_model[0][0],
                    device=input_ids.device,
                ).unsqueeze(0)
                input_ids = torch.cat(
                    [input_ids[:, :current_position + num_accepted], next_token_ids],
                    dim=-1,
                )
                num_ensemble += 1
        else:
            drafts_accepted += gamma

        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[:, 1].min().item()
            del verifier_caches, draft_cache
            gc.collect()
            torch.cuda.empty_cache()
            return (
                input_ids[:, prompt_len:current_position + stop_location + 1].tolist(),
                drafts_accepted / drafts_speculated,
                num_ensemble,
            )

        if use_cache:
            for verifier_idx, verifier_cache in enumerate(verifier_caches):
                num_pruned_kv = min(
                    verifier_input_ids_list[verifier_idx].shape[-1] - verifier_prompt_lens[verifier_idx],
                    verifier_input_ids_list[verifier_idx].shape[-1] - verifier_current_positions[verifier_idx] + 5,
                )
                verifier_caches[verifier_idx] = _prune_cache_tokens(verifier_cache, num_pruned_kv)

        if num_accepted != gamma:
            draft_cache, attention_mask = _rollback_draft_state(
                draft_cache=draft_cache,
                input_ids=input_ids,
                attention_mask=attention_mask,
                current_position=current_position,
                num_accepted=num_accepted,
            )
            num_accepted += new_token_len

        current_position += num_accepted

    del verifier_caches, draft_cache
    gc.collect()
    torch.cuda.empty_cache()

    return input_ids[:, prompt_len:].tolist(), drafts_accepted / drafts_speculated, num_ensemble
