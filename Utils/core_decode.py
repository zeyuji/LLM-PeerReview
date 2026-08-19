from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers.cache_utils import DynamicCache, HybridCache

from Utils.core_token_map import safe_convert_ids_to_tokens
from Utils.util import clean_generation


EPS = 1e-12
DEFAULT_MAX_PROMPT_LENGTH = 4096


@dataclass
class CoreModelBundle:
    name: str
    model: Any
    tokenizer: Any
    device: torch.device


@dataclass
class _ModelState:
    bundle: CoreModelBundle
    sync_prompt_text: str
    prompt_ids: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pending_input_ids: torch.Tensor
    past_key_values: Any = None


@dataclass
class CoreDecodeResult:
    generation: str
    num_generated_tokens: int
    debug_steps: List[Dict[str, object]]


def _cache_length(cache: Any) -> int:
    if cache is None:
        return 0
    if isinstance(cache, tuple):
        return int(cache[0][0].shape[-2])
    if isinstance(cache, DynamicCache):
        return int(getattr(cache, "_seen_tokens", cache.key_cache[0].shape[-2]))
    if isinstance(cache, HybridCache):
        return int(cache.key_cache[0].shape[-2])
    raise ValueError(f"Unsupported cache type: {type(cache)}")


def _prune_cache(cache: Any, num_tokens_to_discard: int):
    if cache is None or num_tokens_to_discard <= 0:
        return cache
    if isinstance(cache, tuple):
        new_cache = []
        for layer_cache in cache:
            if layer_cache is None:
                new_cache.append(None)
                continue
            new_cache.append(tuple(tensor[:, :, :-num_tokens_to_discard, :] for tensor in layer_cache))
        return tuple(new_cache)
    if isinstance(cache, DynamicCache):
        for layer in range(len(cache)):
            cache.key_cache[layer] = cache.key_cache[layer][:, :, :-num_tokens_to_discard, :]
            cache.value_cache[layer] = cache.value_cache[layer][:, :, :-num_tokens_to_discard, :]
        cache._seen_tokens -= num_tokens_to_discard
        return cache
    if isinstance(cache, HybridCache):
        prune_start = cache.key_cache[0].shape[-2] - num_tokens_to_discard
        for layer in range(len(cache.key_cache)):
            cache.key_cache[layer][:, :, prune_start:, :] = 0.0
            cache.value_cache[layer][:, :, prune_start:, :] = 0.0
        return cache
    raise ValueError(f"Unsupported cache type: {type(cache)}")


def _count_common_prefix(old_ids: torch.Tensor, new_ids: torch.Tensor) -> int:
    old_list = old_ids[0].tolist()
    new_list = new_ids[0].tolist()
    common_len = 0
    for old_token_id, new_token_id in zip(old_list, new_list):
        if old_token_id != new_token_id:
            break
        common_len += 1
    return common_len


def _build_sparse_topk_prob_vector(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"CoRE v1 expects logits with shape [1, vocab], but got {tuple(logits.shape)}")

    top_k = min(int(k), int(logits.shape[-1]))
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    probs = torch.softmax(values, dim=-1)

    dense_probs = torch.zeros_like(logits, dtype=torch.float32)
    dense_probs.scatter_(1, indices, probs)

    batch_indices = torch.zeros_like(indices)
    sparse_indices = torch.stack([batch_indices.reshape(-1), indices.reshape(-1)], dim=0)
    sparse_values = probs.reshape(-1)
    sparse_probs = torch.sparse_coo_tensor(
        indices=sparse_indices,
        values=sparse_values,
        size=logits.shape,
        device=logits.device,
    ).coalesce()
    return dense_probs, sparse_probs


def row_entropy(x: torch.Tensor) -> torch.Tensor:
    row_sum = x.sum(dim=1, keepdim=True).clamp_min(EPS)
    p = (x / row_sum).clamp_min(EPS)
    log_p = torch.log(p)
    return -(p * log_p).sum(dim=1, keepdim=True)


def compute_consistency_scores(prob_list: Sequence[torch.Tensor], method: str = "consist-rbf") -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.cat(prob_list, dim=0)
    p_star = probs.mean(dim=0, keepdim=True)
    diff = torch.abs(probs - p_star)
    p_mask = ((probs + probs[0].unsqueeze(0)) > EPS).float()

    if method == "consist-linear":
        token_scores = p_mask * (1 - diff)
    elif method == "consist-rbf":
        beta = 2.0
        base = torch.tensor(beta, device=probs.device, dtype=probs.dtype)
        token_scores = p_mask * (torch.exp(-base * diff) - torch.exp(-base)) / (1 - torch.exp(-base))
    elif method == "consist-power":
        token_scores = p_mask * (1 - diff).clamp(min=0) ** 5.0
    elif method == "consist-rec":
        tmp = 1 / (1 + diff)
        token_scores = p_mask * (tmp - tmp.min()) / (tmp.max() - tmp.min() + EPS)
    else:
        raise ValueError(f"Unsupported CoRE consistency method: {method}")

    nonzero_count = (token_scores != 0).sum(dim=1, keepdim=True).float()
    token_scores_sum = token_scores.sum(dim=1, keepdim=True)
    token_scores = token_scores / token_scores_sum.clamp_min(EPS) * nonzero_count

    model_scores = token_scores_sum / (row_entropy(probs) + EPS)
    model_scores[0] = model_scores[0].clamp_min(model_scores.sum() - model_scores[0])
    model_weights = (model_scores / model_scores.sum().clamp_min(EPS)).squeeze(-1)
    return token_scores, model_weights


def combine_probabilities(prob_list: Sequence[torch.Tensor], variant: str) -> Tuple[torch.Tensor, List[float], Optional[torch.Tensor]]:
    if len(prob_list) == 0:
        raise ValueError("prob_list must not be empty")

    if variant == "vanilla":
        weights = torch.ones(len(prob_list), dtype=prob_list[0].dtype, device=prob_list[0].device)
        weights = weights / weights.sum()
        ensemble_probs = prob_list[0] * weights[0]
        for idx in range(1, len(prob_list)):
            ensemble_probs = ensemble_probs + prob_list[idx] * weights[idx]
        return ensemble_probs, [float(weight) for weight in weights.tolist()], None

    token_scores, weights = compute_consistency_scores(prob_list=prob_list, method=variant)
    ensemble_probs = prob_list[0] * weights[0]
    for idx in range(1, len(prob_list)):
        ensemble_probs = ensemble_probs + prob_list[idx] * weights[idx] * token_scores[idx].unsqueeze(0)
    return ensemble_probs, [float(weight) for weight in weights.tolist()], token_scores


def _build_stop_token_ids(tokenizer) -> List[int]:
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    if tokenizer.pad_token_id is not None:
        stop_ids.append(int(tokenizer.pad_token_id))

    special_tokens = ["<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "<|end|>", "</s>"]
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue
        if unk_token_id is not None and token_id == unk_token_id:
            continue
        stop_ids.append(int(token_id))
    return list(dict.fromkeys(stop_ids))


def _token_text(tokenizer, token_id: int) -> str:
    token = safe_convert_ids_to_tokens(tokenizer, token_id, skip_special_tokens=False)
    return token or ""


def _should_fallback_to_main(main_token: str, ensemble_token: str) -> bool:
    if not main_token or not ensemble_token:
        return False
    return ensemble_token.startswith(main_token) or main_token.startswith(ensemble_token)


def _prepare_prompt_inputs(bundle: CoreModelBundle, prompt: str, max_prompt_length: int) -> torch.Tensor:
    encoded = bundle.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    return encoded["input_ids"].to(bundle.device)


def _initialize_state(bundle: CoreModelBundle, prompt: str, max_prompt_length: int) -> _ModelState:
    prompt_ids = _prepare_prompt_inputs(bundle=bundle, prompt=prompt, max_prompt_length=max_prompt_length)
    attention_mask = torch.ones_like(prompt_ids, device=bundle.device)
    return _ModelState(
        bundle=bundle,
        # Decode the truncated prompt ids once so later re-tokenization stays aligned
        # with the actual cached prefix instead of the original raw prompt.
        sync_prompt_text=bundle.tokenizer.decode(
            prompt_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        prompt_ids=prompt_ids,
        input_ids=prompt_ids.clone(),
        attention_mask=attention_mask,
        pending_input_ids=prompt_ids.clone(),
    )


def _model_forward_next_logits(state: _ModelState) -> torch.Tensor:
    cache_length = _cache_length(state.past_key_values)
    pending_input_ids = state.pending_input_ids
    if pending_input_ids is None or pending_input_ids.shape[-1] == 0:
        raise ValueError(f"Model state for {state.bundle.name} has no pending tokens to evaluate.")

    model_kwargs = {
        "attention_mask": state.attention_mask,
        "past_key_values": state.past_key_values,
        "use_cache": True,
    }
    if cache_length > 0:
        model_kwargs["cache_position"] = torch.arange(
            cache_length,
            cache_length + pending_input_ids.shape[-1],
            device=state.bundle.device,
        )
    else:
        model_kwargs["cache_position"] = torch.arange(
            0,
            pending_input_ids.shape[-1],
            device=state.bundle.device,
        )

    try:
        model_inputs = state.bundle.model.prepare_inputs_for_generation(
            pending_input_ids,
            **model_kwargs,
        )
    except TypeError:
        model_kwargs.pop("cache_position", None)
        model_inputs = state.bundle.model.prepare_inputs_for_generation(
            pending_input_ids,
            **model_kwargs,
        )

    with torch.inference_mode():
        outputs = state.bundle.model(**model_inputs, return_dict=True)
    state.past_key_values = outputs.past_key_values
    state.pending_input_ids = None
    return outputs.logits[:, -1, :].to(dtype=torch.float32, device=state.bundle.device)


def _advance_main_state(state: _ModelState, selected_token_id: int) -> None:
    next_token = torch.tensor([[selected_token_id]], dtype=torch.long, device=state.bundle.device)
    state.input_ids = torch.cat([state.input_ids, next_token], dim=-1)
    state.attention_mask = torch.cat(
        [state.attention_mask, torch.ones((1, 1), dtype=state.attention_mask.dtype, device=state.bundle.device)],
        dim=-1,
    )
    state.pending_input_ids = next_token


def _sync_state_to_common_text(state: _ModelState, generated_text: str) -> None:
    full_input_ids = state.bundle.tokenizer(
        state.sync_prompt_text + generated_text,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"].to(state.bundle.device)
    new_input_ids = full_input_ids
    old_input_ids = state.input_ids
    common_prefix = _count_common_prefix(old_ids=old_input_ids, new_ids=new_input_ids)

    old_length = int(old_input_ids.shape[-1])
    if state.past_key_values is not None and common_prefix < old_length:
        state.past_key_values = _prune_cache(state.past_key_values, old_length - common_prefix)

    state.input_ids = new_input_ids
    state.attention_mask = torch.ones_like(new_input_ids, device=state.bundle.device)
    if state.past_key_values is None:
        state.pending_input_ids = new_input_ids
    else:
        state.pending_input_ids = new_input_ids[:, common_prefix:]
        if state.pending_input_ids.shape[-1] == 0:
            state.past_key_values = _prune_cache(state.past_key_values, 1)
            state.pending_input_ids = new_input_ids[:, -1:]


def run_core_decode(
    prompt: str,
    main_bundle: CoreModelBundle,
    assist_bundles: Sequence[CoreModelBundle],
    assist_to_main_maps: Sequence[torch.Tensor],
    variant: str,
    top_k: int,
    max_new_tokens: int,
    debug: bool = False,
    max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH,
) -> CoreDecodeResult:
    if len(assist_bundles) != len(assist_to_main_maps):
        raise ValueError("assist_bundles and assist_to_main_maps must have the same length.")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive for CoRE decoding, but got {top_k}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive for CoRE decoding, but got {max_new_tokens}")

    main_state = _initialize_state(bundle=main_bundle, prompt=prompt, max_prompt_length=max_prompt_length)
    assist_states = [
        _initialize_state(bundle=bundle, prompt=prompt, max_prompt_length=max_prompt_length)
        for bundle in assist_bundles
    ]
    stop_token_ids = set(_build_stop_token_ids(main_bundle.tokenizer))

    generated_token_ids: List[int] = []
    debug_steps: List[Dict[str, object]] = []
    assist_maps = [assist_to_main_map.to(bundle.device) for assist_to_main_map, bundle in zip(assist_to_main_maps, assist_bundles)]

    for step_idx in range(int(max_new_tokens)):
        main_logits = _model_forward_next_logits(main_state)
        main_greedy_id = int(torch.argmax(main_logits, dim=-1).item())
        if main_greedy_id in stop_token_ids:
            break

        main_probs, _ = _build_sparse_topk_prob_vector(logits=main_logits, k=top_k)
        prob_list = [main_probs.to(main_bundle.device)]

        assist_mapped_probs = []
        for assist_state, assist_to_main_map in zip(assist_states, assist_maps):
            current_logits = _model_forward_next_logits(assist_state)
            _, assist_sparse_probs = _build_sparse_topk_prob_vector(logits=current_logits, k=top_k)
            if assist_sparse_probs.shape[-1] != assist_to_main_map.shape[0]:
                raise ValueError(
                    "CoRE token-map shape mismatch before sparse matmul: "
                    f"assist_probs_shape={tuple(assist_sparse_probs.shape)}, "
                    f"assist_to_main_map_shape={tuple(assist_to_main_map.shape)}, "
                    f"assist_model={assist_state.bundle.name}, "
                    f"main_model={main_bundle.name}"
                )
            mapped = torch.sparse.mm(
                assist_sparse_probs,
                assist_to_main_map,
            )
            mapped_probs = mapped.to_dense() if mapped.is_sparse else mapped
            assist_mapped_probs.append(mapped_probs.to(main_bundle.device, dtype=torch.float32))
            prob_list.append(assist_mapped_probs[-1])

        ensemble_probs, model_weights, token_scores = combine_probabilities(prob_list=prob_list, variant=variant)
        ensemble_token_id = int(torch.argmax(ensemble_probs, dim=-1).item())
        ensemble_token = _token_text(main_bundle.tokenizer, ensemble_token_id)
        main_token = _token_text(main_bundle.tokenizer, main_greedy_id)

        selected_token_id = ensemble_token_id
        selected_token = ensemble_token
        if _should_fallback_to_main(main_token=main_token, ensemble_token=ensemble_token):
            selected_token_id = main_greedy_id
            selected_token = main_token

        if selected_token_id in stop_token_ids:
            break

        generated_token_ids.append(selected_token_id)
        _advance_main_state(state=main_state, selected_token_id=selected_token_id)

        current_generated_text = main_bundle.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for assist_state in assist_states:
            _sync_state_to_common_text(
                state=assist_state,
                generated_text=current_generated_text,
            )

        if debug:
            step_payload = {
                "step": step_idx,
                "selected_token_id": selected_token_id,
                "selected_token": selected_token,
                "main_token_id": main_greedy_id,
                "main_token": main_token,
                "model_weights": model_weights,
            }
            if token_scores is not None:
                selected_scores = []
                for score_row in token_scores:
                    selected_scores.append(float(score_row[selected_token_id].item()))
                step_payload["selected_token_consistency"] = selected_scores
            debug_steps.append(step_payload)

    generation = clean_generation(
        main_bundle.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    return CoreDecodeResult(
        generation=generation,
        num_generated_tokens=len(generated_token_ids),
        debug_steps=debug_steps,
    )
