import copy
import gc
from typing import List, Tuple

import torch
from torch.nn import Module

from caching import prune_cache


def get_top_k_tokens(logits, tokenizer, k=10, internlm=False):
    top_k_indices = torch.topk(logits, k).indices
    logits_list = logits.tolist()

    top_k_values = []
    for idx, val_row in zip(top_k_indices, logits_list):
        val_item = []
        for i in idx:
            val_item.append(val_row[i])
        top_k_values.append(val_item)

    v1 = []
    for val, idx in zip(top_k_values, top_k_indices):
        # Specific prefix handling for InternLM models
        if internlm:
            v1.append(
                {tokenizer.decode([27960, i])[1:]: [v, int(i)] for v, i in zip(val, idx)})
        else:
            v1.append(
                {tokenizer.decode(i): [v, int(i)] for v, i in zip(val, idx)})

    return v1


def get_union_vocab(v1, v2):
    # Extract unique tokens from both dictionaries
    unique_tokens = []
    for v1_tokens, v2_tokens in zip(v1, v2):
        unique_tokens.append(list(set(v1_tokens.keys()) | set(v2_tokens.keys())))

    return unique_tokens


def update_vocab(v1, vu, tokenizer, logits):
    for vu_token, v1_token, logit_ele in zip(vu, v1, logits):
        blank_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        v1_token_ids = []
        for item in v1_token.values():
            v1_token_ids.append(item[1])
        for token in vu_token:
            if token not in v1_token.keys():
                if token != '':
                    subtokens = tokenizer.tokenize(token)
                    subtoken_id = tokenizer.convert_tokens_to_ids(subtokens)
                    # Use the first subtoken ID if the string splits into multiple tokens
                    if subtoken_id and len(subtoken_id) == 1:
                        subtoken_id = subtoken_id[0]
                        logit = logit_ele[subtoken_id]
                    else:
                        for token_id in subtoken_id:
                            if token_id != blank_id:
                                subtoken_id = token_id
                                logit = logit_ele[subtoken_id]
                                break
                else:
                    # Fallback to whitespace token if no ID is found
                    subtoken_id = blank_id
                    logit = logit_ele[subtoken_id]

                v1_token[token] = [logit, subtoken_id]

    v1_new = vocab_softmax(v1)
    return v1_new


def vocab_softmax(v1):
    v1_new = []
    for element in v1:
        ele = {}
        ele_values = list(element.values())
        ele_probs, ele_ids = [], []
        for item in ele_values:
            ele_probs.append(item[0])
            ele_ids.append(item[1])
        ele_probs = torch.softmax(torch.tensor(ele_probs), dim=0)
        for token, prob, ids in zip(element.keys(), ele_probs, ele_ids):
            ele[token] = [prob, ids]
        v1_new.append(ele)

    return v1_new


def average_and_sample(v1, v1_orig, v2, lamda, sharpen_type):
    next_token, v_avg, next_token_id1, next_token_id2 = [], [], [], []
    for element_v1, element_v2 in zip(v1, v2):
        assert len(element_v1) == len(element_v2)
        v_new = {}
        # Linear ensemble of probabilities
        for token1 in element_v1:
            v_new[token1] = [lamda * element_v1[token1][0] + (1 - lamda) * element_v2[token1][0], element_v1[token1][1]]
        v_avg.append(v_new)
        
        # Apply sharpening when the distribution is overly smooth (Max prob < 0.5)
        max_prob = max(v[0].item() for k, v in v_new.items())
        if max_prob < 0.5:
            if sharpen_type == "geom": # Geometric mean sharpening
                v_new = {}       
                v_avg = []     
                for token1 in element_v1:
                    v_new[token1] = [torch.sqrt(element_v1[token1][0] * element_v2[token1][0]), element_v1[token1][1]]
                v_avg.append(v_new)
            else: # Heuristic sharpening: aggregate probabilities of related prefix tokens
                filtered_dict = copy.deepcopy(v_new)
                filtered_keys = [k for k, v in filtered_dict.items() if v[0].item() > 0.1 and k in v1_orig]
                for top_k in filtered_keys:
                    if top_k == " ":
                        continue
                    for _key in v_new:
                        if _key != top_k and _key.startswith(top_k):
                            v_new[top_k][0] = v_new[top_k][0] + filtered_dict[_key][0].item()
            
        probs = []
        for item in v_new.values():
            probs.append(item[0])

        sample_index = probs.index(max(probs))

        i = 0
        for item1 in v_new.keys():
            if i == sample_index:
                next_token.append(item1)
                next_token_id1.extend([element_v1[item1][1]])
                next_token_id2.append(element_v2[item1][1])
            i+=1

    return next_token, v_avg, next_token_id1, next_token_id2


@torch.no_grad()
def safe_generate_unite(
    inputs,
    ver_inputs,
    draft_model: Module,
    ver_model: Module,
    max_length=4096,
    draft_tokenizer=None,
    ver_tokenizer=None,
    gamma: int = 5,
    draft=None,
    verifier=None,
    use_cache: bool = False,
    sharpen_type="geom",
    **kwargs,
) -> Tuple[List[int], float]:
    """
    Args:
        inputs: drafter input sequence of batch size 1.
        ver_inputs: verifier input sequence of batch size 1.
        draft_model (Module): drafter model.
        ver_model (Module): verifier model.
        tokenizer: tokenizer.
        gamma (int): number of drafts generated by the drafter at each step.
        use_cache (bool): whether to use cache.
        sharpen_type: sharpening overly smooth ensemble distribution, "geom" | "heur"
    
    Returns:
        output: generated sequence.
        accept_rate: acceptance rate (number of accepted drafts divided by the number of total drafts).
        num_tokens: lengthe of the generated sequence
        num_ensemble: number of ensemble operations
    """
    draft_cache, ver_cache = None, None
    
    # Define eos tokens and end conditions  
    eos_token_list = [t.eos_token for t in [draft_tokenizer, ver_tokenizer]]
    eos_token_list.extend(["<|end_of_text|>", "<|endoftext|>", "<|im_end|>", "<|end|>", "</s>"])
    eos_token_id = [draft_tokenizer.eos_token_id]
    if draft == "internlm":
        eos_token_id.append(128131)
    stop_tokens = torch.tensor(eos_token_id, dtype=torch.long, device=ver_model.device).unsqueeze(1)

    drafts_accepted, drafts_speculated = .0, .0
    num_ensemble = 0
    
    # Force greedy decoding configuration
    draft_model.generation_config.do_sample = False
    draft_model.generation_config.temperature = 0.
    ver_model.generation_config.do_sample = False
    ver_model.generation_config.temperature = 0.
    
    input_ids = inputs["input_ids"]
    ver_input_ids = ver_inputs["input_ids"]
    
    attention_mask = inputs["attention_mask"]
    ver_attention_mask = ver_inputs["attention_mask"]
    
    prompt_len = input_ids.shape[-1]
    total_len = prompt_len + max_length
    ver_prompt_len = ver_input_ids.shape[-1]

    current_position = prompt_len
    ver_current_position = ver_prompt_len

    while current_position < total_len:                
        input_ids = input_ids.to(draft_model.device)
        draft_logits = []
        
        # 1. Generate
        for k in range(gamma):
            if draft_cache is None:
                cache_position = torch.arange(0, input_ids.shape[-1]).to(draft_model.device)
            else:
                cache_position = torch.arange(draft_cache[0][0].shape[-2], input_ids.shape[-1]).to(draft_model.device)
                    
            drafter_model_kw = {
                "attention_mask": attention_mask,
                "past_key_values": draft_cache,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
            drafter_inputs = draft_model.prepare_inputs_for_generation(input_ids, **drafter_model_kw)
            Md = draft_model(
                **drafter_inputs,
                return_dict=True
            )
            draft_cache = Md.past_key_values
            curr_draft_logits = Md.logits[..., -1, :].to(copy=True, dtype=torch.float32, device=draft_model.device)
            xi = torch.argmax(curr_draft_logits, dim=-1)
            input_ids = torch.cat([input_ids, xi[:, None]], dim=-1)
            draft_logits.append(curr_draft_logits)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        drafts_speculated += gamma
        input_ids = input_ids.to(ver_model.device)

        # Prepare inputs for verifier
        # Retokenization: Map drafter tokens to verifier vocabulary
        retokenized_draft_ids = ver_tokenizer(draft_tokenizer.batch_decode(input_ids[:, prompt_len:]), return_tensors="pt", add_special_tokens=False).to(ver_model.device)["input_ids"]
        
        # Concat with input prompt
        ver_input_ids = torch.cat([ver_input_ids[:, :ver_prompt_len], retokenized_draft_ids], dim=-1)
        ver_input_ids = ver_input_ids.to(torch.int64)
        
        # Align verifier cache position
        if current_position > prompt_len:
            prev_draft_seq = draft_tokenizer.batch_decode(input_ids[:, prompt_len:current_position])[0]
            k = 0
            if len(prev_draft_seq) > len(ver_tokenizer.batch_decode(ver_input_ids[:, ver_prompt_len:])[0]):
                ver_current_position = ver_input_ids.shape[-1]
            else:
                while len(prev_draft_seq) > len(ver_tokenizer.batch_decode(ver_input_ids[:, ver_prompt_len:ver_current_position - gamma + k])[0]):
                    k += 1
                ver_current_position = ver_current_position-gamma+k
        
        # 2. Verify
        ver_attention_mask = torch.ones_like(ver_input_ids)
        if ver_cache is not None:
            ver_cache_position = torch.arange(ver_cache[0][0].shape[-2], ver_input_ids.shape[-1]).to(ver_model.device)
        else:
            ver_cache_position = None
        ver_model_kw = {
            "attention_mask": ver_attention_mask,
            "past_key_values": ver_cache,
            "use_cache": use_cache,
            "cache_position": ver_cache_position,
        }

        ver_inputs = ver_model.prepare_inputs_for_generation(ver_input_ids, **ver_model_kw)
        Mv = ver_model(
            **ver_inputs,
            return_dict=True,
        )

        ver_cache = Mv.past_key_values
        
        # Number of tokens to verify
        num_verify = ver_input_ids.shape[1] - ver_current_position
        max_verify_len = ver_input_ids.shape[1] - ver_current_position
        
        ver_logits = Mv.logits[...,  -max_verify_len-1:-1, :].to(copy=True, dtype=torch.float32, device=ver_model.device) # [1, gamma, vocab_size]
        
        for i in range(max_verify_len):
            ver_logit = ver_logits[:, i]
            max_ver_id = torch.argmax(ver_logit, dim=-1).item()
            ver_i_seq = ver_tokenizer.batch_decode(ver_input_ids[:, ver_prompt_len:ver_current_position+i])[0].lstrip(" ")
            for j in range(gamma):
                # OOV-like verification
                draft_j_seq = draft_tokenizer.batch_decode(input_ids[:, prompt_len:current_position+j])[0].lstrip(" ")
                if draft_j_seq == ver_i_seq:
                    # Not OOV-like tokens
                    draft_prob = torch.softmax(draft_logits[j], dim=-1)[0, input_ids[0, current_position + j]].item()
                    ver_prob = torch.softmax(ver_logit, dim=-1)[0, ver_input_ids[0, ver_current_position + i]].item()
                    # Ensemble distribution verification
                    if (draft_prob + ver_prob) < 1.0 and max_ver_id != ver_input_ids[0, ver_current_position + i].item():
                        num_accepted = j # drafter token position
                        num_verify = i # verifier token position
                        
                    break
                elif len(draft_j_seq) > len(ver_i_seq):
                    break
            if num_verify < max_verify_len:
                break

        # 3. Ensemble (UniTE)
        is_rejected = (num_verify < max_verify_len)
        if is_rejected:
            drafts_accepted += num_accepted
            if num_accepted < gamma:
                v1 = get_top_k_tokens(draft_logits[num_accepted], draft_tokenizer, 10, internlm=(draft == "internlm"))
                v2 = get_top_k_tokens(ver_logits[:, num_verify], ver_tokenizer, 10, internlm=(verifier == "internlm"))
                
                v1_orig_keys = copy.deepcopy(list(v1[0].keys()))

                vu = get_union_vocab(v1, v2)
                v1_new = update_vocab(v1, vu, draft_tokenizer, draft_logits[num_accepted])
                v2_new = update_vocab(v2, vu, ver_tokenizer, ver_logits[:, num_verify])
                
                next_token, v_avg, next_token_id1, next_token_id2 = average_and_sample(v1_new, v1_orig_keys, v2_new, 0.5, sharpen_type)

                if ver_tokenizer.decode(next_token_id2[0]) in eos_token_list:
                    next_token_id1 = [eos_token_id[-1]]
                new_token_len = len(next_token_id1)
                next_token_id1 = torch.tensor(next_token_id1, device=input_ids.device).unsqueeze(0)
                input_ids = torch.cat([input_ids[:, :current_position+num_accepted], next_token_id1], dim=-1)
                num_ensemble += 1

        else:
            # Every tokens are accepted
            drafts_accepted += gamma
            num_accepted = gamma
        
        # Check for EOS to exit early
        """
        if (input_ids[..., current_position:] == eos_token_id).any():
            # Find exact position only if EOS exists
            stop_locations = torch.nonzero((input_ids[..., current_position:] == eos_token_id))
        """
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:], stop_tokens))
        if stop_locations.shape[0] > 0:
            #stop_location = stop_locations[0, 1].item()
            stop_location = stop_locations[:, 1].min().item()
            del ver_cache, draft_cache, Md, Mv
            gc.collect()
            torch.cuda.empty_cache()
            return input_ids[:, prompt_len:current_position + stop_location + 1].tolist(), drafts_accepted / drafts_speculated, num_ensemble

        # KV cache implementation
        if use_cache:
            # Prune ver_cache
            num_pruned_kv = min(ver_input_ids.shape[-1] - ver_prompt_len, ver_input_ids.shape[-1] - ver_current_position + 5)
            ver_cache = prune_cache(ver_cache, num_pruned_kv)
            if num_accepted != gamma:
                # Prune draft_cache if tokens were rejected
                draft_cache = prune_cache(draft_cache, gamma - num_accepted)
        
        if num_accepted != gamma:
            num_accepted += new_token_len
            attention_mask = attention_mask[:, :current_position + num_accepted]
        
        current_position += num_accepted
        del Md, Mv
    
    del ver_cache, draft_cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return input_ids[:, prompt_len:].tolist(), drafts_accepted / drafts_speculated, num_ensemble