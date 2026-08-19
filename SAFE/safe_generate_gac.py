import torch
from torch.nn import Module
from transformers.cache_utils import DynamicCache
from caching import prune_cache
from typing import List, Tuple
import gc
from tqdm import tqdm
from logger import setup_custom_logger

logger = setup_custom_logger("TSP")


def check_byte_mappings(tokenizer):
    """
    Args:
    - tokenizer: An object representing a tokenizer. This tokenizer object must have a method
                 `get_vocab()` that returns a dictionary mapping tokens to their respective
                 token IDs within the tokenizer's vocabulary.

    Returns:
    - If the tokenizer is identified as BBPE based on prefix counts, returns a dictionary for byte values from '<0x00>' to '<0x7F>'.
    - Otherwise, returns a byte_mapping (dict): A dictionary where each key is a string representing a byte value in
                           standard hex format (e.g., '<0x00>', '<0x01>', ..., '<0xFF>'), and each
                           value is the corresponding token ID for that byte representation
                           within the tokenizer's vocabulary.
    """
    vocab = tokenizer.get_vocab()
    g_prefix_count = sum(token.startswith("Ġ") for token in vocab)
    u_prefix_count = sum(token.startswith("▁") for token in vocab)

    byte_mapping = {}

    # For BBPE, handle bytes from 0x00 to 0x7F
    if g_prefix_count > u_prefix_count:
        for byte_val in range(128):  # Limit to 0x00 to 0x7F
            byte_char = chr(byte_val)
            token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(byte_char))[0]
            hex_token = f"<0x{byte_val:02X}>"
            byte_mapping[hex_token] = token_id
    else:
        # For non-BBPE, attempt to find a direct mapping in vocab
        for byte_val in range(256):
            hex_token = f"<0x{byte_val:02X}>"
            # For cases like "\t" being replaced in vocab
            if hex_token == "<0x09>" and hex_token not in vocab:
                continue
            if hex_token not in vocab:
                raise ValueError(
                    f"Token {hex_token} not found in tokenizer's vocabulary."
                )
            byte_mapping[hex_token] = vocab[hex_token]

    return byte_mapping


def get_vocab_union_and_mapping(tokenizers):
    """
    Modified function that creates a union of tokens from the vocabularies of given tokenizers and
    provides a mapping for each tokenizer from its token IDs to the tokens in the unified vocabulary.
    It handles tokens starting with 'Ġ' or '▁' differently to merge similar tokens.

    Args:
    tokenizers (list): A list of tokenizer objects, each with a 'get_vocab()' method that
                       returns a dictionary of tokens and their corresponding IDs in the tokenizer's
                       vocabulary.

    Returns:
    tuple: A tuple containing three elements:
        - vocab_union (set): A set containing the union of all tokens in the vocabularies of the
                             provided tokenizers.
        - tokenizers_mapping (list): A list of dictionaries, where each dictionary corresponds to
                                     a tokenizer from the input list and maps token IDs from the
                                     tokenizer to tokens in the vocab_union.
        - index_to_vocab (dict): A dictionary mapping from unique index to tokens in the vocab_union.
        - byte_mappings_list (list): A list of dictionaries, where each dictionary corresponds to a
                                tokenizer from the input list and provides a mapping of byte value
                                tokens from '<0x00>' to '<0xFF>' to their original token IDs in the
                                tokenizer's vocabulary. This mapping is used to ensure consistency
                                and to facilitate the identification and replacement of these tokens
                                in the unified vocabulary.
    """
    # Initialize a set to store all tokens
    vocab_union = set()
    # Initialize a list to store the mappings for each tokenizer
    tokenizers_mapping = []
    byte_mappings_list = []

    # First, add '<0x00>' to '<0xFF>'
    for byte_val in range(256):
        vocab_union.add(f"<0x{byte_val:02X}>")

    # Process each tokenizer separately
    for tokenizer in tokenizers:
        vocab = tokenizer.get_vocab()
        token_set = set()
        mapping = {}

        # Check and record each tokenizer's mapping for '<0x00>' to '<0xFF>'
        byte_mapping = check_byte_mappings(tokenizer)
        byte_mappings_list.append(byte_mapping)

        if len(byte_mapping) == 128:
            logger.warning(
                "BBPE detected. Please be cautious in usage as currently it only supports applications such as multiple-choice questions eg.(A)"
            )

        # Remove the existing mappings for '<0x00>' to '<0xFF>'
        for hex_token, token_id in byte_mapping.items():
            # Remove tokens from the vocabulary whose token IDs appear in the byte_mapping
            actual_tokens = [token for token, id in vocab.items() if id == token_id]

            if len(actual_tokens) != 1:
                # Raise an error if more than one matching token is found
                raise ValueError(
                    f"Multiple tokens/ Zero token found for token ID {token_id} in tokenizer's vocabulary."
                )
            del vocab[actual_tokens[0]]

        # Detect usage of 'Ġ' and '▁'
        g_prefix_count = sum(token.startswith("Ġ") for token in vocab)
        u_prefix_count = sum(token.startswith("▁") for token in vocab)

        # Process tokens based on prefix type
        if g_prefix_count > u_prefix_count:
            # Handle tokens starting with 'Ġ'
            for token, token_id in vocab.items():
                processed_token = token.replace("Ġ", " ").replace("Ċ", "\n")
                token_set.add(processed_token)
                mapping[token_id] = processed_token
        else:
            # Handle tokens starting with '▁'
            for token, token_id in vocab.items():
                if token.startswith("▁"):
                    processed_token = token.replace("▁", " ")
                else:
                    # For tokens without '▁', use the decode method
                    processed_token = token  # tokenizer.decode([token_id])
                token_set.add(processed_token)
                mapping[token_id] = processed_token

        # Merge into the total vocab_union
        vocab_union = vocab_union.union(token_set)
        # Append the mapping for this tokenizer to the list
        tokenizers_mapping.append(mapping)

    # Generate a mapping for each token in the union to a unique index
    vocab_to_index = {token: i for i, token in enumerate(vocab_union)}

    # Convert vocab_to_index to index_to_vocab
    index_to_vocab = {index: token for token, index in vocab_to_index.items()}

    for tokenizer, byte_mapping, mapping in zip(
        tokenizers, byte_mappings_list, tokenizers_mapping
    ):
        # Update the mappings for each tokenizer to map to the index in the unified vocab
        for token_id, token in mapping.items():
            mapping[token_id] = vocab_to_index[token]

        # Define the extended mapping dictionary
        bbpe_mapping = {
            **{
                f"<0x{hex(i)[2:].upper()}>": chr(i) for i in range(0x30, 0x3A)
            },  # mapping '0' to '9'
            **{
                f"<0x{hex(i)[2:].upper()}>": chr(i) for i in range(0x41, 0x5B)
            },  # mapping 'A' to 'Z'
            **{
                f"<0x{hex(i)[2:].upper()}>": chr(i) for i in range(0x61, 0x7B)
            },  # mapping 'a' to 'z'
        }

        # Add the '<0x00>' to '<0xFF>' mappings for each tokenizer
        for hex_token, original_token_id in byte_mapping.items():
            # First, check the original conditions
            if (
                not all(len(bm) == 128 for bm in byte_mappings_list)
                and len(byte_mapping) == 128
            ):
                # Apply special handling to the specified characters
                if hex_token in bbpe_mapping:
                    logger.warning(
                        f"We force-mapped the BBPE {hex_token} to {bbpe_mapping[hex_token]} in union vocab"
                    )
                    mapping[original_token_id] = vocab_to_index[bbpe_mapping[hex_token]]
                    continue
            mapping[original_token_id] = vocab_to_index[hex_token]

    return vocab_union, tokenizers_mapping, index_to_vocab, byte_mappings_list


def create_mapping_matrix(mapping, union_vocab_size, model_vocab_size):
    """
    Creates a sparse tensor mapping matrix for vocabulary translation.
    
    Args:
    - mapping (dict): Maps model token IDs to unified vocabulary indexes.
    - union_vocab_size (int): Size of the unified vocabulary.
    - model_vocab_size (int): Size of the model's vocabulary.
    
    Returns:
    - torch.sparse_coo_tensor: Sparse tensor in COO format with shape [model_vocab_size, union_vocab_size].
                               Each non-zero element (i, j) indicates a mapping from the i-th token in the
                               model's vocabulary to the j-th token in the unified vocabulary.
    """

    if model_vocab_size == 151646:
        logger.warning(
            "The qwen1.5 series has been detected, where the length of tokenizer.get_vocab() and the vocab_size in the model config are inconsistent. We have forcefully set it to the latter. https://github.com/QwenLM/Qwen1.5/issues/29"
        )
        model_vocab_size = 151936

    indices = []  # Store the coordinates of non-zero elements
    values = []  # Non-zero values, typically 1 for a mapping matrix

    for model_token_id, unified_token_index in mapping.items():
        indices.append([model_token_id, unified_token_index])  # (rows, cols)
        values.append(1.0)

    # Convert to a tensor suitable for COO format
    indices = torch.tensor(
        indices, dtype=torch.long
    ).t()  # Transpose to meet (rows, cols)
    values = torch.tensor(values, dtype=torch.float)

    # Create a sparse tensor
    size = torch.Size([model_vocab_size, union_vocab_size])
    mapping_matrix = torch.sparse_coo_tensor(indices, values, size, device="cuda")

    return mapping_matrix


def find_special_underscore_token(tokenizer):
    """
    Identifies the shortest special token in the tokenizer's vocabulary that starts with '▁',
    which is neither part of any other token nor contains any other token (except '▁' itself).
    '▁' itself and tokens resulting in only whitespace after '▁' is removed are also excluded 
    from the result.
    
    Args:
        tokenizer: An instance of a tokenizer class with a 'get_vocab()' method, returning 
                   a dictionary of tokens and their IDs.

    Returns:
        str: The shortest special token meeting the criteria, with '▁' removed, sorted
             lexicographically to ensure consistency. Raises an error if no such token is found.

    The function first checks the prevalence of tokens starting with 'Ġ' and '▁'. If tokens
    starting with 'Ġ' are more prevalent, it returns an empty string. Otherwise, it proceeds
    to find the shortest token starting with '▁', which is not part of any other token and
    does not contain any other tokens (except for the initial '▁'), and is not just whitespace
    after '▁' is removed. It then removes '▁' from the token before returning it. If no such
    token is found, an error is raised.
    """

    # get tokenizer vocab
    vocab = tokenizer.get_vocab()

    # Count tokens that start with 'Ġ' and '▁'
    count_prefix_G = sum(1 for token in vocab if token.startswith("Ġ"))
    count_prefix_underscore = sum(1 for token in vocab if token.startswith("▁"))

    # Return an empty string if 'Ġ' tokens are more frequent
    if count_prefix_G > count_prefix_underscore:
        return ""

    # Filter tokens that start with '▁'
    underscore_tokens = [
        token for token in vocab if token.startswith("▁") and token != "▁"
    ]

    # Filter tokens that meet the criteria
    special_tokens = []
    for token in tqdm(underscore_tokens, desc="Analyzing tokens"):
        cleaned_token = token[1:]  # remove '▁'

        # Ensure the token is not part of another token, contains no additional tokens besides the first '▁',
        # has no multiple '▁', and is not a space after removing '▁'
        if (
            not any(
                token in other_token
                for other_token in underscore_tokens
                if other_token != token
            )
            and token.count("▁") == 1
            and cleaned_token.strip() != ""
        ):
            special_tokens.append(cleaned_token)

    # Raise an error if no token meets the criteria
    if not special_tokens:
        raise ValueError("No special underscore token found that meets the criteria.")

    # Return the shortest token to ensure consistency
    return min(special_tokens, key=lambda x: (len(x), x))


def get_special_prefix_tokens_for_all(tokenizers):
    """
    This function takes a list of tokenizers and returns a dictionary where each tokenizer is 
    associated with its special prefix token. It utilizes a hypothetical function find_special_underscore_token
    which is assumed to return the special prefix token that each individual tokenizer can handle.
    
    Args:
    tokenizers (list): A list of tokenizer objects. Each tokenizer is assumed to have a 
                       method or functionality that allows the extraction of its special prefix token.
    
    Returns:
    dict: A dictionary where each key is a tokenizer from the input list, and the corresponding 
          value is the special prefix token that the tokenizer can handle, as determined by calling 
          the find_special_underscore_token function.
          
    Example:
    tokenizers = [tokenizer1, tokenizer2, ...]
    special_prefix_tokens = get_special_prefix_tokens_for_all(tokenizers)
    print(special_prefix_tokens)  # Output: {tokenizer1: special_prefix_token1, tokenizer2: special_prefix_token2, ...}
    """

    # Initialize an empty dictionary to store the results
    special_prefix_tokens = {}

    # Iterate through the list of tokenizers
    for tokenizer in tokenizers:
        if tokenizer.vocab_size == 256000:
            logger.info("gemma-it detected, use '¢' as special_prefix_token")
            special_prefix_tokens[tokenizer] = "¢"
            continue
        # Get the special prefix token for each tokenizer
        token = find_special_underscore_token(tokenizer)
        # Store the tokenizer and its special prefix token in the dictionary
        special_prefix_tokens[tokenizer] = token
    return special_prefix_tokens


def setup_union_vocab(models, tokenizers):
    # Determine special prefix tokens for all tokenizers
    special_prefix_tokens_dict = get_special_prefix_tokens_for_all(tokenizers)
    vocab_union, tokenizers_mapping, index_to_vocab, byte_mappings_list = get_vocab_union_and_mapping(
        tokenizers
    )

    model_vocab_size_list = [
        model_actor.config.vocab_size for model_actor in models
    ]

    mapping_matrices = [
        create_mapping_matrix(mapping, len(vocab_union), vocab_size)
        for mapping, tokenizer, vocab_size in zip(tokenizers_mapping, tokenizers, model_vocab_size_list)
    ]
    
    return vocab_union, mapping_matrices, index_to_vocab, byte_mappings_list, special_prefix_tokens_dict


def get_token_ids(tokenizer, token, special_prefix_token, byte_mapping):
    """
    Tokenizes a given token and a special prefix token from the tokenizer's vocabulary, 
    then finds the token IDs for the portion of the given token that does not overlap 
    with the special prefix token. It is particularly useful for identifying unique sub-tokens 
    in tokenization processes. If initial tokenization does not meet expectations,
    it tries using ';' as an alternate special prefix token.

    Args:
    tokenizer: An instance of a tokenizer class with an 'encode' method that converts
               text to a list of token IDs.
    token (str): The token to be tokenized and analyzed.
    special_prefix_token (str): A special prefix token from the tokenizer's vocabulary, used as a 
                                reference point for comparison. It is the shortest token starting with 
                                a specific prefix ('▁' in most cases), which is neither part of any 
                                other token nor contains any other token.
    byte_mapping (dict): A dictionary mapping standard byte representations ('<0x00>' to '<0xFF>')
                         to their token IDs in the tokenizer's vocabulary.

    Returns:
    list: A list of token IDs representing the non-overlapping part of the 'token'
          when tokenized, compared to the tokenization of 'special_prefix_token'.

    The function tries using the provided special_prefix_token, and if tokenization doesn't match as expected,
    it attempts using ';' as an alternate special_prefix_token. If it still doesn't match, it returns
    the token IDs for 'token'.
    """

    # Check if the token is a standard byte representation and return its token ID if found
    if token in byte_mapping:
        return [byte_mapping[token]]

    if byte_mapping != 128:
        prefix_tokens = [special_prefix_token, ";"]

        for prefix_token in prefix_tokens:
            # Tokenize individually
            token_id_list1 = tokenizer.encode(prefix_token, add_special_tokens=False)

            # Tokenize doubled token
            token_id_list2 = tokenizer.encode(
                prefix_token + token, add_special_tokens=False
            )

            # Check if the start of token_id_list2 matches token_id_list1
            if token_id_list2[: len(token_id_list1)] == token_id_list1:
                result = token_id_list2[len(token_id_list1) :]
                if result:
                    return result

        # If tokenization doesn't match as expected with any prefix token, return the token IDs for 'token'
        logger.warning(f"Warning: Token '{token}' may not be tokenized as expected.")
    return tokenizer.encode(token, add_special_tokens=False)


def get_ensemble_token(
    outputs,
    tokenizers,
    sharpen_type,
    mapping_matrices,
    vocab_union,
    index_to_vocab,
    special_prefix_token,
    byte_mappings_list,
):
    eos_token_list = [tokenizer.eos_token for tokenizer in tokenizers]
    eos_token_list.extend(["<|end_of_text|>", "<|endoftext|>", "<|im_end|>", "<|end|>"])

    # Initialize the merged probability vector and store it on the GPU, the first one is the main model
    merged_probs = torch.zeros(
        (outputs[0].size(0), len(vocab_union)), device=outputs[0].device
    )

    v1_probs = torch.sparse.mm(outputs[0].float(), mapping_matrices[0].to(outputs[0].device))
    v2_probs = torch.sparse.mm(outputs[1].float(), mapping_matrices[1].to(outputs[1].device))
    merged_probs += v1_probs.to(merged_probs.device) + v2_probs.to(merged_probs.device)
    
    # sharpening distribution
    v1_top10_tokens = torch.topk(v1_probs[0], 10).indices
    v2_top10_tokens = torch.topk(v2_probs[0], 10).indices
    clone_merged_probs = merged_probs.clone()
    filtered_tokens = torch.nonzero(merged_probs[0]>0.2).squeeze()
    filtered_tokens = filtered_tokens[torch.isin(filtered_tokens, v1_top10_tokens.to(filtered_tokens.device))]
    cand_tokens = list(set(v1_top10_tokens.tolist()) | set(v2_top10_tokens.tolist()))
    max_prob = torch.max(merged_probs).item()
    
    if max_prob < 1.0: # Apply sharpening when distribution is smooth
        if sharpen_type == "geom":
            merged_probs = torch.zeros(
                (outputs[0].size(0), len(vocab_union)), device=outputs[0].device
            )
            merged_probs += torch.sqrt(v1_probs.to(merged_probs.device) * v2_probs.to(merged_probs.device))
        else:
            for top_ind in filtered_tokens:
                top_ind = top_ind.item()
                if index_to_vocab[top_ind] == " ":
                    continue
                for _key in cand_tokens:
                    if _key != top_ind and index_to_vocab[_key].startswith(index_to_vocab[top_ind]):
                        merged_probs[0, top_ind] = merged_probs[0, top_ind] + clone_merged_probs[0, _key].item()
    
    
    max_token_indices = torch.argmax(merged_probs, dim=1)
    max_tokens = [index_to_vocab[index.item()] for index in max_token_indices]

    # Convert to token IDs for each tokenizer
    batch_token_ids = [
        [] for _ in range(len(tokenizers))
    ]  # Initialize list for each model
    for i, tokenizer in enumerate(tokenizers):
        for token in max_tokens:
            if token in eos_token_list:
                token_id = [tokenizer.eos_token_id]
            else:
                # Convert token to corresponding tokenizer's token IDs using special_prefix_token
                token_id = get_token_ids(
                    tokenizer,
                    token,
                    special_prefix_token[tokenizer],
                    byte_mappings_list[i],
                )

            batch_token_ids[i].append(token_id)  # Append token IDs for each batch

    return batch_token_ids

@torch.no_grad()
def safe_generate_gac(
    inputs,
    ver_inputs,
    draft_model: Module,
    ver_model: Module,
    max_length=4096,
    draft_tokenizer = None,
    ver_tokenizer = None,
    vocab_union = None,
    mapping_matrices = None, 
    index_to_vocab = None, 
    byte_mappings_list = None, 
    special_prefix_tokens_dict = None,
    gamma: int = 5,
    draft=None,
    verifier=None,
    use_cache: bool = False,
    sharpen_type = "geom",
) -> Tuple[List[int], float]:
    """
    Args:
        inputs: drafter input sequence of batch size 1.
        verifier_inputs: verifier input sequence of batch size 1.
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
    
    eos_token_id = [draft_tokenizer.eos_token_id]
    if draft == "internlm":
        eos_token_id.append(128131)
    stop_tokens = torch.tensor(eos_token_id, dtype=torch.long, device=ver_model.device).unsqueeze(1)
    
    drafts_accepted, drafts_speculated = .0, .0
    num_ensemble = 0

    # prepare input tensor
    input_ids = inputs["input_ids"]
    ver_input_ids = ver_inputs["input_ids"]
    
    attention_mask = inputs["attention_mask"]
    ver_attention_mask = ver_inputs["attention_mask"]
    
    prompt_len = input_ids.shape[-1]
    total_len = prompt_len + max_length
    
    ver_prompt_len = ver_input_ids.shape[-1]
    
    # ensure greedy decoding
    draft_model.generation_config.do_sample = False
    draft_model.generation_config.temperature = 0.0
    ver_model.generation_config.do_sample = False
    ver_model.generation_config.temperature = 0.0

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
                "cache_position": cache_position
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
            if not ver_tokenizer.batch_decode(ver_input_ids[:, ver_prompt_len:])[0].startswith(prev_draft_seq):
                ver_current_position = ver_input_ids.shape[-1]
            else:
                #while len(prev_draft_seq) > len(ver_tokenizer.batch_decode(ver_input_ids[:, ver_prompt_len:ver_current_position-5+k])[0]):
                while not ver_tokenizer.batch_decode(ver_input_ids[:, ver_prompt_len:ver_current_position-gamma+k])[0].startswith(prev_draft_seq):
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
            "cache_position": ver_cache_position
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
                    draft_prob = torch.softmax(draft_logits[j], dim=-1)[0, input_ids[0, current_position+j]].item()
                    ver_prob = torch.softmax(ver_logit, dim=-1)[0, ver_input_ids[0, ver_current_position + i]].item()
                    # Ensemble distribution verification
                    if draft_prob < 0.5 and (draft_prob + ver_prob) < 1.0 and max_ver_id != ver_input_ids[0, ver_current_position + i].item():
                        num_accepted = j # drafter token position
                        num_verify = i # verifier token position
                        break
                    break
                elif len(draft_j_seq) > len(ver_i_seq):
                    break
            if num_verify < max_verify_len:
                break
        
        # 3. Ensemble (GaC)
        is_rejected = (num_verify < max_verify_len)
        if is_rejected:
            drafts_accepted += num_accepted
            if num_accepted < gamma:
                next_token_ids = get_ensemble_token([torch.softmax(draft_logits[num_accepted], dim=-1), torch.softmax(ver_logits[:, num_verify], dim=-1)], [draft_tokenizer, ver_tokenizer], sharpen_type, mapping_matrices, vocab_union, index_to_vocab, special_prefix_tokens_dict, byte_mappings_list)
                new_token_len = len(next_token_ids[0][0])
                next_token_id1 = torch.tensor(next_token_ids[0][0], device=input_ids.device).unsqueeze(0)
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
            stop_location = stop_locations[0, 1].item()
        """
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:], stop_tokens))
        if stop_locations.shape[0] > 0:
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
