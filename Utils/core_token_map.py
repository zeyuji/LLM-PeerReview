from pathlib import Path
from typing import Dict, Optional, Union

import torch
from tqdm.auto import tqdm


DEFAULT_TOKEN_MAP_CACHE_ROOT = Path("./CoRE_Cache/token_maps")
EPS = 1e-12


def safe_convert_ids_to_tokens(tokenizer, token_id: int, skip_special_tokens: bool = False) -> str:
    token = tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=skip_special_tokens)
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return token or ""


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _normalize_raw_token(token: str) -> str:
    return token.replace("Ġ", "▁").replace("<0x0A>", "\n").replace("Ċ", "\n")


def _normalize_text_token(token: str) -> str:
    return _normalize_raw_token(token).replace("▁", " ")


def _count_vocab_size(tokenizer) -> int:
    candidates = []

    try:
        candidates.append(int(len(tokenizer)))
    except Exception:
        pass

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        candidates.append(int(vocab_size))

    vocab = tokenizer.get_vocab()
    if vocab:
        candidates.append(max(vocab.values()) + 1)

    if not candidates:
        raise ValueError(f"Unable to infer tokenizer vocab size for {type(tokenizer)}")
    return max(candidates)


def _blank_token_id(tokenizer) -> int:
    blank_ids = tokenizer.encode(" ", add_special_tokens=False)
    if blank_ids:
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        for token_id in blank_ids:
            if token_id != unk_token_id:
                return int(token_id)
        return int(blank_ids[0])
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def _first_content_token_id(
    token_ids,
    blank_token_id: int,
    unk_token_id: Optional[int],
    max_token_id_exclusive: Optional[int] = None,
) -> Optional[int]:
    if token_ids is None:
        return None
    for token_id in token_ids:
        if max_token_id_exclusive is not None and int(token_id) >= int(max_token_id_exclusive):
            continue
        if token_id == blank_token_id:
            continue
        if unk_token_id is not None and token_id == unk_token_id:
            continue
        return int(token_id)
    return None


def _build_normalized_vocab(tokenizer, vocab_size: Optional[int] = None) -> Dict[str, int]:
    normalized_vocab: Dict[str, int] = {}
    for token, token_id in tokenizer.get_vocab().items():
        if vocab_size is not None and int(token_id) >= int(vocab_size):
            continue
        normalized_token = _normalize_raw_token(token)
        normalized_vocab.setdefault(normalized_token, int(token_id))
    return normalized_vocab


def _build_normalized_text_id_map(tokenizer, vocab_size: int) -> Dict[str, int]:
    normalized_map: Dict[str, int] = {}
    for token_id in range(vocab_size):
        token = safe_convert_ids_to_tokens(tokenizer, token_id, skip_special_tokens=True)
        if token == "":
            continue
        normalized_map.setdefault(_normalize_text_token(token), token_id)
    return normalized_map


def build_unite_token_map(
    source_tokenizer,
    target_tokenizer,
    source_vocab_size: Optional[int] = None,
    target_vocab_size: Optional[int] = None,
) -> torch.Tensor:
    source_vocab_size = int(source_vocab_size or _count_vocab_size(source_tokenizer))
    target_vocab_size = int(target_vocab_size or _count_vocab_size(target_tokenizer))

    target_vocab = _build_normalized_vocab(target_tokenizer, vocab_size=target_vocab_size)
    blank_token_id = _blank_token_id(target_tokenizer)
    unk_token_id = getattr(target_tokenizer, "unk_token_id", None)

    src_indices = []
    tgt_indices = []
    exact_matches = 0
    fallback_matches = 0
    unresolved = 0

    for source_token_id in tqdm(range(source_vocab_size), desc="Building UniTE token map"):
        source_token = safe_convert_ids_to_tokens(source_tokenizer, source_token_id, skip_special_tokens=True)
        if source_token == "":
            unresolved += 1
            continue

        normalized_raw = _normalize_raw_token(source_token)
        target_token_id = target_vocab.get(normalized_raw)

        if target_token_id is not None:
            exact_matches += 1
        else:
            normalized_text = _normalize_text_token(source_token)
            retokenized_ids = target_tokenizer.encode(normalized_text, add_special_tokens=False)
            target_token_id = _first_content_token_id(
                token_ids=retokenized_ids,
                blank_token_id=blank_token_id,
                unk_token_id=unk_token_id,
                max_token_id_exclusive=target_vocab_size,
            )
            if target_token_id is not None:
                fallback_matches += 1

        if target_token_id is None:
            target_token_id = blank_token_id
            unresolved += 1

        src_indices.append(source_token_id)
        tgt_indices.append(target_token_id)

    indices = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
    values = torch.ones(len(src_indices), dtype=torch.float32)
    token_map = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(source_vocab_size, target_vocab_size),
    ).coalesce()

    print(
        "unite_align\n"
        f"exact matches: {exact_matches}, fraction: {exact_matches / max(source_vocab_size, 1):.4f}\n"
        f"fallback matches: {fallback_matches}, fraction: {fallback_matches / max(source_vocab_size, 1):.4f}\n"
        f"unresolved(blank fallback): {unresolved}, fraction: {unresolved / max(source_vocab_size, 1):.4f}\n"
    )
    return token_map


def build_gac_token_map(
    source_tokenizer,
    target_tokenizer,
    source_vocab_size: Optional[int] = None,
    target_vocab_size: Optional[int] = None,
) -> torch.Tensor:
    source_vocab_size = int(source_vocab_size or _count_vocab_size(source_tokenizer))
    target_vocab_size = int(target_vocab_size or _count_vocab_size(target_tokenizer))
    target_text_to_id = _build_normalized_text_id_map(target_tokenizer, target_vocab_size)

    src_indices = []
    tgt_indices = []
    exact_matches = 0

    for source_token_id in tqdm(range(source_vocab_size), desc="Building GaC token map"):
        source_token = safe_convert_ids_to_tokens(source_tokenizer, source_token_id, skip_special_tokens=True)
        if source_token == "":
            continue
        target_token_id = target_text_to_id.get(_normalize_text_token(source_token))
        if target_token_id is None:
            continue
        src_indices.append(source_token_id)
        tgt_indices.append(target_token_id)
        exact_matches += 1

    indices = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
    values = torch.ones(len(src_indices), dtype=torch.float32)
    token_map = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(source_vocab_size, target_vocab_size),
    ).coalesce()

    print(
        "gac_align\n"
        f"exact matches: {exact_matches}, fraction: {exact_matches / max(source_vocab_size, 1):.4f}\n"
    )
    return token_map


def build_token_map(
    source_tokenizer,
    target_tokenizer,
    method: str,
    source_vocab_size: Optional[int] = None,
    target_vocab_size: Optional[int] = None,
) -> torch.Tensor:
    if method == "unite":
        return build_unite_token_map(
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
        )
    if method == "gac":
        return build_gac_token_map(
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
        )
    raise ValueError(f"Unsupported CoRE alignment method: {method}")


def get_cached_token_map(
    source_name: str,
    target_name: str,
    source_tokenizer,
    target_tokenizer,
    method: str,
    cache_root: Union[Path, str] = DEFAULT_TOKEN_MAP_CACHE_ROOT,
    source_vocab_size: Optional[int] = None,
    target_vocab_size: Optional[int] = None,
) -> torch.Tensor:
    cache_root = Path(cache_root)
    cache_dir = cache_root / method
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_source_vocab_size = int(source_vocab_size or _count_vocab_size(source_tokenizer))
    expected_target_vocab_size = int(target_vocab_size or _count_vocab_size(target_tokenizer))
    cache_path = cache_dir / f"{_sanitize_model_name(source_name)}__to__{_sanitize_model_name(target_name)}.pth"
    if cache_path.exists():
        cached_token_map = torch.load(cache_path, map_location="cpu")
        if tuple(cached_token_map.shape) == (expected_source_vocab_size, expected_target_vocab_size):
            return cached_token_map

        print(
            "Cached CoRE token map shape mismatch detected, rebuilding.\n"
            f"cache_path={cache_path}\n"
            f"cached_shape={tuple(cached_token_map.shape)}\n"
            f"expected_shape={(expected_source_vocab_size, expected_target_vocab_size)}"
        )
        cache_path.unlink()

    token_map = build_token_map(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        method=method,
        source_vocab_size=expected_source_vocab_size,
        target_vocab_size=expected_target_vocab_size,
    ).cpu()
    torch.save(token_map, cache_path)
    return token_map
