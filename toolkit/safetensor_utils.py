"""Small helpers for inspecting safetensor files and mapping token ids to embedding rows."""
from typing import List, Tuple, Dict, Any
import os

import torch
from safetensors.torch import load_file


def find_embedding_keys(safetensor_path: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """Return list of candidate embedding keys with their shapes.

    Heuristic: return keys with 2D tensors whose key name contains 'embed'|'token'|'text'|'wte' or any 2D tensor if none match.
    """
    sd = load_file(safetensor_path)
    candidates = []
    fallback = []
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        if len(v.shape) == 2:
            lname = k.lower()
            if any(x in lname for x in ("embed", "embedding", "token", "text", "wte", "word")):
                candidates.append((k, tuple(v.shape)))
            else:
                fallback.append((k, tuple(v.shape)))
    return candidates if candidates else fallback


def map_token_ids_to_embedding_rows(safetensor_path: str, embedding_key: str, token_ids: List[int]) -> Dict[int, List[float]]:
    """Return a dict mapping token id -> embedding row (as python list of floats).

    Raises if embedding_key not found or token id out of range.
    """
    sd = load_file(safetensor_path)
    if embedding_key not in sd:
        raise KeyError(f"Key {embedding_key} not found in {safetensor_path}")
    emb = sd[embedding_key]
    if len(emb.shape) != 2:
        raise ValueError("Embedding tensor must be 2-D")
    out = {}
    for tid in token_ids:
        if tid < 0 or tid >= emb.shape[0]:
            raise IndexError(f"Token id {tid} out of range for embedding with first-dim {emb.shape[0]}")
        vec = emb[tid]
        out[tid] = vec.cpu().tolist() if hasattr(vec, "cpu") else vec.tolist()
    return out
