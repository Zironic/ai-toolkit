"""Prompt -> token mapping helpers

Functions:
- tokenize_prompt(prompt, tokenizer) -> dict(tokens, ids, offsets)
- merge_subwords_to_words(tokens, offsets) -> list of {word, token_indices, token_texts}
- export_prompt_mapping_json(prompt, tokenizer, safetensor_paths, out_path)

These helpers support word-level labels for attention visualizations.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import os

import torch


def tokenize_prompt(prompt: str, tokenizer) -> Dict[str, Any]:
    """Tokenize `prompt` and return token texts, ids, and optional offsets.

    Returns dict with keys: 'tokens' (list[str]), 'ids' (list[int]), 'offsets' (list[tuple]|None).
    """
    # Prefer fast tokenizer offsets mapping
    try:
        enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = enc["input_ids"]
        offsets = enc.get("offset_mapping", None)
        # Convert to plain lists
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        # Normalize offsets (list of tuples) if present
        if offsets is not None and isinstance(offsets[0][0], torch.Tensor):
            offsets = [tuple(o) for o in offsets]
    except Exception:
        # Fallback: basic tokenization without offsets
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        offsets = None

    return {"tokens": tokens, "ids": input_ids, "offsets": offsets}


def merge_subwords_to_words(tokens: List[str], offsets: Optional[List[tuple]]) -> List[Dict[str, Any]]:
    """Merge BPE/subword tokens into word-level spans.

    If `offsets` is given, use character offsets to group tokens that belong to the same word.
    Otherwise use heuristic based on common BPE prefixes (##, ▁, Ġ).

    Returns a list of dicts: {"word": str, "token_indices": [int], "token_texts": [str]}
    """
    out = []

    if offsets:
        # offsets: list of (start, end) for each token
        current = None
        for i, (tok, off) in enumerate(zip(tokens, offsets)):
            if current is None:
                current = {"token_indices": [i], "token_texts": [tok], "start": off[0], "end": off[1]}
            else:
                # if token begins after the current end -> new word, otherwise continuation/subword
                if off[0] > current["end"]:
                    # finalize previous
                    word_text = "".join(current["token_texts"]) if len(current["token_texts"]) > 1 else current["token_texts"][0]
                    out.append({"word": word_text, "token_indices": current["token_indices"], "token_texts": current["token_texts"]})
                    current = {"token_indices": [i], "token_texts": [tok], "start": off[0], "end": off[1]}
                else:
                    # continuation (subword)
                    current["token_indices"].append(i)
                    current["token_texts"].append(tok)
                    current["end"] = max(current["end"], off[1])
        if current is not None:
            word_text = "".join(current["token_texts"]) if len(current["token_texts"]) > 1 else current["token_texts"][0]
            out.append({"word": word_text, "token_indices": current["token_indices"], "token_texts": current["token_texts"]})
        return out

    # fallback heuristic grouping by common BPE markers
    def is_prefix(token: str) -> bool:
        return token.startswith("##") or token.startswith("▁") or token.startswith("Ġ") or token.startswith("▁")

    current = None
    for i, tok in enumerate(tokens):
        if current is None:
            current = {"token_indices": [i], "token_texts": [tok]}
        else:
            # if token looks like a continuation -> append
            if tok.startswith("##") or tok.startswith("▁"):
                current["token_indices"].append(i)
                current["token_texts"].append(tok)
            else:
                # new word
                word_text = "".join(current["token_texts"]) if len(current["token_texts"]) > 1 else current["token_texts"][0]
                out.append({"word": word_text, "token_indices": current["token_indices"], "token_texts": current["token_texts"]})
                current = {"token_indices": [i], "token_texts": [tok]}
    if current is not None:
        word_text = "".join(current["token_texts"]) if len(current["token_texts"]) > 1 else current["token_texts"][0]
        out.append({"word": word_text, "token_indices": current["token_indices"], "token_texts": current["token_texts"]})

    return out


def export_prompt_mapping_json(prompt: str, tokenizer, safetensor_paths: List[str], out_path: str) -> Dict[str, Any]:
    """Create and write a mapping JSON describing words -> token ids -> (candidate) embedding keys and row indices.

    The produced JSON contains an array of entries: {word, token_ids, token_texts, embedding_candidates: [{file, key, row_indices}]}
    """
    tok = tokenize_prompt(prompt, tokenizer)
    merged = merge_subwords_to_words(tok["tokens"], tok["offsets"])
    max_id = max(tok["ids"]) if len(tok["ids"]) > 0 else 0

    from toolkit.safetensor_utils import find_embedding_keys, map_token_ids_to_embedding_rows

    embedding_candidates = {}
    for sp in safetensor_paths:
        try:
            keys = find_embedding_keys(sp)
        except Exception:
            keys = []
        for key, shape in keys:
            # require that the embedding matrix covers token ids in this prompt
            if shape[0] > max_id:
                embedding_candidates.setdefault(sp, []).append({"key": key, "shape": shape})

    entries = []
    for m in merged:
        token_ids = [tok["ids"][i] for i in m["token_indices"]]
        cand = []
        for sp, ks in embedding_candidates.items():
            for info in ks:
                # attempt to map ids -> rows (rows may be high-dim lists, user can inspect separately)
                rows = map_token_ids_to_embedding_rows(sp, info["key"], token_ids)
                cand.append({"file": sp, "key": info["key"], "row_indices": token_ids})
        entries.append({"word": m["word"], "token_texts": m["token_texts"], "token_ids": token_ids, "embedding_candidates": cand})

    out = {"prompt": prompt, "tokens": tok, "words": entries}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
