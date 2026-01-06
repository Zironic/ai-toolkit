"""Utilities to split a combined LoRA state dict into content/style parts by block indices.

Functions:
- split_lora_state_dict(state_dict, content_blocks, style_blocks) -> (content_sd, style_sd)
- save_split_lora_from_file(in_file, out_content_file, out_style_file, content_blocks, style_blocks, metadata=None, dtype=None)
"""
from typing import Dict, Tuple, Iterable
import os
import torch

from toolkit.kohya_lora import get_block_index


def split_lora_state_dict(state_dict: Dict[str, torch.Tensor], content_blocks: Iterable[int], style_blocks: Iterable[int]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split an in-memory LoRA state_dict into content and style dicts.

    Behavior:
    - If a key maps to a block present in content_blocks include it into content dict.
    - If a key maps to a block present in style_blocks include it into style dict.
    - If a key does not map to a block (get_block_index == -1), include it in BOTH outputs (safe fallback).
    - Overlapping blocks (e.g., 30,31) will result in keys included in both outputs.

    Returns: (content_state_dict, style_state_dict)
    """
    content_sd = {}
    style_sd = {}

    content_set = set(int(x) for x in content_blocks)
    style_set = set(int(x) for x in style_blocks)

    for k, v in state_dict.items():
        # Determine the lora block index by scanning tokens in the key (robust to naming variants)
        idx = -1
        if isinstance(k, str):
            for token in k.split('.'):
                try:
                    idx = get_block_index(token)
                    if idx != -1:
                        break
                except Exception:
                    idx = -1
        else:
            idx = -1

        included_in_content = False
        included_in_style = False

        if idx != -1:
            if idx in content_set:
                content_sd[k] = v.clone().detach().cpu()
                included_in_content = True
            if idx in style_set:
                style_sd[k] = v.clone().detach().cpu()
                included_in_style = True
        else:
            # Not recognized as block-specific: include in both
            content_sd[k] = v.clone().detach().cpu()
            style_sd[k] = v.clone().detach().cpu()
            included_in_content = True
            included_in_style = True

        # If recognized (idx != -1) but not included in either (e.g., block outside both sets), include in both to be safe
        if idx != -1 and not (included_in_content or included_in_style):
            content_sd[k] = v.clone().detach().cpu()
            style_sd[k] = v.clone().detach().cpu()

    return content_sd, style_sd


def save_split_lora_from_file(in_file: str, out_content_file: str, out_style_file: str, content_blocks: Iterable[int], style_blocks: Iterable[int], metadata=None, dtype=None, content_block_dims=None, style_block_dims=None):
    """Load a lora state dict from file, split it, and save two files.

    Supports .safetensors and torch files; output format mirrors requested file extensions.

    Adds metadata to outputs: 'split_type', 'blocks', 'block_dims' (if provided), and 'source_file'.
    For non-safetensor outputs, a sidecar JSON metadata file is written (out_file + '.meta.json').
    """
    import json

    # load
    state_dict = None
    if os.path.splitext(in_file)[1] == ".safetensors":
        try:
            from safetensors.torch import load_file
            state_dict = load_file(in_file)
        except Exception:
            # fallback to torch
            state_dict = torch.load(in_file, map_location='cpu')
    else:
        state_dict = torch.load(in_file, map_location='cpu')

    content_sd, style_sd = split_lora_state_dict(state_dict, content_blocks, style_blocks)

    # Optionally cast dtype
    if dtype is not None:
        for d in (content_sd, style_sd):
            for k in list(d.keys()):
                d[k] = d[k].to(dtype)

    # Prepare metadata objects
    def _make_meta(split_type, blocks, block_dims):
        m = {} if metadata is None else dict(metadata)
        m = {k: str(v) for k, v in m.items()}  # ensure string values for safetensors
        m["split_type"] = split_type
        m["blocks"] = str(list(blocks))
        if block_dims is not None:
            m["block_dims"] = str(list(block_dims))
        m["source_file"] = os.path.basename(in_file)
        return m

    meta_content = _make_meta("content", content_blocks, content_block_dims)
    meta_style = _make_meta("style", style_blocks, style_block_dims)

    # Save
    def _save_dict(d, path, meta):
        if os.path.splitext(path)[1] == ".safetensors":
            try:
                from safetensors.torch import save_file
                save_file(d, path, meta)
                return
            except Exception:
                # fallback to torch
                torch.save(d, path)
                # write sidecar metadata
                meta_path = path + ".meta.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                return
        else:
            torch.save(d, path)
            meta_path = path + ".meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)

    _save_dict(content_sd, out_content_file, meta_content)
    _save_dict(style_sd, out_style_file, meta_style)

    return out_content_file, out_style_file