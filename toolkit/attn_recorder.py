"""Recording attention processor used during inference to capture cross-attention maps.

Usage:
- Create an instance: recorder = RecordingAttnProcessor()
- Call `sd.unet.set_attn_processor(recorder)` before sampling (or replace processors per-layer)
- After sampling: `records = recorder.get_records()` returns list of dicts {module, block_idx, attn: Tensor[B,H,T,S]}
- Utility: `token_to_spatial_masks_from_records` can use these records (see toolkit.attn_recorder.token_to_spatial_masks_from_records)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class RecordingAttnProcessor(nn.Module):
    """Records attention probability tensors during forward.

    This processor mirrors the API expected by UNet attention layers (it is
    compatible with the simple `AttnProcessor` signature in this repo). It
    captures attention probs as [B, H, T, S] tensors and stores them with
    metadata for later analysis.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or "recording_attn"
        self.records: List[Dict[str, Any]] = []

    def clear(self):
        """Clear recorded entries."""
        self.records.clear()

    def get_records(self):
        """Return recorded entries as a shallow copy."""
        return list(self.records)

    def export_numpy(self, out_path: str):
        """Export recorded attentions to a single NumPy .npz file for easy inspection.

        Format: arrays named by index, each value a dict in metadata list with keys:
            - 'module', 'block_idx', 'shape'
        """
        import numpy as _np
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        arrays = {}
        meta = []
        for i, r in enumerate(self.records):
            arrays[f"attn_{i}"] = r["attn"].cpu().numpy()
            meta.append({"module": r.get("module"), "block_idx": r.get("block_idx"), "shape": arrays[f"attn_{i}"].shape})
        arrays["meta"] = _np.array(str(meta))
        _np.savez_compressed(out_path, **arrays)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        block_idx: Optional[int] = None,
        per_block_prompt_embeds: Optional[dict] = None,
    ):
        """Compute attention probs and record them, then return standard output.

        Note: this implementation mirrors the computation pattern used by the
        repo's AttnProcessor and IPAttnProcessor to ensure consistent outputs.
        It expects `attn` to provide the same attributes/methods as the repo
        processors: `to_q`, `to_k`, `to_v`, `head_to_batch_dim`, `get_attention_scores`,
        `batch_to_head_dim`, `to_out`, `residual_connection`, `rescale_output_factor`, and
        optional spatial_norm/group_norm attributes.
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Save a detached CPU copy when possible and reshape to [B, H, T, S]
        attn_copy = attention_probs.detach()
        try:
            # If head-batched, attempt to recover B and H
            if attn_copy.ndim == 3:
                BtimesH, T, S = attn_copy.shape
                nb_heads = getattr(attn, "num_heads", None) or getattr(attn, "heads", None)
                batch_size = getattr(attn, "batch_size", None)
                if batch_size is not None and nb_heads is not None:
                    H = nb_heads
                    B = BtimesH // H
                elif nb_heads is not None and BtimesH % nb_heads == 0:
                    H = nb_heads
                    B = BtimesH // H
                else:
                    B = BtimesH
                    H = 1
                att_final = attn_copy.view(B, H, T, S)
            else:
                att_final = attn_copy
        except Exception:
            att_final = attn_copy

        # record
        module_name = getattr(attn, "module_name", None) or getattr(attn, "name", None)
        self.records.append({"module": module_name, "block_idx": block_idx, "attn": att_final.detach().cpu()})

        # the rest of forward: compute hidden states and return
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def avg_recorded_attentions(records: List[Dict[str, Any]], layers: Optional[List[int]] = None, heads: Optional[List[int]] = None):
    """Return a list of attention tensors [B,H,T,S] selected by layer indices."""
    out = []
    for i, r in enumerate(records):
        if layers is not None and i not in layers:
            continue
        out.append(r["attn"])
    return out


def token_to_spatial_masks_from_records(records: List[Dict[str, Any]], token_indices: List[int], H: int, W: int, layers: Optional[List[int]] = None, heads: Optional[List[int]] = None):
    """Build per-token spatial masks [B, K, H, W] from recorded attn records."""
    from toolkit.token_attention import avg_token_attention_maps

    att_list = avg_recorded_attentions(records, layers=layers, heads=heads)
    if len(att_list) == 0:
        raise RuntimeError("No recorded attentions available")
    head_avg = avg_token_attention_maps(att_list, layers=None, heads=heads)  # [B, T, S]
    B, T, S = head_avg.shape
    if any(ti < 0 or ti >= T for ti in token_indices):
        raise ValueError("token index out of range")
    maps = head_avg[:, token_indices, :]
    # maps shape: [B, K, S]
    if S == H * W:
        maps = maps.view(B, len(token_indices), H, W)
        return maps
    # if S is a perfect square, reshape to sqrt grid and upsample
    import math
    sroot = int(math.sqrt(S))
    if sroot * sroot == S:
        maps2 = maps.view(B, len(token_indices), sroot, sroot)
        maps2 = torch.nn.functional.interpolate(maps2, size=(H, W), mode='bilinear', align_corners=False)
        return maps2
    # fallback: tile flat values to fill the target spatial size
    rep = int(math.ceil((H * W) / S))
    flat = maps.repeat(1, 1, rep)[:, :, : H * W]
    maps2 = flat.view(B, len(token_indices), H, W)
    return maps2