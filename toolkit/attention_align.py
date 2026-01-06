"""Helpers for attention-based alignment supervision.

Functions:
- avg_attention_maps(attentions, token_indices, layers=None, heads=None)
- normalize_map(x, eps=1e-8)
- attention_alignment_loss(att_map, mask, mode='mse')
"""
from typing import List, Optional
import torch
import torch.nn.functional as F


def avg_attention_maps(attentions: List[torch.Tensor], token_indices: List[int], layers: Optional[List[int]] = None, heads: Optional[List[int]] = None) -> torch.Tensor:
    """Average attention tensors across selected layers and heads for given token indices.

    attentions: list of tensors, each shape [B, num_heads, seq_len, seq_len] or [B, num_heads, tgt_len, src_len]
    token_indices: list of token indices in the src sequence to extract (rare token positions)
    returns: tensor [B, len(token_indices), H_att, W_att] (if spatial) or [B, len(token_indices), src_len]
    Note: calling code must map token-related axis to spatial axes when using pixel-attention maps.
    """
    # Stack layers
    stacked = torch.stack([a.detach() if a is not None else torch.zeros(1) for a in attentions], dim=0)
    # stacked shape [L, B, H, T, S] or similar
    # Optionally select layers
    if layers is not None:
        stacked = stacked[layers]
    # Now average over layers
    layer_avg = stacked.mean(dim=0)  # [B, H, T, S]
    B, H, T, S = layer_avg.shape
    # Optionally select heads
    if heads is not None:
        layer_avg = layer_avg[:, heads, ...]
    # average over heads
    head_avg = layer_avg.mean(dim=1)  # [B, T, S]
    # Extract token-specific maps: for each token index, gather head_avg[:, :, token_idx]
    out = []
    for ti in token_indices:
        if ti < 0 or ti >= head_avg.shape[-1]:
            # safe fallback: zeros
            out.append(torch.zeros((B, head_avg.shape[-2]), device=head_avg.device, dtype=head_avg.dtype))
        else:
            out.append(head_avg[:, :, ti])
    # out is list of [B, T] where T is target-length (usually spatial flattened)
    # Stack to [B, K, T]
    out_t = torch.stack(out, dim=1)
    return out_t


def normalize_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize maps to sum-to-1 per sample+channel."""
    orig = x
    s = x.flatten(2).sum(dim=-1, keepdim=True)  # sum over spatial
    return x / (s.unsqueeze(-1) + eps) if s.numel() > 0 else x


def attention_alignment_loss(att_map: torch.Tensor, mask: torch.Tensor, mode: str = 'mse') -> torch.Tensor:
    """Compute loss between attention map and target mask.

    att_map: [B, K, N] or [B, K, H, W] (if 2D spatial)
    mask: [B, K, N] or [B, K, H, W]
    mode: 'mse'|'kl'|'iou'
    returns: scalar loss
    """
    # Ensure shapes match: flatten spatial dims
    if att_map.ndim == 4:
        # [B,K,H,W]
        att = att_map.flatten(2)
    else:
        att = att_map
    if mask.ndim == 4:
        tgt = mask.flatten(2)
    else:
        tgt = mask
    # normalize
    att_n = att / (att.sum(dim=-1, keepdim=True) + 1e-8)
    tgt_n = tgt / (tgt.sum(dim=-1, keepdim=True) + 1e-8)

    if mode == 'mse':
        loss = ((att_n - tgt_n) ** 2).mean()
        return loss
    if mode == 'kl':
        loss = (tgt_n * (tgt_n.log() - att_n.log())).mean()
        return loss
    if mode == 'iou':
        # soft IoU
        inter = (att_n * tgt_n).sum(dim=-1)
        union = (att_n + tgt_n - att_n * tgt_n).sum(dim=-1) + 1e-8
        loss = 1.0 - (inter / union).mean()
        return loss
    raise ValueError(f"Unknown mode: {mode}")