"""Token-based saliency helpers and token->spatial attention masks.

Functions:
- token_zscore_saliency(embeddings, eps=1e-8) -> [B, T]
- token_sigmoid_weights(saliency, scale=10.0, threshold=None) -> [B, T]
- avg_token_attention_maps(attentions, layers=None, heads=None) -> [B, T, S]
- token_attention_mask_from_weights(att_maps, token_weights, normalize=True) -> [B, S]
- token_attention_mask_to_spatial(mask_flat, H, W) -> [B,1,H,W]

These are model-agnostic helpers that operate on collected attention tensors and on token embeddings.
"""
from typing import List, Optional
import torch
import torch.nn.functional as F


def token_zscore_saliency(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute per-token z-score based saliency.

    Given embeddings of shape [B, T, D], compute mean across tokens per sample and std across tokens per sample,
    then compute scalar saliency per token as the mean across dims of abs((emb - mean) / (std + eps)).

    Returns: [B, T] saliency values (non-negative).
    """
    if embeddings.ndim != 3:
        raise ValueError("embeddings must be shape [B, T, D]")
    mean = embeddings.mean(dim=1, keepdim=True)  # [B,1,D]
    std = embeddings.std(dim=1, unbiased=False, keepdim=True)  # [B,1,D]
    z = (embeddings - mean).abs() / (std + eps)  # [B,T,D]
    sal = z.mean(dim=-1)  # [B,T]
    return sal


def token_sigmoid_weights(saliency: torch.Tensor, scale: float = 10.0, threshold: Optional[float] = None) -> torch.Tensor:
    """Convert saliency into [0,1] weights via a scaled sigmoid.

    saliency: [B,T]
    threshold: if provided, scalar or [B,1] array to shift sigmoid. If None, use per-sample mean saliency.
    """
    if saliency.ndim != 2:
        raise ValueError("saliency must be shape [B, T]")
    if threshold is None:
        thr = saliency.mean(dim=1, keepdim=True)  # [B,1]
    else:
        thr = torch.tensor(threshold, device=saliency.device, dtype=saliency.dtype)
        if thr.ndim == 0:
            thr = thr.view(1, 1)
    weights = torch.sigmoid(scale * (saliency - thr))
    return weights


def avg_token_attention_maps(attentions: List[torch.Tensor], layers: Optional[List[int]] = None, heads: Optional[List[int]] = None) -> torch.Tensor:
    """Average attentions across selected layers and heads to produce per-token attention maps.

    attentions: list of tensors [B, H, T, S]
    returns: [B, T, S]
    """
    if len(attentions) == 0:
        raise ValueError("No attentions provided")
    stacked = torch.stack([a.detach() for a in attentions], dim=0)  # [L, B, H, T, S]
    if layers is not None:
        stacked = stacked[layers]
    layer_avg = stacked.mean(dim=0)  # [B, H, T, S]
    if heads is not None:
        layer_avg = layer_avg[:, heads, ...]
    head_avg = layer_avg.mean(dim=1)  # [B, T, S]
    return head_avg


def token_attention_mask_from_weights(att_maps: torch.Tensor, token_weights: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Build a flat mask over source positions by weighting token attention maps.

    att_maps: [B, T, S]
    token_weights: [B, T] weights per token (soft). Returns [B, S]
    """
    if att_maps.ndim != 3:
        raise ValueError("att_maps must be [B, T, S]")
    if token_weights.ndim != 2:
        raise ValueError("token_weights must be [B, T]")
    # Weighted sum over tokens
    mask = torch.einsum('bts,bt->bs', att_maps, token_weights)
    if normalize:
        denom = mask.sum(dim=-1, keepdim=True)
        mask = mask / (denom + 1e-9)
    return mask


def token_attention_mask_to_spatial(mask_flat: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Reshape a flat mask [B, S] to spatial [B,1,H,W].

    Expects S == H*W (flattened spatial)."""
    if mask_flat.ndim != 2:
        raise ValueError("mask_flat must be [B, S]")
    B, S = mask_flat.shape
    if S != H * W:
        raise ValueError(f"S ({S}) does not match H*W ({H*W})")
    out = mask_flat.view(B, 1, H, W)
    return out
