"""Helpers for SplitFlux RCA rank vector construction.

Functions:
- build_splitflux_block_dims(train_config) -> (content_block_dims, style_block_dims)

These return lists of length `num_total_blocks` (LoRANetwork.NUM_OF_BLOCKS * 2 + 1).
"""
from typing import List, Tuple
from toolkit.kohya_lora import LoRANetwork


def _make_filled(n: int, fill: int) -> List[int]:
    return [int(fill) for _ in range(n)]


def build_splitflux_block_dims(train_config) -> Tuple[List[int], List[int]]:
    """Build per-block rank vectors for content and style LoRAs based on train_config.

    Returns: (content_block_dims, style_block_dims)
    """
    content_blocks = list(getattr(train_config, 'splitflux_content_blocks', list(range(20, 30))))
    style_blocks = list(getattr(train_config, 'splitflux_style_blocks', list(range(30, 58))))

    # Ensure list length covers any explicitly referenced block indices in config
    default_blocks = LoRANetwork.NUM_OF_BLOCKS * 2 + 1
    max_cfg_idx = -1
    if len(content_blocks) > 0:
        max_cfg_idx = max(max_cfg_idx, max(content_blocks))
    if len(style_blocks) > 0:
        max_cfg_idx = max(max_cfg_idx, max(style_blocks))
    num_total_blocks = max(default_blocks, max_cfg_idx + 1)

    # defaults
    sec = int(getattr(train_config, 'splitflux_secondary_rank', 16))
    content_primary = int(getattr(train_config, 'splitflux_content_primary_rank', 64))
    content_cnt = int(getattr(train_config, 'splitflux_content_cnt_rank', 48))
    content_res = int(getattr(train_config, 'splitflux_content_res_rank', 16))
    spatial_rank = int(getattr(train_config, 'splitflux_spatial_rank', 32))
    style_primary = int(getattr(train_config, 'splitflux_style_primary_rank', 64))

    content_bd = _make_filled(num_total_blocks, sec)
    style_bd = _make_filled(num_total_blocks, sec)

    content_blocks = list(getattr(train_config, 'splitflux_content_blocks', list(range(20, 30))))
    style_blocks = list(getattr(train_config, 'splitflux_style_blocks', list(range(30, 58))))

    # clamp indices to valid range
    def clamp_idx(i):
        return max(0, min(num_total_blocks - 1, int(i)))

    # assign content primary
    for b in content_blocks:
        content_bd[clamp_idx(b)] = content_primary

    # assign style primary for style blocks
    for b in style_blocks:
        style_bd[clamp_idx(b)] = style_primary

    # assign spatial constrained ranks (override content/style where applicable)
    for b in [30, 31]:
        if 0 <= b < num_total_blocks:
            content_bd[clamp_idx(b)] = spatial_rank
            style_bd[clamp_idx(b)] = spatial_rank

    # Note: paper decomposes content primary into CNT/RES (48 + 16). This helper returns the top-level ranks.
    # The CNT/RES split will be handled by the Visual‑Gated LoRA wrapper later.

    return content_bd, style_bd


def _module_name_matches_block(name: str, block_idx: int) -> bool:
    """Return True if the module name corresponds to the given UNet block index.

    Matches naming patterns such as 'down_blocks_{i}_', 'up_blocks_{i}_', and 'mid_block' for the middle block.
    """
    if f"down_blocks_{block_idx}_" in name:
        return True
    if f"up_blocks_{block_idx}_" in name:
        return True
    # middle block index equals LoRANetwork.NUM_OF_BLOCKS
    if block_idx == LoRANetwork.NUM_OF_BLOCKS and "mid_block" in name:
        return True
    return False


def freeze_unet_blocks(unet: object, block_indices: List[int]) -> List[str]:
    """Freeze parameters for modules that belong to the given UNet block indices.

    Returns list of parameter names that were frozen.
    """
    if block_indices is None:
        return []
    frozen = []
    for name, module in unet.named_modules():
        for bi in block_indices:
            if _module_name_matches_block(name, bi):
                for pn, p in module.named_parameters(recurse=True):
                    full = f"{name}.{pn}"
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen.append(full)
                break
    return frozen


def build_rca_combined_block_dims(train_config) -> List[int]:
    """Return a single per-block dims vector representing the max rank required by content or style LoRA.

    This is useful for configuring a single LoRA network that will be split later.
    """
    content_bd, style_bd = build_splitflux_block_dims(train_config)
    if len(content_bd) != len(style_bd):
        raise ValueError("content/style block dims length mismatch")
    combined = [max(int(c), int(s)) for c, s in zip(content_bd, style_bd)]
    return combined


def complementary_loss(
    content_net,
    style_net,
    weight: float = 0.01,
    normalize: str = 'rnorm',
    eps: float = 1e-8,
    include_names: list = None,
) -> 'torch.Tensor':
    """Compute the complementary (orthogonality) loss between two LoRA networks.

    The loss encourages the low-rank factors of the `style_net` to be orthogonal to those of
    the `content_net`, helping the residual (style) LoRA to specialize on directions not captured
    by the content LoRA.

    Arguments:
      content_net, style_net: objects with attributes `text_encoder_loras` and `unet_loras` (e.g., LoRANetwork)
      weight: scalar multiplier for the returned loss
      normalize: one of {'rnorm', 'scale_inv', 'none'} determining normalization strategy
      eps: small constant to avoid division by zero
      include_names: optional list of LoRA names to restrict which module pairs are considered

    Returns:
      A scalar torch.Tensor with the complementary loss (already multiplied by `weight`).
    """
    import torch

    def _collect_modules(net):
        modules = {}
        for l in getattr(net, 'text_encoder_loras', []) + getattr(net, 'unet_loras', []):
            modules[l.lora_name] = l
        return modules

    def _get_tensor_from_obj(obj):
        # Accept nn.Module (with .weight), Parameter or Tensor
        if hasattr(obj, 'weight'):
            return obj.weight
        if isinstance(obj, torch.nn.Parameter) or isinstance(obj, torch.Tensor):
            return obj
        raise ValueError('Unsupported lora param object')

    def _mat_down(lora):
        # lora_down.weight shape: (r, in, *kernel) or Parameter/Tensor equivalent
        w = _get_tensor_from_obj(lora.lora_down)
        if w.ndim > 2:
            return w.view(w.shape[0], -1)
        return w.view(w.shape[0], -1)

    def _mat_up(lora):
        # lora_up.weight shape: (out, r, *kernel) or (out, r) or Parameter/Tensor equivalent
        w = _get_tensor_from_obj(lora.lora_up)
        if w.ndim > 2:
            return w.view(w.shape[0], w.shape[1])
        return w.view(w.shape[0], w.shape[1])

    content_modules = _collect_modules(content_net)
    style_modules = _collect_modules(style_net)

    # intersect module names
    names = sorted(set(content_modules.keys()) & set(style_modules.keys()))
    if include_names is not None:
        names = [n for n in names if n in include_names]

    # determine device safely
    device = torch.device('cpu')
    if len(content_modules):
        try:
            some = next(iter(content_modules.values()))
            device = _get_tensor_from_obj(some.lora_down).device
        except Exception:
            pass
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for name in names:
        c = content_modules[name]
        s = style_modules[name]
        try:
            D1 = _mat_down(c)
            D2 = _mat_down(s)
            U1 = _mat_up(c)
            U2 = _mat_up(s)
        except Exception:
            # if a module doesn't have expected attributes/shape, skip
            continue

        if D1.shape[1] != D2.shape[1] or U1.shape[0] != U2.shape[0]:
            # incompatible shapes; skip
            continue

        # cross-correlation
        A = D1 @ D2.t()  # (r1, r2)
        B = U1.t() @ U2  # (r1, r2)

        if normalize == 'rnorm':
            n = float(A.shape[0] * A.shape[1])
            loss_m = (A.pow(2).sum() + B.pow(2).sum()) / (n + eps)
        elif normalize == 'scale_inv':
            dn = (D1.pow(2).sum().sqrt() * D2.pow(2).sum().sqrt()).clamp(min=eps)
            un = (U1.pow(2).sum().sqrt() * U2.pow(2).sum().sqrt()).clamp(min=eps)
            loss_m = (A / dn).pow(2).sum() + (B / un).pow(2).sum()
            loss_m = loss_m / (A.numel() + B.numel())
        else:
            loss_m = A.pow(2).sum() + B.pow(2).sum()

        total_loss = total_loss + loss_m
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True)

    total_loss = total_loss / float(count)
    return total_loss * weight
