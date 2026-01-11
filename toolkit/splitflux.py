"""Helpers for SplitFlux RCA rank vector construction.

Functions:
- build_splitflux_block_dims(train_config) -> (content_block_dims, style_block_dims)

These return lists of length `num_total_blocks` (LoRANetwork.NUM_OF_BLOCKS * 2 + 1).
"""
from typing import List, Tuple
try:
    from toolkit.kohya_lora import LoRANetwork
except Exception:
    # Fallback minimal stub for environments where kohya_lora or heavy deps
    # (transformers, torch) may not be importable during lightweight tests
    class LoRANetwork:
        NUM_OF_BLOCKS = 12


def _make_filled(n: int, fill: int) -> List[int]:
    return [int(fill) for _ in range(n)]


def build_splitflux_block_dims(train_config, job_size: int = None, job_alpha: int = None) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Build per-block rank vectors AND alpha vectors for content and style LoRAs based on train_config.

    If `job_size` / `job_alpha` are provided (e.g., when target is LoKr or RCA is enabled), some RCA-specific
    adjustments are made:
      - blocks 1..19 (the early UNet blocks commonly frozen by RCA) are set to 0
      - content blocks are set to `job_size` and their alphas to `job_alpha` (if provided)
      - style primary and content primary use `job_size`/`job_alpha` where requested
      - spatial constrained ranks (blocks 30,31) are set to `job_size//2` and alpha to `job_alpha//2` when `job_size` is provided

    Returns: (content_block_dims, content_block_alphas, style_block_dims, style_block_alphas)
    """
    content_blocks = list(getattr(train_config, 'splitflux_content_blocks', list(range(20, 30))))
    style_blocks = list(getattr(train_config, 'splitflux_style_blocks', list(range(30, 58))))

    # Treat configured block numbers as 1-based (user-facing). Convert to 0-based indices for internal arrays.
    try:
        content_blocks = [int(b) - 1 for b in content_blocks]
    except Exception:
        pass
    try:
        style_blocks = [int(b) - 1 for b in style_blocks]
    except Exception:
        pass

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

    # If a job_size is given (e.g., RCA for LoKr), prefer that for the configured ranks
    if job_size is not None:
        # Per-request: first 19 blocks should be set to 0 when RCA is active
        # Use block indices 0..18 (inclusive) to represent the first 19 UNet blocks
        early_block_indices = list(range(0, 19))
        # set content/style primary to job size
        content_primary = int(job_size)
        style_primary = int(job_size)
        # if a separate alpha was provided prefer that for block alphas
        if job_alpha is not None:
            content_alpha_primary = int(job_alpha)
            style_alpha_primary = int(job_alpha)
        else:
            content_alpha_primary = content_primary
            style_alpha_primary = style_primary
        # spatial rank should be job_size // 2 and corresponding alpha halved (rounded down)
        spatial_rank = max(1, int(job_size // 2))
        spatial_alpha = max(1, int((job_alpha // 2) if job_alpha is not None else spatial_rank))
    else:
        content_alpha_primary = content_primary
        style_alpha_primary = style_primary
        spatial_alpha = spatial_rank
    content_bd = _make_filled(num_total_blocks, sec)
    style_bd = _make_filled(num_total_blocks, sec)

    # default alphas use secondary rank / network defaults
    content_ba = _make_filled(num_total_blocks, sec)
    style_ba = _make_filled(num_total_blocks, sec)

    # content_blocks and style_blocks were already read and converted to 0-based indices above
    # (do not re-read to avoid undoing the 1-based -> 0-based conversion)

    # clamp indices to valid range
    def clamp_idx(i):
        return max(0, min(num_total_blocks - 1, int(i)))

    # assign content primary
    for b in content_blocks:
        content_bd[clamp_idx(b)] = content_primary
        content_ba[clamp_idx(b)] = content_alpha_primary

    # assign style primary for style blocks
    for b in style_blocks:
        style_bd[clamp_idx(b)] = style_primary
        style_ba[clamp_idx(b)] = style_alpha_primary

    # assign spatial constrained ranks (override content/style where applicable)
    # Spatial blocks correspond to 30/31 (1-based) => indices 29 and 30 (0-based)
    for b in [29, 30]:
        if 0 <= b < num_total_blocks:
            content_bd[clamp_idx(b)] = spatial_rank
            style_bd[clamp_idx(b)] = spatial_rank
            content_ba[clamp_idx(b)] = spatial_alpha
            style_ba[clamp_idx(b)] = spatial_alpha

    # Apply early block zeroing for RCA if job_size is provided
    if job_size is not None:
        for bi in early_block_indices:
            if 0 <= bi < num_total_blocks:
                content_bd[bi] = 0
                style_bd[bi] = 0
                content_ba[bi] = 0
                style_ba[bi] = 0

    # Note: paper decomposes content primary into CNT/RES (48 + 16). This helper returns the top-level ranks.
    # The CNT/RES split will be handled by the Visual‑Gated LoRA wrapper later.

    return content_bd, content_ba, style_bd, style_ba


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


def rca_supported_arch(model_config, sd=None) -> bool:
    """Return True if RCA (SplitFlux) should be enabled for the given model architecture.

    Currently RCA is only supported for the original Flux architecture (Flux1).
    Flux2, Z-Image and other transformer variants use different single/double
    stream layouts and therefore should not enable Flux1-style RCA by default.
    """
    try:
        arch = getattr(model_config, 'arch', None)
        if isinstance(arch, str) and arch.lower() == 'flux':
            return True
        return False
    except Exception:
        return False


def build_rca_combined_block_dims(train_config, network_config=None) -> Tuple[List[int], List[float]]:
    """Return a single per-block dims vector and alphas vector representing the max rank required by content or style LoRA.

    If `network_config` is provided and indicates LoKr (or any network), we will use the network-configured
    linear and linear_alpha as the job size/alpha to shape RCA defaults (first 19 zeros,
    spatial rank = job_size//2, content/style primaries = job_size).
    """
    job_size = None
    job_alpha = None
    if network_config is not None:
        try:
            # Handle LoKr 'full_rank' sentinel: prefer the original rank stored in 'rank' when 'lokr_full_rank' is True
            if getattr(network_config, 'lokr_full_rank', False):
                cand = getattr(network_config, 'rank', None)
                if cand is not None and cand > 0 and cand < 1000000:
                    job_size = int(cand)
                # prefer an explicit linear_alpha if it looks sane
                cand_alpha = getattr(network_config, 'linear_alpha', None)
                if cand_alpha is not None and cand_alpha > 0 and cand_alpha < 1000000:
                    job_alpha = int(cand_alpha)
                else:
                    job_alpha = job_size
            else:
                # prefer the linear rank and alpha from the network config when present and sane
                cand = getattr(network_config, 'linear', None)
                if cand is not None and cand > 0 and cand < 1000000:
                    job_size = int(cand)
                cand_alpha = getattr(network_config, 'linear_alpha', None)
                if cand_alpha is not None and cand_alpha > 0 and cand_alpha < 1000000:
                    job_alpha = int(cand_alpha)
        except Exception:
            job_size = None
            job_alpha = None

    content_bd, content_ba, style_bd, style_ba = build_splitflux_block_dims(train_config, job_size=job_size, job_alpha=job_alpha)
    if len(content_bd) != len(style_bd) or len(content_ba) != len(style_ba):
        raise ValueError("content/style block dims/alphas length mismatch")
    combined_dims = [max(int(c), int(s)) for c, s in zip(content_bd, style_bd)]
    combined_alphas = [max(float(ca), float(sa)) for ca, sa in zip(content_ba, style_ba)]
    return combined_dims, combined_alphas


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
