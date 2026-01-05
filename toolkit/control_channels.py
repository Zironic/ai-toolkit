"""Unified helpers for inferring and adapting control image / latent channel shapes.

This module centralizes the various heuristics previously scattered across the
trainer and VideoX wrapper into a small, well-tested API.

Functions:
- infer_expected_in_ch(adapter): return int expected in-channels or None
- adapt_control_images(control_images, adapter): accept 4D or 5D tensors or lists
  and return a 4D tensor or list adapted to the adapter's expectation (trimming,
  padding, frame collapse/mean, etc.).
- adapt_noisy_latents_for_adapter(latents, expected_in): adapt latents channels via
  grouped mean, slice, or pad to match expected_in.

Note: This is intentionally narrow and focused on VideoX/Z-Image needs (the
user asked us to support the specific Alibaba Z-Image Turbo ControlNet).
"""
from typing import Any, Optional, Tuple, Dict
import inspect
import time
import traceback
import weakref
import torch

from .control_util import infer_expected_in_ch
from .print import print_acc

# Runtime sentinel to help verify training jobs pick up this code branch.
# This message is intentionally short and unique so you can grep logs for it:
# "CONTROL_CHANNELS:ASSEMBLE_V1_LOADED"
try:
    try:
        print_acc("[CONTROL_CHANNELS] ASSEMBLE_V1_LOADED")
    except Exception:
        print("[CONTROL_CHANNELS] ASSEMBLE_V1_LOADED")
except Exception:
    # best-effort; don't fail import
    pass

# Lightweight mapping from tensor id -> (weakref to tensor, metadata). Using
# id() avoids relying on tensor truthiness / hashing, which can be problematic
# for multi-element tensors in some Python versions.
_tensor_meta: "dict[int, tuple[weakref.ref, Dict]]" = {}


def _remove_callback(wr: weakref.ref):
    # Remove any entries referencing this weakref
    try:
        for tid, (r, _) in list(_tensor_meta.items()):
            if r is wr:
                _tensor_meta.pop(tid, None)
                break
    except Exception:
        # best-effort; don't fail the caller
        return


def _record_meta(t: torch.Tensor, meta: Dict):
    try:
        tid = id(t)
        _tensor_meta[tid] = (weakref.ref(t, _remove_callback), meta)
    except Exception as e:
        raise RuntimeError(f"Failed to record tensor metadata: {e}") from e


def tag_tensor(t: torch.Tensor, op: str):
    """Attach a small provenance record for `t` with operation `op`.

    Records the calling site (file, line, function), timestamp, and tensor shape/dtype/device.
    """
    if not isinstance(t, torch.Tensor):
        return
    try:
        # capture caller info (skip this function and its caller)
        stack = inspect.stack()
        # prefer the frame two levels up (the immediate caller of the helper)
        frame_info = stack[1]
        meta = {
            'op': op,
            'file': frame_info.filename if frame_info is not None else None,
            'line': frame_info.lineno if frame_info is not None else None,
            'func': frame_info.function if frame_info is not None else None,
            'time': time.time(),
            'shape': tuple(t.shape),
            'dtype': str(getattr(t, 'dtype', None)),
            'device': str(getattr(t, 'device', None)),
            'stack': traceback.format_list(traceback.extract_stack(limit=6)[:-1])
        }
        _record_meta(t, meta)
    except Exception as e:
        raise RuntimeError(f"Failed to tag tensor metadata: {e}") from e

def get_tensor_origin(t: torch.Tensor) -> Optional[Dict]:
    """Return recorded metadata for tensor `t`, or None if none recorded."""
    try:
        entry = _tensor_meta.get(id(t))
        return entry[1] if entry is not None else None
    except Exception:
        return None


def format_origin(t: torch.Tensor) -> str:
    """Return a short human-readable origin string for the tensor, if available."""
    meta = get_tensor_origin(t)
    if not meta:
        return 'unknown-origin'
    file = meta.get('file')
    line = meta.get('line')
    func = meta.get('func')
    op = meta.get('op')
    shape = meta.get('shape')
    return f"{op} at {file}:{line} in {func} (shape={shape})"

def _collapse_frames_if_present(t: torch.Tensor) -> torch.Tensor:
    """Collapse a 5D control tensor [B, C, F, H, W] to [B, C, H, W].

    If F == 1, squeeze; otherwise average across F (grouped mean across frames).
    Tags the resulting tensor with provenance info and logs the operation.
    """
    if t.ndim != 5:
        return t
    if t.shape[2] == 1:
        out = t[:, :, 0, :, :]
        tag_tensor(out, 'collapse_frames:squeeze')
        print_acc(f"[CONTROL_CHANNELS] collapse_frames:squeeze input_shape={tuple(t.shape)} -> out_shape={tuple(out.shape)}")
        return out
    out = t.mean(dim=2)
    tag_tensor(out, 'collapse_frames:mean')
    print_acc(f"[CONTROL_CHANNELS] collapse_frames:mean input_shape={tuple(t.shape)} -> out_shape={tuple(out.shape)}")
    return out


def _trim_or_pad_tensor(ch_tensor: torch.Tensor, expected: int) -> torch.Tensor:
    """Strict: require exact channel match for control tensors.

    VideoX parity: do not silently drop alpha channels or slice/pad control tensors.
    If the channel count does not match `expected`, raise a RuntimeError with an
    actionable message.
    """
    if expected is None:
        return ch_tensor
    B, C, H, W = ch_tensor.shape
    if C == expected:
        print_acc(f"[CONTROL_CHANNELS] _trim_or_pad_tensor: no-op channels ({C}) == expected ({expected}) shape={tuple(ch_tensor.shape)}")
        return ch_tensor
    # Strict: fail early and clearly
    raise RuntimeError(f"Control tensor has {C} channels but expected {expected}; strict VideoX parity requires exact channel match ({expected}). Please provide control_context with the exact channel count.")


def adapt_control_images(control_images: Any, adapter: Any, expected_override: Optional[int] = None) -> Tuple[Any, Optional[int]]:
    """Normalize control images to a 4D tensor or list matching the adapter.

    control_images may be:
    - None
    - torch.Tensor [B, C, H, W]
    - torch.Tensor [B, C, F, H, W]
    - list of torch.Tensor (each [B, C, H, W] or [B, C, F, H, W])

    If `expected_override` is provided, it will be used instead of inferring
    expected channels from the adapter. This makes adaptation deterministic
    when caller already computed expected channels (useful for wrapped adapters).

    Returns: (adapted_control_images, expected_in_ch)
    """
    if control_images is None:
        try:
            expected = expected_override if expected_override is not None else infer_expected_in_ch(adapter)
        except Exception as e:
            raise RuntimeError(f"Failed to infer expected control in-channels from adapter: {e}") from e
        return None, expected

    try:
        expected = expected_override if expected_override is not None else infer_expected_in_ch(adapter)
    except Exception as e:
        raise RuntimeError(f"Failed to infer expected control in-channels from adapter: {e}") from e

    # Fallback heuristics for common adapter shapes used in tests and real adapters.
    # Prefer explicit signals from adapter (control_in_dim), but fall back to conv_in
    # when present, or to Z-Image default of 4 channels for known Z-Image adapters.
    if expected is None:
        try:
            conv_in = getattr(adapter, 'conv_in', None)
            if conv_in is not None and hasattr(conv_in, 'weight'):
                expected = int(conv_in.weight.shape[1])
                print_acc(f"[CONTROL_CHANNELS] adapt_control_images: inferred expected_in from adapter.conv_in -> {expected}")
        except Exception:
            pass
    if expected is None:
        try:
            name = getattr(adapter, 'name_or_path', None) if adapter is not None else None
            name_str = str(name).lower() if name is not None else ''
            for pat in ('zimage', 'z_image', 'z-image', 'videox', 'video_x'):
                if pat in name_str:
                    # Z-Image Turbo canonical default for raw images is 4 channels
                    expected = 4
                    print_acc(f"[CONTROL_CHANNELS] adapt_control_images: detected Z-Image adapter name; defaulting expected_in=4")
                    break
        except Exception:
            pass

    def _process_tensor(t: torch.Tensor) -> torch.Tensor:
        nonlocal expected
        # collapse frames when present
        t_in_shape = tuple(t.shape)
        t = _collapse_frames_if_present(t)
        # ensure 4D
        if t.ndim != 4:
            raise RuntimeError(f"Adapted control tensor must be 4D after frame collapse, got ndim={t.ndim}")
        # Special-case: already-assembled VideoX/Z-Image control_context (33 channels).
        # In this case, do not attempt to trim/pad to the adapter's expected channels;
        # instead, accept and return the tensor unchanged and update the expected
        # value to reflect the control_context's channel count (33).
        try:
            if isinstance(t, torch.Tensor) and int(t.shape[1]) == 33:
                print_acc(f"[CONTROL_CHANNELS] adapt_control_images: detected assembled Z-Image control_context; passing through shape={tuple(t.shape)}")
                tag_tensor(t, 'adapt_control_images:control_context_passthrough')
                expected = 33
                return t
        except Exception:
            # best-effort: if inspection fails, continue with normal logic
            pass
        # adapt channels
        if expected is None:
            print_acc(f"[CONTROL_CHANNELS] adapt_control_images: no expected_in provided; returning shape={tuple(t.shape)} (from {t_in_shape})")
            return t

        # Heuristic: if tensor looks like a pixel image (1/3/4 channels, reasonable spatial size)
        # we allow friendly adaptations such as dropping or adding alpha channels to match
        # the adapter expectation (e.g., 4 -> 3 drop alpha, 3 -> 4 pad alpha)
        looks_like_pixel = isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[1] in (1, 3, 4) and max(t.shape[-2:]) >= 16
        try:
            if looks_like_pixel:
                C = int(t.shape[1])
                if C == 4 and expected == 3:
                    out = t[:, :3, ...]
                    print_acc(f"[CONTROL_CHANNELS] adapt_control_images: trimmed alpha channel 4->3 shape={tuple(out.shape)}")
                    tag_tensor(out, 'adapt_control_images:pixel_trim_alpha')
                    return out
                if C == 3 and expected == 4:
                    pad = torch.zeros((t.shape[0], 1, t.shape[2], t.shape[3]), dtype=t.dtype, device=t.device)
                    out = torch.cat([t, pad], dim=1)
                    print_acc(f"[CONTROL_CHANNELS] adapt_control_images: padded alpha channel 3->4 shape={tuple(out.shape)}")
                    tag_tensor(out, 'adapt_control_images:pixel_pad_alpha')
                    return out
                # falls through to strict enforcement for other mismatches
            out = _trim_or_pad_tensor(t, expected)
        except RuntimeError as e:
            # Provide richer diagnostics: include adapter identity, configured control_in_dim,
            # tensor origin metadata (if available), and whether the tensor looks like pixel images
            from .control_channels import format_origin as _format_origin  # local alias for clarity
            # Build robust adapter representation to avoid ambiguity between a missing
            # adapter name vs adapter object being None.
            try:
                adapter_repr = None if adapter is None else f"{type(adapter).__name__}@{hex(id(adapter))}"
            except Exception:
                adapter_repr = str(adapter)
            adapter_name = getattr(adapter, 'name_or_path', None)
            adapter_cfg_dim = getattr(adapter, 'control_in_dim', None)
            origin = _format_origin(t) if isinstance(t, torch.Tensor) else 'unknown-origin'
            # Heuristic classification for helpful hints
            looks_like = 'pixel-image' if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[1] in (1, 3, 4) and max(t.shape[-2:]) >= 16 else 'latents/unknown'
            hint = ''
            if adapter_cfg_dim == 33 or expected == 33:
                hint = 'Note: VideoX/Z-Image expects a 33-channel control_context (VAE-encoded latents + mask/inpaint). Make sure you encode images via the VAE and call assemble_zimage_control_context.'
            raise RuntimeError(
                f"Control tensor has {t.shape[1]} channels but expected {expected}; adapter_repr={adapter_repr} adapter.name_or_path={adapter_name!r} adapter.control_in_dim={adapter_cfg_dim} origin={origin} looks_like={looks_like}. {hint}"
            ) from e
        print_acc(f"[CONTROL_CHANNELS] adapt_control_images: adapted channels from {t_in_shape} -> {tuple(out.shape)} expected={expected}")
        return out

    if isinstance(control_images, torch.Tensor):
        out = _process_tensor(control_images)
        tag_tensor(out, 'adapt_control_images')
        return out, expected
    if isinstance(control_images, (list, tuple)):
        out_list = [ _process_tensor(t) for t in control_images ]
        for o in out_list:
            tag_tensor(o, 'adapt_control_images')
        return out_list, expected
    # other container types (dicts) are not supported by VideoX path; return as-is
    return control_images, expected


def assemble_zimage_control_context(
    control_latents: torch.Tensor,
    inpaint_latent: Optional[torch.Tensor] = None,
    mask_condition: Optional[torch.Tensor] = None,
    control_in_dim: Optional[int] = None,
    mask_from: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Assemble Z-Image style `control_context` matching VideoX behaviour.

    Behaviour (VideoX parity):
    - If `control_in_dim` is None or equals `control_latents.shape[1]`, return
      `control_latents` unchanged (4D tensor).
    - Otherwise, build a 5D control tensor with a singleton frame dim by
      concatenating: [control_latents.unsqueeze(2), mask_single.unsqueeze(2), inpaint_latent.unsqueeze(2)]
      where `mask_single` == `1 - mask_condition[:, :1]` interpolated to the
      inpaint spatial size using nearest neighbor.

    New option:
    - `mask_from`: optional tensor (pixel images or latents) used to auto-detect
      solid black/white backgrounds and synthesize a `mask_condition` when
      `mask_condition` is not provided. Detection is conservative and will fall
      back to the default (no mask) when no clear background is detected.

    Returns a 5D tensor [B, control_in_dim, 1, H, W] when assembled, or the
    original 4D `control_latents` when no assembly is required.
    """
    if control_latents is None:
        return None
    if not isinstance(control_latents, torch.Tensor):
        raise RuntimeError("control_latents must be a torch.Tensor")
    if control_latents.ndim != 4:
        raise RuntimeError(f"control_latents must be 4D [B,C,H,W], got ndim={control_latents.ndim}")

    C = int(control_latents.shape[1])
    B = int(control_latents.shape[0])
    dev = control_latents.device
    dtype = control_latents.dtype

    # If no special control dim requested or matches latents channels, pass-through
    if control_in_dim is None or control_in_dim == C:
        print_acc(f"[CONTROL_CHANNELS] assemble_zimage_control_context: pass-through channels={C}")
        return control_latents

    import torch.nn.functional as F

    # Determine target spatial size: prefer inpaint_latent if present
    if inpaint_latent is not None and isinstance(inpaint_latent, torch.Tensor) and inpaint_latent.ndim == 4:
        tgt_h, tgt_w = int(inpaint_latent.shape[-2]), int(inpaint_latent.shape[-1])
    else:
        tgt_h, tgt_w = int(control_latents.shape[-2]), int(control_latents.shape[-1])

    # Prepare inpaint latent
    if inpaint_latent is None:
        inpaint_latent = torch.zeros((B, C, tgt_h, tgt_w), dtype=dtype, device=dev)
    else:
        if not isinstance(inpaint_latent, torch.Tensor) or inpaint_latent.ndim != 4:
            raise RuntimeError("inpaint_latent must be a 4D torch.Tensor when provided")
        # If channels mismatch, raise - caller should provide same channel count
        if int(inpaint_latent.shape[1]) != C:
            raise RuntimeError(f"inpaint_latent channel mismatch: expected {C}, got {int(inpaint_latent.shape[1])}")
        # Resize inpaint spatial dims if needed
        if int(inpaint_latent.shape[-2]) != tgt_h or int(inpaint_latent.shape[-1]) != tgt_w:
            inpaint_latent = F.interpolate(inpaint_latent, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)

    # Prepare mask: collapse to single channel and interpolate to inpaint size
    # VideoX default behavior: when no mask is provided, treat mask as all-zero
    # which results in (1 - mask) == 1.0 across the spatial dims.
    # If `mask_from` is provided and `mask_condition` is None, attempt to
    # auto-detect a solid black/white background and synthesize a mask.
    if mask_condition is None and mask_from is not None:
        try:
            # Accept 4D or 5D tensors; if 5D collapse frames by mean
            src = mask_from
            if not isinstance(src, torch.Tensor):
                raise RuntimeError("mask_from must be a torch.Tensor when provided")
            if src.ndim == 5:
                src = src.mean(dim=2)
            if src.ndim != 4:
                raise RuntimeError("mask_from must be 4D or 5D tensor")

            # Normalize to 0..1 range conservatively
            def _to_01(x: torch.Tensor) -> torch.Tensor:
                if not x.is_floating_point():
                    return x.float() / 255.0
                if x.min() < -0.5 and x.max() <= 1.5:
                    return (x + 1.0) / 2.0
                return x

            src01 = _to_01(src)
            # per-pixel 'brightness' via max across channels
            pix = src01.abs().max(dim=1, keepdim=True).values

            black_thresh = 0.02
            white_thresh = 0.98
            frac_threshold = 0.92

            is_black = (pix < black_thresh).float()
            is_white = (pix > white_thresh).float()
            frac_black = is_black.mean(dim=[2,3]).squeeze(1)
            frac_white = is_white.mean(dim=[2,3]).squeeze(1)

            bg_mask = torch.zeros((B, 1, tgt_h, tgt_w), dtype=dtype, device=dev)
            detected = False
            for i in range(B):
                if float(frac_black[i]) >= frac_threshold:
                    mm = is_black[i:i+1]
                    if int(mm.shape[-2]) != tgt_h or int(mm.shape[-1]) != tgt_w:
                        mm = F.interpolate(mm, size=(tgt_h, tgt_w), mode='nearest')
                    bg_mask[i:i+1] = mm.to(dtype=dtype, device=dev)
                    detected = True
                elif float(frac_white[i]) >= frac_threshold:
                    mm = is_white[i:i+1]
                    if int(mm.shape[-2]) != tgt_h or int(mm.shape[-1]) != tgt_w:
                        mm = F.interpolate(mm, size=(tgt_h, tgt_w), mode='nearest')
                    bg_mask[i:i+1] = mm.to(dtype=dtype, device=dev)
                    detected = True

            if detected:
                # `bg_mask` has 1.0 for background pixels; assemble uses mask_single = 1 - mask_condition
                mask_single = 1.0 - bg_mask
            else:
                mask_single = torch.ones((B, 1, tgt_h, tgt_w), dtype=dtype, device=dev)
        except Exception:
            mask_single = torch.ones((B, 1, tgt_h, tgt_w), dtype=dtype, device=dev)
    else:
        if mask_condition is None:
            mask_single = torch.ones((B, 1, tgt_h, tgt_w), dtype=dtype, device=dev)
        else:
            if not isinstance(mask_condition, torch.Tensor):
                raise RuntimeError("mask_condition must be a torch.Tensor when provided")
            # Accept 4D [B,3,H,W] or [B,1,H,W]
            if mask_condition.ndim != 4:
                raise RuntimeError("mask_condition must be 4D [B, C, H, W]")
            # take first channel and invert as VideoX does: (1 - mask[:, :1])
            m = mask_condition[:, :1, ...].to(dtype=dtype, device=dev)
            # Interpolate to inpaint size using nearest
            if int(m.shape[-2]) != tgt_h or int(m.shape[-1]) != tgt_w:
                m = F.interpolate(1.0 - m, size=(tgt_h, tgt_w), mode='nearest')
            else:
                m = 1.0 - m
            mask_single = m

    # Ensure control_latents spatially match inpaint size
    if int(control_latents.shape[-2]) != tgt_h or int(control_latents.shape[-1]) != tgt_w:
        control_latents_rs = F.interpolate(control_latents, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
    else:
        control_latents_rs = control_latents

    # Unsqueeze frame dim (create 5D tensors)
    ctl = control_latents_rs.unsqueeze(2)
    msk = mask_single.unsqueeze(2)
    inp = inpaint_latent.unsqueeze(2)

    # Concatenate along channel dimension
    out = torch.cat([ctl, msk, inp], dim=1)
    tag_tensor(out, 'assemble_zimage_control_context')
    print_acc(f"[CONTROL_CHANNELS] assemble_zimage_control_context: assembled from C={C} -> out_ch={out.shape[1]} shape={tuple(out.shape)} target_control_in_dim={control_in_dim}")

    if out.shape[1] != control_in_dim:
        raise RuntimeError(f"Assembled control_context channels {out.shape[1]} != requested control_in_dim {control_in_dim}")
    return out


def adapt_noisy_latents_for_adapter(latents: torch.Tensor, expected_in: Optional[int]) -> torch.Tensor:
    """Adapt noisy latents to the adapter expected channels.

    Behavior:
    - If expected_in is None or latents is not a 4D tensor, return latents unchanged.
    - If C == expected_in: return unchanged.
    - If C > expected_in and divisible: perform grouped mean to reduce channels.
    - If C > expected_in and not divisible: slice first expected_in channels.
    - If C < expected_in: pad with zeros to reach expected_in.

    This function attempts to perform reasonable adaptations for noisy latents used
    internally during training, while still being explicit about its actions.
    """
    try:
        if expected_in is None or latents is None or not isinstance(latents, torch.Tensor):
            print_acc(f"[CONTROL_CHANNELS] adapt_noisy_latents_for_adapter: no-op (expected_in={expected_in}) latents_type={type(latents)}")
            return latents
        if latents.ndim != 4:
            print_acc(f"[CONTROL_CHANNELS] adapt_noisy_latents_for_adapter: no-op (ndim={latents.ndim})")
            return latents
        C = int(latents.shape[1])
        print_acc(f"[CONTROL_CHANNELS] adapt_noisy_latents_for_adapter: input_ch={C}, expected_in={expected_in}, shape={tuple(latents.shape)}")
        if C == expected_in:
            print_acc(f"[CONTROL_CHANNELS] adapt_noisy_latents_for_adapter: channels already match ({C})")
            return latents
        # Reduce channels via grouped mean when divisible
        if C > expected_in and C % expected_in == 0:
            group = C // expected_in
            B, _, H, W = latents.shape
            lat = latents.view(B, expected_in, group, H, W).mean(dim=2)
            print_acc(f"[CONTROL_CHANNELS] adapt_noisy_latents_for_adapter: grouped mean {C}->{expected_in} (group={group})")
            return lat
        # If fewer channels than expected, pad with zeros
        if C < expected_in:
            pad = torch.zeros((latents.shape[0], expected_in - C, latents.shape[2], latents.shape[3]), dtype=latents.dtype, device=latents.device)
            out = torch.cat([latents, pad], dim=1)
            print_acc(f"[CONTROL_CHANNELS] adapt_noisy_latents_for_adapter: padded {C}->{expected_in}")
            return out
        # Otherwise slice
        out = latents[:, :expected_in, ...]
        print_acc(f"[CONTROL_CHANNELS] adapt_noisy_latents_for_adapter: sliced {C}->{expected_in}")
        return out
    except Exception as e:
        raise RuntimeError(f"adapt_noisy_latents_for_adapter failed: {e}") from e
