"""Compatibility helpers for VideoX/zimage-style ControlNets.

Provide a thin shim that adapts a VideoX-style controlnet to an object
that can be called by `StableDiffusion._predict_noise_zimage` and
by trainer routing code. The wrapper tries common call signatures
and normalizes return values for robustness in mixed environments.
"""
import torch
from typing import Any


class ControlNetLegacyAdapter(torch.nn.Module):
    """Adapter shim that maps legacy ControlNet signatures to VideoX-style.

    Some older or Flux1-style adapters accept kwargs like `controlnet_cond` or
    positional `controlnet_cond` instead of `control_context`. This shim
    provides a `forward(latents_list, timestep, control_context=..., conditioning_scale=...)`
    signature and forwards to the inner adapter using common legacy kw names.
    """

    def __init__(self, inner: Any):
        super().__init__()
        self.inner = inner

    def forward(self, latents_list, timestep, control_context=None, conditioning_scale: float = 1.0, *args, **kwargs):
        # Normalize latents_list -> the legacy adapters often expect a tensor or list
        latents_for_inner = latents_list
        if isinstance(latents_list, list):
            # many legacy adapters accept a stacked tensor; try stacking
            try:
                latents_for_inner = torch.stack(latents_list, dim=0)
            except Exception:
                latents_for_inner = latents_list

        # Attempt to call using common legacy kw names
        call_kwargs = dict(kwargs)
        # prefer `controlnet_cond` name if present
        call_kwargs['controlnet_cond'] = control_context
        # also supply common conditioning names
        call_kwargs['controlnet_conditioning_scale'] = conditioning_scale
        call_kwargs['conditioning_scale'] = conditioning_scale
        try:
            return self.inner(latents_for_inner, timestep, **call_kwargs)
        except Exception as e:
            # try positional fallback (latents, timestep, controlnet_cond)
            try:
                return self.inner(latents_for_inner, timestep, control_context, *args, **kwargs)
            except Exception:
                raise


class VideoXControlnetWrapper(torch.nn.Module):
    """Torch-friendly wrapper for VideoX-style ControlNets.

    This wrapper subclasses `torch.nn.Module` so it works with the project's
    offload helpers which expect modules supporting `.to()` and parameter
    inspection. It delegates forward calls to the inner adapter and enforces
    strict VideoX parity: the adapter MUST accept a `control_context` kwarg
    and MUST NOT require legacy Flux1-style positional-only `encoder_hidden_states`.
    """

    def __init__(self, inner: Any):
        super().__init__()
        self.inner = inner
        # Early validation: ensure the inner adapter presents a VideoX-compatible signature
        # If `inner` is None (used in some tests), do not enforce signature validation here; the None-case is checked at call-time.
        if self.inner is None:
            return
        try:
            import inspect
            # If the adapter exposes an explicit `forward` method, we can statically inspect
            # it and enforce strict VideoX parity at construction time.
            target_fn = getattr(self.inner, 'forward', None)
            if target_fn is None:
                # If `forward` is not present, do not enforce static signature checks here.
                # Allow construction and defer validation to runtime to enable test harnesses
                # and non-module callables to be wrapped and exercised (some tests rely on
                # constructing wrappers around simple callables or objects that provide
                # `.to()` but not a `forward` attribute).
                return

            sig = inspect.signature(target_fn)
            params = sig.parameters
            # If the adapter does not accept `control_context`, allow construction
            # to proceed and attempt a safe runtime shim at call time. We still
            # reject adapters that appear to require positional `encoder_hidden_states`
            # since they are incompatible with strict VideoX routing.
            if 'control_context' not in params:
                # Mark that the adapter may need a legacy shim at runtime
                try:
                    self._may_need_legacy_shim = True
                except Exception:
                    pass
            # Reject Flux1-style adapters that require positional encoder hidden states
            if 'encoder_hidden_states' in params:
                p = params['encoder_hidden_states']
                if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                    raise RuntimeError("Adapter appears to require 'encoder_hidden_states' (Flux1-style). For strict VideoX/Z-Image routing, use an adapter that accepts 'control_context' and does not require 'encoder_hidden_states'.")
        except RuntimeError:
            raise
        except Exception:
            # If signature inspection fails for any reason, fail fast to avoid silent mismatches
            raise RuntimeError("Unable to validate adapter signature for VideoX parity; ensure adapter has a `forward(self, latents_list, timestep, control_context=..., ...)` signature.") from None

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        """Simplified, robust forward wrapper that attempts several common
        call signatures on the inner adapter. Keeps provenance tagging and
        deterministic channel adaptation, but uses a clearer linear flow to
        avoid deeply nested try/except chains that can be fragile.
        """
        # Fail fast if inner adapter is missing to avoid confusing fallback behavior.
        if self.inner is None:
            raise RuntimeError("VideoXControlnetWrapper: inner adapter is None. Ensure the ControlNet adapter was loaded successfully before calling the wrapper.")

        # Helper functions
        def _inner_device():
            try:
                params = getattr(self.inner, 'parameters', None)
                if callable(params):
                    for p in params():
                        return p.device
            except Exception:
                pass
            try:
                buffs = getattr(self.inner, 'buffers', None)
                if callable(buffs):
                    for b in buffs():
                        return b.device
            except Exception:
                pass
            return None

        def _move(obj, device):
            if device is None:
                return obj
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, list):
                return [ _move(x, device) for x in obj ]
            if isinstance(obj, tuple):
                return tuple(_move(list(obj), device))
            if isinstance(obj, dict):
                return { k: _move(v, device) for k, v in obj.items() }
            return obj

        def _trim_channels(obj, expected):
            if expected is None:
                return obj
            if isinstance(obj, torch.Tensor):
                if obj.ndim >= 2 and obj.shape[1] != expected:
                    if obj.shape[1] == expected + 1 and expected in (1, 3):
                        return obj[:, :expected, ...]
                    if obj.shape[1] > expected:
                        return obj[:, :expected, ...]
                    if obj.shape[1] < expected:
                        pad = torch.zeros((obj.shape[0], expected - obj.shape[1], *obj.shape[2:]), dtype=obj.dtype, device=obj.device)
                        return torch.cat([obj, pad], dim=1)
                return obj
            if isinstance(obj, list):
                return [ _trim_channels(x, expected) for x in obj ]
            if isinstance(obj, tuple):
                return tuple(_trim_channels(list(obj), expected))
            if isinstance(obj, dict):
                return { k: _trim_channels(v, expected) for k,v in obj.items() }
            return obj

        def _maybe_restore_output(out, orig_ch):
            try:
                if isinstance(out, torch.Tensor) and orig_ch is not None and out.ndim >= 2 and out.shape[1] != orig_ch:
                    out = _trim_channels(out, orig_ch)
                return out
            except Exception:
                return out

        import re
        # Move inputs to inner device if possible
        dev = _inner_device()
        latents = _move(latents, dev)
        control_context = _move(control_context, dev)

        # Tag inputs (best-effort)
        try:
            from toolkit.control_channels import tag_tensor
            if isinstance(latents, torch.Tensor):
                tag_tensor(latents, 'input:latents')
            if isinstance(control_context, torch.Tensor):
                tag_tensor(control_context, 'input:control_context')
            if isinstance(control_context, (list, tuple)):
                for i, c in enumerate(control_context):
                    if isinstance(c, torch.Tensor):
                        tag_tensor(c, f'input:control_context[{i}]')
        except Exception:
            pass

        # Infer expected channels
        try:
            from toolkit.control_util import infer_expected_in_ch
            expected_in = infer_expected_in_ch(self.inner)
        except Exception as e:
            raise RuntimeError(f"Failed to infer expected in-channels from inner adapter: {e}") from e

        # Adapt control and latents via centralized helpers
        try:
            from toolkit.control_channels import adapt_noisy_latents_for_adapter

            def _looks_like_pixel_images_for_wrapper(obj):
                try:
                    if isinstance(obj, torch.Tensor):
                        if obj.ndim == 4:
                            _, c, h, w = obj.shape
                            return (c in (1, 3, 4) and max(h, w) >= 64)
                        if obj.ndim == 5:
                            _, c, f, h, w = obj.shape
                            return (c in (1, 3, 4) and max(h, w) >= 64)
                    if isinstance(obj, (list, tuple)) and len(obj) > 0:
                        return _looks_like_pixel_images_for_wrapper(obj[0])
                except Exception:
                    return False
                return False

            # If control_context appears to be raw pixel images, this indicates a
            # misrouted call: the caller should either provide pre-encoded control
            # latents (VAE-encoded) or route through `StableDiffusion._predict_noise_zimage`
            # which performs auto-encoding when supported. Fail fast with an actionable
            # error to avoid silent mismatches later.
            def _looks_like_pixel_images_for_wrapper(obj):
                try:
                    if isinstance(obj, torch.Tensor):
                        if obj.ndim == 4:
                            _, c, h, w = obj.shape
                            return (c in (1, 3, 4) and max(h, w) >= 64)
                        if obj.ndim == 5:
                            _, c, f, h, w = obj.shape
                            return (c in (1, 3, 4) and max(h, w) >= 64)
                    if isinstance(obj, (list, tuple)) and len(obj) > 0:
                        return _looks_like_pixel_images_for_wrapper(obj[0])
                except Exception:
                    return False
                return False

            if _looks_like_pixel_images_for_wrapper(control_context):
                # Under the strict policy, any raw pixel images reaching the wrapper
                # indicate an upstream misroute; fail fast with an informative error
                # including adapter identity when available.
                adapter_name = getattr(self.inner, 'name_or_path', getattr(self.inner, 'name', None))
                adapter_cfg_dim = getattr(self.inner, 'control_in_dim', None)
                raise RuntimeError(
                    f"VideoXControlnetWrapper received raw pixel images for adapter={adapter_name!r} (control_in_dim={adapter_cfg_dim!r}); provide pre-encoded control latents via sd.encode_control_images or call StableDiffusion._predict_noise_zimage"
                )

            # When we know the expected input channels, adapt the control images to that
            # expected channel count using the central helper. This ensures consistent
            # behavior (padding/slicing/grouped-mean) regardless of whether inputs look
            # like pixel images or already formed control latents.
            # If the caller supplied a precomputed, assembled Z-Image control_context
            # (either explicitly tagged via `assemble_zimage_control_context` or via a
            # high channel-count tensor that does not look like pixel images), bypass the
            # generic `adapt_control_images` path and treat the supplied tensor as the
            # authoritative assembled control_context. This avoids accidental re-adaptation
            # or trimming that can corrupt precomputed contexts.
            skip_trim = False
            is_precomputed_ctx = False
            precomputed_latents = False
            try:
                from toolkit.control_channels import get_tensor_origin
                if isinstance(control_context, torch.Tensor):
                    meta = get_tensor_origin(control_context)
                    if meta is not None:
                        op = meta.get('op')
                        if op == 'assemble_zimage_control_context':
                            is_precomputed_ctx = True
                        elif isinstance(op, str) and op.startswith('precompute:control_latents'):
                            # Explicit signal that these latents were precomputed by the trainer
                            precomputed_latents = True
            except Exception:
                # metadata lookups are best-effort; continue to heuristic checks
                pass

            # Heuristic fallback and consumer-side assembly: if the incoming `control_context`
            # looks like precomputed raw latents (packed frames or multi-frame tensors), perform
            # deterministic assembly here so the adapter gets an authoritative control_context
            # matching its expected `control_in_dim` (e.g., VideoX 33-channel contexts).
            if not is_precomputed_ctx:
                try:
                    # If we have an explicit precompute tag, prefer deterministic assembly behavior
                    if precomputed_latents:
                        try:
                            from toolkit.control_channels import assemble_zimage_control_context, tag_tensor
                            # Normalize 3D single-sample to 4D
                            t = control_context
                            if isinstance(t, torch.Tensor) and t.ndim == 3:
                                t = t.unsqueeze(0)
                                tag_tensor(t, 'precomputed:unsqueezed_single_sample')
                            if isinstance(t, torch.Tensor) and getattr(t, 'ndim', 0) == 5:
                                # collapse frames prior to assembly (mean)
                                t = t.mean(dim=2)
                            # At this point `t` should be 4D latents (B, C, H, W)
                            if isinstance(t, torch.Tensor) and t.ndim == 4:
                                # attempt to assemble into VideoX control_context (33) when adapter expects 33
                                from toolkit.control_util import infer_expected_in_ch
                                expected = infer_expected_in_ch(self.inner) or None
                                if expected is None and getattr(self.inner, 'name_or_path', None) is not None:
                                    # prefer to coerce to 33 for VideoX-like adapters
                                    from toolkit.control_util import adapter_uses_zimage
                                    if adapter_uses_zimage(self.inner, None):
                                        expected = 33
                                if expected == 33:
                                    base = 4
                                    C = int(t.shape[1])
                                    if C == base:
                                        assembled = assemble_zimage_control_context(t, control_in_dim=expected, mask_from=None)
                                        tag_tensor(assembled, 'precomputed:assembled_control_context')
                                        control_context = assembled
                                        is_precomputed_ctx = True
                                    elif C % base == 0:
                                        # If packed channels already correspond to frames that when assembled
                                        # would yield the expected channel count (e.g., C=16 -> 2*C+1=33),
                                        # assemble directly from the packed channels. Otherwise collapse
                                        # to the base latent channels and assemble from there.
                                        if 2 * C + 1 == expected:
                                            assembled = assemble_zimage_control_context(t, control_in_dim=expected, mask_from=None)
                                            tag_tensor(assembled, 'precomputed:assembled_control_context')
                                            control_context = assembled
                                            is_precomputed_ctx = True
                                        else:
                                            B, C2, H, W = t.shape
                                            F = C2 // base
                                            try:
                                                t3 = t.reshape(B, base, F, H, W).mean(dim=2)
                                            except Exception:
                                                try:
                                                    from toolkit.print import print_acc
                                                    print_acc(f"[CONTROLNET] Failed to reshape packed latents for assembly: shape={tuple(t.shape)}")
                                                except Exception:
                                                    pass
                                                raise
                                            assembled = assemble_zimage_control_context(t3, control_in_dim=expected, mask_from=None)
                                            tag_tensor(assembled, 'precomputed:assembled_control_context')
                                            control_context = assembled
                                            is_precomputed_ctx = True
                                else:
                                    # If adapter expects latent base channels (e.g., 4), collapse frames to base
                                    if expected is not None:
                                        C = int(t.shape[1])
                                        if C % expected == 0:
                                            B, C2, H, W = t.shape
                                            F = C2 // expected
                                            t3 = t.view(B, expected, F, H, W).mean(dim=2)
                                            tag_tensor(t3, 'precomputed:collapsed_latents')
                                            control_context = t3
                                            is_precomputed_ctx = True
                        except Exception:
                            # If deterministic assembly fails, fall back to heuristics below
                            pass
                        # End precomputed handling
                        if is_precomputed_ctx:
                            # proceed to regular flow but skip heuristic assembly
                            pass
                    # Infer expected input channels (may be None)
                    expected = expected_in
                    # If adapter looks like a VideoX adapter and expected is unknown, default to 33
                    if expected is None:
                        try:
                            from toolkit.control_util import adapter_uses_zimage
                            if adapter_uses_zimage(self.inner, None):
                                expected = 33
                        except Exception:
                            pass

                    # Normalize 3D single-sample to 4D
                    t = control_context
                    if isinstance(t, torch.Tensor) and t.ndim == 3:
                        try:
                            t = t.unsqueeze(0)
                            from toolkit.control_channels import tag_tensor
                            tag_tensor(t, 'precomputed:unsqueezed_single_sample')
                            control_context = t
                        except Exception:
                            pass

                    if isinstance(control_context, torch.Tensor) and getattr(control_context, 'ndim', 0) in (4, 5):
                        C = int(control_context.shape[1])
                        looks_like_pixel = (control_context.ndim == 4 and C in (1, 3, 4) and max(control_context.shape[-2:]) >= 64)
                        # If it doesn't look like a pixel image and channels indicate packed latents,
                        # attempt deterministic assembly based on the adapter's expectation
                        if not looks_like_pixel and C > 4:
                            from toolkit.control_channels import assemble_zimage_control_context, tag_tensor
                            # Collapse explicit frame dim if present
                            t2 = control_context
                            if t2.ndim == 5:
                                t2 = t2.mean(dim=2)
                            # If adapter expects VideoX assembled context (33), unpack packed frames
                            # and collapse them to base VAE latents (base=4) before assembly
                            if expected == 33:
                                base = 4
                                if int(t2.shape[1]) == base:
                                    # already base latents, just assemble
                                    assembled = assemble_zimage_control_context(t2, control_in_dim=expected, mask_from=None)
                                    tag_tensor(assembled, 'precomputed:assembled_control_context')
                                    control_context = assembled
                                    is_precomputed_ctx = True
                                elif int(t2.shape[1]) % base == 0:
                                    C = int(t2.shape[1])
                                    # If packed channels already correspond to frames that when assembled
                                    # would yield the expected channel count (e.g., C=16 -> 2*C+1=33),
                                    # assemble directly from the packed channels. Otherwise attempt to
                                    # collapse to base channels and assemble.
                                    if 2 * C + 1 == expected:
                                        assembled = assemble_zimage_control_context(t2, control_in_dim=expected, mask_from=None)
                                        tag_tensor(assembled, 'precomputed:assembled_control_context')
                                        control_context = assembled
                                        is_precomputed_ctx = True
                                    else:
                                        B, C2, H, W = t2.shape
                                        F = C2 // base
                                        try:
                                            t3 = t2.reshape(B, base, F, H, W).mean(dim=2)
                                            assembled = assemble_zimage_control_context(t3, control_in_dim=expected, mask_from=None)
                                            tag_tensor(assembled, 'precomputed:assembled_control_context')
                                            control_context = assembled
                                            is_precomputed_ctx = True
                                        except Exception:
                                            try:
                                                from toolkit.print import print_acc
                                                print_acc(f"[CONTROLNET] Failed to reshape packed latents for assembly: shape={tuple(t2.shape)}")
                                            except Exception:
                                                pass
                                            # Fall back to treating as assembled context if we cannot reshape
                                            pass
                            else:
                                # If adapter expects base latent channels (e.g., 4), collapse packed frames
                                # into base latents using mean across frames
                                if expected is not None and C % expected == 0:
                                    B, C2, H, W = t2.shape
                                    F = C2 // expected
                                    try:
                                        t3 = t2.view(B, expected, F, H, W).mean(dim=2)
                                        tag_tensor(t3, 'precomputed:collapsed_latents')
                                        control_context = t3
                                        is_precomputed_ctx = True
                                    except Exception:
                                        pass
                except Exception:
                    pass

            if is_precomputed_ctx:
                try:
                    from toolkit.control_channels import tag_tensor
                    tag_tensor(control_context, 'precomputed:control_context_passthrough')
                except Exception:
                    pass
                adapted_control_context = control_context
                # Do not trim or adapt precomputed assembled contexts; they are authoritative
                skip_trim = True
            else:
                if expected_in is None:
                    adapted_control_context = control_context
                else:
                    # Strict policy: the wrapper must not attempt to adapt raw pixel images
                    # or apply lossy heuristics. Upstream should either provide pre-encoded
                    # control latents or route through StableDiffusion._predict_noise_zimage
                    # which performs auto-encoding.
                    if _looks_like_pixel_images_for_wrapper(control_context):
                        adapter_name = getattr(self.inner, 'name_or_path', getattr(self.inner, 'name', None))
                        adapter_cfg_dim = getattr(self.inner, 'control_in_dim', None)
                        raise RuntimeError(
                            f"VideoXControlnetWrapper received raw pixel images for adapter={adapter_name!r} (control_in_dim={adapter_cfg_dim!r}); provide pre-encoded control latents via sd.encode_control_images or call StableDiffusion._predict_noise_zimage"
                        )

                    # Helper to process a single latent tensor (3D/4D/5D allowed)
                    def _process_latent_tensor(t: torch.Tensor):
                        # collapse frames if present
                        if t.ndim == 5:
                            t_proc = t.mean(dim=2)
                        else:
                            t_proc = t
                        # normalize to 4D [B,C,H,W]
                        if t_proc.ndim == 3:
                            t_proc = t_proc.unsqueeze(0)
                        if t_proc.ndim != 4:
                            raise RuntimeError(f"Unsupported control latent ndim={t_proc.ndim}")

                        C = int(t_proc.shape[1])

                        # If it already matches expected channels, accept it unchanged
                        if C == expected_in:
                            return t_proc

                        # If expected is 33 (VideoX assembled context), assemble from base latents
                        if expected_in == 33:
                            base = 4
                            if C == base:
                                from toolkit.control_channels import assemble_zimage_control_context, tag_tensor
                                assembled = assemble_zimage_control_context(t_proc, control_in_dim=expected_in, mask_from=None)
                                tag_tensor(assembled, 'precomputed:assembled_control_context')
                                return assembled
                            if C % base == 0:
                                # If packed channels directly map to assembled channels
                                if 2 * C + 1 == expected_in:
                                    from toolkit.control_channels import assemble_zimage_control_context, tag_tensor
                                    assembled = assemble_zimage_control_context(t_proc, control_in_dim=expected_in, mask_from=None)
                                    tag_tensor(assembled, 'precomputed:assembled_control_context')
                                    return assembled
                                # Otherwise, collapse to base latents then assemble
                                B, C2, H, W = t_proc.shape
                                F = C2 // base
                                try:
                                    t3 = t_proc.reshape(B, base, F, H, W).mean(dim=2)
                                except Exception:
                                    raise RuntimeError(f"Failed to reshape packed latents for assembly: shape={tuple(t_proc.shape)}")
                                from toolkit.control_channels import assemble_zimage_control_context, tag_tensor
                                assembled = assemble_zimage_control_context(t3, control_in_dim=expected_in, mask_from=None)
                                tag_tensor(assembled, 'precomputed:assembled_control_context')
                                return assembled
                            raise RuntimeError(f"Control latent channels ({C}) incompatible with expected_in=33")

                        # For non-33 expected channels, allow collapsing frames when channel count is multiple
                        if C % expected_in == 0:
                            B, C2, H, W = t_proc.shape
                            F = C2 // expected_in
                            t3 = t_proc.view(B, expected_in, F, H, W).mean(dim=2)
                            from toolkit.control_channels import tag_tensor
                            tag_tensor(t3, 'precomputed:collapsed_latents')
                            return t3

                        raise RuntimeError(f"Control latent channels ({C}) do not match expected_in={expected_in}; wrapper will not adapt control images")

                    # Process tensor or lists deterministically
                    if isinstance(control_context, torch.Tensor):
                        adapted_control_context = _process_latent_tensor(control_context)
                    elif isinstance(control_context, (list, tuple)):
                        adapted_list = []
                        for item in control_context:
                            if not isinstance(item, torch.Tensor):
                                raise RuntimeError("Unsupported control_context element type; expected torch.Tensor")
                            adapted_list.append(_process_latent_tensor(item))
                        adapted_control_context = adapted_list
                    else:
                        raise RuntimeError("Unsupported control_context type; expected tensor or list of tensors")

            # Do NOT attempt to adapt noisy latents based on the control image channel count.
            # `expected_in` is the control_context channel expectation, not the latent channels.
            # Strict parity: do not modify latents here — let the caller/outer pipeline provide
            # correctly-shaped latents. This avoids heuristics that could corrupt the intended
            # deterministic codepath.
            latents_for_inner = latents

        except Exception as e:
            raise RuntimeError(f"Failed to adapt control_images via toolkit.control_channels.adapt_control_images: {e}") from e

        # Pre-call shapes/logging (best-effort)
        try:
            from toolkit.control_channels import format_origin
            from toolkit.print import print_acc
            def _shape_of(obj):
                if isinstance(obj, torch.Tensor):
                    return tuple(obj.shape)
                if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
                    return tuple(obj[0].shape)
                return None
            lat_shape = _shape_of(latents_for_inner)
            ctrl_shape = _shape_of(adapted_control_context)
            try:
                if isinstance(latents_for_inner, torch.Tensor):
                    tag_tensor(latents_for_inner, 'call:latents_for_inner')
                if isinstance(adapted_control_context, torch.Tensor):
                    tag_tensor(adapted_control_context, 'call:control_context')
            except Exception:
                pass
            print_acc(f"[CONTROLNET-REROUTE] pre-call shapes latents={lat_shape} control_context={ctrl_shape} expected_in={expected_in}")
        except Exception:
            pass

        # Normalize channels when expected known
        try:
            if expected_in is not None:
                adapted_control_context = _trim_channels(adapted_control_context, expected_in)
                def _verify_channels(obj):
                    if isinstance(obj, torch.Tensor):
                        if obj.ndim >= 2 and obj.shape[1] != expected_in:
                            raise RuntimeError(f"Control context channels ({obj.shape[1]}) do not match expected ({expected_in}) after adaptation")
                    if isinstance(obj, (list, tuple)):
                        for x in obj:
                            _verify_channels(x)
                _verify_channels(adapted_control_context)
        except Exception as e:
            raise RuntimeError(f"Failed to normalize control_context to expected_in={expected_in}: {e}") from e

        # Attempt common call signatures using the adapter's callable interface
        callable_inner = self.inner if callable(self.inner) else getattr(self.inner, 'forward', None)
        if callable_inner is None:
            raise RuntimeError("Inner adapter is not callable and has no .forward method")

        # Strict VideoX parity: format latents into the transformer input (list-of-samples),
        # and make a single direct call to the inner adapter. Do NOT attempt adaptive
        # fallbacks here - if shapes mismatch, let the inner error bubble up.
        try:
            inner_dtype = getattr(self.inner, 'dtype', None) or getattr(self.inner, 'torch_dtype', None)
            if isinstance(inner_dtype, torch.dtype) and isinstance(latents, torch.Tensor):
                latents = latents.to(inner_dtype)
        except Exception:
            pass

        if isinstance(latents, torch.Tensor):
            if latents.ndim == 4:
                latent_model_input = latents.unsqueeze(2)  # [B, C, 1, H, W]
            else:
                latent_model_input = latents
            latent_list = list(latent_model_input.unbind(dim=0))
        else:
            latent_list = list(latents)

        callable_inner = self.inner if callable(self.inner) else getattr(self.inner, 'forward', None)
        if callable_inner is None:
            raise RuntimeError("Inner adapter is not callable and has no .forward method")

        # VideoX parity: single call signature and no internal retries
        import inspect
        try:
            target_fn = getattr(self.inner, 'forward', callable_inner)
            sig = inspect.signature(target_fn)
            params = sig.parameters
        except Exception:
            params = {}

        call_kwargs = {}
        # Strict: prefer VideoX-style `control_context` parameter. If the
        # adapter doesn't expose `control_context`, attempt a safe runtime shim
        # which maps common legacy names (e.g., `controlnet_cond`) to the
        # expected `control_context`. Persist the shim to `self.inner` if it
        # succeeds so subsequent calls avoid repeated work.
        if 'control_context' in params:
            call_kwargs['control_context'] = adapted_control_context
        else:
            shim_applied = False
            try:
                from toolkit.controlnet_compat import ControlNetLegacyAdapter
                shim = ControlNetLegacyAdapter(self.inner)
                target_fn = getattr(shim, 'forward', shim)
                sig = inspect.signature(target_fn)
                params = sig.parameters
                if 'control_context' in params:
                    # persist shim for later calls
                    self.inner = shim
                    callable_inner = shim
                    # ensure we pass the adapted control_context into the shim call
                    call_kwargs['control_context'] = adapted_control_context
                    try:
                        from toolkit.print import print_acc
                        print_acc('[CONTROLNET] Applied legacy shim inside VideoXControlnetWrapper for runtime compatibility')
                    except Exception:
                        pass
                    shim_applied = True
            except Exception:
                shim_applied = False

            if not shim_applied:
                raise RuntimeError("Adapter is not VideoX-compatible: missing required parameter 'control_context'. Ensure you use a Z-Image / VideoX-style adapter for strict routing.")

        # map conditioning/scale name
        for scale_name in ('control_context_scale', 'conditioning_scale', 'controlnet_conditioning_scale', 'controlnet_conditioning_scale', 'controlnet_scale', 'scale'):
            if scale_name in params:
                call_kwargs[scale_name] = conditioning_scale
                break

        # First try calling with a tensor-form `latents_for_inner` (works with many adapters)
        pos_args_tensor = [latents_for_inner if 'latents_for_inner' in locals() else latents, timestep]
        pos_args_list = [latent_list, timestep]

        # remember original channel count so we can restore outputs to the expected shape
        orig_ch = None
        try:
            if isinstance(latents, torch.Tensor):
                orig_ch = int(latents.shape[1])
            elif isinstance(latents, (list, tuple)) and len(latents) > 0 and isinstance(latents[0], torch.Tensor):
                orig_ch = int(latents[0].shape[1])
        except Exception:
            orig_ch = None

        import inspect as _inspect

        # If adapter requires a positional `encoder_hidden_states` arg with no default,
        # this likely indicates a Flux1/hidden-states style adapter which is not
        # compatible with strict VideoX routing. Fail with an actionable error.
        try:
            if 'encoder_hidden_states' in params:
                p = params['encoder_hidden_states']
                if p.default is _inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                    raise RuntimeError("Adapter appears to require 'encoder_hidden_states' (Flux1-style). For strict VideoX/Z-Image routing, use an adapter that accepts 'control_context' and does not require 'encoder_hidden_states'.")
        except RuntimeError:
            raise
        except Exception:
            # ignore signature inspection failures
            pass

        # Perform the single, strict call using VideoX list-of-samples parity (latent list form)
        try:
            raw_out = callable_inner(*pos_args_list, *args, **call_kwargs, **kwargs)
        except Exception as e:
            # Let errors bubble up; include some context
            raise RuntimeError(f"VideoXControlnetWrapper: calling inner adapter failed: {e}") from e

        # Restore output channels to the original latents channel count when possible
        try:
            return _maybe_restore_output(raw_out, orig_ch)
        except Exception:
            return raw_out

        try:
            # Some implementations expect signature (latents, timestep, cap_feats, control_context)
            # Call without passing conditioning_scale as kw to avoid duplicate-kw errors
            return self.inner(latents_for_inner if 'latents_for_inner' in locals() else latents, timestep, None, control_context, *args, **kwargs)
        except Exception as e:
            last_exc = e
            # Include origin metadata for easier debugging
            try:
                from toolkit.control_channels import format_origin, get_tensor_origin
                ctrl_origin = format_origin(control_context) if isinstance(control_context, torch.Tensor) else ','.join((format_origin(c) if isinstance(c, torch.Tensor) else str(type(c)) for c in control_context)) if isinstance(control_context, (list, tuple)) else 'unknown'
            except Exception:
                ctrl_origin = 'unknown'
            try:
                lat_origin = format_origin(latents_for_inner) if isinstance(latents_for_inner, torch.Tensor) else ('list' if isinstance(latents_for_inner, (list, tuple)) else 'unknown')
            except Exception:
                lat_origin = 'unknown'

            # Try to detect offending tensor by inspecting the exception message
            offending = 'unknown'
            try:
                msg = str(last_exc)
                import re
                m = re.search(r"expected input\[.*\] to have (\d+) channels, but got (\d+) channels", msg)
                if m:
                    got_ch = int(m.group(2))
                    def _ch_of(obj):
                        if isinstance(obj, torch.Tensor) and obj.ndim >= 2:
                            return int(obj.shape[1])
                        if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
                            return int(obj[0].shape[1])
                        return None
                    lat_ch = _ch_of(latents_for_inner)
                    ctrl_ch = _ch_of(control_context)
                    if lat_ch == got_ch:
                        offending = 'latents'
                    elif ctrl_ch == got_ch:
                        offending = 'control_context'
            except Exception:
                offending = 'unknown'

            # Emit detailed diagnostics to help debugging in failing job logs
            try:
                from toolkit.print import print_acc
                print_acc("[CONTROLNET-REROUTE] Diagnostics disabled in training mode; original error will be raised")
                print_acc(f"[CONTROLNET-REROUTE] last_exc={last_exc} offending={offending} latents_shape={getattr(latents_for_inner,'shape',None)} control_context_shape={getattr(control_context,'shape',None)}")
            except Exception:
                pass

            raise TypeError(f"VideoXControlnetWrapper: inner controlnet rejected known call signatures: {last_exc}; offending={offending}; control_context_origin={ctrl_origin}; latents_origin={lat_origin}") from last_exc

        # Preferred: (latents, timestep, control_context, conditioning_scale=...)
        try:
            return _attempt_call(lambda control_context=control_context: self.inner(latents_for_inner if 'latents_for_inner' in locals() else latents, timestep, control_context, conditioning_scale=conditioning_scale, *args, **kwargs))
        except TypeError:
            pass
        except Exception:
            # fall through to other call signatures; we'll include the final error if all fail
            last_exc = None

        # Fallback: (latents, timestep, control_context)
        try:
            return _attempt_call(lambda control_context=control_context: self.inner(latents_for_inner if 'latents_for_inner' in locals() else latents, timestep, control_context, *args, **kwargs))
        except TypeError:
            pass
        except Exception:
            last_exc = None

        # Some implementations expect kwargs like `controlnet_cond`
        try:
            return _attempt_call(lambda control_context=control_context: self.inner(latents_for_inner if 'latents_for_inner' in locals() else latents, timestep, controlnet_cond=control_context, conditioning_scale=conditioning_scale, *args, **kwargs))
        except TypeError:
            pass
        except Exception:
            last_exc = None

        # Last fallback: some implementations expect (latents, timestep, cap_feats, control_context)
        try:
            return _attempt_call(lambda control_context=control_context: self.inner(latents_for_inner if 'latents_for_inner' in locals() else latents, timestep, None, control_context, conditioning_scale=conditioning_scale, *args, **kwargs))
        except Exception as e:
            # Include origin metadata for easier debugging
            try:
                from toolkit.control_channels import format_origin, get_tensor_origin
                ctrl_origin = format_origin(control_context) if isinstance(control_context, torch.Tensor) else ','.join((format_origin(c) if isinstance(c, torch.Tensor) else str(type(c)) for c in control_context)) if isinstance(control_context, (list, tuple)) else 'unknown'
            except Exception:
                ctrl_origin = 'unknown'
            try:
                lat_origin = format_origin(latents_for_inner) if isinstance(latents_for_inner, torch.Tensor) else ('list' if isinstance(latents_for_inner, (list, tuple)) else 'unknown')
            except Exception:
                lat_origin = 'unknown'
            # attempt to detect offending tensor by message
            offending = 'unknown'
            try:
                msg = str(e)
                import re
                m = re.search(r"expected input\[.*\] to have (\d+) channels, but got (\d+) channels", msg)
                if m:
                    got_ch = int(m.group(2))
                    def _ch_of(obj):
                        if isinstance(obj, torch.Tensor) and obj.ndim >= 2:
                            return int(obj.shape[1])
                        if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
                            return int(obj[0].shape[1])
                        return None
                    lat_ch = _ch_of(latents_for_inner)
                    ctrl_ch = _ch_of(control_context)
                    if lat_ch == got_ch:
                        offending = 'latents'
                    elif ctrl_ch == got_ch:
                        offending = 'control_context'
            except Exception:
                offending = 'unknown'
            raise TypeError(f"VideoXControlnetWrapper: inner controlnet rejected known call signatures: {e}; offending={offending}; control_context_origin={ctrl_origin}; latents_origin={lat_origin}") from e
    # Delegate common nn.Module helpers to the inner module where appropriate
    def to(self, *args, **kwargs):
        try:
            # If inner supports .to(), forward the call
            self.inner.to(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Forwarding .to to inner adapter failed: {e}") from e
        return self

    def parameters(self, *args, **kwargs):
        try:
            return self.inner.parameters(*args, **kwargs)
        except Exception:
            return super().parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        try:
            return self.inner.state_dict(*args, **kwargs)
        except Exception:
            return super().state_dict(*args, **kwargs)

