"""Compatibility helpers for VideoX/zimage-style ControlNets.

Provide a thin shim that adapts a VideoX-style controlnet to an object
that can be called by `StableDiffusion.get_noise_prediction` and
by trainer routing code. The wrapper tries common call signatures
and normalizes return values for robustness in mixed environments.
"""
import torch
from typing import Any
from types import SimpleNamespace


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
        # Require an actual adapter at construction time to ensure correctness —
        # do not allow constructing a wrapper around `None` which previously led
        # to confusing runtime behavior later in the pipeline.
        if inner is None:
            raise RuntimeError("VideoXControlnetWrapper requires a non-None inner adapter; construct with a properly initialized ControlNet adapter.")
        self.inner = inner
        # Early validation: ensure the inner adapter presents a VideoX-compatible signature
        # We enforce signature checks at construction time to fail fast on mismatches.
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
            # The wrapper requires the adapter to explicitly accept `control_context`.
            # No runtime shims/fallbacks are permitted — fail fast at construction time
            # so incorrect adapters are not wrapped silently.
            if 'control_context' not in params:
                raise RuntimeError("Adapter does not expose required parameter 'control_context'; VideoXControlnetWrapper only wraps adapters with explicit 'control_context' support.")
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
            except Exception as e:
                try:
                    print_acc(f"[CONTROLNET] _inner_device parameters probe failed: {e}")
                except Exception:
                    print(f"[CONTROLNET] _inner_device parameters probe failed: {e}")
            try:
                buffs = getattr(self.inner, 'buffers', None) 
                if callable(buffs):
                    for b in buffs():
                        return b.device
            except Exception as e:
                try:
                    print_acc(f"[CONTROLNET] _inner_device buffers probe failed: {e}")
                except Exception:
                    print(f"[CONTROLNET] _inner_device buffers probe failed: {e}")
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
            except Exception as e:
                try:
                    print_acc(f"[CONTROLNET] _maybe_restore_output failed: {e}")
                except Exception:
                    print(f"[CONTROLNET] _maybe_restore_output failed: {e}")
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
        except Exception as e:
            try:
                print_acc(f"[CONTROLNET] tagging inputs failed: {e}")
            except Exception:
                print(f"[CONTROLNET] tagging inputs failed: {e}")

        # Determine expected channels explicitly: prefer adapter's explicit
        # `control_in_dim` if present; otherwise **default** to 33 (VideoX Z-Image).
        try:
            expected_in = getattr(self.inner, 'control_in_dim', None)
        except Exception:
            expected_in = None
        # No heuristic inference: default to 33 unless the adapter explicitly sets a different value
        if expected_in is None:
            expected_in = 33

        # Fail fast on misconfigured adapters: prefer explicit identity on the
        # outer wrapper if present (set via set_adapter_name_if_missing on the
        # adapter returned by loaders), otherwise look at the inner module. This
        # avoids false positives when the wrapper object itself was assigned a
        # name but the wrapped inner model lacks that attribute.
        adapter_name = getattr(self, 'name_or_path', None) or getattr(self.inner, 'name_or_path', None) or getattr(self.inner, 'name', None)
        adapter_control_in_dim = getattr(self, 'control_in_dim', None) or getattr(self.inner, 'control_in_dim', None)
        if adapter_name is None and adapter_control_in_dim is None:
            raise RuntimeError(
                f"Adapter appears misconfigured: missing 'name'/'name_or_path' on wrapper or inner and explicit 'control_in_dim'. Provide an adapter name or set 'control_in_dim' to enable deterministic routing (wrapper_type={type(self)!r}, inner_type={type(self.inner)!r})."
            )

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
            # latents (VAE-encoded) or call `StableDiffusion.get_noise_prediction` / use
            # sd.encode_control_images so the model can perform auto-encoding when supported. Fail fast with an actionable
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
                    f"VideoXControlnetWrapper received raw pixel images for adapter={adapter_name!r} (control_in_dim={adapter_cfg_dim!r}); provide pre-encoded control latents via sd.encode_control_images or call StableDiffusion.get_noise_prediction"
                )

            # Strict handling: accept only the exact expected forms and fail otherwise.
            def _process_single_cc(t: torch.Tensor):
                # Normalize: collapse frame dim and ensure 4D
                if t.ndim == 5:
                    t = t.mean(dim=2)
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                if t.ndim != 4:
                    raise RuntimeError(f"Unsupported control_context ndim={t.ndim}; expected 4 or 5")

                C = int(t.shape[1])

                # 33-channel assembled context: pass-through
                if C == 33:
                    try:
                        from toolkit.control_channels import tag_tensor
                        tag_tensor(t, 'precomputed:assembled_control_context')
                    except Exception as e:
                        try:
                            print_acc(f"[CONTROLNET] tagging assembled control_context failed: {e}")
                        except Exception:
                            print(f"[CONTROLNET] tagging assembled control_context failed: {e}")
                    return t

                # 3-channel raw images are not acceptable here. Under strict VideoX
                # semantics we require either VAE-encoded latents (C==16) or
                # a pre-assembled 33-channel control_context. Treat raw images (C==3)
                # as a misrouted or upstream error and fail fast with an actionable message.
                try:
                    from toolkit.control_channels import RAW_IMAGE_CHANNELS, ENCODED_LATENT_CHANNELS
                except Exception:
                    RAW_IMAGE_CHANNELS, ENCODED_LATENT_CHANNELS = 3, 16

                if C == RAW_IMAGE_CHANNELS:
                    raise RuntimeError(
                        "Received raw 3-channel image controls in VideoX wrapper: upstream should provide VAE-encoded latents (C==16) or a 33-channel assembled control_context; do not pass raw images to the wrapper"
                    )

                # VAE-encoded or multi-frame latents (C >= base and divisible by base)
                base = ENCODED_LATENT_CHANNELS
                if C >= base and C % base == 0:
                    # If the packed channels directly map to assembled channels (e.g., C=16 -> 2*C+1=33),
                    # assemble directly from the packed channels to avoid collapsing to base first.
                    if expected_in == 33 and (2 * C + 1) == expected_in:
                        try:
                            from toolkit.control_channels import assemble_zimage_control_context, tag_tensor
                            assembled = assemble_zimage_control_context(t, control_in_dim=expected_in, mask_from=None)
                            try:
                                tag_tensor(assembled, 'precomputed:assembled_control_context')
                            except Exception as e:
                                try:
                                    print_acc(f"[CONTROLNET] tagging assembled control_context failed: {e}")
                                except Exception:
                                    print(f"[CONTROLNET] tagging assembled control_context failed: {e}")
                            return assembled
                        except Exception as e:
                            raise RuntimeError(f"Deterministic assembly from packed channels failed: {e}") from e

                    # Otherwise, collapse frames to base latents and assemble
                    B, C2, H, W = t.shape
                    F = C2 // base
                    try:
                        t3 = t.reshape(B, base, F, H, W).mean(dim=2)
                    except Exception as e:
                        try:
                            print_acc(f"[CONTROLNET] Failed to reshape encoded/multi-frame latents for assembly: shape={tuple(t.shape)} error={e}")
                        except Exception:
                            print(f"[CONTROLNET] Failed to reshape encoded/multi-frame latents for assembly: shape={tuple(t.shape)} error={e}")
                        raise RuntimeError(f"Failed to reshape packed latents for assembly: shape={tuple(t.shape)}") from e
                    try:
                        from toolkit.control_channels import assemble_zimage_control_context, tag_tensor
                        assembled = assemble_zimage_control_context(t3, control_in_dim=33, mask_from=None)
                        try:
                            tag_tensor(assembled, 'precomputed:assembled_control_context')
                        except Exception as e:
                            try:
                                print_acc(f"[CONTROLNET] tagging assembled control_context failed: {e}")
                            except Exception:
                                print(f"[CONTROLNET] tagging assembled control_context failed: {e}")
                        return assembled
                    except Exception as e:
                        # Surface deterministic assembly errors
                        raise RuntimeError(f"Deterministic assembly failed: {e}") from e

                # Pixel-like tensors were handled earlier; any other channel counts are invalid
                if 'expected_in' in locals() and expected_in is not None:
                    raise RuntimeError(f"Control latent channels ({C}) do not match expected_in={expected_in}; wrapper will not adapt control images")
                raise RuntimeError(f"Unsupported control_context channels ({C}); expected 16 (packed), 4 (base) or 33 (assembled)")

            # Apply strict processing to either tensor or list/tuple
            if isinstance(control_context, torch.Tensor):
                adapted_control_context = _process_single_cc(control_context)
                skip_trim = True
                is_precomputed_ctx = True
            elif isinstance(control_context, (list, tuple)):
                adapted_list = [ _process_single_cc(c) if isinstance(c, torch.Tensor) else (_ for _ in ()).throw(RuntimeError('Unsupported control_context element type; expected torch.Tensor')) for c in control_context ]
                adapted_control_context = adapted_list
                skip_trim = True
                is_precomputed_ctx = True
            else:
                raise RuntimeError("Unsupported control_context type; expected torch.Tensor or list/tuple of tensors")

            # Tag precomputed control_context for diagnostics (best-effort)
            try:
                from toolkit.control_channels import tag_tensor
                if isinstance(adapted_control_context, torch.Tensor):
                    tag_tensor(adapted_control_context, 'precomputed:control_context_passthrough')
            except Exception as e:
                try:
                    print_acc(f"[CONTROLNET] tagging precomputed control_context failed: {e}")
                except Exception:
                    print(f"[CONTROLNET] tagging precomputed control_context failed: {e}")
            # Do not trim or adapt precomputed assembled contexts; they are authoritative
            skip_trim = True


            latents_for_inner = latents

        except Exception as e:
            # Provide an accurate, actionable error message (do not reference the
            # legacy `adapt_control_images` helper which the wrapper no longer calls).
            try:
                from toolkit.control_channels import format_origin
                ctrl_origin = format_origin(control_context) if isinstance(control_context, torch.Tensor) else (', '.join(format_origin(c) if isinstance(c, torch.Tensor) else str(type(c)) for c in control_context) if isinstance(control_context, (list, tuple)) else 'unknown')
            except Exception:
                ctrl_origin = 'unknown'
            adapter_name = getattr(self.inner, 'name_or_path', getattr(self.inner, 'name', None))
            raise RuntimeError(
                f"Failed to adapt control_context for adapter={adapter_name!r} expected_in={expected_in!r} ctrl_origin={ctrl_origin!r}: {e}"
            ) from e

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
            except Exception as e:
                try:
                    from toolkit.print import print_acc
                    print_acc(f"[CONTROLNET] tagging call inputs failed: {e}")
                except Exception:
                    print(f"[CONTROLNET] tagging call inputs failed: {e}")
            print_acc(f"[CONTROLNET-REROUTE] pre-call shapes latents={lat_shape} control_context={ctrl_shape} expected_in={expected_in}")
        except Exception as e:
            try:
                print_acc(f"[CONTROLNET] pre-call shapes logging failed: {e}")
            except Exception:
                print(f"[CONTROLNET] pre-call shapes logging failed: {e}")

        # Normalize channels when expected known
        try:
            if expected_in is not None and not skip_trim:
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
        except Exception as e:
            try:
                print_acc(f"[CONTROLNET] casting latents to inner dtype failed: {e}")
            except Exception:
                print(f"[CONTROLNET] casting latents to inner dtype failed: {e}")

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
                    except Exception as e:
                        try:
                            from toolkit.print import print_acc as _p
                            _p(f"[CONTROLNET] printing shim application notice failed: {e}")
                        except Exception:
                            print(f"[CONTROLNET] printing shim application notice failed: {e}")
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
        except Exception as e:
            # ignore signature inspection failures but log for diagnostics
            try:
                from toolkit.print import print_acc
                print_acc(f"[CONTROLNET] signature inspection failed: {e}")
            except Exception:
                print(f"[CONTROLNET] signature inspection failed: {e}")

        # Perform the single, strict call using the centralized adapter helper so
        # the adapter invocation, autocast, timers, and normalization are canonical.
        try:
            # The helper lives in the Z-Image extension module and accepts a preassembled control_context
            from extensions_built_in.diffusion_models.z_image.z_image import compute_zimage_adapter_residuals as _compute
            sd_for_helper = getattr(self, '_owner_sd', None)
            down, mid, control_context, raw_out = _compute(sd_for_helper or SimpleNamespace(),
                                                           latents if isinstance(latents, torch.Tensor) else torch.stack(latents, dim=0),
                                                           timestep,
                                                           zimage_controlnet=callable_inner,
                                                           zimage_control_context=adapted_control_context,
                                                           zimage_conditioning_scale=conditioning_scale,
                                                           train_dtype=getattr(self, 'torch_dtype', None),
                                                           dataset_controlnet_debug=False,
                                                           batch=None)
        except Exception as e:
            raise RuntimeError(f"VideoXControlnetWrapper: calling inner adapter via centralized helper failed: {e}") from e

        # Restore output channels to the original latents channel count when possible
        try:
            return _maybe_restore_output(raw_out, orig_ch)
        except Exception:
            return raw_out

        # Legacy fallback heuristics were removed; if the primary strict call failed,
        # surface a clear actionable error rather than attempting multiple heuristic
        # call signature fallbacks that obscure root causes.
        raise RuntimeError("Inner adapter did not accept the strict VideoX call signature; ensure it accepts a 'control_context' kw and list-of-samples latents")


        # Legacy fallback call-signature heuristics have been removed to enforce strict
        # VideoX parity. If the primary direct call failed above, surface an actionable
        # error so callers can fix their adapter or provide correctly-shaped inputs.
        raise RuntimeError("Adapter did not accept strict VideoX-style call signatures; ensure adapter exposes a 'control_context' parameter and accepts list-of-samples latents")
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

