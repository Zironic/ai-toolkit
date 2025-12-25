"""Lightweight model utilities for inference tasks.

This module provides helpers to resolve local model paths and to load models
in a restricted, inference-only mode. The goal is to reuse the same
resolution and loading behavior used by training, but avoid the heavy or
stateful operations (LoRA fusion, persistent saves, conversion side-effects)
that are unnecessary or undesired during short-lived evaluation runs.
"""
from typing import Optional, Dict, Any
import os

from .paths import MODELS_PATH, ORIG_CONFIGS_ROOT, DIFFUSERS_CONFIGS_ROOT
from .config_modules import ModelConfig
from .stable_diffusion_model import StableDiffusion


# Conservative whitelist of ModelConfig fields allowed for evaluation jobs.
# Avoid allowing fields that trigger training-only behavior (lora fusion, save paths, etc.).
ALLOWED_MODEL_CONFIG_KEYS = {
    'name_or_path',
    'arch',
    'low_vram',
    'quantize',
    'qtype',
    'dtype',
    'vae_dtype',
    'vae_path',
    'te_dtype',
    # add other safe fields if needed
} 


def resolve_local_model_path(name_or_path: str) -> Optional[str]:
    """Return a local path if the model appears to be downloaded locally.

    Checks a set of common directories used by the project (MODELS_PATH,
    toolkit/diffusers_configs, toolkit/orig_configs) for a folder matching
    the given name_or_path or the repository name component.
    """
    if not name_or_path:
        return None

    # if it's already a path on disk, accept it
    if os.path.exists(name_or_path):
        return name_or_path

    candidates = []
    if MODELS_PATH:
        candidates.append(os.path.join(MODELS_PATH, name_or_path))
        if '/' in name_or_path:
            candidates.append(os.path.join(MODELS_PATH, name_or_path.split('/')[-1]))
        candidates.append(os.path.join(MODELS_PATH, name_or_path.replace('/', '_')))

    candidates.extend([
        os.path.join(DIFFUSERS_CONFIGS_ROOT, name_or_path),
        os.path.join(DIFFUSERS_CONFIGS_ROOT, name_or_path.split('/')[-1] if '/' in name_or_path else name_or_path),
        os.path.join(ORIG_CONFIGS_ROOT, name_or_path),
        os.path.join(ORIG_CONFIGS_ROOT, name_or_path.split('/')[-1] if '/' in name_or_path else name_or_path),
    ])

    for c in candidates:
        try:
            if c and os.path.exists(c):
                return c
        except Exception:
            continue
    return None


def sanitize_model_config(cfg: Dict[str, Any], apply_lora: bool = False) -> Dict[str, Any]:
    """Return a sanitized copy of a model config dict containing only allowed keys.

    If apply_lora=True, also allow assistant/inference lora keys so they can be applied
    in-memory for evaluation.
    """
    if not cfg:
        return {}
    allowed = set(ALLOWED_MODEL_CONFIG_KEYS)
    if apply_lora:
        allowed = allowed.union({'lora_path', 'assistant_lora_path', 'inference_lora_path'})
    return {k: v for k, v in cfg.items() if k in allowed}


def load_model_for_inference(model_or_name: Any, device: str = 'cpu', dtype: str = 'bf16', apply_lora: bool = False) -> StableDiffusion:
    """Load a StableDiffusion instance suitable for inference/evaluation.

    model_or_name can be either a `str` (name_or_path) or a dict of ModelConfig
    kwargs (e.g., parsed from a JSON file). For dicts, only whitelisted keys
    are used. The returned StableDiffusion instance is loaded (sd.load_model())
    but the loader will avoid doing destructive/side-effectful operations unless
    `apply_lora=True` which allows non-persistent in-memory LoRA/application for
    fidelity with training models.

    Default dtype for inference/eval is `bf16` to match training defaults and
    improve throughput on supported hardware.
    """
    # Build a ModelConfig safely
    if isinstance(model_or_name, dict):
        safe_cfg = sanitize_model_config(model_or_name, apply_lora=apply_lora)
        # ensure required field present
        if 'name_or_path' not in safe_cfg:
            raise ValueError('model_config must include "name_or_path"')
        mc = ModelConfig(**safe_cfg)
    else:
        # string -> try to resolve local path first
        resolved = resolve_local_model_path(str(model_or_name))
        mc = ModelConfig(name_or_path=resolved or str(model_or_name))

    # If the caller provided an explicit dtype for inference and the ModelConfig
    # does not already specify VAE/TE dtypes, set them to the requested dtype.
    # This mirrors the training flow where the training dtype propagates into
    # ModelConfig and avoids mixed-precision mismatches at inference time.
    if dtype:
        try:
            if getattr(mc, 'vae_dtype', None) is None:
                mc.vae_dtype = dtype
        except Exception:
            pass
        try:
            if getattr(mc, 'te_dtype', None) is None:
                mc.te_dtype = dtype
        except Exception:
            pass

    # If apply_lora is False, explicitly disable known stateful lora/refiner fields in case a caller passed them
    if not apply_lora:
        for attr in ['lora_path', 'assistant_lora_path', 'inference_lora_path', 'refiner_name_or_path']:
            if hasattr(mc, attr):
                setattr(mc, attr, None)

    # Choose the appropriate model class (stable diffusion or other registered models)
    from .util.get_model import get_model_class

    ModelClass = get_model_class(mc)
    model_instance = ModelClass(device=device, model_config=mc, dtype=dtype)

    # Load model (may download or instantiate the pipeline)
    model_instance.load_model()

    # Ensure VAE parameters and buffers match the requested dtype to avoid conv bias/input
    # mismatches that can occur when parts of the model remain in float while inputs are fp16/bf16.
    # This is a conservative, non-invasive post-load step restricted to the VAE only.
    import torch

    torch_dtype = None
    try:
        if dtype in ("fp16", "float16"):
            torch_dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype in ("fp32", "float32") or dtype is None:
            torch_dtype = torch.float32
    except Exception:
        torch_dtype = None

    if torch_dtype is not None:
        vae = getattr(model_instance, 'vae', None)
        if vae is not None:
            try:
                # cast parameters
                for p in vae.parameters(recurse=True):
                    try:
                        p.data = p.data.to(torch_dtype)
                        if p.grad is not None:
                            p.grad.data = p.grad.data.to(torch_dtype)
                    except Exception:
                        # best-effort; continue if something can't be cast
                        continue
                # cast buffers (eg running stats)
                for name, buf in vae.named_buffers(recurse=True):
                    try:
                        buf.data = buf.data.to(torch_dtype)
                    except Exception:
                        continue
                # also move the module (safe no-op if already moved)
                try:
                    vae.to(torch_dtype)
                except Exception:
                    # some VAEs require device arg; ignore if this fails
                    pass
            except Exception as e:
                print(f"Warning: failed to cast VAE params/buffers to dtype {dtype}: {e}")

    # If requested, apply LoRA/adapter weights in-memory (non-persistent) after loading
    if apply_lora:
        try:
            if getattr(mc, 'assistant_lora_path', None):
                # For models like ZImage, load_training_adapter will be invoked inside load_model
                # if assistant_lora_path was part of the model_config passed. If not, we try a best-effort
                # hook here in case some models need an explicit call.
                if hasattr(model_instance, 'load_training_adapter'):
                    # Some models expect the transformer instance; attempt to call if available
                    try:
                        transformer = getattr(model_instance, 'transformer', None) or getattr(model_instance, 'model', None)
                        if transformer is not None:
                            model_instance.load_training_adapter(transformer)
                            print(f"Applied assistant LO-RA in-memory for {mc.name_or_path}")
                    except Exception as e:
                        print(f"Warning: failed to apply in-memory assistant LoRA via hook: {e}")
        except Exception as e:
            print(f"Warning: failed to apply in-memory LoRA: {e}")

    return model_instance
