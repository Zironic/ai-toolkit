"""Fixed SKC projector-vector injection for Krea2 training and sampling.

``skc3vo`` is a rank-1 adapter that touches ``txtfusion.projector`` and nothing
else: it adds ``(projector_input * vector).sum(-1) * strength`` to the projector
output, where ``vector`` is 12 values along the projector's slot axis. This
module loads that 12-vector from an skc3vo / z0-style safetensors file and
installs it as a forward hook, so the perturbation is present in every model
forward -- during training (so a correction LoRA learns to compensate for it)
and during sampling previews.

The vector is FIXED (no grad). Gradients from the training loss flow through the
additive delta to the trainable LoRA on the downstream blocks, never to the
vector. The delta math is identical to ``projector_delta_forward`` in
``scripts/capture_krea_txtfusion_forward.py`` (the same injection the probe /
fork scripts use), so a run trained here reproduces the projector perturbation
those experiments measured -- the 12-slot axis is the LAST axis of the projector
input, reduced with ``sum(-1, keepdim=True)``.
"""
from typing import Tuple

import torch


def extract_projector_vector(state_dict: dict) -> Tuple[str, torch.Tensor]:
    """Return (source_key, 12-value vector) from an skc3vo/z0 state dict.

    Handles both the ``.diff`` single-tensor form (z0jglf) and the rank-1
    ``lora_A``/``lora_B`` pair (skc3vo), under any of the known key prefixes.
    """
    diff_keys = [
        "diffusion_model.txtfusion.projector.diff",
        "transformer.text_fusion.projector.diff",
        "txtfusion.projector.diff",
    ]
    for key in diff_keys:
        if key in state_dict:
            vector = state_dict[key].detach().float().reshape(-1)
            if vector.numel() != 12:
                raise ValueError(
                    f"{key} must contain 12 projector values, got {tuple(state_dict[key].shape)}"
                )
            return key, vector

    pairs = [
        (
            "transformer.text_fusion.projector.lora_A.weight",
            "transformer.text_fusion.projector.lora_B.weight",
        ),
        (
            "diffusion_model.txtfusion.projector.lora_A.weight",
            "diffusion_model.txtfusion.projector.lora_B.weight",
        ),
        (
            "txtfusion.projector.lora_A.weight",
            "txtfusion.projector.lora_B.weight",
        ),
    ]
    for a_key, b_key in pairs:
        if a_key in state_dict and b_key in state_dict:
            a = state_dict[a_key].detach().float()
            b = state_dict[b_key].detach().float()
            delta = torch.matmul(b, a).reshape(-1)
            if delta.numel() != 12:
                raise ValueError(
                    f"{b_key} @ {a_key} must produce 12 projector values, got {tuple(delta.shape)}"
                )
            return f"{b_key} @ {a_key}", delta

    raise ValueError("No txtfusion.projector diff or LoRA A/B pair found")


def make_skc_projector_hook(vector: torch.Tensor, strength: float):
    """Build a forward hook that adds the fixed SKC projector delta.

    Projector input ``x`` has shape ``(..., 2560, 12)``; the delta reduces the
    trailing 12-slot axis and is added to the ``(..., 2560, 1)`` output. This
    mirrors ``projector_delta_forward`` in the probe scripts exactly.
    """
    v = vector.detach().float().reshape(1, 1, 12)
    s = float(strength)

    def hook(module, inputs, output):
        x = inputs[0]
        vv = v.to(device=x.device, dtype=x.dtype)
        delta = (x * vv).sum(dim=-1, keepdim=True) * s
        return output + delta

    return hook


def install_skc_projector_injection(
    transformer: torch.nn.Module,
    lora_path: str,
    strength: float,
):
    """Load the SKC vector from ``lora_path`` and hook it onto ``txtfusion.projector``.

    Returns ``(source_key, handle)``; ``handle.remove()`` disables the injection.
    """
    from safetensors.torch import load_file

    state_dict = load_file(lora_path)
    source, vector = extract_projector_vector(state_dict)
    projector = transformer.txtfusion.projector
    handle = projector.register_forward_hook(make_skc_projector_hook(vector, strength))
    return source, handle
